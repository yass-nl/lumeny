"""
Diagnostic: compare live-style (7d) vs backtest-style (40d) inference
specifically at hours where the backtest found tradeable signals.

We replay the last 7 days hour-by-hour, find tradeable signals in the
backtest-style run, then check if the live-style run agrees.
"""

import asyncio
import os
import sys
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8')

BACKEND_DIR = Path(__file__).parent / 'backend'
sys.path.insert(0, str(BACKEND_DIR))

from dotenv import load_dotenv
load_dotenv()

from features import PAIRS, compute_features_for_pair, resample_ohlcv
from data_service import fetch_historical_bars

MODELS_DIR = BACKEND_DIR / 'models_5.1'

q50_bundle = joblib.load(MODELS_DIR / '3_quants' / 'model_1H_Q50.joblib')
q25_bundle = joblib.load(MODELS_DIR / '3_quants' / 'model_1H_Q25.joblib')
q75_bundle = joblib.load(MODELS_DIR / '3_quants' / 'model_1H_Q75.joblib')
meta_bundle = joblib.load(MODELS_DIR / 'meta' / 'meta_confidence.joblib')

feature_cols = q50_bundle['feature_cols']
meta_feature_cols = meta_bundle['meta_feature_cols']

AVG_SPREAD = 0.00028
MIN_Q50_THRESHOLD = AVG_SPREAD * 0.5
META_THRESHOLD = 0.55

REPLAY_DAYS = 7


def predict_at_hour(features_df, pair, hour_ts):
    """Run prediction using features up to and including hour_ts."""
    feat_up_to = features_df[features_df.index <= hour_ts]
    if feat_up_to.empty:
        return None

    X = feat_up_to[feature_cols].ffill().fillna(0)
    latest = X.iloc[[-1]]

    q50 = float(q50_bundle['model'].predict(latest)[0])
    q25 = float(q25_bundle['model'].predict(latest)[0])
    q75 = float(q75_bundle['model'].predict(latest)[0])

    abs_q50 = abs(q50)
    iqr = q75 - q25

    meta_row = latest.copy()
    meta_row['Q50_oof'] = q50
    meta_row['Q25_oof'] = q25
    meta_row['Q75_oof'] = q75
    meta_row['abs_Q50'] = abs_q50
    meta_row['iqr'] = iqr
    meta_row['conf_ratio'] = abs_q50 / max(iqr, 1e-10)

    X_meta = meta_row[meta_feature_cols].fillna(0)
    meta_proba = float(meta_bundle['model'].predict_proba(X_meta)[0, 1])

    is_tradeable = meta_proba > META_THRESHOLD and abs_q50 > MIN_Q50_THRESHOLD

    return {
        'pair': pair,
        'hour': str(hour_ts),
        'q50': q50, 'q25': q25, 'q75': q75,
        'meta_proba': meta_proba,
        'abs_q50': abs_q50, 'iqr': iqr,
        'is_tradeable': is_tradeable,
    }


async def main():
    now = datetime.utcnow()
    to_date = now.strftime('%Y-%m-%d')

    # Determine replay window: last 7 days, hour by hour
    # We need data BEFORE the replay window for features warmup
    # Backtest-style: fetch 40 days of 1m
    # Live-style: for each replay hour, we simulate fetching 7 days ending at that hour

    print(f"Fetching data for all pairs...")
    print(f"Backtest-style: 40 days of 1m")
    print(f"Live-style: 7 days of 1m (same data, sliced)\n")

    from_40d = (now - timedelta(days=40)).strftime('%Y-%m-%d')

    all_bt_features = {}  # pair -> DataFrame
    all_1m_data = {}      # pair -> full 40d 1m DataFrame

    for pair in PAIRS:
        print(f"  {pair}...", end=" ", flush=True)
        await asyncio.sleep(0.3)
        df_1m = await fetch_historical_bars(pair, 1, 'minute', from_40d, to_date)
        if df_1m.empty:
            print("NO DATA")
            continue

        # Filter weekends
        df_1m = df_1m[~((df_1m.index.dayofweek == 5) |
                        ((df_1m.index.dayofweek == 6) & (df_1m.index.hour < 21)))]

        all_1m_data[pair] = df_1m

        # Backtest-style features (full 40d)
        df_5m = resample_ohlcv(df_1m, '5min')
        df_15m = resample_ohlcv(df_1m, '15min')
        feat = compute_features_for_pair(pair, df_1m, df_5m, df_15m)
        all_bt_features[pair] = feat
        print(f"{len(df_1m):,} 1m bars, {len(feat)} feature hours")

    # Determine replay hours (last 7 days)
    replay_start = now - timedelta(days=REPLAY_DAYS)
    replay_start = replay_start.replace(minute=0, second=0, microsecond=0)

    # First pass: find tradeable signals in backtest-style
    print(f"\n{'='*90}")
    print(f"SCANNING FOR TRADEABLE SIGNALS (backtest-style, last {REPLAY_DAYS} days)")
    print(f"{'='*90}")

    bt_tradeable = []

    for pair in PAIRS:
        if pair not in all_bt_features:
            continue
        feat = all_bt_features[pair]
        hours_in_range = feat[feat.index >= replay_start].index

        for hour_ts in hours_in_range:
            result = predict_at_hour(feat, pair, hour_ts)
            if result and result['is_tradeable']:
                bt_tradeable.append(result)

    print(f"Found {len(bt_tradeable)} tradeable signals in backtest-style\n")

    if not bt_tradeable:
        print("No tradeable signals found in the last 7 days!")
        return

    # Second pass: for each tradeable signal, compute live-style features
    # Live-style = 7 days of 1m ending at that hour
    print(f"{'='*90}")
    print(f"COMPARING: backtest-style vs live-style at tradeable hours")
    print(f"{'='*90}")
    print(f"{'Hour':>20} {'Pair':>8} {'BT meta':>9} {'Live meta':>10} {'Diff':>8} {'BT trade':>9} {'Live trade':>10}")
    print("-" * 80)

    matches = 0
    mismatches = 0
    detail_diffs = []

    for bt_result in sorted(bt_tradeable, key=lambda x: x['hour']):
        pair = bt_result['pair']
        hour_ts = pd.Timestamp(bt_result['hour'])

        if pair not in all_1m_data:
            continue

        # Simulate live: slice 1m data to 7 days before this hour
        # Live runs at HH:00:30 after hour closes, so it sees all 1m bars
        # with timestamps up to HH:59 of the target hour.
        # The hour_ts IS the hour (e.g., 21:00 means 21:00-21:59 data).
        # We need 1m bars from [hour_ts - 7d, hour_ts + 59min].
        sim_now = hour_ts + timedelta(hours=1)  # simulate "now" = start of next hour
        start_7d = sim_now - timedelta(days=7)

        df_1m_full = all_1m_data[pair]
        df_1m_live = df_1m_full[(df_1m_full.index >= start_7d) &
                                (df_1m_full.index + timedelta(minutes=1) <= sim_now)]

        if len(df_1m_live) < 120:
            print(f"{str(hour_ts):>20} {pair:>8}  -- insufficient 1m data for live-style ({len(df_1m_live)} bars) --")
            continue

        df_5m_live = resample_ohlcv(df_1m_live, '5min')
        df_15m_live = resample_ohlcv(df_1m_live, '15min')
        df_5m_live = df_5m_live[df_5m_live.index + timedelta(minutes=5) <= sim_now]
        df_15m_live = df_15m_live[df_15m_live.index + timedelta(minutes=15) <= sim_now]

        feat_live = compute_features_for_pair(pair, df_1m_live, df_5m_live, df_15m_live)

        if feat_live.empty or hour_ts not in feat_live.index:
            print(f"{str(hour_ts):>20} {pair:>8}  -- hour not in live features --")
            continue

        live_result = predict_at_hour(feat_live, pair, hour_ts)
        if live_result is None:
            print(f"{str(hour_ts):>20} {pair:>8}  -- live prediction failed --")
            continue

        diff = live_result['meta_proba'] - bt_result['meta_proba']
        bt_trade = "YES" if bt_result['is_tradeable'] else "no"
        live_trade = "YES" if live_result['is_tradeable'] else "no"

        agree = bt_result['is_tradeable'] == live_result['is_tradeable']
        marker = "" if agree else " *** MISMATCH ***"

        if agree:
            matches += 1
        else:
            mismatches += 1

        print(f"{str(hour_ts):>20} {pair:>8} {bt_result['meta_proba']:>9.4f} {live_result['meta_proba']:>10.4f} {diff:>+8.4f} {bt_trade:>9} {live_trade:>10}{marker}")

        detail_diffs.append({
            'hour': str(hour_ts),
            'pair': pair,
            'bt_meta': bt_result['meta_proba'],
            'live_meta': live_result['meta_proba'],
            'diff': diff,
            'bt_q50': bt_result['q50'],
            'live_q50': live_result['q50'],
            'bt_tradeable': bt_result['is_tradeable'],
            'live_tradeable': live_result['is_tradeable'],
        })

    # Summary
    print(f"\n{'='*90}")
    print(f"SUMMARY")
    print(f"{'='*90}")
    print(f"Tradeable signals in backtest-style: {len(bt_tradeable)}")
    print(f"Agreement (both tradeable): {matches}")
    print(f"Mismatches: {mismatches}")

    if detail_diffs:
        diffs_arr = [d['diff'] for d in detail_diffs]
        print(f"\nMeta_proba difference stats:")
        print(f"  Mean diff: {np.mean(diffs_arr):+.6f}")
        print(f"  Max diff:  {max(diffs_arr):+.6f}")
        print(f"  Min diff:  {min(diffs_arr):+.6f}")
        print(f"  Std diff:  {np.std(diffs_arr):.6f}")

        # Show mismatches in detail
        mismatched = [d for d in detail_diffs if d['bt_tradeable'] != d['live_tradeable']]
        if mismatched:
            print(f"\nMISMATCHED SIGNALS (backtest says trade, live disagrees or vice versa):")
            for d in mismatched:
                print(f"  {d['hour']}  {d['pair']:>8}  BT_meta={d['bt_meta']:.4f}  Live_meta={d['live_meta']:.4f}  "
                      f"BT_Q50={d['bt_q50']:+.6f}  Live_Q50={d['live_q50']:+.6f}")


if __name__ == '__main__':
    asyncio.run(main())
