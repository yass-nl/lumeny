"""
Replay every hour from market open (Sun Mar 15 21:00 UTC) to now (Tue Mar 17 ~15:43 UTC).
For each hour, compute features and run prediction for all 15 pairs.
Show ALL tradeable signals and the full meta_proba distribution.
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


def predict_at_hour(features_df, hour_ts):
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

    direction = 'bullish' if q50 > 0 else 'bearish'
    is_tradeable = meta_proba > META_THRESHOLD and abs_q50 > MIN_Q50_THRESHOLD

    return {
        'q50': q50,
        'meta_proba': meta_proba,
        'is_tradeable': is_tradeable,
        'direction': direction,
        'abs_q50': abs_q50,
    }


async def main():
    now = datetime.utcnow()
    to_date = now.strftime('%Y-%m-%d')

    # Market opened Sunday Mar 15 21:00 UTC
    # We need data before that for features warmup
    # Fetch 40 days of 1m for full feature context
    from_date = (now - timedelta(days=40)).strftime('%Y-%m-%d')

    # Replay window: Sun Mar 15 21:00 -> now
    replay_start = datetime(2026, 3, 15, 21, 0, 0)
    replay_end = now.replace(minute=0, second=0, microsecond=0)

    print(f"Replay window: {replay_start} -> {replay_end}")
    print(f"Fetching 40 days of 1m data for all 15 pairs...\n")

    all_features = {}  # pair -> DataFrame

    for pair in PAIRS:
        print(f"  {pair}...", end=" ", flush=True)
        await asyncio.sleep(0.3)
        df_1m = await fetch_historical_bars(pair, 1, 'minute', from_date, to_date)
        if df_1m.empty:
            print("NO DATA")
            continue

        # Filter weekends
        df_1m = df_1m[~((df_1m.index.dayofweek == 5) |
                        ((df_1m.index.dayofweek == 6) & (df_1m.index.hour < 21)))]
        # Drop future bars
        now_naive = now.replace(tzinfo=None) if now.tzinfo else now
        df_1m = df_1m[df_1m.index + timedelta(minutes=1) <= now_naive]

        df_5m = resample_ohlcv(df_1m, '5min')
        df_15m = resample_ohlcv(df_1m, '15min')

        feat = compute_features_for_pair(pair, df_1m, df_5m, df_15m)
        all_features[pair] = feat
        print(f"{len(feat)} feature hours")

    # Generate all replay hours
    hours = []
    h = replay_start
    while h <= replay_end:
        hours.append(h)
        h += timedelta(hours=1)

    print(f"\nTotal replay hours: {len(hours)}")
    print(f"From {hours[0]} to {hours[-1]}")

    # Run predictions for every pair at every hour
    print(f"\n{'='*100}")
    print(f"HOUR-BY-HOUR REPLAY: ALL TRADEABLE SIGNALS")
    print(f"{'='*100}")

    all_trades = []
    hourly_max_meta = []

    for hour_ts in hours:
        hour_pd = pd.Timestamp(hour_ts)
        hour_results = []

        for pair in PAIRS:
            if pair not in all_features:
                continue
            feat = all_features[pair]
            if hour_pd not in feat.index:
                continue

            result = predict_at_hour(feat, hour_pd)
            if result:
                result['pair'] = pair
                result['hour'] = hour_ts
                hour_results.append(result)

        if not hour_results:
            continue

        # Track max meta for this hour
        max_meta = max(r['meta_proba'] for r in hour_results)
        avg_meta = np.mean([r['meta_proba'] for r in hour_results])
        tradeable = [r for r in hour_results if r['is_tradeable']]
        hourly_max_meta.append({
            'hour': hour_ts,
            'max_meta': max_meta,
            'avg_meta': avg_meta,
            'n_tradeable': len(tradeable),
            'weekday': hour_ts.strftime('%A'),
        })

        if tradeable:
            for t in tradeable:
                all_trades.append(t)
                print(f"  {hour_ts} ({hour_ts.strftime('%a')})  {t['pair']:>8}  "
                      f"meta={t['meta_proba']:.4f}  q50={t['q50']:+.7f}  {t['direction']}")

    # Summary
    print(f"\n{'='*100}")
    print(f"SUMMARY")
    print(f"{'='*100}")
    print(f"Total hours replayed: {len(hours)}")
    print(f"Total tradeable signals: {len(all_trades)}")

    if all_trades:
        print(f"\nTradeable signals by day/hour:")
        by_day = {}
        for t in all_trades:
            day = t['hour'].strftime('%Y-%m-%d %A')
            if day not in by_day:
                by_day[day] = []
            by_day[day].append(t)
        for day, trades in sorted(by_day.items()):
            print(f"  {day}: {len(trades)} trades")
            for t in trades:
                print(f"    {t['hour'].strftime('%H:%M')} {t['pair']:>8} meta={t['meta_proba']:.4f} {t['direction']}")

    # Hourly meta distribution
    print(f"\n{'='*100}")
    print(f"HOURLY MAX META DISTRIBUTION")
    print(f"{'='*100}")
    print(f"{'Hour':<25} {'Day':<10} {'Max meta':>9} {'Avg meta':>9} {'#Trades':>8}")
    print("-" * 65)
    for h in hourly_max_meta:
        marker = " <-- TRADES" if h['n_tradeable'] > 0 else ""
        print(f"{str(h['hour']):<25} {h['weekday']:<10} {h['max_meta']:>9.4f} {h['avg_meta']:>9.4f} {h['n_tradeable']:>8}{marker}")


if __name__ == '__main__':
    asyncio.run(main())
