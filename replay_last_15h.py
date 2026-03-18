"""
Replay the last 15 hourly inference cycles locally using Polygon REST data.
Replicates exact same logic as paper_trading.py log_predictions().

Usage:
    python replay_last_15h.py
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

warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

# ── paths ──
BACKEND_DIR = Path(__file__).parent / 'backend'
sys.path.insert(0, str(BACKEND_DIR))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / '.env')

from features import PAIRS, compute_features_for_pair, resample_ohlcv
from data_service import fetch_historical_bars

# ── Load retrained v5.1 models locally ──
MODELS_DIR = BACKEND_DIR / 'models_5.1'

print("Loading models from", MODELS_DIR)

quant_models = {}
feature_cols = None
for q_name in ['Q25', 'Q50', 'Q75']:
    q_int = int(q_name[1:])
    path = MODELS_DIR / '3_quants' / f'model_1H_Q{q_int}.joblib'
    bundle = joblib.load(path)
    quant_models[q_name] = bundle['model']
    if feature_cols is None:
        feature_cols = bundle['feature_cols']
    print(f"  Loaded {q_name}: {path.name} ({path.stat().st_size / 1e6:.1f} MB)")

meta_path = MODELS_DIR / 'meta' / 'meta_confidence.joblib'
meta_bundle = joblib.load(meta_path)
meta_model = meta_bundle['model']
meta_feature_cols = meta_bundle['meta_feature_cols']
print(f"  Loaded meta: {meta_path.name} ({meta_path.stat().st_size / 1e6:.1f} MB)")

AVG_SPREAD = 0.00028
MIN_Q50_THRESHOLD = AVG_SPREAD * 0.5  # 0.00014
META_THRESHOLD = 0.55

N_CYCLES = 15  # replay last 15 hours


def predict(features_df, pair):
    """Exact replica of inference.py Predictor.predict()."""
    X = features_df[feature_cols].ffill().fillna(0)
    latest = X.iloc[[-1]]

    q_preds = {}
    for q_name in ['Q25', 'Q50', 'Q75']:
        model = quant_models[q_name]
        q_preds[q_name] = float(model.predict(latest)[0])

    q50 = q_preds['Q50']
    q25 = q_preds['Q25']
    q75 = q_preds['Q75']

    direction = 'bullish' if q50 > 0 else ('bearish' if q50 < 0 else 'neutral')

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
    meta_proba = float(meta_model.predict_proba(X_meta)[0, 1])

    is_tradeable = meta_proba > META_THRESHOLD and abs_q50 > MIN_Q50_THRESHOLD

    return {
        'pair': pair,
        'direction': direction,
        'q25': round(q25, 8),
        'q50': round(q50, 8),
        'q75': round(q75, 8),
        'meta_proba': round(meta_proba, 4),
        'is_tradeable': is_tradeable,
        'abs_q50': round(abs_q50, 8),
        'iqr': round(iqr, 8),
    }


async def fetch_pair_data(pair, now_utc):
    """Fetch 1m (7d) and resample 5m/15m — matches training/backtest pipeline."""
    to_date = now_utc.strftime('%Y-%m-%d')
    from_date = (now_utc - timedelta(days=7)).strftime('%Y-%m-%d')

    await asyncio.sleep(0.3)
    df_1m = await fetch_historical_bars(pair, 1, 'minute', from_date, to_date)

    data = {}
    if not df_1m.empty:
        # Filter weekends
        df_1m = df_1m[~((df_1m.index.dayofweek == 5) |
                        ((df_1m.index.dayofweek == 6) & (df_1m.index.hour < 21)))]
        data['1m'] = df_1m
        data['5m'] = resample_ohlcv(df_1m, '5min')
        data['15m'] = resample_ohlcv(df_1m, '15min')
    else:
        data['1m'] = df_1m
        data['5m'] = pd.DataFrame()
        data['15m'] = pd.DataFrame()

    return data


async def main():
    now_utc = datetime.now(timezone.utc).replace(tzinfo=None)
    current_hour = now_utc.replace(minute=0, second=0, microsecond=0)

    # The 15 cycles: each cycle uses data up to the end of that hour
    # Last cycle = last fully closed hour, going back 15 hours
    last_closed_hour = current_hour - timedelta(hours=1)
    cycle_hours = [last_closed_hour - timedelta(hours=i) for i in range(N_CYCLES - 1, -1, -1)]

    print(f"\nNow UTC: {now_utc}")
    print(f"Replaying {N_CYCLES} cycles: {cycle_hours[0]} -> {cycle_hours[-1]}")
    print(f"{'='*100}")

    # Fetch data for all pairs once (we'll slice per cycle)
    print("\nFetching data from Polygon...")
    all_pair_data = {}
    for pair in PAIRS:
        print(f"  {pair}...", end=" ", flush=True)
        data = await fetch_pair_data(pair, now_utc)
        bars_1m = len(data.get('1m', pd.DataFrame()))
        bars_5m = len(data.get('5m', pd.DataFrame()))
        print(f"1m={bars_1m}, 5m={bars_5m}")
        all_pair_data[pair] = data

    # Run each cycle
    all_results = []

    for cycle_hour in cycle_hours:
        print(f"\n{'-'*100}")
        print(f"CYCLE: {cycle_hour} (data cutoff = bars closing at or before {cycle_hour})")
        print(f"{'-'*100}")

        cycle_cutoff = cycle_hour  # only use bars that closed by this time

        tf_bar_durations = {'1m': 1, '5m': 5, '15m': 15}

        for pair in PAIRS:
            try:
                data = all_pair_data[pair]

                # Slice data to only include bars fully closed by cycle_cutoff
                # A bar at timestamp T with duration D closes at T + D minutes
                # So we keep bars where T + D <= cycle_cutoff
                sliced = {}
                for tf_name, df in data.items():
                    if df.empty:
                        sliced[tf_name] = df
                        continue
                    dur_min = tf_bar_durations[tf_name]
                    bar_close_time = df.index + timedelta(minutes=dur_min)
                    mask = bar_close_time <= cycle_cutoff
                    sliced[tf_name] = df[mask]

                df_1m = sliced.get('1m', pd.DataFrame())
                df_5m = sliced.get('5m', pd.DataFrame())
                df_15m = sliced.get('15m', pd.DataFrame())

                if len(df_1m) < 120:
                    continue

                # Compute features
                features_df = compute_features_for_pair(pair, df_1m, df_5m, df_15m)
                if features_df.empty:
                    continue

                result = predict(features_df, pair)
                result['cycle_hour'] = str(cycle_hour)
                result['n_feature_rows'] = len(features_df)
                result['last_feature_time'] = str(features_df.index[-1])
                all_results.append(result)

                marker = " *** TRADEABLE ***" if result['is_tradeable'] else ""
                print(f"  {pair:8s}  dir={result['direction']:8s}  "
                      f"Q50={result['q50']:+.6f}  meta={result['meta_proba']:.4f}  "
                      f"|Q50|={result['abs_q50']:.6f}  iqr={result['iqr']:.6f}"
                      f"{marker}")

            except Exception as e:
                print(f"  {pair:8s}  ERROR: {e}")

        # Summary for this cycle
        cycle_results = [r for r in all_results if r['cycle_hour'] == str(cycle_hour)]
        tradeable = [r for r in cycle_results if r['is_tradeable']]
        meta_vals = [r['meta_proba'] for r in cycle_results]
        if meta_vals:
            print(f"\n  Cycle summary: {len(cycle_results)} pairs | "
                  f"meta_proba range [{min(meta_vals):.4f}, {max(meta_vals):.4f}] | "
                  f"tradeable: {len(tradeable)}")
        if tradeable:
            for t in tradeable:
                print(f"    >> TRADE: {t['pair']} {t['direction']} meta={t['meta_proba']:.4f} Q50={t['q50']:+.6f}")

    # Final summary
    print(f"\n{'='*100}")
    print("FINAL SUMMARY")
    print(f"{'='*100}")

    total_preds = len(all_results)
    total_tradeable = sum(1 for r in all_results if r['is_tradeable'])
    all_meta = [r['meta_proba'] for r in all_results]

    print(f"Total predictions: {total_preds} ({N_CYCLES} cycles x {len(PAIRS)} pairs)")
    print(f"Tradeable signals: {total_tradeable}")
    if all_meta:
        print(f"Meta proba: min={min(all_meta):.4f}  max={max(all_meta):.4f}  "
              f"mean={np.mean(all_meta):.4f}  median={np.median(all_meta):.4f}")

        # Distribution
        bins = [0, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.80, 1.0]
        hist, _ = np.histogram(all_meta, bins=bins)
        print(f"\nMeta proba distribution:")
        for i in range(len(bins) - 1):
            bar = '█' * hist[i]
            print(f"  [{bins[i]:.2f}-{bins[i+1]:.2f}): {hist[i]:4d}  {bar}")

    if total_tradeable > 0:
        print(f"\nTradeable signals detail:")
        for r in all_results:
            if r['is_tradeable']:
                print(f"  {r['cycle_hour']}  {r['pair']:8s}  {r['direction']:8s}  "
                      f"meta={r['meta_proba']:.4f}  Q50={r['q50']:+.6f}")


if __name__ == '__main__':
    asyncio.run(main())
