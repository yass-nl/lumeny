"""
Replay prediction id=9: USDJPY 1H at candle T=14:00 2026-03-04.

Fetches the same data from Polygon REST that the live system would have had,
computes features with the live code, runs inference, and compares to the
actual prediction output.

Actual prediction:
  direction=bullish, probability=0.7474, raw_p_down=0.001
  q10=0.0062, q25=0.0117, q50=0.0133, q75=0.0155, q90=0.0473
  entry_price=157.234, low_conviction=1
"""

import sys
sys.path.insert(0, 'backend')

import asyncio
import os

# Load .env
from pathlib import Path
env_path = Path(__file__).parent.parent / '.env'
if env_path.exists():
    for line in env_path.read_text().splitlines():
        if '=' in line and not line.startswith('#'):
            k, v = line.split('=', 1)
            os.environ.setdefault(k.strip(), v.strip())

import pandas as pd
import numpy as np
import joblib
from pathlib import Path

from data_service import fetch_historical_bars
from features import build_feature_row, resample_ohlcv, PAIRS
from inference import Predictor

# The prediction was logged at 15:06 UTC on 2026-03-04
# Candle T=14:00 means last complete 1H candle ends at 15:00
# The system fetched data up to "now" and dropped the last candle
# So the last 1H candle in the buffer would be 14:00 (close = 15:00 close)

PAIR = 'USDJPY'
PAIR_ID = PAIRS.index(PAIR)

# Actual prediction values to compare against
ACTUAL = {
    'direction': 'bullish',
    'probability': 0.7474,
    'raw_p_down': 0.001,
    'q10': 0.0062,
    'q25': 0.0117,
    'q50': 0.0133,
    'q75': 0.0155,
    'q90': 0.0473,
    'entry_price': 157.234,
    'low_conviction': True,
}


async def main():
    print("=" * 70)
    print("REPLAY: Prediction #9 — USDJPY 1H at 2026-03-04 T=14:00")
    print("=" * 70)

    # ── Step 1: Fetch data exactly like the live system ──
    print("\n[1] Fetching data from Polygon REST (same as live system)...")

    # The live system fetches up to "today" and drops last candle
    # We simulate what it had at ~15:05 UTC on 2026-03-04
    to_date = '2026-03-04'

    buffers = {}
    tf_fetch_config = {
        '5m':  (5,   'minute', 3),
        '15m': (15,  'minute', 7),
        '1H':  (1,   'hour',   850),
    }

    for pair in PAIRS:
        buffers[pair] = {}
        for tf_name, (multiplier, timespan, lookback_days) in tf_fetch_config.items():
            from_dt = pd.Timestamp(to_date) - pd.Timedelta(days=lookback_days)
            from_date = from_dt.strftime('%Y-%m-%d')
            try:
                df = await fetch_historical_bars(pair, multiplier, timespan, from_date, to_date)
                if not df.empty:
                    df = df[df.index.dayofweek < 5]
                    # Drop last candle (may be incomplete) — same as live
                    if len(df) > 1:
                        df = df.iloc[:-1]
                    buffers[pair][tf_name] = df
                    if pair == PAIR:
                        print(f"    {pair} {tf_name}: {len(df)} candles "
                              f"({df.index[0]} to {df.index[-1]})")
            except Exception as e:
                print(f"    {pair} {tf_name} error: {e}")

        # Trim all timeframes to what was available at prediction time
        # The prediction used candle T=14:00, so last 1H candle = 14:00
        cutoff_1h = pd.Timestamp('2026-03-04 14:00:00')
        cutoff_sub = pd.Timestamp('2026-03-04 14:59:00')
        for tf_name in list(buffers[pair].keys()):
            df = buffers[pair][tf_name]
            if tf_name in ('5m', '15m'):
                buffers[pair][tf_name] = df[df.index <= cutoff_sub]
            else:
                buffers[pair][tf_name] = df[df.index <= cutoff_1h]

        # Resample 4H and 1D from 1H
        if '1H' in buffers[pair] and not buffers[pair]['1H'].empty:
            df_1h = buffers[pair]['1H']
            for tf_name, rule in [('4H', '4h'), ('1D', '1D')]:
                df_r = resample_ohlcv(df_1h, rule)
                if not df_r.empty:
                    buffers[pair][tf_name] = df_r
                    if pair == PAIR:
                        print(f"    {pair} {tf_name}: {len(df_r)} candles (resampled)")

    # Verify last 1H candle
    last_1h = buffers[PAIR]['1H'].index[-1]
    last_close = buffers[PAIR]['1H']['close'].iloc[-1]
    print(f"\n    Last 1H candle: {last_1h}")
    print(f"    Last close: {last_close}")
    print(f"    Expected entry_price: {ACTUAL['entry_price']}")
    print(f"    Match: {abs(last_close - ACTUAL['entry_price']) < 0.01}")

    # ── Step 2: Compute features ──
    print("\n[2] Computing features with live code...")

    closes_1h = {}
    for pair in PAIRS:
        if pair in buffers and '1H' in buffers[pair]:
            closes_1h[pair] = buffers[pair]['1H']['close']

    model_bundle = joblib.load('backend/models/model_1H_Q50.joblib')
    expected_cols = model_bundle['feature_cols']

    features_df = build_feature_row(
        ohlcv_by_tf=buffers[PAIR],
        closes_1h_all_pairs=closes_1h,
        pair=PAIR,
        pair_id=PAIR_ID,
        expected_cols=expected_cols,
    )
    print(f"    Features shape: {features_df.shape}")
    print(f"    Last feature timestamp: {features_df.index[-1]}")

    # Show the actual feature row that would be fed to the model
    latest = features_df.iloc[-1]
    nan_count = latest.isna().sum()
    print(f"    NaN features in latest row: {nan_count}/{len(latest)}")

    if nan_count > 0:
        nan_cols = latest[latest.isna()].index.tolist()
        print(f"    NaN columns: {nan_cols}")

    # ── Step 3: Run inference ──
    print("\n[3] Running inference...")
    predictor = Predictor()
    result = predictor.predict(features_df, PAIR)

    h1 = result['horizons']['1H']

    print(f"\n    REPLAYED prediction:")
    print(f"      direction:      {h1['direction']}")
    print(f"      probability:    {h1['probability']}")
    print(f"      raw_p_down:     {h1['raw_p_down']}")
    print(f"      cal_p_down:     {h1['calibrated_p_down']}")
    print(f"      cal_p_up:       {h1['calibrated_p_up']}")
    print(f"      signal_strength: {h1['signal_strength']}")
    print(f"      low_conviction: {h1['low_conviction']}")
    print(f"      expected_move:  {h1['expected_move_pct']}")
    print(f"      quantiles:     {h1['quantiles']}")

    # ── Step 4: Compare ──
    print(f"\n{'=' * 70}")
    print("COMPARISON: Replayed vs Actual prediction #9")
    print(f"{'=' * 70}")

    checks = [
        ('direction',      h1['direction'],       ACTUAL['direction']),
        ('probability',    h1['probability'],      ACTUAL['probability']),
        ('raw_p_down',     h1['raw_p_down'],       ACTUAL['raw_p_down']),
        ('Q10',            h1['quantiles']['Q10'], ACTUAL['q10']),
        ('Q25',            h1['quantiles']['Q25'], ACTUAL['q25']),
        ('Q50',            h1['quantiles']['Q50'], ACTUAL['q50']),
        ('Q75',            h1['quantiles']['Q75'], ACTUAL['q75']),
        ('Q90',            h1['quantiles']['Q90'], ACTUAL['q90']),
        ('low_conviction', h1['low_conviction'],   ACTUAL['low_conviction']),
    ]

    all_match = True
    for name, replayed, actual in checks:
        if isinstance(replayed, float) and isinstance(actual, float):
            match = abs(replayed - actual) < 0.0001
        else:
            match = replayed == actual
        status = 'OK' if match else 'MISMATCH'
        if not match:
            all_match = False
        print(f"  {name:20s}  replayed={str(replayed):>12s}  actual={str(actual):>12s}  [{status}]")

    print(f"\n  VERDICT: {'ALL MATCH - prediction is reproducible' if all_match else 'DIFFERENCES FOUND'}")

    # ── Step 5: Also compare against notebook-style computation ──
    # If we have the training parquet, we can also check what features differ
    # between the live data and training data distribution
    print(f"\n{'=' * 70}")
    print("FEATURE DISTRIBUTION CHECK")
    print(f"{'=' * 70}")

    # Load training features for USDJPY to compare distributions
    nb_features = pd.read_parquet('backend/data/features/USDJPY_features.parquet')
    live_row = features_df.iloc[-1]

    compare_cols = [c for c in expected_cols if c in nb_features.columns]
    outliers = []
    for col in compare_cols:
        live_val = live_row[col]
        if pd.isna(live_val):
            continue
        nb_col = nb_features[col].dropna()
        if len(nb_col) == 0:
            continue
        mean = nb_col.mean()
        std = nb_col.std()
        if std > 0:
            z_score = (live_val - mean) / std
            if abs(z_score) > 3:
                outliers.append((col, live_val, mean, std, z_score))

    if outliers:
        print(f"\n  Features >3 sigma from training distribution ({len(outliers)}):")
        for col, val, mean, std, z in sorted(outliers, key=lambda x: -abs(x[4])):
            print(f"    {col:35s}  live={val:>12.6f}  train_mean={mean:>12.6f}  z={z:>+.1f}")
    else:
        print("\n  All features within 3 sigma of training distribution.")


if __name__ == '__main__':
    asyncio.run(main())
