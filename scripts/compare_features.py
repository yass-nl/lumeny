"""
Definitive feature comparison:
  - Load processed OHLCV (same source as notebook)
  - Compute features with live features.py code
  - Compare against notebook's saved feature parquet
  - Report every mismatch
"""

import sys
sys.path.insert(0, 'backend')

import pandas as pd
import numpy as np
from pathlib import Path

from features import compute_features, compute_time_features, compute_cross_tf_features
from features import compute_cross_pair_correlations, build_feature_row, PAIRS

PROCESSED_DIR = Path('backend/data/processed')
FEATURES_DIR = Path('backend/data/features')

PAIR = 'USDJPY'
PAIR_ID = PAIRS.index(PAIR)

# Pick a few test timestamps near end of data
TEST_TIMESTAMPS = [
    '2025-12-31 15:00:00',
    '2025-12-15 10:00:00',
    '2025-11-20 14:00:00',
    '2025-10-01 08:00:00',
    '2025-06-15 12:00:00',
]

print("=" * 70)
print(f"FEATURE COMPARISON: {PAIR}")
print("=" * 70)

# ── Step 1: Load notebook features ──
print("\n[1] Loading notebook features parquet...")
nb_features = pd.read_parquet(FEATURES_DIR / f'{PAIR}_features.parquet')
print(f"    Shape: {nb_features.shape}")
print(f"    Columns: {len(nb_features.columns)}")

# ── Step 2: Load processed OHLCV (same source as notebook) ──
print("\n[2] Loading processed OHLCV...")
ohlcv_by_tf = {}
for tf in ['5m', '15m', '1H', '4H', '1D']:
    path = PROCESSED_DIR / f'{PAIR}_{tf}.parquet'
    ohlcv_by_tf[tf] = pd.read_parquet(path)
    print(f"    {tf}: {ohlcv_by_tf[tf].shape}")

# ── Step 3: Load all pairs' 1H closes for cross-pair correlations ──
print("\n[3] Loading all pairs' 1H closes...")
closes_1h = {}
for pair in PAIRS:
    df_1h = pd.read_parquet(PROCESSED_DIR / f'{pair}_1H.parquet')
    closes_1h[pair] = df_1h['close']
    print(f"    {pair}: {len(df_1h)} rows")

# ── Step 4: Get expected columns from model ──
print("\n[4] Loading model to get expected_cols...")
import joblib
model_bundle = joblib.load('backend/models/model_1H_Q50.joblib')
expected_cols = model_bundle['feature_cols']
print(f"    Model expects {len(expected_cols)} features")

# ── Step 5: Compute features with live code ──
print("\n[5] Computing features with live code (build_feature_row)...")
live_features = build_feature_row(
    ohlcv_by_tf=ohlcv_by_tf,
    closes_1h_all_pairs=closes_1h,
    pair=PAIR,
    pair_id=PAIR_ID,
    expected_cols=expected_cols,
)
print(f"    Shape: {live_features.shape}")

# ── Step 6: Compare at each test timestamp ──
print("\n[6] Comparing features at test timestamps...")
print("=" * 70)

# Features that are in both
nb_cols = set(nb_features.columns)
live_cols = set(live_features.columns)

# The notebook has 229 cols (features + labels), live has 208 (model cols only)
# Compare only the features the model actually uses
compare_cols = [c for c in expected_cols if c in nb_cols]
print(f"\nComparing {len(compare_cols)} features (model-expected that exist in both)")

missing_in_nb = [c for c in expected_cols if c not in nb_cols]
if missing_in_nb:
    print(f"WARNING: {len(missing_in_nb)} model features missing from notebook: {missing_in_nb}")

total_mismatches = 0
total_checked = 0

for ts_str in TEST_TIMESTAMPS:
    ts = pd.Timestamp(ts_str)

    if ts not in nb_features.index:
        print(f"\n  {ts_str}: NOT IN notebook parquet, skipping")
        continue
    if ts not in live_features.index:
        print(f"\n  {ts_str}: NOT IN live features, skipping")
        continue

    print(f"\n{'-' * 70}")
    print(f"  TIMESTAMP: {ts_str}")
    print(f"{'-' * 70}")

    nb_row = nb_features.loc[ts]
    live_row = live_features.loc[ts]

    mismatches = []
    close_matches = []
    nan_mismatches = []

    for col in compare_cols:
        nb_val = nb_row[col]
        live_val = live_row[col]
        total_checked += 1

        nb_nan = pd.isna(nb_val)
        live_nan = pd.isna(live_val)

        if nb_nan and live_nan:
            continue  # Both NaN — match

        if nb_nan != live_nan:
            nan_mismatches.append((col, nb_val, live_val))
            total_mismatches += 1
            continue

        # Both have values — check closeness
        if nb_val == 0 and live_val == 0:
            continue

        abs_diff = abs(nb_val - live_val)
        rel_diff = abs_diff / (abs(nb_val) + 1e-15)

        if rel_diff > 0.01:  # >1% relative difference
            mismatches.append((col, nb_val, live_val, rel_diff))
            total_mismatches += 1
        elif rel_diff > 0.001:  # 0.1-1% — close but not exact
            close_matches.append((col, nb_val, live_val, rel_diff))

    if nan_mismatches:
        print(f"\n  NaN MISMATCHES ({len(nan_mismatches)}):")
        for col, nb_val, live_val in nan_mismatches[:20]:
            print(f"    {col:35s}  notebook={nb_val!s:>15s}  live={live_val!s:>15s}")

    if mismatches:
        print(f"\n  VALUE MISMATCHES >1% ({len(mismatches)}):")
        for col, nb_val, live_val, rel in sorted(mismatches, key=lambda x: -x[3])[:30]:
            print(f"    {col:35s}  notebook={nb_val:>15.8f}  live={live_val:>15.8f}  diff={rel:.4%}")

    if close_matches:
        print(f"\n  CLOSE MATCHES 0.1-1% ({len(close_matches)}):")
        for col, nb_val, live_val, rel in sorted(close_matches, key=lambda x: -x[3])[:10]:
            print(f"    {col:35s}  notebook={nb_val:>15.8f}  live={live_val:>15.8f}  diff={rel:.4%}")

    if not mismatches and not nan_mismatches and not close_matches:
        print(f"  ALL {len(compare_cols)} FEATURES MATCH PERFECTLY")

# ── Summary ──
print(f"\n{'=' * 70}")
print(f"SUMMARY")
print(f"{'=' * 70}")
print(f"Total feature-checks: {total_checked}")
print(f"Total mismatches: {total_mismatches}")
if total_checked > 0:
    print(f"Match rate: {(total_checked - total_mismatches) / total_checked:.2%}")
