"""
Per-Hour Threshold Calibration
Loads last 3 years of features_6 data across all 15 pairs,
computes per-hour percentile cutoffs for the 6 rule features,
and saves the threshold table to backend/models_6/hour_thresholds.parquet
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

PAIRS = [
    'EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD', 'USDCAD', 'NZDUSD',
    'EURJPY', 'GBPJPY', 'EURGBP', 'EURAUD', 'AUDJPY', 'CADJPY', 'CHFJPY', 'AUDNZD',
]

FEATURES = [
    'volume_cv',       # gate
    'epps_1m_15m',     # gate
    'momentum_shift',  # direction (use absolute value for threshold, sign for direction)
    'accel_mean',      # direction (use absolute value for threshold, sign for direction)
    'realized_skew',   # direction support
    'kyle_lambda_r2',  # gate support
]

PERCENTILES = [50, 60, 70, 75, 80, 85, 90]

CUTOFF_DATE = pd.Timestamp(datetime.utcnow() - timedelta(days=3*365)).tz_localize(None)

print(f'Loading features_6 — last 3 years (from {CUTOFF_DATE.date()})')
print(f'Pairs: {len(PAIRS)}')

frames = []
for pair in PAIRS:
    path = Path(f'backend/data/features_6/{pair}_features.parquet')
    if not path.exists():
        print(f'  {pair}: NOT FOUND')
        continue
    df = pd.read_parquet(path, columns=FEATURES)
    df = df[df.index >= CUTOFF_DATE]
    df['pair'] = pair
    frames.append(df)
    print(f'  {pair}: {len(df):,} rows')

df_all = pd.concat(frames).sort_index()
print(f'\nTotal rows: {len(df_all):,}')
print(f'Date range: {df_all.index.min().date()} to {df_all.index.max().date()}')

# For direction features, threshold on absolute value
# (we threshold |momentum_shift| > cutoff, then use sign for direction)
df_all['abs_momentum_shift'] = df_all['momentum_shift'].abs()
df_all['abs_accel_mean']     = df_all['accel_mean'].abs()
df_all['abs_realized_skew']  = df_all['realized_skew'].abs()

THRESHOLD_FEATURES = [
    'volume_cv',
    'epps_1m_15m',
    'abs_momentum_shift',
    'abs_accel_mean',
    'abs_realized_skew',
    'kyle_lambda_r2',
]

# ── Compute per-hour percentiles ──
rows = []
print(f'\nComputing per-hour percentiles...')
for h in range(24):
    sub = df_all[df_all.index.hour == h]
    row = {'hour': h, 'n_rows': len(sub)}
    for feat in THRESHOLD_FEATURES:
        if feat not in sub.columns:
            for p in PERCENTILES:
                row[f'{feat}_p{p}'] = np.nan
            continue
        vals = sub[feat].dropna()
        for p in PERCENTILES:
            row[f'{feat}_p{p}'] = np.percentile(vals, p)
    rows.append(row)

thresholds = pd.DataFrame(rows).set_index('hour')

# ── Print the table ──
print(f'\n{"="*100}')
print('PER-HOUR THRESHOLD TABLE (p70 and p80)')
print(f'{"="*100}')

print(f'\n--- GATE FEATURES ---')
print(f'{"Hour":>6} {"N rows":>8}  {"vol_cv p70":>12} {"vol_cv p80":>12}  {"epps p70":>10} {"epps p80":>10}  {"klr2 p70":>10} {"klr2 p80":>10}')
print('-' * 90)
for h, row in thresholds.iterrows():
    print(f'{h:>6} {int(row.n_rows):>8}  '
          f'{row["volume_cv_p70"]:>12.4f} {row["volume_cv_p80"]:>12.4f}  '
          f'{row["epps_1m_15m_p70"]:>10.3f} {row["epps_1m_15m_p80"]:>10.3f}  '
          f'{row["kyle_lambda_r2_p70"]:>10.4f} {row["kyle_lambda_r2_p80"]:>10.4f}')

print(f'\n--- DIRECTION FEATURES (absolute values) ---')
print(f'{"Hour":>6} {"N rows":>8}  {"mom_shift p70":>15} {"mom_shift p80":>15}  {"accel_mean p70":>16} {"accel_mean p80":>16}')
print('-' * 82)
for h, row in thresholds.iterrows():
    print(f'{h:>6} {int(row.n_rows):>8}  '
          f'{row["abs_momentum_shift_p70"]:>15.6f} {row["abs_momentum_shift_p80"]:>15.6f}  '
          f'{row["abs_accel_mean_p70"]:>16.6f} {row["abs_accel_mean_p80"]:>16.6f}')

# ── Sanity check: ratio of p70 values across hours ──
print(f'\n--- SANITY CHECK: p70 ratio (21h / median-of-all-hours) ---')
print('Should confirm 21h is elevated but not the only elevated hour')
for feat in THRESHOLD_FEATURES:
    col = f'{feat}_p70'
    if col not in thresholds.columns:
        continue
    vals = thresholds[col].dropna()
    h21  = thresholds.loc[21, col]
    med  = vals.median()
    print(f'  {feat:<25}  21h p70={h21:.5f}  all-hours median p70={med:.5f}  ratio={h21/med:.2f}x')

# ── Save ──
out_path = Path('backend/models_6/hour_thresholds.parquet')
out_path.parent.mkdir(parents=True, exist_ok=True)
thresholds.to_parquet(out_path)
print(f'\nSaved to {out_path}')
print(f'Shape: {thresholds.shape}')
