"""
Feature Pattern Audit — 18-22h UTC vs other hours
Computes top 25 Q50 features over last 2 weeks and compares
the distribution at 18-22h UTC vs all other hours.
"""

import pandas as pd
import numpy as np
import joblib
import warnings
from pathlib import Path
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')

import sys
sys.path.insert(0, str(Path(__file__).parent))
from test_capital_sim_3 import (
    fetch_bars, compute_features_for_pair, PAIRS
)

# ── Config ──
LOOKBACK_DAYS = 14
EDGE_HOURS    = {18, 19, 20, 21, 22}

TOP_FEATURES = [
    'accel_mean', 'momentum_shift', 'volume_cv',
    'dist_from_24h_high', 'realized_skew', 'lower_wick_ratio',
    'upper_wick_ratio', 'volume_ratio_6', 'volume_ratio_24',
    'dist_from_24h_low', 'range_pos_24', 'accel_skew',
    'rv_yang_zhang_delta_3h', 'kyle_lambda_change', 'rv_yang_zhang_delta_6h',
    'range_return_ratio', 'info_accel', 'accel_std', 'body_ratio',
    'range_pos_48', 'entropy_volume_divergence', 'tail_ratio_95_5',
    'epps_1m_15m', 'abs_Q50', 'conf_ratio',
]

# ── Fetch & build features ──
end_dt   = datetime.utcnow()
start_dt = end_dt - timedelta(days=LOOKBACK_DAYS + 2)  # +2 for warmup

print(f'Feature Pattern Audit — last {LOOKBACK_DAYS} days')
print(f'Edge window: {sorted(EDGE_HOURS)}h UTC vs all other hours')
print(f'Fetching {len(PAIRS)} pairs...\n')

# Load Q50 + meta models for abs_Q50 / conf_ratio
q50_bundle  = joblib.load('backend/models_6/3_quants/model_1H_Q50.joblib')
q25_bundle  = joblib.load('backend/models_6/3_quants/model_1H_Q25.joblib')
q75_bundle  = joblib.load('backend/models_6/3_quants/model_1H_Q75.joblib')
q50_model   = q50_bundle['model']
q25_model   = q25_bundle['model']
q75_model   = q75_bundle['model']
feat_cols   = q50_model.feature_name_

all_rows = []

import asyncio

async def fetch_all():
    from_date = start_dt.strftime('%Y-%m-%d')
    to_date   = end_dt.strftime('%Y-%m-%d')
    cutoff    = pd.Timestamp(end_dt - timedelta(days=LOOKBACK_DAYS)).tz_localize(None)

    for pair in PAIRS:
        try:
            print(f'  {pair}...', end=' ', flush=True)
            df_1m = await fetch_bars(pair, 1, 'minute', from_date, to_date)
            if df_1m.empty:
                print('NO DATA')
                continue

            df_5m  = df_1m.resample('5min').agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum'}).dropna()
            df_15m = df_1m.resample('15min').agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum'}).dropna()
            df_1h  = df_1m.resample('1h').agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum'}).dropna()

            data = {pair: {'1m': df_1m, '5m': df_5m, '15m': df_15m, '1h': df_1h}}
            df_feat = compute_features_for_pair(pair, data)
            if df_feat is None or len(df_feat) == 0:
                print('NO FEATURES')
                continue

            # Run inference
            X = df_feat[feat_cols].ffill().fillna(0)
            q50 = q50_model.predict(X)
            q25 = q25_model.predict(X)
            q75 = q75_model.predict(X)
            df_feat['abs_Q50']    = np.abs(q50)
            df_feat['conf_ratio'] = np.abs(q50) / np.clip(q75 - q25, 1e-10, None)

            # Keep only last LOOKBACK_DAYS
            df_feat = df_feat[df_feat.index >= cutoff]
            all_rows.append(df_feat)
            print(f'{len(df_feat)} hours')
        except Exception as e:
            print(f'ERROR: {e}')

asyncio.run(fetch_all())

if not all_rows:
    print('No data fetched.')
    exit()

df = pd.concat(all_rows).sort_index()
print(f'\nTotal rows: {len(df):,}')

# ── Split edge vs other ──
in_edge   = df.index.hour.isin(EDGE_HOURS)
df_edge   = df[in_edge]
df_other  = df[~in_edge]

print(f'Edge hours rows:  {len(df_edge):,}')
print(f'Other hours rows: {len(df_other):,}')

# ── Print comparison table ──
print(f'\n{"Feature":<30} {"Edge mean":>12} {"Other mean":>12} {"Edge std":>10} {"Other std":>10} {"Ratio":>8}')
print('-' * 88)

results = []
for feat in TOP_FEATURES:
    if feat not in df.columns:
        continue
    e_mean = df_edge[feat].mean()
    o_mean = df_other[feat].mean()
    e_std  = df_edge[feat].std()
    o_std  = df_other[feat].std()
    # ratio: how much larger is edge vs other (absolute)
    ratio  = e_mean / o_mean if abs(o_mean) > 1e-10 else float('nan')
    results.append((feat, e_mean, o_mean, e_std, o_std, ratio))

results.sort(key=lambda x: abs(x[5] - 1) if not np.isnan(x[5]) else 0, reverse=True)

for feat, e_mean, o_mean, e_std, o_std, ratio in results:
    flag = ' <<<' if not np.isnan(ratio) and abs(ratio - 1) > 0.3 else ''
    print(f'{feat:<30} {e_mean:>12.5f} {o_mean:>12.5f} {e_std:>10.5f} {o_std:>10.5f} {ratio:>8.2f}{flag}')

# ── abs_Q50 percentile breakdown by hour ──
print(f'\n--- abs_Q50 median by hour ---')
print(f'{"Hour":>6} {"Count":>7} {"abs_Q50 med":>13} {"conf_ratio med":>16}')
print('-' * 46)
for h in range(24):
    sub = df[df.index.hour == h]
    if len(sub) == 0:
        continue
    flag = ' <<<' if h in EDGE_HOURS else ''
    print(f'{h:>6}  {len(sub):>6,}  {sub["abs_Q50"].median():>12.6f}  {sub["conf_ratio"].median():>15.4f}{flag}')

# ── Hour-by-hour breakdown of key state features ──
STATE_FEATURES = [
    'rv_yang_zhang_delta_3h',
    'rv_yang_zhang_delta_6h',
    'kyle_lambda_change',
    'volume_cv',
    'epps_1m_15m',
    'accel_mean',
]

# Compute per-feature, per-hour: median and 75th percentile
# Using only already-loaded df — no re-fetch
print(f'\n--- Key state features by hour (median) ---')
header = f'{"Hour":>6}' + ''.join(f'{f[:14]:>16}' for f in STATE_FEATURES)
print(header)
print('-' * (6 + 16 * len(STATE_FEATURES)))

# Get edge-hour baselines for comparison
edge_medians = {f: df[df.index.hour.isin(EDGE_HOURS)][f].median() for f in STATE_FEATURES if f in df.columns}

for h in range(24):
    sub = df[df.index.hour == h]
    if len(sub) == 0:
        continue
    vals = []
    flags = []
    for f in STATE_FEATURES:
        if f not in df.columns:
            vals.append(float('nan'))
            flags.append(False)
            continue
        med = sub[f].median()
        vals.append(med)
        # flag if within 50% of edge-hour median
        e_med = edge_medians.get(f, 0)
        flags.append(abs(e_med) > 1e-10 and abs(med) >= abs(e_med) * 0.5)

    row = f'{h:>6}' + ''.join(f'{v:>16.6f}' for v in vals)
    flag = ' <<<' if h in EDGE_HOURS else (' ***' if any(flags) else '')
    print(row + flag)

print()
print('<<< = edge window   *** = at least one feature within 50% of edge-hour level')

# ── Meta proba distribution when volume_cv + epps spike off-hours ──
meta_bundle = joblib.load('backend/models_6/meta/meta_confidence.joblib')
meta_model  = meta_bundle['model']
meta_feat_cols = meta_bundle['meta_feature_cols']

# Run meta on all rows that have Q50 > threshold
MIN_Q50 = 0.00028 * 0.7
tradeable = df[df['abs_Q50'] > MIN_Q50].copy()

# Build meta feature matrix — need Q25/Q75 too
q25_model = joblib.load('backend/models_6/3_quants/model_1H_Q25.joblib')['model']
q75_model = joblib.load('backend/models_6/3_quants/model_1H_Q75.joblib')['model']
X_all = tradeable[feat_cols].ffill().fillna(0)
tradeable['Q50_oof']    = q50_model.predict(X_all)
tradeable['Q25_oof']    = q25_model.predict(X_all)
tradeable['Q75_oof']    = q75_model.predict(X_all)
tradeable['iqr']        = tradeable['Q75_oof'] - tradeable['Q25_oof']
tradeable['conf_ratio'] = tradeable['abs_Q50'] / tradeable['iqr'].clip(lower=1e-10)

X_meta = tradeable[[c for c in meta_feat_cols if c in tradeable.columns]].ffill().fillna(0)
tradeable['meta_proba'] = meta_model.predict_proba(X_meta)[:, 1]

print(f'\n--- Meta proba: state-regime filter vs clock filter ---')
print(f'Tradeable rows (|Q50|>threshold): {len(tradeable):,}')

# Define regime conditions
vol_cv_thresh   = 0.35
epps_thresh     = 7.0
in_regime       = (tradeable['volume_cv'] > vol_cv_thresh) & (tradeable['epps_1m_15m'] > epps_thresh)
in_clock        = tradeable.index.hour.isin(EDGE_HOURS)
in_regime_only  = in_regime & ~in_clock
in_clock_only   = in_clock & ~in_regime
in_both         = in_regime & in_clock
in_neither      = ~in_regime & ~in_clock

print(f'\nCondition                    Rows   meta>0.5   avg_meta   med_meta')
print('-' * 68)
for label, mask in [
    ('Clock (19-21h)',            in_clock),
    ('Regime (vol_cv+epps)',      in_regime),
    ('Both',                      in_both),
    ('Regime only (off-hours)',   in_regime_only),
    ('Clock only (low regime)',   in_clock_only),
    ('Neither',                   in_neither),
]:
    sub = tradeable[mask]
    if len(sub) == 0:
        print(f'{label:<28}    0       -          -          -')
        continue
    n_above = (sub['meta_proba'] > 0.5).sum()
    print(f'{label:<28} {len(sub):>5}  {n_above:>5} ({n_above/len(sub):.0%})  {sub["meta_proba"].mean():.3f}      {sub["meta_proba"].median():.3f}')

# Per-hour: how often does the regime fire off-hours
print(f'\n--- Off-hours regime hits (volume_cv>{vol_cv_thresh} & epps>{epps_thresh}) ---')
print(f'{"Hour":>6} {"Total":>7} {"Regime":>8} {"Rate":>7} {"meta>0.5":>10} {"avg_meta":>10}')
print('-' * 52)
for h in range(24):
    sub = tradeable[tradeable.index.hour == h]
    if len(sub) == 0:
        continue
    reg = sub[(sub['volume_cv'] > vol_cv_thresh) & (sub['epps_1m_15m'] > epps_thresh)]
    rate = len(reg) / len(sub) if len(sub) > 0 else 0
    n_meta = (reg['meta_proba'] > 0.5).sum() if len(reg) > 0 else 0
    avg_meta = reg['meta_proba'].mean() if len(reg) > 0 else float('nan')
    flag = ' <<<' if h in EDGE_HOURS else ''
    print(f'{h:>6} {len(sub):>7} {len(reg):>8} {rate:>7.1%} {n_meta:>10} {avg_meta:>10.3f}{flag}')
