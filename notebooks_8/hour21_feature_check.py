"""
Hour 21 Feature Check
======================
Are the features that fire at hour 21 genuine market signals or daily-reset artifacts?

Method:
  - Split signal bars into hour-21 vs all other hours
  - For each feature, compute mean value at hour-21 vs other hours
  - Flag features that are suspiciously different at hour-21
  - Specifically watch: range_pos_*, dist_from_*, ret_1d, ret_3d (daily-reset candidates)
    vs corr_*, beta_*, csi_*, atr_72 (genuine market state)
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path

SCRIPT_DIR     = Path(__file__).parent
FEATURES_DIR   = SCRIPT_DIR / '../backend/data/features_9'
MFE_MODEL_PATH = SCRIPT_DIR / '../backend/models_9/mfe_q50/model_1H_Q50.joblib'

TRAIN_END  = '2024-06-30'
MFE_THRESH = 70.0
COOLDOWN_H = 72

print('Loading model + features...')
bundle       = joblib.load(MFE_MODEL_PATH)
mfe_model    = bundle['model']
feature_cols = bundle['feature_cols']

dfs     = [pd.read_parquet(f) for f in sorted(FEATURES_DIR.glob('*_features.parquet'))]
df      = pd.concat(dfs).sort_index()
df_test = df[df.index > TRAIN_END].copy()

print('Running MFE model...')
X = df_test[feature_cols].ffill().fillna(0)
df_test['q50_mfe'] = mfe_model.predict(X)

# Apply cooldown
candidates = df_test[df_test['q50_mfe'] >= MFE_THRESH].sort_index()
cooldown_until = {}
kept = []
for ts, row in candidates.iterrows():
    pair = row['pair']
    if pair in cooldown_until and ts < cooldown_until[pair]:
        continue
    cooldown_until[pair] = ts + pd.Timedelta(hours=COOLDOWN_H)
    kept.append(ts)

df_sig = candidates.loc[kept].copy()
df_sig['hour'] = pd.to_datetime(df_sig.index).hour
df_sig['is_h21'] = df_sig['hour'] == 21

h21   = df_sig[df_sig['is_h21']]
other = df_sig[~df_sig['is_h21']]

print(f'\n  Signal bars: {len(df_sig):,}  |  hour-21: {len(h21):,}  |  other: {len(other):,}')

# ── Feature comparison: hour-21 vs other ─────────────────────────────────────
print(f'\n  Computing feature means at hour-21 vs other hours...')

rows = []
for col in feature_cols:
    if col not in df_sig.columns:
        continue
    v21   = h21[col].dropna()
    voth  = other[col].dropna()
    if len(v21) < 20 or len(voth) < 20:
        continue
    m21  = v21.mean()
    moth = voth.mean()
    std  = df_sig[col].std()
    if std < 1e-10:
        continue
    # Normalised difference: how many std devs apart
    diff_std = (m21 - moth) / std
    rows.append({
        'feature':   col,
        'mean_h21':  m21,
        'mean_other':moth,
        'diff_std':  diff_std,
        'abs_diff':  abs(diff_std),
    })

res = pd.DataFrame(rows).sort_values('abs_diff', ascending=False)

print(f'\n{"="*70}')
print(f'  TOP 40 FEATURES MOST DIFFERENT AT HOUR-21 vs OTHER HOURS')
print(f'  (sorted by |normalised difference| in std devs)')
print(f'  Positive diff_std = feature is HIGHER at hour-21')
print(f'{"="*70}')
print(f'\n  {"Feature":<40}  {"h21_mean":>10}  {"other_mean":>10}  {"diff_std":>9}  {"Type"}')
print(f'  {"-"*85}')

# Classify features as artifact-suspect or genuine
ARTIFACT_KEYWORDS = ['range_pos', 'dist_from', 'ret_1d', 'ret_3d', 'ret_1w',
                     'daily', 'session', 'range_width_24', 'range_width_48',
                     'dist_5d', 'pct_rank', 'vr_', 'dow_', 'is_month', 'is_quarter']
GENUINE_KEYWORDS  = ['corr_', 'beta_', 'csi_', 'atr_72', 'peer_', 'relstr_',
                     'vol_trend', 'vol_regime', 'hurst', 'fractal', 'entropy',
                     'realized_skew', 'kyle_', 'vpin']

def classify(feat):
    fl = feat.lower()
    if any(k in fl for k in ARTIFACT_KEYWORDS):
        return 'ARTIFACT?'
    if any(k in fl for k in GENUINE_KEYWORDS):
        return 'genuine'
    return ''

for _, row in res.head(40).iterrows():
    tag = classify(row['feature'])
    marker = ' <<' if tag == 'ARTIFACT?' else ''
    print(f'  {row["feature"]:<40}  {row["mean_h21"]:>10.4f}  {row["mean_other"]:>10.4f}  '
          f'{row["diff_std"]:>+9.3f}  {tag}{marker}')

# ── Summary: artifact vs genuine among top offenders ─────────────────────────
top50 = res.head(50)
top50['type'] = top50['feature'].apply(classify)
n_artifact = (top50['type'] == 'ARTIFACT?').sum()
n_genuine  = (top50['type'] == 'genuine').sum()
n_unknown  = (top50['type'] == '').sum()

print(f'\n{"="*70}')
print(f'  SUMMARY — top 50 most-different features:')
print(f'    ARTIFACT? (daily-reset candidates): {n_artifact}')
print(f'    genuine   (market state features):  {n_genuine}')
print(f'    unknown:                            {n_unknown}')
print(f'{"="*70}')

# ── Check specific artifact suspects explicitly ───────────────────────────────
print(f'\n  Key artifact-suspect features at hour-21:')
suspects = [c for c in feature_cols if any(k in c for k in ['range_pos_24', 'dist_from_24h',
            'ret_1d', 'range_width_24', 'dist_5d_high', 'dist_5d_low'])]
print(f'\n  {"Feature":<35}  {"h21_mean":>10}  {"other_mean":>10}  {"diff_std":>9}')
print(f'  {"-"*68}')
for feat in suspects:
    if feat not in res['feature'].values:
        continue
    row = res[res['feature'] == feat].iloc[0]
    print(f'  {feat:<35}  {row["mean_h21"]:>10.4f}  {row["mean_other"]:>10.4f}  {row["diff_std"]:>+9.3f}')

print(f'\n  Key genuine market-state features at hour-21:')
genuine = [c for c in feature_cols if any(k in c for k in ['corr_usdjpy_1w', 'beta_usdjpy_1w',
           'csi_jpy', 'vol_trend', 'atr_72', 'hurst'])]
print(f'\n  {"Feature":<35}  {"h21_mean":>10}  {"other_mean":>10}  {"diff_std":>9}')
print(f'  {"-"*68}')
for feat in genuine:
    if feat not in res['feature'].values:
        continue
    row = res[res['feature'] == feat].iloc[0]
    print(f'  {feat:<35}  {row["mean_h21"]:>10.4f}  {row["mean_other"]:>10.4f}  {row["diff_std"]:>+9.3f}')
