"""
SHAP Direction Analysis — Q50 model
Answers: which features drive DIRECTION (UP vs DOWN) vs REGIME (confidence level)?

Method:
1. Load Q50 model + training data
2. Sample 5000 rows where |Q50 prediction| > threshold (tradeable signals only)
3. Compute SHAP values
4. For each feature: compute mean SHAP when prediction is UP vs DOWN
   - Regime features: SHAP contribution is large regardless of direction (same sign both ways, or large absolute)
   - Direction features: SHAP flips sign between UP and DOWN predictions

Output:
- Table sorted by |SHAP_up - SHAP_down| — top = most direction-sensitive features
- Table sorted by |SHAP_up + SHAP_down| / 2 — top = most regime/confidence features
- Hour-of-day breakdown: mean |SHAP| for top direction features, to see if they peak at 21h
"""

import pandas as pd
import numpy as np
import joblib
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

import shap

# ── Load model ──
print('Loading Q50 model...')
bundle   = joblib.load('backend/models_6/3_quants/model_1H_Q50.joblib')
model    = bundle['model']
feat_cols = model.feature_name_

# ── Load training data ──
print('Loading features...')
df_maj = pd.read_parquet('backend/data/features_2/all_pairs_microstructure.parquet')
df_cro = pd.read_parquet('backend/data/features_3/all_pairs_microstructure.parquet')
df_all = pd.concat([df_maj, df_cro]).sort_index()
del df_maj, df_cro

print(f'Total rows: {len(df_all):,}  ({df_all.index.min().date()} to {df_all.index.max().date()})')

# Keep only feature columns that exist
available = [c for c in feat_cols if c in df_all.columns]
missing   = [c for c in feat_cols if c not in df_all.columns]
if missing:
    print(f'Missing {len(missing)} feature cols (will fill 0): {missing[:5]}...')

X = df_all[available].reindex(columns=feat_cols, fill_value=0).ffill().fillna(0)

# ── Run inference to get Q50 predictions ──
print('Running Q50 inference...')
q50_pred = model.predict(X)
abs_q50  = np.abs(q50_pred)

# ── Sample tradeable rows ──
AVG_SPREAD = 0.00028
MIN_Q50    = AVG_SPREAD * 0.7  # same as sim threshold

tradeable_mask = abs_q50 > MIN_Q50
print(f'Tradeable rows (|Q50|>{MIN_Q50:.6f}): {tradeable_mask.sum():,} ({tradeable_mask.mean():.1%} of all)')

# Sample 5000 for SHAP (TreeExplainer is fast but 1M rows would be slow)
rng = np.random.default_rng(42)
tradeable_idx = np.where(tradeable_mask)[0]
sample_n = min(5000, len(tradeable_idx))
sampled  = rng.choice(tradeable_idx, size=sample_n, replace=False)
sampled  = np.sort(sampled)

X_sample  = X.iloc[sampled]
q50_sample = q50_pred[sampled]
hour_sample = df_all.index[sampled].hour

print(f'SHAP sample: {sample_n:,} rows')

# ── Compute SHAP values ──
print('Computing SHAP values (TreeExplainer)...')
explainer   = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_sample)  # shape: (n_samples, n_features)
print('Done.')

shap_df = pd.DataFrame(shap_values, columns=feat_cols, index=X_sample.index)
shap_df['q50_pred'] = q50_sample
shap_df['hour']     = hour_sample

# ── Split UP vs DOWN ──
up_mask   = shap_df['q50_pred'] > 0
down_mask = shap_df['q50_pred'] < 0

shap_up   = shap_df[up_mask][feat_cols].mean()
shap_down = shap_df[down_mask][feat_cols].mean()
shap_abs  = shap_df[feat_cols].abs().mean()

# Build analysis table
results = pd.DataFrame({
    'mean_shap_up':   shap_up,
    'mean_shap_down': shap_down,
    'mean_abs_shap':  shap_abs,
})
results['direction_sensitivity'] = (results['mean_shap_up'] - results['mean_shap_down']).abs()
results['regime_signal']         = (results['mean_shap_up'] + results['mean_shap_down']).abs() / 2
results['direction_ratio']       = results['direction_sensitivity'] / (results['mean_abs_shap'] + 1e-12)

# ── Print: Direction-driving features ──
print(f'\n{"="*90}')
print('TOP DIRECTION FEATURES (SHAP flips sign between UP and DOWN predictions)')
print('High direction_sensitivity = feature says "UP" when prediction is UP, "DOWN" when DOWN')
print(f'{"="*90}')
print(f'{"Feature":<30} {"SHAP(UP)":>12} {"SHAP(DOWN)":>12} {"Flip magnitude":>16} {"Dir ratio":>10}')
print('-' * 82)
top_dir = results.sort_values('direction_sensitivity', ascending=False).head(20)
for feat, row in top_dir.iterrows():
    print(f'{feat:<30} {row.mean_shap_up:>12.6f} {row.mean_shap_down:>12.6f} {row.direction_sensitivity:>16.6f} {row.direction_ratio:>10.3f}')

# ── Print: Regime/confidence features ──
print(f'\n{"="*90}')
print('TOP REGIME FEATURES (SHAP is large and same-direction regardless of UP/DOWN prediction)')
print('High regime_signal = feature boosts or suppresses confidence regardless of direction')
print(f'{"="*90}')
print(f'{"Feature":<30} {"SHAP(UP)":>12} {"SHAP(DOWN)":>12} {"Regime signal":>15} {"Mean |SHAP|":>12}')
print('-' * 82)
top_reg = results.sort_values('regime_signal', ascending=False).head(20)
for feat, row in top_reg.iterrows():
    print(f'{feat:<30} {row.mean_shap_up:>12.6f} {row.mean_shap_down:>12.6f} {row.regime_signal:>15.6f} {row.mean_abs_shap:>12.6f}')

# ── Hour breakdown for top direction + regime features ──
TOP_DIR_FEATS = list(top_dir.head(6).index)
TOP_REG_FEATS = list(top_reg.head(6).index)
FOCUS_FEATS   = list(dict.fromkeys(TOP_DIR_FEATS + TOP_REG_FEATS))  # deduplicated

print(f'\n{"="*90}')
print('MEAN |SHAP| BY HOUR — do these features peak at 21h UTC?')
print(f'{"="*90}')
header = f'{"Hour":>6} {"N":>6}' + ''.join(f'{f[:13]:>15}' for f in FOCUS_FEATS)
print(header)
print('-' * (14 + 15 * len(FOCUS_FEATS)))

EDGE_HOURS = {18, 19, 20, 21, 22}
for h in range(24):
    mask = shap_df['hour'] == h
    sub  = shap_df[mask]
    if len(sub) == 0:
        continue
    vals = [sub[f].abs().mean() if f in sub.columns else float('nan') for f in FOCUS_FEATS]
    row  = f'{h:>6} {len(sub):>6}' + ''.join(f'{v:>15.6f}' for v in vals)
    flag = ' <<<' if h in EDGE_HOURS else ''
    print(row + flag)

# ── Correlation between SHAP of regime vs direction features at each hour ──
print(f'\n{"="*90}')
print('PER-HOUR: mean |SHAP| for direction features vs regime features')
print('If direction SHAP peaks at 21h but regime SHAP does not — direction is hour-dependent')
print('If both peak at 21h — the model requires BOTH simultaneously')
print(f'{"="*90}')
print(f'{"Hour":>6} {"N":>6} {"Dir SHAP":>12} {"Regime SHAP":>14} {"Ratio D/R":>12}')
print('-' * 54)
for h in range(24):
    mask = shap_df['hour'] == h
    sub  = shap_df[mask]
    if len(sub) == 0:
        continue
    dir_shap = sub[[f for f in TOP_DIR_FEATS if f in sub.columns]].abs().mean().mean()
    reg_shap = sub[[f for f in TOP_REG_FEATS if f in sub.columns]].abs().mean().mean()
    ratio    = dir_shap / (reg_shap + 1e-12)
    flag     = ' <<<' if h in EDGE_HOURS else ''
    print(f'{h:>6} {len(sub):>6} {dir_shap:>12.6f} {reg_shap:>14.6f} {ratio:>12.4f}{flag}')

print()
print('<<< = edge window (18-22h UTC)')
print()
print('KEY QUESTION: if Dir/Regime ratio is similar across hours,')
print('the model uses them proportionally everywhere — suggesting')
print('per-hour thresholds could unlock the same structure off-hours.')

# ── Explicit check: where do volume_cv and epps_1m_15m rank? ──
print(f'\n{"="*90}')
print('EXPLICIT CHECK: volume_cv and epps_1m_15m SHAP rank and values')
print('(These were top feature-importance features — why not in top SHAP?)')
print(f'{"="*90}')

results_full = results.copy()
results_full['rank_dir']    = results_full['direction_sensitivity'].rank(ascending=False).astype(int)
results_full['rank_regime'] = results_full['regime_signal'].rank(ascending=False).astype(int)
results_full['rank_abs']    = results_full['mean_abs_shap'].rank(ascending=False).astype(int)

for feat in ['volume_cv', 'epps_1m_15m', 'kyle_lambda_change', 'accel_mean', 'momentum_shift']:
    if feat not in results_full.index:
        print(f'{feat:<30}  NOT IN MODEL FEATURES (filled with 0)')
        continue
    row = results_full.loc[feat]
    print(f'{feat:<30}  SHAP(UP)={row.mean_shap_up:+.6f}  SHAP(DOWN)={row.mean_shap_down:+.6f}  '
          f'|SHAP|={row.mean_abs_shap:.6f}  rank_abs=#{row.rank_abs}  rank_dir=#{row.rank_dir}  rank_regime=#{row.rank_regime}')

# Also show distribution of volume_cv in tradeable vs all
print()
print(f'volume_cv in FULL dataset (all rows):')
if 'volume_cv' in df_all.columns:
    vc_all = df_all['volume_cv'].dropna()
    print(f'  mean={vc_all.mean():.4f}  median={vc_all.median():.4f}  p75={vc_all.quantile(0.75):.4f}  p90={vc_all.quantile(0.90):.4f}')
    vc_trade = df_all.iloc[sampled]['volume_cv'].dropna()
    print(f'volume_cv in TRADEABLE sample (|Q50|>threshold):')
    print(f'  mean={vc_trade.mean():.4f}  median={vc_trade.median():.4f}  p75={vc_trade.quantile(0.75):.4f}  p90={vc_trade.quantile(0.90):.4f}')
    print(f'  => selection uplift: {vc_trade.mean()/vc_all.mean():.2f}x higher in tradeable sample')
if 'epps_1m_15m' in df_all.columns:
    ep_all   = df_all['epps_1m_15m'].dropna()
    ep_trade = df_all.iloc[sampled]['epps_1m_15m'].dropna()
    print(f'epps_1m_15m in FULL dataset:       mean={ep_all.mean():.4f}  median={ep_all.median():.4f}')
    print(f'epps_1m_15m in TRADEABLE sample:   mean={ep_trade.mean():.4f}  median={ep_trade.median():.4f}')
    print(f'  => selection uplift: {ep_trade.mean()/ep_all.mean():.2f}x higher in tradeable sample')
