"""
8h Model — Directional Edge Analysis
======================================
Among bars where Q50_8h > threshold (model says big move coming),
compare feature values between:
  - LONG-dominant bars: mfe_long_8h > mfe_short_8h (price ran more up)
  - SHORT-dominant bars: mfe_short_8h > mfe_long_8h (price ran more down)

Features with large, consistent difference between long/short populations
are candidates for directional filters.

Uses within-ATR-bucket analysis to ensure signal is not just vol.
"""

import joblib, pandas as pd, numpy as np
from pathlib import Path
from scipy import stats

TRAIN_END    = '2024-06-30'
Q50_THRESH   = 30
FEATURES_DIR = Path('backend/data/features_9')

MICRO_FEATURES = [
    'kyle_lambda', 'kyle_lambda_ma12', 'kyle_lambda_change',
    'kyle_lambda_delta_3h', 'kyle_lambda_delta_6h', 'kyle_lambda_delta_12h',
    'order_imbalance', 'order_imbalance_intensity', 'buy_volume_frac',
    'amihud_illiq', 'volume_cv',
    'entropy_returns', 'entropy_norm', 'entropy_volume_divergence', 'kl_proxy',
    'autocorr_1', 'autocorr_2', 'autocorr_5', 'sum_abs_autocorr', 'noise_to_signal',
    'fractal_dim', 'vr_5', 'vr_10', 'vr_z5', 'runs_z',
    'realized_skew', 'realized_kurt', 'tail_ratio_95_5', 'hill_tail_index',
    'accel_mean', 'accel_std', 'accel_skew',
    'momentum_shift', 'ret_concentration', 'vol_clustering_ac1',
    'epps_1m_5m', 'epps_1m_15m', 'epps_5m_15m',
    'info_accel', 'range_return_ratio',
    'jump_ratio', 'jump_z', 'jump_intensity', 'jump_mean_size', 'jump_asymmetry',
    'rv_close', 'rv_parkinson', 'rv_garman_klass', 'rv_rogers_satchell', 'rv_yang_zhang',
    # session/calendar
    'is_ny', 'is_asia', 'is_london',
    'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',
    # momentum
    'ret_1h', 'ret_4h', 'ret_1d', 'ret_2d', 'ret_1w', 'ret_2w',
    'mom_4h', 'mom_12h', 'mom_24h',
    # structural
    'dist_from_24h_high', 'dist_from_24h_low',
    'range_width_24', 'rv_zscore_24',
    'atr_ratio_6_24', 'atr_ratio_6_72',
]

# ── Load model and data ───────────────────────────────────────────────────────
bundle       = joblib.load('backend/models_9/mfe_q50_8h/model_1H_Q50.joblib')
model        = bundle['model']
feature_cols = bundle['feature_cols']

print('Loading data...')
dfs = [pd.read_parquet(f) for f in sorted(FEATURES_DIR.glob('*_features.parquet'))]
df  = pd.concat(dfs).sort_index()
print(f'  {len(df):,} rows')

# 8h MFE targets
df['mfe_long_8h']  = df['mfe_long_pips'].where(df['trail_long_bars']  <= 8)
df['mfe_short_8h'] = df['mfe_short_pips'].where(df['trail_short_bars'] <= 8)

# Test set only (unseen)
df_test = df[df.index > TRAIN_END].copy()
df_test = df_test.dropna(subset=['mfe_long_8h', 'mfe_short_8h', 'atr_24'])
print(f'  {len(df_test):,} test rows with valid 8h MFE')

# Predict Q50
for col in feature_cols:
    if col not in df_test.columns:
        df_test[col] = 0.0
X    = df_test[feature_cols].ffill().fillna(0)
df_test['q50'] = model.predict(X)

# Filter: only high-Q50 bars (model says big move coming)
df_sig = df_test[df_test['q50'] >= Q50_THRESH].copy()
print(f'  {len(df_sig):,} bars with Q50 >= {Q50_THRESH}')

# Label direction: long-dominant vs short-dominant
df_sig['long_dom']  = df_sig['mfe_long_8h']  > df_sig['mfe_short_8h']
df_sig['short_dom'] = df_sig['mfe_short_8h'] > df_sig['mfe_long_8h']
df_sig['margin']    = (df_sig['mfe_long_8h'] - df_sig['mfe_short_8h']).abs()

long_bars  = df_sig[df_sig['long_dom']]
short_bars = df_sig[df_sig['short_dom']]
print(f'  Long-dominant:  {len(long_bars):,} ({len(long_bars)/len(df_sig):.1%})')
print(f'  Short-dominant: {len(short_bars):,} ({len(short_bars)/len(df_sig):.1%})')

# ── Feature comparison ────────────────────────────────────────────────────────
avail = [f for f in MICRO_FEATURES if f in df_sig.columns]

results = []
for feat in avail:
    l = long_bars[feat].dropna()
    s = short_bars[feat].dropna()
    if len(l) < 50 or len(s) < 50:
        continue

    # Mean difference (long - short), normalized by pooled std
    mean_l = l.mean()
    mean_s = s.mean()
    pooled_std = np.sqrt((l.std()**2 + s.std()**2) / 2)
    effect_size = (mean_l - mean_s) / pooled_std if pooled_std > 0 else 0

    # T-test
    t, p = stats.ttest_ind(l, s, equal_var=False)

    results.append({
        'feature':     feat,
        'mean_long':   mean_l,
        'mean_short':  mean_s,
        'diff':        mean_l - mean_s,
        'effect_size': effect_size,
        'p_value':     p,
    })

df_res = pd.DataFrame(results).sort_values('effect_size', key=abs, ascending=False)

print(f'\n{"="*85}')
print(f'  FEATURE VALUES: LONG-DOMINANT vs SHORT-DOMINANT BARS (Q50 >= {Q50_THRESH})')
print(f'  Effect size = (mean_long - mean_short) / pooled_std')
print(f'  Positive = feature higher when price goes up. Negative = higher when price goes down.')
print(f'{"="*85}')
print(f'  {"Feature":<35} {"EffectSz":>9} {"Mean_L":>10} {"Mean_S":>10} {"p-value":>12}')
print(f'  {"-"*80}')

for _, row in df_res.iterrows():
    sig = '***' if row['p_value'] < 0.001 else '**' if row['p_value'] < 0.01 else '*' if row['p_value'] < 0.05 else ''
    print(f'  {row["feature"]:<35} {row["effect_size"]:>9.4f} {row["mean_long"]:>10.4f} {row["mean_short"]:>10.4f} {row["p_value"]:>12.2e} {sig}')


# ── Within-ATR-bucket consistency ─────────────────────────────────────────────
print(f'\n{"="*85}')
print(f'  TOP FEATURES — WITHIN-ATR-BUCKET CONSISTENCY')
print(f'  Does the long/short difference hold across all vol regimes?')
print(f'{"="*85}')

df_sig['atr_q'] = pd.qcut(df_sig['atr_24'], q=5, labels=['Q1','Q2','Q3','Q4','Q5'])
top_features = df_res[df_res['p_value'] < 0.001].head(20)['feature'].tolist()

print(f'  {"Feature":<35} {"Avg_ES":>8} {"Cons":>6}   {"Q1":>7} {"Q2":>7} {"Q3":>7} {"Q4":>7} {"Q5":>7}')
print(f'  {"-"*83}')

for feat in top_features:
    bucket_es = []
    for b in ['Q1','Q2','Q3','Q4','Q5']:
        sub = df_sig[df_sig['atr_q']==b]
        l = sub[sub['long_dom']][feat].dropna()
        s = sub[sub['short_dom']][feat].dropna()
        if len(l) < 20 or len(s) < 20:
            bucket_es.append(np.nan)
            continue
        pstd = np.sqrt((l.std()**2 + s.std()**2) / 2)
        es = (l.mean() - s.mean()) / pstd if pstd > 0 else 0
        bucket_es.append(es)

    avg_es = np.nanmean(bucket_es)
    signs  = [np.sign(e) == np.sign(avg_es) for e in bucket_es if not np.isnan(e)]
    cons   = np.mean(signs)
    flag   = ' ***' if cons == 1.0 and abs(avg_es) > 0.05 else ''
    vals   = '  '.join([f'{e:>7.4f}' if not np.isnan(e) else '    nan' for e in bucket_es])
    print(f'  {feat:<35} {avg_es:>8.4f} {cons:>6.0%}   {vals}{flag}')


# ── Summary: best directional candidates ──────────────────────────────────────
print(f'\n{"="*85}')
print(f'  BEST DIRECTIONAL CANDIDATES')
print(f'  Consistent across all ATR buckets, effect size > 0.05')
print(f'{"="*85}')

for feat in top_features:
    bucket_es = []
    for b in ['Q1','Q2','Q3','Q4','Q5']:
        sub = df_sig[df_sig['atr_q']==b]
        l = sub[sub['long_dom']][feat].dropna()
        s = sub[sub['short_dom']][feat].dropna()
        if len(l) < 20 or len(s) < 20:
            bucket_es.append(np.nan); continue
        pstd = np.sqrt((l.std()**2 + s.std()**2) / 2)
        bucket_es.append((l.mean()-s.mean())/pstd if pstd > 0 else 0)
    avg_es = np.nanmean(bucket_es)
    signs  = [np.sign(e)==np.sign(avg_es) for e in bucket_es if not np.isnan(e)]
    if np.mean(signs)==1.0 and abs(avg_es) > 0.05:
        direction = 'LONG when HIGH' if avg_es > 0 else 'LONG when LOW'
        print(f'  {feat:<35} effect={avg_es:+.4f}  -> {direction}')
