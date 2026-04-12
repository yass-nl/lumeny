"""
Signal Bar Direction Analysis — Per-Pair
=========================================
For each pair individually: on signal bars (q50_mfe >= 70), which features
best predict whether the next 72h move is LONG or SHORT?

This surfaces pair-specific directional drivers — what explains why
USDJPY is 36% LONG while GBPUSD is 74% LONG at the same time.

Output per pair:
  - Baseline bias (LONG%, avg fwd)
  - Top 10 features by |correlation| with direction
  - Threshold test on top 3 features (quartile buckets)
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from scipy import stats

SCRIPT_DIR     = Path(__file__).parent
FEATURES_DIR   = SCRIPT_DIR / '../backend/data/features_9'
PROCESSED_DIR  = SCRIPT_DIR / '../backend/data/processed'
MFE_MODEL_PATH = SCRIPT_DIR / '../backend/models_9/mfe_q50/model_1H_Q50.joblib'

TRAIN_END   = '2024-06-30'
MFE_THRESH  = 70.0
TOP_N       = 10
MIN_SAMPLES = 30   # minimum signal bars per pair to run analysis
FWD_BARS    = 72   # 72h horizon

JPY_PAIRS = {'USDJPY','EURJPY','GBPJPY','AUDJPY','CADJPY','CHFJPY'}

# ── Load model + features ─────────────────────────────────────────────────────
print('Loading MFE model...')
bundle       = joblib.load(MFE_MODEL_PATH)
mfe_model    = bundle['model']
feature_cols = bundle['feature_cols']
print(f'  {bundle["n_iters"]} iters, {len(feature_cols)} features')

print('\nLoading features_9...')
dfs     = [pd.read_parquet(f) for f in sorted(FEATURES_DIR.glob('*_features.parquet'))]
df      = pd.concat(dfs).sort_index()
df_test = df[df.index > TRAIN_END].copy()
print(f'  Test rows: {len(df_test):,}')

print('Running MFE model...')
X = df_test[feature_cols].ffill().fillna(0)
df_test['q50_mfe'] = mfe_model.predict(X)

# ── Load 1H closes for fwd return ────────────────────────────────────────────
print('\nLoading 1H closes...')
close_all = {}
for pair in df_test['pair'].unique():
    fpath = PROCESSED_DIR / f'{pair}_1H.parquet'
    if fpath.exists():
        close_all[pair] = pd.read_parquet(fpath)['close'].sort_index()

fwd_rows = []
for pair, c in close_all.items():
    fwd = (c.shift(-FWD_BARS) - c)
    pip = 0.01 if pair in JPY_PAIRS else 0.0001
    fwd_pips = fwd / pip
    fwd_rows.append(pd.DataFrame({'pair': pair, 'fwd': fwd_pips}))
fwd_df = pd.concat(fwd_rows)
fwd_df.index.name = 'datetime'

df_test = (
    df_test.reset_index()
    .rename(columns={df_test.reset_index().columns[0]: 'datetime'})
    .merge(fwd_df.reset_index(), on=['datetime','pair'], how='left')
    .set_index('datetime')
)
df_test.index.name = None

# ── Filter to signal bars ─────────────────────────────────────────────────────
df_sig = df_test[
    (df_test['q50_mfe'] >= MFE_THRESH) &
    df_test['fwd'].notna()
].copy()
df_sig['direction'] = np.sign(df_sig['fwd'])

print(f'\n  Total signal bars: {len(df_sig):,}')
print(f'  Pairs: {sorted(df_sig["pair"].unique())}')


def corr_analysis(sub, feature_cols, min_n=20):
    """Run pointbiserialr for each feature against direction. Return sorted DataFrame."""
    y = sub['direction'].values
    rows = []
    for col in feature_cols:
        if col not in sub.columns:
            continue
        x = sub[col].values.astype(float)
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() < min_n:
            continue
        r, p = stats.pointbiserialr(y[mask] > 0, x[mask])
        mean_long  = x[mask & (y > 0)].mean() if (mask & (y > 0)).sum() > 0 else np.nan
        mean_short = x[mask & (y < 0)].mean() if (mask & (y < 0)).sum() > 0 else np.nan
        rows.append({
            'feature':    col,
            'corr':       r,
            'abs_corr':   abs(r),
            'p_value':    p,
            'mean_long':  mean_long,
            'mean_short': mean_short,
            'diff':       mean_long - mean_short if not np.isnan(mean_long) else np.nan,
        })
    return pd.DataFrame(rows).sort_values('abs_corr', ascending=False) if rows else pd.DataFrame()


# ── Per-pair analysis ─────────────────────────────────────────────────────────
pairs = sorted(df_sig['pair'].unique())

for pair in pairs:
    sub = df_sig[df_sig['pair'] == pair].copy()
    n   = len(sub)
    if n < MIN_SAMPLES:
        print(f'\n  {pair}: only {n} signal bars — skipping')
        continue

    pct_long = (sub['direction'] == 1).mean()
    avg_fwd  = sub['fwd'].mean()
    med_fwd  = sub['fwd'].median()
    bias     = 'LONG ' if pct_long > 0.55 else ('SHORT' if pct_long < 0.45 else 'FLAT ')

    print(f'\n{"="*70}')
    print(f'  {pair}   N={n}   LONG={pct_long:.1%}   avg={avg_fwd:+.1f}p   med={med_fwd:+.1f}p   [{bias}]')
    print(f'{"="*70}')

    res = corr_analysis(sub, feature_cols, min_n=20)
    if res.empty:
        print('  (not enough data for any feature)')
        continue

    # Top features table
    print(f'\n  {"Feature":<40}  {"corr":>7}  {"p":>8}  {"mean_L":>9}  {"mean_S":>9}  {"diff":>9}')
    print(f'  {"-"*88}')
    for _, row in res.head(TOP_N).iterrows():
        sig = '***' if row['p_value'] < 0.001 else ('** ' if row['p_value'] < 0.01 else ('*  ' if row['p_value'] < 0.05 else '   '))
        print(f'  {row["feature"]:<40}  {row["corr"]:>+7.4f}  {sig}  '
              f'{row["mean_long"]:>+9.4f}  {row["mean_short"]:>+9.4f}  {row["diff"]:>+9.4f}')

    # Threshold test on top 3
    print(f'\n  Threshold test (top 3 features):')
    for feat in res.head(3)['feature'].tolist():
        if feat not in sub.columns:
            continue
        vals = sub[feat].dropna()
        if len(vals) < 20:
            continue
        p25, p50, p75 = vals.quantile([0.25, 0.5, 0.75])
        corr_val = res.loc[res['feature'] == feat, 'corr'].values[0]
        print(f'\n    {feat}  (corr={corr_val:+.4f}  p25={p25:.4f}  p50={p50:.4f}  p75={p75:.4f})')
        print(f'    {"Bucket":<22} {"N":>5}  {"LONG%":>7}  {"Avg_fwd":>9}')
        print(f'    {"-"*48}')
        buckets = [
            (f'< p25 ({p25:.3f})',  sub[sub[feat] <  p25]),
            (f'p25-p50',            sub[(sub[feat] >= p25) & (sub[feat] < p50)]),
            (f'p50-p75',            sub[(sub[feat] >= p50) & (sub[feat] < p75)]),
            (f'> p75 ({p75:.3f})',  sub[sub[feat] >= p75]),
        ]
        for label, bucket in buckets:
            if len(bucket) < 5:
                continue
            pl  = (bucket['direction'] == 1).mean()
            avg = bucket['fwd'].mean()
            print(f'    {label:<22} {len(bucket):>5}  {pl:>7.1%}  {avg:>+9.1f}p')

print(f'\n{"="*70}')
print(f'  DONE')
print(f'{"="*70}')
