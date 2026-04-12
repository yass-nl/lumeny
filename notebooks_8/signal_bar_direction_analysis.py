"""
Signal Bar Direction Analysis
==============================
On bars where MFE model fires (q50_mfe >= 70), what features correlate
most strongly with the direction of the subsequent 72h move?

This identifies what market structure looks like when big moves happen
AND they go up vs down — the potential source of directional edge.

Output:
  1. Top features correlated with fwd_72h direction on signal bars
  2. Same broken down by pair (is the bias pair-specific or universal?)
  3. Distribution of signal bars over time (is it regime-clustered?)
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

TRAIN_END  = '2024-06-30'
MFE_THRESH = 70.0
TOP_N      = 30

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

# Build fwd_8h per row
fwd_rows = []
for pair, c in close_all.items():
    fwd = (c.shift(-72) - c)
    pip = 0.01 if pair in JPY_PAIRS else 0.0001
    fwd_pips = fwd / pip
    fwd_rows.append(pd.DataFrame({'pair': pair, 'fwd_8h': fwd_pips}))  # col name kept for compatibility
fwd_df = pd.concat(fwd_rows)
fwd_df.index.name = 'datetime'

# Join to test set
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
    df_test['fwd_8h'].notna()
].copy()

df_sig['direction'] = np.sign(df_sig['fwd_8h'])  # +1 = up, -1 = down

print(f'\n  Signal bars: {len(df_sig):,}')
print(f'  LONG  (fwd>0): {(df_sig["direction"]==1).sum()} ({(df_sig["direction"]==1).mean():.1%})')
print(f'  SHORT (fwd<0): {(df_sig["direction"]==-1).sum()} ({(df_sig["direction"]==-1).mean():.1%})')
print(f'  Avg fwd_8h: {df_sig["fwd_8h"].mean():+.1f}p  median: {df_sig["fwd_8h"].median():+.1f}p')


# ── 1. Feature correlation with direction ─────────────────────────────────────
print(f'\n{"="*70}')
print(f'  TOP {TOP_N} FEATURES CORRELATED WITH FWD_8H DIRECTION')
print(f'  (on signal bars only, N={len(df_sig):,})')
print(f'{"="*70}')

y = df_sig['direction'].values  # +1/-1

results = []
for col in feature_cols:
    if col not in df_sig.columns:
        continue
    x = df_sig[col].values.astype(float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 50:
        continue
    r, p = stats.pointbiserialr(y[mask] > 0, x[mask])
    # Also compute mean of feature when LONG vs SHORT
    mean_long  = x[mask & (y > 0)].mean()
    mean_short = x[mask & (y < 0)].mean()
    results.append({
        'feature':     col,
        'corr':        r,
        'abs_corr':    abs(r),
        'p_value':     p,
        'mean_long':   mean_long,
        'mean_short':  mean_short,
        'diff':        mean_long - mean_short,
    })

res = pd.DataFrame(results).sort_values('abs_corr', ascending=False)

print(f'\n  {"Feature":<40}  {"corr":>7}  {"p":>8}  {"mean_L":>9}  {"mean_S":>9}  {"diff":>9}')
print(f'  {"-"*88}')
for _, row in res.head(TOP_N).iterrows():
    sig = '***' if row['p_value'] < 0.001 else ('** ' if row['p_value'] < 0.01 else '*  ')
    print(f'  {row["feature"]:<40}  {row["corr"]:>+7.4f}  {sig}  '
          f'{row["mean_long"]:>+9.4f}  {row["mean_short"]:>+9.4f}  {row["diff"]:>+9.4f}')


# ── 2. Per-pair direction bias ────────────────────────────────────────────────
print(f'\n{"="*70}')
print(f'  PER-PAIR DIRECTION BIAS ON SIGNAL BARS')
print(f'{"="*70}')
print(f'  {"Pair":<10} {"N":>5}  {"LONG%":>6}  {"Avg_fwd":>9}  {"Med_fwd":>9}  {"Bias"}')
print(f'  {"-"*58}')
for pair in sorted(df_sig['pair'].unique()):
    sub = df_sig[df_sig['pair'] == pair]
    pct_long = (sub['direction'] == 1).mean()
    avg_fwd  = sub['fwd_8h'].mean()
    med_fwd  = sub['fwd_8h'].median()
    bias = 'LONG ' if pct_long > 0.55 else ('SHORT' if pct_long < 0.45 else 'FLAT ')
    print(f'  {pair:<10} {len(sub):>5}  {pct_long:>6.1%}  {avg_fwd:>+9.1f}p  {med_fwd:>+9.1f}p  {bias}')


# ── 3. Time distribution of signal bars ──────────────────────────────────────
print(f'\n{"="*70}')
print(f'  SIGNAL BAR DISTRIBUTION OVER TIME')
print(f'  (are signals clustered in specific regimes?)')
print(f'{"="*70}')
df_sig['month'] = pd.to_datetime(df_sig.index).to_period('M')
monthly = df_sig.groupby('month').agg(
    N=('fwd_8h','count'),
    pct_long=('direction', lambda x: (x==1).mean()),
    avg_fwd=('fwd_8h','mean'),
).reset_index()
print(f'\n  {"Month":<10} {"N":>5}  {"LONG%":>7}  {"Avg_fwd":>9}')
print(f'  {"-"*36}')
for _, row in monthly.iterrows():
    bar = '+' * min(int(row['N'] / 5), 30)
    print(f'  {str(row["month"]):<10} {int(row["N"]):>5}  {row["pct_long"]:>7.1%}  {row["avg_fwd"]:>+9.1f}p  {bar}')


# ── 4. Top features breakdown by pair ────────────────────────────────────────
top_features = res.head(10)['feature'].tolist()

print(f'\n{"="*70}')
print(f'  TOP 10 FEATURES — VALUE WHEN LONG vs SHORT, BY PAIR')
print(f'{"="*70}')

for feat in top_features:
    if feat not in df_sig.columns:
        continue
    print(f'\n  {feat}  (overall corr={res.loc[res["feature"]==feat,"corr"].values[0]:+.4f})')
    print(f'  {"Pair":<10} {"N_L":>5}  {"N_S":>5}  {"Mean_L":>9}  {"Mean_S":>9}  {"Diff":>9}')
    print(f'  {"-"*52}')
    for pair in sorted(df_sig['pair'].unique()):
        sub = df_sig[df_sig['pair'] == pair]
        longs  = sub.loc[sub['direction']==1,  feat].dropna()
        shorts = sub.loc[sub['direction']==-1, feat].dropna()
        if len(longs) < 5 or len(shorts) < 5:
            continue
        diff = longs.mean() - shorts.mean()
        print(f'  {pair:<10} {len(longs):>5}  {len(shorts):>5}  '
              f'{longs.mean():>+9.4f}  {shorts.mean():>+9.4f}  {diff:>+9.4f}')


# ── 5. Feature threshold test ─────────────────────────────────────────────────
print(f'\n{"="*70}')
print(f'  THRESHOLD TEST — TOP 5 FEATURES')
print(f'  (does splitting on feature value create directional edge?)')
print(f'{"="*70}')

for feat in res.head(5)['feature'].tolist():
    if feat not in df_sig.columns:
        continue
    vals = df_sig[feat].dropna()
    p25, p50, p75 = vals.quantile([0.25, 0.5, 0.75])
    corr_sign = res.loc[res['feature']==feat,'corr'].values[0]

    print(f'\n  {feat}  (corr={corr_sign:+.4f}  p25={p25:.4f}  p50={p50:.4f}  p75={p75:.4f})')
    print(f'  {"Bucket":<20} {"N":>5}  {"LONG%":>7}  {"Avg_fwd":>9}')
    print(f'  {"-"*46}')

    buckets = [
        (f'< p25 ({p25:.3f})',  df_sig[df_sig[feat] <  p25]),
        (f'p25-p50',            df_sig[(df_sig[feat] >= p25) & (df_sig[feat] < p50)]),
        (f'p50-p75',            df_sig[(df_sig[feat] >= p50) & (df_sig[feat] < p75)]),
        (f'> p75 ({p75:.3f})',  df_sig[df_sig[feat] >= p75]),
    ]
    for label, sub in buckets:
        if len(sub) < 10:
            continue
        pct_long = (sub['direction'] == 1).mean()
        avg_fwd  = sub['fwd_8h'].mean()
        print(f'  {label:<20} {len(sub):>5}  {pct_long:>7.1%}  {avg_fwd:>+9.1f}p')
