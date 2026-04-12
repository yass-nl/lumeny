"""
Direction System — All Bars Test
==================================
Tests the SAME pair-specific directional rules crafted in notebooks_8
but applied to ALL bars (not just MFE signal bars).

Goal: measure raw directional accuracy of the regime indicators
when triggered everywhere, not just where MFE model fires.

Method:
  - Walk every bar per pair (last 18 months, i.e. since ~2024-10-11)
  - Pre-filter: MFE q50 score >= 30 AND hour between 7-20 UTC
  - Apply pair-specific direction rules to get LONG / SHORT / skip
  - Apply 72h cooldown per pair (one signal per 72h)
  - Measure:
      - Directional accuracy (% correct vs fwd_72h sign)
      - Average 72h move in pip in the predicted direction
      - Distribution by LONG vs SHORT calls

Pairs and rules (from RELAXED version):
  USDJPY  -> always SHORT
  AUDUSD  -> LONG if beta_gbpusd_1w > 0.775 OR atr_24 < 40.8
  GBPUSD  -> LONG if csi_usd_24h < 0.004
  EURUSD  -> LONG if corr_audusd_24h < 0.22
  NZDUSD  -> LONG if dist_5d_high > 0.35
  USDCHF  -> LONG if corr_eurusd_1w > -0.60
  CHFJPY  -> LONG if corr_usdjpy_1w > 0.40, SHORT if < 0.26, else skip
  CADJPY  -> LONG if vol_trend < 1.15, SHORT if >= 1.15
  AUDJPY  -> LONG if beta_usdjpy_1w > 0.74
  EURJPY  -> LONG if beta_eurusd_1w > 0.38
  GBPJPY  -> LONG if beta_eurusd_1w > 0.50  (proxy from EURJPY pattern)
  EURAUD  -> LONG if corr_audusd_24h < 0.22 (EUR driver same as EURUSD)
  AUDNZD  -> LONG if corr_regime_audusd > 0.0
  EURGBP  -> SHORT if csi_usd_24h > 0.004 (EUR/GBP dynamics)
"""

import numpy as np
import pandas as pd
from pathlib import Path

SCRIPT_DIR    = Path(__file__).parent
FEATURES_DIR  = SCRIPT_DIR / '../backend/data/features_9'
PROCESSED_DIR = SCRIPT_DIR / '../backend/data/processed'

# Last 18 months from today (2026-04-11)
START_DATE = '2024-10-11'
COOLDOWN_H = 72
FWD_BARS   = 72

JPY_PAIRS = {'USDJPY', 'EURJPY', 'GBPJPY', 'AUDJPY', 'CADJPY', 'CHFJPY'}

MFE_MODEL_PATH = SCRIPT_DIR / '../backend/models_9/mfe_q50/model_1H_Q50.joblib'
MFE_THRESH     = 50.0
HOURS_ALLOWED  = set(range(7, 21))   # 7-20 UTC inclusive

PAIRS_ALL = [
    'EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD', 'USDCAD', 'NZDUSD',
    'EURJPY', 'GBPJPY', 'EURGBP', 'EURAUD', 'AUDJPY', 'CADJPY', 'CHFJPY', 'AUDNZD',
]

# ── Load MFE model ────────────────────────────────────────────────────────────
import joblib
print('Loading MFE model...')
bundle       = joblib.load(MFE_MODEL_PATH)
mfe_model    = bundle['model']
feature_cols = bundle['feature_cols']
print(f'  {bundle["n_iters"]} iters, {len(feature_cols)} features')

# ── Load features_9 ───────────────────────────────────────────────────────────
print('Loading features_9...')
dfs = [pd.read_parquet(f) for f in sorted(FEATURES_DIR.glob('*_features.parquet'))]
df  = pd.concat(dfs).sort_index()
df  = df[df.index >= START_DATE].copy()
print(f'  Rows: {len(df):,}  |  period: {df.index.min().date()} to {df.index.max().date()}')
print(f'  Pairs: {sorted(df["pair"].unique())}')

# ── Load 1H closes for fwd return ────────────────────────────────────────────
print('\nLoading 1H closes for forward returns...')
fwd_frames = []
for pair in PAIRS_ALL:
    fpath = PROCESSED_DIR / f'{pair}_1H.parquet'
    if not fpath.exists():
        print(f'  Missing: {fpath}')
        continue
    c   = pd.read_parquet(fpath)['close'].sort_index()
    pip = 0.01 if pair in JPY_PAIRS else 0.0001
    fwd_72h = (c.shift(-FWD_BARS) - c) / pip
    fwd_frames.append(pd.DataFrame({'pair': pair, 'fwd_72h': fwd_72h}))

fwd_df = pd.concat(fwd_frames)
fwd_df.index.name = 'datetime'

# Merge fwd_72h into main df
df = (
    df.reset_index()
    .rename(columns={df.reset_index().columns[0]: 'datetime'})
    .merge(fwd_df.reset_index(), on=['datetime', 'pair'], how='left')
    .set_index('datetime')
)
df.index.name = None
print(f'  fwd_72h non-null: {df["fwd_72h"].notna().sum():,}')

# ── Run MFE model + hour filter ───────────────────────────────────────────────
print('\nRunning MFE model...')
X = df[feature_cols].ffill().fillna(0)
df['q50_mfe'] = mfe_model.predict(X)

df['hour'] = pd.to_datetime(df.index).hour
n_before = len(df)
df = df[(df['q50_mfe'] >= MFE_THRESH) & df['hour'].isin(HOURS_ALLOWED)].copy()
print(f'  After MFE>=30 + hours 7-20: {len(df):,} bars  (was {n_before:,})')

# ── Direction rules ────────────────────────────────────────────────────────────

def apply_direction_rules(df):
    """
    Apply pair-specific directional rules.
    Returns pd.Series with values: +1 (LONG), -1 (SHORT), NaN (skip).
    """
    dirs = pd.Series(np.nan, index=df.index)
    pair = df['pair']

    # USDJPY -> always SHORT
    dirs = dirs.where(pair != 'USDJPY', -1.0)

    # AUDUSD -> LONG if beta_gbpusd_1w > 0.775 OR atr_24 < 40.8
    m = pair == 'AUDUSD'
    long_cond = m & (df.get('beta_gbpusd_1w', pd.Series(np.nan, index=df.index)).gt(0.775) |
                     df.get('atr_24',         pd.Series(np.nan, index=df.index)).lt(40.8))
    dirs = dirs.where(~long_cond, 1.0).where(~(m & ~long_cond), np.nan)

    # GBPUSD -> LONG if csi_usd_24h < 0.004
    m = pair == 'GBPUSD'
    long_cond = m & df.get('csi_usd_24h', pd.Series(np.nan, index=df.index)).lt(0.004)
    dirs = dirs.where(~long_cond, 1.0).where(~(m & ~long_cond), np.nan)

    # EURUSD -> LONG if corr_audusd_24h < 0.22
    m = pair == 'EURUSD'
    long_cond = m & df.get('corr_audusd_24h', pd.Series(np.nan, index=df.index)).lt(0.22)
    dirs = dirs.where(~long_cond, 1.0).where(~(m & ~long_cond), np.nan)

    # NZDUSD -> LONG if dist_5d_high > 0.35
    m = pair == 'NZDUSD'
    long_cond = m & df.get('dist_5d_high', pd.Series(np.nan, index=df.index)).gt(0.35)
    dirs = dirs.where(~long_cond, 1.0).where(~(m & ~long_cond), np.nan)

    # USDCHF -> LONG if corr_eurusd_1w > -0.60
    m = pair == 'USDCHF'
    long_cond = m & df.get('corr_eurusd_1w', pd.Series(np.nan, index=df.index)).gt(-0.60)
    dirs = dirs.where(~long_cond, 1.0).where(~(m & ~long_cond), np.nan)

    # CHFJPY -> LONG if corr_usdjpy_1w > 0.40, SHORT if < 0.26, else NaN
    m = pair == 'CHFJPY'
    c_val = df.get('corr_usdjpy_1w', pd.Series(np.nan, index=df.index))
    long_cond  = m & c_val.gt(0.40)
    short_cond = m & c_val.lt(0.26)
    dirs = dirs.where(~long_cond, 1.0).where(~short_cond, -1.0).where(~(m & ~long_cond & ~short_cond), np.nan)

    # CADJPY -> LONG if vol_trend < 1.15, SHORT if >= 1.15
    m = pair == 'CADJPY'
    vt = df.get('vol_trend', pd.Series(np.nan, index=df.index))
    long_cond  = m & vt.lt(1.15)
    short_cond = m & vt.ge(1.15)
    dirs = dirs.where(~long_cond, 1.0).where(~short_cond, -1.0)

    # AUDJPY -> LONG if beta_usdjpy_1w > 0.74
    m = pair == 'AUDJPY'
    long_cond = m & df.get('beta_usdjpy_1w', pd.Series(np.nan, index=df.index)).gt(0.74)
    dirs = dirs.where(~long_cond, 1.0).where(~(m & ~long_cond), np.nan)

    # EURJPY -> LONG if beta_eurusd_1w > 0.38
    m = pair == 'EURJPY'
    long_cond = m & df.get('beta_eurusd_1w', pd.Series(np.nan, index=df.index)).gt(0.38)
    dirs = dirs.where(~long_cond, 1.0).where(~(m & ~long_cond), np.nan)

    # GBPJPY -> LONG if beta_eurusd_1w > 0.50
    m = pair == 'GBPJPY'
    long_cond = m & df.get('beta_eurusd_1w', pd.Series(np.nan, index=df.index)).gt(0.50)
    dirs = dirs.where(~long_cond, 1.0).where(~(m & ~long_cond), np.nan)

    # EURAUD -> LONG if corr_audusd_24h < 0.22
    m = pair == 'EURAUD'
    long_cond = m & df.get('corr_audusd_24h', pd.Series(np.nan, index=df.index)).lt(0.22)
    dirs = dirs.where(~long_cond, 1.0).where(~(m & ~long_cond), np.nan)

    # AUDNZD -> LONG if corr_regime_audusd > 0.0
    m = pair == 'AUDNZD'
    long_cond = m & df.get('corr_regime_audusd', pd.Series(np.nan, index=df.index)).gt(0.0)
    dirs = dirs.where(~long_cond, 1.0).where(~(m & ~long_cond), np.nan)

    # EURGBP -> SHORT if csi_usd_24h > 0.004
    m = pair == 'EURGBP'
    short_cond = m & df.get('csi_usd_24h', pd.Series(np.nan, index=df.index)).gt(0.004)
    dirs = dirs.where(~short_cond, -1.0).where(~(m & ~short_cond), np.nan)

    # USDCAD -> no rule yet, skip
    dirs = dirs.where(pair != 'USDCAD', np.nan)

    return dirs


print('\nApplying direction rules...')
df['direction'] = apply_direction_rules(df)
n_with_dir = df['direction'].notna().sum()
print(f'  Bars with a direction call: {n_with_dir:,} out of {len(df):,}')

# ── Apply 72h cooldown per pair ───────────────────────────────────────────────
print('\nApplying 72h cooldown per pair...')
candidates = df[df['direction'].notna() & df['fwd_72h'].notna()].sort_index().copy()

cooldown_until = {}
kept = []
for ts, row in candidates.iterrows():
    pair = row['pair']
    if pair in cooldown_until and ts < cooldown_until[pair]:
        continue
    cooldown_until[pair] = ts + pd.Timedelta(hours=COOLDOWN_H)
    kept.append(ts)

df_sig = candidates.loc[kept].copy()
total  = len(df_sig)
days   = (df_sig.index.max() - df_sig.index.min()).days
months = max(days / 30, 0.1)

print(f'  Signals after cooldown: {total:,}  ({total/months:.0f}/mo)')

# ── Accuracy metrics ──────────────────────────────────────────────────────────
df_sig['fwd_dir'] = np.sign(df_sig['fwd_72h'])
df_sig['correct'] = (df_sig['direction'] == df_sig['fwd_dir']).astype(float)
df_sig['result_pips'] = df_sig['direction'] * df_sig['fwd_72h']  # pips gained in predicted direction

print(f'\n{"="*70}')
print(f'  OVERALL RESULTS  (MFE>=50, hours 7-20 UTC, 72h cooldown, last 18 months)')
print(f'{"="*70}')
print(f'  Total signals : {total:,}  ({total/months:.0f}/mo)')
print(f'  Directional accuracy: {df_sig["correct"].mean():.1%}')
print(f'  Avg 72h move (in direction): {df_sig["result_pips"].mean():+.1f} pips')
print(f'  Med 72h move (in direction): {df_sig["result_pips"].median():+.1f} pips')

# Baseline: always take the biased direction per pair
n_long  = (df_sig['direction'] ==  1).sum()
n_short = (df_sig['direction'] == -1).sum()
print(f'\n  LONG calls : {n_long:,}  ({n_long/total:.1%})')
print(f'  SHORT calls: {n_short:,}  ({n_short/total:.1%})')

# ── Per-pair breakdown ────────────────────────────────────────────────────────
print(f'\n{"="*70}')
print(f'  PER-PAIR BREAKDOWN')
print(f'{"="*70}')
print(f'\n  {"Pair":<10}  {"N":>5}  {"N/mo":>6}  {"Dir":>6}  {"Acc%":>7}  {"AvgPip":>8}  {"MedPip":>8}  {"LONG%":>7}  Rule')
print(f'  {"-"*88}')

RULES = {
    'USDJPY': 'always SHORT',
    'AUDUSD': 'beta_gbpusd_1w>0.775 OR atr_24<40.8',
    'GBPUSD': 'csi_usd_24h<0.004',
    'EURUSD': 'corr_audusd_24h<0.22',
    'NZDUSD': 'dist_5d_high>0.35',
    'USDCHF': 'corr_eurusd_1w>-0.60',
    'CHFJPY': 'corr_usdjpy_1w: >0.40=L, <0.26=S',
    'CADJPY': 'vol_trend: <1.15=L, >=1.15=S',
    'AUDJPY': 'beta_usdjpy_1w>0.74',
    'EURJPY': 'beta_eurusd_1w>0.38',
    'GBPJPY': 'beta_eurusd_1w>0.50',
    'EURAUD': 'corr_audusd_24h<0.22',
    'AUDNZD': 'corr_regime_audusd>0.0',
    'EURGBP': 'csi_usd_24h>0.004 -> SHORT',
    'USDCAD': '(no rule)',
}

pair_results = []
for pair in sorted(df_sig['pair'].unique()):
    sub = df_sig[df_sig['pair'] == pair]
    if len(sub) < 5:
        continue
    n       = len(sub)
    acc     = sub['correct'].mean()
    avg_pip = sub['result_pips'].mean()
    med_pip = sub['result_pips'].median()
    pct_l   = (sub['direction'] == 1).mean()
    dir_str = 'LONG' if pct_l > 0.8 else ('SHORT' if pct_l < 0.2 else f'MIX')
    rule    = RULES.get(pair, '?')
    print(f'  {pair:<10}  {n:>5}  {n/months:>6.1f}  {dir_str:>6}  {acc:>7.1%}  {avg_pip:>+8.1f}  {med_pip:>+8.1f}  {pct_l:>7.1%}  {rule}')
    pair_results.append({'pair': pair, 'n': n, 'acc': acc, 'avg_pip': avg_pip})

# ── Monthly accuracy breakdown ────────────────────────────────────────────────
print(f'\n{"="*70}')
print(f'  MONTHLY ACCURACY')
print(f'{"="*70}')
df_sig['month'] = pd.to_datetime(df_sig.index).to_period('M')
monthly = df_sig.groupby('month').agg(
    N        = ('correct', 'count'),
    accuracy = ('correct', 'mean'),
    avg_pip  = ('result_pips', 'mean'),
).reset_index()

print(f'\n  {"Month":<10}  {"N":>5}  {"Acc%":>7}  {"AvgPip":>8}')
print(f'  {"-"*36}')
for _, row in monthly.iterrows():
    acc_flag = ' <<' if row['accuracy'] < 0.45 else (' >>' if row['accuracy'] > 0.65 else '')
    print(f'  {str(row["month"]):<10}  {int(row["N"]):>5}  {row["accuracy"]:>7.1%}  {row["avg_pip"]:>+8.1f}{acc_flag}')

# ── Accuracy by direction call ────────────────────────────────────────────────
print(f'\n{"="*70}')
print(f'  ACCURACY SPLIT: LONG calls vs SHORT calls')
print(f'{"="*70}')
for d, label in [(1, 'LONG '), (-1, 'SHORT')]:
    sub = df_sig[df_sig['direction'] == d]
    if len(sub) == 0:
        continue
    print(f'\n  {label}  N={len(sub):,}')
    print(f'    Accuracy  : {sub["correct"].mean():.1%}')
    print(f'    Avg pip   : {sub["result_pips"].mean():+.1f}')
    print(f'    Med pip   : {sub["result_pips"].median():+.1f}')

# ── Comparison: direction system vs always-LONG baseline ─────────────────────
print(f'\n{"="*70}')
print(f'  BASELINE COMPARISON')
print(f'{"="*70}')
baseline_long  = (df_sig['fwd_dir'] == 1).mean()
baseline_short = (df_sig['fwd_dir'] == -1).mean()
print(f'  Always LONG accuracy on these bars  : {baseline_long:.1%}')
print(f'  Always SHORT accuracy on these bars : {baseline_short:.1%}')
print(f'  Direction system accuracy           : {df_sig["correct"].mean():.1%}')
print(f'  Lift over best naive baseline       : {df_sig["correct"].mean() - max(baseline_long, baseline_short):+.1%}')

print(f'\n{"="*70}')
print(f'  DONE')
print(f'{"="*70}')
