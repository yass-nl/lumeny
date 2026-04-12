"""
Stress Test — Pair-Specific Direction Rules
============================================
Uses parquet test set (post 2024-06-30) only.
MFE model: mfe_q50 (72h horizon model).
Horizon: 72h fixed forward return.
Cooldown: 72h per pair.

Rules tested:
  1. Always SHORT / Always LONG / Random  (baselines)
  2. Universal USD Strength               — original approach (kept for comparison)
  3. Pair-specific rules                  — derived from per-pair direction analysis:
       USDJPY  → always SHORT (36% LONG bias, structural)
       AUDUSD  → LONG if beta_gbpusd_1w > 0.77 AND atr_24 < median
       GBPUSD  → LONG if csi_usd_24h < p25 (USD weak)
       EURUSD  → LONG if corr_audusd_24h < p25 AND vol_regime_5d < p75
       NZDUSD  → LONG if dist_5d_high > p75 (near 5d high = continuation)
       USDCHF  → LONG if corr_eurusd_1w > -0.40 (less inverse = USD driving)
       CHFJPY  → LONG if corr_usdjpy_1w in [p50, p75] range
       CADJPY  → LONG if vol_trend < p25, SHORT if vol_trend > p75
       Others  → skip (no clean signal found)
"""

import sys
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

SCRIPT_DIR     = Path(__file__).parent
FEATURES_DIR   = SCRIPT_DIR / '../backend/data/features_9'
PROCESSED_DIR  = SCRIPT_DIR / '../backend/data/processed'
MFE_MODEL_PATH = SCRIPT_DIR / '../backend/models_9/mfe_q50/model_1H_Q50.joblib'

TRAIN_END  = '2024-06-30'
MFE_THRESH = 70.0
COOLDOWN_H = 72     # match 72h horizon

SPREAD_PIPS = {
    'EURUSD': 0.6, 'GBPUSD': 0.8, 'USDJPY': 1.0, 'USDCHF': 0.7,
    'AUDUSD': 0.6, 'USDCAD': 1.2, 'NZDUSD': 0.9,
    'EURJPY': 1.4, 'GBPJPY': 2.1, 'EURGBP': 0.7,
    'EURAUD': 2.1, 'AUDJPY': 1.5, 'CADJPY': 1.6, 'CHFJPY': 2.5, 'AUDNZD': 2.0,
}

JPY_PAIRS = {'USDJPY', 'EURJPY', 'GBPJPY', 'AUDJPY', 'CADJPY', 'CHFJPY'}

PAIRS_ALL = [
    'EURUSD','GBPUSD','USDJPY','USDCHF','AUDUSD','USDCAD','NZDUSD',
    'EURJPY','GBPJPY','EURGBP','EURAUD','AUDJPY','CADJPY','CHFJPY','AUDNZD',
]

# USD-denominated pairs and their sign convention (+1 = USD strengthens when pair rises)
# EURUSD rises = USD weakens → sign = -1
# USDJPY rises = USD strengthens → sign = +1
USD_PAIRS = {
    'EURUSD': -1, 'GBPUSD': -1, 'AUDUSD': -1, 'NZDUSD': -1, 'USDCAD': +1,
    'USDCHF': +1, 'USDJPY': +1,
}

# ── Load MFE model ────────────────────────────────────────────────────────────
print('Loading MFE model...')
mfe_bundle   = joblib.load(MFE_MODEL_PATH)
mfe_model    = mfe_bundle['model']
feature_cols = mfe_bundle['feature_cols']
print(f'  MFE: {mfe_bundle["n_iters"]} iters, {len(feature_cols)} features')

# ── Load features_9 ───────────────────────────────────────────────────────────
print('\nLoading features_9...')
dfs = [pd.read_parquet(f) for f in sorted(FEATURES_DIR.glob('*_features.parquet'))]
df  = pd.concat(dfs).sort_index()
df_test = df[df.index > TRAIN_END].copy()
print(f'  Test rows: {len(df_test):,}')

# ── Run MFE model ─────────────────────────────────────────────────────────────
print('Running MFE model...')
X = df_test[feature_cols].ffill().fillna(0)
df_test['q50_mfe'] = mfe_model.predict(X)

# ── Load 1H processed data for all pairs ─────────────────────────────────────
print('\nLoading 1H processed data...')
close_all = {}
high_all  = {}
low_all   = {}
for pair in PAIRS_ALL:
    fpath = PROCESSED_DIR / f'{pair}_1H.parquet'
    if not fpath.exists():
        print(f'  Missing: {fpath}')
        continue
    df_1h = pd.read_parquet(fpath).sort_index()
    close_all[pair] = df_1h['close']
    high_all[pair]  = df_1h['high']
    low_all[pair]   = df_1h['low']
    print(f'  {pair}: {len(df_1h):,} bars')

# ── Compute fwd_72h and USD agree — everything else comes from df_test features ─
print('\nComputing indicators...')

# Forward return series per pair
fwd_frames = []
for pair in PAIRS_ALL:
    if pair not in close_all:
        continue
    c   = close_all[pair]
    pip = 0.01 if pair in JPY_PAIRS else 0.0001
    fwd_72h = (c.shift(-72) - c) / pip
    tmp = pd.DataFrame({'pair': pair, 'fwd_72h': fwd_72h}, index=c.index)
    tmp.index.name = 'datetime'
    fwd_frames.append(tmp.reset_index())

df_fwd = pd.concat(fwd_frames, ignore_index=True)

# USD agree (for universal baseline rule)
print('Computing cross-pair indicators...')
usd_rets = {}
for pair, sign in USD_PAIRS.items():
    if pair in close_all:
        c = close_all[pair]
        usd_rets[pair] = sign * (c / c.shift(168) - 1)
usd_agree_df = pd.DataFrame(usd_rets)
usd_agree    = (usd_agree_df > 0).sum(axis=1) / usd_agree_df.notna().sum(axis=1)
usd_agree.name = 'usd_agree'
usd_agree    = usd_agree.reset_index()
usd_agree.columns = ['datetime', 'usd_agree']

# ── Filter to MFE candidates — join fwd_72h and usd_agree ────────────────────
df_cands = df_test[df_test['q50_mfe'] >= MFE_THRESH].copy()

df_cands = (
    df_cands.reset_index()
    .rename(columns={df_cands.reset_index().columns[0]: 'datetime'})
    .merge(df_fwd,    on=['datetime', 'pair'], how='left')
    .merge(usd_agree, on='datetime',           how='left')
    .set_index('datetime')
)
df_cands.index.name = None
print(f'\n  Candidate bars after join: {len(df_cands):,}')

# Verify required feature columns are present (they come from df_test/features_9)
required = ['beta_gbpusd_1w', 'atr_24', 'csi_usd_24h', 'corr_audusd_24h',
            'vol_regime_5d', 'dist_5d_high', 'corr_eurusd_1w',
            'corr_usdjpy_1w', 'vol_trend']
missing = [c for c in required if c not in df_cands.columns]
if missing:
    print(f'  WARNING: missing feature cols: {missing}')
else:
    print(f'  All required feature columns present.')

# ── Compute direction signals ─────────────────────────────────────────────────

# ── Baseline: Universal USD Strength (same as before, for comparison)
df_cands['usd_strength_dir'] = np.where(
    df_cands['usd_agree'].isna(), np.nan,
    np.where(df_cands['usd_agree'] > 0.5, -1.0, 1.0)
)

# ── Pair-specific rules — TIGHT version (original) ───────────────────────────
pair_col = df_cands['pair']

def build_dirs_tight(df_cands, pair_col):
    dirs = pd.Series(np.nan, index=df_cands.index)
    dirs = dirs.where(pair_col != 'USDJPY', -1.0)
    audusd_mask = pair_col == 'AUDUSD'
    audusd_long = audusd_mask & df_cands['beta_gbpusd_1w'].gt(0.775) & df_cands['atr_24'].lt(40.8)
    dirs = dirs.where(~audusd_long, 1.0).where(~(audusd_mask & ~audusd_long), np.nan)
    gbpusd_mask = pair_col == 'GBPUSD'
    gbpusd_long = gbpusd_mask & df_cands['csi_usd_24h'].lt(-0.007)
    dirs = dirs.where(~gbpusd_long, 1.0).where(~(gbpusd_mask & ~gbpusd_long), np.nan)
    eurusd_mask = pair_col == 'EURUSD'
    eurusd_long = eurusd_mask & df_cands['corr_audusd_24h'].lt(-0.10) & df_cands['vol_regime_5d'].lt(1.65)
    dirs = dirs.where(~eurusd_long, 1.0).where(~(eurusd_mask & ~eurusd_long), np.nan)
    nzdusd_mask = pair_col == 'NZDUSD'
    nzdusd_long = nzdusd_mask & df_cands['dist_5d_high'].gt(0.55)
    dirs = dirs.where(~nzdusd_long, 1.0).where(~(nzdusd_mask & ~nzdusd_long), np.nan)
    usdchf_mask = pair_col == 'USDCHF'
    usdchf_long = usdchf_mask & df_cands['corr_eurusd_1w'].gt(-0.40)
    dirs = dirs.where(~usdchf_long, 1.0).where(~(usdchf_mask & ~usdchf_long), np.nan)
    chfjpy_mask  = pair_col == 'CHFJPY'
    chfjpy_long  = chfjpy_mask & df_cands['corr_usdjpy_1w'].between(0.52, 0.68)
    chfjpy_short = chfjpy_mask & df_cands['corr_usdjpy_1w'].lt(0.26)
    dirs = dirs.where(~chfjpy_long, 1.0).where(~chfjpy_short, -1.0).where(~(chfjpy_mask & ~chfjpy_long & ~chfjpy_short), np.nan)
    cadjpy_mask  = pair_col == 'CADJPY'
    cadjpy_long  = cadjpy_mask & df_cands['vol_trend'].lt(0.96)
    cadjpy_short = cadjpy_mask & df_cands['vol_trend'].gt(1.27)
    dirs = dirs.where(~cadjpy_long, 1.0).where(~cadjpy_short, -1.0).where(~(cadjpy_mask & ~cadjpy_long & ~cadjpy_short), np.nan)
    other_pairs = ~pair_col.isin(['USDJPY','AUDUSD','GBPUSD','EURUSD','NZDUSD','USDCHF','CHFJPY','CADJPY'])
    dirs = dirs.where(~other_pairs, np.nan)
    return dirs

# ── Pair-specific rules — RELAXED version (v1) ───────────────────────────────
def build_dirs_relaxed(df_cands, pair_col):
    dirs = pd.Series(np.nan, index=df_cands.index)
    dirs = dirs.where(pair_col != 'USDJPY', -1.0)
    audusd_mask = pair_col == 'AUDUSD'
    audusd_long = audusd_mask & (df_cands['beta_gbpusd_1w'].gt(0.775) | df_cands['atr_24'].lt(40.8))
    dirs = dirs.where(~audusd_long, 1.0).where(~(audusd_mask & ~audusd_long), np.nan)
    gbpusd_mask = pair_col == 'GBPUSD'
    gbpusd_long = gbpusd_mask & df_cands['csi_usd_24h'].lt(0.004)
    dirs = dirs.where(~gbpusd_long, 1.0).where(~(gbpusd_mask & ~gbpusd_long), np.nan)
    eurusd_mask = pair_col == 'EURUSD'
    eurusd_long = eurusd_mask & df_cands['corr_audusd_24h'].lt(0.22)
    dirs = dirs.where(~eurusd_long, 1.0).where(~(eurusd_mask & ~eurusd_long), np.nan)
    nzdusd_mask = pair_col == 'NZDUSD'
    nzdusd_long = nzdusd_mask & df_cands['dist_5d_high'].gt(0.35)
    dirs = dirs.where(~nzdusd_long, 1.0).where(~(nzdusd_mask & ~nzdusd_long), np.nan)
    usdchf_mask = pair_col == 'USDCHF'
    usdchf_long = usdchf_mask & df_cands['corr_eurusd_1w'].gt(-0.60)
    dirs = dirs.where(~usdchf_long, 1.0).where(~(usdchf_mask & ~usdchf_long), np.nan)
    chfjpy_mask  = pair_col == 'CHFJPY'
    chfjpy_long  = chfjpy_mask & df_cands['corr_usdjpy_1w'].gt(0.40)
    chfjpy_short = chfjpy_mask & df_cands['corr_usdjpy_1w'].lt(0.26)
    dirs = dirs.where(~chfjpy_long, 1.0).where(~chfjpy_short, -1.0).where(~(chfjpy_mask & ~chfjpy_long & ~chfjpy_short), np.nan)
    cadjpy_mask  = pair_col == 'CADJPY'
    cadjpy_long  = cadjpy_mask & df_cands['vol_trend'].lt(1.15)
    cadjpy_short = cadjpy_mask & df_cands['vol_trend'].ge(1.15)
    dirs = dirs.where(~cadjpy_long, 1.0).where(~cadjpy_short, -1.0)
    audjpy_mask = pair_col == 'AUDJPY'
    audjpy_long = audjpy_mask & df_cands['beta_usdjpy_1w'].gt(1.14)
    dirs = dirs.where(~audjpy_long, 1.0).where(~(audjpy_mask & ~audjpy_long), np.nan)
    eurjpy_mask = pair_col == 'EURJPY'
    eurjpy_long = eurjpy_mask & df_cands['beta_eurusd_1w'].gt(0.63)
    dirs = dirs.where(~eurjpy_long, 1.0).where(~(eurjpy_mask & ~eurjpy_long), np.nan)
    other_pairs = ~pair_col.isin(['USDJPY','AUDUSD','GBPUSD','EURUSD','NZDUSD','USDCHF','CHFJPY','CADJPY','AUDJPY','EURJPY'])
    dirs = dirs.where(~other_pairs, np.nan)
    return dirs

# ── Pair-specific rules — VERY RELAXED version (v2) ──────────────────────────
# Push every threshold to p50 or remove entirely; open up remaining pairs
def build_dirs_very_relaxed(df_cands, pair_col):
    dirs = pd.Series(np.nan, index=df_cands.index)

    # USDJPY → always SHORT (structural, keep)
    dirs = dirs.where(pair_col != 'USDJPY', -1.0)

    # AUDUSD → always LONG (78% bias, just take all signals)
    dirs = dirs.where(pair_col != 'AUDUSD', 1.0)

    # GBPUSD → always LONG (74% bias)
    dirs = dirs.where(pair_col != 'GBPUSD', 1.0)

    # EURUSD → LONG if corr_audusd_24h < p75 (0.64, very wide)
    eurusd_mask = pair_col == 'EURUSD'
    eurusd_long = eurusd_mask & df_cands['corr_audusd_24h'].lt(0.64)
    dirs = dirs.where(~eurusd_long, 1.0).where(~(eurusd_mask & ~eurusd_long), np.nan)

    # NZDUSD → LONG if dist_5d_high > 0.20 (very loose, just not at 5d lows)
    nzdusd_mask = pair_col == 'NZDUSD'
    nzdusd_long = nzdusd_mask & df_cands['dist_5d_high'].gt(0.20)
    dirs = dirs.where(~nzdusd_long, 1.0).where(~(nzdusd_mask & ~nzdusd_long), np.nan)

    # USDCHF → always LONG (55% bias + atr_24 > p75 = 100% LONG, but let all fire)
    dirs = dirs.where(pair_col != 'USDCHF', 1.0)

    # CHFJPY → LONG if corr_usdjpy_1w > 0.26 (just exclude the confirmed SHORT zone)
    #        → SHORT if corr_usdjpy_1w < 0.26
    chfjpy_mask  = pair_col == 'CHFJPY'
    chfjpy_long  = chfjpy_mask & df_cands['corr_usdjpy_1w'].ge(0.26)
    chfjpy_short = chfjpy_mask & df_cands['corr_usdjpy_1w'].lt(0.26)
    dirs = dirs.where(~chfjpy_long, 1.0).where(~chfjpy_short, -1.0)

    # CADJPY → LONG if vol_trend < 1.27 (p75), SHORT if > 1.27
    cadjpy_mask  = pair_col == 'CADJPY'
    cadjpy_long  = cadjpy_mask & df_cands['vol_trend'].lt(1.27)
    cadjpy_short = cadjpy_mask & df_cands['vol_trend'].ge(1.27)
    dirs = dirs.where(~cadjpy_long, 1.0).where(~cadjpy_short, -1.0)

    # AUDJPY → LONG if beta_usdjpy_1w > 0.74 (p25, was p75=1.14)
    audjpy_mask = pair_col == 'AUDJPY'
    audjpy_long = audjpy_mask & df_cands['beta_usdjpy_1w'].gt(0.74)
    dirs = dirs.where(~audjpy_long, 1.0).where(~(audjpy_mask & ~audjpy_long), np.nan)

    # EURJPY → LONG if beta_eurusd_1w > 0.38 (p50, was 0.63)
    eurjpy_mask = pair_col == 'EURJPY'
    eurjpy_long = eurjpy_mask & df_cands['beta_eurusd_1w'].gt(0.38)
    dirs = dirs.where(~eurjpy_long, 1.0).where(~(eurjpy_mask & ~eurjpy_long), np.nan)

    # GBPJPY → LONG if corr_usdjpy_3d > 0.68 (p25 from analysis)
    gbpjpy_mask = pair_col == 'GBPJPY'
    gbpjpy_long = gbpjpy_mask & df_cands['corr_usdjpy_3d'].gt(0.68)
    dirs = dirs.where(~gbpjpy_long, 1.0).where(~(gbpjpy_mask & ~gbpjpy_long), np.nan)

    # AUDNZD → LONG if corr_audjpy_1w < 0.23 (low corr = AUD independent = bullish NZD divergence)
    audnzd_mask = pair_col == 'AUDNZD'
    audnzd_long = audnzd_mask & df_cands['corr_audjpy_1w'].lt(0.23)
    dirs = dirs.where(~audnzd_long, 1.0).where(~(audnzd_mask & ~audnzd_long), np.nan)

    # EURAUD → LONG if csi_eur_72h < p25 (EUR weak = EURAUD bullish — wait, EUR weak = EURAUD falls)
    # From analysis: csi_eur_72h < p25 → 61.5% LONG, +38p. Keep.
    euraud_mask = pair_col == 'EURAUD'
    euraud_long = euraud_mask & df_cands['csi_eur_72h'].lt(-0.002)
    dirs = dirs.where(~euraud_long, 1.0).where(~(euraud_mask & ~euraud_long), np.nan)

    return dirs

df_cands['pair_specific_dir']              = build_dirs_tight(df_cands, pair_col)
df_cands['pair_specific_relaxed_dir']      = build_dirs_relaxed(df_cands, pair_col)
df_cands['pair_specific_very_relaxed_dir'] = build_dirs_very_relaxed(df_cands, pair_col)


# ── Simulation ────────────────────────────────────────────────────────────────
def simulate(df_cands, direction_col, label, seed=42):
    """
    Walk candidates chronologically per pair with cooldown.
    Result = direction * fwd_72h / pip - spread
    """
    rng = np.random.default_rng(seed)
    cooldown_until = {}
    trades = []

    for ts, row in df_cands.sort_index().iterrows():
        pair = row['pair']

        if pair in cooldown_until and ts < cooldown_until[pair]:
            continue

        if direction_col == 'long':
            direction = 1
        elif direction_col == 'short':
            direction = -1
        elif direction_col == 'random':
            direction = 1 if rng.random() >= 0.5 else -1
        else:
            val = row.get(direction_col, np.nan)
            if pd.isna(val):
                continue
            direction = int(val)

        fwd = row.get('fwd_72h', np.nan)
        if pd.isna(fwd):
            continue

        sp     = SPREAD_PIPS.get(pair, 1.5)
        result = direction * fwd - sp   # fwd_72h already in pips

        cooldown_until[pair] = ts + pd.Timedelta(hours=COOLDOWN_H)

        trades.append({
            'ts':      ts,
            'pair':    pair,
            'dir':     direction,
            'q50_mfe': row['q50_mfe'],
            'result':  result,
            'win':     result > 0,
            'year':    ts.year,
            'month':   ts.to_period('M'),
        })

    return pd.DataFrame(trades)


def stats(tr, label=''):
    if len(tr) == 0:
        print(f'  {label:<52} N=0 (no trades)')
        return
    r = tr['result'].dropna()
    months = max((tr['ts'].max() - tr['ts'].min()).days / 30, 0.1)
    wr  = r.gt(0).mean()
    avg = r.mean()
    wins = r[r>0]; losses = r[r<=0]
    pf  = wins.sum() / abs(losses.sum()) if len(losses) > 0 and losses.sum() != 0 else np.nan
    n_long  = (tr['dir'] ==  1).sum()
    n_short = (tr['dir'] == -1).sum()
    print(f'  {label:<52} N={len(tr):>5,} ({len(tr)/months:>4.0f}/mo)  '
          f'L={n_long} S={n_short}  WR={wr:.1%}  avg={avg:>+7.1f}p  PF={pf:.3f}')


# ── Run all simulations ───────────────────────────────────────────────────────
print('\nSimulating...')
tr_long          = simulate(df_cands, 'long',                         'Always LONG')
tr_short         = simulate(df_cands, 'short',                        'Always SHORT')
tr_random        = simulate(df_cands, 'random',                       'Random direction')
tr_usd           = simulate(df_cands, 'usd_strength_dir',             'USD Strength (universal)')
tr_tight         = simulate(df_cands, 'pair_specific_dir',            'Pair-specific TIGHT')
tr_relaxed       = simulate(df_cands, 'pair_specific_relaxed_dir',    'Pair-specific RELAXED')
tr_very_relaxed  = simulate(df_cands, 'pair_specific_very_relaxed_dir','Pair-specific VERY RELAXED')


# ── Results ───────────────────────────────────────────────────────────────────
print(f'\n{"="*80}')
print(f'  RESULTS — 72h horizon | MFE >= {MFE_THRESH} | cooldown={COOLDOWN_H}h')
print(f'{"="*80}')
print(f'  {"Rule":<52} {"N":>5}  {"L/S"}  {"WR":>6}  {"Avg":>8}  {"PF":>7}')
print(f'  {"-"*80}')
stats(tr_long,         'Always LONG')
stats(tr_short,        'Always SHORT')
stats(tr_random,       'Random direction')
print(f'  {"-"*80}')
stats(tr_usd,          'USD Strength (universal)')
stats(tr_tight,        'Pair-specific TIGHT')
stats(tr_relaxed,      'Pair-specific RELAXED')
stats(tr_very_relaxed, 'Pair-specific VERY RELAXED')


# ── Per-pair breakdown ────────────────────────────────────────────────────────
def per_pair(tr, label):
    print(f'\n  {label}')
    print(f'  {"Pair":<10} {"N":>5}  {"L/S":>7}  {"WR":>6}  {"Avg":>8}')
    print(f'  {"-"*44}')
    for pair in sorted(tr['pair'].unique()):
        sub = tr[tr['pair'] == pair]['result']
        if len(sub) < 3: continue
        nl = (tr.loc[tr['pair']==pair,'dir']==1).sum()
        ns = (tr.loc[tr['pair']==pair,'dir']==-1).sum()
        print(f'  {pair:<10} {len(sub):>5}  {nl:>3}/{ns:<3}  {sub.gt(0).mean():>6.1%}  {sub.mean():>+8.2f}p')

print(f'\n{"="*80}')
print(f'  PER-PAIR BREAKDOWN')
print(f'{"="*80}')
per_pair(tr_short,        'Baseline: Always SHORT')
per_pair(tr_tight,        'Pair-specific TIGHT')
per_pair(tr_relaxed,      'Pair-specific RELAXED')
per_pair(tr_very_relaxed, 'Pair-specific VERY RELAXED')


# ── Year-by-year ──────────────────────────────────────────────────────────────
def year_by_year(tr, label):
    print(f'\n  {label}')
    print(f'  {"Year":<6} {"N":>5}  {"WR":>6}  {"Avg":>8}  {"Total":>10}')
    for year in sorted(tr['year'].unique()):
        s = tr[tr['year']==year]['result']
        print(f'  {year:<6} {len(s):>5}  {s.gt(0).mean():>6.1%}  {s.mean():>+8.2f}p  {s.sum():>10.0f}p')

print(f'\n{"="*80}')
print(f'  YEAR-BY-YEAR')
print(f'{"="*80}')
year_by_year(tr_short,        'Always SHORT (baseline)')
year_by_year(tr_usd,          'USD Strength (universal)')
year_by_year(tr_tight,        'Pair-specific TIGHT')
year_by_year(tr_relaxed,      'Pair-specific RELAXED')
year_by_year(tr_very_relaxed, 'Pair-specific VERY RELAXED')
