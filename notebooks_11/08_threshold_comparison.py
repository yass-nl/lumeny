"""
MFE Threshold Comparison: Q50 >= 50 vs Q50 >= 70
==================================================
Last 18 months, hours 7-20 UTC, 72h cooldown per pair.

For each threshold:
  - Signal volume
  - MFE model accuracy: actual MFE within 72h >= threshold  -> accurate
  - Direction system accuracy: price at h+72 in predicted direction
  - Overlap: % of signals shared between both thresholds

Direction accuracy definition:
  - LONG  correct if close[t+72] > close[t]
  - SHORT correct if close[t+72] < close[t]

MFE accuracy definition:
  - Signal at threshold T is accurate if max(high[t:t+72]) - close[t] >= T * pip  (LONG)
    or close[t] - min(low[t:t+72]) >= T * pip  (SHORT, using signed direction)
  - Without direction: actual MFE (unsigned best excursion in either direction) >= T * pip
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path

SCRIPT_DIR    = Path(__file__).parent
FEATURES_DIR  = SCRIPT_DIR / '../backend/data/features_9'
PROCESSED_DIR = SCRIPT_DIR / '../backend/data/processed'
MFE_MODEL     = SCRIPT_DIR / '../backend/models_9/mfe_q50/model_1H_Q50.joblib'

START_DATE    = '2024-10-11'
COOLDOWN_H    = 72
FWD_H         = 72
HOURS_ALLOWED = set(range(7, 21))

THRESH_A      = 50.0   # pips
THRESH_B      = 70.0   # pips

JPY_PAIRS = {'USDJPY', 'EURJPY', 'GBPJPY', 'AUDJPY', 'CADJPY', 'CHFJPY'}
PAIRS_ALL = [
    'EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD', 'USDCAD', 'NZDUSD',
    'EURJPY', 'GBPJPY', 'EURGBP', 'EURAUD', 'AUDJPY', 'CADJPY', 'CHFJPY', 'AUDNZD',
]


# ── Load MFE model ─────────────────────────────────────────────────────────────
print('Loading MFE model...')
bundle       = joblib.load(MFE_MODEL)
mfe_model    = bundle['model']
feature_cols = bundle['feature_cols']
print(f'  {len(feature_cols)} features, {bundle["n_iters"]} iters')

# ── Load features_9 ────────────────────────────────────────────────────────────
print('Loading features_9...')
dfs = [pd.read_parquet(f) for f in sorted(FEATURES_DIR.glob('*_features.parquet'))]
df  = pd.concat(dfs).sort_index()
df  = df[df.index >= START_DATE].copy()
print(f'  {len(df):,} rows  |  {df.index.min().date()} to {df.index.max().date()}')

# ── Run MFE model ──────────────────────────────────────────────────────────────
print('Scoring MFE model...')
X           = df[feature_cols].ffill().fillna(0)
df['q50']   = mfe_model.predict(X)
df['hour']  = pd.to_datetime(df.index).hour
df_filtered = df[df['hour'].isin(HOURS_ALLOWED)].copy()
print(f'  Hours 7-20 filtered: {len(df_filtered):,} bars')

# ── Direction rules (RELAXED) ──────────────────────────────────────────────────
def apply_direction_rules(df):
    dirs = pd.Series(np.nan, index=df.index)
    pair = df['pair']
    def col(name):
        return df.get(name, pd.Series(np.nan, index=df.index))

    dirs = dirs.where(pair != 'USDJPY', -1.0)

    m = pair == 'AUDUSD'
    lc = m & (col('beta_gbpusd_1w').gt(0.775) | col('atr_24').lt(40.8))
    dirs = dirs.where(~lc, 1.0).where(~(m & ~lc), np.nan)

    m = pair == 'GBPUSD'
    lc = m & col('csi_usd_24h').lt(0.004)
    dirs = dirs.where(~lc, 1.0).where(~(m & ~lc), np.nan)

    m = pair == 'EURUSD'
    lc = m & col('corr_audusd_24h').lt(0.22)
    dirs = dirs.where(~lc, 1.0).where(~(m & ~lc), np.nan)

    m = pair == 'NZDUSD'
    lc = m & col('dist_5d_high').gt(0.35)
    dirs = dirs.where(~lc, 1.0).where(~(m & ~lc), np.nan)

    m = pair == 'USDCHF'
    lc = m & col('corr_eurusd_1w').gt(-0.60)
    dirs = dirs.where(~lc, 1.0).where(~(m & ~lc), np.nan)

    m  = pair == 'CHFJPY'
    cv = col('corr_usdjpy_1w')
    lc = m & cv.gt(0.40)
    sc = m & cv.lt(0.26)
    dirs = dirs.where(~lc, 1.0).where(~sc, -1.0).where(~(m & ~lc & ~sc), np.nan)

    m  = pair == 'CADJPY'
    vt = col('vol_trend')
    lc = m & vt.lt(1.15)
    sc = m & vt.ge(1.15)
    dirs = dirs.where(~lc, 1.0).where(~sc, -1.0)

    m  = pair == 'AUDJPY'
    lc = m & col('beta_usdjpy_1w').gt(0.74)
    dirs = dirs.where(~lc, 1.0).where(~(m & ~lc), np.nan)

    m  = pair == 'EURJPY'
    lc = m & col('beta_eurusd_1w').gt(0.38)
    dirs = dirs.where(~lc, 1.0).where(~(m & ~lc), np.nan)

    m  = pair == 'GBPJPY'
    lc = m & col('beta_eurusd_1w').gt(0.50)
    dirs = dirs.where(~lc, 1.0).where(~(m & ~lc), np.nan)

    m  = pair == 'EURAUD'
    lc = m & col('corr_audusd_24h').lt(0.22)
    dirs = dirs.where(~lc, 1.0).where(~(m & ~lc), np.nan)

    m  = pair == 'AUDNZD'
    lc = m & col('corr_regime_audusd').gt(0.0)
    dirs = dirs.where(~lc, 1.0).where(~(m & ~lc), np.nan)

    m  = pair == 'EURGBP'
    sc = m & col('csi_usd_24h').gt(0.004)
    dirs = dirs.where(~sc, -1.0).where(~(m & ~sc), np.nan)

    dirs = dirs.where(pair != 'USDCAD', np.nan)
    return dirs


df_filtered['direction'] = apply_direction_rules(df_filtered)

# ── Load 1H OHLCV for realized outcomes ───────────────────────────────────────
print('Loading 1H price data...')
ohlcv = {}
for pair in PAIRS_ALL:
    fpath = PROCESSED_DIR / f'{pair}_1H.parquet'
    if fpath.exists():
        ohlcv[pair] = pd.read_parquet(fpath).sort_index()

# ── Cooldown + signal extraction per threshold ────────────────────────────────
def extract_signals(df_in, thresh):
    """
    Filter df to q50 >= thresh, apply 72h cooldown, compute realized outcomes.
    Returns DataFrame with one row per signal.
    """
    pool = df_in[df_in['q50'] >= thresh].copy()

    # Cooldown (integer-position safe)
    pool_r = pool.reset_index()
    ts_col = pool_r.columns[0]
    cooldown_until = {}
    kept = []
    for i, row in pool_r.iterrows():
        ts   = row[ts_col]
        pair = row['pair']
        if pair in cooldown_until and ts < cooldown_until[pair]:
            continue
        cooldown_until[pair] = ts + pd.Timedelta(hours=COOLDOWN_H)
        kept.append(i)
    sigs = pool_r.iloc[kept].set_index(ts_col).copy()
    sigs.index.name = None

    # Realized outcomes
    fwd_72h     = []
    actual_mfe  = []   # unsigned best excursion in direction (or max of both if no dir)
    mfe_acc     = []   # did actual MFE >= thresh pips?
    dir_acc     = []   # did price move in predicted direction at h+72?
    has_dir     = []

    for ts, row in sigs.iterrows():
        pair    = row['pair']
        pip     = 0.01 if pair in JPY_PAIRS else 0.0001
        dir_val = row.get('direction', np.nan)

        if pair not in ohlcv:
            fwd_72h.append(np.nan); actual_mfe.append(np.nan)
            mfe_acc.append(np.nan); dir_acc.append(np.nan); has_dir.append(False)
            continue

        close_s = ohlcv[pair]['close']
        high_s  = ohlcv[pair]['high']
        low_s   = ohlcv[pair]['low']
        pos     = close_s.index.searchsorted(ts)

        if pos >= len(close_s) or pos + FWD_H >= len(close_s):
            fwd_72h.append(np.nan); actual_mfe.append(np.nan)
            mfe_acc.append(np.nan); dir_acc.append(np.nan); has_dir.append(False)
            continue

        entry     = close_s.iloc[pos]
        exit_72h  = close_s.iloc[pos + FWD_H]
        h_window  = high_s.iloc[pos:pos + FWD_H + 1]
        l_window  = low_s.iloc[pos:pos + FWD_H + 1]

        fwd_move  = (exit_72h - entry) / pip
        fwd_72h.append(round(fwd_move, 1))

        # Actual MFE: unsigned max excursion in EITHER direction
        up_excursion   = (h_window.max() - entry) / pip
        down_excursion = (entry - l_window.min()) / pip
        best_excursion = max(up_excursion, down_excursion)
        actual_mfe.append(round(best_excursion, 1))

        # MFE accuracy: did best excursion reach the threshold?
        mfe_acc.append(1 if best_excursion >= thresh else 0)

        # Direction accuracy
        if pd.notna(dir_val):
            correct = (dir_val == 1 and fwd_move > 0) or (dir_val == -1 and fwd_move < 0)
            dir_acc.append(1 if correct else 0)
            has_dir.append(True)
        else:
            dir_acc.append(np.nan)
            has_dir.append(False)

    sigs['fwd_72h']    = fwd_72h
    sigs['actual_mfe'] = actual_mfe
    sigs['mfe_acc']    = mfe_acc
    sigs['dir_acc']    = dir_acc
    sigs['has_dir']    = has_dir
    return sigs


print('\nExtracting signals...')
sigs_50 = extract_signals(df_filtered, THRESH_A)
sigs_70 = extract_signals(df_filtered, THRESH_B)

# Signals with direction
sigs_50_dir = sigs_50[sigs_50['has_dir']].copy()
sigs_70_dir = sigs_70[sigs_70['has_dir']].copy()

# ── Overlap analysis ──────────────────────────────────────────────────────────
# A signal is "shared" if the same (pair, timestamp) appears in both sets
idx_50 = set(zip(sigs_50.index, sigs_50['pair']))
idx_70 = set(zip(sigs_70.index, sigs_70['pair']))
overlap = idx_50 & idx_70

# ── Monthly stability helper ──────────────────────────────────────────────────
def monthly_dir_acc(sigs):
    s = sigs[sigs['dir_acc'].notna()].copy()
    s['month'] = pd.to_datetime(s.index).to_period('M')
    return s.groupby('month').agg(
        N      = ('dir_acc', 'count'),
        acc    = ('dir_acc', 'mean'),
        avg_fwd= ('fwd_72h', lambda x: (sigs.loc[x.index, 'direction'] * x).mean()
                   if False else (sigs.loc[x.index, 'direction'] * sigs.loc[x.index, 'fwd_72h']).mean()),
    )

# Helper for display
def pct(n, d):
    return f'{n/d:.1%}' if d > 0 else 'n/a'

# ── Period info ───────────────────────────────────────────────────────────────
days  = (sigs_50.index.max() - sigs_50.index.min()).days
months = max(days / 30, 0.1)

# ── OUTPUT ────────────────────────────────────────────────────────────────────
W = 72
print(f'\n{"="*W}')
print(f'  MFE THRESHOLD COMPARISON  |  hours 7-20 UTC  |  72h cooldown')
print(f'  Period: {START_DATE} to present  ({days} days, ~{months:.0f} months)')
print(f'{"="*W}')

for label, sigs, sigs_d, thresh in [
    (f'MFE >= {THRESH_A:.0f}p',  sigs_50, sigs_50_dir, THRESH_A),
    (f'MFE >= {THRESH_B:.0f}p',  sigs_70, sigs_70_dir, THRESH_B),
]:
    n_total   = len(sigs)
    n_dir     = len(sigs_d)
    n_mfe_ok  = sigs['mfe_acc'].notna()
    mfe_acc   = sigs.loc[n_mfe_ok, 'mfe_acc'].mean()
    n_mfe_ev  = n_mfe_ok.sum()

    dir_n     = sigs_d['dir_acc'].notna().sum()
    dir_acc   = sigs_d['dir_acc'].mean()
    avg_fwd   = (sigs_d['direction'] * sigs_d['fwd_72h']).mean()
    med_fwd   = (sigs_d['direction'] * sigs_d['fwd_72h']).median()

    long_n    = (sigs_d['direction'] == 1).sum()
    short_n   = (sigs_d['direction'] == -1).sum()

    print(f'\n  -- {label} --------------------------------------------')
    print(f'  Total signals (all pairs):      {n_total:>5,}  ({n_total/months:>5.0f}/month)')
    print(f'  Signals with direction call:    {n_dir:>5,}  ({n_dir/months:>5.0f}/month)  LONG={long_n} SHORT={short_n}')
    print()
    print(f'  MFE MODEL accuracy')
    print(f'    evaluated (window closed):    {n_mfe_ev:>5,}')
    print(f'    actual MFE >= {thresh:.0f}p:          {mfe_acc:.1%}  (price reached {thresh:.0f}p in 72h)')
    print(f'    avg actual MFE (unsigned):    {sigs.loc[n_mfe_ok,"actual_mfe"].mean():.1f}p  med={sigs.loc[n_mfe_ok,"actual_mfe"].median():.1f}p')
    print()
    print(f'  DIRECTION SYSTEM accuracy')
    print(f'    evaluated (window closed):    {dir_n:>5,}')
    print(f'    directional accuracy:         {dir_acc:.1%}  (price at h+72 in predicted direction)')
    print(f'    avg in-direction move:        {avg_fwd:+.1f}p  med={med_fwd:+.1f}p')

print(f'\n{"="*W}')
print(f'  OVERLAP ANALYSIS')
print(f'{"="*W}')
print(f'  Signals at {THRESH_A:.0f}p threshold:   {len(idx_50):,}')
print(f'  Signals at {THRESH_B:.0f}p threshold:   {len(idx_70):,}')
print(f'  Shared (same pair + timestamp):  {len(overlap):,}')
print(f'  {THRESH_B:.0f}p is subset of {THRESH_A:.0f}p:        {pct(len(overlap), len(idx_70))} of {THRESH_B:.0f}p signals also appear at {THRESH_A:.0f}p')
print(f'  {THRESH_A:.0f}p signals also at {THRESH_B:.0f}p:        {pct(len(overlap), len(idx_50))} of {THRESH_A:.0f}p signals are also at {THRESH_B:.0f}p')

# How does direction accuracy differ on the SHARED vs EXCLUSIVE subsets?
shared_mask_50 = sigs_50_dir.apply(lambda r: (r.name, r['pair']) in overlap, axis=1)
shared_mask_70 = sigs_70_dir.apply(lambda r: (r.name, r['pair']) in overlap, axis=1)

excl_50  = sigs_50_dir[~shared_mask_50]   # only at 50, not at 70
shared_d = sigs_50_dir[shared_mask_50]    # at both thresholds

def dir_summary(s, label):
    if len(s) == 0:
        print(f'  {label}: N=0')
        return
    acc = s['dir_acc'].mean()
    avg = (s['direction'] * s['fwd_72h']).mean()
    print(f'  {label}: N={len(s):,}  dir_acc={acc:.1%}  avg_in_dir={avg:+.1f}p')

print()
dir_summary(shared_d, f'Shared signals (at both thresholds)')
dir_summary(excl_50,  f'Exclusive to {THRESH_A:.0f}p (not at {THRESH_B:.0f}p)')

print(f'\n{"="*W}')
print(f'  MONTHLY DIRECTION ACCURACY')
print(f'{"="*W}')

def print_monthly(sigs_d, label):
    s = sigs_d[sigs_d['dir_acc'].notna()].copy()
    s['month'] = pd.to_datetime(s.index).to_period('M')
    grp = s.groupby('month')
    print(f'\n  {label}:')
    print(f'  {"Month":<10} {"N":>5}  {"DirAcc":>7}  {"AvgInDir":>9}')
    print(f'  {"-"*36}')
    for mo, g in grp:
        acc  = g['dir_acc'].mean()
        avg  = (g['direction'] * g['fwd_72h']).mean()
        flag = ' <<' if acc < 0.45 else (' >>' if acc > 0.65 else '')
        print(f'  {str(mo):<10} {len(g):>5}  {acc:>7.1%}  {avg:>+9.1f}p{flag}')

print_monthly(sigs_50_dir, f'MFE >= {THRESH_A:.0f}p + Direction')
print_monthly(sigs_70_dir, f'MFE >= {THRESH_B:.0f}p + Direction')

print(f'\n{"="*W}')
print(f'  PER-PAIR (direction accuracy)')
print(f'{"="*W}')
print(f'  {"Pair":<8}  {"N50":>5}  {"Acc50":>6}  {"AvgDir50":>9}  |  {"N70":>5}  {"Acc70":>6}  {"AvgDir70":>9}')
print(f'  {"-"*60}')
all_pairs = sorted(sigs_50_dir['pair'].unique())
for pair in all_pairs:
    s50 = sigs_50_dir[sigs_50_dir['pair'] == pair]
    s70 = sigs_70_dir[sigs_70_dir['pair'] == pair]
    def fmt(s):
        if len(s) == 0: return f'{"":>5}  {"":>6}  {"":>9}'
        acc = s['dir_acc'].mean()
        avg = (s['direction'] * s['fwd_72h']).mean()
        return f'{len(s):>5}  {acc:>6.1%}  {avg:>+9.1f}p'
    print(f'  {pair:<8}  {fmt(s50)}  |  {fmt(s70)}')

print(f'\n{"="*W}')
print(f'  HOUR DISTRIBUTION  (UTC, hours 7-20)')
print(f'{"="*W}')
print(f'  {"Hour":>5}  {"N50":>5}  {"Acc50":>6}  {"AvgDir50":>9}  |  {"N70":>5}  {"Acc70":>6}  {"AvgDir70":>9}')
print(f'  {"-"*62}')

sigs_50_dir['hour'] = pd.to_datetime(sigs_50_dir.index).hour
sigs_70_dir['hour'] = pd.to_datetime(sigs_70_dir.index).hour

for h in range(7, 21):
    s50h = sigs_50_dir[sigs_50_dir['hour'] == h]
    s70h = sigs_70_dir[sigs_70_dir['hour'] == h]
    def fmt_h(s):
        if len(s) == 0: return f'{"":>5}  {"":>6}  {"":>9}'
        acc = s['dir_acc'].mean()
        avg = (s['direction'] * s['fwd_72h']).mean()
        return f'{len(s):>5}  {acc:>6.1%}  {avg:>+9.1f}p'
    print(f'  {h:>5}h  {fmt_h(s50h)}  |  {fmt_h(s70h)}')

print(f'\n{"="*W}')
print(f'  DONE')
print(f'{"="*W}')
