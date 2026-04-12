"""
Large-move setup scanner.

Goal: find combinations of features at bar T where the reversion move
over the next 2-8 bars is large enough (~0.75+ ATR) to be clearly
tradeable after spread.

Approach:
  1. Start with base combo (slope_close_3h + intrabar_slope + intrabar_momentum)
     as the directional filter
  2. Add magnitude filters on top: extreme residual, high body_ratio,
     slope_alignment, compression, envelope_squeeze
  3. For each sub-filter, report: n_setups, mean_rev, median_rev, % > 0.75 ATR,
     WR, payoff ratio (with SL=0.5 ATR), EV

Magnitude threshold: feature in top/bottom 30% (extreme values)
"""

import sys, os
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd
import numpy as np
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR   = os.path.dirname(SCRIPT_DIR)

PAIRS = [
    'AUDJPY','AUDNZD','AUDUSD','CADJPY','CHFJPY',
    'EURAUD','EURGBP','EURJPY','EURUSD','GBPJPY',
    'GBPUSD','NZDUSD','USDCAD','USDCHF','USDJPY',
]

BASE_FEATS = ['slope_close_3h', 'intrabar_slope', 'intrabar_momentum']

# Magnitude amplifiers — features that predict larger moves when extreme
# Format: (feature, direction) where direction = 'high' means high value -> bigger move
#                                                 'low'  means low value -> bigger move
#                                                 'abs'  means extreme (either sign) -> bigger
AMPLIFIERS = [
    ('residual_6h',         'abs'),   # large residual = far from regression = more to revert
    ('residual_12h',        'abs'),
    ('slope_alignment_3_12','high'),  # short slopes well aligned = momentum mature
    ('body_ratio',          'high'),  # large body = committed bar = sharper reversion
    ('envelope_squeeze_6h', 'abs'),   # channel tilted strongly
    ('compression_6_24',    'low'),   # tight range breaking = more energy
    ('close_position',      'abs'),   # closed near extreme of bar
    ('intrabar_close_pos',  'abs'),   # intrabar also extreme
    ('curvature_6h',        'abs'),   # acceleration in either direction
    ('slope_close_6h',      'abs'),   # short-term trend strength
]

LOAD_FEATS = BASE_FEATS + [f for f, _ in AMPLIFIERS] + ['atr_24']
EXIT_BAR   = 2    # primary exit
SL_FRAC    = 0.5  # SL at 0.5 ATR
LARGE_THR  = 0.75 # ATR threshold for "large move"
AMP_PCT    = 0.30 # top/bottom 30% for extreme filter

# ── Load ───────────────────────────────────────────────────────────────────────
print("Loading data...")
frames = []
for pair in PAIRS:
    feat  = pd.read_parquet(os.path.join(ROOT_DIR, 'backend', 'data', 'features_8', f'{pair}_geometric.parquet'))
    price = pd.read_parquet(os.path.join(ROOT_DIR, 'backend', 'data', 'processed', f'{pair}_1H.parquet'))
    m = feat[LOAD_FEATS].join(price[['high','low','close']], how='inner')
    m['pair'] = pair
    atr   = m['atr_24'].clip(lower=1e-10).values
    close = m['close'].values
    high  = m['high'].values
    low   = m['low'].values
    # Forward returns ATR-normalized
    for h in [2, 4, 8]:
        r = np.empty(len(m)); r[:] = np.nan
        r[:-h] = np.log(close[h:] / close[:-h]) / atr[:-h]
        m[f'ret_{h}H'] = r
    # SL-managed PnL at T+2 exit (SHORT for bull, LONG for bear)
    # We'll compute this per-setup below
    frames.append(m)

df = pd.concat(frames).reset_index(drop=True)
df = df.dropna(subset=BASE_FEATS)
print(f"Bars: {len(df):,}  |  Pairs: {df['pair'].nunique()}")

close_ = df['close'].values
high_  = df['high'].values
low_   = df['low'].values
atr_   = df['atr_24'].clip(lower=1e-10).values
N      = len(df)

# ── Base setup mask ────────────────────────────────────────────────────────────
base_sig   = np.stack([np.sign(df[f].values) for f in BASE_FEATS], axis=1).sum(axis=1)
full_bull  = base_sig == 3
full_bear  = base_sig == -3
base_setup = full_bull | full_bear

# Signed reversion label: positive = reversion working
rev_2H = np.where(full_bull, -df['ret_2H'].values, df['ret_2H'].values)
rev_4H = np.where(full_bull, -df['ret_4H'].values, df['ret_4H'].values)
rev_8H = np.where(full_bull, -df['ret_8H'].values, df['ret_8H'].values)

# ── SL-managed PnL function ────────────────────────────────────────────────────
def compute_pnl_sl(mask, direction, sl_frac=SL_FRAC, exit_bar=EXIT_BAR):
    """direction: -1=SHORT (bull setup), +1=LONG (bear setup)"""
    idx = np.where(mask)[0]
    idx = idx[idx < N - exit_bar]
    pnl = np.empty(len(idx))
    atr_at = atr_[idx]
    for i, t in enumerate(idx):
        sl_abs = atr_at[i] * sl_frac
        entry  = close_[t]
        sl_lvl = entry - direction * sl_abs
        stopped = False
        for k in range(1, exit_bar + 1):
            if direction == -1 and high_[t+k] >= sl_lvl:
                pnl[i] = -sl_abs; stopped = True; break
            elif direction == 1 and low_[t+k] <= sl_lvl:
                pnl[i] = -sl_abs; stopped = True; break
        if not stopped:
            pnl[i] = direction * (close_[t + exit_bar] - entry)
    return pnl / atr_at   # ATR-normalized

# ── Baseline: base combo without any amplitude filter ─────────────────────────
def report_setup(mask, label=''):
    rev = rev_2H[mask]
    rev4 = rev_4H[mask]
    valid_rev = rev[~np.isnan(rev)]
    valid_rev4 = rev4[~np.isnan(rev4)]
    n = mask.sum()

    bull_m = mask & full_bull
    bear_m = mask & full_bear
    pnl_b  = compute_pnl_sl(bull_m, -1) if bull_m.sum() > 10 else np.array([])
    pnl_be = compute_pnl_sl(bear_m, +1) if bear_m.sum() > 10 else np.array([])
    pnl    = np.concatenate([pnl_b, pnl_be]) if len(pnl_b) and len(pnl_be) else (pnl_b if len(pnl_b) else pnl_be)

    wr      = (pnl > 0).mean()  if len(pnl) else np.nan
    wins    = pnl[pnl > 0];  losses = pnl[pnl < 0]
    mw      = wins.mean()   if len(wins)   else np.nan
    ml      = losses.mean() if len(losses) else np.nan
    ratio   = mw / abs(ml)  if (not np.isnan(mw) and ml != 0) else np.nan
    ev      = pnl.mean()    if len(pnl) else np.nan
    pct_lg  = (valid_rev4 > LARGE_THR).mean() if len(valid_rev4) else np.nan
    med4    = np.median(valid_rev4) if len(valid_rev4) else np.nan
    mean4   = valid_rev4.mean()     if len(valid_rev4) else np.nan

    return {
        'label': label, 'n': n,
        'mean_rev4H': mean4, 'median_rev4H': med4,
        'pct_large': pct_lg,
        'wr': wr, 'payoff': ratio, 'ev_atr': ev,
        'n_bull': bull_m.sum(), 'n_bear': bear_m.sum(),
    }

# ── Print helper ───────────────────────────────────────────────────────────────
def print_row(r):
    pct_base_n = r['n'] / base_setup.sum() * 100
    print(f"  {r['label']:<52} n={r['n']:>7,}({pct_base_n:>4.1f}%) "
          f"mean4H={r['mean_rev4H']:>7.4f} med4H={r['median_rev4H']:>7.4f} "
          f">0.75ATR={r['pct_large']:>5.1%} "
          f"WR={r['wr']:>5.3f} pay={r['payoff']:>5.3f} EV={r['ev_atr']:>7.4f}")

# ── Run ────────────────────────────────────────────────────────────────────────
print()
print("=" * 120)
print("LARGE MOVE SCANNER  —  base: slope_close_3h + intrabar_slope + intrabar_momentum")
print(f"Amplitude filter: top/bottom {AMP_PCT*100:.0f}% of each amplifier feature")
print(f"SL={SL_FRAC} ATR  |  Exit=T+{EXIT_BAR}  |  Large move threshold={LARGE_THR} ATR at T+4H")
print("=" * 120)

# Baseline
print("\n[ BASELINE — no amplitude filter ]")
base_r = report_setup(base_setup, 'BASE (no filter)')
print_row(base_r)

# Single amplifier filters
print("\n[ SINGLE AMPLITUDE FILTER ]")
print(f"  {'Filter':<52} {'n':>8} {'mean4H':>8} {'med4H':>8} {'>0.75ATR':>9} {'WR':>6} {'pay':>6} {'EV':>8}")
print("  " + "-"*110)

single_results = []
for feat, direction in AMPLIFIERS:
    col = df[feat].values
    if direction == 'high':
        thr = np.nanpercentile(col, (1 - AMP_PCT) * 100)
        amp_mask = col >= thr
    elif direction == 'low':
        thr = np.nanpercentile(col, AMP_PCT * 100)
        amp_mask = col <= thr
    else:  # abs
        thr = np.nanpercentile(np.abs(col[~np.isnan(col)]), (1 - AMP_PCT) * 100)
        amp_mask = np.abs(col) >= thr

    mask = base_setup & amp_mask
    if mask.sum() < 500: continue
    r = report_setup(mask, f'{feat} ({direction})')
    single_results.append((r, feat, direction, amp_mask))
    print_row(r)

# Double amplifier filters (top combinations)
print("\n[ DOUBLE AMPLITUDE FILTER ]")
print(f"  {'Filter':<52} {'n':>8} {'mean4H':>8} {'med4H':>8} {'>0.75ATR':>9} {'WR':>6} {'pay':>6} {'EV':>8}")
print("  " + "-"*110)

double_results = []
for (r1, f1, d1, m1), (r2, f2, d2, m2) in combinations(single_results, 2):
    mask = base_setup & m1 & m2
    if mask.sum() < 300: continue
    r = report_setup(mask, f'{f1} + {f2}')
    double_results.append(r)

double_results.sort(key=lambda x: x['mean_rev4H'], reverse=True)
for r in double_results[:20]:
    print_row(r)

# Triple amplifier filters
print("\n[ TRIPLE AMPLITUDE FILTER — top 15 by mean 4H reversion ]")
print(f"  {'Filter':<52} {'n':>8} {'mean4H':>8} {'med4H':>8} {'>0.75ATR':>9} {'WR':>6} {'pay':>6} {'EV':>8}")
print("  " + "-"*110)

triple_results = []
for (r1,f1,d1,m1),(r2,f2,d2,m2),(r3,f3,d3,m3) in combinations(single_results, 3):
    mask = base_setup & m1 & m2 & m3
    if mask.sum() < 150: continue
    r = report_setup(mask, f'{f1} + {f2} + {f3}')
    triple_results.append(r)

triple_results.sort(key=lambda x: x['mean_rev4H'], reverse=True)
for r in triple_results[:15]:
    print_row(r)

print(f"\nDone.")
