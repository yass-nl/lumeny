"""
Exhaustion reversion scan.

Idea: when price is far from its MA (here: regression line = residual_*),
the move is "exhausted" and a correction is likely.

Signal: residual_Xh is in the top/bottom N% (extreme stretch)
Direction: residual > 0 (above MA) -> SHORT correction
           residual < 0 (below MA) -> LONG correction

We test:
  - Windows: 6h, 12h, 24h
  - Thresholds: top/bottom 5%, 10%, 15%, 20%
  - Horizons: t+1H to t+12H (correction may take longer than reversion)
  - SL simulation at 0.5 ATR (exit T+4H)

Per-pair percentile normalization to avoid ATR scale bias across pairs.
Exclude hours 20-21 UTC (untradeable dead zone).
"""

import sys, os
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd
import numpy as np
from scipy.stats import rankdata
import warnings
warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR   = os.path.dirname(SCRIPT_DIR)

PAIRS = [
    'AUDJPY','AUDNZD','AUDUSD','CADJPY','CHFJPY',
    'EURAUD','EURGBP','EURJPY','EURUSD','GBPJPY',
    'GBPUSD','NZDUSD','USDCAD','USDCHF','USDJPY',
]

DEAD_HOURS     = {20, 21}
RESIDUAL_FEATS = ['residual_6h', 'residual_12h', 'residual_24h']
THRESHOLDS     = [0.05, 0.10, 0.15, 0.20]
HORIZONS       = [1, 2, 4, 6, 8, 12]
LOAD_FEATS     = RESIDUAL_FEATS + ['atr_24']

# ── Load ───────────────────────────────────────────────────────────────────────
print("Loading data...")
frames = []
for pair in PAIRS:
    feat  = pd.read_parquet(os.path.join(ROOT_DIR, 'backend', 'data', 'features_8', f'{pair}_geometric.parquet'))
    price = pd.read_parquet(os.path.join(ROOT_DIR, 'backend', 'data', 'processed', f'{pair}_1H.parquet'))
    m = feat[LOAD_FEATS].join(price[['high', 'low', 'close']], how='inner')
    m['pair'] = pair
    m['hour'] = m.index.hour
    m['year'] = m.index.year
    frames.append(m)

df = pd.concat(frames)

# Exclude dead hours
df = df[~df['hour'].isin(DEAD_HOURS)].copy()

# Per-pair percentile rank AFTER filtering (NaN-safe)
for f in RESIDUAL_FEATS:
    df[f'{f}_pct'] = np.nan
    for pair in PAIRS:
        mask = (df['pair'] == pair).values
        col  = df[f].values[mask]
        pct  = np.full(len(col), np.nan)
        valid = ~np.isnan(col)
        if valid.sum() > 0:
            pct[valid] = rankdata(col[valid], method='average') / valid.sum()
        idx = np.where(mask)[0]
        df.iloc[idx, df.columns.get_loc(f'{f}_pct')] = pct

df = df.reset_index(drop=True)

close_ = df['close'].values
high_  = df['high'].values
low_   = df['low'].values
atr_   = df['atr_24'].clip(lower=1e-10).values
N      = len(df)

print(f"Bars after hour filter: {N:,}  |  Pairs: {df['pair'].nunique()}")
print(f"Percentile range check residual_24h_pct: {df['residual_24h_pct'].min():.4f} - {df['residual_24h_pct'].max():.4f}")

# Precompute forward returns (log, ATR-normalized, positional)
fwd = {}
for h in HORIZONS:
    r = np.empty(N); r[:] = np.nan
    r[:-h] = np.log(close_[h:] / close_[:-h]) / atr_[:-h]
    fwd[h] = r

# ── SL simulation ──────────────────────────────────────────────────────────────
def sim_sl(mask, direction, sl_frac, exit_bar):
    idx = np.where(mask)[0]
    idx = idx[idx < N - exit_bar]
    if len(idx) == 0:
        return np.array([])
    sl_abs = atr_[idx] * sl_frac
    entry  = close_[idx]
    sl_lvl = entry - direction * sl_abs
    pnl    = np.empty(len(idx))
    for i, t in enumerate(idx):
        stopped = False
        for k in range(1, exit_bar + 1):
            if direction == -1 and high_[t+k] >= sl_lvl[i]:
                pnl[i] = -sl_abs[i]; stopped = True; break
            elif direction == 1 and low_[t+k] <= sl_lvl[i]:
                pnl[i] = -sl_abs[i]; stopped = True; break
        if not stopped:
            pnl[i] = direction * (close_[t + exit_bar] - entry[i])
    return pnl / atr_[idx]

def summarize(pnl_all):
    if len(pnl_all) == 0:
        return np.nan, np.nan, np.nan
    wins   = pnl_all[pnl_all > 0]
    losses = pnl_all[pnl_all < 0]
    mw     = wins.mean()   if len(wins)   > 0 else np.nan
    ml     = losses.mean() if len(losses) > 0 else np.nan
    payoff = mw / abs(ml)  if (not np.isnan(mw) and ml is not None and ml != 0) else np.nan
    ev     = pnl_all.mean()
    return ev, payoff, len(pnl_all)

# ── Scan per residual feature ──────────────────────────────────────────────────
print()
print("=" * 115)
print("EXHAUSTION REVERSION SCAN  --  Signal: price stretched from regression line")
print("Direction: residual > 0 -> SHORT  |  residual < 0 -> LONG")
print("Exclude: hours 20-21 UTC  |  Per-pair percentile normalization")
print("=" * 115)

all_results = []

for feat in RESIDUAL_FEATS:
    pct_col = df[f'{feat}_pct'].values

    print(f"\n{'='*80}")
    print(f"FEATURE: {feat}")
    print(f"{'='*80}")
    print(f"  {'Threshold':>10} {'n_setups':>10} {'WR_1H':>8} {'WR_2H':>8} {'WR_4H':>8} {'WR_8H':>8} {'mean_1H':>9} {'mean_4H':>9} {'EV_SL05(4H)':>12} {'payoff':>8}")
    print("  " + "-" * 105)

    for thr in THRESHOLDS:
        above = pct_col >= (1 - thr)   # above MA -> SHORT
        below = pct_col <= thr          # below MA -> LONG
        n_setups = int(above.sum() + below.sum())

        wrs   = {}
        means = {}
        for h in HORIZONS:
            lbl   = fwd[h]
            valid = ~np.isnan(lbl)
            na = (above & valid).sum()
            nb = (below & valid).sum()
            wr_a = (lbl[above & valid] < 0).mean() if na > 10 else np.nan
            wr_b = (lbl[below & valid] > 0).mean() if nb > 10 else np.nan
            wrs[h] = np.nanmean([wr_a, wr_b])
            m_a = -lbl[above & valid].mean() if na > 10 else np.nan
            m_b =  lbl[below & valid].mean() if nb > 10 else np.nan
            means[h] = np.nanmean([m_a, m_b])

        pnl_s = sim_sl(above, -1, 0.5, exit_bar=4)
        pnl_l = sim_sl(below, +1, 0.5, exit_bar=4)
        pnl_a = np.concatenate([pnl_s, pnl_l]) if (len(pnl_s) and len(pnl_l)) else (pnl_s if len(pnl_s) else pnl_l)
        ev, payoff, nt = summarize(pnl_a)

        print(f"  top/bot {thr*100:.0f}%{'':<3} {n_setups:>10,} "
              f"{wrs.get(1,np.nan):>8.4f} {wrs.get(2,np.nan):>8.4f} "
              f"{wrs.get(4,np.nan):>8.4f} {wrs.get(8,np.nan):>8.4f} "
              f"{means.get(1,np.nan):>9.4f} {means.get(4,np.nan):>9.4f} "
              f"{ev:>12.4f} {payoff:>8.3f}")

        all_results.append({
            'feature': feat, 'threshold': thr, 'n': n_setups,
            'wr_1H': wrs.get(1), 'wr_4H': wrs.get(4), 'wr_8H': wrs.get(8),
            'mean_1H': means.get(1), 'mean_4H': means.get(4),
            'ev_sl05_4H': ev, 'payoff_sl05_4H': payoff,
        })

# ── Combined: all 3 residuals agree ──────────────────────────────────────────
print()
print("=" * 115)
print("COMBINED: all 3 residuals in top/bot X% simultaneously")
print("= price stretched across 6h + 12h + 24h regression lines at once")
print("=" * 115)

for thr in THRESHOLDS:
    above_all = (
        (df['residual_6h_pct'].values  >= (1 - thr)) &
        (df['residual_12h_pct'].values >= (1 - thr)) &
        (df['residual_24h_pct'].values >= (1 - thr))
    )
    below_all = (
        (df['residual_6h_pct'].values  <= thr) &
        (df['residual_12h_pct'].values <= thr) &
        (df['residual_24h_pct'].values <= thr)
    )
    n = int(above_all.sum() + below_all.sum())

    wrs = {}
    means = {}
    for h in HORIZONS:
        lbl   = fwd[h]
        valid = ~np.isnan(lbl)
        na = (above_all & valid).sum()
        nb = (below_all & valid).sum()
        wr_a = (lbl[above_all & valid] < 0).mean() if na > 5 else np.nan
        wr_b = (lbl[below_all & valid] > 0).mean() if nb > 5 else np.nan
        wrs[h] = np.nanmean([wr_a, wr_b])
        m_a = -lbl[above_all & valid].mean() if na > 5 else np.nan
        m_b =  lbl[below_all & valid].mean() if nb > 5 else np.nan
        means[h] = np.nanmean([m_a, m_b])

    pnl_s = sim_sl(above_all, -1, 0.5, exit_bar=4)
    pnl_l = sim_sl(below_all, +1, 0.5, exit_bar=4)
    pnl_a = np.concatenate([pnl_s, pnl_l]) if (len(pnl_s) and len(pnl_l)) else (pnl_s if len(pnl_s) else pnl_l)
    ev, payoff, nt = summarize(pnl_a)

    print(f"\n  top/bot {thr*100:.0f}%  n={n:,}  (above={above_all.sum():,}  below={below_all.sum():,})")
    print(f"  WR:   1H={wrs.get(1,np.nan):.4f}  2H={wrs.get(2,np.nan):.4f}  4H={wrs.get(4,np.nan):.4f}  8H={wrs.get(8,np.nan):.4f}  12H={wrs.get(12,np.nan):.4f}")
    print(f"  Mean corr (ATR): 1H={means.get(1,np.nan):.4f}  4H={means.get(4,np.nan):.4f}  8H={means.get(8,np.nan):.4f}")
    print(f"  SL=0.5 ATR exit=4H:  EV={ev:.4f}  payoff={payoff:.3f}  n_trades={nt:,}")

# ── Year-by-year for residual_24h top/bot 10% ────────────────────────────────
print()
print("=" * 115)
print("YEAR-BY-YEAR: residual_24h top/bot 10%  (SL=0.5 ATR, exit=4H)")
print("=" * 115)

thr     = 0.10
pct_col = df['residual_24h_pct'].values
above_m = pct_col >= (1 - thr)
below_m = pct_col <= thr
year_   = df['year'].values

print(f"\n  {'Year':>6} {'n':>7} {'WR_1H':>8} {'WR_4H':>8} {'mean_4H':>9} {'EV_SL05':>9} {'payoff':>8}")
print("  " + "-" * 62)

yearly = []
for yr in sorted(df['year'].unique()):
    yr_mask = year_ == yr
    ab = above_m & yr_mask
    bl = below_m & yr_mask

    lbl1 = fwd[1]
    lbl4 = fwd[4]
    v1 = ~np.isnan(lbl1)
    v4 = ~np.isnan(lbl4)

    wr1 = np.nanmean([(lbl1[ab & v1] < 0).mean() if (ab & v1).sum() > 5 else np.nan,
                      (lbl1[bl & v1] > 0).mean() if (bl & v1).sum() > 5 else np.nan])
    wr4 = np.nanmean([(lbl4[ab & v4] < 0).mean() if (ab & v4).sum() > 5 else np.nan,
                      (lbl4[bl & v4] > 0).mean() if (bl & v4).sum() > 5 else np.nan])
    m4  = np.nanmean([-lbl4[ab & v4].mean() if (ab & v4).sum() > 5 else np.nan,
                       lbl4[bl & v4].mean() if (bl & v4).sum() > 5 else np.nan])

    pnl_s = sim_sl(ab, -1, 0.5, exit_bar=4)
    pnl_l = sim_sl(bl, +1, 0.5, exit_bar=4)
    pnl_y = np.concatenate([pnl_s, pnl_l]) if (len(pnl_s) and len(pnl_l)) else (pnl_s if len(pnl_s) else pnl_l)
    ev, payoff, nt = summarize(pnl_y)

    flag = " +" if ev > 0 else " -"
    print(f"  {int(yr):>6} {nt:>7,} {wr1:>8.4f} {wr4:>8.4f} {m4:>9.4f} {ev:>9.4f} {payoff:>8.3f}{flag}")
    yearly.append({'year': yr, 'ev': ev, 'n': nt})

if yearly:
    ydf = pd.DataFrame(yearly)
    print(f"\n  Mean EV: {ydf['ev'].mean():.4f}  |  Profitable years: {(ydf['ev']>0).sum()}/{len(ydf)}")

pd.DataFrame(all_results).to_csv(os.path.join(SCRIPT_DIR, 'exhaustion_results.csv'), index=False)
print(f"\nSaved to notebooks_10/exhaustion_results.csv")
