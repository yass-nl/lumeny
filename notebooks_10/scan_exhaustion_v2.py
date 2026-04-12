"""
Exhaustion feature scan v2.

Tests ALL candidate exhaustion features, not just residuals.
Each feature is evaluated as an exhaustion signal — "price stretched/overextended".

For directional features (signed): high value = bullish exhaustion -> SHORT
For magnitude features (unsigned, always high = exhausted): high = exhausted

Signal logic:
  - Signed features: top X% -> SHORT, bottom X% -> LONG
  - Unsigned/abs features: top X% -> trade opposite to recent direction
    (we detect direction from bar_direction or slope_close_3h sign)

Exclude hours 20-21 UTC.
Per-pair percentile normalization.
Exit at T+4H with SL=0.5 ATR.
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

DEAD_HOURS = {20, 21}

# All candidate exhaustion features with their directionality
# 'signed'   : positive = bullish exhaustion -> top% = SHORT, bottom% = LONG
# 'unsigned' : high magnitude = exhausted (need direction from bar_direction)
CANDIDATES = [
    # Residuals (price vs regression) — signed
    ('residual_6h',          'signed'),
    ('residual_12h',         'signed'),
    ('residual_24h',         'signed'),
    ('4h_residual',          'signed'),
    # Range position — signed (high = near top of range = overbought)
    ('range_pos_12h',        'signed'),
    ('range_pos_24h',        'signed'),
    ('range_pos_48h',        'signed'),
    ('4h_close_pos',         'signed'),
    ('close_position',       'signed'),
    # Slope alignment — signed (high = short > long slope = momentum mature, due for fade)
    ('slope_alignment_3_12', 'signed'),
    ('slope_alignment_6_24', 'signed'),
    # Curvature — signed (positive = accelerating up = bull exhaustion)
    ('curvature_6h',         'signed'),
    ('curvature_12h',        'signed'),
    ('curvature_24h',        'signed'),
    # Envelope squeeze — signed (positive = channel tilted up)
    ('envelope_squeeze_6h',  'signed'),
    ('envelope_squeeze_12h', 'signed'),
    ('envelope_squeeze_24h', 'signed'),
    # Consecutive bars — unsigned (high consec_bull -> SHORT, high consec_bear -> LONG)
    ('consec_bull',          'bull_only'),   # high -> SHORT
    ('consec_bear',          'bear_only'),   # high -> LONG
    # Intrabar — signed
    ('intrabar_slope',       'signed'),
    ('intrabar_momentum',    'signed'),
    ('intrabar_close_pos',   'signed'),
    # Slope close — signed
    ('slope_close_3h',       'signed'),
    ('slope_close_6h',       'signed'),
    ('slope_close_12h',      'signed'),
    ('slope_close_24h',      'signed'),
    # 4H slopes — signed
    ('4h_slope_8h',          'signed'),
    ('4h_slope_16h',         'signed'),
    ('4h_slope_24h',         'signed'),
]

LOAD_FEATS = list(set(f for f, _ in CANDIDATES)) + ['atr_24', 'bar_direction']
THRESHOLDS = [0.05, 0.10, 0.15, 0.20]
EXIT_BAR   = 4
SL_FRAC    = 0.5

# ── Load ───────────────────────────────────────────────────────────────────────
print("Loading data...")
frames = []
for pair in PAIRS:
    feat  = pd.read_parquet(os.path.join(ROOT_DIR, 'backend', 'data', 'features_8', f'{pair}_geometric.parquet'))
    price = pd.read_parquet(os.path.join(ROOT_DIR, 'backend', 'data', 'processed', f'{pair}_1H.parquet'))
    avail = [f for f in LOAD_FEATS if f in feat.columns]
    m = feat[avail].join(price[['high', 'low', 'close']], how='inner')
    m['pair'] = pair
    m['hour'] = m.index.hour
    m['year'] = m.index.year
    frames.append(m)

df = pd.concat(frames)
df = df[~df['hour'].isin(DEAD_HOURS)].copy()
df = df.reset_index(drop=True)

# Per-pair percentile rank (NaN-safe)
print("Computing per-pair percentiles...")
for feat, _ in CANDIDATES:
    if feat not in df.columns:
        continue
    df[f'{feat}_pct'] = np.nan
    for pair in PAIRS:
        mask = (df['pair'] == pair).values
        col  = df[feat].values[mask]
        pct  = np.full(len(col), np.nan)
        valid = ~np.isnan(col)
        if valid.sum() > 0:
            pct[valid] = rankdata(col[valid], method='average') / valid.sum()
        idx = np.where(mask)[0]
        for i, v in zip(idx, pct):
            df.at[i, f'{feat}_pct'] = v

close_ = df['close'].values
high_  = df['high'].values
low_   = df['low'].values
atr_   = df['atr_24'].clip(lower=1e-10).values
N      = len(df)

print(f"Bars after filter: {N:,}")

# Precompute forward returns
fwd = {}
for h in [1, 2, 4, 8]:
    r = np.full(N, np.nan)
    r[:-h] = np.log(close_[h:] / close_[:-h]) / atr_[:-h]
    fwd[h] = r

# ── SL sim ────────────────────────────────────────────────────────────────────
def sim_sl(mask, direction):
    idx = np.where(mask)[0]
    idx = idx[idx < N - EXIT_BAR]
    if len(idx) == 0:
        return np.array([])
    sl_abs = atr_[idx] * SL_FRAC
    entry  = close_[idx]
    sl_lvl = entry - direction * sl_abs
    pnl    = np.empty(len(idx))
    for i, t in enumerate(idx):
        stopped = False
        for k in range(1, EXIT_BAR + 1):
            if direction == -1 and high_[t+k] >= sl_lvl[i]:
                pnl[i] = -sl_abs[i]; stopped = True; break
            elif direction == 1 and low_[t+k] <= sl_lvl[i]:
                pnl[i] = -sl_abs[i]; stopped = True; break
        if not stopped:
            pnl[i] = direction * (close_[t + EXIT_BAR] - entry[i])
    return pnl / atr_[idx]

def stats(pnl_all):
    if len(pnl_all) == 0:
        return np.nan, np.nan, np.nan
    wins   = pnl_all[pnl_all > 0]
    losses = pnl_all[pnl_all < 0]
    mw     = wins.mean()   if len(wins)   > 0 else np.nan
    ml     = losses.mean() if len(losses) > 0 else np.nan
    payoff = mw / abs(ml)  if (not np.isnan(mw) and not np.isnan(ml) and ml != 0) else np.nan
    return pnl_all.mean(), payoff, len(pnl_all)

# ── Scan ──────────────────────────────────────────────────────────────────────
print()
print("=" * 120)
print("EXHAUSTION FEATURE SCAN  --  all candidates, threshold=top/bot 10%")
print(f"Exit T+{EXIT_BAR}H  |  SL={SL_FRAC} ATR  |  Exclude 20-21 UTC  |  Per-pair percentile")
print("=" * 120)
print()
print(f"  {'Feature':<28} {'type':<10} {'n_above':>9} {'n_below':>9} "
      f"{'WR_1H':>7} {'WR_4H':>7} {'WR_8H':>7} "
      f"{'mean_1H':>8} {'mean_4H':>8} "
      f"{'EV':>8} {'payoff':>7}  note")
print("  " + "-" * 115)

results = []
THR = 0.10  # main scan at 10%

for feat, ftype in CANDIDATES:
    if feat not in df.columns:
        continue
    pct = df[f'{feat}_pct'].values

    if ftype == 'signed':
        above = pct >= (1 - THR)   # overbought -> SHORT
        below = pct <= THR          # oversold   -> LONG
        short_mask = above
        long_mask  = below
    elif ftype == 'bull_only':
        # consec_bull high -> SHORT
        short_mask = pct >= (1 - THR)
        long_mask  = np.zeros(N, dtype=bool)
    elif ftype == 'bear_only':
        # consec_bear high -> LONG
        short_mask = np.zeros(N, dtype=bool)
        long_mask  = pct >= (1 - THR)
    else:
        continue

    n_above = short_mask.sum()
    n_below = long_mask.sum()
    if n_above + n_below < 200:
        continue

    # WR at each horizon
    wrs   = {}
    means = {}
    for h in [1, 2, 4, 8]:
        lbl   = fwd[h]
        valid = ~np.isnan(lbl)
        na = (short_mask & valid).sum()
        nb = (long_mask  & valid).sum()
        wr_a = (lbl[short_mask & valid] < 0).mean() if na > 10 else np.nan
        wr_b = (lbl[long_mask  & valid] > 0).mean() if nb > 10 else np.nan
        wrs[h]   = np.nanmean([x for x in [wr_a, wr_b] if not np.isnan(x)])
        m_a = -lbl[short_mask & valid].mean() if na > 10 else np.nan
        m_b =  lbl[long_mask  & valid].mean() if nb > 10 else np.nan
        means[h] = np.nanmean([x for x in [m_a, m_b] if not np.isnan(x)])

    pnl_s = sim_sl(short_mask, -1)
    pnl_l = sim_sl(long_mask,  +1)
    pnl_a = np.concatenate([pnl_s, pnl_l]) if (len(pnl_s) and len(pnl_l)) else (pnl_s if len(pnl_s) else pnl_l)
    ev, payoff, nt = stats(pnl_a)

    note = ''
    if not np.isnan(wrs.get(1, np.nan)) and wrs.get(1, 0) > 0.525:
        note += 'WR1H>52.5% '
    if not np.isnan(ev) and ev > 0.02:
        note += 'EV>0.02 '
    if not np.isnan(payoff) and payoff > 2.5:
        note += 'pay>2.5'

    print(f"  {feat:<28} {ftype:<10} {n_above:>9,} {n_below:>9,} "
          f"{wrs.get(1,np.nan):>7.4f} {wrs.get(4,np.nan):>7.4f} {wrs.get(8,np.nan):>7.4f} "
          f"{means.get(1,np.nan):>8.4f} {means.get(4,np.nan):>8.4f} "
          f"{ev:>8.4f} {payoff:>7.3f}  {note}")

    results.append({
        'feature': feat, 'type': ftype, 'threshold': THR,
        'n_short': n_above, 'n_long': n_below,
        'wr_1H': wrs.get(1), 'wr_4H': wrs.get(4), 'wr_8H': wrs.get(8),
        'mean_1H': means.get(1), 'mean_4H': means.get(4),
        'ev': ev, 'payoff': payoff,
    })

# ── Top features by WR_1H ────────────────────────────────────────────────────
res_df = pd.DataFrame(results).sort_values('wr_1H', ascending=False)
print()
print("=" * 120)
print("TOP 10 FEATURES BY WR_1H")
print("=" * 120)
for _, r in res_df.head(10).iterrows():
    print(f"  {r['feature']:<28} WR_1H={r['wr_1H']:.4f}  WR_4H={r['wr_4H']:.4f}  EV={r['ev']:.4f}  payoff={r['payoff']:.3f}  n={(r['n_short']+r['n_long']):,.0f}")

# ── Top features by EV ───────────────────────────────────────────────────────
res_df2 = pd.DataFrame(results).sort_values('ev', ascending=False)
print()
print("=" * 120)
print("TOP 10 FEATURES BY EV (ATR-normalized, SL=0.5 ATR, exit=4H)")
print("=" * 120)
for _, r in res_df2.head(10).iterrows():
    print(f"  {r['feature']:<28} EV={r['ev']:.4f}  payoff={r['payoff']:.3f}  WR_1H={r['wr_1H']:.4f}  WR_4H={r['wr_4H']:.4f}  n={(r['n_short']+r['n_long']):,.0f}")

# ── Two-feature combinations of top candidates ────────────────────────────────
print()
print("=" * 120)
print("TWO-FEATURE COMBINATIONS  --  top candidates combined (AND logic)")
print("Both features must be in extreme percentile simultaneously")
print("=" * 120)
print()

# Take top 8 by WR_1H as combo candidates
top_feats = res_df.head(8)['feature'].tolist()

from itertools import combinations

combo_results = []
for f1, f2 in combinations(top_feats, 2):
    t1 = dict(CANDIDATES)[f1]
    t2 = dict(CANDIDATES)[f2]

    pct1 = df[f'{f1}_pct'].values
    pct2 = df[f'{f2}_pct'].values

    def get_masks(f, ftype, pct):
        if ftype == 'signed':
            return pct >= (1 - THR), pct <= THR
        elif ftype == 'bull_only':
            return pct >= (1 - THR), np.zeros(N, dtype=bool)
        elif ftype == 'bear_only':
            return np.zeros(N, dtype=bool), pct >= (1 - THR)
        return np.zeros(N, dtype=bool), np.zeros(N, dtype=bool)

    s1, l1 = get_masks(f1, t1, pct1)
    s2, l2 = get_masks(f2, t2, pct2)

    short_m = s1 & s2
    long_m  = l1 & l2
    n = short_m.sum() + long_m.sum()
    if n < 100:
        continue

    wrs = {}
    for h in [1, 4, 8]:
        lbl   = fwd[h]
        valid = ~np.isnan(lbl)
        na = (short_m & valid).sum()
        nb = (long_m  & valid).sum()
        wr_a = (lbl[short_m & valid] < 0).mean() if na > 5 else np.nan
        wr_b = (lbl[long_m  & valid] > 0).mean() if nb > 5 else np.nan
        wrs[h] = np.nanmean([x for x in [wr_a, wr_b] if not np.isnan(x)])

    pnl_s = sim_sl(short_m, -1)
    pnl_l = sim_sl(long_m,  +1)
    pnl_a = np.concatenate([pnl_s, pnl_l]) if (len(pnl_s) and len(pnl_l)) else (pnl_s if len(pnl_s) else pnl_l)
    ev, payoff, nt = stats(pnl_a)

    combo_results.append({
        'combo': f'{f1} + {f2}', 'n': n,
        'wr_1H': wrs.get(1), 'wr_4H': wrs.get(4), 'wr_8H': wrs.get(8),
        'ev': ev, 'payoff': payoff,
    })

combo_results.sort(key=lambda x: x['wr_1H'] if not np.isnan(x['wr_1H']) else 0, reverse=True)
print(f"  {'Combination':<55} {'n':>8} {'WR_1H':>7} {'WR_4H':>7} {'WR_8H':>7} {'EV':>8} {'payoff':>7}")
print("  " + "-" * 105)
for r in combo_results:
    print(f"  {r['combo']:<55} {r['n']:>8,} {r['wr_1H']:>7.4f} {r['wr_4H']:>7.4f} {r['wr_8H']:>7.4f} {r['ev']:>8.4f} {r['payoff']:>7.3f}")

# Save
pd.DataFrame(results).to_csv(os.path.join(SCRIPT_DIR, 'exhaustion_v2_results.csv'), index=False)
print(f"\nSaved to notebooks_10/exhaustion_v2_results.csv")
