"""
SL asymmetry test.

For each setup (bull/bear consensus), simulate a trade with:
  - Entry: close[T]  (market order at bar close)
  - Direction: SHORT for bull setup (reversion expected), LONG for bear setup
  - Stop loss: fixed at SL_pips above entry (short) or below entry (long)
  - Exit: at close[T+2] IF stop not hit, else at SL price

SL hit detection uses bar high/low (conservative):
  - SHORT: SL hit if high[T+k] >= entry + SL  (worst case, assume hit at SL)
  - LONG:  SL hit if low[T+k]  <= entry - SL

We test multiple SL sizes (in ATR fractions and absolute pips).
Goal: find combos + SL combos where mean win > mean loss (positive asymmetry).
"""

import sys, os
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR   = os.path.dirname(SCRIPT_DIR)

PAIRS = [
    'AUDJPY','AUDNZD','AUDUSD','CADJPY','CHFJPY',
    'EURAUD','EURGBP','EURJPY','EURUSD','GBPJPY',
    'GBPUSD','NZDUSD','USDCAD','USDCHF','USDJPY',
]

FEAT_COLS = ['slope_close_3h', 'slope_close_6h', 'intrabar_slope', 'intrabar_momentum', 'bar_direction']

COMBOS = {
    'slope_close_3h + intrabar_slope + intrabar_momentum':
        ['slope_close_3h', 'intrabar_slope', 'intrabar_momentum'],
    'slope_close_3h + slope_close_6h + intrabar_slope + intrabar_momentum':
        ['slope_close_3h', 'slope_close_6h', 'intrabar_slope', 'intrabar_momentum'],
    'bar_direction + slope_close_3h + intrabar_momentum':
        ['bar_direction', 'slope_close_3h', 'intrabar_momentum'],
}

# SL sizes as fraction of ATR_24 at signal bar
SL_ATR_FRACS = [0.25, 0.50, 0.75, 1.0, 1.5]
EXIT_BAR = 2   # exit at close[T+2] if not stopped

# ── Load data ──────────────────────────────────────────────────────────────────
print("Loading data...")
frames = []
for pair in PAIRS:
    feat  = pd.read_parquet(os.path.join(ROOT_DIR, 'backend', 'data', 'features_8', f'{pair}_geometric.parquet'))
    feat  = feat[FEAT_COLS + ['atr_24']].copy()
    price = pd.read_parquet(os.path.join(ROOT_DIR, 'backend', 'data', 'processed', f'{pair}_1H.parquet'))
    merged = feat.join(price[['open','high','low','close']], how='inner')
    merged['pair'] = pair
    frames.append(merged)

df = pd.concat(frames).reset_index(drop=True)
df = df.dropna(subset=FEAT_COLS)

close  = df['close'].values
high   = df['high'].values
low    = df['low'].values
atr    = df['atr_24'].clip(lower=1e-10).values
N      = len(df)

print(f"Total bars: {N:,}  |  Pairs: {df['pair'].nunique()}")
print()

# ── Simulate SL-managed trade ──────────────────────────────────────────────────
def simulate_trades(setup_mask, direction, sl_sizes_abs):
    """
    direction: +1 = LONG (bear reversion), -1 = SHORT (bull reversion)
    sl_sizes_abs: array of SL distances in price units (per bar, vectorized)
    Returns dict of SL_frac -> array of PnL per trade (in price units)
    """
    idx = np.where(setup_mask)[0]
    # Remove last EXIT_BAR bars to avoid out-of-bounds
    idx = idx[idx < N - EXIT_BAR]

    results = {}
    for sl_frac in SL_ATR_FRACS:
        sl_abs = atr[idx] * sl_frac   # SL distance per trade

        entry   = close[idx]
        sl_lvl  = entry - direction * sl_abs   # SL price level

        pnl = np.empty(len(idx))

        for i, (t, sl, entry_p, sl_l) in enumerate(zip(idx, sl_abs, entry, sl_lvl)):
            stopped = False
            for k in range(1, EXIT_BAR + 1):
                if direction == -1:  # SHORT: stopped if high >= sl_level
                    if high[t + k] >= sl_l:
                        pnl[i] = -sl   # lose SL amount
                        stopped = True
                        break
                else:  # LONG: stopped if low <= sl_level
                    if low[t + k] <= sl_l:
                        pnl[i] = -sl
                        stopped = True
                        break
            if not stopped:
                # Exit at close[T+EXIT_BAR]
                pnl[i] = direction * (close[t + EXIT_BAR] - entry_p)

        results[sl_frac] = pnl

    return results, idx

# ── Run per combo ──────────────────────────────────────────────────────────────
all_results = []

print("=" * 90)
print("SL ASYMMETRY TEST  —  EXIT at close[T+2] or SL (whichever first)")
print("SL hit: SHORT -> bar high >= SL level  |  LONG -> bar low <= SL level")
print("=" * 90)

for combo_name, feats in COMBOS.items():
    size = len(feats)
    signals   = np.stack([np.sign(df[f].values) for f in feats], axis=1)
    consensus = signals.sum(axis=1)
    full_bull  = consensus == size    # SHORT trade
    full_bear  = consensus == -size   # LONG trade

    print(f"\n{'='*70}")
    print(f"COMBO: {combo_name}")
    print(f"  Bull setups (SHORT): {full_bull.sum():,}  |  Bear setups (LONG): {full_bear.sum():,}")
    print(f"{'='*70}")

    bull_results, bull_idx = simulate_trades(full_bull, direction=-1, sl_sizes_abs=None)
    bear_results, bear_idx = simulate_trades(full_bear, direction=+1, sl_sizes_abs=None)

    print(f"\n  {'SL(ATR)':>8} {'SL%Hit':>8} {'WR':>7} {'MeanW':>9} {'MeanL':>9} {'Ratio':>7} {'EV(ATR)':>9} {'n':>7}")
    print("  " + "-" * 72)

    for sl_frac in SL_ATR_FRACS:
        b_pnl = bull_results[sl_frac]
        be_pnl = bear_results[sl_frac]
        all_pnl = np.concatenate([b_pnl, be_pnl])

        # Normalize by ATR at signal bar for comparability
        b_atr  = atr[bull_idx]
        be_atr = atr[bear_idx]
        all_atr = np.concatenate([b_atr, be_atr])
        pnl_norm = all_pnl / all_atr   # in ATR units

        n        = len(all_pnl)
        sl_hits  = (all_pnl < 0) & (np.abs(all_pnl / all_atr + sl_frac) < 0.01)
        sl_pct   = sl_hits.mean()

        wins     = pnl_norm[pnl_norm > 0]
        losses   = pnl_norm[pnl_norm < 0]
        wr       = len(wins) / n
        mean_w   = wins.mean()  if len(wins)  > 0 else 0
        mean_l   = losses.mean() if len(losses) > 0 else 0
        ratio    = mean_w / abs(mean_l) if mean_l != 0 else np.nan
        ev       = pnl_norm.mean()

        flag = " <<" if ratio > 1.0 and wr > 0.45 else ""

        print(f"  {sl_frac:>8.2f} {sl_pct:>8.3f} {wr:>7.3f} {mean_w:>9.4f} {mean_l:>9.4f} {ratio:>7.3f} {ev:>9.5f} {n:>7,}{flag}")

        all_results.append({
            'combo': combo_name, 'sl_frac': sl_frac,
            'sl_hit_rate': sl_pct, 'wr': wr,
            'mean_win_atr': mean_w, 'mean_loss_atr': mean_l,
            'payoff_ratio': ratio, 'ev_atr': ev, 'n': n,
        })

# ── Per-pair breakdown for best combo ─────────────────────────────────────────
print()
print("=" * 90)
print("PER-PAIR BREAKDOWN  —  best combo, SL=0.5 ATR")
print("=" * 90)
best_feats = ['slope_close_3h', 'intrabar_slope', 'intrabar_momentum']
best_size  = 3

print(f"\n  {'Pair':<10} {'SL%Hit':>8} {'WR':>7} {'MeanW(ATR)':>11} {'MeanL(ATR)':>11} {'Ratio':>7} {'EV(ATR)':>9} {'n':>7}")
print("  " + "-" * 72)

for pair in sorted(df['pair'].unique()):
    sub    = df[df['pair'] == pair].copy()
    sidx   = sub.index  # original positions in df
    pos    = np.where(df['pair'].values == pair)[0]

    sig_p  = np.stack([np.sign(sub[f].values) for f in best_feats], axis=1).sum(axis=1)
    fb_p   = sig_p == best_size
    be_p   = sig_p == -best_size

    close_p = close[pos]
    high_p  = high[pos]
    low_p   = low[pos]
    atr_p   = atr[pos]
    Np      = len(pos)

    sl_frac = 0.5
    all_pnl_pair = []
    all_atr_pair = []

    for mask, direction in [(fb_p, -1), (be_p, +1)]:
        local_idx = np.where(mask)[0]
        local_idx = local_idx[local_idx < Np - EXIT_BAR]
        if len(local_idx) == 0:
            continue
        sl_abs = atr_p[local_idx] * sl_frac
        entry  = close_p[local_idx]
        sl_lvl = entry - direction * sl_abs

        for i, (t, sl, ep, sl_l) in enumerate(zip(local_idx, sl_abs, entry, sl_lvl)):
            stopped = False
            pnl_val = 0
            for k in range(1, EXIT_BAR + 1):
                if direction == -1:
                    if high_p[t+k] >= sl_l:
                        pnl_val = -sl; stopped = True; break
                else:
                    if low_p[t+k] <= sl_l:
                        pnl_val = -sl; stopped = True; break
            if not stopped:
                pnl_val = direction * (close_p[t + EXIT_BAR] - ep)
            all_pnl_pair.append(pnl_val)
            all_atr_pair.append(atr_p[t])

    if len(all_pnl_pair) == 0:
        continue

    pnl_arr  = np.array(all_pnl_pair)
    atr_arr  = np.array(all_atr_pair)
    pnl_norm = pnl_arr / atr_arr

    wins_  = pnl_norm[pnl_norm > 0]
    losses_= pnl_norm[pnl_norm < 0]
    sl_h   = (pnl_arr < 0) & (np.abs(np.abs(pnl_norm) - sl_frac) < 0.01)
    wr_    = len(wins_) / len(pnl_norm)
    mw_    = wins_.mean()  if len(wins_)   > 0 else 0
    ml_    = losses_.mean() if len(losses_) > 0 else 0
    ratio_ = mw_ / abs(ml_) if ml_ != 0 else np.nan
    ev_    = pnl_norm.mean()
    flag   = " <<" if ratio_ > 1.0 else ""

    print(f"  {pair:<10} {sl_h.mean():>8.3f} {wr_:>7.3f} {mw_:>11.4f} {ml_:>11.4f} {ratio_:>7.3f} {ev_:>9.5f} {len(pnl_norm):>7,}{flag}")

res_df = pd.DataFrame(all_results)
res_df.to_csv(os.path.join(SCRIPT_DIR, 'sl_asymmetry_results.csv'), index=False)
print(f"\nSaved to notebooks_10/sl_asymmetry_results.csv")
