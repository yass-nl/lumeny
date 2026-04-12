"""
Reversion profile test.

For each setup (bull/bear consensus), instead of cumulative return at t+H,
we look at the INCREMENTAL return at each bar t+1, t+2, ..., t+8 separately.

This tells us: does the reversion happen gradually (smooth drift) or is it
concentrated in one specific bar within the window?

Incremental label at t+k = log(close[T+k] / close[T+k-1]) / ATR[T]
(ATR normalized at signal bar T, so magnitudes are comparable across bars)
"""

import sys
import os
import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

sys.stdout.reconfigure(encoding='utf-8')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR   = os.path.dirname(SCRIPT_DIR)
DATA_1H    = os.path.join(ROOT_DIR, 'backend', 'data', 'processed')
FEAT_DIR   = os.path.join(ROOT_DIR, 'backend', 'data', 'features_8')

HORIZONS = list(range(1, 9))  # t+1 through t+8

PAIRS = [
    'AUDJPY','AUDNZD','AUDUSD','CADJPY','CHFJPY',
    'EURAUD','EURGBP','EURJPY','EURUSD','GBPJPY',
    'GBPUSD','NZDUSD','USDCAD','USDCHF','USDJPY',
]

COMBOS = {
    'slope_close_3h + intrabar_slope + intrabar_momentum':
        ['slope_close_3h', 'intrabar_slope', 'intrabar_momentum'],
    'slope_close_3h + slope_close_6h + intrabar_slope + intrabar_momentum':
        ['slope_close_3h', 'slope_close_6h', 'intrabar_slope', 'intrabar_momentum'],
    'bar_direction + slope_close_3h + intrabar_momentum':
        ['bar_direction', 'slope_close_3h', 'intrabar_momentum'],
}

FEAT_COLS = list(set(f for feats in COMBOS.values() for f in feats))

# ── Load data ──────────────────────────────────────────────────────────────────
print("Loading data...")
frames = []
for pair in PAIRS:
    feat  = pd.read_parquet(os.path.join(FEAT_DIR, f'{pair}_geometric.parquet'))
    feat  = feat[FEAT_COLS + ['atr_24']].copy()
    price = pd.read_parquet(os.path.join(DATA_1H, f'{pair}_1H.parquet'))[['close']]
    merged = feat.join(price, how='inner')
    merged['pair'] = pair

    atr = merged['atr_24'].clip(lower=1e-10).values
    close = merged['close'].values

    # Incremental log return at each bar k, normalized by ATR at signal bar T
    for k in HORIZONS:
        # close[T+k] / close[T+k-1], normalized by atr[T]
        inc = np.empty(len(merged))
        inc[:] = np.nan
        inc[:-k] = np.log(close[k:] / close[k-1:-1]) / atr[:-k]
        merged[f'inc_{k}H'] = inc

    frames.append(merged)

df = pd.concat(frames).reset_index(drop=False)
df = df.dropna(subset=FEAT_COLS)
print(f"Total bars: {len(df):,}  |  Pairs: {df['pair'].nunique()}")

# ── Profile test ───────────────────────────────────────────────────────────────
print()
print("=" * 85)
print("INCREMENTAL REVERSION PROFILE")
print("Each column = mean incremental ATR-normalized return AT that bar (not cumulative)")
print("Bull setup -> expect NEGATIVE incremental returns (reversion)")
print("=" * 85)

all_results = []

for combo_name, feats in COMBOS.items():
    size = len(feats)
    signals = np.stack([np.sign(df[f].values) for f in feats], axis=1)
    consensus = signals.sum(axis=1)
    full_bull = consensus == size
    full_bear = consensus == -size

    n_bull = full_bull.sum()
    n_bear = full_bear.sum()

    print(f"\n--- {combo_name} ---")
    print(f"n_bull={n_bull:,}  n_bear={n_bear:,}")
    print()
    print(f"  {'Bar':<8} {'Bull_WR':>8} {'Bull_mean':>10} {'Bear_WR':>8} {'Bear_mean':>10}  {'Combined_WR':>12}  note")
    print("  " + "-" * 72)

    cumul_bull = np.zeros(n_bull)
    cumul_bear = np.zeros(n_bear)

    for k in HORIZONS:
        col = f'inc_{k}H'
        inc_all = df[col].values
        valid   = ~np.isnan(inc_all)

        inc     = inc_all[valid]
        fb      = full_bull[valid]
        be      = full_bear[valid]

        inc_bull = inc[fb]
        inc_bear = inc[be]

        # Bull setup: reversion = incremental return is negative
        bull_wr   = (inc_bull < 0).mean()
        bull_mean = inc_bull.mean()

        # Bear setup: reversion = incremental return is positive
        bear_wr   = (inc_bear > 0).mean()
        bear_mean = inc_bear.mean()

        combined_wr = (bull_wr + bear_wr) / 2

        # Z-test on combined
        n_wins  = (inc_bull < 0).sum() + (inc_bear > 0).sum()
        n_total = len(inc_bull) + len(inc_bear)
        z       = (combined_wr - 0.5) / np.sqrt(0.25 / n_total)

        note = "<< peak" if abs(combined_wr - 0.5) == max(
            abs((( (inc_all[full_bull & ~np.isnan(inc_all)] < 0).mean() +
                   (inc_all[full_bear & ~np.isnan(inc_all)] > 0).mean()) / 2) - 0.5)
            for inc_all2 in [df[f'inc_{j}H'].values for j in HORIZONS]
            for inc_all in [inc_all2]
        ) else ""

        print(f"  t+{k}H{'':<4} {bull_wr:>8.4f} {bull_mean:>10.4f} {bear_wr:>8.4f} {bear_mean:>10.4f}  {combined_wr:>12.4f}  Z={z:.1f}")

        all_results.append({
            'combo': combo_name,
            'bar': k,
            'bull_wr': bull_wr,
            'bull_mean': bull_mean,
            'bear_wr': bear_wr,
            'bear_mean': bear_mean,
            'combined_wr': combined_wr,
            'z': z,
        })

# ── Summary table: incremental WR across bars ──────────────────────────────────
res_df = pd.DataFrame(all_results)

print()
print("=" * 85)
print("SUMMARY: Combined WR at each incremental bar (is the signal front-loaded or spread?)")
print("=" * 85)
pivot = res_df.pivot(index='combo', columns='bar', values='combined_wr')
pivot.columns = [f't+{k}' for k in pivot.columns]
print(pivot.round(4).to_string())

print()
print("=" * 85)
print("SUMMARY: Mean incremental return (bull side, negative = reversion working)")
print("=" * 85)
pivot2 = res_df.pivot(index='combo', columns='bar', values='bull_mean')
pivot2.columns = [f't+{k}' for k in pivot2.columns]
print(pivot2.round(4).to_string())

res_df.to_csv(os.path.join(SCRIPT_DIR, 'reversion_profile_results.csv'), index=False)
print(f"\nSaved to notebooks_10/reversion_profile_results.csv")
