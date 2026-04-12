"""
Multi-horizon reversion test.

Takes the top combos from the feature combo scan and tests whether
the reversion signal holds not just at t+1H but through t+2H, t+4H, t+8H.

Label: ATR-normalized cumulative log return from close[T] to close[T+H]
       i.e. log(close[T+H] / close[T]) / ATR_24[T]

Setup signal is computed at bar T using the same consensus logic as the combo scan.
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

HORIZONS = [1, 2, 3, 4, 6, 8]  # bars forward

PAIRS = [
    'AUDJPY','AUDNZD','AUDUSD','CADJPY','CHFJPY',
    'EURAUD','EURGBP','EURJPY','EURUSD','GBPJPY',
    'GBPUSD','NZDUSD','USDCAD','USDCHF','USDJPY',
]

COMBOS = {
    'intrabar_slope + intrabar_momentum':
        ['intrabar_slope', 'intrabar_momentum'],
    'slope_close_3h + intrabar_slope + intrabar_momentum':
        ['slope_close_3h', 'intrabar_slope', 'intrabar_momentum'],
    'bar_direction + intrabar_slope + intrabar_momentum':
        ['bar_direction', 'intrabar_slope', 'intrabar_momentum'],
    'bar_direction + slope_close_3h + intrabar_momentum':
        ['bar_direction', 'slope_close_3h', 'intrabar_momentum'],
    'bar_direction + slope_close_3h':
        ['bar_direction', 'slope_close_3h'],
    'slope_close_3h + slope_close_6h + intrabar_slope + intrabar_momentum':
        ['slope_close_3h', 'slope_close_6h', 'intrabar_slope', 'intrabar_momentum'],
}

FEAT_COLS = list(set(f for feats in COMBOS.values() for f in feats))

# ── Load and merge data ────────────────────────────────────────────────────────
print("Loading data...")
frames = []
for pair in PAIRS:
    # Features
    feat = pd.read_parquet(os.path.join(FEAT_DIR, f'{pair}_geometric.parquet'))
    feat = feat[FEAT_COLS + ['atr_24']].copy()

    # 1H close prices
    price = pd.read_parquet(os.path.join(DATA_1H, f'{pair}_1H.parquet'))[['close']]

    # Merge on datetime index
    merged = feat.join(price, how='inner')
    merged['pair'] = pair

    # Compute forward returns for each horizon (log return, ATR-normalized)
    for h in HORIZONS:
        fwd_log_ret = np.log(merged['close'].shift(-h) / merged['close'])
        atr = merged['atr_24'].clip(lower=1e-10)
        merged[f'label_{h}H'] = fwd_log_ret / atr

    frames.append(merged)

df = pd.concat(frames)
df = df.dropna(subset=FEAT_COLS)
print(f"Total bars: {len(df):,}  |  Pairs: {df['pair'].nunique()}")
print()

# ── Run multi-horizon test ─────────────────────────────────────────────────────
print("=" * 90)
print("REVERSION WIN RATE BY COMBO AND HORIZON")
print("Interpretation: bull consensus -> expect negative label (reversion)")
print("=" * 90)

results = []

for combo_name, feats in COMBOS.items():
    size = len(feats)
    signals = np.stack([np.sign(df[f].values) for f in feats], axis=1)
    consensus = signals.sum(axis=1)
    full_bull = consensus == size
    full_bear = consensus == -size

    print(f"\n{combo_name}  (n_bull={full_bull.sum():,}  n_bear={full_bear.sum():,})")
    print(f"  {'Horizon':<10} {'RevWR':>7} {'BullWR':>8} {'BearWR':>8} {'MeanLbl_B':>10} {'MeanLbl_Be':>11} {'Z':>7} {'p':>10}")
    print("  " + "-" * 75)

    for h in HORIZONS:
        col = f'label_{h}H'
        lbl_all = df[col].values        # full array, NaN at tail of each pair
        valid   = ~np.isnan(lbl_all)    # positional mask, same length as df

        # Apply valid mask to both label and consensus masks
        lbl     = lbl_all[valid]
        fb_h    = full_bull[valid]
        be_h    = full_bear[valid]

        lbl_bull = lbl[fb_h]
        lbl_bear = lbl[be_h]
        n_bull   = fb_h.sum()
        n_bear   = be_h.sum()

        if n_bull < 100 or n_bear < 100:
            continue

        wr_bull  = (lbl_bull < 0).mean()
        wr_bear  = (lbl_bear > 0).mean()
        rev_wr   = (wr_bull + wr_bear) / 2
        n_total  = n_bull + n_bear
        n_wins   = (lbl_bull < 0).sum() + (lbl_bear > 0).sum()

        z = (rev_wr - 0.5) / np.sqrt(0.25 / n_total)
        p = stats.binomtest(n_wins, n_total, p=0.5, alternative='greater').pvalue
        p_str = f'{p:.2e}' if p > 1e-300 else '~0'

        mean_bull = lbl_bull.mean()
        mean_bear = lbl_bear.mean()

        print(f"  t+{h}H{'':<7} {rev_wr:>7.4f} {wr_bull:>8.4f} {wr_bear:>8.4f} "
              f"{mean_bull:>10.4f} {mean_bear:>11.4f} {z:>7.1f} {p_str:>10}")

        results.append({
            'combo': combo_name,
            'horizon': h,
            'rev_wr': rev_wr,
            'bull_wr': wr_bull,
            'bear_wr': wr_bear,
            'mean_label_bull': mean_bull,
            'mean_label_bear': mean_bear,
            'n_bull': n_bull,
            'n_bear': n_bear,
            'z': z,
            'p': p,
        })

# ── Summary: best horizon per combo ───────────────────────────────────────────
res_df = pd.DataFrame(results)
res_df.to_csv(os.path.join(SCRIPT_DIR, 'multihorizon_results.csv'), index=False)

print()
print("=" * 90)
print("DECAY PROFILE — does WR hold or fade as horizon extends?")
print("=" * 90)
pivot = res_df.pivot(index='combo', columns='horizon', values='rev_wr')
pivot.columns = [f't+{h}H' for h in pivot.columns]
print(pivot.round(4).to_string())

print(f"\nSaved to notebooks_10/multihorizon_results.csv")
