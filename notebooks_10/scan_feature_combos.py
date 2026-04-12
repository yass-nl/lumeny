"""
Feature combination scanner for directional edge.

For each combination of signed features:
  1. Binarize each feature: +1 if > 0 (bullish), -1 if < 0 (bearish)
  2. Sum -> consensus score
  3. When all agree (full bull or full bear), check reversion win rate
  4. Report: win rate, mean label, n_bars, edge score

Edge is measured as reversion: bull consensus -> expect negative label (mean reversion).
"""

import os
import sys
import pandas as pd
import numpy as np
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

# Force UTF-8 output on Windows
sys.stdout.reconfigure(encoding='utf-8')

# ── Config ─────────────────────────────────────────────────────────────────────
COMBO_SIZES = [2, 3, 4]
MIN_BARS    = 500    # min bars per side (bull/bear) to include a combo
TOP_N       = 30     # top combos to print per size
ALL_PAIRS   = True

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR    = os.path.dirname(SCRIPT_DIR)
OUTPUT_CSV  = os.path.join(SCRIPT_DIR, 'feature_combo_results.csv')

# Reversion features: when positive -> next bar tends to be negative
CANDIDATE_FEATS = [
    'bar_direction',
    'slope_close_3h',
    'slope_close_6h',
    'slope_close_12h',
    'residual_6h',
    'residual_12h',
    'residual_24h',
    'intrabar_slope',
    'intrabar_momentum',
    'high_slope_6h',
    'low_slope_6h',
    'high_slope_12h',
    'envelope_squeeze_6h',
    'envelope_squeeze_12h',
    '4h_slope_8h',
    '4h_slope_16h',
]

# ── Load data ──────────────────────────────────────────────────────────────────
print("Loading data...")
parquet = 'all_pairs_geometric.parquet' if ALL_PAIRS else 'EURUSD_geometric.parquet'
df = pd.read_parquet(os.path.join(ROOT_DIR, 'backend', 'data', 'features_8', parquet))
df = df.dropna(subset=['label_1H'])
df = df[CANDIDATE_FEATS + ['label_1H', 'pair']].dropna()
print(f"Bars: {len(df):,}  |  Pairs: {df['pair'].nunique()}")

# Pre-compute binarized signals as numpy arrays (fast indexing)
bin_arrays = {}
for f in CANDIDATE_FEATS:
    bin_arrays[f] = np.sign(df[f].values)

label = df['label_1H'].values
N = len(df)

# ── Combination scanner ────────────────────────────────────────────────────────
results = []

for size in COMBO_SIZES:
    combos = list(combinations(CANDIDATE_FEATS, size))
    print(f"Testing {len(combos)} combinations of size {size}...")

    for combo in combos:
        # Stack binarized signals and sum
        signals = np.stack([bin_arrays[f] for f in combo], axis=1)  # (N, size)
        consensus = signals.sum(axis=1)

        full_bull = consensus == size
        full_bear = consensus == -size

        n_bull = full_bull.sum()
        n_bear = full_bear.sum()

        if n_bull < MIN_BARS or n_bear < MIN_BARS:
            continue

        lbl_bull = label[full_bull]
        lbl_bear = label[full_bear]

        # Reversion: bull setup -> expect negative label (short trade wins)
        bull_rev_wr  = (lbl_bull < 0).mean()
        bull_rev_mean = lbl_bull.mean()

        # Reversion: bear setup -> expect positive label (long trade wins)
        bear_rev_wr  = (lbl_bear > 0).mean()
        bear_rev_mean = lbl_bear.mean()

        rev_wr   = (bull_rev_wr + bear_rev_wr) / 2
        rev_mean = (abs(bull_rev_mean) + abs(bear_rev_mean)) / 2

        n_total  = n_bull + n_bear
        coverage = n_total / N
        edge     = (rev_wr - 0.5) * coverage * 100

        results.append({
            'combo':        '|'.join(combo),  # store as string for CSV
            'size':         size,
            'rev_wr':       rev_wr,
            'bull_rev_wr':  bull_rev_wr,
            'bear_rev_wr':  bear_rev_wr,
            'mean_label':   rev_mean,
            'n_bull':       n_bull,
            'n_bear':       n_bear,
            'n_total':      n_total,
            'coverage_pct': coverage * 100,
            'edge_score':   edge,
        })

results_df = pd.DataFrame(results)

# ── Save ───────────────────────────────────────────────────────────────────────
results_df.to_csv(OUTPUT_CSV, index=False)
print(f"Saved {len(results_df)} results to {OUTPUT_CSV}")

# ── Print results ──────────────────────────────────────────────────────────────
print()
print("=" * 80)
print("TOP COMBINATIONS BY REVERSION WIN RATE")
print("=" * 80)

for size in COMBO_SIZES:
    sub = results_df[results_df['size'] == size]
    if sub.empty:
        continue
    top = sub.nlargest(TOP_N, 'rev_wr')
    print(f"\n-- Size {size} --")
    print(f"{'Combo':<65} {'RevWR':>6} {'BullWR':>7} {'BearWR':>7} {'MeanLbl':>8} {'N':>7} {'Cov%':>5} {'Edge':>6}")
    print("-" * 115)
    for _, row in top.iterrows():
        combo_str = row['combo'].replace('|', ' + ')
        print(f"{combo_str:<65} {row['rev_wr']:>6.3f} {row['bull_rev_wr']:>7.3f} {row['bear_rev_wr']:>7.3f} "
              f"{row['mean_label']:>8.4f} {row['n_total']:>7,} {row['coverage_pct']:>5.1f}% {row['edge_score']:>6.3f}")

print()
print("=" * 80)
print("TOP 40 BY EDGE SCORE (win_rate_deviation x coverage)")
print("=" * 80)
for _, row in results_df.nlargest(40, 'edge_score').iterrows():
    combo_str = row['combo'].replace('|', ' + ')
    print(f"[{row['size']}] {combo_str:<72} RevWR={row['rev_wr']:.3f}  Edge={row['edge_score']:.3f}  N={row['n_total']:,}")
