"""
Consecutive bar fingerprint scan.

For each bar T, check if the next 4 bars (T+1 to T+4) are ALL bullish or ALL bearish.
"Bullish bar" = close > open (bar_direction = +1)
"Bearish bar" = close < open (bar_direction = -1)

Then: what are the feature values at bar T in each case?

Goal: find features whose average value at T is significantly different
between the "4 consecutive bull" and "4 consecutive bear" setup —
i.e. a fingerprint that precedes sustained directional moves.

This is NOT reversion — this is CONTINUATION.
"""

import sys, os
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR   = os.path.dirname(SCRIPT_DIR)

PAIRS = [
    'AUDJPY','AUDNZD','AUDUSD','CADJPY','CHFJPY',
    'EURAUD','EURGBP','EURJPY','EURUSD','GBPJPY',
    'GBPUSD','NZDUSD','USDCAD','USDCHF','USDJPY',
]

# All explicitly directional features from the table
DIRECTIONAL_FEATS = [
    'bar_direction',
    'wick_asymmetry',
    'slope_close_3h',
    'slope_close_6h',
    'slope_close_12h',
    'slope_close_24h',
    'high_slope_6h',
    'high_slope_12h',
    'high_slope_24h',
    'low_slope_6h',
    'low_slope_12h',
    'low_slope_24h',
    'envelope_squeeze_6h',
    'envelope_squeeze_12h',
    'envelope_squeeze_24h',
    'residual_6h',
    'residual_12h',
    'residual_24h',
    'curvature_6h',
    'curvature_12h',
    'curvature_24h',
    'slope_alignment_3_12',
    'slope_alignment_6_24',
    'intrabar_slope',
    'intrabar_momentum',
    '4h_slope_8h',
    '4h_slope_16h',
    '4h_slope_24h',
    '4h_residual',
]

# ── Load ───────────────────────────────────────────────────────────────────────
print("Loading data...")
frames = []
for pair in PAIRS:
    feat  = pd.read_parquet(os.path.join(ROOT_DIR, 'backend', 'data', 'features_8', f'{pair}_geometric.parquet'))
    price = pd.read_parquet(os.path.join(ROOT_DIR, 'backend', 'data', 'processed', f'{pair}_1H.parquet'))
    m = feat[DIRECTIONAL_FEATS].join(price[['open', 'close']], how='inner')
    m['pair'] = pair
    frames.append(m)

df = pd.concat(frames).reset_index(drop=True)
df = df.dropna(subset=DIRECTIONAL_FEATS)
print(f"Bars: {len(df):,}  |  Pairs: {df['pair'].nunique()}")

close_ = df['close'].values
open_  = df['open'].values
N      = len(df)

# ── Label: are the next 4 bars all bull or all bear? ──────────────────────────
# Bar direction of bar T+k = sign(close[T+k] - open[T+k])
bar_dir = np.sign(close_ - open_)   # +1, -1, 0

# For each bar T, check T+1, T+2, T+3, T+4
next4_bull = np.zeros(N, dtype=bool)
next4_bear = np.zeros(N, dtype=bool)

for t in range(N - 4):
    dirs = bar_dir[t+1:t+5]
    if np.all(dirs == 1):
        next4_bull[t] = True
    elif np.all(dirs == -1):
        next4_bear[t] = True

n_bull = next4_bull.sum()
n_bear = next4_bear.sum()
n_total = N

print(f"\n4-consecutive bull setups: {n_bull:,}  ({n_bull/n_total*100:.2f}% of bars)")
print(f"4-consecutive bear setups: {n_bear:,}  ({n_bear/n_total*100:.2f}% of bars)")
print(f"Neither:                   {n_total-n_bull-n_bear:,}  ({(n_total-n_bull-n_bear)/n_total*100:.2f}% of bars)")

# ── Feature fingerprint ────────────────────────────────────────────────────────
print()
print("=" * 100)
print("FEATURE FINGERPRINT AT BAR T")
print("Comparing: T followed by 4 bull bars  vs  T followed by 4 bear bars")
print("Asymmetry = bull_mean - bear_mean  (positive = feature higher before bull runs)")
print("=" * 100)
print()
print(f"  {'Feature':<28} {'Bull_mean':>10} {'Bear_mean':>10} {'Asymmetry':>11} {'|Asym|/std':>11} {'p-value':>12}  interpretation")
print("  " + "-" * 100)

results = []
for feat in DIRECTIONAL_FEATS:
    col = df[feat].values

    bull_vals = col[next4_bull]
    bear_vals = col[next4_bear]
    all_vals  = col[~np.isnan(col)]

    bull_mean = np.nanmean(bull_vals)
    bear_mean = np.nanmean(bear_vals)
    asymmetry = bull_mean - bear_mean

    # Normalize by overall std to make features comparable
    overall_std = np.nanstd(all_vals)
    asym_norm   = abs(asymmetry) / overall_std if overall_std > 0 else 0

    # T-test
    t_stat, p_val = stats.ttest_ind(
        bull_vals[~np.isnan(bull_vals)],
        bear_vals[~np.isnan(bear_vals)],
        equal_var=False
    )

    # Interpretation: does the feature value PREDICT the direction?
    # If feature > 0 tends to precede bull runs AND feature < 0 tends to precede bear runs
    # -> continuation signal (feature aligns with next move)
    # If opposite -> contrarian / no signal
    aligns_bull = bull_mean > 0 and bear_mean < 0
    aligns_bear = bull_mean < 0 and bear_mean > 0
    if aligns_bull:
        interp = "continuation (+ -> bull, - -> bear)"
    elif aligns_bear:
        interp = "contrarian   (+ -> bear, - -> bull)"
    elif bull_mean > bear_mean:
        interp = "partial+ (higher before bull)"
    else:
        interp = "partial- (lower before bull)"

    results.append({
        'feature': feat,
        'bull_mean': bull_mean,
        'bear_mean': bear_mean,
        'asymmetry': asymmetry,
        'asym_norm': asym_norm,
        'p_value': p_val,
        'interpretation': interp,
    })

# Sort by normalized asymmetry
results.sort(key=lambda x: x['asym_norm'], reverse=True)

for r in results:
    sig = '***' if r['p_value'] < 0.001 else ('**' if r['p_value'] < 0.01 else ('*' if r['p_value'] < 0.05 else ''))
    print(f"  {r['feature']:<28} {r['bull_mean']:>10.4f} {r['bear_mean']:>10.4f} "
          f"{r['asymmetry']:>+11.4f} {r['asym_norm']:>11.4f} {r['p_value']:>12.2e} {sig}  {r['interpretation']}")

# ── Combined score: at bar T, how many features align with the eventual direction? ──
print()
print("=" * 100)
print("ALIGNMENT SCORE: how many features point in the eventual direction at bar T?")
print("(Signed features: positive = bullish. Score = fraction pointing correctly)")
print("=" * 100)

# Only use features with clear continuation pattern and strong asymmetry
cont_feats = [r['feature'] for r in results if 'continuation' in r['interpretation'] and r['asym_norm'] > 0.05]
print(f"\nContinuation features (n={len(cont_feats)}): {cont_feats}")

if cont_feats:
    feat_vals = np.stack([df[f].values for f in cont_feats], axis=1)  # (N, n_feats)
    feat_signs = np.sign(feat_vals)

    # Alignment score for bull setups at T: fraction of features > 0
    bull_scores = feat_signs[next4_bull].mean(axis=1)  # per bar
    bear_scores = feat_signs[next4_bear].mean(axis=1)

    print(f"\nWhen next 4 bars are BULL — mean alignment score: {bull_scores.mean():.4f}  (1.0 = all features bullish)")
    print(f"When next 4 bars are BEAR — mean alignment score: {bear_scores.mean():.4f}  (-1.0 = all features bearish)")
    print(f"All bars — mean alignment score:                  {feat_signs.mean(axis=1).mean():.4f}")
    print()

    # Distribution of alignment scores
    print("Score distribution when next 4 bars BULL:")
    for thr in [0.0, 0.25, 0.50, 0.75, 1.0]:
        pct = (bull_scores >= thr).mean()
        print(f"  >= {thr:.2f}: {pct:.3f}  ({(bull_scores >= thr).sum():,} bars)")
    print()
    print("Score distribution when next 4 bars BEAR:")
    for thr in [0.0, -0.25, -0.50, -0.75, -1.0]:
        pct = (bear_scores <= thr).mean()
        print(f"  <= {thr:.2f}: {pct:.3f}  ({(bear_scores <= thr).sum():,} bars)")

# Save
pd.DataFrame(results).to_csv(os.path.join(SCRIPT_DIR, 'consecutive_fingerprint_results.csv'), index=False)
print(f"\nSaved to notebooks_10/consecutive_fingerprint_results.csv")
