"""
DIAGNOSTIC CHECKS — Is the backtest legitimate?
"""
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

FEATURES_DIR  = Path('backend/data/features')
MODELS_DIR    = Path('backend/models_2')
PROCESSED_DIR = Path('backend/data/processed')

print("Loading dataset...")
df = pd.read_parquet(FEATURES_DIR / 'all_pairs_features_labels.parquet')
df_test  = df[df.index > '2024-06-30'].copy()
df_train = df[df.index <= '2024-06-30'].copy()

HORIZONS       = ['1H', '4H', '1D']
QUANTILES      = [0.10, 0.25, 0.50, 0.75, 0.90]
QUANTILE_NAMES = ['Q10', 'Q25', 'Q50', 'Q75', 'Q90']
ALL_HORIZONS   = ['1H', '4H', '1D', '7D']
label_cols     = [f'label_{h}' for h in ALL_HORIZONS]
feature_cols   = [c for c in df.columns if c not in label_cols + ['pair']]

print(f"Test set:  {len(df_test):,} rows")
print(f"Train set: {len(df_train):,} rows")
print()

# ===================================================================
# CHECK 1: Label integrity - does label_1H match manual shift(-1)?
# ===================================================================
print("="*65)
print("CHECK 1: Label integrity - does label_1H match shift(-1)?")
print("="*65)

df_1h_eur = pd.read_parquet(PROCESSED_DIR / 'EURUSD_1H.parquet')
close = df_1h_eur['close']
manual_label = np.log(close.shift(-1) / close)
stored = df[df['pair'] == 'EURUSD']['label_1H']
common = manual_label.index.intersection(stored.index)
diff = (manual_label.loc[common] - stored.loc[common]).abs()
max_diff = diff.max()
nonzero = (diff > 1e-10).sum()

print(f"  Max |manual - stored|: {max_diff:.2e}")
print(f"  Rows with diff > 1e-10: {nonzero:,} / {len(common):,}")
if max_diff < 1e-8:
    print("  PASS: Labels match perfectly.")
else:
    print("  FAIL: LABELS DO NOT MATCH!")
print()

# ===================================================================
# CHECK 2: Are any label columns in feature_cols? (direct leakage)
# ===================================================================
print("="*65)
print("CHECK 2: Label columns accidentally in feature_cols?")
print("="*65)

leaked_labels = [c for c in feature_cols if 'label' in c.lower()]
print(f"  Feature cols containing 'label': {leaked_labels}")
if not leaked_labels:
    print("  PASS: No label columns in features.")
else:
    print("  FAIL: LABEL LEAKAGE IN FEATURES!")
print()

# ===================================================================
# CHECK 3: Feature-label Spearman correlations (test set)
# High correlation means a feature is encoding the future
# ===================================================================
print("="*65)
print("CHECK 3: Top feature-label correlations (test set, label_1H)")
print("="*65)

from scipy.stats import spearmanr

label_col = 'label_1H'
test_clean = df_test[df_test[label_col].notna()].copy()
X_test = test_clean[feature_cols].ffill().bfill()
y_test = test_clean[label_col]

corrs = {}
for col in feature_cols:
    try:
        vals = X_test[col].values
        if np.std(vals) > 0:
            r, _ = spearmanr(vals, y_test.values, nan_policy='omit')
            corrs[col] = abs(r)
    except Exception:
        pass

corr_series = pd.Series(corrs).sort_values(ascending=False)
print(f"  Top 20 |Spearman r| with label_1H:")
print(f"  {'Feature':<38} {'|r|':>6}")
print(f"  {'-'*46}")
for feat, r in corr_series.head(20).items():
    flag = " <-- SUSPICIOUS" if r > 0.10 else ""
    print(f"  {feat:<38} {r:.4f}{flag}")
print()

# ===================================================================
# CHECK 4: Train/test index overlap
# ===================================================================
print("="*65)
print("CHECK 4: Train/test timestamp overlap")
print("="*65)

train_idx = set(df_train.index)
test_idx  = set(df_test.index)
overlap = train_idx & test_idx
print(f"  Overlap: {len(overlap)} timestamps")
print(f"  {'PASS: No overlap.' if not overlap else 'FAIL: TIMESTAMPS OVERLAP!'}")
print()

# ===================================================================
# CHECK 5: Naive baselines
# ===================================================================
print("="*65)
print("CHECK 5: Naive baselines on test set")
print("="*65)

for horizon in ['1H', '4H', '1D']:
    lab = f'label_{horizon}'
    sub = df_test[df_test[lab].notna()][lab]
    pct_up   = (sub > 0).mean()
    best_naive = max(pct_up, 1 - pct_up)
    print(f"  {horizon}: {len(sub):,} rows | % up={pct_up:.1%} | best naive={best_naive:.1%}")
print()

# ===================================================================
# CHECK 6: Model predictions distribution
# If 88% are marked high-confidence and 92% of those are correct,
# but the model outputs probabilities in a narrow band,
# that would be suspicious (model just outputs 0.80-0.85 for everything)
# ===================================================================
print("="*65)
print("CHECK 6: Model output probability distribution (test set)")
print("="*65)

# Load models
models = {}
for horizon in HORIZONS:
    for q, q_name in zip(QUANTILES, QUANTILE_NAMES):
        path = MODELS_DIR / f'model_{horizon}_Q{int(q*100)}.joblib'
        bundle = joblib.load(path)
        models[(horizon, q_name)] = bundle['model']
bundle0 = joblib.load(MODELS_DIR / 'model_1H_Q50.joblib')
model_feature_cols = bundle0['feature_cols']

def derive_p_down(q_vals):
    qs   = np.array(QUANTILES)
    vals = np.sort(q_vals)
    if vals[0] <= 0 <= vals[-1]:
        return float(np.interp(0, vals, qs))
    elif vals[-1] < 0:
        slope = (qs[-1] - qs[-2]) / (vals[-1] - vals[-2] + 1e-10)
        return float(np.clip(qs[-1] + slope * (0 - vals[-1]), 0.90, 0.999))
    else:
        slope = (qs[1] - qs[0]) / (vals[1] - vals[0] + 1e-10)
        return float(np.clip(qs[0] + slope * (0 - vals[0]), 0.001, 0.10))

print("  Running inference on test set (sample of 5000 rows per horizon)...")
SPREAD_THRESHOLDS = {'1H': 0.0008, '4H': 0.0020, '1D': 0.0060}

for horizon in HORIZONS:
    lab = f'label_{horizon}'
    sub = df_test[df_test[lab].notna()].copy()
    sub = sub.sample(min(5000, len(sub)), random_state=42)
    X = sub[model_feature_cols].ffill().bfill()
    y = sub[lab]

    probs = []
    for i in range(len(X)):
        row = X.iloc[[i]]
        q_vals_raw = np.array([
            float(models[(horizon, q_name)].predict(row)[0])
            for q_name in QUANTILE_NAMES
        ])
        q_vals = np.sort(q_vals_raw)
        spread = q_vals[-1] - q_vals[0]
        p_down = derive_p_down(q_vals)
        p_up   = 1.0 - p_down
        low_conviction = spread < SPREAD_THRESHOLDS[horizon]
        if low_conviction:
            p_down = 0.5 + (p_down - 0.5) * 0.5
            p_up   = 1.0 - p_down
        prob = max(p_down, p_up)
        direction = 'bearish' if p_down > p_up else 'bullish'
        actual_dir = 'bullish' if float(y.iloc[i]) > 0 else 'bearish'
        probs.append({
            'prob': prob,
            'direction': direction,
            'correct': int(direction == actual_dir),
            'low_conviction': low_conviction,
        })

    probs_df = pd.DataFrame(probs)

    # Distribution of probabilities
    bins = [0.5, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.0]
    print(f"\n  {horizon} probability distribution (n={len(probs_df):,}):")
    print(f"    {'Bucket':<12} {'N':>6} {'%total':>8} {'Accuracy':>10}")
    print(f"    {'-'*40}")
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (probs_df['prob'] >= lo) & (probs_df['prob'] < hi)
        n = mask.sum()
        if n == 0:
            continue
        pct = n / len(probs_df)
        acc = probs_df.loc[mask, 'correct'].mean()
        print(f"    {lo:.0%}-{hi:.0%}     {n:>6}   {pct:>7.1%}   {acc:>9.1%}")

    overall_acc = probs_df['correct'].mean()
    pct_high = (probs_df['prob'] >= 0.70).mean()
    low_conv_pct = probs_df['low_conviction'].mean()
    print(f"    Overall accuracy: {overall_acc:.1%} | High-conf (>=70%): {pct_high:.1%} | Low-conviction: {low_conv_pct:.1%}")

print()

# ===================================================================
# CHECK 7: Does the model perform better on train set vs test set?
# If train accuracy is 95%+ but test is 73%, that suggests overfitting
# (not leakage, but important to know)
# ===================================================================
print("="*65)
print("CHECK 7: Train vs test accuracy (Q50 sign accuracy, sample)")
print("="*65)

for horizon in ['1H', '4H', '1D']:
    lab = f'label_{horizon}'
    model = models[(horizon, 'Q50')]

    # Train sample
    tr = df_train[df_train[lab].notna()].sample(min(3000, len(df_train)), random_state=42)
    X_tr = tr[model_feature_cols].ffill().bfill()
    y_tr = tr[lab]
    pred_tr = model.predict(X_tr)
    acc_tr = ((pred_tr > 0) == (y_tr.values > 0)).mean()

    # Test sample
    te = df_test[df_test[lab].notna()].sample(min(3000, len(df_test)), random_state=42)
    X_te = te[model_feature_cols].ffill().bfill()
    y_te = te[lab]
    pred_te = model.predict(X_te)
    acc_te = ((pred_te > 0) == (y_te.values > 0)).mean()

    gap = acc_tr - acc_te
    flag = " <-- OVERFIT" if gap > 0.15 else (" <-- SUSPICIOUS" if gap > 0.08 else "")
    print(f"  {horizon} Q50 sign accuracy: train={acc_tr:.1%}  test={acc_te:.1%}  gap={gap:+.1%}{flag}")

print()

# ===================================================================
# CHECK 8: Does accuracy vary by pair? Uniform high accuracy
# across ALL pairs and horizons is itself suspicious
# ===================================================================
print("="*65)
print("CHECK 8: Accuracy breakdown by pair × horizon (Q50 sign, test set)")
print("="*65)

PAIRS = ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD', 'USDCAD', 'NZDUSD']

print(f"  {'Pair':<10}", end='')
for h in ['1H', '4H', '1D']:
    print(f"  {h:>8}", end='')
print()
print(f"  {'-'*38}")

for pair in PAIRS:
    print(f"  {pair:<10}", end='')
    for horizon in ['1H', '4H', '1D']:
        lab = f'label_{horizon}'
        sub = df_test[(df_test['pair'] == pair) & df_test[lab].notna()]
        if len(sub) < 10:
            print(f"  {'N/A':>8}", end='')
            continue
        X = sub[model_feature_cols].ffill().bfill()
        y = sub[lab]
        pred = models[(horizon, 'Q50')].predict(X)
        acc = ((pred > 0) == (y.values > 0)).mean()
        print(f"  {acc:>7.1%}", end='')
    print()

print()

print("="*65)
print("DIAGNOSTICS COMPLETE")
print("="*65)
