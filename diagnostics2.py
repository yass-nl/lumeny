"""
DEEP DIVE: What's special about rows that get 70-75% confidence?
And why does accuracy jump from 73% at 65-70% to 89% at 70-75%?
This non-monotonic pattern is the red flag.
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

df = pd.read_parquet(FEATURES_DIR / 'all_pairs_features_labels.parquet')
df_test = df[df.index > '2024-06-30'].copy()

ALL_HORIZONS   = ['1H', '4H', '1D', '7D']
label_cols     = [f'label_{h}' for h in ALL_HORIZONS]
feature_cols   = [c for c in df.columns if c not in label_cols + ['pair']]

HORIZONS       = ['1H', '4H', '1D']
QUANTILES      = [0.10, 0.25, 0.50, 0.75, 0.90]
QUANTILE_NAMES = ['Q10', 'Q25', 'Q50', 'Q75', 'Q90']
SPREAD_THRESHOLDS = {'1H': 0.0008, '4H': 0.0020, '1D': 0.0060}

models = {}
for horizon in HORIZONS:
    for q, q_name in zip(QUANTILES, QUANTILE_NAMES):
        path = MODELS_DIR / f'model_{horizon}_Q{int(q*100)}.joblib'
        models[(horizon, q_name)] = joblib.load(path)['model']
model_feature_cols = joblib.load(MODELS_DIR / 'model_1H_Q50.joblib')['feature_cols']

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

# ===================================================================
# DEEP CHECK A: For 1H, investigate the non-monotonic accuracy jump
# Why is 70-75% accuracy at 89% but 75-80% only at 79%?
# Look at the Q50 value (expected move) and label distribution in each bucket
# ===================================================================
print("="*65)
print("DEEP CHECK A: Non-monotonic accuracy - what differs between buckets?")
print("="*65)

horizon = '1H'
lab = f'label_{horizon}'
sub = df_test[df_test[lab].notna()].sample(10000, random_state=42)
X = sub[model_feature_cols].ffill().bfill()
y = sub[lab]

rows = []
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
    rows.append({
        'prob': prob,
        'direction': direction,
        'p_down': p_down,
        'correct': int(direction == actual_dir),
        'low_conviction': int(low_conviction),
        'spread': spread,
        'q10': q_vals[0], 'q25': q_vals[1], 'q50': q_vals[2],
        'q75': q_vals[3], 'q90': q_vals[4],
        'actual_label': float(y.iloc[i]),
    })

res = pd.DataFrame(rows)

buckets = [(0.50,0.55),(0.55,0.60),(0.60,0.65),(0.65,0.70),(0.70,0.75),(0.75,0.80),(0.80,0.85),(0.85,0.90),(0.90,1.0)]
print(f"  {'Bucket':<12} {'N':>5} {'Acc':>7} {'Low%':>7} {'Spread':>10} {'Q50(bps)':>10} {'|Label|(bps)':>14}")
print(f"  {'-'*70}")
for lo, hi in buckets:
    mask = (res['prob'] >= lo) & (res['prob'] < hi)
    s = res[mask]
    if len(s) < 5:
        continue
    avg_spread = s['spread'].mean() * 10000
    avg_q50 = s['q50'].abs().mean() * 10000
    avg_actual = s['actual_label'].abs().mean() * 10000
    low_pct = s['low_conviction'].mean()
    print(f"  {lo:.0%}-{hi:.0%}     {len(s):>5}  {s['correct'].mean():>6.1%}  {low_pct:>6.1%}  {avg_spread:>9.2f}  {avg_q50:>9.2f}  {avg_actual:>13.2f}")

print()

# ===================================================================
# DEEP CHECK B: The low_conviction flag — when it fires, accuracy should
# be ~50%, but the 70-75% bucket seems to have models pushing high prob
# after the 0.5x compression. What if the compression is being gamed?
# Check: among rows with low_conviction=False, what is the prob distribution?
# ===================================================================
print("="*65)
print("DEEP CHECK B: Accuracy WITH vs WITHOUT low_conviction flag")
print("="*65)

for horizon in HORIZONS:
    lab = f'label_{horizon}'
    sub = df_test[df_test[lab].notna()].sample(min(8000, len(df_test)), random_state=42)
    X = sub[model_feature_cols].ffill().bfill()
    y = sub[lab]

    rows = []
    for i in range(len(X)):
        row = X.iloc[[i]]
        q_vals_raw = np.array([float(models[(horizon, q_name)].predict(row)[0]) for q_name in QUANTILE_NAMES])
        q_vals = np.sort(q_vals_raw)
        spread = q_vals[-1] - q_vals[0]
        p_down = derive_p_down(q_vals)
        p_up   = 1.0 - p_down
        low_conviction = spread < SPREAD_THRESHOLDS[horizon]
        p_down_adj = 0.5 + (p_down - 0.5) * 0.5 if low_conviction else p_down
        p_up_adj   = 1.0 - p_down_adj
        prob = max(p_down_adj, p_up_adj)
        direction = 'bearish' if p_down_adj > p_up_adj else 'bullish'
        actual_dir = 'bullish' if float(y.iloc[i]) > 0 else 'bearish'
        rows.append({
            'prob': prob,
            'correct': int(direction == actual_dir),
            'low_conviction': int(low_conviction),
        })

    r = pd.DataFrame(rows)
    normal = r[r['low_conviction'] == 0]
    low    = r[r['low_conviction'] == 1]
    print(f"\n  {horizon}:")
    print(f"    Normal (n={len(normal):,}):  acc={normal['correct'].mean():.1%}  | high-conf(>=70%): {(normal['prob']>=0.70).mean():.1%}")
    print(f"    Low conv (n={len(low):,}):   acc={low['correct'].mean():.1%}  | high-conf(>=70%): {(low['prob']>=0.70).mean():.1%}")
    if len(low) > 0 and (low['prob'] >= 0.70).mean() > 0.05:
        print(f"    WARNING: {(low['prob']>=0.70).mean():.1%} of LOW-CONVICTION rows still output prob>=70%! That's wrong.")

print()

# ===================================================================
# DEEP CHECK C: Timestamp alignment check
# Is the feature for row T using data from T or T+1?
# Compare log_ret_1_1H at row T against actual return T->T+1
# ===================================================================
print("="*65)
print("DEEP CHECK C: Feature timestamp alignment")
print("  Is log_ret_1_1H[T] = log(close[T]/close[T-1]) or log(close[T+1]/close[T])?")
print("="*65)

for pair in ['EURUSD', 'GBPUSD']:
    df_1h = pd.read_parquet(PROCESSED_DIR / f'{pair}_1H.parquet')
    close = df_1h['close']

    stored_feat = df[df['pair'] == pair]['log_ret_1_1H']
    correct_past = np.log(close / close.shift(1))       # past return (no lookahead)
    future_ret   = np.log(close.shift(-1) / close)       # future return (leakage!)

    common = stored_feat.index.intersection(correct_past.index)
    diff_past   = (stored_feat.loc[common] - correct_past.loc[common]).abs()
    diff_future = (stored_feat.loc[common] - future_ret.loc[common]).abs()

    mean_diff_past   = diff_past.mean()
    mean_diff_future = diff_future.mean()

    print(f"\n  {pair}:")
    print(f"    mean|feat - past_ret|:   {mean_diff_past:.2e}")
    print(f"    mean|feat - future_ret|: {mean_diff_future:.2e}")
    if mean_diff_past < mean_diff_future:
        print(f"    PASS: Feature matches past return (no lookahead)")
    else:
        print(f"    FAIL: Feature matches FUTURE return (LOOKAHEAD DETECTED!)")

print()

# ===================================================================
# DEEP CHECK D: The 4H feature alignment check
# The highest correlated feature is log_ret_1_4H (r=0.31 with label_1H)
# Is this legitimate (momentum) or is it using future 4H data?
# A 4H candle closes at 04:00, 08:00, 12:00, 16:00, 20:00, 00:00
# At 1H bar for 15:00, what 4H data is available? The 12:00-16:00 bar hasn't closed yet.
# So the "last completed 4H bar" is the 08:00-12:00 bar.
# BUT: if reindex(method='ffill') is used, the 12:00 4H bar is forward-filled to 13:00, 14:00, 15:00...
# When does the 4H bar for [12:00-16:00) become available? At 16:00.
# So at 15:00, we should only have the 4H bar that closed at 12:00.
# Check: is log_ret_1_4H[T=15:00] = log(close_4H[12:00] / close_4H[08:00])?
# Or is it = log(close_4H[16:00] / close_4H[12:00])? (future!)
# ===================================================================
print("="*65)
print("DEEP CHECK D: 4H feature alignment - is current or NEXT 4H bar used?")
print("="*65)

pair = 'EURUSD'
df_4h = pd.read_parquet(PROCESSED_DIR / f'{pair}_4H.parquet')
close_4h = df_4h['close']

# Compute log_ret_1 for 4H — this is log(close[T] / close[T-1]) in 4H space
log_ret_1_4h = np.log(close_4h / close_4h.shift(1))

# Get stored feature
stored_4h_feat = df[df['pair'] == pair]['log_ret_1_4H']

# How it SHOULD look after ffill to 1H:
# at 1H bar 15:00 UTC — the last closed 4H bar is 08:00 (closed at 12:00)
# So log_ret_1_4H[15:00] should = log(close_4H[12:00] / close_4H[08:00])
# Wait — actually 4H bars open at 00, 04, 08, 12, 16, 20
# At 1H bar 15:00, which 4H bar has closed? The one that started at 08:00 and closed at 12:00.
# ffill: log_ret_1_4H is computed on 4H index, then ffilled to 1H index.
# Value at 4H[12:00] = log(close_4H[12:00] / close_4H[08:00])
# This gets ffilled to 1H bars 12:00, 13:00, 14:00, 15:00 -> OK, that's past data.
# Value at 4H[16:00] = log(close_4H[16:00] / close_4H[12:00]) -> gets ffilled to 16:00, 17:00...

# Verify by checking one specific time point
print(f"\n  4H bars (EURUSD, first available):")
print(f"  {df_4h.index[:5].tolist()}")
print(f"\n  Stored 4H log_ret_1 values (1H index sample around a 4H bar close):")

# Find a 4H bar close time and look at surrounding 1H bars
sample_4h_close = df_4h.index[100]  # some middle 4H bar
surrounding = [sample_4h_close - pd.Timedelta(hours=i) for i in range(5, -6, -1)]
print(f"\n  Around 4H bar close at {sample_4h_close}:")
for ts in surrounding:
    val = stored_4h_feat.get(ts, float('nan')) if ts in stored_4h_feat.index else float('nan')
    is_4h_close = ts in log_ret_1_4h.index
    expected = log_ret_1_4h.get(ts, float('nan')) if ts in log_ret_1_4h.index else float('nan')
    print(f"    {ts}  log_ret_1_4H={val:.6f}  4H_bar_close={is_4h_close}  4H_raw={expected:.6f}")

# The key test: at a 1H bar BEFORE the 4H close, does log_ret_1_4H
# match the CURRENT (open) 4H bar or the PREVIOUS (closed) 4H bar?
print(f"\n  Interpretation:")
print(f"  If log_ret_1_4H is the SAME for 1H bars before and after a 4H close,")
print(f"  then ffill is working correctly (using last CLOSED 4H bar).")
print(f"  If it CHANGES at each 1H bar within the 4H period, there's a problem.")

# Check: correlation of log_ret_1_4H with label_1H
from scipy.stats import spearmanr
test_eur = df_test[df_test['pair'] == 'EURUSD'][['log_ret_1_4H', 'label_1H']].dropna()
r, p = spearmanr(test_eur['log_ret_1_4H'], test_eur['label_1H'])
print(f"\n  Spearman(log_ret_1_4H, label_1H) for EURUSD test set: r={r:.4f}, p={p:.2e}")
print(f"  This is the HIGHEST correlated feature (r=0.31 overall).")
print(f"  Is this legitimate momentum or leakage?")

# The 4H log_ret_1 at 1H timestamp T contains the PREVIOUS completed 4H bar's return
# Correlation of 0.31 with next-1H return is suspicious but let's check:
# Does log_ret_1_4H predict label_1H BETTER on the 1st bar after a 4H close vs later bars?
df_eur_full = df[df['pair'] == 'EURUSD'].copy()
df_4h_eur = pd.read_parquet(PROCESSED_DIR / f'EURUSD_4H.parquet')
four_hour_close_times = set(df_4h_eur.index.tolist())
df_eur_full['is_4h_close_bar'] = df_eur_full.index.isin(four_hour_close_times)

# Bar 1 after 4H close = bar where 4H data just refreshed
test_eur2 = df_eur_full[df_eur_full.index > '2024-06-30'][['log_ret_1_4H','label_1H','is_4h_close_bar']].dropna()
just_after = test_eur2[test_eur2['is_4h_close_bar']]
other = test_eur2[~test_eur2['is_4h_close_bar']]

if len(just_after) > 50 and len(other) > 50:
    r_after, _ = spearmanr(just_after['log_ret_1_4H'], just_after['label_1H'])
    r_other, _ = spearmanr(other['log_ret_1_4H'], other['label_1H'])
    print(f"\n  Correlation split by 4H bar boundary:")
    print(f"    At 4H close bars (n={len(just_after)}):   r={r_after:.4f}")
    print(f"    Other 1H bars   (n={len(other)}):   r={r_other:.4f}")
    if abs(r_after) > abs(r_other) * 1.5:
        print(f"    WARNING: Much higher correlation right at 4H close — possible lookahead in 4H feature!")
    else:
        print(f"    Similar correlation across bar types — likely legitimate momentum signal.")

print()

# ===================================================================
# DEEP CHECK E: The real question — is the model's edge real?
# Randomly permute the features (break any signal) and measure accuracy
# If accuracy stays high, the model is using something degenerate
# ===================================================================
print("="*65)
print("DEEP CHECK E: Permutation test (shuffle features, does accuracy collapse?)")
print("="*65)

horizon = '1H'
lab = f'label_{horizon}'
sub = df_test[df_test[lab].notna()].sample(2000, random_state=42)
X_real = sub[model_feature_cols].ffill().bfill().values
y_real = (sub[lab].values > 0).astype(int)

# Real accuracy
q50_model = models[(horizon, 'Q50')]
preds_real = q50_model.predict(X_real)
acc_real = ((preds_real > 0) == y_real).mean()

# Permuted accuracy (shuffle rows completely)
np.random.seed(42)
X_perm = X_real.copy()
np.random.shuffle(X_perm)
preds_perm = q50_model.predict(X_perm)
acc_perm = ((preds_perm > 0) == y_real).mean()

# Column-wise shuffle (destroy structure but keep marginals)
X_col_perm = X_real.copy()
for j in range(X_col_perm.shape[1]):
    np.random.shuffle(X_col_perm[:, j])
preds_col_perm = q50_model.predict(X_col_perm)
acc_col_perm = ((preds_col_perm > 0) == y_real).mean()

print(f"\n  {horizon} Q50 sign accuracy:")
print(f"    Real features:           {acc_real:.1%}")
print(f"    Row-shuffled features:   {acc_perm:.1%}  (should be ~50%)")
print(f"    Col-shuffled features:   {acc_col_perm:.1%}  (should be ~50%)")

if acc_perm > 0.55 or acc_col_perm > 0.55:
    print(f"    WARNING: Accuracy stays high even with shuffled features!")
    print(f"    This suggests the model has a structural bias (e.g., always predicts one direction).")
else:
    print(f"    Accuracy collapses with shuffled features — model is using the features legitimately.")

print()

print("="*65)
print("DEEP DIAGNOSTICS COMPLETE")
print("="*65)
