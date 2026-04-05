import json
from pathlib import Path

nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11.0"}
    },
    "cells": []
}

def md(src): return {"cell_type": "markdown", "metadata": {}, "source": src, "id": "md"}
def code(src): return {"cell_type": "code", "metadata": {}, "source": src, "outputs": [], "execution_count": None, "id": "code"}

cells = [

md("# LumenY 9 — Binary Classifier: Will Price Reach MA200 Within 72H?\n\n**Setup**: Price is currently below both MA50 and MA200.\n\n**Target**: Will price close above MA200 within the next 72 bars (72H on 1H data)?\n\n**Features**: features_10 (geometric + MA context)\n\n**Model**: LightGBM binary classifier"),

code("""\
import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
import joblib
import warnings
import gc
warnings.filterwarnings('ignore')

from pathlib import Path
from sklearn.metrics import roc_auc_score, log_loss, classification_report

FEATURES_10_DIR = Path('../backend/data/features_10')
PROCESSED_DIR   = Path('../backend/data/processed')
MODELS_DIR      = Path('../backend/models_9/ma_cross')
MODELS_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_END = '2024-06-30'
MAX_BARS  = 72   # 72H horizon

PAIRS = [
    'EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD', 'USDCAD', 'NZDUSD',
    'EURJPY', 'GBPJPY', 'EURGBP', 'EURAUD', 'AUDJPY', 'CADJPY', 'CHFJPY', 'AUDNZD',
]
PIP_SIZE = {
    'EURUSD':0.0001,'GBPUSD':0.0001,'USDCHF':0.0001,'AUDUSD':0.0001,
    'USDCAD':0.0001,'NZDUSD':0.0001,'EURGBP':0.0001,'EURAUD':0.0001,'AUDNZD':0.0001,
    'USDJPY':0.01,'EURJPY':0.01,'GBPJPY':0.01,'AUDJPY':0.01,'CADJPY':0.01,'CHFJPY':0.01,
}

print('Ready.')
print(f'Target: price below both MA50+MA200 now -> will it reach MA200 within {MAX_BARS}H?')
print(f'Training cutoff: {TRAIN_END}')
"""),

md("## 1. Compute Binary Label"),

code("""\
print('Computing labels from raw OHLCV...')
label_parts = []

for pair in PAIRS:
    ohlc = pd.read_parquet(PROCESSED_DIR / f'{pair}_1H.parquet')[['close']]
    ohlc.index = pd.to_datetime(ohlc.index)

    c     = ohlc['close'].values
    ma50  = pd.Series(c).rolling(50).mean().values
    ma200 = pd.Series(c).rolling(200).mean().values
    n     = len(c)

    label_long  = np.full(n, np.nan)   # below both, will reach MA200?
    label_short = np.full(n, np.nan)   # above both, will reach MA200 (from above)?

    for i in range(n - MAX_BARS):
        if np.isnan(ma50[i]) or np.isnan(ma200[i]):
            continue

        # LONG setup: price below both MAs
        if c[i] < ma50[i] and c[i] < ma200[i]:
            target = ma200[i]
            reached = any(c[i+k] >= target for k in range(1, MAX_BARS + 1))
            label_long[i] = 1.0 if reached else 0.0

        # SHORT setup: price above both MAs
        if c[i] > ma50[i] and c[i] > ma200[i]:
            target = ma200[i]
            reached = any(c[i+k] <= target for k in range(1, MAX_BARS + 1))
            label_short[i] = 1.0 if reached else 0.0

    tmp = pd.DataFrame({
        'label_long':  label_long,
        'label_short': label_short,
        'pair': pair,
    }, index=ohlc.index)
    label_parts.append(tmp)

df_labels = pd.concat(label_parts).sort_index()
print(f'Labels computed: {len(df_labels):,} rows')

# Stats
long_pop  = df_labels['label_long'].dropna()
short_pop = df_labels['label_short'].dropna()
print(f'\\nLong population  (below both MAs): {len(long_pop):,}  hit rate={long_pop.mean():.3f}')
print(f'Short population (above both MAs): {len(short_pop):,}  hit rate={short_pop.mean():.3f}')
"""),

md("## 2. Build Dataset"),

code("""\
print('Loading features_10...')
dfs = []
for pair in PAIRS:
    feat = pd.read_parquet(FEATURES_10_DIR / f'{pair}_features.parquet')
    feat = feat.drop(columns=[c for c in feat.columns if c.startswith('label_')], errors='ignore')
    dfs.append(feat)

df_feat = pd.concat(dfs).sort_index()
print(f'features_10: {df_feat.shape}')

# Join features with labels on (timestamp + pair)
df_f = df_feat.reset_index()
df_l = df_labels.reset_index()
ts_col = df_f.columns[0]
df = pd.merge(df_f, df_l, on=[ts_col, 'pair'], how='inner')
df = df.set_index(ts_col).sort_index()
del df_feat, df_labels, df_f, df_l
print(f'Merged: {df.shape}')

# We train TWO models: one for long setups, one for short setups
# For now focus on LONG (below both MAs -> will reach MA200)
# Filter to only bars where price is below both MAs
df_long  = df[df['both_below'] == 1].dropna(subset=['label_long']).copy()
df_short = df[df['both_above'] == 1].dropna(subset=['label_short']).copy()

print(f'\\nLong dataset  (both_below=1): {len(df_long):,}  label mean={df_long["label_long"].mean():.3f}')
print(f'Short dataset (both_above=1): {len(df_short):,}  label mean={df_short["label_short"].mean():.3f}')

# Feature cols: everything except labels and pair
label_cols   = ['label_long', 'label_short']
drop_cols    = label_cols + ['pair', 'both_below', 'both_above']
feature_cols = [c for c in df_long.columns if c not in drop_cols]
print(f'Feature cols: {len(feature_cols)}')
"""),

md("## 3. Walk-Forward Splits"),

code("""\
def walk_forward_splits(n, n_splits=5, test_ratio=0.1):
    test_size = int(n * test_ratio)
    splits = []
    for i in range(n_splits):
        test_start = int(n * 0.5) + i * (int(n * 0.5) // n_splits)
        test_end   = test_start + test_size
        if test_end > n:
            break
        splits.append((list(range(0, test_start)), list(range(test_start, test_end))))
    return splits

# Train/test split
df_long_train = df_long[df_long.index <= TRAIN_END]
df_long_test  = df_long[df_long.index >  TRAIN_END]
print(f'Long train: {len(df_long_train):,}  test: {len(df_long_test):,}')

splits = walk_forward_splits(len(df_long_train))
print(f'Walk-forward splits: {len(splits)}')
for i, (tr, te) in enumerate(splits):
    print(f'  Fold {i+1}: train={len(tr):,}  val={len(te):,}')
"""),

md("## 4. Train LightGBM Classifier"),

code("""\
params = {
    'objective':         'binary',
    'metric':            'auc',
    'boosting_type':     'gbdt',
    'n_estimators':      3000,
    'learning_rate':     0.02,
    'num_leaves':        64,
    'max_depth':         6,
    'min_child_samples': 50,
    'feature_fraction':  0.7,
    'bagging_fraction':  0.8,
    'bagging_freq':      5,
    'reg_alpha':         0.1,
    'reg_lambda':        0.1,
    'random_state':      42,
    'n_jobs':            -1,
    'verbose':           -1,
    'device':            'gpu',
}

X_train = df_long_train[feature_cols].ffill().fillna(0)
y_train = df_long_train['label_long']

print(f'Training samples: {len(X_train):,}')
print(f'Positive rate: {y_train.mean():.3f}')

oof_preds  = np.full(len(X_train), np.nan)
best_iters = []
cv_aucs    = []

for fold, (tr_idx, val_idx) in enumerate(splits):
    X_tr, y_tr = X_train.iloc[tr_idx], y_train.iloc[tr_idx]
    X_val, y_val = X_train.iloc[val_idx], y_train.iloc[val_idx]

    model = lgb.LGBMClassifier(**params)
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
              callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])

    preds = model.predict_proba(X_val)[:, 1]
    oof_preds[val_idx] = preds
    auc = roc_auc_score(y_val, preds)
    cv_aucs.append(auc)
    best_iters.append(model.best_iteration_)
    print(f'  Fold {fold+1}: AUC={auc:.4f}  iters={model.best_iteration_}')
    del model; gc.collect()

print(f'\\nCV AUC: {np.mean(cv_aucs):.4f} +/- {np.std(cv_aucs):.4f}')
print(f'Best iters: {best_iters}')

# Train final model
avg_iter = max(50, int(np.mean(best_iters)))
final_model = lgb.LGBMClassifier(**{**params, 'n_estimators': avg_iter})
final_model.fit(X_train, y_train)

joblib.dump({
    'model':        final_model,
    'feature_cols': feature_cols,
    'train_end':    TRAIN_END,
    'max_bars':     MAX_BARS,
    'direction':    'long',
    'cv_auc':       np.mean(cv_aucs),
    'n_iters':      avg_iter,
}, MODELS_DIR / 'model_long.joblib')
print(f'\\nModel saved -> {MODELS_DIR}/model_long.joblib')
del final_model; gc.collect()
"""),

md("## 5. Test Set Evaluation"),

code("""\
bundle = joblib.load(MODELS_DIR / 'model_long.joblib')
X_test = df_long_test[feature_cols].ffill().fillna(0)
y_test = df_long_test['label_long']

proba = bundle['model'].predict_proba(X_test)[:, 1]
auc   = roc_auc_score(y_test, proba)
ll    = log_loss(y_test, proba)

print(f'Test AUC:       {auc:.4f}')
print(f'Test log-loss:  {ll:.4f}')
print(f'Positive rate:  {y_test.mean():.3f}')
print(f'Test rows:      {len(y_test):,}')

results = pd.DataFrame({
    'actual': y_test.values,
    'proba':  proba,
    'pair':   df_long_test['pair'].values,
}, index=y_test.index)

# Precision at different probability thresholds
print(f'\\nPrecision / Recall at probability thresholds:')
print(f'{"Thresh":>8} {"N":>7} {"% kept":>7} {"Precision":>10} {"Recall":>8} {"Lift":>7}')
print('-' * 55)
base = y_test.mean()
for t in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    mask = results['proba'] >= t
    if mask.sum() < 20: continue
    s = results[mask]
    prec   = s['actual'].mean()
    recall = s['actual'].sum() / results['actual'].sum()
    lift   = prec / base
    print(f'{t:>8.1f} {mask.sum():>7,} {mask.mean():>7.1%} {prec:>10.3f} {recall:>8.3f} {lift:>7.2f}x')

# Per-pair breakdown
print(f'\\nPer-pair (all thresholds):')
print(f'{"Pair":>10} {"N":>6} {"Base rate":>10} {"AUC":>8}')
print('-' * 38)
for pair, g in results.groupby('pair'):
    if len(g) < 50 or g['actual'].nunique() < 2: continue
    pair_auc = roc_auc_score(g['actual'], g['proba'])
    print(f'{pair:>10} {len(g):>6,} {g["actual"].mean():>10.3f} {pair_auc:>8.4f}')
"""),

md("## 6. Feature Importance"),

code("""\
bundle = joblib.load(MODELS_DIR / 'model_long.joblib')
importance = pd.Series(bundle['model'].feature_importances_, index=feature_cols)
importance = importance.sort_values(ascending=True).tail(30)

fig, ax = plt.subplots(figsize=(10, 10))
fig.patch.set_facecolor('#080c14')
ax.set_facecolor('#080c14')
ax.barh(importance.index, importance.values, color='#4fc3f7', alpha=0.8)
ax.tick_params(colors='white', labelsize=8)
ax.set_title('Feature Importance — MA Cross Model (Long)', color='white', fontsize=12)
for spine in ax.spines.values(): spine.set_edgecolor('#1a2332')
plt.tight_layout()
plt.show()
"""),

md("## 7. Calibration — Probability vs Actual Hit Rate"),

code("""\
# Bin by predicted probability, show actual hit rate per bin
results['prob_bin'] = pd.cut(results['proba'], bins=10)
cal = results.groupby('prob_bin', observed=True).agg(
    n=('actual','count'),
    actual_rate=('actual','mean'),
    avg_proba=('proba','mean')
).reset_index()

print('Calibration (predicted proba vs actual hit rate):')
print(f'{"Bin":<22} {"N":>6} {"Pred":>8} {"Actual":>8} {"Diff":>8}')
print('-' * 58)
for _, row in cal.iterrows():
    diff = row.actual_rate - row.avg_proba
    print(f'{str(row.prob_bin):<22} {int(row.n):>6} {row.avg_proba:>8.3f} {row.actual_rate:>8.3f} {diff:>+8.3f}')
"""),

]

nb['cells'] = cells

out = Path('notebooks_9/02_model_training.ipynb')
with open(out, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print(f'Written: {out}')
