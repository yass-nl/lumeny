import json
from pathlib import Path

with open('notebooks_9/02_model_training.ipynb') as f:
    nb = json.load(f)

def md(src): return {"cell_type": "markdown", "metadata": {}, "source": src, "id": "md"}
def code(src): return {"cell_type": "code", "metadata": {}, "source": src, "outputs": [], "execution_count": None, "id": "code"}

nb['cells'][0] = md("# LumenY 9 — Episode Low Detector\n\n**Setup**: Price is below both MA50 and MA200 (below-both episode).\n\n**Target**: Is the current bar's low the lowest low of the entire episode?\n\nIf yes → enter long, ride the full swing (~70-120p) back up through both MAs.\n\n**Features**: features_10 (geometric + MA context)\n\n**Model**: LightGBM binary classifier")

nb['cells'][1] = code("""\
import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
import joblib
import warnings
import gc
warnings.filterwarnings('ignore')

from pathlib import Path
from sklearn.metrics import roc_auc_score, log_loss

FEATURES_10_DIR = Path('../backend/data/features_10')
PROCESSED_DIR   = Path('../backend/data/processed')
MODELS_DIR      = Path('../backend/models_10/ma_cross')
MODELS_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_END = '2024-06-30'
SPREAD_PIPS = 1.5

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
print('Target: is this bar the episode low? (lowest low of the entire below-both episode)')
print(f'Training cutoff: {TRAIN_END}')
""")

new_cells = [

md("## 1. Compute Episode Low Label\n\nFor each below-both episode, find the bar with the lowest low. That bar gets label=1, all others in the episode get label=0."),

code("""\
print('Computing episode low labels...')
label_parts = []

for pair in PAIRS:
    pip  = PIP_SIZE[pair]
    ohlc = pd.read_parquet(PROCESSED_DIR / f'{pair}_1H.parquet')[['close', 'high', 'low']]
    ohlc.index = pd.to_datetime(ohlc.index)

    c     = ohlc['close'].values
    lo    = ohlc['low'].values
    ma50  = pd.Series(c).rolling(50).mean().values
    ma200 = pd.Series(c).rolling(200).mean().values
    n     = len(c)

    both_below = (c < ma50) & (c < ma200)
    label      = np.full(n, np.nan)

    i = 0
    while i < n:
        if np.isnan(ma50[i]) or np.isnan(ma200[i]) or not both_below[i]:
            i += 1
            continue

        # Start of a below-both episode
        ep_start = i
        while i < n and (np.isnan(ma50[i]) or np.isnan(ma200[i]) or both_below[i]
                         or (not both_below[i] and not ((c[i] > ma50[i]) and (c[i] > ma200[i])))):
            # continue episode while below both or between MAs
            if np.isnan(ma50[i]) or np.isnan(ma200[i]):
                i += 1
                break
            if (c[i] > ma50[i]) and (c[i] > ma200[i]):
                break  # crossed above both = episode ends
            i += 1
        ep_end = i  # exclusive

        # Find bar with lowest low in this episode
        ep_lows = lo[ep_start:ep_end]
        if len(ep_lows) == 0:
            continue
        min_idx = ep_start + int(np.argmin(ep_lows))

        # Label all bars in episode: 1 for the low bar, 0 for the rest
        for j in range(ep_start, ep_end):
            if np.isnan(ma50[j]) or np.isnan(ma200[j]):
                continue
            label[j] = 1.0 if j == min_idx else 0.0

    tmp = pd.DataFrame({'label': label, 'pair': pair}, index=ohlc.index)
    label_parts.append(tmp)

df_labels = pd.concat(label_parts).sort_index()
pop = df_labels['label'].dropna()
print(f'Labeled bars: {len(pop):,}')
print(f'Episode lows (label=1): {int(pop.sum()):,}  ({pop.mean():.4f} = 1 low per {1/pop.mean():.0f} bars)')
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

df_f = df_feat.reset_index()
df_l = df_labels.reset_index()
ts_col = df_f.columns[0]
df = pd.merge(df_f, df_l, on=[ts_col, 'pair'], how='inner')
df = df.set_index(ts_col).sort_index()
del df_feat, df_labels, df_f, df_l; gc.collect()

# Only bars inside episodes (label is not NaN)
df_sig = df[df['label'].notna()].copy()
print(f'Episode bars: {len(df_sig):,}  positive rate={df_sig["label"].mean():.4f}')

drop_cols    = ['label', 'pair', 'both_below', 'both_above', 'label_long', 'label_short']
feature_cols = [c for c in df_sig.columns if c not in drop_cols]
print(f'Feature cols: {len(feature_cols)}')
"""),

md("## 3. Train LightGBM"),

code("""\
df_train = df_sig[df_sig.index <= TRAIN_END].copy()
df_test  = df_sig[df_sig.index >  TRAIN_END].copy()
print(f'Train: {len(df_train):,}  pos_rate={df_train["label"].mean():.4f}')
print(f'Test:  {len(df_test):,}   pos_rate={df_test["label"].mean():.4f}')

X_train = df_train[feature_cols].ffill().fillna(0)
y_train = df_train['label']
X_test  = df_test[feature_cols].ffill().fillna(0)
y_test  = df_test['label']

# Class imbalance: ~1 positive per episode length bars
pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
print(f'Positive weight (scale_pos_weight): {pos_weight:.1f}')

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
    'scale_pos_weight':  pos_weight,
    'random_state':      42,
    'n_jobs':            -1,
    'verbose':           -1,
    'device':            'gpu',
}

n = len(X_train)
n_splits, test_size = 5, int(n * 0.1)
cv_aucs, best_iters = [], []

for fold in range(n_splits):
    test_start = int(n * 0.5) + fold * (int(n * 0.5) // n_splits)
    test_end   = test_start + test_size
    if test_end > n: break
    tr_idx  = list(range(0, test_start))
    val_idx = list(range(test_start, test_end))

    m = lgb.LGBMClassifier(**params)
    m.fit(X_train.iloc[tr_idx], y_train.iloc[tr_idx],
          eval_set=[(X_train.iloc[val_idx], y_train.iloc[val_idx])],
          callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])

    preds = m.predict_proba(X_train.iloc[val_idx])[:, 1]
    auc   = roc_auc_score(y_train.iloc[val_idx], preds)
    cv_aucs.append(auc)
    best_iters.append(m.best_iteration_)
    print(f'  Fold {fold+1}: AUC={auc:.4f}  iters={m.best_iteration_}')
    del m; gc.collect()

print(f'\\nCV AUC: {np.mean(cv_aucs):.4f} +/- {np.std(cv_aucs):.4f}')

avg_iter = max(50, int(np.mean(best_iters)))
final_model = lgb.LGBMClassifier(**{**params, 'n_estimators': avg_iter})
final_model.fit(X_train, y_train)

joblib.dump({
    'model':        final_model,
    'feature_cols': feature_cols,
    'train_end':    TRAIN_END,
    'cv_auc':       np.mean(cv_aucs),
}, MODELS_DIR / 'model_ep_low.joblib')
print(f'Model saved -> {MODELS_DIR}/model_ep_low.joblib')
del final_model; gc.collect()
"""),

md("## 4. Test Set Evaluation"),

code("""\
bundle = joblib.load(MODELS_DIR / 'model_ep_low.joblib')
proba  = bundle['model'].predict_proba(X_test)[:, 1]
auc    = roc_auc_score(y_test, proba)
print(f'Test AUC:      {auc:.4f}')
print(f'Positive rate: {y_test.mean():.4f}')
print(f'Test rows:     {len(y_test):,}')

results = pd.DataFrame({
    'actual': y_test.values,
    'proba':  proba,
    'pair':   df_test['pair'].values,
}, index=y_test.index)

base = y_test.mean()
print(f'\\nPrecision at probability thresholds (base={base:.4f}):')
print(f'{"Thresh":>8} {"N":>8} {"% kept":>7} {"Precision":>10} {"Recall":>8} {"Lift":>7}')
print('-' * 55)
for t in [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5]:
    mask = results['proba'] >= t
    if mask.sum() < 10: continue
    s      = results[mask]
    prec   = s['actual'].mean()
    recall = s['actual'].sum() / results['actual'].sum()
    lift   = prec / base
    print(f'{t:>8.2f} {mask.sum():>8,} {mask.mean():>7.1%} {prec:>10.4f} {recall:>8.3f} {lift:>7.2f}x')
"""),

md("## 5. Feature Importance"),

code("""\
bundle     = joblib.load(MODELS_DIR / 'model_ep_low.joblib')
importance = pd.Series(bundle['model'].feature_importances_, index=feature_cols)
importance = importance.sort_values(ascending=True).tail(30)

fig, ax = plt.subplots(figsize=(10, 10))
fig.patch.set_facecolor('#080c14')
ax.set_facecolor('#080c14')
ax.barh(importance.index, importance.values, color='#4fc3f7', alpha=0.8)
ax.tick_params(colors='white', labelsize=8)
ax.set_title('Feature Importance — Episode Low Detector', color='white', fontsize=12)
for spine in ax.spines.values(): spine.set_edgecolor('#1a2332')
plt.tight_layout()
plt.show()
"""),

md("## 6. EV Simulation\n\nFor bars flagged as episode low (proba >= threshold): enter long at close, TP = first bar where price crosses above both MAs, timeout = close at T+120H, spread = 1.5p."),

code("""\
print('EV simulation: enter at close, TP=first close above both MAs, timeout=T+120H')
print(f'{"Thresh":>8} {"N":>7} {"WR":>7} {"Avg win":>9} {"Avg loss":>9} {"EV":>8} {"Sharpe":>8}')
print('-' * 65)

for t in [0.01, 0.02, 0.05, 0.1, 0.2]:
    mask = results['proba'] >= t
    if mask.sum() < 10: continue
    sigs = results[mask]

    pnls = []
    for pair in PAIRS:
        pip  = PIP_SIZE[pair]
        ohlc = pd.read_parquet(PROCESSED_DIR / f'{pair}_1H.parquet')[['close','high','low']]
        ohlc.index = pd.to_datetime(ohlc.index)
        c_arr  = ohlc['close'].values
        ma50_s = pd.Series(c_arr).rolling(50).mean()
        ma200_s= pd.Series(c_arr).rolling(200).mean()

        filt = sigs[sigs['pair'] == pair]
        for ts, row in filt.iterrows():
            if ts not in ohlc.index: continue
            iloc  = ohlc.index.get_loc(ts)
            entry = ohlc['close'].iloc[iloc]

            pnl = None
            for k in range(1, 121):  # up to 120H
                if iloc + k >= len(ohlc): break
                c_k    = ohlc['close'].iloc[iloc + k]
                ma50_k = ma50_s.iloc[iloc + k]
                ma200_k= ma200_s.iloc[iloc + k]
                if pd.notna(ma50_k) and pd.notna(ma200_k) and c_k > ma50_k and c_k > ma200_k:
                    pnl = (c_k - entry) / pip - SPREAD_PIPS
                    break

            if pnl is None:
                c_end = ohlc['close'].iloc[min(iloc + 120, len(ohlc)-1)]
                pnl   = (c_end - entry) / pip - SPREAD_PIPS
            pnls.append(pnl)

    pnls   = np.array(pnls)
    wins   = pnls[pnls > 0]
    losses = pnls[pnls <= 0]
    ev     = pnls.mean()
    sharpe = ev / pnls.std() * np.sqrt(252) if pnls.std() > 0 else 0
    print(f'{t:>8.2f} {len(pnls):>7,} {(pnls>0).mean():>7.1%} '
          f'{wins.mean() if len(wins) else 0:>9.1f} '
          f'{losses.mean() if len(losses) else 0:>9.1f} '
          f'{ev:>8.2f} {sharpe:>8.3f}')
"""),

]

nb['cells'].extend(new_cells)

with open('notebooks_9/02_model_training.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)

print(f'Done. Total cells: {len(nb["cells"])}')
