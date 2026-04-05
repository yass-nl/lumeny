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

md("# LumenY 9 — Meta Model: When Is the Base Model's Signal Profitable?\n\n**Idea**: The base model (AUC=0.80) predicts whether price will reach MA200 within 72H. But raw probabilities don't translate to positive EV because:\n- High-proba signals often fire when price is already close to MA200 (small move)\n- Model fires on fresh breakdowns below MA50 (bad entries)\n- Timeout exits at T+72 bleed losses\n\n**Meta model**: Train a second classifier that sees:\n- Base model probability\n- Distance to MA200 in pips\n- How long price has been below MA50 (bars_below_ma50)\n- Hour of day\n- All features_10 columns\n\n**Target**: actual PnL > 0 (TP=MA200 at entry, timeout=T+72 close, spread=1.5p)"),

code("""\
import pandas as pd
import numpy as np
import lightgbm as lgb
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

TRAIN_END    = '2024-06-30'
MAX_BARS     = 72
SPREAD_PIPS  = 1.5
PIP_SIZE = {
    'EURUSD':0.0001,'GBPUSD':0.0001,'USDCHF':0.0001,'AUDUSD':0.0001,
    'USDCAD':0.0001,'NZDUSD':0.0001,'EURGBP':0.0001,'EURAUD':0.0001,'AUDNZD':0.0001,
    'USDJPY':0.01,'EURJPY':0.01,'GBPJPY':0.01,'AUDJPY':0.01,'CADJPY':0.01,'CHFJPY':0.01,
}

PAIRS = [
    'EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD', 'USDCAD', 'NZDUSD',
    'EURJPY', 'GBPJPY', 'EURGBP', 'EURAUD', 'AUDJPY', 'CADJPY', 'CHFJPY', 'AUDNZD',
]

print('Ready.')
"""),

md("## 1. Load Base Model, Compute Probabilities on All Signals"),

code("""\
bundle = joblib.load(MODELS_DIR / 'model_long.joblib')
base_model    = bundle['model']
feature_cols  = bundle['feature_cols']
print(f'Base model loaded. Feature cols: {len(feature_cols)}')
print(f'Train end: {bundle["train_end"]}  Max bars: {bundle["max_bars"]}')

# Load all features_10 and compute base model proba
parts = []
for pair in PAIRS:
    f = pd.read_parquet(FEATURES_10_DIR / f'{pair}_features.parquet')
    f.index = pd.to_datetime(f.index)
    parts.append(f)

df_all = pd.concat(parts).sort_index()
print(f'All features: {df_all.shape}')

# Only look at bars where price is below both MAs (both_below=1)
df_sig = df_all[df_all['both_below'] == 1].copy()
print(f'Both_below signals: {len(df_sig):,}')

# Compute base model proba
X = df_sig[feature_cols].ffill().fillna(0)
df_sig['base_proba'] = base_model.predict_proba(X)[:, 1]
print(f'Base proba computed. Mean={df_sig["base_proba"].mean():.3f}')
print(f'Distribution:')
for t in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    n = (df_sig['base_proba'] >= t).sum()
    print(f'  >= {t:.1f}: {n:,} ({n/len(df_sig):.1%})')

del df_all, X; gc.collect()
"""),

md("## 2. Walk Forward: Compute Actual PnL per Signal\n\nFor each signal: TP = MA200 at entry (use high prices), timeout = close at T+72.\nPnL = (TP - entry)/pip - spread if TP hit, else (close[T+72] - entry)/pip - spread."),

code("""\
print('Computing actual PnL for each signal...')
print('(TP=MA200 at entry using highs, timeout=close[T+72], spread=1.5p)')

ohlc_cache = {}
for pair in PAIRS:
    ohlc = pd.read_parquet(PROCESSED_DIR / f'{pair}_1H.parquet')[['close','high','low']]
    ohlc.index = pd.to_datetime(ohlc.index)
    ohlc_cache[pair] = ohlc

pnl_list = []
for pair in PAIRS:
    pip  = PIP_SIZE[pair]
    ohlc = ohlc_cache[pair]
    mask = df_sig['pair'] == pair

    # Compute MA200 for entry price reference
    c    = ohlc['close']
    ma50  = c.rolling(50).mean()
    ma200 = c.rolling(200).mean()

    for ts, row in df_sig[mask].iterrows():
        if ts not in ohlc.index:
            continue
        iloc   = ohlc.index.get_loc(ts)
        entry  = ohlc['close'].iloc[iloc]

        if ts not in ma200.index or pd.isna(ma200.loc[ts]):
            continue
        target = ma200.loc[ts]

        # Walk forward MAX_BARS bars
        highs  = ohlc['high'].iloc[iloc+1 : iloc+MAX_BARS+1].values
        closes = ohlc['close'].iloc[iloc+1 : iloc+MAX_BARS+1].values

        pnl = None
        for k in range(len(highs)):
            if highs[k] >= target:
                pnl = (target - entry) / pip - SPREAD_PIPS
                break

        if pnl is None:
            # Timeout exit at T+72 close
            if len(closes) > 0:
                pnl = (closes[-1] - entry) / pip - SPREAD_PIPS
            else:
                pnl = -SPREAD_PIPS

        pnl_list.append({
            'ts':           ts,
            'pair':         pair,
            'pnl':          pnl,
            'entry':        entry,
            'target':       target,
            'pips_to_ma200': (target - entry) / pip,
        })

df_pnl = pd.DataFrame(pnl_list).set_index('ts').sort_index()
print(f'Signals with PnL: {len(df_pnl):,}')
print(f'Win rate (pnl>0): {(df_pnl["pnl"] > 0).mean():.3f}')
print(f'Mean PnL: {df_pnl["pnl"].mean():.2f}p')
print(f'Mean win: {df_pnl[df_pnl["pnl"]>0]["pnl"].mean():.2f}p')
print(f'Mean loss: {df_pnl[df_pnl["pnl"]<=0]["pnl"].mean():.2f}p')
"""),

md("## 3. Build Meta Dataset\n\nJoin base proba + features_10 + actual PnL. Target = pnl > 0."),

code("""\
# Join: bring in base_proba and all features_10 cols not already in df_pnl
cols_to_join = [c for c in df_sig.columns if c not in df_pnl.columns]
df_meta = df_pnl.join(df_sig[cols_to_join], how='inner')

# Add hour
df_meta['hour'] = df_meta.index.hour

# Deduplicate columns (safety)
df_meta = df_meta.loc[:, ~df_meta.columns.duplicated()]

print(f'Meta dataset: {df_meta.shape}')
print(f'Target (pnl>0): {(df_meta["pnl"] > 0).mean():.3f}')

# Split train/test
df_meta_train = df_meta[df_meta.index <= TRAIN_END].copy()
df_meta_test  = df_meta[df_meta.index >  TRAIN_END].copy()
print(f'Train: {len(df_meta_train):,}  Test: {len(df_meta_test):,}')
print(f'Train positive rate: {(df_meta_train["pnl"] > 0).mean():.3f}')
print(f'Test  positive rate: {(df_meta_test["pnl"] > 0).mean():.3f}')

# Meta feature cols
exclude = {'pnl', 'entry', 'target', 'pair', 'label_long', 'label_short',
           'both_below', 'both_above'}
meta_feature_cols = [c for c in df_meta.columns if c not in exclude]
print(f'Meta features: {len(meta_feature_cols)}')
print('  Includes: base_proba, pips_to_ma200, bars_below_ma50, hour, all features_10...')
"""),

md("## 4. Train Meta Model"),

code("""\
params = {
    'objective':         'binary',
    'metric':            'auc',
    'boosting_type':     'gbdt',
    'n_estimators':      2000,
    'learning_rate':     0.02,
    'num_leaves':        32,
    'max_depth':         5,
    'min_child_samples': 30,
    'feature_fraction':  0.7,
    'bagging_fraction':  0.8,
    'bagging_freq':      5,
    'reg_alpha':         0.1,
    'reg_lambda':        0.2,
    'random_state':      42,
    'n_jobs':            -1,
    'verbose':           -1,
    'device':            'gpu',
}

X_tr = df_meta_train[meta_feature_cols].ffill().fillna(0)
y_tr = (df_meta_train['pnl'] > 0).astype(int)

X_te = df_meta_test[meta_feature_cols].ffill().fillna(0)
y_te = (df_meta_test['pnl'] > 0).astype(int)

print(f'Train: {len(X_tr):,}  pos_rate={y_tr.mean():.3f}')
print(f'Test:  {len(X_te):,}  pos_rate={y_te.mean():.3f}')

# Walk-forward CV on train set
n = len(X_tr)
n_splits = 5
test_size = int(n * 0.1)
cv_aucs, best_iters = [], []

for fold in range(n_splits):
    test_start = int(n * 0.5) + fold * (int(n * 0.5) // n_splits)
    test_end   = test_start + test_size
    if test_end > n:
        break
    tr_idx  = list(range(0, test_start))
    val_idx = list(range(test_start, test_end))

    m = lgb.LGBMClassifier(**params)
    m.fit(X_tr.iloc[tr_idx], y_tr.iloc[tr_idx],
          eval_set=[(X_tr.iloc[val_idx], y_tr.iloc[val_idx])],
          callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])

    preds = m.predict_proba(X_tr.iloc[val_idx])[:, 1]
    auc   = roc_auc_score(y_tr.iloc[val_idx], preds)
    cv_aucs.append(auc)
    best_iters.append(m.best_iteration_)
    print(f'  Fold {fold+1}: AUC={auc:.4f}  iters={m.best_iteration_}')
    del m; gc.collect()

print(f'\\nCV AUC: {np.mean(cv_aucs):.4f} +/- {np.std(cv_aucs):.4f}')

# Train final meta model
avg_iter = max(50, int(np.mean(best_iters)))
meta_model = lgb.LGBMClassifier(**{**params, 'n_estimators': avg_iter})
meta_model.fit(X_tr, y_tr)

joblib.dump({
    'model':             meta_model,
    'meta_feature_cols': meta_feature_cols,
    'train_end':         TRAIN_END,
    'max_bars':          MAX_BARS,
    'spread_pips':       SPREAD_PIPS,
    'n_iters':           avg_iter,
    'cv_auc':            np.mean(cv_aucs),
}, MODELS_DIR / 'model_long_meta.joblib')
print(f'Meta model saved -> {MODELS_DIR}/model_long_meta.joblib')
"""),

md("## 5. Test Set Evaluation"),

code("""\
meta_proba = meta_model.predict_proba(X_te)[:, 1]
auc = roc_auc_score(y_te, meta_proba)
print(f'Meta model test AUC: {auc:.4f}')
print(f'Positive rate: {y_te.mean():.3f}')
print(f'Test rows: {len(y_te):,}')

results = df_meta_test.copy()
results['meta_proba'] = meta_proba

print(f'\\nPrecision / EV at meta model thresholds:')
print(f'{"Thresh":>8} {"N":>7} {"WR":>7} {"Mean PnL":>10} {"Mean win":>10} {"Mean loss":>10}')
print('-' * 60)
for t in [0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]:
    mask = results['meta_proba'] >= t
    if mask.sum() < 20: continue
    s   = results[mask]
    wr  = (s['pnl'] > 0).mean()
    ev  = s['pnl'].mean()
    mw  = s[s['pnl']>0]['pnl'].mean() if (s['pnl']>0).any() else 0
    ml  = s[s['pnl']<=0]['pnl'].mean() if (s['pnl']<=0).any() else 0
    print(f'{t:>8.2f} {mask.sum():>7,} {wr:>7.1%} {ev:>10.2f} {mw:>10.1f} {ml:>10.1f}')
"""),

md("## 6. Combined Filter: base_proba >= X AND meta_proba >= Y"),

code("""\
print('Combined base + meta filter:')
print(f'{"Base":>6} {"Meta":>6} {"N":>7} {"WR":>7} {"EV":>8} {"Mean win":>10} {"Mean loss":>10}')
print('-' * 65)
for bt in [0.5, 0.6]:
    for mt in [0.5, 0.55, 0.6, 0.65, 0.7]:
        mask = (results['base_proba'] >= bt) & (results['meta_proba'] >= mt)
        if mask.sum() < 15: continue
        s  = results[mask]
        wr = (s['pnl'] > 0).mean()
        ev = s['pnl'].mean()
        mw = s[s['pnl']>0]['pnl'].mean() if (s['pnl']>0).any() else 0
        ml = s[s['pnl']<=0]['pnl'].mean() if (s['pnl']<=0).any() else 0
        print(f'{bt:>6.1f} {mt:>6.2f} {mask.sum():>7,} {wr:>7.1%} {ev:>8.2f} {mw:>10.1f} {ml:>10.1f}')
"""),

md("## 7. Feature Importance of Meta Model"),

code("""\
import matplotlib.pyplot as plt

importance = pd.Series(meta_model.feature_importances_, index=meta_feature_cols)
importance = importance.sort_values(ascending=True).tail(30)

fig, ax = plt.subplots(figsize=(10, 10))
fig.patch.set_facecolor('#080c14')
ax.set_facecolor('#080c14')
ax.barh(importance.index, importance.values, color='#4fc3f7', alpha=0.8)
ax.tick_params(colors='white', labelsize=8)
ax.set_title('Feature Importance — Meta Model (Long)', color='white', fontsize=12)
for spine in ax.spines.values(): spine.set_edgecolor('#1a2332')
plt.tight_layout()
plt.show()
"""),

md("## 8. Per-Pair EV at Best Meta Threshold"),

code("""\
# Find best combined threshold: EV > 0 and N >= 30
best_bt, best_mt = 0.5, 0.6
mask = (results['base_proba'] >= best_bt) & (results['meta_proba'] >= best_mt)
print(f'Using base>={best_bt}, meta>={best_mt}: N={mask.sum():,}, WR={(results[mask]["pnl"]>0).mean():.1%}, EV={results[mask]["pnl"].mean():.2f}p')

print(f'\\nPer-pair breakdown:')
print(f'{"Pair":>10} {"N":>6} {"WR":>7} {"EV":>8} {"Mean win":>10} {"Mean loss":>10}')
print('-'*55)
for pair, g in results[mask].groupby('pair'):
    if len(g) < 5: continue
    wr = (g['pnl'] > 0).mean()
    ev = g['pnl'].mean()
    mw = g[g['pnl']>0]['pnl'].mean() if (g['pnl']>0).any() else 0
    ml = g[g['pnl']<=0]['pnl'].mean() if (g['pnl']<=0).any() else 0
    print(f'{pair:>10} {len(g):>6} {wr:>7.1%} {ev:>8.2f} {mw:>10.1f} {ml:>10.1f}')
"""),

md("## 9. Distribution Over Time (signal frequency)"),

code("""\
sub = results[(results['base_proba'] >= best_bt) & (results['meta_proba'] >= best_mt)].copy()

# Monthly signal count
sub['month'] = sub.index.to_period('M')
monthly = sub.groupby('month').agg(n=('pnl','count'), wr=('pnl', lambda x: (x>0).mean()), ev=('pnl','mean'))
print(f'Monthly signal distribution (base>={best_bt}, meta>={best_mt}):')
print(f'{"Month":>10} {"N":>5} {"WR":>7} {"EV":>8}')
print('-'*35)
for month, row in monthly.iterrows():
    bar = '#' * int(row['n'] / monthly['n'].max() * 20)
    print(f'{str(month):>10} {int(row["n"]):>5} {row["wr"]:>7.1%} {row["ev"]:>8.2f}  {bar}')

print(f'\\nAverage signals/month: {monthly["n"].mean():.1f}')
print(f'Overall WR: {(sub["pnl"]>0).mean():.1%}')
print(f'Overall EV: {sub["pnl"].mean():.2f}p')
"""),

md("## 10. SL Simulation on Meta-Filtered Signals"),

code("""\
print('SL simulation on meta-filtered signals (TP=MA200, timeout=T+72)')
print(f'Filter: base>={best_bt}, meta>={best_mt}')
print()

# Gather sim data for filtered signals
sim_data = []
for pair in PAIRS:
    pip  = PIP_SIZE[pair]
    ohlc = ohlc_cache[pair]
    filt = results[(results['pair'] == pair) &
                   (results['base_proba'] >= best_bt) &
                   (results['meta_proba'] >= best_mt)]
    for ts, row in filt.iterrows():
        if ts not in ohlc.index: continue
        iloc = ohlc.index.get_loc(ts)
        sim_data.append({
            'pair':   pair,
            'entry':  row['entry'],
            'target': row['target'],
            'pip':    pip,
            'lows':   ohlc['low'].iloc[iloc+1:iloc+MAX_BARS+1].values,
            'highs':  ohlc['high'].iloc[iloc+1:iloc+MAX_BARS+1].values,
            'closes': ohlc['close'].iloc[iloc+1:iloc+MAX_BARS+1].values,
            'pips_to_ma200': row['pips_to_ma200'],
        })

print(f'Signals to simulate: {len(sim_data):,}')
print(f'{"SL":>8} {"N TP":>8} {"N SL":>8} {"N TO":>8} {"WR":>7} {"Avg win":>9} {"Avg loss":>9} {"EV":>8} {"Sharpe":>8}')
print('-'*80)
for sl_pips in [30, 40, 50, 60, 75, 100, 150, 999]:
    pnls = []
    for s in sim_data:
        pip    = s['pip']
        entry  = s['entry']
        target = s['target']
        sl_lvl = entry - sl_pips * pip
        pnl = None
        for k in range(len(s['lows'])):
            if s['lows'][k] <= sl_lvl:
                pnl = -sl_pips - SPREAD_PIPS; break
            if s['highs'][k] >= target:
                pnl = (target - entry) / pip - SPREAD_PIPS; break
        if pnl is None:
            pnl = (s['closes'][-1] - entry) / pip - SPREAD_PIPS if len(s['closes']) > 0 else -SPREAD_PIPS
        pnls.append(pnl)
    pnls   = np.array(pnls)
    wins   = pnls[pnls > 0]
    losses = pnls[pnls <= 0]
    n_sl   = (pnls <= -(sl_pips - 0.1)).sum() if sl_pips < 999 else 0
    n_tp   = (pnls > 0).sum()
    n_to   = len(pnls) - n_sl - n_tp
    ev     = pnls.mean()
    sharpe = ev / pnls.std() * np.sqrt(252) if pnls.std() > 0 else 0
    lbl    = f'{sl_pips}p' if sl_pips < 999 else 'none'
    print(f'{lbl:>8} {n_tp:>8,} {n_sl:>8,} {n_to:>8,} {(pnls>0).mean():>7.1%} '
          f'{wins.mean() if len(wins) else 0:>9.1f} '
          f'{losses.mean() if len(losses) else 0:>9.1f} '
          f'{ev:>8.2f} {sharpe:>8.3f}')
"""),

]

nb['cells'] = cells

out = Path('notebooks_9/03_meta_model.ipynb')
with open(out, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print(f'Written: {out}')
print(f'Cells: {len(nb["cells"])}')
