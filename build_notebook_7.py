"""Build notebooks_7/03_model_training.ipynb — Binary classifier version"""
import json

cells = []

def md(source):
    return {'cell_type': 'markdown', 'id': f'md{len(cells)}', 'metadata': {}, 'source': source}

def code(source):
    return {'cell_type': 'code', 'id': f'cd{len(cells)}', 'metadata': {}, 'execution_count': None, 'outputs': [], 'source': source}

# ── Cell 0 — title ────────────────────────────────────────────────────────────
cells.append(md(
    "# LumenY 7 — Binary Directional Classifier (15M) — 7 Majors\n\n"
    "**Approach:** Binary classification on 15M bars.\n\n"
    "**Label:**\n"
    "- `+1` if return_15M > spread (clear up move)\n"
    "- `-1` if return_15M < -spread (clear down move)\n"
    "- `0` ambiguous (|return| <= spread, no trade)\n\n"
    "**Model:** Two LightGBM binary classifiers — one for UP, one for DOWN.\n\n"
    "**Signal logic:** Trade when P(up) or P(down) exceeds threshold.\n\n"
    "**Features:** features_7 microstructure (64 features, 5M bars, filtered to 15M-aligned)\n\n"
    "**Models saved to:** `backend/models_7/classifier/`"
))

# ── Cell 1 — imports ──────────────────────────────────────────────────────────
cells.append(md("## 0. Setup"))
cells.append(code(
    "import pandas as pd\n"
    "import numpy as np\n"
    "import lightgbm as lgb\n"
    "import matplotlib.pyplot as plt\n"
    "import joblib\n"
    "import warnings\n"
    "import gc\n"
    "warnings.filterwarnings('ignore')\n"
    "\n"
    "from pathlib import Path\n"
    "from sklearn.metrics import roc_auc_score, classification_report\n"
    "\n"
    "FEATURES_DIR = Path('../backend/data/features_7')\n"
    "MODELS_DIR   = Path('../backend/models_7/classifier')\n"
    "MODELS_DIR.mkdir(parents=True, exist_ok=True)\n"
    "\n"
    "TRAIN_END  = '2024-06-30'\n"
    "SPREAD     = 0.00010   # label threshold — 1 pip on majors\n"
    "\n"
    "print('Ready.')\n"
    "print(f'Training cutoff: {TRAIN_END}')\n"
    "print(f'Spread threshold: {SPREAD}')"
))

# ── Cell 2 — load & label ─────────────────────────────────────────────────────
cells.append(md("## 1. Load Data & Build Labels"))
cells.append(code(
    "df_all = pd.read_parquet(FEATURES_DIR / 'all_pairs_microstructure.parquet')\n"
    "\n"
    "# Filter to 15M-aligned bars only\n"
    "df_all = df_all[df_all.index.minute.isin([0, 15, 30, 45])].copy()\n"
    "print(f'15M-aligned rows: {len(df_all):,}')\n"
    "\n"
    "# Build directional label\n"
    "ret = df_all['label_15m']\n"
    "df_all['label'] = 0\n"
    "df_all.loc[ret >  SPREAD, 'label'] = 1   # UP\n"
    "df_all.loc[ret < -SPREAD, 'label'] = -1  # DOWN\n"
    "\n"
    "# Feature columns\n"
    "drop_cols    = ['label_5m', 'label_15m', 'label_1h', 'pair', 'label']\n"
    "feature_cols = [c for c in df_all.columns if c not in drop_cols]\n"
    "\n"
    "# Train / test split\n"
    "df_train = df_all[df_all.index <= TRAIN_END].copy()\n"
    "df_test  = df_all[df_all.index >  TRAIN_END].copy()\n"
    "\n"
    "# Class balance\n"
    "for name, df in [('Train', df_train), ('Test', df_test)]:\n"
    "    vc = df['label'].value_counts().sort_index()\n"
    "    total = len(df)\n"
    "    print(f'{name}: UP={vc.get(1,0):,} ({100*vc.get(1,0)/total:.1f}%)  '\n"
    "          f'DOWN={vc.get(-1,0):,} ({100*vc.get(-1,0)/total:.1f}%)  '\n"
    "          f'SKIP={vc.get(0,0):,} ({100*vc.get(0,0)/total:.1f}%)  '\n"
    "          f'Total={total:,}')\n"
    "\n"
    "print(f'\\nFeatures: {len(feature_cols)}')\n"
    "print(f'Date range: {df_all.index.min().date()} to {df_all.index.max().date()}')"
))

# ── Cell 3 — walk forward ─────────────────────────────────────────────────────
cells.append(md("## 2. Walk-Forward Splits"))
cells.append(code(
    "def walk_forward_splits(n, n_splits=5, test_ratio=0.1):\n"
    "    test_size = int(n * test_ratio)\n"
    "    splits = []\n"
    "    for i in range(n_splits):\n"
    "        test_start = int(n * 0.5) + i * (int(n * 0.5) // n_splits)\n"
    "        test_end   = test_start + test_size\n"
    "        if test_end > n:\n"
    "            break\n"
    "        splits.append((list(range(0, test_start)), list(range(test_start, test_end))))\n"
    "    return splits\n"
    "\n"
    "# Use all rows for feature matrix (label=0 rows included as negatives for both classifiers)\n"
    "valid_mask = df_train['label_15m'].notna()\n"
    "X_clean    = df_train[feature_cols][valid_mask].ffill().fillna(0)\n"
    "y_clean    = df_train['label'][valid_mask]\n"
    "\n"
    "splits = walk_forward_splits(len(X_clean))\n"
    "print(f'Training rows: {len(X_clean):,}')\n"
    "print(f'Walk-forward splits: {len(splits)}')\n"
    "for i, (tr_idx, te_idx) in enumerate(splits):\n"
    "    print(f'  Fold {i+1}: train -> {X_clean.index[tr_idx[-1]].date()} '\n"
    "          f'({len(tr_idx):,}) | test {X_clean.index[te_idx[0]].date()} '\n"
    "          f'-> {X_clean.index[te_idx[-1]].date()} ({len(te_idx):,})')"
))

# ── Cell 4 — model params ─────────────────────────────────────────────────────
cells.append(md("## 3. Model Config\n\n"
    "Two binary classifiers:\n"
    "- **UP model:** P(return > spread) — label_up = 1 if label==+1, else 0\n"
    "- **DOWN model:** P(return < -spread) — label_down = 1 if label==-1, else 0\n\n"
    "Both treat label=0 bars as negatives (ambiguous bars are non-events)."))
cells.append(code(
    "def get_clf_params():\n"
    "    return {\n"
    "        'objective':         'binary',\n"
    "        'metric':            'auc',\n"
    "        'boosting_type':     'gbdt',\n"
    "        'n_estimators':      3000,\n"
    "        'learning_rate':     0.02,\n"
    "        'num_leaves':        64,\n"
    "        'max_depth':         6,\n"
    "        'min_child_samples': 50,\n"
    "        'feature_fraction':  0.7,\n"
    "        'bagging_fraction':  0.8,\n"
    "        'bagging_freq':      5,\n"
    "        'reg_alpha':         0.1,\n"
    "        'reg_lambda':        0.1,\n"
    "        'is_unbalance':      True,\n"
    "        'random_state':      42,\n"
    "        'n_jobs':            -1,\n"
    "        'verbose':           -1,\n"
    "        'device':            'gpu',\n"
    "    }\n"
    "\n"
    "print('Params ready.')"
))

# ── Cell 5 — train ────────────────────────────────────────────────────────────
cells.append(md("## 4. Train UP & DOWN Classifiers"))
cells.append(code(
    "# Binary targets\n"
    "y_up   = (y_clean ==  1).astype(int)  # 1 = up move, 0 = everything else\n"
    "y_down = (y_clean == -1).astype(int)  # 1 = down move, 0 = everything else\n"
    "\n"
    "print(f'UP   positives: {y_up.sum():,} ({100*y_up.mean():.1f}%)')\n"
    "print(f'DOWN positives: {y_down.sum():,} ({100*y_down.mean():.1f}%)')\n"
    "\n"
    "results_cv = {}\n"
    "\n"
    "for direction, y_bin in [('UP', y_up), ('DOWN', y_down)]:\n"
    "    print(f'\\n--- Training {direction} classifier ---')\n"
    "    params     = get_clf_params()\n"
    "    oof_proba  = np.full(len(X_clean), np.nan)\n"
    "    best_iters = []\n"
    "    aucs       = []\n"
    "\n"
    "    for fold, (tr_idx, te_idx) in enumerate(splits):\n"
    "        X_tr, y_tr = X_clean.iloc[tr_idx], y_bin.iloc[tr_idx]\n"
    "        X_te, y_te = X_clean.iloc[te_idx], y_bin.iloc[te_idx]\n"
    "\n"
    "        model = lgb.LGBMClassifier(**params)\n"
    "        model.fit(X_tr, y_tr, eval_set=[(X_te, y_te)],\n"
    "                  callbacks=[lgb.early_stopping(50, verbose=False),\n"
    "                             lgb.log_evaluation(-1)])\n"
    "\n"
    "        proba = model.predict_proba(X_te)[:, 1]\n"
    "        oof_proba[te_idx] = proba\n"
    "        auc = roc_auc_score(y_te, proba)\n"
    "        aucs.append(auc)\n"
    "        best_iters.append(model.best_iteration_)\n"
    "        print(f'  Fold {fold+1}: AUC={auc:.4f}, iters={model.best_iteration_}')\n"
    "\n"
    "    # Filter out degenerate folds (< 10 iters)\n"
    "    valid_iters = [x for x in best_iters if x >= 10]\n"
    "    avg_iter    = max(100, int(np.mean(valid_iters))) if valid_iters else 100\n"
    "    print(f'  CV AUC: {np.mean(aucs):.4f} | Valid iters: {valid_iters} -> avg={avg_iter}')\n"
    "\n"
    "    # Train final model on all training data\n"
    "    final = lgb.LGBMClassifier(**{**params, 'n_estimators': avg_iter})\n"
    "    final.fit(X_clean, y_bin)\n"
    "\n"
    "    joblib.dump({\n"
    "        'model':        final,\n"
    "        'direction':    direction,\n"
    "        'feature_cols': feature_cols,\n"
    "        'train_end':    TRAIN_END,\n"
    "        'spread':       SPREAD,\n"
    "        'cv_auc':       np.mean(aucs),\n"
    "        'n_iters':      avg_iter,\n"
    "    }, MODELS_DIR / f'model_15M_{direction}.joblib')\n"
    "\n"
    "    results_cv[direction] = {'aucs': aucs, 'iters': best_iters, 'oof': oof_proba}\n"
    "    print(f'  Saved model_15M_{direction}.joblib')\n"
    "    del final, model; gc.collect()\n"
    "\n"
    "print('\\nTraining complete.')"
))

# ── Cell 6 — test evaluation ──────────────────────────────────────────────────
cells.append(md("## 5. Test Set Evaluation"))
cells.append(code(
    "# Prepare test set\n"
    "valid_test   = df_test['label_15m'].notna()\n"
    "X_test_clean = df_test[feature_cols][valid_test].ffill().fillna(0)\n"
    "y_test_label = df_test['label'][valid_test]\n"
    "y_test_ret   = df_test['label_15m'][valid_test]\n"
    "pairs_test   = df_test['pair'][valid_test]\n"
    "\n"
    "n_test_days  = (X_test_clean.index.max() - X_test_clean.index.min()).days\n"
    "\n"
    "# Load models and predict\n"
    "proba_up   = joblib.load(MODELS_DIR / 'model_15M_UP.joblib')['model'].predict_proba(X_test_clean)[:, 1]\n"
    "proba_down = joblib.load(MODELS_DIR / 'model_15M_DOWN.joblib')['model'].predict_proba(X_test_clean)[:, 1]\n"
    "\n"
    "results = pd.DataFrame({\n"
    "    'ret':        y_test_ret.values,\n"
    "    'label':      y_test_label.values,\n"
    "    'proba_up':   proba_up,\n"
    "    'proba_down': proba_down,\n"
    "    'pair':       pairs_test.values,\n"
    "}, index=X_test_clean.index)\n"
    "\n"
    "# Signal: take the stronger of the two probabilities\n"
    "results['signal']    = 0\n"
    "results['signal_p']  = np.maximum(results['proba_up'], results['proba_down'])\n"
    "results.loc[results['proba_up']   > results['proba_down'], 'signal'] =  1\n"
    "results.loc[results['proba_down'] > results['proba_up'],   'signal'] = -1\n"
    "\n"
    "print(f'Test set: {len(results):,} rows ({results.index.min().date()} -> {results.index.max().date()})')\n"
    "print(f'Baseline (no model): UP={100*(results[\"label\"]==1).mean():.1f}%  '\n"
    "      f'DOWN={100*(results[\"label\"]==-1).mean():.1f}%  '\n"
    "      f'SKIP={100*(results[\"label\"]==0).mean():.1f}%')\n"
    "\n"
    "# Probability threshold sweep\n"
    "print(f'\\n{\"Threshold\":>10} {\"Trades\":>8} {\"Tr/day\":>8} {\"WinRate\":>9} '\n"
    "      f'{\"EV/trade\":>11} {\"Total PnL\":>12} {\"Sharpe\":>8}')\n"
    "print('-' * 72)\n"
    "\n"
    "for thresh in [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]:\n"
    "    mask = results['signal_p'] > thresh\n"
    "    if mask.sum() < 10:\n"
    "        continue\n"
    "    s   = results[mask]\n"
    "    # P&L: signal direction * actual return - spread\n"
    "    pnl = s['signal'] * s['ret'] - SPREAD\n"
    "    wr  = (s['signal'] == s['label']).mean()\n"
    "    ev  = pnl.mean()\n"
    "    tot = pnl.sum()\n"
    "    shr = (pnl.mean() / pnl.std()) * np.sqrt(252 * 24 * 4) if pnl.std() > 0 else 0\n"
    "    tpd = mask.sum() / n_test_days\n"
    "    flag = ' <<<' if ev > 0 else ''\n"
    "    print(f'{thresh:>10.2f} {mask.sum():>8,} {tpd:>8.1f} {wr:>8.1%} '\n"
    "          f'{ev:>11.6f} {tot:>12.4f} {shr:>8.2f}{flag}')\n"
    "\n"
    "print(f'\\nSpread cost: {SPREAD}')"
))

# ── Cell 7 — per pair ─────────────────────────────────────────────────────────
cells.append(md("## 6. Per-Pair Breakdown"))
cells.append(code(
    "BEST_THRESH = 0.55  # adjust after seeing sweep results above\n"
    "\n"
    "filtered = results[results['signal_p'] > BEST_THRESH]\n"
    "print(f'Per-pair results (signal_p > {BEST_THRESH}):')\n"
    "print(f'\\n{\"Pair\":<10} {\"Trades\":>8} {\"Tr/day\":>8} {\"WinRate\":>9} '\n"
    "      f'{\"EV/trade\":>11} {\"Total PnL\":>12} {\"Sharpe\":>8}')\n"
    "print('-' * 72)\n"
    "\n"
    "for pair in sorted(filtered['pair'].unique()):\n"
    "    p   = filtered[filtered['pair'] == pair]\n"
    "    pnl = p['signal'] * p['ret'] - SPREAD\n"
    "    wr  = (p['signal'] == p['label']).mean()\n"
    "    shr = (pnl.mean() / pnl.std()) * np.sqrt(252 * 24 * 4) if pnl.std() > 0 else 0\n"
    "    tpd = len(p) / n_test_days\n"
    "    flag = ' <<<' if pnl.mean() > 0 else ''\n"
    "    print(f'{pair:<10} {len(p):>8,} {tpd:>8.1f} {wr:>8.1%} '\n"
    "          f'{pnl.mean():>11.6f} {pnl.sum():>12.4f} {shr:>8.2f}{flag}')"
))

# ── Cell 8 — hour of day analysis ────────────────────────────────────────────
cells.append(md("## 7. Edge by Hour of Day"))
cells.append(code(
    "filtered = results[results['signal_p'] > BEST_THRESH].copy()\n"
    "filtered['hour'] = filtered.index.hour\n"
    "filtered['pnl']  = filtered['signal'] * filtered['ret'] - SPREAD\n"
    "filtered['correct'] = (filtered['signal'] == filtered['label']).astype(int)\n"
    "\n"
    "hourly = filtered.groupby('hour').agg(\n"
    "    trades=('pnl', 'count'),\n"
    "    wr=('correct', 'mean'),\n"
    "    ev=('pnl', 'mean'),\n"
    "    total_pnl=('pnl', 'sum')\n"
    ").reset_index()\n"
    "\n"
    "print(f'{\"Hour\":>6} {\"Trades\":>8} {\"WinRate\":>9} {\"EV/trade\":>11} {\"Total PnL\":>12}')\n"
    "print('-' * 52)\n"
    "for _, row in hourly.iterrows():\n"
    "    flag = ' <<<' if row['ev'] > 0 else ''\n"
    "    print(f'{int(row[\"hour\"]):>6} {int(row[\"trades\"]):>8} {row[\"wr\"]:>8.1%} '\n"
    "          f'{row[\"ev\"]:>11.6f} {row[\"total_pnl\"]:>12.4f}{flag}')"
))

# ── Cell 9 — equity curves ────────────────────────────────────────────────────
cells.append(md("## 8. Equity Curves"))
cells.append(code(
    "fig, axes = plt.subplots(2, 2, figsize=(16, 10))\n"
    "fig.patch.set_facecolor('#080c14')\n"
    "\n"
    "thresholds_plot = [0.45, 0.50, 0.55, 0.60]\n"
    "\n"
    "for ax, thresh in zip(axes.flatten(), thresholds_plot):\n"
    "    ax.set_facecolor('#080c14')\n"
    "    subset  = results[results['signal_p'] > thresh]\n"
    "    pnl     = subset['signal'] * subset['ret'] - SPREAD\n"
    "    cum_pnl = pnl.cumsum()\n"
    "\n"
    "    for pair in sorted(subset['pair'].unique()):\n"
    "        p        = subset[subset['pair'] == pair]\n"
    "        pair_pnl = (p['signal'] * p['ret'] - SPREAD).cumsum()\n"
    "        ax.plot(pair_pnl.index, pair_pnl.values, alpha=0.3, linewidth=0.7)\n"
    "\n"
    "    ax.plot(cum_pnl.index, cum_pnl.values, color='#4fc3f7', linewidth=2)\n"
    "    ax.axhline(0, color=(1,1,1,0.2), linewidth=1, linestyle='--')\n"
    "\n"
    "    wr  = (subset['signal'] == subset['label']).mean()\n"
    "    ev  = pnl.mean()\n"
    "    shr = (pnl.mean() / pnl.std()) * np.sqrt(252 * 24 * 4) if pnl.std() > 0 else 0\n"
    "    ax.set_title(f'Threshold={thresh} ({len(subset):,} trades)\\n'\n"
    "                 f'WR: {wr:.1%}, EV: {ev:.6f}, Sharpe: {shr:.2f}',\n"
    "                 color='white', fontsize=9)\n"
    "    ax.tick_params(colors='white')\n"
    "    ax.set_ylabel('Cum PnL (after spread)', color='white', fontsize=8)\n"
    "    for spine in ax.spines.values(): spine.set_edgecolor('#1a2332')\n"
    "\n"
    "plt.suptitle('Binary Classifier 15M — Equity Curves on Unseen Test Set',\n"
    "             color='white', fontsize=13)\n"
    "plt.tight_layout()\n"
    "plt.show()"
))

# ── Cell 10 — feature importance ─────────────────────────────────────────────
cells.append(md("## 9. Feature Importance"))
cells.append(code(
    "fig, axes = plt.subplots(1, 2, figsize=(18, 10))\n"
    "fig.patch.set_facecolor('#080c14')\n"
    "\n"
    "for ax, direction in zip(axes, ['UP', 'DOWN']):\n"
    "    bundle     = joblib.load(MODELS_DIR / f'model_15M_{direction}.joblib')\n"
    "    importance = pd.Series(bundle['model'].feature_importances_, index=feature_cols)\n"
    "    importance = importance.sort_values(ascending=True).tail(25)\n"
    "\n"
    "    ax.barh(importance.index, importance.values, color='#4fc3f7', alpha=0.8)\n"
    "    ax.set_facecolor('#080c14')\n"
    "    ax.tick_params(colors='white', labelsize=7)\n"
    "    ax.set_title(f'{direction} Classifier — Top 25 Features', color='white', fontsize=11)\n"
    "    for spine in ax.spines.values(): spine.set_edgecolor('#1a2332')\n"
    "\n"
    "plt.suptitle('Feature Importance — 15M Binary Classifier', color='white', fontsize=13)\n"
    "plt.tight_layout()\n"
    "plt.show()"
))

# ── Cell 11 — summary ─────────────────────────────────────────────────────────
cells.append(md("## 10. Summary"))
cells.append(code(
    "print('=' * 70)\n"
    "print('BINARY CLASSIFIER 15M — COMPLETE')\n"
    "print('=' * 70)\n"
    "\n"
    "for direction in ['UP', 'DOWN']:\n"
    "    b = joblib.load(MODELS_DIR / f'model_15M_{direction}.joblib')\n"
    "    print(f'\\n{direction} model: AUC={b[\"cv_auc\"]:.4f}, iters={b[\"n_iters\"]}  '\n"
    "          f'| {(MODELS_DIR / f\"model_15M_{direction}.joblib\").stat().st_size/1e6:.1f} MB')\n"
    "\n"
    "print(f'\\n-- Test Set Results --')\n"
    "for thresh in [0.45, 0.50, 0.55, 0.60, 0.65]:\n"
    "    mask = results['signal_p'] > thresh\n"
    "    if mask.sum() < 10: continue\n"
    "    s   = results[mask]\n"
    "    pnl = s['signal'] * s['ret'] - SPREAD\n"
    "    wr  = (s['signal'] == s['label']).mean()\n"
    "    ev  = pnl.mean()\n"
    "    shr = (pnl.mean() / pnl.std()) * np.sqrt(252 * 24 * 4) if pnl.std() > 0 else 0\n"
    "    tpd = mask.sum() / n_test_days\n"
    "    flag = ' <<<' if ev > 0 else ''\n"
    "    print(f'  P>{thresh:.2f}: WR={wr:.1%}  EV={ev:.6f}  Sharpe={shr:.2f}  Tr/day={tpd:.1f}{flag}')\n"
    "\n"
    "print(f'\\nTest period: {results.index.min().date()} -> {results.index.max().date()}')\n"
    "print(f'Training cutoff: {TRAIN_END}')\n"
    "print(f'Spread threshold: {SPREAD}')\n"
    "print(f'Features: {len(feature_cols)}')\n"
    "print(f'Pairs: {sorted(results[\"pair\"].unique())}')"
))

# ── Write notebook ─────────────────────────────────────────────────────────────
nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"}
    },
    "cells": cells
}

out_path = r"C:\Users\noual\lumeny\notebooks_7\03_model_training.ipynb"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Notebook written: {out_path}")
print(f"Cells: {len(cells)}")
