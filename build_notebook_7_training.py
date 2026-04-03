"""Build notebooks_7/03_model_training.ipynb — Geometric 3Q model"""
import json

cells = []

def md(source):
    return {'cell_type': 'markdown', 'id': f'md{len(cells)}', 'metadata': {}, 'source': source}

def code(source):
    return {'cell_type': 'code', 'id': f'cd{len(cells)}', 'metadata': {}, 'execution_count': None, 'outputs': [], 'source': source}

# ── Cell 0 — Title ────────────────────────────────────────────────────────────
cells.append(md(
    "# LumenY 7 — 3-Quantile Regression (1H) — Geometric Features\n\n"
    "Same architecture as notebooks_6/03_model_training_NEW.ipynb but trained\n"
    "on **pure geometric features** (features_8) instead of microstructure.\n\n"
    "**Goal:** Find 1H directional edge distributed across all sessions,\n"
    "not concentrated at Asian open.\n\n"
    "- **Features:** `backend/data/features_8/all_pairs_geometric.parquet` (54 features)\n"
    "- **Target:** `label_1H` — next 1H log return\n"
    "- **Pairs:** all 15\n"
    "- **Training cutoff:** 2024-06-30\n"
    "- **Models saved to:** `backend/models_7/3_quants/`"
))

# ── Cell 1 — Imports ──────────────────────────────────────────────────────────
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
    "from sklearn.metrics import mean_pinball_loss\n"
    "\n"
    "FEATURES_DIR = Path('../backend/data/features_8')\n"
    "MODELS_DIR   = Path('../backend/models_7/3_quants')\n"
    "MODELS_DIR.mkdir(parents=True, exist_ok=True)\n"
    "\n"
    "QUANTILES      = [0.25, 0.50, 0.75]\n"
    "QUANTILE_NAMES = ['Q25', 'Q50', 'Q75']\n"
    "\n"
    "TRAIN_END  = '2024-06-30'\n"
    "AVG_SPREAD = 0.00028\n"
    "\n"
    "print('Ready.')\n"
    "print(f'Training cutoff: {TRAIN_END}')\n"
    "print(f'Avg spread: {AVG_SPREAD}')"
))

# ── Cell 2 — Load data ────────────────────────────────────────────────────────
cells.append(md("## 1. Load Dataset"))
cells.append(code(
    "df_all = pd.read_parquet(FEATURES_DIR / 'all_pairs_geometric.parquet')\n"
    "print(f'Shape: {df_all.shape}')\n"
    "print(f'Date range: {df_all.index.min().date()} to {df_all.index.max().date()}')\n"
    "print(f'Pairs: {sorted(df_all[\"pair\"].unique())}')\n"
    "\n"
    "drop_cols    = ['pair', 'label_1H']\n"
    "feature_cols = [c for c in df_all.columns if c not in drop_cols]\n"
    "\n"
    "TARGET_COL = 'label_1H'\n"
    "\n"
    "df_train = df_all[df_all.index <= TRAIN_END].copy()\n"
    "df_test  = df_all[df_all.index >  TRAIN_END].copy()\n"
    "\n"
    "print(f'\\nFeatures:  {len(feature_cols)}')\n"
    "print(f'Train set: {len(df_train):,} rows  ({df_train.index.min().date()} to {df_train.index.max().date()})')\n"
    "print(f'Test set:  {len(df_test):,} rows  ({df_test.index.min().date()} to {df_test.index.max().date()})')\n"
    "print(f'\\nTarget stats (train):')\n"
    "print(df_train[TARGET_COL].describe())"
))

# ── Cell 3 — Walk forward ─────────────────────────────────────────────────────
cells.append(md("## 2. Walk-Forward Splits"))
cells.append(code(
    "def walk_forward_splits(n, n_splits=5, test_ratio=0.1):\n"
    "    \"\"\"Expanding window walk-forward splits within training data.\"\"\"\n"
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
    "valid_mask = df_train[TARGET_COL].notna()\n"
    "X_clean    = df_train[feature_cols][valid_mask].ffill().fillna(0)\n"
    "y_clean    = df_train[TARGET_COL][valid_mask]\n"
    "\n"
    "splits = walk_forward_splits(len(X_clean))\n"
    "print(f'Training samples: {len(X_clean):,}')\n"
    "print(f'Walk-forward splits: {len(splits)}')\n"
    "for i, (tr_idx, te_idx) in enumerate(splits):\n"
    "    print(f'  Fold {i+1}: train -> {X_clean.index[tr_idx[-1]].date()} '\n"
    "          f'({len(tr_idx):,}) | test {X_clean.index[te_idx[0]].date()} '\n"
    "          f'-> {X_clean.index[te_idx[-1]].date()} ({len(te_idx):,})')"
))

# ── Cell 4 — LightGBM params ──────────────────────────────────────────────────
cells.append(md("## 3. LightGBM Quantile Config"))
cells.append(code(
    "def get_lgbm_params(quantile):\n"
    "    return {\n"
    "        'objective':         'quantile',\n"
    "        'alpha':             quantile,\n"
    "        'metric':            'quantile',\n"
    "        'boosting_type':     'gbdt',\n"
    "        'n_estimators':      10000,\n"
    "        'learning_rate':     0.02,\n"
    "        'num_leaves':        64,\n"
    "        'max_depth':         6,\n"
    "        'min_child_samples': 50,\n"
    "        'feature_fraction':  0.7,\n"
    "        'bagging_fraction':  0.8,\n"
    "        'bagging_freq':      5,\n"
    "        'reg_alpha':         0.1,\n"
    "        'reg_lambda':        0.1,\n"
    "        'random_state':      42,\n"
    "        'n_jobs':            -1,\n"
    "        'verbose':           -1,\n"
    "        'device':            'gpu',\n"
    "    }\n"
    "\n"
    "print('Params ready.')"
))

# ── Cell 5 — Training ─────────────────────────────────────────────────────────
cells.append(md("## 4. Train 3 Quantile Models"))
cells.append(code(
    "print(f'Training samples: {len(X_clean):,}')\n"
    "print(f'Target mean: {y_clean.mean():.6f}, std: {y_clean.std():.6f}')\n"
    "\n"
    "oof_preds  = {q: np.full(len(X_clean), np.nan) for q in QUANTILES}\n"
    "best_iters = {q: [] for q in QUANTILES}\n"
    "cv_pinball = {q: [] for q in QUANTILES}\n"
    "\n"
    "for q, q_name in zip(QUANTILES, QUANTILE_NAMES):\n"
    "    print(f'\\nTraining {q_name} (alpha={q})...')\n"
    "    params = get_lgbm_params(q)\n"
    "\n"
    "    for fold, (tr_idx, te_idx) in enumerate(splits):\n"
    "        X_tr, y_tr = X_clean.iloc[tr_idx], y_clean.iloc[tr_idx]\n"
    "        X_te, y_te = X_clean.iloc[te_idx], y_clean.iloc[te_idx]\n"
    "\n"
    "        model = lgb.LGBMRegressor(**params)\n"
    "        model.fit(X_tr, y_tr, eval_set=[(X_te, y_te)],\n"
    "                  callbacks=[lgb.early_stopping(50, verbose=False),\n"
    "                             lgb.log_evaluation(-1)])\n"
    "\n"
    "        preds = model.predict(X_te)\n"
    "        oof_preds[q][te_idx] = preds\n"
    "        cv_pinball[q].append(mean_pinball_loss(y_te, preds, alpha=q))\n"
    "        best_iters[q].append(model.best_iteration_)\n"
    "        print(f'  Fold {fold+1}: pinball={cv_pinball[q][-1]:.6f}, iters={model.best_iteration_}')\n"
    "\n"
    "    # Exclude degenerate folds from iter average\n"
    "    valid_iters = [x for x in best_iters[q] if x >= 10]\n"
    "    avg_iter    = max(50, int(np.mean(valid_iters))) if valid_iters else 50\n"
    "    print(f'  CV Pinball: {np.mean(cv_pinball[q]):.6f} | Valid iters: {valid_iters} -> avg={avg_iter}')\n"
    "\n"
    "    # Final model on all training data\n"
    "    final_model = lgb.LGBMRegressor(**{**params, 'n_estimators': avg_iter})\n"
    "    final_model.fit(X_clean, y_clean)\n"
    "\n"
    "    q_int = int(q * 100)\n"
    "    joblib.dump({\n"
    "        'model':        final_model,\n"
    "        'quantile':     q,\n"
    "        'horizon':      '1H',\n"
    "        'feature_cols': feature_cols,\n"
    "        'train_end':    TRAIN_END,\n"
    "        'avg_spread':   AVG_SPREAD,\n"
    "        'cv_pinball':   np.mean(cv_pinball[q]),\n"
    "        'n_iters':      avg_iter,\n"
    "    }, MODELS_DIR / f'model_1H_Q{q_int}.joblib')\n"
    "\n"
    "    size_mb = (MODELS_DIR / f'model_1H_Q{q_int}.joblib').stat().st_size / 1024 / 1024\n"
    "    print(f'  Saved model_1H_Q{q_int}.joblib ({size_mb:.1f} MB)')\n"
    "    del final_model, model; gc.collect()\n"
    "\n"
    "print(f'\\nAll 3 models saved to {MODELS_DIR}')"
))

# ── Cell 6 — Test evaluation ──────────────────────────────────────────────────
cells.append(md("## 5. Test Set Evaluation"))
cells.append(code(
    "X_test       = df_test[feature_cols].ffill().fillna(0)\n"
    "y_test       = df_test[TARGET_COL]\n"
    "valid_test   = y_test.notna()\n"
    "X_test_clean = X_test[valid_test]\n"
    "y_test_clean = y_test[valid_test]\n"
    "\n"
    "q_preds_test = {}\n"
    "for q, q_name in zip(QUANTILES, QUANTILE_NAMES):\n"
    "    q_int = int(q * 100)\n"
    "    bundle = joblib.load(MODELS_DIR / f'model_1H_Q{q_int}.joblib')\n"
    "    q_preds_test[q_name] = bundle['model'].predict(X_test_clean)\n"
    "    print(f'{q_name}: mean={q_preds_test[q_name].mean():.6f}, std={q_preds_test[q_name].std():.6f}')\n"
    "\n"
    "results = pd.DataFrame({\n"
    "    'actual_return': y_test_clean.values,\n"
    "    'Q25':           q_preds_test['Q25'],\n"
    "    'Q50':           q_preds_test['Q50'],\n"
    "    'Q75':           q_preds_test['Q75'],\n"
    "    'pair':          df_test.loc[valid_test, 'pair'].values,\n"
    "}, index=y_test_clean.index)\n"
    "\n"
    "results['pred_dir']   = np.sign(results['Q50'])\n"
    "results['actual_dir'] = np.sign(results['actual_return'])\n"
    "results['abs_Q50']    = results['Q50'].abs()\n"
    "results['iqr']        = results['Q75'] - results['Q25']\n"
    "results['hour']       = results.index.hour\n"
    "\n"
    "n_test_days = (results.index.max() - results.index.min()).days\n"
    "print(f'\\nTest set: {len(results):,} rows ({results.index.min().date()} -> {results.index.max().date()})')\n"
    "\n"
    "thresholds = {\n"
    "    'All hours (no filter)': 0,\n"
    "    '|Q50| > 0.5x spread':  AVG_SPREAD * 0.5,\n"
    "    '|Q50| > 1x spread':    AVG_SPREAD,\n"
    "    '|Q50| > 2x spread':    AVG_SPREAD * 2,\n"
    "    '|Q50| > 3x spread':    AVG_SPREAD * 3,\n"
    "    '|Q50| > 5x spread':    AVG_SPREAD * 5,\n"
    "}\n"
    "\n"
    "print(f'\\n{\"Filter\":<25} {\"Trades\":>8} {\"Tr/day\":>8} {\"WinRate\":>9} {\"EV/trade\":>11} {\"Total PnL\":>12} {\"Sharpe\":>8}')\n"
    "print('-' * 88)\n"
    "\n"
    "for name, thresh in thresholds.items():\n"
    "    mask = results['abs_Q50'] > thresh\n"
    "    if mask.sum() < 50: continue\n"
    "    s   = results[mask]\n"
    "    pnl = s['pred_dir'] * s['actual_return'] - AVG_SPREAD\n"
    "    wr  = (s['pred_dir'] == s['actual_dir']).mean()\n"
    "    ev  = pnl.mean()\n"
    "    tot = pnl.sum()\n"
    "    shr = (pnl.mean() / pnl.std()) * np.sqrt(252 * 24) if pnl.std() > 0 else 0\n"
    "    tpd = mask.sum() / n_test_days\n"
    "    flag = ' <<<' if ev > 0 else ''\n"
    "    print(f'{name:<25} {mask.sum():>8,} {tpd:>8.1f} {wr:>8.1%} {ev:>11.6f} {tot:>12.4f} {shr:>8.2f}{flag}')\n"
    "\n"
    "print(f'\\nSpread cost: {AVG_SPREAD}')"
))

# ── Cell 7 — Per pair breakdown ───────────────────────────────────────────────
cells.append(md("## 6. Per-Pair Breakdown"))
cells.append(code(
    "filtered = results[results['abs_Q50'] > AVG_SPREAD * 0.5]\n"
    "\n"
    "print(f'Per-pair results (|Q50| > 0.5x spread):')\n"
    "print(f'\\n{\"Pair\":<10} {\"Trades\":>8} {\"Tr/day\":>8} {\"WinRate\":>9} {\"EV/trade\":>11} {\"Total PnL\":>12} {\"Sharpe\":>8}')\n"
    "print('-' * 75)\n"
    "\n"
    "for pair in sorted(filtered['pair'].unique()):\n"
    "    p   = filtered[filtered['pair'] == pair]\n"
    "    pnl = p['pred_dir'] * p['actual_return'] - AVG_SPREAD\n"
    "    wr  = (p['pred_dir'] == p['actual_dir']).mean()\n"
    "    shr = (pnl.mean() / pnl.std()) * np.sqrt(252 * 24) if pnl.std() > 0 else 0\n"
    "    tpd = len(p) / n_test_days\n"
    "    flag = ' <<<' if pnl.mean() > 0 else ''\n"
    "    print(f'{pair:<10} {len(p):>8,} {tpd:>8.1f} {wr:>8.1%} {pnl.mean():>11.6f} {pnl.sum():>12.4f} {shr:>8.2f}{flag}')"
))

# ── Cell 8 — Hour of day breakdown ────────────────────────────────────────────
cells.append(md(
    "## 7. Edge by Hour of Day\n\n"
    "Key diagnostic: is the edge distributed across sessions or concentrated?\n"
    "Compare with current model which clusters 75%+ of trades at 19-22 UTC."
))
cells.append(code(
    "filtered = results[results['abs_Q50'] > AVG_SPREAD * 0.5].copy()\n"
    "filtered['pnl']     = filtered['pred_dir'] * filtered['actual_return'] - AVG_SPREAD\n"
    "filtered['correct'] = (filtered['pred_dir'] == filtered['actual_dir']).astype(int)\n"
    "\n"
    "hourly = filtered.groupby('hour').agg(\n"
    "    trades=('pnl', 'count'),\n"
    "    wr=('correct', 'mean'),\n"
    "    ev=('pnl', 'mean'),\n"
    "    total_pnl=('pnl', 'sum')\n"
    ").reset_index()\n"
    "\n"
    "print(f'{\"Hour UTC\":<10} {\"Trades\":>8} {\"% of total\":>12} {\"WinRate\":>9} {\"EV/trade\":>11} {\"Total PnL\":>12}')\n"
    "print('-' * 65)\n"
    "total_trades = hourly['trades'].sum()\n"
    "for _, row in hourly.iterrows():\n"
    "    flag = ' <<<' if row['ev'] > 0 else ''\n"
    "    pct  = 100 * row['trades'] / total_trades\n"
    "    print(f'{int(row[\"hour\"]):>8}   {int(row[\"trades\"]):>8} {pct:>11.1f}% {row[\"wr\"]:>8.1%} '\n"
    "          f'{row[\"ev\"]:>11.6f} {row[\"total_pnl\"]:>12.4f}{flag}')\n"
    "\n"
    "# Session summary\n"
    "print(f'\\n--- Session Summary ---')\n"
    "sessions = {\n"
    "    'Asia    (00-07)': filtered['hour'].between(0, 6),\n"
    "    'London  (07-13)': filtered['hour'].between(7, 12),\n"
    "    'Overlap (13-16)': filtered['hour'].between(13, 15),\n"
    "    'NY      (16-22)': filtered['hour'].between(16, 21),\n"
    "    'Dead    (22-24)': filtered['hour'].between(22, 23),\n"
    "}\n"
    "for name, mask in sessions.items():\n"
    "    s   = filtered[mask]\n"
    "    if len(s) == 0: continue\n"
    "    pnl = s['pnl']\n"
    "    wr  = s['correct'].mean()\n"
    "    pct = 100 * len(s) / len(filtered)\n"
    "    flag = ' <<<' if pnl.mean() > 0 else ''\n"
    "    print(f'  {name}: {len(s):>6,} trades ({pct:.1f}%)  WR={wr:.1%}  EV={pnl.mean():.6f}{flag}')"
))

# ── Cell 9 — Equity curves ────────────────────────────────────────────────────
cells.append(md("## 8. Equity Curves"))
cells.append(code(
    "fig, axes = plt.subplots(2, 2, figsize=(16, 10))\n"
    "fig.patch.set_facecolor('#080c14')\n"
    "\n"
    "strategies = {\n"
    "    'All hours':          results,\n"
    "    '|Q50| > 0.5x spread': results[results['abs_Q50'] > AVG_SPREAD * 0.5],\n"
    "    '|Q50| > 1x spread':   results[results['abs_Q50'] > AVG_SPREAD],\n"
    "    '|Q50| > 2x spread':   results[results['abs_Q50'] > AVG_SPREAD * 2],\n"
    "}\n"
    "\n"
    "for ax, (name, subset) in zip(axes.flatten(), strategies.items()):\n"
    "    ax.set_facecolor('#080c14')\n"
    "    pnl     = subset['pred_dir'] * subset['actual_return'] - AVG_SPREAD\n"
    "    cum_pnl = pnl.cumsum()\n"
    "\n"
    "    for pair in sorted(subset['pair'].unique()):\n"
    "        p        = subset[subset['pair'] == pair]\n"
    "        pair_pnl = (p['pred_dir'] * p['actual_return'] - AVG_SPREAD).cumsum()\n"
    "        ax.plot(pair_pnl.index, pair_pnl.values, alpha=0.3, linewidth=0.7)\n"
    "\n"
    "    ax.plot(cum_pnl.index, cum_pnl.values, color='#4fc3f7', linewidth=2)\n"
    "    ax.axhline(0, color=(1,1,1,0.2), linewidth=1, linestyle='--')\n"
    "\n"
    "    wr  = (subset['pred_dir'] == subset['actual_dir']).mean()\n"
    "    ev  = pnl.mean()\n"
    "    shr = (pnl.mean() / pnl.std()) * np.sqrt(252 * 24) if pnl.std() > 0 else 0\n"
    "    ax.set_title(f'{name} ({len(subset):,} trades)\\nWR: {wr:.1%}, EV: {ev:.6f}, Sharpe: {shr:.2f}',\n"
    "                 color='white', fontsize=9)\n"
    "    ax.tick_params(colors='white')\n"
    "    ax.set_ylabel('Cum PnL (after spread)', color='white', fontsize=8)\n"
    "    for spine in ax.spines.values(): spine.set_edgecolor('#1a2332')\n"
    "\n"
    "plt.suptitle('Geometric 3Q Model — Equity Curves on Unseen Test Set', color='white', fontsize=13)\n"
    "plt.tight_layout()\n"
    "plt.show()"
))

# ── Cell 10 — Feature importance ─────────────────────────────────────────────
cells.append(md("## 9. Feature Importance"))
cells.append(code(
    "fig, axes = plt.subplots(1, 2, figsize=(18, 10))\n"
    "fig.patch.set_facecolor('#080c14')\n"
    "\n"
    "for ax, (q_name, q_int) in zip(axes, [('Q25', 25), ('Q50', 50)]):\n"
    "    bundle     = joblib.load(MODELS_DIR / f'model_1H_Q{q_int}.joblib')\n"
    "    importance = pd.Series(bundle['model'].feature_importances_, index=feature_cols)\n"
    "    importance = importance.sort_values(ascending=True).tail(25)\n"
    "\n"
    "    ax.barh(importance.index, importance.values, color='#4fc3f7', alpha=0.8)\n"
    "    ax.set_facecolor('#080c14')\n"
    "    ax.tick_params(colors='white', labelsize=7)\n"
    "    ax.set_title(f'{q_name} — Top 25 Features', color='white', fontsize=11)\n"
    "    for spine in ax.spines.values(): spine.set_edgecolor('#1a2332')\n"
    "\n"
    "plt.suptitle('Feature Importance — Geometric Model', color='white', fontsize=13)\n"
    "plt.tight_layout()\n"
    "plt.show()"
))

# ── Cell 11 — OOF calibration ─────────────────────────────────────────────────
cells.append(md("## 10. OOF Quantile Calibration"))
cells.append(code(
    "valid_oof = ~np.isnan(oof_preds[QUANTILES[0]])\n"
    "y_oof     = y_clean.values[valid_oof]\n"
    "\n"
    "print('OOF Quantile Coverage:')\n"
    "print(f'{\"Quantile\":<10} {\"Target\":<10} {\"Actual\":<10} {\"Gap\"}')\n"
    "print('-' * 42)\n"
    "for q, q_name in zip(QUANTILES, QUANTILE_NAMES):\n"
    "    preds_oof = oof_preds[q][valid_oof]\n"
    "    coverage  = np.mean(y_oof <= preds_oof)\n"
    "    gap       = coverage - q\n"
    "    flag      = 'OK' if abs(gap) < 0.03 else 'NEEDS CALIBRATION'\n"
    "    print(f'{q_name:<10} {q:<10.2f} {coverage:<10.3f} {gap:+.3f}  {flag}')"
))

# ── Cell 12 — Summary ─────────────────────────────────────────────────────────
cells.append(md("## 11. Summary"))
cells.append(code(
    "print('=' * 70)\n"
    "print('GEOMETRIC 3Q MODEL — TRAINING COMPLETE')\n"
    "print('=' * 70)\n"
    "\n"
    "print(f'\\nModels saved to: {MODELS_DIR.resolve()}')\n"
    "for f in sorted(MODELS_DIR.glob('*.joblib')):\n"
    "    print(f'  {f.name:<25} {f.stat().st_size / 1024 / 1024:.1f} MB')\n"
    "\n"
    "print(f'\\n-- CV Results --')\n"
    "for q, q_name in zip(QUANTILES, QUANTILE_NAMES):\n"
    "    valid_iters = [x for x in best_iters[q] if x >= 10]\n"
    "    print(f'  {q_name}: pinball={np.mean(cv_pinball[q]):.6f}, '\n"
    "          f'valid_iters={valid_iters}, avg={max(50, int(np.mean(valid_iters)))}')\n"
    "\n"
    "print(f'\\n-- Test Set (UNSEEN, after {TRAIN_END}) --')\n"
    "for name, thresh in thresholds.items():\n"
    "    mask = results['abs_Q50'] > thresh\n"
    "    if mask.sum() < 50: continue\n"
    "    s   = results[mask]\n"
    "    pnl = s['pred_dir'] * s['actual_return'] - AVG_SPREAD\n"
    "    ev  = pnl.mean()\n"
    "    wr  = (s['pred_dir'] == s['actual_dir']).mean()\n"
    "    shr = (pnl.mean() / pnl.std()) * np.sqrt(252 * 24) if pnl.std() > 0 else 0\n"
    "    tpd = mask.sum() / n_test_days\n"
    "    flag = ' <<<' if ev > 0 else ''\n"
    "    print(f'  {name:<25} WR={wr:.1%}  EV={ev:.6f}  Sharpe={shr:.2f}  Tr/day={tpd:.1f}{flag}')\n"
    "\n"
    "print(f'\\nTest period: {results.index.min().date()} -> {results.index.max().date()}')\n"
    "print(f'Training cutoff: {TRAIN_END}')\n"
    "print(f'Features: {len(feature_cols)} geometric features')\n"
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
