import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

# Cell 0: markdown
cells.append(nbf.v4.new_markdown_cell(
"# Direction Model v2\n"
"**Changes vs v1:**\n"
"- Target: `sign(mfe_long_pips > mfe_short_pips)` instead of 8h log return\n"
"- Features: directional-only subset (flow, momentum, range, CSI/relstr, candle)\n"
"- Drops microstructure features designed to detect *when* moves happen\n"
))

# Cell 1: imports & config
cells.append(nbf.v4.new_code_cell(
'import joblib, warnings\n'
'import numpy as np\n'
'import pandas as pd\n'
'import lightgbm as lgb\n'
'from pathlib import Path\n'
'from sklearn.metrics import log_loss, accuracy_score, roc_auc_score\n'
'warnings.filterwarnings("ignore")\n'
'\n'
'FEATURES_DIR   = Path("../backend/data/features_9")\n'
'MFE_MODEL_PATH = Path("../backend/models_9/mfe_q50_8h/model_1H_Q50.joblib")\n'
'OUTPUT_DIR     = Path("../backend/models_9/dir_v2_8h")\n'
'OUTPUT_DIR.mkdir(parents=True, exist_ok=True)\n'
'\n'
'TRAIN_END  = "2024-06-30"\n'
'MFE_THRESH = 30.0\n'
'N_FOLDS    = 5\n'
'\n'
'FLOW_FEATURES = [\n'
'    "order_imbalance", "order_imbalance_intensity", "buy_volume_frac",\n'
'    "order_imbalance_delta_3h", "order_imbalance_delta_6h", "order_imbalance_delta_12h",\n'
']\n'
'MOMENTUM_FEATURES = [\n'
'    "ret_1d", "ret_3d", "ret_1w", "ret_2w",\n'
'    "momentum_shift", "accel_mean",\n'
']\n'
'RANGE_FEATURES = [\n'
'    "range_pos_24", "range_pos_48", "range_pos_5d", "range_pos_10d",\n'
'    "dist_from_24h_high", "dist_from_24h_low",\n'
'    "dist_5d_high", "dist_5d_low",\n'
'    "range_width_24", "range_width_48", "range_width_5d",\n'
']\n'
'CANDLE_FEATURES = [\n'
'    "body_ratio", "candle_direction", "consec_bullish", "consec_bearish",\n'
'    "upper_wick_ratio", "lower_wick_ratio",\n'
']\n'
'SESSION_FEATURES = [\n'
'    "hour_sin", "hour_cos", "dow_sin", "dow_cos",\n'
'    "is_london", "is_ny", "is_asia", "is_overlap",\n'
'    "is_month_end", "days_to_friday",\n'
']\n'
'VOL_CONTEXT = ["atr_ratio_6_24", "atr_ratio_6_72", "vol_regime_5d", "vol_regime_10d", "vol_trend"]\n'
'\n'
'print("Config loaded.")\n'
))

# Cell 2: load data
cells.append(nbf.v4.new_code_cell(
'print("Loading features_9...")\n'
'dfs = [pd.read_parquet(f) for f in sorted(FEATURES_DIR.glob("*_features.parquet"))]\n'
'df  = pd.concat(dfs).sort_index()\n'
'print(f"  Total rows: {len(df):,} | Pairs: {df[\'pair\'].nunique()}")\n'
'\n'
'cols = df.columns.tolist()\n'
'CSI_RELSTR_FEATURES = [c for c in cols if c.startswith(("csi_", "relstr_", "peer_", "corr_", "beta_"))]\n'
'print(f"  CSI/relstr/peer/corr/beta: {len(CSI_RELSTR_FEATURES)}")\n'
'\n'
'FEATURE_COLS = []\n'
'for group in [FLOW_FEATURES, MOMENTUM_FEATURES, RANGE_FEATURES,\n'
'              CANDLE_FEATURES, CSI_RELSTR_FEATURES, SESSION_FEATURES, VOL_CONTEXT]:\n'
'    FEATURE_COLS += [f for f in group if f in cols]\n'
'seen = set()\n'
'FEATURE_COLS = [f for f in FEATURE_COLS if not (f in seen or seen.add(f))]\n'
'print(f"  Directional feature cols: {len(FEATURE_COLS)}")\n'
))

# Cell 3: target & filter
cells.append(nbf.v4.new_code_cell(
'df["dir_target"] = (df["mfe_long_pips"] > df["mfe_short_pips"]).astype(int)\n'
'\n'
'print("Running MFE model...")\n'
'mfe_bundle    = joblib.load(MFE_MODEL_PATH)\n'
'mfe_model     = mfe_bundle["model"]\n'
'mfe_feat_cols = mfe_bundle["feature_cols"]\n'
'for col in mfe_feat_cols:\n'
'    if col not in df.columns:\n'
'        df[col] = 0.0\n'
'X_mfe = df[mfe_feat_cols].ffill().fillna(0)\n'
'df["q50_mfe"] = mfe_model.predict(X_mfe)\n'
'\n'
'df_sig = df[\n'
'    (df["q50_mfe"] >= MFE_THRESH) &\n'
'    df["mfe_long_pips"].notna() &\n'
'    df["mfe_short_pips"].notna()\n'
'].copy()\n'
'print(f"  Signal bars (MFE>={MFE_THRESH}): {len(df_sig):,}")\n'
'print(f"  Long: {df_sig[\'dir_target\'].mean():.2%}  |  Short: {1-df_sig[\'dir_target\'].mean():.2%}")\n'
'\n'
'df_train = df_sig[df_sig.index <= TRAIN_END].copy()\n'
'df_test  = df_sig[df_sig.index >  TRAIN_END].copy()\n'
'print(f"  Train: {len(df_train):,} | Test: {len(df_test):,}")\n'
))

# Cell 4: CV
cells.append(nbf.v4.new_code_cell(
'df_train_sorted = df_train.sort_index()\n'
'X_all = df_train_sorted[FEATURE_COLS].ffill().fillna(0).to_numpy()\n'
'y_all = df_train_sorted["dir_target"].to_numpy()\n'
'n     = len(X_all)\n'
'\n'
'fold_size = n // N_FOLDS\n'
'folds_pos = []\n'
'for i in range(N_FOLDS):\n'
'    tr_end = fold_size * (i + 1)\n'
'    te_end = min(fold_size * (i + 2), n)\n'
'    if te_end <= tr_end: break\n'
'    folds_pos.append((slice(0, tr_end), slice(tr_end, te_end)))\n'
'\n'
'LGB_PARAMS = dict(\n'
'    objective="binary", metric="binary_logloss",\n'
'    learning_rate=0.05, num_leaves=31,\n'
'    min_child_samples=50, feature_fraction=0.7,\n'
'    bagging_fraction=0.8, bagging_freq=5,\n'
'    lambda_l1=0.1, lambda_l2=0.1, verbose=-1,\n'
')\n'
'\n'
'oof_preds = np.full(n, np.nan)\n'
'oof_iters = []\n'
'\n'
'for fold_i, (tr, te) in enumerate(folds_pos):\n'
'    X_tr, y_tr = X_all[tr], y_all[tr]\n'
'    X_te, y_te = X_all[te], y_all[te]\n'
'    dtrain = lgb.Dataset(X_tr, label=y_tr)\n'
'    dval   = lgb.Dataset(X_te, label=y_te, reference=dtrain)\n'
'    m = lgb.train(LGB_PARAMS, dtrain, num_boost_round=500,\n'
'                  valid_sets=[dval],\n'
'                  callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])\n'
'    preds = m.predict(X_te)\n'
'    oof_preds[te] = preds\n'
'    oof_iters.append(m.best_iteration)\n'
'    print(f"  Fold {fold_i+1}: iters={m.best_iteration} | "\n'
'          f"acc={accuracy_score(y_te, preds>=0.5):.3f} | "\n'
'          f"AUC={roc_auc_score(y_te, preds):.3f} | "\n'
'          f"logloss={log_loss(y_te, preds):.4f}")\n'
'\n'
'mask = ~np.isnan(oof_preds)\n'
'oof_acc  = accuracy_score(y_all[mask], oof_preds[mask] >= 0.5)\n'
'oof_auc  = roc_auc_score(y_all[mask], oof_preds[mask])\n'
'best_iters = int(np.mean(oof_iters) * 1.1)\n'
'print(f"OOF: acc={oof_acc:.3f} | AUC={oof_auc:.3f} | final iters -> {best_iters}")\n'
))

# Cell 5: final model + test eval
cells.append(nbf.v4.new_code_cell(
'X_train_full = df_train_sorted[FEATURE_COLS].ffill().fillna(0).to_numpy()\n'
'y_train_full = df_train_sorted["dir_target"].to_numpy()\n'
'final_model = lgb.train(LGB_PARAMS, lgb.Dataset(X_train_full, label=y_train_full),\n'
'                        num_boost_round=best_iters, callbacks=[lgb.log_evaluation(-1)])\n'
'print(f"Final model: {best_iters} iters, {len(FEATURE_COLS)} features")\n'
'\n'
'X_test     = df_test[FEATURE_COLS].ffill().fillna(0).to_numpy()\n'
'y_test     = df_test["dir_target"].to_numpy()\n'
'test_preds = final_model.predict(X_test)\n'
'\n'
'test_acc  = accuracy_score(y_test, test_preds >= 0.5)\n'
'test_auc  = roc_auc_score(y_test, test_preds)\n'
'print(f"Test (post {TRAIN_END}):")\n'
'print(f"  Accuracy : {test_acc:.3f}  (baseline always-short: {1-y_test.mean():.3f})")\n'
'print(f"  AUC      : {test_auc:.3f}  (baseline: 0.500)")\n'
'print(f"  Log loss : {log_loss(y_test, test_preds):.4f}")\n'
))

# Cell 6: threshold sweep
cells.append(nbf.v4.new_code_cell(
'print("Threshold sweep:")\n'
'print(f"  {\'Thresh\':>8} {\'N_long\':>8} {\'N_short\':>8} {\'WR_long\':>9} {\'WR_short\':>9} {\'Overall\':>9}")\n'
'print(f"  {\'-\'*60}")\n'
'for thresh in [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]:\n'
'    pred_dir = (test_preds >= thresh).astype(int)\n'
'    mask_l = pred_dir == 1; mask_s = pred_dir == 0\n'
'    acc_l = y_test[mask_l].mean()      if mask_l.sum() > 0 else float("nan")\n'
'    acc_s = (1-y_test[mask_s]).mean()  if mask_s.sum() > 0 else float("nan")\n'
'    print(f"  {thresh:>8.2f} {mask_l.sum():>8,} {mask_s.sum():>8,} "\n'
'          f"{acc_l:>9.3f} {acc_s:>9.3f} {accuracy_score(y_test, pred_dir):>9.3f}")\n'
'\n'
'print("\\nPrediction bucket -> actual long rate:")\n'
'for lo, hi in [(0,.3),(.3,.4),(.4,.5),(.5,.6),(.6,.7),(.7,1.01)]:\n'
'    mask = (test_preds >= lo) & (test_preds < hi)\n'
'    if mask.sum() < 10: continue\n'
'    print(f"  [{lo:.1f}-{hi:.1f}): n={mask.sum():>5,}  long_rate={y_test[mask].mean():.3f}")\n'
))

# Cell 7: feature importance
cells.append(nbf.v4.new_code_cell(
'import matplotlib.pyplot as plt\n'
'fig, ax = plt.subplots(figsize=(10, 8), facecolor="#1a2332")\n'
'ax.set_facecolor("#1a2332")\n'
'importance = pd.Series(\n'
'    final_model.feature_importance(importance_type="gain"), index=FEATURE_COLS\n'
').sort_values(ascending=False).head(30)\n'
'colors = ["#4fc3f7" if i < 10 else "#546e8a" for i in range(len(importance))]\n'
'ax.barh(range(len(importance)), importance.values[::-1], color=colors[::-1])\n'
'ax.set_yticks(range(len(importance)))\n'
'ax.set_yticklabels(importance.index[::-1], color="white", fontsize=8)\n'
'ax.set_xlabel("Gain", color="white"); ax.tick_params(colors="white")\n'
'ax.set_title("Direction v2 — Top 30 Features by Gain", color="white", fontsize=12)\n'
'plt.tight_layout()\n'
'plt.savefig(OUTPUT_DIR / "feature_importance_v2.png", dpi=120, bbox_inches="tight", facecolor="#1a2332")\n'
'plt.show()\n'
'print("\\nTop 20 features:")\n'
'for feat, imp in importance.head(20).items():\n'
'    print(f"  {feat:<40} {imp:>10.0f}")\n'
))

# Cell 8: per-pair breakdown
cells.append(nbf.v4.new_code_cell(
'df_te = df_test.copy()\n'
'df_te["pred"]     = test_preds\n'
'df_te["pred_dir"] = (test_preds >= 0.5).astype(int)\n'
'df_te["correct"]  = (df_te["pred_dir"] == df_te["dir_target"])\n'
'print(f"{\'Pair\':<10} {\'N\':>6} {\'Acc\':>7} {\'AUC\':>7} {\'Long%\':>7} {\'Pred_L%\':>8} {\'Base\':>7}")\n'
'print("-" * 60)\n'
'for pair in sorted(df_te["pair"].unique()):\n'
'    s = df_te[df_te["pair"] == pair]\n'
'    if len(s) < 30: continue\n'
'    acc  = s["correct"].mean()\n'
'    base = 1 - s["dir_target"].mean()\n'
'    try: auc = roc_auc_score(s["dir_target"], s["pred"])\n'
'    except: auc = float("nan")\n'
'    print(f"{pair:<10} {len(s):>6,} {acc:>7.3f} {auc:>7.3f} "\n'
'          f"{s[\'dir_target\'].mean():>7.2%} {s[\'pred_dir\'].mean():>8.2%} {base:>7.3f}")\n'
))

# Cell 9: save
cells.append(nbf.v4.new_code_cell(
'bundle = {\n'
'    "model":        final_model,\n'
'    "feature_cols": FEATURE_COLS,\n'
'    "target":       "sign(mfe_long > mfe_short)",\n'
'    "train_end":    TRAIN_END,\n'
'    "mfe_thresh":   MFE_THRESH,\n'
'    "n_iters":      best_iters,\n'
'    "oof_acc":      oof_acc,\n'
'    "oof_auc":      oof_auc,\n'
'    "test_acc":     test_acc,\n'
'    "test_auc":     test_auc,\n'
'}\n'
'out_path = OUTPUT_DIR / "model_1H_dir_v2.joblib"\n'
'joblib.dump(bundle, out_path)\n'
'print(f"Saved: {out_path}")\n'
'print(f"OOF  acc={oof_acc:.3f} | AUC={oof_auc:.3f}")\n'
'print(f"Test acc={test_acc:.3f} | AUC={test_auc:.3f}")\n'
))

nb.cells = cells
nbf.write(nb, r"C:/Users/noual/lumeny/notebooks_8/05_dir_model_v2.ipynb")
print("Written: notebooks_8/05_dir_model_v2.ipynb")
