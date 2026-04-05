import json

with open('notebooks_9/02_model_training.ipynb') as f:
    nb = json.load(f)

def md(src, id_): return {'cell_type': 'markdown', 'metadata': {}, 'source': src, 'id': id_}
def code(src, id_): return {'cell_type': 'code', 'metadata': {}, 'source': src, 'outputs': [], 'execution_count': None, 'id': id_}

# ── Cell 2: label markdown ──────────────────────────────────────────────────
nb['cells'][2] = md(
    '## 1. Compute Episode Low / High Labels\n\n'
    'For each below-both episode: bar with lowest low = label_long=1.\n'
    'For each above-both episode: bar with highest high = label_short=1.',
    'md1'
)

# ── Cell 3: label code ─────────────────────────────────────────────────────
nb['cells'][3] = code(
r"""print('Computing episode low/high labels...')
label_parts = []

for pair in PAIRS:
    pip  = PIP_SIZE[pair]
    ohlc = pd.read_parquet(PROCESSED_DIR / f'{pair}_1H.parquet')[['close', 'high', 'low']]
    ohlc.index = pd.to_datetime(ohlc.index)

    c     = ohlc['close'].values
    hi    = ohlc['high'].values
    lo    = ohlc['low'].values
    ma50  = pd.Series(c).rolling(50).mean().values
    ma200 = pd.Series(c).rolling(200).mean().values
    n     = len(c)

    both_below = (c < ma50) & (c < ma200)
    both_above = (c > ma50) & (c > ma200)
    label_long  = np.full(n, np.nan)
    label_short = np.full(n, np.nan)

    # LONG: episode low
    i = 0
    while i < n:
        if np.isnan(ma50[i]) or np.isnan(ma200[i]) or not both_below[i]:
            i += 1; continue
        ep_start = i
        while i < n:
            if np.isnan(ma50[i]) or np.isnan(ma200[i]): i += 1; break
            if both_above[i]: break
            i += 1
        ep_end = i
        ep_lows = lo[ep_start:ep_end]
        if len(ep_lows) == 0: continue
        min_idx = ep_start + int(np.argmin(ep_lows))
        for j in range(ep_start, ep_end):
            if np.isnan(ma50[j]) or np.isnan(ma200[j]): continue
            label_long[j] = 1.0 if j == min_idx else 0.0

    # SHORT: episode high
    i = 0
    while i < n:
        if np.isnan(ma50[i]) or np.isnan(ma200[i]) or not both_above[i]:
            i += 1; continue
        ep_start = i
        while i < n:
            if np.isnan(ma50[i]) or np.isnan(ma200[i]): i += 1; break
            if both_below[i]: break
            i += 1
        ep_end = i
        ep_highs = hi[ep_start:ep_end]
        if len(ep_highs) == 0: continue
        max_idx = ep_start + int(np.argmax(ep_highs))
        for j in range(ep_start, ep_end):
            if np.isnan(ma50[j]) or np.isnan(ma200[j]): continue
            label_short[j] = 1.0 if j == max_idx else 0.0

    tmp = pd.DataFrame({'label_long': label_long, 'label_short': label_short, 'pair': pair}, index=ohlc.index)
    label_parts.append(tmp)

df_labels = pd.concat(label_parts).sort_index()
long_pop  = df_labels['label_long'].dropna()
short_pop = df_labels['label_short'].dropna()
print(f'Long  labeled: {len(long_pop):,}  hit rate={long_pop.mean():.4f}  (1 per {1/long_pop.mean():.0f} bars)')
print(f'Short labeled: {len(short_pop):,}  hit rate={short_pop.mean():.4f}  (1 per {1/short_pop.mean():.0f} bars)')
""", 'c3')

# ── Cell 5: build dataset ──────────────────────────────────────────────────
nb['cells'][5] = code(
r"""print('Loading features_10...')
dfs = []
for pair in PAIRS:
    feat = pd.read_parquet(FEATURES_10_DIR / f'{pair}_features.parquet')
    feat = feat.drop(columns=[c for c in feat.columns if c.startswith('label_')], errors='ignore')
    dfs.append(feat)

df_feat = pd.concat(dfs).sort_index()
print(f'features_10: {df_feat.shape}')

df_f   = df_feat.reset_index()
df_l   = df_labels.reset_index()
ts_col = df_f.columns[0]
df     = pd.merge(df_f, df_l, on=[ts_col, 'pair'], how='inner')
df     = df.set_index(ts_col).sort_index()
del df_feat, df_f, df_l; gc.collect()

drop_cols    = ['label_long', 'label_short', 'pair', 'both_below', 'both_above']
feature_cols = [c for c in df.columns if c not in drop_cols]
print(f'Feature cols: {len(feature_cols)}')

df_sig_long  = df[(df['both_below'] == 1) & df['label_long'].notna()].copy()
df_sig_short = df[(df['both_above'] == 1) & df['label_short'].notna()].copy()
print(f'Long  episode bars: {len(df_sig_long):,}   pos_rate={df_sig_long["label_long"].mean():.4f}')
print(f'Short episode bars: {len(df_sig_short):,}  pos_rate={df_sig_short["label_short"].mean():.4f}')
""", 'c5')

# ── Cell 7: train both models ──────────────────────────────────────────────
nb['cells'][7] = code(
r"""def train_ep_model(X_tr, y_tr, label):
    pos_weight = (y_tr == 0).sum() / (y_tr == 1).sum()
    params = {
        'objective': 'binary', 'metric': 'auc', 'boosting_type': 'gbdt',
        'n_estimators': 3000, 'learning_rate': 0.02, 'num_leaves': 64,
        'max_depth': 6, 'min_child_samples': 50, 'feature_fraction': 0.7,
        'bagging_fraction': 0.8, 'bagging_freq': 5, 'reg_alpha': 0.1,
        'reg_lambda': 0.1, 'scale_pos_weight': pos_weight,
        'random_state': 42, 'n_jobs': -1, 'verbose': -1, 'device': 'gpu',
    }
    n = len(X_tr)
    cv_aucs, best_iters = [], []
    for fold in range(5):
        test_start = int(n * 0.5) + fold * (int(n * 0.5) // 5)
        test_end   = test_start + int(n * 0.1)
        if test_end > n: break
        tr_idx  = list(range(test_start))
        val_idx = list(range(test_start, test_end))
        m = lgb.LGBMClassifier(**params)
        m.fit(X_tr.iloc[tr_idx], y_tr.iloc[tr_idx],
              eval_set=[(X_tr.iloc[val_idx], y_tr.iloc[val_idx])],
              callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])
        auc = roc_auc_score(y_tr.iloc[val_idx], m.predict_proba(X_tr.iloc[val_idx])[:, 1])
        cv_aucs.append(auc); best_iters.append(m.best_iteration_)
        print(f'  [{label}] Fold {fold+1}: AUC={auc:.4f}  iters={m.best_iteration_}')
        del m; gc.collect()
    print(f'  [{label}] CV AUC: {np.mean(cv_aucs):.4f} +/- {np.std(cv_aucs):.4f}')
    avg_iter = max(50, int(np.mean(best_iters)))
    final = lgb.LGBMClassifier(**{**params, 'n_estimators': avg_iter})
    final.fit(X_tr, y_tr)
    return final, float(np.mean(cv_aucs))

# LONG
df_train_l = df_sig_long[df_sig_long.index <= TRAIN_END]
df_test_l  = df_sig_long[df_sig_long.index >  TRAIN_END]
X_train_l  = df_train_l[feature_cols].ffill().fillna(0)
y_train_l  = df_train_l['label_long']
X_test_l   = df_test_l[feature_cols].ffill().fillna(0)
y_test_l   = df_test_l['label_long']
print(f'LONG  train: {len(X_train_l):,}  pos_rate={y_train_l.mean():.4f}')
long_model, long_cv = train_ep_model(X_train_l, y_train_l, 'LONG')
proba_l = long_model.predict_proba(X_test_l)[:, 1]
print(f'LONG  test AUC: {roc_auc_score(y_test_l, proba_l):.4f}')
joblib.dump({'model': long_model, 'feature_cols': feature_cols, 'train_end': TRAIN_END, 'cv_auc': long_cv, 'direction': 'long'}, MODELS_DIR / 'model_ep_low.joblib')
print('Long model saved.')
del long_model; gc.collect()

# SHORT
df_train_s = df_sig_short[df_sig_short.index <= TRAIN_END]
df_test_s  = df_sig_short[df_sig_short.index >  TRAIN_END]
X_train_s  = df_train_s[feature_cols].ffill().fillna(0)
y_train_s  = df_train_s['label_short']
X_test_s   = df_test_s[feature_cols].ffill().fillna(0)
y_test_s   = df_test_s['label_short']
print(f'SHORT train: {len(X_train_s):,}  pos_rate={y_train_s.mean():.4f}')
short_model, short_cv = train_ep_model(X_train_s, y_train_s, 'SHORT')
proba_s = short_model.predict_proba(X_test_s)[:, 1]
print(f'SHORT test AUC: {roc_auc_score(y_test_s, proba_s):.4f}')
joblib.dump({'model': short_model, 'feature_cols': feature_cols, 'train_end': TRAIN_END, 'cv_auc': short_cv, 'direction': 'short'}, MODELS_DIR / 'model_ep_high.joblib')
print('Short model saved.')
del short_model; gc.collect()
""", 'c7')

# ── Cell 9: evaluation both ────────────────────────────────────────────────
nb['cells'][9] = code(
r"""bundle_l = joblib.load(MODELS_DIR / 'model_ep_low.joblib')
bundle_s = joblib.load(MODELS_DIR / 'model_ep_high.joblib')
proba_l  = bundle_l['model'].predict_proba(X_test_l)[:, 1]
proba_s  = bundle_s['model'].predict_proba(X_test_s)[:, 1]

results_long  = pd.DataFrame({'actual': y_test_l.values, 'proba': proba_l, 'pair': df_test_l['pair'].values}, index=y_test_l.index)
results_short = pd.DataFrame({'actual': y_test_s.values, 'proba': proba_s, 'pair': df_test_s['pair'].values}, index=y_test_s.index)

for label, results, y_test in [('LONG', results_long, y_test_l), ('SHORT', results_short, y_test_s)]:
    auc  = roc_auc_score(y_test, results['proba'])
    base = y_test.mean()
    print(f'--- {label} ---  AUC={auc:.4f}  base={base:.4f}  N={len(y_test):,}')
    print(f'{"Thresh":>8} {"N":>8} {"% kept":>7} {"Precision":>10} {"Recall":>8} {"Lift":>7}')
    print('-' * 55)
    for t in [0.05, 0.1, 0.2, 0.3, 0.5]:
        mask = results['proba'] >= t
        if mask.sum() < 10: continue
        s      = results[mask]
        prec   = s['actual'].mean()
        recall = s['actual'].sum() / results['actual'].sum()
        print(f'{t:>8.2f} {mask.sum():>8,} {mask.mean():>7.1%} {prec:>10.4f} {recall:>8.3f} {prec/base:>7.2f}x')
    print()
""", 'c9')

# ── Cell 11: feature importance both ──────────────────────────────────────
nb['cells'][11] = code(
r"""import matplotlib.pyplot as plt

for fname, label in [('model_ep_low.joblib', 'Long — Episode Low'), ('model_ep_high.joblib', 'Short — Episode High')]:
    bundle     = joblib.load(MODELS_DIR / fname)
    importance = pd.Series(bundle['model'].feature_importances_, index=feature_cols)
    importance = importance.sort_values(ascending=True).tail(25)
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor('#080c14'); ax.set_facecolor('#080c14')
    ax.barh(importance.index, importance.values, color='#4fc3f7', alpha=0.8)
    ax.tick_params(colors='white', labelsize=8)
    ax.set_title(f'Feature Importance — {label}', color='white', fontsize=12)
    for spine in ax.spines.values(): spine.set_edgecolor('#1a2332')
    plt.tight_layout(); plt.show()
""", 'c11')

# ── Cell 13: EV sim both ───────────────────────────────────────────────────
nb['cells'][13] = code(
r"""def ev_sim(results, direction):
    SPREAD_PIPS = 1.5
    sign = 1 if direction == 'long' else -1
    print(f'EV simulation ({direction.upper()}): TP=first close {"above" if direction=="long" else "below"} both MAs, timeout=T+120H')
    print(f'{"Thresh":>8} {"N":>7} {"WR":>7} {"Avg win":>9} {"Avg loss":>9} {"EV":>8} {"Sharpe":>8}')
    print('-' * 65)
    for t in [0.05, 0.1, 0.2, 0.3, 0.5]:
        mask = results['proba'] >= t
        if mask.sum() < 10: continue
        sigs = results[mask]
        pnls = []
        for pair in PAIRS:
            pip  = PIP_SIZE[pair]
            ohlc = pd.read_parquet(PROCESSED_DIR / f'{pair}_1H.parquet')[['close']]
            ohlc.index = pd.to_datetime(ohlc.index)
            c_arr   = ohlc['close'].values
            ma50_s  = pd.Series(c_arr).rolling(50).mean()
            ma200_s = pd.Series(c_arr).rolling(200).mean()
            filt = sigs[sigs['pair'] == pair]
            for ts, row in filt.iterrows():
                if ts not in ohlc.index: continue
                iloc  = ohlc.index.get_loc(ts)
                entry = ohlc['close'].iloc[iloc]
                pnl   = None
                for k in range(1, 121):
                    if iloc + k >= len(ohlc): break
                    c_k     = ohlc['close'].iloc[iloc+k]
                    ma50_k  = ma50_s.iloc[iloc+k]
                    ma200_k = ma200_s.iloc[iloc+k]
                    if pd.notna(ma50_k) and pd.notna(ma200_k):
                        if direction == 'long'  and c_k > ma50_k and c_k > ma200_k:
                            pnl = (c_k - entry) / pip - SPREAD_PIPS; break
                        elif direction == 'short' and c_k < ma50_k and c_k < ma200_k:
                            pnl = (entry - c_k) / pip - SPREAD_PIPS; break
                if pnl is None:
                    c_end = ohlc['close'].iloc[min(iloc+120, len(ohlc)-1)]
                    pnl   = sign * (c_end - entry) / pip - SPREAD_PIPS
                pnls.append(pnl)
        pnls   = np.array(pnls)
        wins   = pnls[pnls > 0]; losses = pnls[pnls <= 0]
        ev     = pnls.mean()
        sharpe = ev / pnls.std() * np.sqrt(252) if pnls.std() > 0 else 0
        print(f'{t:>8.2f} {len(pnls):>7,} {(pnls>0).mean():>7.1%} '
              f'{wins.mean() if len(wins) else 0:>9.1f} '
              f'{losses.mean() if len(losses) else 0:>9.1f} '
              f'{ev:>8.2f} {sharpe:>8.3f}')

ev_sim(results_long,  'long')
print()
ev_sim(results_short, 'short')
""", 'c13')

# ── Append analysis cells ──────────────────────────────────────────────────
nb['cells'].append(md('## 7. Hour Distribution', 'md7'))
nb['cells'].append(code(
r"""for results, label in [(results_long, 'LONG'), (results_short, 'SHORT')]:
    sub = results[results['proba'] >= 0.2].copy()
    sub['hour'] = sub.index.hour
    by_hour = sub.groupby('hour').agg(n=('actual','count'), hit_rate=('actual','mean'))
    total = by_hour['n'].sum()
    print(f'Hour distribution ({label}, proba>=0.2), total={int(total):,}')
    print(f'{"Hour":>5} {"N":>6} {"%":>6} {"Hit%":>7}')
    print('-'*30)
    for hour, row in by_hour.iterrows():
        bar = '#' * int(row['n'] / total * 30)
        print(f'{hour:>5} {int(row["n"]):>6} {row["n"]/total:>6.1%} {row["hit_rate"]:>7.1%}  {bar}')
    print()
""", 'c_hour'))

nb['cells'].append(md('## 8. Per-Pair Breakdown', 'md8'))
nb['cells'].append(code(
r"""for results, label in [(results_long, 'LONG'), (results_short, 'SHORT')]:
    sub = results[results['proba'] >= 0.2]
    print(f'{label} per-pair (proba>=0.2):')
    print(f'{"Pair":>10} {"N":>6} {"Hit%":>7} {"AUC":>8}')
    print('-'*35)
    for pair, g in sub.groupby('pair'):
        if len(g) < 20 or g['actual'].nunique() < 2: continue
        print(f'{pair:>10} {len(g):>6,} {g["actual"].mean():>7.1%} {roc_auc_score(g["actual"], g["proba"]):>8.4f}')
    print()
""", 'c_pair'))

nb['cells'].append(md('## 9. Entry Quality — How Close to Actual Low/High?', 'md9'))
nb['cells'].append(code(
r"""for results, label, direction in [(results_long, 'LONG', 'long'), (results_short, 'SHORT', 'short')]:
    r2 = results.copy()
    r2['gap']   = r2.index.to_series().diff().dt.total_seconds() > 7200
    r2['ep_id'] = r2['gap'].cumsum()
    early, late, bars_diffs, slippages = 0, 0, [], []
    for pair in PAIRS:
        pip  = PIP_SIZE[pair]
        ohlc = pd.read_parquet(PROCESSED_DIR / f'{pair}_1H.parquet')[['close','high','low']]
        ohlc.index = pd.to_datetime(ohlc.index)
        pr = r2[r2['pair'] == pair]
        for ep_id, ep in pr.groupby('ep_id'):
            entry_ts  = ep['proba'].idxmax()
            actual_ep = ep[ep['actual'] == 1]
            if len(actual_ep) == 0: continue
            actual_ts = actual_ep.index[0]
            if entry_ts not in ohlc.index or actual_ts not in ohlc.index: continue
            ep_price  = ohlc['close'].loc[entry_ts]
            act_price = ohlc['low'].loc[actual_ts] if direction == 'long' else ohlc['high'].loc[actual_ts]
            bars_diff = (entry_ts - actual_ts).total_seconds() / 3600
            slippage  = (ep_price - act_price) / pip if direction == 'long' else (act_price - ep_price) / pip
            bars_diffs.append(bars_diff); slippages.append(slippage)
            if bars_diff < 0: early += 1
            else: late += 1
    total = early + late
    print(f'{label} entry quality ({total} episodes):')
    print(f'  Entered before actual extreme: {early} ({early/total:.1%})')
    print(f'  Entered after  actual extreme: {late}  ({late/total:.1%})')
    print(f'  Bars diff  — mean={np.mean(bars_diffs):.1f}H  median={np.median(bars_diffs):.1f}H')
    print(f'  Slippage   — mean={np.mean(slippages):.1f}p   median={np.median(slippages):.1f}p')
    print()
""", 'c_quality'))

with open('notebooks_9/02_model_training.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)
print(f'Done. Total cells: {len(nb["cells"])}')
