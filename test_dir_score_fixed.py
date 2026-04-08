"""
Directional scoring backtest — FIXED version.
Computes cross-pair features from parquet (same logic as live pipeline).
Evaluates directional accuracy using actual t+8h close price move,
NOT mfe_long > mfe_short (which was a flawed proxy metric).
"""

import pandas as pd, numpy as np
from pathlib import Path

TRAIN_END    = '2024-06-30'
FEATURES_DIR = Path('backend/data/features_9')
EXIT_H       = 8   # fixed exit horizon in hours

PAIRS = [
    'EURUSD','GBPUSD','USDJPY','USDCHF','AUDUSD','USDCAD','NZDUSD',
    'EURJPY','GBPJPY','EURGBP','EURAUD','AUDJPY','CADJPY','CHFJPY','AUDNZD',
]
CURRENCY_SIGN = {
    'EURUSD':{'EUR':+1,'USD':-1},'GBPUSD':{'GBP':+1,'USD':-1},
    'USDJPY':{'USD':+1,'JPY':-1},'USDCHF':{'USD':+1,'CHF':-1},
    'AUDUSD':{'AUD':+1,'USD':-1},'USDCAD':{'USD':+1,'CAD':-1},
    'NZDUSD':{'NZD':+1,'USD':-1},'EURJPY':{'EUR':+1,'JPY':-1},
    'GBPJPY':{'GBP':+1,'JPY':-1},'EURGBP':{'EUR':+1,'GBP':-1},
    'EURAUD':{'EUR':+1,'AUD':-1},'AUDJPY':{'AUD':+1,'JPY':-1},
    'CADJPY':{'CAD':+1,'JPY':-1},'CHFJPY':{'CHF':+1,'JPY':-1},
    'AUDNZD':{'AUD':+1,'NZD':-1},
}

SCORING_RULES = [
    ('range_return_ratio', +1, 3.0),
    ('noise_to_signal',    -1, 2.0),
    ('autocorr_1',         +1, 2.0),
    ('fractal_dim',        -1, 1.5),
    ('vr_z5',              +1, 1.5),
    ('epps_1m_5m',         -1, 1.5),
    ('vol_clustering_ac1', -1, 1.0),
    ('vr_10',              +1, 1.0),
    ('runs_z',             -1, 1.0),
    ('rv_zscore_24',       +1, 1.0),
]
SCORE_MAX    = sum(w for _, _, w in SCORING_RULES)
SCORE_THRESH = 0.7 * SCORE_MAX


# ── Load all parquets ─────────────────────────────────────────────────────────
print('Loading parquet data...')
parquet_data = {}
for pair in PAIRS:
    path = FEATURES_DIR / f'{pair}_features.parquet'
    if not path.exists(): continue
    df = pd.read_parquet(path)
    df = df[~df.index.duplicated(keep='first')].sort_index()
    parquet_data[pair] = df

print(f'  Loaded {len(parquet_data)} pairs, {sum(len(v) for v in parquet_data.values()):,} total rows')


# ── Reconstruct 1h log returns for each pair ─────────────────────────────────
# Each pair's return is stored as peer_X_ret_1h in another pair's parquet
print('Reconstructing 1h return series...')
ret_1h = {}
for pair in PAIRS:
    found = False
    # Try to find this pair's return as a peer column in any other pair's parquet
    pair_key = pair.lower()
    for host_pair, df_host in parquet_data.items():
        col = f'peer_{pair_key}_ret_1h'
        if col in df_host.columns:
            ret_1h[pair] = df_host[col].copy()
            found = True
            break
    if not found:
        # EURAUD: reconstruct from EURUSD - AUDUSD returns
        if pair == 'EURAUD' and 'AUDUSD' in parquet_data and 'EURUSD' in parquet_data:
            eu = parquet_data['AUDUSD'].get('peer_eurusd_ret_1h')
            au = parquet_data['EURUSD'].get('peer_audusd_ret_1h')
            if eu is not None and au is not None:
                idx = eu.index.union(au.index)
                ret_1h[pair] = eu.reindex(idx) - au.reindex(idx)
                found = True
        if not found:
            print(f'  WARNING: could not find return for {pair}')

print(f'  Returns available for: {sorted(ret_1h.keys())}')


# ── Reconstruct close price from cumulative returns ───────────────────────────
# We need close price for corr/beta/relstr computation
# Start with 1.0 and cumulate log returns
print('Reconstructing close prices...')
close_1h = {}
for pair, ret in ret_1h.items():
    ret_clean = ret.fillna(0)
    close_1h[pair] = np.exp(ret_clean.cumsum())  # relative price index


# ── Compute cross-pair features (identical to live pipeline) ─────────────────
def compute_cross_pair_features(close_1h_all, ret_1h_all):
    returns_df = pd.DataFrame({p: r for p, r in ret_1h_all.items()})

    currencies = ['EUR','USD','GBP','JPY','AUD','NZD','CAD','CHF']
    csi = {}
    for ccy in currencies:
        comps = [CURRENCY_SIGN[p][ccy] * returns_df[p]
                 for p in PAIRS if ccy in CURRENCY_SIGN.get(p,{}) and p in returns_df]
        if comps:
            csi[f'csi_{ccy.lower()}'] = pd.concat(comps, axis=1).mean(axis=1)
    csi_df = pd.DataFrame(csi)
    csi_rolling = {}
    for col in csi_df.columns:
        csi_rolling[f'{col}_24h'] = csi_df[col].rolling(24, min_periods=8).sum()
        csi_rolling[f'{col}_72h'] = csi_df[col].rolling(72, min_periods=24).sum()
    csi_rolling_df = pd.DataFrame(csi_rolling)

    result = {}
    for pair in PAIRS:
        if pair not in returns_df.columns:
            continue
        r      = returns_df[pair]
        c_pair = close_1h_all[pair]
        cols   = {}
        for peer in [p for p in PAIRS if p != pair]:
            if peer not in returns_df.columns:
                continue
            p_ret  = returns_df[peer]
            c_peer = close_1h_all[peer]
            sl     = peer.lower()
            for w, lbl in [(24,'24h'),(72,'3d'),(168,'1w')]:
                cols[f'corr_{sl}_{lbl}'] = r.rolling(w, min_periods=w//2).corr(p_ret)
            cols[f'corr_regime_{sl}'] = cols[f'corr_{sl}_24h'] - cols[f'corr_{sl}_1w']
            for w, lbl in [(24,'24h'),(168,'1w')]:
                cov = r.rolling(w, min_periods=w//2).cov(p_ret)
                var = p_ret.rolling(w, min_periods=w//2).var().clip(lower=1e-12)
                cols[f'beta_{sl}_{lbl}'] = cov / var
            cols[f'relstr_{sl}_1h']  = r - p_ret
            cols[f'relstr_{sl}_4h']  = np.log(c_pair/c_pair.shift(4))  - np.log(c_peer/c_peer.shift(4))
            cols[f'relstr_{sl}_24h'] = np.log(c_pair/c_pair.shift(24)) - np.log(c_peer/c_peer.shift(24))
            cols[f'peer_{sl}_ret_1h']  = p_ret
            cols[f'peer_{sl}_ret_4h']  = np.log(c_peer/c_peer.shift(4))
            cols[f'peer_{sl}_ret_24h'] = np.log(c_peer/c_peer.shift(24))
        for col in csi_df.columns:
            cols[col]          = csi_df[col]
            cols[f'{col}_24h'] = csi_rolling_df[f'{col}_24h']
            cols[f'{col}_72h'] = csi_rolling_df[f'{col}_72h']
        result[pair] = pd.DataFrame(cols, index=r.index).astype(np.float32)
    return result


print('Computing cross-pair features...')
cross_features = compute_cross_pair_features(close_1h, ret_1h)



# ── Train-set medians for scoring ─────────────────────────────────────────────
df_all_train = pd.concat([v[v.index <= TRAIN_END] for v in parquet_data.values()])
TRAIN_MEDIANS = {}
for feat, _, _ in SCORING_RULES:
    if feat in df_all_train.columns:
        TRAIN_MEDIANS[feat] = float(df_all_train[feat].median())
print(f'Train medians loaded for {len(TRAIN_MEDIANS)}/{len(SCORING_RULES)} scoring features')


# ── Per-pair scoring and signal collection ────────────────────────────────────
# Direction = actual t+8h close price move, NOT mfe_long > mfe_short
print('\nProcessing pairs...')
all_results = []

for pair in PAIRS:
    if pair not in parquet_data:
        continue
    if pair not in close_1h:
        continue

    df_micro = parquet_data[pair]
    df_cross = cross_features.get(pair)

    if df_cross is not None:
        df_full = df_micro.join(df_cross.reindex(df_micro.index), how='left', rsuffix='_cp')
    else:
        df_full = df_micro.copy()

    df_test = df_full[df_full.index > TRAIN_END].copy()
    if len(df_test) < 100:
        continue

    # Compute scores for all bars
    scores = np.zeros(len(df_test))
    for feat, direction, weight in SCORING_RULES:
        if feat not in df_test.columns:
            continue
        thresh = TRAIN_MEDIANS.get(feat, np.nan)
        if np.isnan(thresh):
            continue
        val  = df_test[feat].values
        vote = np.where(val > thresh, +1, -1)
        scores += direction * vote * weight
    df_test['score'] = scores

    # Filter to signal bars only
    df_cands = df_test[np.abs(df_test['score']) >= SCORE_THRESH].copy()
    if len(df_cands) < 10:
        continue

    # Compute actual t+8h direction from reconstructed close prices
    closes = close_1h[pair]
    rows = []
    for ts, row in df_cands.iterrows():
        try:
            pos = closes.index.get_loc(ts)
        except KeyError:
            continue
        exit_pos = pos + EXIT_H
        if exit_pos >= len(closes):
            continue
        move = closes.iloc[exit_pos] - closes.iloc[pos]
        signal = 1 if row['score'] >= SCORE_THRESH else -1
        correct = (signal == 1 and move > 0) or (signal == -1 and move < 0)
        rows.append({
            'ts':      ts,
            'pair':    pair,
            'signal':  signal,
            'score':   row['score'],
            'move':    move * signal,   # positive = correct direction
            'correct': correct,
            'year':    ts.year,
        })

    if rows:
        all_results.append(pd.DataFrame(rows).set_index('ts'))
    print(f'  {pair}: {len(rows):,} signals')

df_all = pd.concat(all_results).sort_index()
months = (df_all.index.max() - df_all.index.min()).days / 30
n_long  = (df_all['signal'] == 1).sum()
n_short = (df_all['signal'] == -1).sum()
acc     = df_all['correct'].mean()
print(f'\nTotal signals: {len(df_all):,} over {months:.0f} months | L/S: {n_long}/{n_short} | Acc: {acc:.1%}')


# ── Results ───────────────────────────────────────────────────────────────────
print(f'\n{"="*80}')
print(f'  DIRECTIONAL SCORING BACKTEST — actual t+{EXIT_H}h price direction')
print(f'  Threshold: {SCORE_THRESH:.1f}/{SCORE_MAX:.1f} (0.7x) | Train medians pre-{TRAIN_END}')
print(f'{"="*80}')

print(f'\n--- Score threshold sweep ---')
print(f'  {"Thresh":>6} {"N":>7} {"L/S":>10} {"Acc":>7} {"Avg_move":>10} {"Per_month":>10}')
print(f'  {"-"*55}')
for thresh_frac in [0.5, 0.6, 0.7, 0.8, 0.9]:
    tv = thresh_frac * SCORE_MAX
    sub = df_all[np.abs(df_all['score']) >= tv]
    if len(sub) < 50: continue
    nl = (sub['signal']==1).sum(); ns = (sub['signal']==-1).sum()
    print(f'  {thresh_frac:.1f}x   {len(sub):>7,} {nl}/{ns:<7} {sub["correct"].mean():>7.1%} {sub["move"].mean():>10.3f} {len(sub)/months:>10.0f}')

print(f'\n--- Year-by-year at 0.7x ---')
print(f'  {"Year":<6} {"N":>6} {"L/S":>8} {"Acc":>7} {"Avg_move":>10}')
print(f'  {"-"*42}')
tv = 0.7 * SCORE_MAX
sub = df_all[np.abs(df_all['score']) >= tv]
for y in sorted(sub['year'].unique()):
    s = sub[sub['year']==y]
    nl = (s['signal']==1).sum(); ns = (s['signal']==-1).sum()
    print(f'  {y:<6} {len(s):>6,} {nl}/{ns:<6} {s["correct"].mean():>7.1%} {s["move"].mean():>10.3f}')

print(f'\n--- Per-pair at 0.7x ---')
print(f'  {"Pair":<10} {"N":>6} {"L/S":>8} {"Acc":>7} {"Avg_move":>10}')
print(f'  {"-"*46}')
for pair in sorted(sub['pair'].unique()):
    s = sub[sub['pair']==pair]
    nl = (s['signal']==1).sum(); ns = (s['signal']==-1).sum()
    print(f'  {pair:<10} {len(s):>6,} {nl}/{ns:<6} {s["correct"].mean():>7.1%} {s["move"].mean():>10.5f}')

print(f'\n--- Score bucket accuracy at 0.7x ---')
print(f'  {"Score_abs":<12} {"N":>6} {"Acc":>7} {"Avg_move":>10}')
print(f'  {"-"*38}')
sub['score_abs'] = sub['score'].abs()
for lo, hi in [(10.85,11.5),(11.5,12.5),(12.5,13.5),(13.5,15.5)]:
    s = sub[(sub['score_abs']>=lo)&(sub['score_abs']<hi)]
    if len(s) < 5: continue
    print(f'  {lo:.1f}-{hi:.1f}      {len(s):>6,} {s["correct"].mean():>7.1%} {s["move"].mean():>10.5f}')
