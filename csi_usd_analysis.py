import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import sys
sys.path.insert(0, 'backend')
from live_features_extra import compute_momentum_calendar_features
from features import PIP_SIZE

PROCESSED_DIR = Path('backend/data/processed')
FEATURES_DIR  = Path('backend/data/features_9')
TRAIN_END     = '2024-06-30'

PAIRS = ['EURUSD','GBPUSD','USDJPY','USDCHF','AUDUSD','USDCAD','NZDUSD',
         'EURJPY','GBPJPY','EURGBP','EURAUD','AUDJPY','CADJPY','CHFJPY','AUDNZD']

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
pip_map = {p: 0.01 if 'JPY' in p else 0.0001 for p in PAIRS}

# Load 1H closes
closes = {}
for pair in PAIRS:
    p = PROCESSED_DIR / f'{pair}_1H.parquet'
    if p.exists():
        closes[pair] = pd.read_parquet(p)['close'].sort_index()

returns_df = pd.DataFrame({p: np.log(c/c.shift(1)) for p,c in closes.items()})

# Compute cross-pair features
def compute_cross_features(pair):
    r = returns_df[pair]
    c_pair = closes[pair]
    cols = {}
    currencies = ['EUR','USD','GBP','JPY','AUD','NZD','CAD','CHF']
    for ccy in currencies:
        comps = [CURRENCY_SIGN[p][ccy] * returns_df[p]
                 for p in PAIRS if ccy in CURRENCY_SIGN.get(p,{}) and p in returns_df]
        if comps:
            csi = pd.concat(comps, axis=1).mean(axis=1)
            cols[f'csi_{ccy.lower()}']     = csi
            cols[f'csi_{ccy.lower()}_24h'] = csi.rolling(24,  min_periods=8).sum()
            cols[f'csi_{ccy.lower()}_72h'] = csi.rolling(72, min_periods=24).sum()
    for peer in [p for p in PAIRS if p != pair]:
        if peer not in returns_df.columns: continue
        p_ret  = returns_df[peer]
        c_peer = closes[peer]
        sl = peer.lower()
        for w, lbl in [(24,'24h'),(72,'3d'),(168,'1w')]:
            cols[f'corr_{sl}_{lbl}'] = r.rolling(w, min_periods=w//2).corr(p_ret)
        cols[f'corr_regime_{sl}'] = cols[f'corr_{sl}_24h'] - cols[f'corr_{sl}_1w']
        for w, lbl in [(24,'24h'),(168,'1w')]:
            cov = r.rolling(w, min_periods=w//2).cov(p_ret)
            var = p_ret.rolling(w, min_periods=w//2).var().clip(lower=1e-12)
            cols[f'beta_{sl}_{lbl}'] = cov / var
        cols[f'relstr_{sl}_1h']    = r - p_ret
        cols[f'relstr_{sl}_4h']    = np.log(c_pair/c_pair.shift(4))  - np.log(c_peer/c_peer.shift(4))
        cols[f'relstr_{sl}_24h']   = np.log(c_pair/c_pair.shift(24)) - np.log(c_peer/c_peer.shift(24))
        cols[f'peer_{sl}_ret_1h']  = p_ret
        cols[f'peer_{sl}_ret_4h']  = np.log(c_peer/c_peer.shift(4))
        cols[f'peer_{sl}_ret_24h'] = np.log(c_peer/c_peer.shift(24))
    return pd.DataFrame(cols, index=r.index).astype(np.float32)

# Load mfe_q50 model
bundle    = joblib.load('backend/models_9/mfe_q50/model_1H_Q50.joblib')
feat_cols = bundle['feature_cols']
print(f'mfe_q50: {len(feat_cols)} features, {bundle["n_iters"]} iters')

# Score all pairs
all_cands = []
for pair in PAIRS:
    feat_path = FEATURES_DIR / f'{pair}_features.parquet'
    if not feat_path.exists(): continue
    df_base  = pd.read_parquet(feat_path)
    df_extra = compute_momentum_calendar_features(
        closes[pair].to_frame('close').assign(open=closes[pair], high=closes[pair], low=closes[pair], volume=1),
        pip_map[pair])
    df_cross = compute_cross_features(pair)
    new_extra = df_extra.reindex(df_base.index)[[c for c in df_extra.columns if c not in df_base.columns]]
    new_cross = df_cross.reindex(df_base.index)[[c for c in df_cross.columns if c not in df_base.columns]]
    df_base  = df_base.join(new_extra, how='left')
    df_base  = df_base.join(new_cross, how='left')
    for c in feat_cols:
        if c not in df_base.columns: df_base[c] = 0.0
    df_test = df_base[df_base.index > TRAIN_END].copy()
    X = df_test[feat_cols].ffill().fillna(0)
    df_test['q50_mfe'] = bundle['model'].predict(X)
    df_test['pair']    = pair
    all_cands.append(df_test[['pair','q50_mfe']])
    print(f'  {pair}: {(df_test["q50_mfe"]>=70).sum()} candidates')

df_all   = pd.concat(all_cands).sort_index()
df_cands = df_all[df_all['q50_mfe'] >= 70.0].copy()
print(f'\nTotal candidates (mfe_q50 >= 70): {len(df_cands)}')
print(f'Pair distribution:')
print(df_cands['pair'].value_counts().to_string())

# Add csi_usd
usd_comps   = [CURRENCY_SIGN[p]['USD'] * returns_df[p]
               for p in PAIRS if 'USD' in CURRENCY_SIGN.get(p,{}) and p in returns_df]
csi_usd     = pd.concat(usd_comps, axis=1).mean(axis=1)
csi_usd_24h = csi_usd.rolling(24, min_periods=8).sum()
csi_usd_72h = csi_usd.rolling(72, min_periods=24).sum()
csi_df = pd.DataFrame({'csi_usd': csi_usd, 'csi_usd_24h': csi_usd_24h, 'csi_usd_72h': csi_usd_72h})
df_cands = df_cands.join(csi_df, how='left')

# Forward returns
rows = []
for pair in PAIRS:
    if pair not in closes: continue
    c = closes[pair]; pip = pip_map[pair]
    rows.append(pd.DataFrame({
        'pair':    pair,
        'fwd_8h':  (c.shift(-8)  - c) / pip,
        'fwd_24h': (c.shift(-24) - c) / pip,
        'fwd_72h': (c.shift(-72) - c) / pip,
    }))
fwd_df    = pd.concat(rows).reset_index().rename(columns={'index':'datetime'})
cands_r   = df_cands.reset_index().rename(columns={df_cands.index.name or 'index': 'datetime'})
merged    = cands_r.merge(fwd_df, on=['datetime','pair'], how='left').dropna(subset=['csi_usd_24h','fwd_8h'])

USD_SIGN      = {p: CURRENCY_SIGN[p].get('USD', 0) for p in PAIRS}
merged['usd_sign'] = merged['pair'].map(USD_SIGN)
merged['usd_pred'] = merged['csi_usd_24h'] * merged['usd_sign']
merged['usd_dir']  = np.where(merged['usd_sign']==0, np.nan,
                     np.where(merged['usd_pred'] > 0, 1.0, -1.0))

has_usd = merged[merged['usd_sign'] != 0].copy()
no_usd  = merged[merged['usd_sign'] == 0].copy()

print(f'\nUSD-leg N={len(has_usd)}, Cross N={len(no_usd)}')

print(f'\n--- USD-leg pairs: csi_usd_24h accuracy ---')
print(f'  {"Pair":<10} {"N":>6}  {"acc_8h":>7}  {"acc_24h":>8}  {"acc_72h":>8}')
for pair in sorted(has_usd['pair'].unique()):
    sub = has_usd[has_usd['pair']==pair]
    a8  = (np.sign(sub['usd_dir']) == np.sign(sub['fwd_8h'])).mean()
    a24 = (np.sign(sub['usd_dir']) == np.sign(sub['fwd_24h'])).mean()
    a72 = (np.sign(sub['usd_dir']) == np.sign(sub['fwd_72h'])).mean()
    print(f'  {pair:<10} {len(sub):>6}  {a8:>7.1%}  {a24:>8.1%}  {a72:>8.1%}')

print(f'\n--- csi_usd_24h sign vs fwd (all pairs) ---')
pos = merged[merged['csi_usd_24h'] > 0]
neg = merged[merged['csi_usd_24h'] < 0]
print(f'  USD strong (>0): N={len(pos)}  avg_fwd8h={pos["fwd_8h"].mean():+.1f}p  pct_up={(pos["fwd_8h"]>0).mean():.1%}')
print(f'  USD weak   (<0): N={len(neg)}  avg_fwd8h={neg["fwd_8h"].mean():+.1f}p  pct_up={(neg["fwd_8h"]>0).mean():.1%}')

print(f'\n--- csi_usd_24h quartile vs fwd_8h ---')
merged['q_usd'] = pd.qcut(merged['csi_usd_24h'], 4, labels=['Q1_weak','Q2','Q3','Q4_strong'])
for qt in ['Q1_weak','Q2','Q3','Q4_strong']:
    sub = merged[merged['q_usd']==qt]
    print(f'  {qt:<12}: N={len(sub):>5}  avg_fwd8h={sub["fwd_8h"].mean():+.1f}p  pct_up={(sub["fwd_8h"]>0).mean():.1%}  avg_fwd24h={sub["fwd_24h"].mean():+.1f}p')

print(f'\n--- per pair: USD strong vs weak fwd_8h ---')
print(f'  {"Pair":<10} {"N":>5}  {"USD_str_avg":>12}  {"USD_wk_avg":>11}  {"delta":>7}  {"str_acc":>8}  {"wk_acc":>7}')
for pair in sorted(PAIRS):
    sub = merged[merged['pair']==pair]
    if len(sub) < 10: continue
    sp = sub[sub['csi_usd_24h']>0];  sw = sub[sub['csi_usd_24h']<0]
    if len(sp)<3 or len(sw)<3: continue
    print(f'  {pair:<10} {len(sub):>5}  {sp["fwd_8h"].mean():>+12.1f}p  {sw["fwd_8h"].mean():>+11.1f}p  {sp["fwd_8h"].mean()-sw["fwd_8h"].mean():>+7.1f}p  {(sp["fwd_8h"]>0).mean():>8.1%}  {(sw["fwd_8h"]>0).mean():>7.1%}')
