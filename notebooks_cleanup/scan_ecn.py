import pandas as pd, numpy as np, joblib, warnings; warnings.filterwarnings('ignore')
from pathlib import Path
from collections import defaultdict

PAIRS = ['AUDJPY','AUDNZD','AUDUSD','CADJPY','CHFJPY','EURAUD','EURGBP','EURJPY',
         'EURUSD','GBPJPY','GBPUSD','NZDUSD','USDCAD','USDCHF','USDJPY']
PIP = {'AUDJPY':0.01,'AUDNZD':0.0001,'AUDUSD':0.0001,'CADJPY':0.01,'CHFJPY':0.01,
       'EURAUD':0.0001,'EURGBP':0.0001,'EURJPY':0.01,'EURUSD':0.0001,'GBPJPY':0.01,
       'GBPUSD':0.0001,'NZDUSD':0.0001,'USDCAD':0.0001,'USDCHF':0.0001,'USDJPY':0.01}
MAX_1H = {'AUDJPY':250,'AUDNZD':80,'AUDUSD':120,'CADJPY':250,'CHFJPY':250,'EURAUD':150,
          'EURGBP':100,'EURJPY':300,'EURUSD':150,'GBPJPY':300,'GBPUSD':150,'NZDUSD':100,
          'USDCAD':150,'USDCHF':120,'USDJPY':300}

# ECN spreads during London/NY hours
SPREAD = {'EURUSD':0.2,'GBPUSD':0.4,'USDJPY':0.3,'USDCHF':0.4,'USDCAD':0.4,
          'AUDUSD':0.3,'NZDUSD':0.5,'EURGBP':0.5,'EURJPY':0.5,'GBPJPY':0.8,
          'AUDJPY':0.8,'CADJPY':1.0,'CHFJPY':1.0,'EURAUD':0.8,'AUDNZD':1.0}

F6 = Path('backend/data/features_6')
F8 = Path('backend/data/features_8')

m = joblib.load('backend/models_9/mfe_q50/model_1H_Q50.joblib')
model = m['model']; feat_cols = m['feature_cols']
closes_all = {p: pd.read_parquet(f'backend/data/processed/{p}_1H.parquet')['close'] for p in PAIRS}

# Currency strength
CUR_PAIRS = {
    'USD': [('EURUSD',-1),('GBPUSD',-1),('AUDUSD',-1),('NZDUSD',-1),('USDCAD',+1),('USDCHF',+1),('USDJPY',+1)],
    'EUR': [('EURUSD',+1),('EURGBP',+1),('EURAUD',+1),('EURJPY',+1)],
    'GBP': [('GBPUSD',+1),('EURGBP',-1),('GBPJPY',+1)],
    'AUD': [('AUDUSD',+1),('AUDJPY',+1),('AUDNZD',+1),('EURAUD',-1)],
    'CAD': [('USDCAD',-1),('CADJPY',+1)],
    'CHF': [('USDCHF',-1),('CHFJPY',+1)],
    'JPY': [('USDJPY',-1),('EURJPY',-1),('GBPJPY',-1),('AUDJPY',-1),('CADJPY',-1),('CHFJPY',-1)],
    'NZD': [('NZDUSD',+1),('AUDNZD',-1)],
}
cur_strength = {}
for cur, ps in CUR_PAIRS.items():
    parts = [sign * np.log(closes_all[p]).diff() for p, sign in ps]
    avg = pd.concat(parts, axis=1).mean(axis=1)
    cur_strength[f'{cur}_24h'] = avg.rolling(24).sum()
    cur_strength[f'{cur}_6h']  = avg.rolling(6).sum()
cur_df = pd.DataFrame(cur_strength)

PAIR_CURRENCIES = {
    'AUDJPY':('AUD','JPY'), 'AUDNZD':('AUD','NZD'), 'AUDUSD':('AUD','USD'),
    'CADJPY':('CAD','JPY'), 'CHFJPY':('CHF','JPY'), 'EURAUD':('EUR','AUD'),
    'EURGBP':('EUR','GBP'), 'EURJPY':('EUR','JPY'), 'EURUSD':('EUR','USD'),
    'GBPJPY':('GBP','JPY'), 'GBPUSD':('GBP','USD'), 'NZDUSD':('NZD','USD'),
    'USDCAD':('USD','CAD'), 'USDCHF':('USD','CHF'), 'USDJPY':('USD','JPY'),
}

MA_PERIODS = [8, 24, 72, 168, 480]

strat_names = [
    'sys2_strict',
    'sys2+mfe85',
    'fade_ma5_only',
    'fade_ma5+mfe85',
    'fade_ma4+mfe85',
    'cur_fade+mfe85',
    'fade_ma5+cur+mfe85',
    'stacked_revert+mfe85',
]
all_pnls = {s: [] for s in strat_names}

for pair in PAIRS:
    print(f'  {pair}...', flush=True)
    pip = PIP[pair]; sp = SPREAD[pair]

    df9 = pd.read_parquet(f'backend/data/features_9/{pair}_features.parquet')
    X = pd.DataFrame({c: df9[c] if c in df9.columns else np.nan for c in feat_cols}, index=df9.index)
    pred_r = pd.Series(model.predict(X), index=df9.index).rank(pct=True)

    df6 = pd.read_parquet(F6 / f'{pair}_features.parquet')
    df8 = pd.read_parquet(F8 / f'{pair}_geometric.parquet'); df8['pair'] = pair
    df6r = df6.reset_index(); df8r = df8.reset_index(); idx = df6r.columns[0]
    df8r = df8r.drop(columns=[c for c in df8r.columns if c in df6r.columns and c not in [idx,'pair']], errors='ignore')
    df = pd.merge(df6r, df8r, on=[idx,'pair'], how='inner').set_index(idx).sort_index()
    for col in ['kyle_lambda_delta_3h','residual_12h','rv_zscore_24','realized_skew','vr_5','close_position','slope_close_3h']:
        df[f'{col}_r'] = df[col].rank(pct=True)

    close_1h = closes_all[pair]
    mas = {p: close_1h.rolling(p).mean() for p in MA_PERIODS}
    ac = pd.DataFrame({p: (close_1h > mas[p]).astype(int) for p in MA_PERIODS}).sum(axis=1)

    base, quote = PAIR_CURRENCIES[pair]
    bstr = cur_df.get(f'{base}_24h', pd.Series(0, index=cur_df.index))
    qstr = cur_df.get(f'{quote}_24h', pd.Series(0, index=cur_df.index))
    diff24 = (bstr - qstr).reindex(close_1h.index).rank(pct=True)

    fwd4h = (close_1h.shift(-4) - close_1h) / pip
    ret_1h = close_1h.diff().abs() / pip
    holiday = ~(((close_1h.index.month==12) & (close_1h.index.day.isin([24,25,26,31]))) |
                ((close_1h.index.month==1)  & (close_1h.index.day.isin([1,2]))))
    quality = (ret_1h <= MAX_1H[pair]) & holiday

    common = df.index.intersection(pred_r.index).intersection(close_1h.index).intersection(diff24.index)
    d   = df.reindex(common)
    pr  = pred_r.reindex(common)
    fwd = fwd4h.reindex(common)
    qual = quality.reindex(common).fillna(False)
    ac_c = ac.reindex(common)
    dr   = diff24.reindex(common)
    mask = qual & (common.hour >= 7) & (common.hour <= 16)
    mfe85 = pr > 0.85

    sys2_l = mask & (d['kyle_lambda_delta_3h_r']>0.70) & (d['residual_12h_r']<0.25) & \
             (d['rv_zscore_24_r']>0.85) & (d['realized_skew_r']<0.30) & (d['vr_5_r']<0.30)
    sys2_s = mask & (d['kyle_lambda_delta_3h_r']<0.30) & (d['residual_12h_r']>0.75) & \
             (d['rv_zscore_24_r']>0.85) & (d['realized_skew_r']>0.70) & (d['vr_5_r']<0.30)

    stacked_l = mask & mfe85 & (d['close_position_r']<0.25) & (d['residual_12h_r']<0.25) & (d['slope_close_3h_r']<0.30)
    stacked_s = mask & mfe85 & (d['close_position_r']>0.75) & (d['residual_12h_r']>0.75) & (d['slope_close_3h_r']>0.70)

    for strat, l, s in [
        ('sys2_strict',          sys2_l,                                  sys2_s),
        ('sys2+mfe85',           sys2_l & mfe85,                          sys2_s & mfe85),
        ('fade_ma5_only',        mask & ((5-ac_c) >= 5),                  mask & (ac_c >= 5)),
        ('fade_ma5+mfe85',       mask & mfe85 & ((5-ac_c) >= 5),          mask & mfe85 & (ac_c >= 5)),
        ('fade_ma4+mfe85',       mask & mfe85 & ((5-ac_c) >= 4),          mask & mfe85 & (ac_c >= 4)),
        ('cur_fade+mfe85',       mask & mfe85 & (dr < 0.35),              mask & mfe85 & (dr > 0.65)),
        ('fade_ma5+cur+mfe85',   mask & mfe85 & ((5-ac_c)>=5) & (dr<0.40), mask & mfe85 & (ac_c>=5) & (dr>0.60)),
        ('stacked_revert+mfe85', stacked_l,                               stacked_s),
    ]:
        pnl = pd.concat([(fwd[l] - sp).dropna(), (-fwd[s] - sp).dropna()])
        all_pnls[strat].append(pnl)

print()
print(f'ECN SPREADS — all strategies, 4H fixed exit, entry 7-16 UTC')
print(f'{"Strategy":<25} {"N":>8} {"/mo":>5} {"WR":>7} {"EV":>8} {"Sh":>7} | {"18m_N":>6} {"18mWR":>7} {"18mEV":>8}')
print('-'*95)
for strat in strat_names:
    t = pd.concat(all_pnls[strat]).sort_index()
    if len(t) < 5: continue
    nm = (t.index.max() - t.index.min()).days / 30
    sh = (t.mean() / t.std()) * np.sqrt(252*24/4) if t.std() > 0 else 0
    cut = t.index.max() - pd.DateOffset(months=18)
    r = t[t.index >= cut]
    sh_r = (r.mean() / r.std()) * np.sqrt(252*24/4) if len(r) > 5 and r.std() > 0 else 0
    print(f'{strat:<25} {len(t):>8,} {len(t)/nm:>5.0f} {(t>0).mean():>7.1%} {t.mean():>+8.2f} {sh:>+7.2f} | {len(r):>6,} {(r>0).mean():>7.1%} {r.mean():>+8.2f}')
