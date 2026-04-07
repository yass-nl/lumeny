import pandas as pd, numpy as np, warnings; warnings.filterwarnings('ignore')

PAIRS = ['AUDJPY','AUDNZD','AUDUSD','CADJPY','CHFJPY','EURAUD','EURGBP','EURJPY',
         'EURUSD','GBPJPY','GBPUSD','NZDUSD','USDCAD','USDCHF','USDJPY']
PIP = {'AUDJPY':0.01,'AUDNZD':0.0001,'AUDUSD':0.0001,'CADJPY':0.01,'CHFJPY':0.01,
       'EURAUD':0.0001,'EURGBP':0.0001,'EURJPY':0.01,'EURUSD':0.0001,'GBPJPY':0.01,
       'GBPUSD':0.0001,'NZDUSD':0.0001,'USDCAD':0.0001,'USDCHF':0.0001,'USDJPY':0.01}
SPREAD={'EURUSD':0.2,'GBPUSD':0.4,'USDJPY':0.3,'USDCHF':0.4,'USDCAD':0.4,
        'AUDUSD':0.3,'NZDUSD':0.5,'EURGBP':0.5,'EURJPY':0.5,'GBPJPY':0.8,
        'AUDJPY':0.8,'CADJPY':1.0,'CHFJPY':1.0,'EURAUD':0.8,'AUDNZD':1.0}

COOLDOWN = 24
MA25_RISE_LOOKBACK = 5
TP_LOOKBACK = 5
NO_CROSS_EXIT = 5
MAX_HOLD = 96
BARS_PER_YEAR = 252 * 24

FEATURES = [
    'kyle_lambda', 'kyle_lambda_change', 'kyle_lambda_delta_3h',
    'kyle_lambda_delta_6h', 'kyle_lambda_delta_12h', 'kyle_lambda_r2',
    'order_imbalance', 'order_imbalance_delta_3h', 'order_imbalance_delta_6h',
    'order_imbalance_intensity', 'buy_volume_frac',
    'vol_trend', 'vol_regime_5d', 'volume_ratio_6', 'volume_ratio_24',
    'accel_mean', 'momentum_shift', 'info_accel',
]

raw_signals = []  # list of dicts with pnl + all feature ranks

for pair in PAIRS:
    print(f'  {pair}...', flush=True)
    pip=PIP[pair]; sp=SPREAD[pair]
    df9=pd.read_parquet(f'backend/data/features_9/{pair}_features.parquet')
    df1h=pd.read_parquet(f'backend/data/processed/{pair}_1H.parquet')
    close=df1h['close']
    ma25=close.rolling(25).mean()
    ma50=close.rolling(50).mean()
    close_arr=close.values; ma25_arr=ma25.values; ma50_arr=ma50.values
    idx=close.index

    # rank features within pair
    feat_ranks = {}
    for feat in FEATURES:
        if feat in df9.columns:
            feat_ranks[feat] = df9[feat].reindex(idx).rank(pct=True)

    last_entry_i = -COOLDOWN - 1
    for i in range(55, len(close_arr) - MAX_HOLD):
        if np.isnan(ma25_arr[i]) or np.isnan(ma50_arr[i]): continue
        if np.isnan(ma25_arr[i - MA25_RISE_LOOKBACK]): continue
        ma25_now=ma25_arr[i]; ma50_now=ma50_arr[i]; ma25_5ago=ma25_arr[i-MA25_RISE_LOOKBACK]
        long_sig  = (ma25_now > ma25_5ago) and (ma25_now < ma50_now)
        short_sig = (ma25_now < ma25_5ago) and (ma25_now > ma50_now)
        if not (long_sig or short_sig): continue
        if i - last_entry_i < COOLDOWN: continue
        last_entry_i = i

        direction = 1 if long_sig else -1
        entry = close_arr[i]
        ts = idx[i]

        crossed_ma50 = False
        exit_pips = None
        exit_reason = 'timeout'
        h = 0
        for h in range(1, MAX_HOLD+1):
            j=i+h
            if j>=len(close_arr): break
            price_now=close_arr[j]; move=direction*(price_now-entry)/pip
            ma25_j=ma25_arr[j]; ma50_j=ma50_arr[j]
            if direction==1 and price_now>ma50_j: crossed_ma50=True
            if direction==-1 and price_now<ma50_j: crossed_ma50=True
            if h==NO_CROSS_EXIT and not crossed_ma50:
                exit_pips=move; exit_reason='no_cross'; break
            if h >= TP_LOOKBACK+2:
                ma25_j1=ma25_arr[j-1]
                if not np.isnan(ma25_j) and not np.isnan(ma25_j1) and not np.isnan(ma25_arr[j-TP_LOOKBACK]):
                    ma25_ref=ma25_arr[j-TP_LOOKBACK]
                    if direction==1 and ma25_j<ma25_j1 and ma25_j1<ma25_ref:
                        exit_pips=move; exit_reason='tp_ma'; break
                    if direction==-1 and ma25_j>ma25_j1 and ma25_j1>ma25_ref:
                        exit_pips=move; exit_reason='tp_ma'; break
        if exit_pips is None:
            exit_pips=direction*(close_arr[min(i+MAX_HOLD,len(close_arr)-1)]-entry)/pip

        row = {'ts': ts, 'pair': pair, 'pnl': exit_pips-sp, 'direction': direction}
        for feat, ranks in feat_ranks.items():
            v = ranks.iloc[i] if i < len(ranks) else np.nan
            row[feat+'_r'] = v
        raw_signals.append(row)

df_all = pd.DataFrame(raw_signals).set_index('ts').sort_index()
t_base = df_all['pnl']
nm = (df_all.index.max()-df_all.index.min()).days/30
cut = df_all.index.max()-pd.DateOffset(months=18)

def stats(t):
    if len(t) < 20: return None
    wr=(t>0).mean(); ev=t.mean()
    sh=(ev/t.std())*np.sqrt(BARS_PER_YEAR) if t.std()>0 else 0
    npm=len(t)/nm
    t18=t[t.index>=cut]
    wr18=(t18>0).mean() if len(t18)>10 else 0
    ev18=t18.mean() if len(t18)>10 else 0
    return {'n':len(t),'npm':npm,'wr':wr,'ev':ev,'sh':sh,'pnl_mo':ev*npm,
            'n18':len(t18),'wr18':wr18,'ev18':ev18}

baseline = stats(t_base)
print(f'\nBaseline: N={baseline["n"]:,} {baseline["npm"]:.0f}/mo WR={baseline["wr"]:.1%} EV={baseline["ev"]:+.2f} Sh={baseline["sh"]:+.2f}')
print()
print(f'{"Feature":<30} {"Dir":<5} {"Thresh":<7} {"N":>6} {"/mo":>5} {"WR":>7} {"EV":>8} {"Sh":>7} | {"18mWR":>7} {"18mEV":>8}')
print('='*105)

results = []
for feat in FEATURES:
    col = feat+'_r'
    if col not in df_all.columns: continue
    for thresh, direction_label in [(0.25,'LOW'), (0.75,'HIGH')]:
        if direction_label == 'LOW':
            filt = df_all[col] < thresh
        else:
            filt = df_all[col] > thresh
        sub = df_all[filt]['pnl']
        s = stats(sub)
        if s is None: continue
        results.append({**s, 'feat': feat, 'dir': direction_label, 'thresh': thresh})

results_df = pd.DataFrame(results).sort_values('ev', ascending=False)
for _, r in results_df.iterrows():
    marker = '***' if r['ev'] > 0 else ''
    print(f'{r.feat:<30} {r.dir:<5} {"<25%" if r.dir=="LOW" else ">75%":<7} '
          f'{r.n:>6,} {r.npm:>5.0f} {r.wr:>7.1%} {r.ev:>+8.2f} {r.sh:>+7.2f} | '
          f'{r.wr18:>7.1%} {r.ev18:>+8.2f}  {marker}')
