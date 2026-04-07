import pandas as pd, numpy as np, joblib, warnings; warnings.filterwarnings('ignore')

PAIRS = ['AUDJPY','AUDNZD','AUDUSD','CADJPY','CHFJPY','EURAUD','EURGBP','EURJPY',
         'EURUSD','GBPJPY','GBPUSD','NZDUSD','USDCAD','USDCHF','USDJPY']
PIP = {'AUDJPY':0.01,'AUDNZD':0.0001,'AUDUSD':0.0001,'CADJPY':0.01,'CHFJPY':0.01,
       'EURAUD':0.0001,'EURGBP':0.0001,'EURJPY':0.01,'EURUSD':0.0001,'GBPJPY':0.01,
       'GBPUSD':0.0001,'NZDUSD':0.0001,'USDCAD':0.0001,'USDCHF':0.0001,'USDJPY':0.01}
MAX_1H={'AUDJPY':250,'AUDNZD':80,'AUDUSD':120,'CADJPY':250,'CHFJPY':250,'EURAUD':150,
        'EURGBP':100,'EURJPY':300,'EURUSD':150,'GBPJPY':300,'GBPUSD':150,'NZDUSD':100,
        'USDCAD':150,'USDCHF':120,'USDJPY':300}
SPREAD={'EURUSD':0.2,'GBPUSD':0.4,'USDJPY':0.3,'USDCHF':0.4,'USDCAD':0.4,
        'AUDUSD':0.3,'NZDUSD':0.5,'EURGBP':0.5,'EURJPY':0.5,'GBPJPY':0.8,
        'AUDJPY':0.8,'CADJPY':1.0,'CHFJPY':1.0,'EURAUD':0.8,'AUDNZD':1.0}

m = joblib.load('backend/models_9/mfe_q50/model_1H_Q50.joblib')
model=m['model']; feat_cols=m['feature_cols']
closes_all={p: pd.read_parquet(f'backend/data/processed/{p}_1H.parquet')['close'] for p in PAIRS}

CUR_PAIRS = {
    'USD':[('EURUSD',-1),('GBPUSD',-1),('AUDUSD',-1),('NZDUSD',-1),('USDCAD',+1),('USDCHF',+1),('USDJPY',+1)],
    'EUR':[('EURUSD',+1),('EURGBP',+1),('EURAUD',+1),('EURJPY',+1)],
    'GBP':[('GBPUSD',+1),('EURGBP',-1),('GBPJPY',+1)],
    'AUD':[('AUDUSD',+1),('AUDJPY',+1),('AUDNZD',+1),('EURAUD',-1)],
    'CAD':[('USDCAD',-1),('CADJPY',+1)],
    'CHF':[('USDCHF',-1),('CHFJPY',+1)],
    'JPY':[('USDJPY',-1),('EURJPY',-1),('GBPJPY',-1),('AUDJPY',-1),('CADJPY',-1),('CHFJPY',-1)],
    'NZD':[('NZDUSD',+1),('AUDNZD',-1)],
}
cur_strength={}
for cur,ps in CUR_PAIRS.items():
    parts=[sign*np.log(closes_all[p]).diff() for p,sign in ps]
    avg=pd.concat(parts,axis=1).mean(axis=1)
    cur_strength[f'{cur}_24h']=avg.rolling(24).sum()
cur_df=pd.DataFrame(cur_strength)
PAIR_CUR={'AUDJPY':('AUD','JPY'),'AUDNZD':('AUD','NZD'),'AUDUSD':('AUD','USD'),
    'CADJPY':('CAD','JPY'),'CHFJPY':('CHF','JPY'),'EURAUD':('EUR','AUD'),
    'EURGBP':('EUR','GBP'),'EURJPY':('EUR','JPY'),'EURUSD':('EUR','USD'),
    'GBPJPY':('GBP','JPY'),'GBPUSD':('GBP','USD'),'NZDUSD':('NZD','USD'),
    'USDCAD':('USD','CAD'),'USDCHF':('USD','CHF'),'USDJPY':('USD','JPY')}
MA_PERIODS=[8,24,72,168,480]

# SL configs to test on 24H no TP:
# ('label', sl_type, param)
# sl_type='fixed'    : exit if move <= -param at any hour
# sl_type='trail'    : trailing stop, starts at -param, moves up with peak
# sl_type='none'     : no SL
raw_signals = []

SL_CONFIGS = [
    ('no SL',            'none',  0),
    ('fixed -50',        'fixed', 50),
    ('fixed -75',        'fixed', 75),
    ('fixed -100',       'fixed', 100),
    ('fixed -150',       'fixed', 150),
    ('trail -100',       'trail', 100),
    ('trail -75',        'trail', 75),
    ('trail -50',        'trail', 50),
]

HOLDS = [24]
TPS   = [999]
results = {(h,tp): [] for h in HOLDS for tp in TPS}
# sl results keyed by label
sl_results = {cfg[0]: [] for cfg in SL_CONFIGS}

for pair in PAIRS:
    print(f'  {pair}...', flush=True)
    pip=PIP[pair]; sp=SPREAD[pair]
    df9=pd.read_parquet(f'backend/data/features_9/{pair}_features.parquet')
    Xnp=np.column_stack([df9[c].values if c in df9.columns else np.full(len(df9),np.nan) for c in feat_cols])
    pred_raw=pd.Series(model.predict(Xnp), index=df9.index); del Xnp

    close_1h=closes_all[pair]
    mas={p: close_1h.rolling(p).mean() for p in MA_PERIODS}
    ac=pd.DataFrame({p:(close_1h>mas[p]).astype(int) for p in MA_PERIODS}).sum(axis=1)
    base,quote=PAIR_CUR[pair]
    diff24=(cur_df.get(f'{base}_24h',pd.Series(0,index=cur_df.index)) -
            cur_df.get(f'{quote}_24h',pd.Series(0,index=cur_df.index))).reindex(close_1h.index).rank(pct=True)

    ret_1h=close_1h.diff().abs()/pip
    holiday=~(((close_1h.index.month==12)&(close_1h.index.day.isin([24,25,26,31])))|
              ((close_1h.index.month==1)&(close_1h.index.day.isin([1,2]))))
    quality=(ret_1h<=MAX_1H[pair])&holiday

    common=pred_raw.index.intersection(close_1h.index).intersection(diff24.index)
    pr=pred_raw.reindex(common)
    qual=quality.reindex(common).fillna(False)
    ac_c=ac.reindex(common); dr=diff24.reindex(common)

    mfe_ok=pr>70
    if 'kyle_lambda_delta_6h' not in df9.columns: continue
    r=df9['kyle_lambda_delta_6h'].reindex(common).rank(pct=True)
    kl=r<0.25

    mask=qual
    long_c  = mask & mfe_ok & ((5-ac_c)>=5) & (dr<0.40) & kl
    short_c = mask & mfe_ok & (ac_c>=5)     & (dr>0.60) & kl
    sig = common[long_c | short_c]
    close_arr=close_1h.values; close_idx=close_1h.index

    for ts in sig:
        direction = 1 if long_c.loc[ts] else -1
        pos = close_idx.get_loc(ts)
        entry = close_arr[pos]
        path = []
        for h in range(1, 25):
            p = pos+h
            if p >= len(close_arr): break
            path.append(direction*(close_arr[p]-entry)/pip)
        if len(path) < 24: continue
        raw_signals.append({'ts':ts,'pair':pair,'direction':direction,'path':path,'sp':sp})

print(f'  Raw signals: {len(raw_signals):,}')

# Apply 24H cooldown per pair — sort all signals by time, block pair for 24H after each trade
raw_signals.sort(key=lambda x: x['ts'])
last_trade = {}
accepted = []
for sig in raw_signals:
    ts, pair = sig['ts'], sig['pair']
    if pair in last_trade and (ts - last_trade[pair]).total_seconds() < 24*3600:
        continue
    last_trade[pair] = ts
    accepted.append(sig)

print(f'  After 24H cooldown: {len(accepted):,}')

for sig in accepted:
    ts=sig['ts']; sp=sig['sp']; pair=sig['pair']; h_path=sig['path']

    # hold/tp grid
    for hold in HOLDS:
        for tp in TPS:
            exit_pips = h_path[-1]
            for move in h_path:
                if move <= -100:
                    exit_pips = -100; break
                if move >= tp:
                    exit_pips = tp; break
            results[(hold,tp)].append({'ts':ts,'pnl':exit_pips-sp,'pair':pair})

    # SL configs
    for cfg_label, sl_type, sl_param in SL_CONFIGS:
        if sl_type == 'none':
            exit_pips = h_path[-1]
        elif sl_type == 'fixed':
            exit_pips = h_path[-1]
            for move in h_path:
                if move <= -sl_param:
                    exit_pips = -sl_param; break
        elif sl_type == 'trail':
            peak = 0.0
            exit_pips = h_path[-1]
            for move in h_path:
                if move > peak: peak = move
                if move <= peak - sl_param:
                    exit_pips = peak - sl_param; break
        sl_results[cfg_label].append({'ts':ts,'pnl':exit_pips-sp,'pair':pair})

all_pnl = results[(24,999)]

cut = pd.Timestamp('2023-10-01')

print()
print('SL=-100 | scanning hold periods and take-profits')
print(f'{"Hold":>6} {"TP":>6} | {"N":>7} {"/mo":>5} {"WR":>7} {"EV":>8} {"Sh":>7} {"PnL/mo":>8} | {"18mWR":>7} {"18mEV":>8} {"18mSh":>7} {"18mPnL/mo":>10}')
print('='*105)
prev_hold = None
for hold in HOLDS:
    for tp in TPS:
        rows = results[(hold,tp)]
        if not rows: continue
        t = pd.DataFrame(rows).set_index('ts').sort_index()['pnl']
        nm=(t.index.max()-t.index.min()).days/30
        wr=(t>0).mean(); ev=t.mean()
        sh=(ev/t.std())*np.sqrt(252) if t.std()>0 else 0
        npm=len(t)/nm
        t18=t[t.index>=cut]
        wr18=(t18>0).mean() if len(t18)>5 else 0
        ev18=t18.mean() if len(t18)>5 else 0
        sh18=(ev18/t18.std())*np.sqrt(252) if len(t18)>5 and t18.std()>0 else 0
        npm18=len(t18)/18
        tp_str='no TP' if tp==999 else f'+{tp}'
        if prev_hold and hold!=prev_hold: print()
        print(f'{hold:>5}H {tp_str:>6} | {len(t):>7,} {npm:>5.0f} {wr:>7.1%} {ev:>+8.1f} {sh:>+7.2f} {ev*npm:>+8.0f} | '
              f'{wr18:>7.1%} {ev18:>+8.1f} {sh18:>+7.2f} {ev18*npm18:>+10.0f}')
        prev_hold = hold

print()
print('SL CONFIGS on 24H no TP:')
print(f'{"Config":<16} | {"N":>7} {"/mo":>5} {"WR":>7} {"EV":>8} {"Sh":>7} {"PnL/mo":>8} | {"18mWR":>7} {"18mEV":>8} {"18mSh":>7} {"18mPnL/mo":>10}')
print('='*100)
for cfg_label, _, _ in SL_CONFIGS:
    rows = sl_results[cfg_label]
    if not rows: continue
    t = pd.DataFrame(rows).set_index('ts').sort_index()['pnl']
    nm=(t.index.max()-t.index.min()).days/30
    wr=(t>0).mean(); ev=t.mean()
    sh=(ev/t.std())*np.sqrt(252) if t.std()>0 else 0
    npm=len(t)/nm
    t18=t[t.index>=cut]
    wr18=(t18>0).mean() if len(t18)>5 else 0
    ev18=t18.mean() if len(t18)>5 else 0
    sh18=(ev18/t18.std())*np.sqrt(252) if len(t18)>5 and t18.std()>0 else 0
    npm18=len(t18)/18
    print(f'{cfg_label:<16} | {len(t):>7,} {npm:>5.0f} {wr:>7.1%} {ev:>+8.1f} {sh:>+7.2f} {ev*npm:>+8.0f} | '
          f'{wr18:>7.1%} {ev18:>+8.1f} {sh18:>+7.2f} {ev18*npm18:>+10.0f}')

# detailed report uses trail -50
all_pnl2 = pd.DataFrame(sl_results['trail -50']).set_index('ts').sort_index()
t_all = all_pnl2
t = t_all['pnl']
t18 = t[t.index>=cut]
nm_all = (t.index.max()-t.index.min()).days/30

wins18 = t18[t18>0]; losses18 = t18[t18<=0]
sh18 = (t18.mean()/t18.std())*np.sqrt(252)

print()
print('='*65)
print('DETAIL: fade+cur+MFE>70 + kl_delta_6h<25 | Trail-50 | 24H | ALL HOURS')
print('='*65)
print(f'\nFULL HISTORY ({len(t):,} trades, {nm_all:.0f} months):')
print(f'  WR:      {(t>0).mean():.1%}')
print(f'  EV:      {t.mean():+.1f} pips/trade')
sh_all=(t.mean()/t.std())*np.sqrt(252)
print(f'  Sharpe:  {sh_all:+.2f}')
print(f'  /month:  {len(t)/nm_all:.0f} trades,  {t.mean()*len(t)/nm_all:+.0f} pips/mo')

print(f'\nLAST 18 MONTHS ({len(t18):,} trades):')
print(f'  WR:         {(t18>0).mean():.1%}')
print(f'  EV:         {t18.mean():+.1f} pips/trade')
print(f'  Sharpe:     {sh18:+.2f}')
print(f'  /month:     {len(t18)/18:.0f} trades,  {t18.sum()/18:+.0f} pips/mo')
print(f'  Avg win:    {wins18.mean():+.1f}  |  Avg loss: {losses18.mean():+.1f}  |  Ratio: {abs(wins18.mean()/losses18.mean()):.2f}x')
print(f'  Max loss:   {losses18.min():+.1f} pips')

print(f'\nPIP DISTRIBUTION (18m):')
buckets_pip=[(-999,-100),(-100,-50),(-50,-20),(-20,0),(0,20),(20,50),(50,100),(100,200),(200,999)]
for lo,hi in buckets_pip:
    n=((t18>lo)&(t18<=hi)).sum(); pct=n/len(t18)
    bar='#'*int(pct*40)
    print(f'  [{lo:>5} to {hi:>4}]: {n:>4} ({pct:>5.1%}) {bar}')

print(f'\nMONTHLY BREAKDOWN (last 18m):')
print(f'  {"Month":<10} {"N":>4} {"WR":>7} {"AvgW":>7} {"AvgL":>7} {"EV":>8} {"Sh":>7} {"CumPnL":>9}')
print('  '+'-'*68)
cum=0
for (yr,mo),g in t18.groupby([t18.index.year,t18.index.month]):
    if len(g)<1: continue
    w=g[g>0]; l=g[g<=0]
    ev=g.mean()
    sh_m=(ev/g.std())*np.sqrt(252) if len(g)>1 and g.std()>0 else 0
    cum+=g.sum()
    aw=w.mean() if len(w)>0 else 0
    al=l.mean() if len(l)>0 else 0
    flag = '<--' if ev < 0 else ''
    print(f'  {yr}-{mo:02d}    {len(g):>4}  {(g>0).mean():>7.1%}  {aw:>+7.1f}  {al:>+7.1f}  {ev:>+8.1f}  {sh_m:>+7.2f}  {cum:>+9.0f}  {flag}')

print(f'\nPER-PAIR (18m):')
print(f'  {"Pair":<10} {"N":>5} {"WR":>7} {"EV":>8} {"PnL":>8}')
print('  '+'-'*42)
df18=t_all[t_all.index>=cut]
for pair,g in df18.groupby('pair'):
    tp=g['pnl']
    if len(tp)<3: continue
    print(f'  {pair:<10} {len(tp):>5} {(tp>0).mean():>7.1%} {tp.mean():>+8.1f} {tp.sum():>+8.0f}')

print(f'\nTRADES BY HOUR OF DAY (18m, trail -50):')
t_trail = pd.DataFrame(sl_results['trail -50']).set_index('ts').sort_index()
t_trail18 = t_trail[t_trail.index>=cut]
print(f'  {"Hour":>5} {"N":>5} {"WR":>7} {"EV":>8} {"PnL":>8}')
print('  '+'-'*38)
for hr in range(24):
    g = t_trail18['pnl'][t_trail18.index.hour==hr]
    if len(g)<3: continue
    print(f'  {hr:>5}h {len(g):>5} {(g>0).mean():>7.1%} {g.mean():>+8.1f} {g.sum():>+8.0f}')
