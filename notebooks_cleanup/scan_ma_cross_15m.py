import pandas as pd, numpy as np, warnings; warnings.filterwarnings('ignore')

PAIRS = ['AUDJPY','AUDNZD','AUDUSD','CADJPY','CHFJPY','EURAUD','EURGBP','EURJPY',
         'EURUSD','GBPJPY','GBPUSD','NZDUSD','USDCAD','USDCHF','USDJPY']
PIP = {'AUDJPY':0.01,'AUDNZD':0.0001,'AUDUSD':0.0001,'CADJPY':0.01,'CHFJPY':0.01,
       'EURAUD':0.0001,'EURGBP':0.0001,'EURJPY':0.01,'EURUSD':0.0001,'GBPJPY':0.01,
       'GBPUSD':0.0001,'NZDUSD':0.0001,'USDCAD':0.0001,'USDCHF':0.0001,'USDJPY':0.01}
SPREAD={'EURUSD':0.2,'GBPUSD':0.4,'USDJPY':0.3,'USDCHF':0.4,'USDCAD':0.4,
        'AUDUSD':0.3,'NZDUSD':0.5,'EURGBP':0.5,'EURJPY':0.5,'GBPJPY':0.8,
        'AUDJPY':0.8,'CADJPY':1.0,'CHFJPY':1.0,'EURAUD':0.8,'AUDNZD':1.0}

BARS_PER_YEAR = 252 * 24
MAX_HOLD = 96   # safety cap
COOLDOWN = 24   # 24H cooldown per pair
MA25_RISE_LOOKBACK = 5   # MA25 must be rising over last 5 bars for long entry
TP_LOOKBACK = 5          # TP when MA25 turns: current < 5 bars ago (long), 2 consecutive
NO_CROSS_EXIT = 5        # exit if price hasn't exceeded MA50 within 5 bars

raw_signals = []

for pair in PAIRS:
    print(f'  {pair}...', flush=True)
    pip = PIP[pair]; sp = SPREAD[pair]
    df = pd.read_parquet(f'backend/data/processed/{pair}_1H.parquet')
    close = df['close']
    ma25 = close.rolling(25).mean()
    ma50 = close.rolling(50).mean()
    close_arr = close.values
    ma25_arr = ma25.values
    ma50_arr = ma50.values
    idx = close.index

    last_entry_i = -COOLDOWN - 1
    for i in range(55, len(close_arr) - MAX_HOLD):
        if np.isnan(ma25_arr[i]) or np.isnan(ma50_arr[i]): continue
        if np.isnan(ma25_arr[i - MA25_RISE_LOOKBACK]): continue

        ma25_now  = ma25_arr[i]
        ma50_now  = ma50_arr[i]
        ma25_5ago = ma25_arr[i - MA25_RISE_LOOKBACK]

        # Long: MA25 rising + MA25 below MA50 (approaching from below)
        long_sig  = (ma25_now > ma25_5ago) and (ma25_now < ma50_now)
        # Short: MA25 falling + MA25 above MA50 (approaching from above)
        short_sig = (ma25_now < ma25_5ago) and (ma25_now > ma50_now)

        if not (long_sig or short_sig): continue
        if i - last_entry_i < COOLDOWN: continue
        last_entry_i = i

        direction = 1 if long_sig else -1
        entry = close_arr[i]

        exit_pips = None
        exit_reason = 'timeout'
        crossed_ma50 = False

        for h in range(1, MAX_HOLD + 1):
            j = i + h
            if j >= len(close_arr): break
            price_now = close_arr[j]
            move = direction * (price_now - entry) / pip
            ma25_j = ma25_arr[j]
            ma50_j = ma50_arr[j]

            # Check if price exceeded MA50 (cross materialised)
            if direction == 1 and price_now > ma50_j:
                crossed_ma50 = True
            if direction == -1 and price_now < ma50_j:
                crossed_ma50 = True

            # Timeout exit: price never exceeded MA50 within 5 bars
            if h == NO_CROSS_EXIT and not crossed_ma50:
                exit_pips = move
                exit_reason = 'no_cross'
                break

            # TP: MA25 turns against direction — requires 2 consecutive bars
            # long: MA25[j] < MA25[j-1] AND MA25[j-1] < MA25[j-1-TP_LOOKBACK]
            if h >= TP_LOOKBACK + 2:
                ma25_j1 = ma25_arr[j - 1]  # previous bar
                if not np.isnan(ma25_j) and not np.isnan(ma25_j1) and not np.isnan(ma25_arr[j - TP_LOOKBACK]):
                    ma25_ref = ma25_arr[j - TP_LOOKBACK]
                    if direction == 1 and ma25_j < ma25_j1 and ma25_j1 < ma25_ref:
                        exit_pips = move
                        exit_reason = 'tp_ma'
                        break
                    if direction == -1 and ma25_j > ma25_j1 and ma25_j1 > ma25_ref:
                        exit_pips = move
                        exit_reason = 'tp_ma'
                        break

        if exit_pips is None:
            exit_pips = direction * (close_arr[min(i+MAX_HOLD, len(close_arr)-1)] - entry) / pip

        raw_signals.append({
            'ts': idx[i], 'pair': pair, 'direction': direction,
            'exit_pips': exit_pips, 'exit_reason': exit_reason,
            'hold_bars': h, 'sp': sp
        })

print(f'  Signals after cooldown: {len(raw_signals):,}')

nm = (raw_signals[-1]['ts'] - raw_signals[0]['ts']).days / 30
cut = raw_signals[-1]['ts'] - pd.DateOffset(months=18)

pnl_all = [{'ts': s['ts'], 'pnl': s['exit_pips'] - s['sp'],
             'pair': s['pair'], 'reason': s['exit_reason'],
             'hold_h': s['hold_bars']} for s in raw_signals]
df_r = pd.DataFrame(pnl_all).set_index('ts').sort_index()
t = df_r['pnl']
t18 = t[t.index >= cut]

wins = t[t>0]; losses = t[t<=0]
wins18 = t18[t18>0]; losses18 = t18[t18<=0]
sh = (t.mean()/t.std())*np.sqrt(BARS_PER_YEAR) if t.std()>0 else 0
sh18 = (t18.mean()/t18.std())*np.sqrt(BARS_PER_YEAR) if t18.std()>0 else 0

print()
print('MA25 rising+below MA50 entry | TP=MA25 turns | Exit if no cross in 5H | 24H cooldown')
print('='*65)
avg_hold = df_r['hold_h'].mean()
avg_hold18 = df_r[df_r.index>=cut]['hold_h'].mean()
print(f'\nFULL HISTORY ({len(t):,} trades, {nm:.0f} months):')
print(f'  WR:         {(t>0).mean():.1%}')
print(f'  EV:         {t.mean():+.2f} pips/trade')
print(f'  Sharpe:     {sh:+.2f}')
print(f'  /month:     {len(t)/nm:.0f} trades,  {t.mean()*len(t)/nm:+.0f} pips/mo')
print(f'  Avg win:    {wins.mean():+.1f}  |  Avg loss: {losses.mean():+.1f}  |  Ratio: {abs(wins.mean()/losses.mean()):.2f}x')
print(f'  Avg hold:   {avg_hold:.1f}H')

print(f'\nLAST 18 MONTHS ({len(t18):,} trades):')
print(f'  WR:         {(t18>0).mean():.1%}')
print(f'  EV:         {t18.mean():+.2f} pips/trade')
print(f'  Sharpe:     {sh18:+.2f}')
print(f'  /month:     {len(t18)/18:.0f} trades,  {t18.sum()/18:+.0f} pips/mo')
print(f'  Avg win:    {wins18.mean():+.1f}  |  Avg loss: {losses18.mean():+.1f}')
print(f'  Avg hold:   {avg_hold18:.1f}H')

print(f'\nEXIT REASON breakdown (all):')
for reason, g in df_r.groupby('reason'):
    print(f'  {reason:<10} {len(g):>6,} ({len(g)/len(df_r):.1%})  WR={( g.pnl>0).mean():.1%}  EV={g.pnl.mean():+.2f}  AvgHold={g.hold_h.mean():.1f}H')

print(f'\nPIP DISTRIBUTION (18m):')
buckets_pip=[(-999,-20),(-20,-10),(-10,0),(0,10),(10,20),(20,50),(50,100),(100,999)]
for lo,hi in buckets_pip:
    n=((t18>lo)&(t18<=hi)).sum(); pct=n/len(t18) if len(t18)>0 else 0
    bar='#'*int(pct*40)
    print(f'  [{lo:>5} to {hi:>4}]: {n:>4} ({pct:>5.1%}) {bar}')

print(f'\nMONTHLY (last 18m):')
print(f'  {"Month":<10} {"N":>4} {"WR":>7} {"EV":>8} {"CumPnL":>9}')
print('  '+'-'*42)
cum=0
for (yr,mo),g in t18.groupby([t18.index.year,t18.index.month]):
    if len(g)<1: continue
    ev=g.mean(); cum+=g.sum()
    flag='<--' if ev<0 else ''
    print(f'  {yr}-{mo:02d}    {len(g):>4}  {(g>0).mean():>7.1%}  {ev:>+8.2f}  {cum:>+9.0f}  {flag}')

print(f'\nPER-PAIR (18m):')
df18=df_r[df_r.index>=cut]
print(f'  {"Pair":<10} {"N":>5} {"WR":>7} {"EV":>8} {"PnL":>8}')
for pair,g in df18.groupby('pair'):
    tp=g['pnl']
    print(f'  {pair:<10} {len(tp):>5} {(tp>0).mean():>7.1%} {tp.mean():>+8.2f} {tp.sum():>+8.0f}')
