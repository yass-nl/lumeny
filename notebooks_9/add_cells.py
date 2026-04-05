import json
from pathlib import Path

with open('notebooks_9/02_model_training.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

def md(src): return {'cell_type': 'markdown', 'metadata': {}, 'source': src, 'id': 'md'}
def code(src): return {'cell_type': 'code', 'metadata': {}, 'source': src, 'outputs': [], 'execution_count': None, 'id': 'code'}

new_cells = [

md('## 8. EV Per Trade'),

code("""\
SPREAD_PIPS = 1.5
PIP_SIZE = {
    'EURUSD':0.0001,'GBPUSD':0.0001,'USDCHF':0.0001,'AUDUSD':0.0001,
    'USDCAD':0.0001,'NZDUSD':0.0001,'EURGBP':0.0001,'EURAUD':0.0001,'AUDNZD':0.0001,
    'USDJPY':0.01,'EURJPY':0.01,'GBPJPY':0.01,'AUDJPY':0.01,'CADJPY':0.01,'CHFJPY':0.01,
}

ohlc_cache = {}
for pair in PAIRS:
    o = pd.read_parquet(PROCESSED_DIR / f'{pair}_1H.parquet')[['close']]
    o.index = pd.to_datetime(o.index)
    o['ma200'] = o['close'].rolling(200).mean()
    ohlc_cache[pair] = o

results['ma200_at_entry'] = np.nan
results['entry_price']    = np.nan
for pair in PAIRS:
    mask = results['pair'] == pair
    idx  = results[mask].index
    ohlc = ohlc_cache[pair].reindex(idx)
    results.loc[mask, 'ma200_at_entry'] = ohlc['ma200'].values
    results.loc[mask, 'entry_price']    = ohlc['close'].values

results['pip']          = results['pair'].map(PIP_SIZE)
results['pips_to_ma200'] = (results['ma200_at_entry'] - results['entry_price']) / results['pip']

print(f'EV Analysis by Probability Threshold (spread={SPREAD_PIPS}p, SL baked in label)')
print(f'{"Thresh":>8} {"N":>7} {"Hit%":>7} {"Avg pips to MA200":>19} {"EV (hits only)":>16} {"EV (zero miss)":>16}')
print('-' * 80)
for t in [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    mask = results['proba'] >= t
    if mask.sum() < 20: continue
    s        = results[mask]
    hit_rate = s['actual'].mean()
    hits     = s[s['actual'] == 1]
    avg_win  = hits['pips_to_ma200'].mean() if len(hits) > 0 else 0
    ev_hits  = avg_win - SPREAD_PIPS
    ev_zero  = hit_rate * avg_win - SPREAD_PIPS
    print(f'{t:>8.1f} {mask.sum():>7,} {hit_rate:>7.1%} {s["pips_to_ma200"].mean():>19.1f} {ev_hits:>16.1f} {ev_zero:>16.1f}')

print(f'\\nPer-pair EV at proba >= 0.5:')
print(f'{"Pair":>10} {"N":>6} {"Hit%":>7} {"Avg pips to MA200":>19} {"EV (zero miss)":>16}')
print('-' * 62)
sub = results[results['proba'] >= 0.5]
for pair, g in sub.groupby('pair'):
    if len(g) < 10: continue
    hit_rate = g['actual'].mean()
    avg_win  = g[g['actual']==1]['pips_to_ma200'].mean() if g['actual'].sum() > 0 else 0
    ev       = hit_rate * avg_win - SPREAD_PIPS
    print(f'{pair:>10} {len(g):>6,} {hit_rate:>7.1%} {avg_win:>19.1f} {ev:>16.1f}')
"""),

md('## 9. EV Filtered by Distance to MA200'),

code("""\
print('EV filtered to pips_to_ma200 > X\\n')
for min_dist in [20, 30, 50, 75]:
    sub = results[results['pips_to_ma200'] >= min_dist]
    if len(sub) < 50: continue
    print(f'=== pips_to_ma200 >= {min_dist} (N={len(sub):,}, {len(sub)/len(results):.1%} of test) ===')
    print(f'{"Thresh":>8} {"N":>7} {"Hit%":>7} {"Avg pips to MA200":>19} {"EV (zero miss)":>16}')
    print('-' * 65)
    for t in [0.4, 0.5, 0.6, 0.7, 0.8]:
        mask = (results['proba'] >= t) & (results['pips_to_ma200'] >= min_dist)
        if mask.sum() < 20: continue
        s        = results[mask]
        hit_rate = s['actual'].mean()
        avg_win  = s[s['actual']==1]['pips_to_ma200'].mean() if s['actual'].sum() > 0 else 0
        ev       = hit_rate * avg_win - SPREAD_PIPS
        print(f'{t:>8.1f} {mask.sum():>7,} {hit_rate:>7.1%} {avg_win:>19.1f} {ev:>16.1f}')
    print()
"""),

md('## 10. Hour Distribution'),

code("""\
sub = results[(results['proba'] >= 0.5) & (results['pips_to_ma200'] >= 50)].copy()
sub['hour'] = sub.index.hour
by_hour = sub.groupby('hour').agg(n=('actual','count'), hit_rate=('actual','mean'), avg_pips=('pips_to_ma200','mean'))
total = by_hour['n'].sum()
print(f'Hour distribution (proba>=0.5, pips_to_ma200>=50), total={int(total):,}')
print(f'{"Hour":>5} {"N":>6} {"%":>6} {"Hit%":>7} {"Avg pips":>10}')
print('-'*42)
for hour, row in by_hour.iterrows():
    bar = '#' * int(row['n'] / total * 40)
    print(f'{hour:>5} {int(row["n"]):>6} {row["n"]/total:>6.1%} {row["hit_rate"]:>7.1%} {row["avg_pips"]:>10.1f}  {bar}')
"""),

md('## 11. Hold Time on Winning Trades'),

code("""\
print('Computing hold times on winning trades (proba>=0.5, pips_to_ma200>=50)...')
hold_times = []
for pair in PAIRS:
    ohlc = pd.read_parquet(PROCESSED_DIR / f'{pair}_1H.parquet')[['high']]
    ohlc.index = pd.to_datetime(ohlc.index)
    mask = (results['pair'] == pair) & (results['proba'] >= 0.5) & (results['pips_to_ma200'] >= 50) & (results['actual'] == 1)
    for ts, row in results[mask].iterrows():
        if ts not in ohlc.index: continue
        iloc   = ohlc.index.get_loc(ts)
        target = row['ma200_at_entry']
        for k in range(1, MAX_BARS + 1):
            if iloc + k >= len(ohlc): break
            if ohlc['high'].iloc[iloc + k] >= target:
                hold_times.append({'pair': pair, 'bars': k, 'pips': row['pips_to_ma200']})
                break

df_hold = pd.DataFrame(hold_times)
print(f'Winning trades: {len(df_hold):,}')
print(f'\\nHold time distribution (hours):')
print(df_hold['bars'].describe().round(1))
print(f'\\nBuckets:')
for lo, hi in [(1,6),(6,12),(12,24),(24,48),(48,72)]:
    mask = (df_hold['bars'] >= lo) & (df_hold['bars'] < hi)
    print(f'  {lo:>3}-{hi:>3}H: {mask.sum():>5,} ({mask.mean():.1%})')
print(f'\\nAvg hold by pair:')
for pair, g in df_hold.groupby('pair'):
    print(f'  {pair:>10}: avg={g["bars"].mean():.1f}H  median={g["bars"].median():.0f}H')
"""),

md('## 12. Stop Loss Simulation'),

code("""\
print('Stop Loss Simulation (proba>=0.5, pips_to_ma200>=50)')
print('TP = MA200 at entry, SL = fixed pips below entry, timeout = T+72 close')
print()

sim_results = []
for pair in PAIRS:
    pip  = PIP_SIZE[pair]
    ohlc = pd.read_parquet(PROCESSED_DIR / f'{pair}_1H.parquet')[['close','high','low']]
    ohlc.index = pd.to_datetime(ohlc.index)
    mask = (results['pair'] == pair) & (results['proba'] >= 0.5) & (results['pips_to_ma200'] >= 50)
    for ts, row in results[mask].iterrows():
        if ts not in ohlc.index: continue
        iloc = ohlc.index.get_loc(ts)
        sim_results.append({
            'pair':          pair,
            'pips_to_ma200': row['pips_to_ma200'],
            'lows':          ohlc['low'].iloc[iloc+1:iloc+MAX_BARS+1].values,
            'highs':         ohlc['high'].iloc[iloc+1:iloc+MAX_BARS+1].values,
            'closes':        ohlc['close'].iloc[iloc+1:iloc+MAX_BARS+1].values,
            'entry':         row['entry_price'],
            'target':        row['ma200_at_entry'],
            'pip':           pip,
        })

print(f'{"SL":>8} {"N TP":>8} {"N SL":>8} {"N timeout":>10} {"WR":>7} {"Avg win":>9} {"Avg loss":>9} {"EV":>8} {"Sharpe":>8}')
print('-'*82)
for sl_pips in [30, 40, 50, 60, 75, 100, 150, 999]:
    pnls = []
    for s in sim_results:
        pip    = s['pip']
        entry  = s['entry']
        target = s['target']
        sl_lvl = entry - sl_pips * pip
        pnl = None
        for k in range(len(s['lows'])):
            if s['lows'][k] <= sl_lvl:
                pnl = -sl_pips - SPREAD_PIPS; break
            if s['highs'][k] >= target:
                pnl = (target - entry) / pip - SPREAD_PIPS; break
        if pnl is None:
            pnl = (s['closes'][-1] - entry) / pip - SPREAD_PIPS if len(s['closes']) > 0 else -SPREAD_PIPS
        pnls.append(pnl)
    pnls   = np.array(pnls)
    wins   = pnls[pnls > 0]
    losses = pnls[pnls <= 0]
    n_sl   = (pnls <= -(sl_pips - 0.1)).sum() if sl_pips < 999 else 0
    n_tp   = (pnls > 0).sum()
    n_to   = len(pnls) - n_sl - n_tp
    ev     = pnls.mean()
    sharpe = ev / pnls.std() * np.sqrt(252) if pnls.std() > 0 else 0
    lbl    = f'{sl_pips}p' if sl_pips < 999 else 'none'
    print(f'{lbl:>8} {n_tp:>8,} {n_sl:>8,} {n_to:>10,} {(pnls>0).mean():>7.1%} '
          f'{wins.mean() if len(wins) else 0:>9.1f} '
          f'{losses.mean() if len(losses) else 0:>9.1f} '
          f'{ev:>8.2f} {sharpe:>8.3f}')

print(f'\\nPer-pair at SL=50p:')
print(f'{"Pair":>10} {"N":>5} {"WR":>7} {"Avg win":>9} {"Avg loss":>9} {"EV":>8}')
print('-'*52)
for pair in PAIRS:
    pair_sims = [s for s in sim_results if s['pair'] == pair]
    if not pair_sims: continue
    pnls = []
    for s in pair_sims:
        pip    = s['pip']
        entry  = s['entry']
        target = s['target']
        sl_lvl = entry - 50 * pip
        pnl = None
        for k in range(len(s['lows'])):
            if s['lows'][k] <= sl_lvl:
                pnl = -50 - SPREAD_PIPS; break
            if s['highs'][k] >= target:
                pnl = (target - entry) / pip - SPREAD_PIPS; break
        if pnl is None:
            pnl = (s['closes'][-1] - entry) / pip - SPREAD_PIPS if len(s['closes']) > 0 else -SPREAD_PIPS
        pnls.append(pnl)
    pnls   = np.array(pnls)
    wins   = pnls[pnls > 0]
    losses = pnls[pnls <= 0]
    print(f'{pair:>10} {len(pnls):>5} {(pnls>0).mean():>7.1%} '
          f'{wins.mean() if len(wins) else 0:>9.1f} '
          f'{losses.mean() if len(losses) else 0:>9.1f} '
          f'{pnls.mean():>8.2f}')
"""),

]

nb['cells'].extend(new_cells)

with open('notebooks_9/02_model_training.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print(f'Done. Total cells: {len(nb["cells"])}')
