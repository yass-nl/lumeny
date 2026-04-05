import pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')
from pathlib import Path

PIP_SIZE = {'EURUSD':0.0001,'GBPUSD':0.0001,'AUDUSD':0.0001,'NZDUSD':0.0001,
    'USDCAD':0.0001,'USDCHF':0.0001,'USDJPY':0.01,'EURJPY':0.01,'GBPJPY':0.01,
    'AUDJPY':0.01,'CADJPY':0.01,'CHFJPY':0.01,'EURAUD':0.0001,'EURGBP':0.0001,'AUDNZD':0.0001}
MAX_1H_MOVE = {'EURUSD':150,'GBPUSD':200,'AUDUSD':120,'NZDUSD':100,'USDCAD':150,
    'USDCHF':150,'USDJPY':300,'EURJPY':300,'GBPJPY':350,'AUDJPY':250,
    'CADJPY':250,'CHFJPY':250,'EURAUD':150,'EURGBP':100,'AUDNZD':80}

# Three spread scenarios
SPREADS = {
    'tight (ECN best)':   {'EURUSD':0.5,'GBPUSD':0.8,'AUDUSD':0.8,'NZDUSD':1.2,'USDCAD':1.0,
                            'USDCHF':1.0,'USDJPY':0.8,'EURJPY':1.2,'GBPJPY':2.0,'AUDJPY':2.0,
                            'CADJPY':2.0,'CHFJPY':2.0,'EURAUD':1.5,'EURGBP':0.8,'AUDNZD':2.0},
    'realistic (retail)': {'EURUSD':1.0,'GBPUSD':1.5,'AUDUSD':1.5,'NZDUSD':2.0,'USDCAD':2.0,
                            'USDCHF':2.0,'USDJPY':1.5,'EURJPY':2.0,'GBPJPY':3.0,'AUDJPY':3.0,
                            'CADJPY':3.0,'CHFJPY':3.0,'EURAUD':3.0,'EURGBP':1.5,'AUDNZD':3.0},
    'wide (news/stress)': {'EURUSD':2.5,'GBPUSD':3.5,'AUDUSD':3.0,'NZDUSD':4.0,'USDCAD':4.0,
                            'USDCHF':4.0,'USDJPY':3.0,'EURJPY':5.0,'GBPJPY':7.0,'AUDJPY':7.0,
                            'CADJPY':7.0,'CHFJPY':7.0,'EURAUD':6.0,'EURGBP':3.5,'AUDNZD':7.0},
    'worst case (x3)':    {'EURUSD':3.0,'GBPUSD':4.5,'AUDUSD':4.5,'NZDUSD':6.0,'USDCAD':6.0,
                            'USDCHF':6.0,'USDJPY':4.5,'EURJPY':6.0,'GBPJPY':9.0,'AUDJPY':9.0,
                            'CADJPY':9.0,'CHFJPY':9.0,'EURAUD':9.0,'EURGBP':4.5,'AUDNZD':9.0},
}

F6 = Path('backend/data/features_6'); F8 = Path('backend/data/features_8')
HOLD_H = 4

dfs = []
for pair, pip in PIP_SIZE.items():
    df6 = pd.read_parquet(F6/f'{pair}_features.parquet')
    df8 = pd.read_parquet(F8/f'{pair}_geometric.parquet'); df8['pair'] = pair
    df6r = df6.reset_index(); df8r = df8.reset_index(); idx = df6r.columns[0]
    df8r = df8r.drop(columns=[c for c in df8r.columns if c in df6r.columns and c not in [idx,'pair']],errors='ignore')
    df = pd.merge(df6r,df8r,on=[idx,'pair'],how='inner').set_index(idx).sort_index()
    df1h = pd.read_parquet(f'backend/data/processed/{pair}_1H.parquet')
    if 'datetime' in df1h.columns: df1h = df1h.set_index('datetime')
    df1h.index = pd.to_datetime(df1h.index)
    close = df1h['close'].reindex(df.index)
    fwd = (close.shift(-HOLD_H) - close) / pip
    ret_1h = ((close.shift(-1) - close) / pip).abs()
    holiday = ~(((df.index.month==12)&(df.index.day.isin([24,25,26,31])))|((df.index.month==1)&(df.index.day.isin([1,2]))))
    clean = (ret_1h.reindex(df.index) <= MAX_1H_MOVE[pair]) & holiday
    liquid = (df.index.hour >= 7) & (df.index.hour <= 21)
    df = df[clean & liquid].copy(); df['fwd_pips'] = fwd
    dfs.append(df)

big = pd.concat(dfs).dropna(subset=['fwd_pips','realized_skew','residual_12h','rv_zscore_24','kyle_lambda_delta_3h'])
for col in ['realized_skew','residual_12h','rv_zscore_24','kyle_lambda_delta_3h','vr_5']:
    if col in big.columns:
        big[f'{col}_r'] = big.groupby('pair')[col].rank(pct=True)

SIGNAL_L = (big['kyle_lambda_delta_3h_r']>0.70) & (big['residual_12h_r']<0.25) & \
           (big['rv_zscore_24_r']>0.85) & (big['realized_skew_r']<0.30) & (big['vr_5_r']<0.30)
SIGNAL_S = (big['kyle_lambda_delta_3h_r']<0.30) & (big['residual_12h_r']>0.75) & \
           (big['rv_zscore_24_r']>0.85) & (big['realized_skew_r']>0.70) & (big['vr_5_r']<0.30)

sig = pd.Series(0, index=big.index)
sig[SIGNAL_L] = 1; sig[SIGNAL_S] = -1
trades = big[sig != 0].copy()
trades['signal'] = sig[sig != 0]
trades['fwd'] = trades['fwd_pips']

print(f'Total signals: {len(trades):,}  ({len(trades)/((trades.index.max()-trades.index.min()).days/30):.0f}/mo)')
print(f'Long: {(trades["signal"]==1).sum():,}  Short: {(trades["signal"]==-1).sum():,}')
print(f'Avg absolute move: {(trades["signal"]*trades["fwd"]).mean():.1f} pips (before spread)')
print()

cut18 = trades.index.max() - pd.DateOffset(months=18)

print(f'{"Scenario":<25} {"Full EV":>9} {"Full WR":>8} {"Full Sh":>8} | {"18m EV":>9} {"18m WR":>8} {"18m Sh":>8} {"18m Total":>11}')
print('-' * 95)

for scenario, sp_dict in SPREADS.items():
    # Apply per-pair spread
    spread_series = trades['pair'].map(sp_dict)
    pnl = trades['signal'] * trades['fwd'] - spread_series

    full_sh = (pnl.mean()/pnl.std())*np.sqrt(252*24/HOLD_H)
    r = pnl[pnl.index >= cut18]
    r_sh = (r.mean()/r.std())*np.sqrt(252*24/HOLD_H) if len(r)>10 and r.std()>0 else 0
    flag = ' <<<' if r.mean() > 0 else ''
    print(f'{scenario:<25} {pnl.mean():>+9.2f} {(pnl>0).mean():>8.1%} {full_sh:>+8.2f} | {r.mean():>+9.2f} {(r>0).mean():>8.1%} {r_sh:>+8.2f} {r.sum():>+11.0f}{flag}')

print()
print('=' * 95)
print('PER-PAIR BREAKDOWN — realistic spread — full history and last 18m')
print()
sp_real = SPREADS['realistic (retail)']
spread_series = trades['pair'].map(sp_real)
pnl = trades['signal'] * trades['fwd'] - spread_series

print(f'{"Pair":<10} {"Trades":>7} {"WR":>7} {"EV":>8} {"Total":>9} | {"18mTrd":>7} {"18mWR":>7} {"18mEV":>8} {"18mTotal":>10}')
print('-' * 85)
for pair in sorted(trades['pair'].unique()):
    mask = trades['pair'] == pair
    p = pnl[mask].dropna()
    r = p[p.index >= cut18]
    flag = ' <<<' if (len(r)>3 and r.mean()>3) else ''
    print(f'{pair:<10} {len(p):>7,} {(p>0).mean():>7.1%} {p.mean():>+8.2f} {p.sum():>+9.0f} | {len(r):>7,} {(r>0).mean():>7.1%} {r.mean():>+8.2f} {r.sum():>+10.0f}{flag}')

print()
print('=' * 95)
print('BREAK-EVEN SPREAD ANALYSIS (what spread kills this system?)')
print()
# Find break-even spread per pair
raw_pnl = trades['signal'] * trades['fwd']  # zero spread
print(f'Avg gross PnL/trade (zero spread): {raw_pnl.mean():.2f} pips')
print(f'The system breaks even when spread = avg gross PnL = {raw_pnl.mean():.1f} pips per trade')
print()
print('Per-pair break-even spread:')
for pair in sorted(trades['pair'].unique()):
    mask = trades['pair'] == pair
    raw = raw_pnl[mask].dropna()
    real_sp = sp_real[pair]
    print(f'  {pair:<10} gross EV: {raw.mean():>+6.2f} pips  |  realistic spread: {real_sp:.1f}  |  break-even spread: {raw.mean():.1f}  |  margin: {raw.mean()-real_sp:>+5.1f} pips')
