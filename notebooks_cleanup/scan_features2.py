import pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')
from pathlib import Path

PIP_SIZE = {'EURUSD':0.0001,'GBPUSD':0.0001,'AUDUSD':0.0001,'NZDUSD':0.0001,
    'USDCAD':0.0001,'USDCHF':0.0001,'USDJPY':0.01,'EURJPY':0.01,'GBPJPY':0.01,
    'AUDJPY':0.01,'CADJPY':0.01,'CHFJPY':0.01,'EURAUD':0.0001,'EURGBP':0.0001,'AUDNZD':0.0001}
SPREAD_PIPS = {'EURUSD':1.0,'GBPUSD':1.5,'AUDUSD':1.5,'NZDUSD':2.0,'USDCAD':2.0,
    'USDCHF':2.0,'USDJPY':1.5,'EURJPY':2.0,'GBPJPY':3.0,'AUDJPY':3.0,
    'CADJPY':3.0,'CHFJPY':3.0,'EURAUD':3.0,'EURGBP':1.5,'AUDNZD':3.0}
MAX_1H_MOVE = {'EURUSD':150,'GBPUSD':200,'AUDUSD':120,'NZDUSD':100,'USDCAD':150,
    'USDCHF':150,'USDJPY':300,'EURJPY':300,'GBPJPY':350,'AUDJPY':250,
    'CADJPY':250,'CHFJPY':250,'EURAUD':150,'EURGBP':100,'AUDNZD':80}
F6 = Path('backend/data/features_6')
F8 = Path('backend/data/features_8')
HOLD_H = 4

dfs = []
for pair, pip in PIP_SIZE.items():
    df6 = pd.read_parquet(F6 / f'{pair}_features.parquet')
    df8 = pd.read_parquet(F8 / f'{pair}_geometric.parquet')
    df8['pair'] = pair
    df6r = df6.reset_index()
    df8r = df8.reset_index()
    idx = df6r.columns[0]
    drop_dup = [c for c in df8r.columns if c in df6r.columns and c not in [idx, 'pair']]
    df8r = df8r.drop(columns=drop_dup, errors='ignore')
    df = pd.merge(df6r, df8r, on=[idx, 'pair'], how='inner').set_index(idx).sort_index()
    df1h = pd.read_parquet(f'backend/data/processed/{pair}_1H.parquet')
    if 'datetime' in df1h.columns:
        df1h = df1h.set_index('datetime')
    df1h.index = pd.to_datetime(df1h.index)
    close = df1h['close'].reindex(df.index)
    fwd = (close.shift(-HOLD_H) - close) / pip
    ret_1h = ((close.shift(-1) - close) / pip).abs()
    holiday = ~(
        ((df.index.month == 12) & (df.index.day.isin([24, 25, 26, 31]))) |
        ((df.index.month == 1) & (df.index.day.isin([1, 2])))
    )
    clean = (ret_1h.reindex(df.index) <= MAX_1H_MOVE[pair]) & holiday
    liquid = (df.index.hour >= 7) & (df.index.hour <= 21)
    df = df[clean & liquid].copy()
    df['fwd_pips'] = fwd
    df['spread'] = SPREAD_PIPS[pair]
    dfs.append(df)

big = pd.concat(dfs).dropna(subset=['fwd_pips', 'realized_skew', 'residual_12h', 'rv_zscore_24', 'kyle_lambda_delta_3h'])

rank_cols = ['realized_skew', 'residual_12h', 'rv_zscore_24', 'kyle_lambda_delta_3h',
             'kyle_lambda_delta_6h', 'jump_asymmetry', 'entropy_norm', 'accel_mean',
             'momentum_shift', 'info_accel', 'residual_6h', 'residual_24h',
             'slope_close_3h', 'slope_close_6h', 'vpin_4h', 'vpin_12h', 'rv_zscore_24']
for col in rank_cols:
    if col in big.columns:
        big[f'{col}_r'] = big.groupby('pair')[col].rank(pct=True)


def test(name, lm, sm):
    sig = pd.Series(0, index=big.index)
    sig[lm] = 1
    sig[sm] = -1
    pnl = sig * big['fwd_pips'] - big['spread'] * sig.abs()
    t = pnl[sig != 0].dropna()
    if len(t) < 30:
        print(f'{name}: too few ({len(t)})')
        return
    nm = (t.index.max() - t.index.min()).days / 30
    sh = (t.mean() / t.std()) * np.sqrt(252 * 24 / HOLD_H)
    cut = t.index.max() - pd.DateOffset(months=18)
    r = t[t.index >= cut]
    sh_r = (r.mean() / r.std()) * np.sqrt(252 * 24 / HOLD_H) if len(r) > 10 and r.std() > 0 else 0
    flag = ' <<<' if r.mean() > 0.5 else ''
    print(f'{name:<55} {len(t):>6,} ({len(t)/nm:>3.0f}/mo) WR:{(t>0).mean():>5.1%} EV:{t.mean():>+5.2f} Sh:{sh:>+4.2f} | 18m EV:{r.mean():>+5.2f} Sh:{sh_r:>+4.2f}{flag}')


BASE_L = (big['kyle_lambda_delta_3h_r'] > 0.70) & (big['residual_12h_r'] < 0.25) & (big['rv_zscore_24_r'] > 0.70)
BASE_S = (big['kyle_lambda_delta_3h_r'] < 0.30) & (big['residual_12h_r'] > 0.75) & (big['rv_zscore_24_r'] > 0.70)

print('=== BASELINE ===')
test('Kyle70+Resid25+rv_z70 (baseline)', BASE_L, BASE_S)

print()
print('=== ADD SKEW FILTER ===')
for thresh in [0.30, 0.35, 0.40, 0.45]:
    test(f'+ skew_r < {thresh}',
         BASE_L & (big['realized_skew_r'] < thresh),
         BASE_S & (big['realized_skew_r'] > 1 - thresh))

print()
print('=== ADD JUMP ASYMMETRY ===')
for thresh in [0.55, 0.60, 0.65, 0.70]:
    test(f'+ jump_asym_r > {thresh} (downside jumps = buy dip)',
         BASE_L & (big['jump_asymmetry_r'] > thresh),
         BASE_S & (big['jump_asymmetry_r'] < 1 - thresh))

print()
print('=== RESIDUAL TIMEFRAME ===')
test('residual_6h instead of 12h',
     (big['kyle_lambda_delta_3h_r'] > 0.70) & (big['residual_6h_r'] < 0.25) & (big['rv_zscore_24_r'] > 0.70),
     (big['kyle_lambda_delta_3h_r'] < 0.30) & (big['residual_6h_r'] > 0.75) & (big['rv_zscore_24_r'] > 0.70))
test('residual_24h instead of 12h',
     (big['kyle_lambda_delta_3h_r'] > 0.70) & (big['residual_24h_r'] < 0.25) & (big['rv_zscore_24_r'] > 0.70),
     (big['kyle_lambda_delta_3h_r'] < 0.30) & (big['residual_24h_r'] > 0.75) & (big['rv_zscore_24_r'] > 0.70))

print()
print('=== SLOPE CONFIRMATION ===')
test('+ slope_3h_r < 0.35 (momentum waning)',
     BASE_L & (big['slope_close_3h_r'] < 0.35),
     BASE_S & (big['slope_close_3h_r'] > 0.65))
if 'slope_close_6h_r' in big.columns:
    test('+ slope_6h_r < 0.35',
         BASE_L & (big['slope_close_6h_r'] < 0.35),
         BASE_S & (big['slope_close_6h_r'] > 0.65))

print()
print('=== KYLE DELTA TIMEFRAME ===')
test('kyle_delta_6h instead of 3h',
     (big['kyle_lambda_delta_6h_r'] > 0.70) & (big['residual_12h_r'] < 0.25) & (big['rv_zscore_24_r'] > 0.70),
     (big['kyle_lambda_delta_6h_r'] < 0.30) & (big['residual_12h_r'] > 0.75) & (big['rv_zscore_24_r'] > 0.70))

print()
print('=== VOL THRESHOLD ===')
test('rv_z threshold 80',
     (big['kyle_lambda_delta_3h_r'] > 0.70) & (big['residual_12h_r'] < 0.25) & (big['rv_zscore_24_r'] > 0.80),
     (big['kyle_lambda_delta_3h_r'] < 0.30) & (big['residual_12h_r'] > 0.75) & (big['rv_zscore_24_r'] > 0.80))
test('rv_z threshold 85',
     (big['kyle_lambda_delta_3h_r'] > 0.70) & (big['residual_12h_r'] < 0.25) & (big['rv_zscore_24_r'] > 0.85),
     (big['kyle_lambda_delta_3h_r'] < 0.30) & (big['residual_12h_r'] > 0.75) & (big['rv_zscore_24_r'] > 0.85))
test('rv_z threshold 90',
     (big['kyle_lambda_delta_3h_r'] > 0.70) & (big['residual_12h_r'] < 0.25) & (big['rv_zscore_24_r'] > 0.90),
     (big['kyle_lambda_delta_3h_r'] < 0.30) & (big['residual_12h_r'] > 0.75) & (big['rv_zscore_24_r'] > 0.90))

print()
print('=== PER-PAIR BREAKDOWN (baseline) ===')
sig = pd.Series(0, index=big.index)
sig[BASE_L] = 1
sig[BASE_S] = -1
pnl = sig * big['fwd_pips'] - big['spread'] * sig.abs()
trades_df = big[sig != 0].copy()
trades_df['pnl'] = pnl[sig != 0]
cut = trades_df.index.max() - pd.DateOffset(months=18)
print(f'{"Pair":<10} {"AllTrd":>7} {"AllEV":>7} {"18mTrd":>7} {"18mEV":>7} {"18mSh":>7}')
for pair in sorted(trades_df['pair'].unique()):
    p = trades_df[trades_df['pair'] == pair]['pnl'].dropna()
    r = p[p.index >= cut]
    sh_r = (r.mean() / r.std()) * np.sqrt(252 * 24 / HOLD_H) if len(r) > 5 and r.std() > 0 else 0
    flag = ' <<<' if r.mean() > 0 else ''
    print(f'{pair:<10} {len(p):>7,} {p.mean():>+7.2f} {len(r):>7,} {r.mean():>+7.2f} {sh_r:>+7.2f}{flag}')
