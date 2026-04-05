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
F6 = Path('backend/data/features_6'); F8 = Path('backend/data/features_8')
HOLD_H = 4

dfs = []
for pair, pip in PIP_SIZE.items():
    df6 = pd.read_parquet(F6 / f'{pair}_features.parquet')
    df8 = pd.read_parquet(F8 / f'{pair}_geometric.parquet'); df8['pair'] = pair
    df6r = df6.reset_index(); df8r = df8.reset_index(); idx = df6r.columns[0]
    df8r = df8r.drop(columns=[c for c in df8r.columns if c in df6r.columns and c not in [idx, 'pair']], errors='ignore')
    df = pd.merge(df6r, df8r, on=[idx, 'pair'], how='inner').set_index(idx).sort_index()
    df1h = pd.read_parquet(f'backend/data/processed/{pair}_1H.parquet')
    if 'datetime' in df1h.columns: df1h = df1h.set_index('datetime')
    df1h.index = pd.to_datetime(df1h.index)
    close = df1h['close'].reindex(df.index)
    fwd = (close.shift(-HOLD_H) - close) / pip
    ret_1h = ((close.shift(-1) - close) / pip).abs()
    holiday = ~(((df.index.month==12) & (df.index.day.isin([24,25,26,31]))) | ((df.index.month==1) & (df.index.day.isin([1,2]))))
    clean = (ret_1h.reindex(df.index) <= MAX_1H_MOVE[pair]) & holiday
    liquid = (df.index.hour >= 7) & (df.index.hour <= 21)
    df = df[clean & liquid].copy(); df['fwd_pips'] = fwd; df['spread'] = SPREAD_PIPS[pair]
    dfs.append(df)

big = pd.concat(dfs).dropna(subset=['fwd_pips', 'realized_skew', 'residual_12h', 'rv_zscore_24', 'kyle_lambda_delta_3h'])

for col in ['realized_skew', 'residual_12h', 'rv_zscore_24', 'kyle_lambda_delta_3h',
            'hurst_6h', 'vr_5', 'vr_z5', 'autocorr_1', 'compression_6_24',
            'atr_ratio_6_24', 'hurst_change', 'slope_close_3h', 'vpin_4h']:
    if col in big.columns:
        big[f'{col}_r'] = big.groupby('pair')[col].rank(pct=True)

BASE_L = (big['kyle_lambda_delta_3h_r'] > 0.70) & (big['residual_12h_r'] < 0.25) & (big['rv_zscore_24_r'] > 0.85) & (big['realized_skew_r'] < 0.30)
BASE_S = (big['kyle_lambda_delta_3h_r'] < 0.30) & (big['residual_12h_r'] > 0.75) & (big['rv_zscore_24_r'] > 0.85) & (big['realized_skew_r'] > 0.70)


def report(name, lm, sm, show_years=False, show_pairs=False):
    sig = pd.Series(0, index=big.index)
    sig[lm] = 1; sig[sm] = -1
    pnl = sig * big['fwd_pips'] - big['spread'] * sig.abs()
    t = pnl[sig != 0].dropna()
    if len(t) < 20:
        print(f'{name}: too few ({len(t)})')
        return
    nm = (t.index.max() - t.index.min()).days / 30
    sh = (t.mean() / t.std()) * np.sqrt(252 * 24 / HOLD_H)
    cut = t.index.max() - pd.DateOffset(months=18)
    r = t[t.index >= cut]
    sh_r = (r.mean() / r.std()) * np.sqrt(252 * 24 / HOLD_H) if len(r) > 10 and r.std() > 0 else 0
    wins = t[t > 0]; losses = t[t <= 0]
    flag = ' <<<' if r.mean() > 5.0 else (' <<' if r.mean() > 2.0 else '')
    print(f'{name:<60} {len(t):>5,} ({len(t)/nm:>3.0f}/mo)  WR:{(t>0).mean():>5.1%}  AvgW:{wins.mean():>5.1f}  AvgL:{losses.mean():>5.1f}  EV:{t.mean():>+6.2f}  Sh:{sh:>+5.2f} | 18m EV:{r.mean():>+6.2f}  Sh:{sh_r:>+5.2f}{flag}')
    if show_years:
        print(f'  {"Year":>6} {"Trades":>7} {"WR":>7} {"EV":>8} {"Total":>9}')
        for yr in sorted(t.index.year.unique()):
            yt = t[t.index.year == yr]
            neg = ' neg' if yt.mean() < -2 else ''
            print(f'  {yr:>6} {len(yt):>7,} {(yt>0).mean():>7.1%} {yt.mean():>+8.2f} {yt.sum():>+9.0f}{neg}')
    if show_pairs:
        trades_df = big[sig != 0].copy(); trades_df['pnl'] = pnl[sig != 0]
        p18 = trades_df[trades_df.index >= cut]
        print(f'  {"Pair":<10} {"18mTrd":>7} {"18mEV":>8} {"18mSh":>8}')
        for pair in sorted(p18['pair'].unique()):
            pp = p18[p18['pair'] == pair]['pnl'].dropna()
            sh_p = (pp.mean() / pp.std()) * np.sqrt(252 * 24 / HOLD_H) if len(pp) > 5 and pp.std() > 0 else 0
            flag2 = ' <<<' if pp.mean() > 3 else ''
            print(f'  {pair:<10} {len(pp):>7,} {pp.mean():>+8.2f} {sh_p:>+8.2f}{flag2}')


print('=' * 100)
print('BASELINE')
report('baseline (Kyle70+Resid25+rv_z85+skew30)', BASE_L, BASE_S, show_years=True)

print()
print('=' * 100)
print('BEST SINGLE REGIME ADDITIONS')
report('+ vr_5_r < 0.30 (strong MR regime)',
       BASE_L & (big['vr_5_r'] < 0.30), BASE_S & (big['vr_5_r'] < 0.30))
report('+ compression_6_24_r < 0.25 (vol squeeze)',
       BASE_L & (big['compression_6_24_r'] < 0.25), BASE_S & (big['compression_6_24_r'] < 0.25))
report('+ hurst_r < 0.40 + vr_5_r < 0.40 (dual MR)',
       BASE_L & (big['hurst_6h_r'] < 0.40) & (big['vr_5_r'] < 0.40),
       BASE_S & (big['hurst_6h_r'] < 0.40) & (big['vr_5_r'] < 0.40))

print()
print('=' * 100)
print('VR THRESHOLD SCAN + COMBINATIONS')
for vr_t in [0.25, 0.30, 0.35, 0.40]:
    report(f'+ vr_5_r<{vr_t} + hurst_r<0.40',
           BASE_L & (big['vr_5_r'] < vr_t) & (big['hurst_6h_r'] < 0.40),
           BASE_S & (big['vr_5_r'] < vr_t) & (big['hurst_6h_r'] < 0.40))

print()
for vr_t in [0.25, 0.30, 0.35]:
    report(f'+ vr_5_r<{vr_t} + compression<0.30',
           BASE_L & (big['vr_5_r'] < vr_t) & (big['compression_6_24_r'] < 0.30),
           BASE_S & (big['vr_5_r'] < vr_t) & (big['compression_6_24_r'] < 0.30))

print()
print('=' * 100)
print('BEST COMBINATIONS — DETAILED VIEW')

# The jewel: vr_5 low = strong mean-reverting
report('vr_5_r<0.30 alone (no baseline)',
       big['vr_5_r'] < 0.30, pd.Series(False, index=big.index))  # just long to check

best_l = BASE_L & (big['vr_5_r'] < 0.30)
best_s = BASE_S & (big['vr_5_r'] < 0.30)
report('BEST: baseline + vr_5_r<0.30', best_l, best_s, show_years=True, show_pairs=True)

print()
best2_l = BASE_L & (big['hurst_6h_r'] < 0.40) & (big['vr_5_r'] < 0.40)
best2_s = BASE_S & (big['hurst_6h_r'] < 0.40) & (big['vr_5_r'] < 0.40)
report('BEST2: baseline + hurst<40 + vr<40', best2_l, best2_s, show_years=True, show_pairs=True)
