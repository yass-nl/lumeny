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
    df = df[clean & liquid].copy()
    df['fwd_pips'] = fwd
    df['spread'] = SPREAD_PIPS[pair]
    dfs.append(df)

big = pd.concat(dfs).dropna(subset=['fwd_pips', 'realized_skew', 'residual_12h', 'rv_zscore_24', 'kyle_lambda_delta_3h'])

# Rank all features per-pair
rank_cols = [
    'realized_skew', 'residual_12h', 'rv_zscore_24', 'kyle_lambda_delta_3h',
    # regime candidates
    'hurst_6h', 'vr_5', 'vr_z5', 'autocorr_1', 'fractal_dim_6h',
    'compression_6_24', 'curvature_12h', 'curvature_6h', 'atr_ratio_6_24',
    'vol_clustering_ac1', 'entropy_norm', 'vpin_4h', 'vpin_12h',
    'hurst_change', 'vr_5_delta_3h', 'vr_5_delta_6h',
    'hurst_6h_delta_3h', 'hurst_6h_delta_12h',
    'noise_to_signal', 'runs_z', 'sum_abs_autocorr',
    'atr_ratio_6_72', 'vol_of_vol', 'entropy_volume_divergence',
    'compression_12_48', '4h_compression', 'envelope_squeeze_12h',
    'slope_close_3h', 'slope_close_6h',
]
for col in rank_cols:
    if col in big.columns:
        big[f'{col}_r'] = big.groupby('pair')[col].rank(pct=True)

# Baseline: the system we found
BASE_L = (big['kyle_lambda_delta_3h_r'] > 0.70) & (big['residual_12h_r'] < 0.25) & (big['rv_zscore_24_r'] > 0.85) & (big['realized_skew_r'] < 0.30)
BASE_S = (big['kyle_lambda_delta_3h_r'] < 0.30) & (big['residual_12h_r'] > 0.75) & (big['rv_zscore_24_r'] > 0.85) & (big['realized_skew_r'] > 0.70)


def test(name, lm, sm):
    sig = pd.Series(0, index=big.index)
    sig[lm] = 1; sig[sm] = -1
    pnl = sig * big['fwd_pips'] - big['spread'] * sig.abs()
    t = pnl[sig != 0].dropna()
    if len(t) < 20:
        print(f'{name:<55} too few trades ({len(t)})')
        return
    nm = (t.index.max() - t.index.min()).days / 30
    sh = (t.mean() / t.std()) * np.sqrt(252 * 24 / HOLD_H)
    cut = t.index.max() - pd.DateOffset(months=18)
    r = t[t.index >= cut]
    sh_r = (r.mean() / r.std()) * np.sqrt(252 * 24 / HOLD_H) if len(r) > 10 and r.std() > 0 else 0
    flag = ' <<<' if r.mean() > 2.0 else ''
    print(f'{name:<55} {len(t):>6,} ({len(t)/nm:>3.0f}/mo)  WR:{(t>0).mean():>5.1%}  EV:{t.mean():>+5.2f}  Sh:{sh:>+5.2f} | 18m EV:{r.mean():>+5.2f}  Sh:{sh_r:>+5.2f}{flag}')


print('BASELINE (Kyle70+Resid25+rv_z85+skew30)')
test('baseline', BASE_L, BASE_S)
print()

# ── 1. HURST: mean-reverting regime (H < 0.5 = mean-reverting, our edge is reversion)
print('=== HURST (mean-reverting regime) ===')
for thresh in [0.30, 0.35, 0.40, 0.45, 0.50]:
    test(f'+ hurst_r < {thresh} (MR regime)',
         BASE_L & (big['hurst_6h_r'] < thresh),
         BASE_S & (big['hurst_6h_r'] < thresh))

print()

# ── 2. VARIANCE RATIO: vr_5 < 1 means mean-reverting
print('=== VARIANCE RATIO (MR regime) ===')
for thresh in [0.30, 0.35, 0.40, 0.45, 0.50]:
    test(f'+ vr_5_r < {thresh} (MR regime)',
         BASE_L & (big['vr_5_r'] < thresh),
         BASE_S & (big['vr_5_r'] < thresh))

print()

# ── 3. AUTOCORR: negative = mean-reverting
print('=== AUTOCORR (negative = MR) ===')
for thresh in [0.25, 0.30, 0.35, 0.40, 0.45]:
    test(f'+ autocorr_1_r < {thresh} (negative autocorr)',
         BASE_L & (big['autocorr_1_r'] < thresh),
         BASE_S & (big['autocorr_1_r'] < thresh))

print()

# ── 4. CURVATURE: high curvature = turning point
print('=== CURVATURE (turning point signal) ===')
# curvature_12h > 0 means concave up (bottom), < 0 = concave down (top)
# for long: want curvature_12h > 0 (curving upward = bottom forming)
# for short: want curvature_12h < 0 (curving downward = top forming)
for thresh in [0.55, 0.60, 0.65, 0.70, 0.75]:
    test(f'+ curvature_12h directional (r>{thresh} long, r<{1-thresh:.2f} short)',
         BASE_L & (big['curvature_12h_r'] > thresh),
         BASE_S & (big['curvature_12h_r'] < 1 - thresh))

print()

# ── 5. COMPRESSION: low compression = volatility squeeze about to expand
print('=== COMPRESSION (squeeze) ===')
for thresh in [0.25, 0.30, 0.35, 0.40]:
    test(f'+ compression_6_24_r < {thresh} (vol squeeze)',
         BASE_L & (big['compression_6_24_r'] < thresh),
         BASE_S & (big['compression_6_24_r'] < thresh))

print()

# ── 6. ATR RATIO: short vol expanding vs long vol
print('=== ATR RATIO (vol expansion) ===')
for thresh in [0.60, 0.65, 0.70, 0.75, 0.80]:
    test(f'+ atr_ratio_6_24_r > {thresh} (vol expanding)',
         BASE_L & (big['atr_ratio_6_24_r'] > thresh),
         BASE_S & (big['atr_ratio_6_24_r'] > thresh))

print()

# ── 7. VPIN: elevated informed trading
print('=== VPIN (informed flow) ===')
for thresh in [0.55, 0.60, 0.65, 0.70, 0.75]:
    test(f'+ vpin_4h_r > {thresh}',
         BASE_L & (big['vpin_4h_r'] > thresh),
         BASE_S & (big['vpin_4h_r'] > thresh))

print()

# ── 8. HURST CHANGE: hurst falling = transitioning to MR
print('=== HURST CHANGE (transitioning to MR) ===')
for thresh in [0.25, 0.30, 0.35, 0.40]:
    test(f'+ hurst_change_r < {thresh} (hurst falling)',
         BASE_L & (big['hurst_change_r'] < thresh),
         BASE_S & (big['hurst_change_r'] < thresh))

print()

# ── 9. COMBINATIONS of best regime filters
print('=== REGIME COMBINATIONS ===')
# MR regime: hurst low + vr low + autocorr negative
mr_regime = (big['hurst_6h_r'] < 0.40) & (big['vr_5_r'] < 0.40)
test('baseline + MR regime (hurst<40 & vr<40)',
     BASE_L & mr_regime, BASE_S & mr_regime)

mr_regime2 = (big['hurst_6h_r'] < 0.35) & (big['autocorr_1_r'] < 0.40)
test('baseline + MR regime (hurst<35 & autocorr<40)',
     BASE_L & mr_regime2, BASE_S & mr_regime2)

# Curvature + MR
test('baseline + curvature directional + hurst<40',
     BASE_L & (big['curvature_12h_r'] > 0.60) & (big['hurst_6h_r'] < 0.40),
     BASE_S & (big['curvature_12h_r'] < 0.40) & (big['hurst_6h_r'] < 0.40))

# Vol expansion in MR regime
test('baseline + atr_expand + hurst<40',
     BASE_L & (big['atr_ratio_6_24_r'] > 0.65) & (big['hurst_6h_r'] < 0.40),
     BASE_S & (big['atr_ratio_6_24_r'] > 0.65) & (big['hurst_6h_r'] < 0.40))

# Full regime combo
test('baseline + hurst<40 + vr<40 + curvature directional',
     BASE_L & (big['hurst_6h_r'] < 0.40) & (big['vr_5_r'] < 0.40) & (big['curvature_12h_r'] > 0.60),
     BASE_S & (big['hurst_6h_r'] < 0.40) & (big['vr_5_r'] < 0.40) & (big['curvature_12h_r'] < 0.40))
