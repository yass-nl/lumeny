"""
Scan for directional information CONDITIONAL on the system signal firing.
We already know a reversion is likely -- now which way?

For each feature, we measure:
- Among signal bars, does feature value predict whether next 4H goes UP or DOWN?
- We use both raw value and the direction it implies
- Key: we want features that work WITHIN the signal population, not globally
"""
import pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')
from pathlib import Path
from scipy import stats

PIP_SIZE = {'AUDJPY':0.01,'AUDNZD':0.0001,'AUDUSD':0.0001,'CADJPY':0.01,'CHFJPY':0.01,
    'EURAUD':0.0001,'EURGBP':0.0001,'EURJPY':0.01,'EURUSD':0.0001,'NZDUSD':0.0001,'USDCAD':0.0001}
MAX_1H_MOVE = {'AUDJPY':250,'AUDNZD':80,'AUDUSD':120,'CADJPY':250,'CHFJPY':250,
    'EURAUD':150,'EURGBP':100,'EURJPY':300,'EURUSD':150,'NZDUSD':100,'USDCAD':150}
F6 = Path('backend/data/features_6'); F8 = Path('backend/data/features_8')
PROC = Path('backend/data/processed')
ENTRY_START, ENTRY_END = 7, 16
HOLD_H = 4

# ── Load ──────────────────────────────────────────────────────────────────────
dfs = []
for pair, pip in PIP_SIZE.items():
    df6 = pd.read_parquet(F6/f'{pair}_features.parquet')
    df8 = pd.read_parquet(F8/f'{pair}_geometric.parquet'); df8['pair'] = pair
    df6r = df6.reset_index(); df8r = df8.reset_index(); idx = df6r.columns[0]
    df8r = df8r.drop(columns=[c for c in df8r.columns if c in df6r.columns and c not in [idx,'pair']],errors='ignore')
    df = pd.merge(df6r,df8r,on=[idx,'pair'],how='inner').set_index(idx).sort_index()
    df1h = pd.read_parquet(PROC/f'{pair}_1H.parquet')
    if 'datetime' in df1h.columns: df1h = df1h.set_index('datetime')
    df1h.index = pd.to_datetime(df1h.index)
    close = df1h['close'].reindex(df.index)
    fwd = (close.shift(-HOLD_H) - close) / pip
    ret_1h = ((close.shift(-1) - close) / pip).abs().reindex(df.index)
    holiday = ~(((df.index.month==12)&(df.index.day.isin([24,25,26,31])))|((df.index.month==1)&(df.index.day.isin([1,2]))))
    clean = (ret_1h <= MAX_1H_MOVE[pair]) & holiday
    df = df[clean].copy()
    df['fwd_pips'] = fwd
    df['close'] = close
    dfs.append(df)

data = pd.concat(dfs).dropna(subset=['fwd_pips','realized_skew','residual_12h','rv_zscore_24','kyle_lambda_delta_3h','vr_5'])

# Per-pair ranks for signal conditions
for col in ['kyle_lambda_delta_3h','residual_12h','rv_zscore_24','realized_skew','vr_5']:
    data[f'{col}_r'] = data.groupby('pair')[col].rank(pct=True)

# ── Isolate signal population ─────────────────────────────────────────────────
in_window = (data.index.hour >= ENTRY_START) & (data.index.hour <= ENTRY_END)

# Long candidates: conditions say "revert UP"
long_cond = (data['kyle_lambda_delta_3h_r']>0.70)&(data['residual_12h_r']<0.25)&\
            (data['rv_zscore_24_r']>0.85)&(data['realized_skew_r']<0.30)&(data['vr_5_r']<0.30)
# Short candidates: conditions say "revert DOWN"
short_cond = (data['kyle_lambda_delta_3h_r']<0.30)&(data['residual_12h_r']>0.75)&\
             (data['rv_zscore_24_r']>0.85)&(data['realized_skew_r']>0.70)&(data['vr_5_r']<0.30)

data['_direction'] = 0
data.loc[long_cond & in_window, '_direction'] = 1
data.loc[short_cond & in_window, '_direction'] = -1
signals = data[data['_direction'] != 0].copy().reset_index(drop=True)
signals['direction'] = signals['_direction']
data.drop(columns=['_direction'], inplace=True)

# Label: did price actually go in the signal direction?
signals['correct'] = (signals['fwd_pips'] * signals['direction']) > 0
print(f'Signal population: {len(signals):,} bars')
print(f'Long: {(signals["direction"]==1).sum():,}  Short: {(signals["direction"]==-1).sum():,}')
print(f'Base win rate (direction correct): {signals["correct"].mean():.1%}')
print()

# ── Features to test ─────────────────────────────────────────────────────────
# Exclude: signal conditions themselves, time features, pair, forward-looking
skip = {'pair','fwd_pips','close','direction','correct',
        'kyle_lambda_delta_3h_r','residual_12h_r','rv_zscore_24_r','realized_skew_r','vr_5_r',
        'hour_sin','hour_cos','dow_sin','dow_cos','is_london','is_ny','is_overlap','is_asia',
        'label_1H','mfe_long_pips','mfe_short_pips','trail_long_bars','trail_short_bars',
        'mfe_atr_24','rv','entropy_norm_delta_6h','hurst_6h_delta_6h'}

# Also skip intrabar direction features (describe current bar, not predictive lag)
skip_patterns = ['intrabar_','bar_direction','candle_direction','body_ratio',
                 'upper_wick','lower_wick','wick_asymmetry','prev_close_pos','prev_body_ratio',
                 'consec_bull','consec_bear','consec_bullish','consec_bearish','close_position']

feat_cols = []
for c in signals.columns:
    if c in skip: continue
    if any(p in c for p in skip_patterns): continue
    if not pd.api.types.is_numeric_dtype(signals[c]): continue
    feat_cols.append(c)

print(f'Testing {len(feat_cols)} features for directional signal within the signal population...')
print()

# ── For each feature: does it discriminate correct vs wrong direction? ────────
# Key insight: for LONG signals, feature X might say "yes, go long" or "no, fade"
# We test: within long signals, does higher feature value -> more likely correct?
# Within short signals, does lower feature value -> more likely correct?

results = []
for col in feat_cols:
    try:
        vals = signals[col].dropna()
        if len(vals) < 50: continue
        sig_sub = signals.loc[vals.index]

        # Rank within signal population (pair-agnostic for now)
        ranks = vals.rank(pct=True)

        # Test 1: point-biserial correlation between feature rank and correctness
        corr, pval = stats.pointbiserialr(ranks, sig_sub['correct'])

        # Test 2: within LONG signals specifically
        long_sub = sig_sub[sig_sub['direction'] == 1]
        if len(long_sub) > 20:
            long_vals = vals.reindex(long_sub.index).dropna()
            long_correct = long_sub['correct'].reindex(long_vals.index)
            long_corr, long_pval = stats.pointbiserialr(long_vals.rank(pct=True), long_correct)
            # Win rate when feature is high vs low (top/bot 30%)
            hi = long_vals[long_vals >= long_vals.quantile(0.70)]
            lo = long_vals[long_vals <= long_vals.quantile(0.30)]
            wr_hi = long_correct.reindex(hi.index).mean()
            wr_lo = long_correct.reindex(lo.index).mean()
        else:
            long_corr, long_pval, wr_hi, wr_lo = 0, 1, 0.5, 0.5

        results.append({
            'feature': col,
            'corr_all': corr,
            'abs_corr': abs(corr),
            'pval': pval,
            'long_corr': long_corr,
            'long_pval': long_pval,
            'wr_hi_feat': wr_hi,   # win rate when feature high (on long signals)
            'wr_lo_feat': wr_lo,   # win rate when feature low (on long signals)
            'wr_spread': wr_hi - wr_lo,  # discrimination power
            'abs_wr_spread': abs(wr_hi - wr_lo),
        })
    except Exception as e:
        print(f'  ERROR {col}: {e}')
        pass

res = pd.DataFrame(results).sort_values('abs_wr_spread', ascending=False)

print('TOP FEATURES BY WIN-RATE SPREAD (within long signal population)')
print('wr_hi = win rate when feature is HIGH | wr_lo = win rate when feature is LOW')
print('wr_spread > 0 means HIGH feature value = more likely correct long')
print()
print(f'{"Feature":<35} {"WR_hi":>8} {"WR_lo":>8} {"Spread":>8} {"LongCorr":>10} {"pval":>10}  Interpretation')
print('-'*105)
for _, row in res.head(40).iterrows():
    interp = 'HIGH=bullish' if row['wr_spread'] > 0 else 'HIGH=bearish'
    sig_flag = ' *' if row['long_pval'] < 0.05 else ''
    print(f'{row["feature"]:<35} {row["wr_hi_feat"]:>8.1%} {row["wr_lo_feat"]:>8.1%} {row["wr_spread"]:>+8.1%} {row["long_corr"]:>+10.3f} {row["long_pval"]:>10.3f}  {interp}{sig_flag}')

print()
print('='*105)
print('TOP FEATURES SORTED BY STATISTICAL SIGNIFICANCE (p < 0.05)')
sig_res = res[res['long_pval'] < 0.05].sort_values('abs_wr_spread', ascending=False)
print(f'{"Feature":<35} {"WR_hi":>8} {"WR_lo":>8} {"Spread":>8} {"LongCorr":>10} {"pval":>10}  Interpretation')
print('-'*105)
for _, row in sig_res.head(20).iterrows():
    interp = 'HIGH=bullish' if row['wr_spread'] > 0 else 'HIGH=bearish'
    print(f'{row["feature"]:<35} {row["wr_hi_feat"]:>8.1%} {row["wr_lo_feat"]:>8.1%} {row["wr_spread"]:>+8.1%} {row["long_corr"]:>+10.3f} {row["long_pval"]:>10.3f}  {interp}')

print()
print('='*105)
print('WITHIN-POPULATION WIN RATE BY QUANTILE — top 10 features')
for _, row in sig_res.head(10).iterrows():
    col = row['feature']
    long_sub = signals[signals['direction'] == 1].copy()
    vals = long_sub[col].dropna()
    long_sub = long_sub.loc[vals.index]
    print(f'\n  {col}  (spread: {row["wr_spread"]:+.1%})')
    for lo_q, hi_q in [(0,0.2),(0.2,0.4),(0.4,0.6),(0.6,0.8),(0.8,1.0)]:
        mask = (vals.rank(pct=True) >= lo_q) & (vals.rank(pct=True) < hi_q)
        wr = long_sub.loc[mask[mask].index, 'correct'].mean()
        n  = mask.sum()
        bar = '█' * int(wr * 20)
        print(f'    Q{lo_q:.0%}-{hi_q:.0%}: WR {wr:.1%} (n={n:,})  {bar}')
