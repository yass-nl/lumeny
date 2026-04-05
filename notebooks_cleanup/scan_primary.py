"""
Test autocorr_1, info_accel, accel_skew as PRIMARY directional signal.
No reversion framework. Pure direction prediction.
Entry 7-16 UTC, stop/target exit, exit before 20 UTC.
"""
import pandas as pd, numpy as np, warnings, itertools
warnings.filterwarnings('ignore')
from pathlib import Path

PIP_SIZE = {'AUDJPY':0.01,'AUDNZD':0.0001,'AUDUSD':0.0001,'CADJPY':0.01,'CHFJPY':0.01,
    'EURAUD':0.0001,'EURGBP':0.0001,'EURJPY':0.01,'EURUSD':0.0001,'NZDUSD':0.0001,'USDCAD':0.0001}
SPREAD_PIPS = {'AUDJPY':3.0,'AUDNZD':3.0,'AUDUSD':1.5,'CADJPY':3.0,'CHFJPY':3.0,
    'EURAUD':3.0,'EURGBP':1.5,'EURJPY':2.0,'EURUSD':1.0,'NZDUSD':2.0,'USDCAD':2.0}
MAX_1H_MOVE = {'AUDJPY':250,'AUDNZD':80,'AUDUSD':120,'CADJPY':250,'CHFJPY':250,
    'EURAUD':150,'EURGBP':100,'EURJPY':300,'EURUSD':150,'NZDUSD':100,'USDCAD':150}
F6  = Path('backend/data/features_6')
F8  = Path('backend/data/features_8')
PROC = Path('backend/data/processed')
ENTRY_START, ENTRY_END = 7, 16
MAX_HOLD = 8

# ── Load ──────────────────────────────────────────────────────────────────────
pair_data = {}
for pair, pip in PIP_SIZE.items():
    df6 = pd.read_parquet(F6/f'{pair}_features.parquet')
    df8 = pd.read_parquet(F8/f'{pair}_geometric.parquet'); df8['pair'] = pair
    df6r = df6.reset_index(); df8r = df8.reset_index(); idx = df6r.columns[0]
    df8r = df8r.drop(columns=[c for c in df8r.columns if c in df6r.columns and c not in [idx,'pair']],errors='ignore')
    df = pd.merge(df6r,df8r,on=[idx,'pair'],how='inner').set_index(idx).sort_index()
    df1h = pd.read_parquet(PROC/f'{pair}_1H.parquet')
    if 'datetime' in df1h.columns: df1h = df1h.set_index('datetime')
    df1h.index = pd.to_datetime(df1h.index)
    df['close'] = df1h['close'].reindex(df.index)
    df['open']  = df1h['open'].reindex(df.index)
    df['high']  = df1h['high'].reindex(df.index)
    df['low']   = df1h['low'].reindex(df.index)
    ret_1h = ((df1h['close'].shift(-1)-df1h['close'])/pip).abs().reindex(df.index)
    holiday = ~(((df.index.month==12)&(df.index.day.isin([24,25,26,31])))|((df.index.month==1)&(df.index.day.isin([1,2]))))
    df = df[holiday & (ret_1h<=MAX_1H_MOVE[pair])].copy()
    df = df.dropna(subset=['close','autocorr_1','info_accel','accel_skew'])
    # Rank all useful features per pair
    rank_cols = [
        'autocorr_1','info_accel','accel_skew','accel_mean','momentum_shift',
        'noise_to_signal','entropy_volume_divergence','sum_abs_autocorr',
        'kyle_lambda_delta_3h','kyle_lambda','kyle_lambda_change',
        'order_imbalance','order_imbalance_delta_3h','order_imbalance_delta_6h','order_imbalance_delta_12h',
        'hurst_6h','vr_5','vr_z5','fractal_dim_6h',
        'rv_zscore_24','atr_ratio_6_24','vol_of_vol',
        'slope_close_3h','slope_close_6h','slope_close_12h','slope_close_24h',
        'curvature_6h','curvature_12h','curvature_24h',
        'residual_6h','residual_12h','residual_24h',
        'range_pos_24h','range_pos_24','range_pos_48h',
        'realized_skew','realized_kurt',
        'jump_ratio_delta_3h','jump_ratio_delta_6h',
        'vpin_4h','vpin_12h',
        'entropy_norm','entropy_change_6h',
        '4h_close_pos','4h_residual','4h_slope_8h',
    ]
    for col in rank_cols:
        if col in df.columns:
            df[f'{col}_r'] = df[col].rank(pct=True)
    pair_data[pair] = (df, df1h[['open','high','low','close']])

print(f'Loaded {len(pair_data)} pairs.')


# ── Stop/target simulator ─────────────────────────────────────────────────────
def simulate(long_fn, short_fn, stop_p, target_p):
    max_entry = min(ENTRY_END, 19 - 1)
    rows = []
    for pair, (df, ohlc) in pair_data.items():
        pip = PIP_SIZE[pair]; sp = SPREAD_PIPS[pair]
        d = df[(df.index.hour >= ENTRY_START) & (df.index.hour <= max_entry)].copy()
        if len(d) == 0: continue
        try:
            sig_l = long_fn(d)
            sig_s = short_fn(d)
        except Exception:
            continue
        sig = pd.Series(0, index=d.index)
        sig[sig_l] = 1; sig[sig_s] = -1
        sig_bars = sig[sig != 0]
        if len(sig_bars) == 0: continue
        results = {}
        for ts, direction in sig_bars.items():
            try:
                pos = ohlc.index.get_loc(ts)
            except KeyError:
                continue
            if pos + 1 >= len(ohlc): continue
            entry_time = ohlc.index[pos+1]
            if entry_time.hour > 19: continue
            entry = ohlc.iloc[pos+1]['open']
            stop_price   = entry - direction * stop_p   * pip
            target_price = entry + direction * target_p * pip
            pnl = None
            for k in range(1, MAX_HOLD+1):
                if pos+1+k >= len(ohlc): break
                bar = ohlc.iloc[pos+1+k]
                if ohlc.index[pos+1+k].hour >= 20:
                    pnl = direction*(ohlc.iloc[pos+k]['close']-entry)/pip - sp; break
                if direction == 1:
                    if bar['low']  <= stop_price:   pnl = -stop_p - sp; break
                    if bar['high'] >= target_price: pnl = +target_p - sp; break
                else:
                    if bar['high'] >= stop_price:   pnl = -stop_p - sp; break
                    if bar['low']  <= target_price: pnl = +target_p - sp; break
            if pnl is None:
                eb = min(pos+1+MAX_HOLD, len(ohlc)-1)
                pnl = direction*(ohlc.iloc[eb]['close']-entry)/pip - sp
            results[ts] = pnl
        if results:
            rows.append(pd.Series(results))
    if not rows: return pd.Series(dtype=float)
    return pd.concat(rows).sort_index()


def report(name, t, show_years=False):
    if len(t) < 10: print(f'{name}: insufficient trades ({len(t)})'); return
    nm  = (t.index.max()-t.index.min()).days/30
    sh  = (t.mean()/t.std())*np.sqrt(252*6)
    cut = t.index.max()-pd.DateOffset(months=18)
    r   = t[t.index>=cut]
    sh_r = (r.mean()/r.std())*np.sqrt(252*6) if len(r)>5 and r.std()>0 else 0
    wins = t[t>0]; losses = t[t<=0]
    flag = ' <<<' if ((t>0).mean()>0.48 and t.mean()>0) else ''
    print(f'{name:<58} {len(t):>5,}({len(t)/nm:>4.0f}/mo) WR:{(t>0).mean():>5.1%} EV:{t.mean():>+5.2f} '
          f'W:{wins.mean():>5.1f} L:{losses.mean():>5.1f} Sh:{sh:>+5.2f} | '
          f'18m:{len(r):>4,} WR:{(r>0).mean():>5.1%} EV:{r.mean():>+5.2f} Sh:{sh_r:>+5.2f}{flag}')
    if show_years:
        for yr in sorted(t.index.year.unique()):
            yt = t[t.index.year==yr]
            m = ' *' if yt.mean()>1 else (' -' if yt.mean()<-1 else '')
            print(f'  {yr} {len(yt):>5,} WR:{(yt>0).mean():.1%} EV:{yt.mean():>+6.2f} Tot:{yt.sum():>+8.0f}{m}')


# ── Step 1: raw directional power of the three features alone ─────────────────
print()
print('STEP 1: RAW DIRECTIONAL POWER (4H fixed exit, no stop/target)')
print('Does autocorr_1 / info_accel / accel_skew predict next bar direction?')
print()

for pair, (df, ohlc) in pair_data.items():
    pip = PIP_SIZE[pair]
    d = df[(df.index.hour >= ENTRY_START) & (df.index.hour <= ENTRY_END)].copy()
    close = d['close']
    fwd4h = (close.shift(-4) - close) / pip
    for col, direction in [('autocorr_1_r','high=up'),('info_accel_r','high=up'),('accel_skew_r','high=dn')]:
        if col not in d.columns: continue
    break  # just show for EURUSD

pair, pip = 'EURUSD', 0.0001
df, ohlc = pair_data[pair]
d = df[(df.index.hour >= ENTRY_START) & (df.index.hour <= ENTRY_END)].copy()
close = d['close']
fwd4h = (close.shift(-4) - close) / pip

print(f'EURUSD — 4H forward return by feature quintile (entry 7-16 UTC):')
for col, label in [('autocorr_1_r','autocorr_1 (positive = trending)'),
                   ('info_accel_r', 'info_accel (high = accelerating info)'),
                   ('accel_skew_r', 'accel_skew (low = decel skew = exhaustion)')]:
    if col not in d.columns: continue
    print(f'\n  {label}:')
    for lo, hi in [(0,.2),(.2,.4),(.4,.6),(.6,.8),(.8,1.)]:
        mask = (d[col]>=lo) & (d[col]<hi)
        fwd = fwd4h[mask].dropna()
        print(f'    Q{lo:.0%}-{hi:.0%}: mean={fwd.mean():>+6.2f} pips  up%={(fwd>0).mean():.1%}  n={len(fwd):,}')

# ── Step 2: momentum continuation signal ──────────────────────────────────────
print()
print('='*115)
print('STEP 2: MOMENTUM CONTINUATION — autocorr_1 high = trending, enter in direction')
print('Signal: price is trending (autocorr_1 > threshold) -> enter in slope direction')
print()

STOP_P, TGT_P = 10, 20

for ac_thresh in [0.60, 0.65, 0.70, 0.75]:
    for ia_thresh in [0.55, 0.60, 0.65]:
        # Long: autocorr high (trending) + slope up + info_accel high (flow accelerating)
        long_fn  = lambda d, a=ac_thresh, i=ia_thresh: (d['autocorr_1_r']>a) & (d['slope_close_3h_r']>0.55) & (d['info_accel_r']>i)
        short_fn = lambda d, a=ac_thresh, i=ia_thresh: (d['autocorr_1_r']>a) & (d['slope_close_3h_r']<0.45) & (d['info_accel_r']>i)
        t = simulate(long_fn, short_fn, STOP_P, TGT_P)
        report(f'AC>{ac_thresh} + slope_3h + IA>{ia_thresh}  [cont]', t)

print()
print('='*115)
print('STEP 3: EXHAUSTION REVERSAL — accel_skew low = deceleration = reversal')
print('Signal: price accelerated but now decelerating (accel_skew < threshold) -> fade direction')
print()

for ask_thresh in [0.25, 0.30, 0.35, 0.40]:
    for am_thresh in [0.25, 0.30, 0.35]:
        # accel_skew low + accel_mean low = deceleration -> fade the current slope
        long_fn  = lambda d, s=ask_thresh, m=am_thresh: (d['accel_skew_r']<s) & (d['accel_mean_r']<m) & (d['slope_close_3h_r']<0.40)
        short_fn = lambda d, s=ask_thresh, m=am_thresh: (d['accel_skew_r']<s) & (d['accel_mean_r']<m) & (d['slope_close_3h_r']>0.60)
        t = simulate(long_fn, short_fn, STOP_P, TGT_P)
        report(f'accel_skew<{ask_thresh} + accel_mean<{am_thresh} + slope_fade [exhaust]', t)

print()
print('='*115)
print('STEP 4: INFO FLOW SURGE — info_accel + autocorr in same direction as slope')
print('Hypothesis: info flow surge + trend confirms -> continuation')
print()

for stop_p, tgt_p in [(8,16),(10,20),(10,25),(12,20),(15,25)]:
    long_fn  = lambda d: (d['info_accel_r']>0.70) & (d['autocorr_1_r']>0.65) & (d['slope_close_6h_r']>0.60)
    short_fn = lambda d: (d['info_accel_r']>0.70) & (d['autocorr_1_r']>0.65) & (d['slope_close_6h_r']<0.40)
    t = simulate(long_fn, short_fn, stop_p, tgt_p)
    report(f'IA>70 + AC>65 + slope6h [cont] stop:{stop_p} tgt:{tgt_p}', t)

print()
print('='*115)
print('STEP 5: OPEN SEARCH — scan all feature pairs for directional signal')
print('For each (feature_A high, feature_B condition) -> long/short')
print()

# Systematic: for each pair of ranked features, test as direction signal
# Long: feature A high + slope up; Short: feature A high + slope down
candidates = [
    ('autocorr_1_r', 0.65, 'high'),
    ('info_accel_r', 0.65, 'high'),
    ('accel_skew_r', 0.35, 'low'),
    ('accel_mean_r', 0.35, 'low'),
    ('kyle_lambda_r', 0.65, 'high'),
    ('kyle_lambda_delta_3h_r', 0.65, 'high'),
    ('kyle_lambda_change_r', 0.65, 'high'),
    ('order_imbalance_r', 0.65, 'high'),
    ('order_imbalance_delta_3h_r', 0.65, 'high'),
    ('vpin_4h_r', 0.65, 'high'),
    ('hurst_6h_r', 0.65, 'high'),
    ('vr_5_r', 0.35, 'low'),
    ('entropy_norm_r', 0.35, 'low'),
    ('rv_zscore_24_r', 0.65, 'high'),
    ('curvature_24h_r', 0.65, 'high'),
    ('4h_close_pos_r', 0.65, 'high'),
    ('momentum_shift_r', 0.35, 'low'),
    ('noise_to_signal_r', 0.35, 'low'),
]

slope_cols = ['slope_close_3h_r', 'slope_close_6h_r', 'slope_close_24h_r']

print(f'{"Signal":<55} {"Trades":>6} {"/mo":>5} {"WR":>7} {"EV":>7} {"Sh":>7} | {"18mWR":>7} {"18mEV":>7} {"18mSh":>7}')
print('-'*115)
results_open = []
for feat, thresh, direction in candidates:
    for slope_col in slope_cols:
        for stop_p, tgt_p in [(8,16),(10,20)]:
            try:
                if direction == 'high':
                    cond = lambda d, f=feat, t=thresh: d[f] > t
                else:
                    cond = lambda d, f=feat, t=thresh: d[f] < t

                long_fn  = lambda d, c=cond, s=slope_col: c(d) & (d[s] > 0.55)
                short_fn = lambda d, c=cond, s=slope_col: c(d) & (d[s] < 0.45)
                t = simulate(long_fn, short_fn, stop_p, tgt_p)
                if len(t) < 100: continue
                nm  = (t.index.max()-t.index.min()).days/30
                sh  = (t.mean()/t.std())*np.sqrt(252*6)
                cut = t.index.max()-pd.DateOffset(months=18)
                r   = t[t.index>=cut]
                sh_r = (r.mean()/r.std())*np.sqrt(252*6) if len(r)>10 and r.std()>0 else 0
                label = f'{feat}{">" if direction=="high" else "<"}{thresh}+{slope_col}[s{stop_p}t{tgt_p}]'
                wr = (t>0).mean()
                ev = t.mean()
                flag = ' <<<' if (wr>0.50 and ev>0) else ''
                if wr > 0.48 or (r>0).mean() > 0.50:
                    print(f'{label:<55} {len(t):>6,} {len(t)/nm:>5.0f} {wr:>7.1%} {ev:>+7.2f} {sh:>+7.2f} | {(r>0).mean():>7.1%} {r.mean():>+7.2f} {sh_r:>+7.2f}{flag}')
                results_open.append({'label':label,'wr':wr,'ev':ev,'sh':sh,'wr18':( r>0).mean(),'ev18':r.mean(),'sh18':sh_r,'n':len(t),'npm':len(t)/nm})
            except Exception:
                pass
