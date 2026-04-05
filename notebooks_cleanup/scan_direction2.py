"""
Test combining directional filters on top of the existing signal.
Goal: push win rate from 52% toward 55-60% in the signal population.

Significant directional features found:
  HIGH=bullish: info_accel, autocorr_1, entropy_volume_divergence,
                curvature_24h, jump_ratio_delta_6h, order_imbalance_delta_12h
  HIGH=bearish: accel_mean, noise_to_signal, realized_skew (already in signal),
                momentum_shift, sum_abs_autocorr, accel_skew
"""
import pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')
from pathlib import Path

PIP_SIZE = {'AUDJPY':0.01,'AUDNZD':0.0001,'AUDUSD':0.0001,'CADJPY':0.01,'CHFJPY':0.01,
    'EURAUD':0.0001,'EURGBP':0.0001,'EURJPY':0.01,'EURUSD':0.0001,'NZDUSD':0.0001,'USDCAD':0.0001}
SPREAD_PIPS = {'AUDJPY':3.0,'AUDNZD':3.0,'AUDUSD':1.5,'CADJPY':3.0,'CHFJPY':3.0,
    'EURAUD':3.0,'EURGBP':1.5,'EURJPY':2.0,'EURUSD':1.0,'NZDUSD':2.0,'USDCAD':2.0}
MAX_1H_MOVE = {'AUDJPY':250,'AUDNZD':80,'AUDUSD':120,'CADJPY':250,'CHFJPY':250,
    'EURAUD':150,'EURGBP':100,'EURJPY':300,'EURUSD':150,'NZDUSD':100,'USDCAD':150}
F6 = Path('backend/data/features_6'); F8 = Path('backend/data/features_8')
PROC = Path('backend/data/processed')
ENTRY_START, ENTRY_END, HOLD_H = 7, 16, 4
STOP_PIPS, TARGET_PIPS, MAX_HOLD = 5, 40, 8

# ── Load per-pair data ────────────────────────────────────────────────────────
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
    df = df.dropna(subset=['realized_skew','residual_12h','rv_zscore_24','kyle_lambda_delta_3h','vr_5','close'])
    for col in ['kyle_lambda_delta_3h','residual_12h','rv_zscore_24','realized_skew','vr_5',
                'info_accel','autocorr_1','entropy_volume_divergence','accel_mean',
                'noise_to_signal','momentum_shift','sum_abs_autocorr','accel_skew',
                'curvature_24h','jump_ratio_delta_6h','order_imbalance_delta_12h',
                'accel_std','jump_ratio','slope_close_6h','curvature_12h']:
        if col in df.columns:
            df[f'{col}_r'] = df[col].rank(pct=True)
    pair_data[pair] = (df, df1h[['open','high','low','close']])
print(f'Loaded {len(pair_data)} pairs.')


def simulate(stop_p, target_p, extra_long_filter=None, extra_short_filter=None):
    """Run stop/target simulation with optional extra directional filters."""
    rows = []
    max_entry = min(ENTRY_END, 19 - 1)

    for pair, (df, ohlc) in pair_data.items():
        pip = PIP_SIZE[pair]; sp = SPREAD_PIPS[pair]
        d = df[(df.index.hour >= ENTRY_START) & (df.index.hour <= max_entry)].copy()
        if len(d) == 0: continue

        # Base signal
        sig_l = (d['kyle_lambda_delta_3h_r']>0.70)&(d['residual_12h_r']<0.25)&\
                (d['rv_zscore_24_r']>0.85)&(d['realized_skew_r']<0.30)&(d['vr_5_r']<0.30)
        sig_s = (d['kyle_lambda_delta_3h_r']<0.30)&(d['residual_12h_r']>0.75)&\
                (d['rv_zscore_24_r']>0.85)&(d['realized_skew_r']>0.70)&(d['vr_5_r']<0.30)

        # Apply extra directional filters
        if extra_long_filter is not None:
            sig_l = sig_l & extra_long_filter(d)
        if extra_short_filter is not None:
            sig_s = sig_s & extra_short_filter(d)

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
            next_bar = ohlc.iloc[pos+1]
            if ohlc.index[pos+1].hour > 19: continue
            entry = next_bar['open']
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
                    if bar['high'] >= target_price: pnl = target_p - sp; break
                else:
                    if bar['high'] >= stop_price:   pnl = -stop_p - sp; break
                    if bar['low']  <= target_price: pnl = target_p - sp; break
            if pnl is None:
                exit_bar = min(pos+1+MAX_HOLD, len(ohlc)-1)
                pnl = direction*(ohlc.iloc[exit_bar]['close']-entry)/pip - sp
            results[ts] = pnl
        rows.append(pd.Series(results))

    if not rows: return pd.Series(dtype=float)
    return pd.concat(rows).sort_index()


def report(name, t, show_years=False):
    if len(t) < 10: print(f'{name}: too few trades'); return
    nm = (t.index.max()-t.index.min()).days/30
    sh = (t.mean()/t.std())*np.sqrt(252*6)
    cut = t.index.max()-pd.DateOffset(months=18)
    r = t[t.index>=cut]
    sh_r = (r.mean()/r.std())*np.sqrt(252*6) if len(r)>5 and r.std()>0 else 0
    wins = t[t>0]; losses = t[t<=0]
    flag = ' <<<' if (t.mean()>0 and (t>0).mean()>0.48) else ''
    print(f'{name:<55} {len(t):>5,}({len(t)/nm:>3.0f}/mo) WR:{(t>0).mean():>5.1%} EV:{t.mean():>+5.2f} W:{wins.mean():>5.1f} L:{losses.mean():>5.1f} Sh:{sh:>+5.2f} | 18m WR:{(r>0).mean():>5.1%} EV:{r.mean():>+5.2f} Sh:{sh_r:>+5.2f}{flag}')
    if show_years:
        for yr in sorted(t.index.year.unique()):
            yt = t[t.index.year==yr]
            marker = ' *' if yt.mean()>1 else (' -' if yt.mean()<-1 else '')
            print(f'  {yr} {len(yt):>5,} WR:{(yt>0).mean():.1%} EV:{yt.mean():>+6.2f} Tot:{yt.sum():>+7.0f}{marker}')


# ── Baseline ──────────────────────────────────────────────────────────────────
print('BASELINE (stop 5, target 40, no directional filter)')
base = simulate(STOP_PIPS, TARGET_PIPS)
report('baseline', base)
print()

# ── Single directional filter tests ──────────────────────────────────────────
print('='*110)
print('SINGLE DIRECTIONAL FILTERS  (each added on top of base signal)')
print()

filters = {
    # Bullish for longs: HIGH value = more likely to revert up
    'info_accel_r > 0.60':          lambda d: d['info_accel_r'] > 0.60,
    'info_accel_r > 0.65':          lambda d: d['info_accel_r'] > 0.65,
    'info_accel_r > 0.70':          lambda d: d['info_accel_r'] > 0.70,
    'autocorr_1_r > 0.60':          lambda d: d['autocorr_1_r'] > 0.60,
    'autocorr_1_r > 0.65':          lambda d: d['autocorr_1_r'] > 0.65,
    'autocorr_1_r > 0.70':          lambda d: d['autocorr_1_r'] > 0.70,
    'entropy_vol_div_r > 0.60':     lambda d: d['entropy_volume_divergence_r'] > 0.60,
    'entropy_vol_div_r > 0.65':     lambda d: d['entropy_volume_divergence_r'] > 0.65,
    'jump_ratio_delta6h_r > 0.60':  lambda d: d['jump_ratio_delta_6h_r'] > 0.60,
    'order_imb_delta12h_r > 0.60':  lambda d: d['order_imbalance_delta_12h_r'] > 0.60,
    'curvature_24h_r > 0.60':       lambda d: d['curvature_24h_r'] > 0.60,
    # Bearish for longs means we SKIP long when high (anti-confirmation)
    'accel_mean_r < 0.40':          lambda d: d['accel_mean_r'] < 0.40,
    'accel_mean_r < 0.35':          lambda d: d['accel_mean_r'] < 0.35,
    'noise_to_signal_r < 0.40':     lambda d: d['noise_to_signal_r'] < 0.40,
    'momentum_shift_r < 0.40':      lambda d: d['momentum_shift_r'] < 0.40,
    'accel_skew_r < 0.40':          lambda d: d['accel_skew_r'] < 0.40,
    'sum_abs_autocorr_r < 0.40':    lambda d: d['sum_abs_autocorr_r'] < 0.40,
}

for name, fn in filters.items():
    try:
        t = simulate(STOP_PIPS, TARGET_PIPS,
                     extra_long_filter=fn,
                     extra_short_filter=fn)  # same logic applies symmetrically for shorts
        report(name, t)
    except Exception as e:
        print(f'{name}: ERROR {e}')

print()
print('='*110)
print('COMBINATIONS of top directional filters')
print()

combos = {
    'autocorr>65 + info_accel>60':
        lambda d: (d['autocorr_1_r']>0.65) & (d['info_accel_r']>0.60),
    'autocorr>65 + accel_mean<40':
        lambda d: (d['autocorr_1_r']>0.65) & (d['accel_mean_r']<0.40),
    'autocorr>65 + accel_skew<40':
        lambda d: (d['autocorr_1_r']>0.65) & (d['accel_skew_r']<0.40),
    'info_accel>65 + accel_mean<40':
        lambda d: (d['info_accel_r']>0.65) & (d['accel_mean_r']<0.40),
    'info_accel>65 + accel_skew<40':
        lambda d: (d['info_accel_r']>0.65) & (d['accel_skew_r']<0.40),
    'autocorr>65 + info_accel>60 + accel_mean<40':
        lambda d: (d['autocorr_1_r']>0.65) & (d['info_accel_r']>0.60) & (d['accel_mean_r']<0.40),
    'autocorr>65 + info_accel>60 + accel_skew<40':
        lambda d: (d['autocorr_1_r']>0.65) & (d['info_accel_r']>0.60) & (d['accel_skew_r']<0.40),
    'autocorr>65 + info_accel>60 + noise<40':
        lambda d: (d['autocorr_1_r']>0.65) & (d['info_accel_r']>0.60) & (d['noise_to_signal_r']<0.40),
    'autocorr>70 + info_accel>65 + accel_skew<35':
        lambda d: (d['autocorr_1_r']>0.70) & (d['info_accel_r']>0.65) & (d['accel_skew_r']<0.35),
    'autocorr>65 + entropy_vol_div>60 + accel_mean<40':
        lambda d: (d['autocorr_1_r']>0.65) & (d['entropy_volume_divergence_r']>0.60) & (d['accel_mean_r']<0.40),
    'info_accel>60 + noise<40 + accel_skew<40':
        lambda d: (d['info_accel_r']>0.60) & (d['noise_to_signal_r']<0.40) & (d['accel_skew_r']<0.40),
    'autocorr>65 + momentum_shift<40':
        lambda d: (d['autocorr_1_r']>0.65) & (d['momentum_shift_r']<0.40),
    'autocorr>65 + sum_autocorr<40':
        lambda d: (d['autocorr_1_r']>0.65) & (d['sum_abs_autocorr_r']<0.40),
}

for name, fn in combos.items():
    try:
        t = simulate(STOP_PIPS, TARGET_PIPS, extra_long_filter=fn, extra_short_filter=fn)
        report(name, t)
    except Exception as e:
        print(f'{name}: ERROR {e}')

print()
print('='*110)
print('BEST COMBO YEAR-BY-YEAR + DIFFERENT STOP/TARGET')
print()

# Retest best combos with different stop/target to find ideal R:R at higher win rate
best_fn = lambda d: (d['autocorr_1_r']>0.65) & (d['info_accel_r']>0.60) & (d['accel_skew_r']<0.40)
print('autocorr>65 + info_accel>60 + accel_skew<40 — stop/target grid:')
for stop_p, tgt_p in [(5,15),(5,20),(5,25),(5,30),(8,15),(8,20),(8,25),(10,20),(10,25),(10,30)]:
    t = simulate(stop_p, tgt_p, extra_long_filter=best_fn, extra_short_filter=best_fn)
    if len(t) < 10: continue
    nm = (t.index.max()-t.index.min()).days/30
    sh = (t.mean()/t.std())*np.sqrt(252*6)
    cut = t.index.max()-pd.DateOffset(months=18)
    r = t[t.index>=cut]
    sh_r = (r.mean()/r.std())*np.sqrt(252*6) if len(r)>5 and r.std()>0 else 0
    print(f'  stop:{stop_p:>3} tgt:{tgt_p:>3} ({tgt_p/stop_p:.1f}x)  {len(t):>4,}({len(t)/nm:.0f}/mo)  WR:{(t>0).mean():.1%}  EV:{t.mean():>+5.2f}  Sh:{sh:>+5.2f} | 18m WR:{(r>0).mean():.1%}  EV:{r.mean():>+5.2f}  Sh:{sh_r:>+5.2f}')
