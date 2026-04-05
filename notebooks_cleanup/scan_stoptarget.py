import pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')
from pathlib import Path
import itertools

PIP_SIZE = {'AUDJPY':0.01,'AUDNZD':0.0001,'AUDUSD':0.0001,'CADJPY':0.01,'CHFJPY':0.01,
    'EURAUD':0.0001,'EURGBP':0.0001,'EURJPY':0.01,'EURUSD':0.0001,'NZDUSD':0.0001,'USDCAD':0.0001}
SPREAD_PIPS = {'AUDJPY':3.0,'AUDNZD':3.0,'AUDUSD':1.5,'CADJPY':3.0,'CHFJPY':3.0,
    'EURAUD':3.0,'EURGBP':1.5,'EURJPY':2.0,'EURUSD':1.0,'NZDUSD':2.0,'USDCAD':2.0}
MAX_1H_MOVE = {'AUDJPY':250,'AUDNZD':80,'AUDUSD':120,'CADJPY':250,'CHFJPY':250,
    'EURAUD':150,'EURGBP':100,'EURJPY':300,'EURUSD':150,'NZDUSD':100,'USDCAD':150}
F6 = Path('backend/data/features_6')
F8 = Path('backend/data/features_8')
PROC = Path('backend/data/processed')

# Entry window: 7-16 UTC, exit within session (bar+max_hold exit <= 19 UTC)
ENTRY_START = 7
ENTRY_END   = 16
MAX_HOLD    = 8  # bars to search for stop/target hit

# ── Load per-pair feature data ────────────────────────────────────────────────
print('Loading features...')
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

    ret_1h = ((df1h['close'].shift(-1) - df1h['close']) / pip).abs().reindex(df.index)
    holiday = ~(((df.index.month==12)&(df.index.day.isin([24,25,26,31])))|((df.index.month==1)&(df.index.day.isin([1,2]))))
    df = df[holiday & (ret_1h <= MAX_1H_MOVE[pair])].copy()
    df = df.dropna(subset=['realized_skew','residual_12h','rv_zscore_24','kyle_lambda_delta_3h','vr_5'])

    # Attach OHLC for stop/target simulation
    df['open']  = df1h['open'].reindex(df.index)
    df['high']  = df1h['high'].reindex(df.index)
    df['low']   = df1h['low'].reindex(df.index)
    df['close'] = df1h['close'].reindex(df.index)

    # Per-pair ranks
    for col in ['kyle_lambda_delta_3h','residual_12h','rv_zscore_24','realized_skew','vr_5']:
        df[f'{col}_r'] = df[col].rank(pct=True)

    # Store full OHLC series for forward bar lookup
    pair_data[pair] = (df, df1h[['open','high','low','close']])

print(f'Loaded {len(pair_data)} pairs.')


def make_signal(d):
    sig_l = (d['kyle_lambda_delta_3h_r']>0.70)&(d['residual_12h_r']<0.25)&\
            (d['rv_zscore_24_r']>0.85)&(d['realized_skew_r']<0.30)&(d['vr_5_r']<0.30)
    sig_s = (d['kyle_lambda_delta_3h_r']<0.30)&(d['residual_12h_r']>0.75)&\
            (d['rv_zscore_24_r']>0.85)&(d['realized_skew_r']>0.70)&(d['vr_5_r']<0.30)
    sig = pd.Series(0, index=d.index)
    sig[sig_l] = 1; sig[sig_s] = -1
    return sig


def simulate_stop_target(pair, df, ohlc_full, signal, stop_pips, target_pips, spread, pip, max_hold=8):
    """
    For each signal bar, walk forward bar-by-bar through OHLC.
    Assume worst-case order: for longs, low hit before high within a bar.
    Entry at next bar's open + spread.
    Returns series of PnL indexed like signal.
    """
    results = {}
    sig_bars = signal[signal != 0]

    for entry_ts, direction in sig_bars.items():
        # Find the bar index in ohlc_full
        try:
            pos = ohlc_full.index.get_loc(entry_ts)
        except KeyError:
            continue

        # Entry at open of next bar (post-signal bar)
        if pos + 1 >= len(ohlc_full):
            continue
        next_bar = ohlc_full.iloc[pos + 1]
        entry_price = next_bar['open']
        entry_time  = ohlc_full.index[pos + 1]

        # Don't enter if exit would go past 19 UTC
        if entry_time.hour > 19:
            continue

        stop_price   = entry_price - direction * stop_pips   * pip
        target_price = entry_price + direction * target_pips * pip

        pnl = None
        for k in range(1, max_hold + 1):
            if pos + 1 + k >= len(ohlc_full):
                break
            bar = ohlc_full.iloc[pos + 1 + k]
            bar_time = ohlc_full.index[pos + 1 + k]

            # Hard session cap: exit at close of bar 19 UTC
            if bar_time.hour >= 20:
                exit_price = ohlc_full.iloc[pos + k]['close']
                pnl = direction * (exit_price - entry_price) / pip - spread
                break

            if direction == 1:
                # Long: assume low hits before high (conservative)
                if bar['low'] <= stop_price:
                    pnl = -stop_pips - spread
                    break
                if bar['high'] >= target_price:
                    pnl = target_pips - spread
                    break
            else:
                # Short: assume high hits before low (conservative)
                if bar['high'] >= stop_price:
                    pnl = -stop_pips - spread
                    break
                if bar['low'] <= target_price:
                    pnl = target_pips - spread
                    break

        if pnl is None:
            # Time exit: close of last bar checked
            exit_bar = min(pos + 1 + max_hold, len(ohlc_full) - 1)
            exit_price = ohlc_full.iloc[exit_bar]['close']
            pnl = direction * (exit_price - entry_price) / pip - spread

        results[entry_ts] = pnl

    return pd.Series(results)


def run_all_pairs(stop_pips, target_pips, entry_start=7, entry_end=16):
    rows = []
    max_entry = min(entry_end, 19 - 1)  # at least 1 bar room
    for pair, (df, ohlc_full) in pair_data.items():
        pip = PIP_SIZE[pair]
        sp  = SPREAD_PIPS[pair]
        d   = df[(df.index.hour >= entry_start) & (df.index.hour <= max_entry)].copy()
        if len(d) == 0: continue
        sig = make_signal(d)
        sig = sig[sig != 0]
        pnl = simulate_stop_target(pair, d, ohlc_full, sig, stop_pips, target_pips, sp, pip, MAX_HOLD)
        rows.append(pnl)
    if not rows: return pd.Series(dtype=float)
    return pd.concat(rows).sort_index()


# ── Grid search over stop / target combos ────────────────────────────────────
stops   = [5, 8, 10, 12, 15, 20]
targets = [10, 15, 20, 25, 30, 40]

print()
print('STOP/TARGET GRID SEARCH  |  entry 7-16 UTC, exit <= 19 UTC, max 8 bars')
print()
print(f'{"Stop":>6} {"Target":>8} {"R:R":>5} {"Trades":>7} {"/mo":>5} {"WR":>7} {"EV":>8} {"Sh":>7} | {"18mTrd":>7} {"18mEV":>8} {"18mSh":>7}')
print('-'*95)

best = []
for stop_p, tgt_p in itertools.product(stops, targets):
    if tgt_p <= stop_p: continue
    t = run_all_pairs(stop_p, tgt_p)
    if len(t) < 20: continue
    nm   = (t.index.max()-t.index.min()).days/30
    sh   = (t.mean()/t.std())*np.sqrt(252*6)  # ~6 trades/day across pairs
    cut  = t.index.max()-pd.DateOffset(months=18)
    r    = t[t.index>=cut]
    sh_r = (r.mean()/r.std())*np.sqrt(252*6) if len(r)>5 and r.std()>0 else 0
    rr   = tgt_p / stop_p
    flag = ' <<<' if (r.mean()>1.0 and sh_r>0.5) else ''
    print(f'{stop_p:>6} {tgt_p:>8} {rr:>5.1f}x {len(t):>7,} {len(t)/nm:>5.0f} {(t>0).mean():>7.1%} {t.mean():>+8.2f} {sh:>+7.2f} | {len(r):>7,} {r.mean():>+8.2f} {sh_r:>+7.2f}{flag}')
    best.append((r.mean(), stop_p, tgt_p, t, r))

print()
print('='*95)
print('YEAR-BY-YEAR — top 3 by 18m EV')
best.sort(reverse=True)
for rev, stop_p, tgt_p, t_all, r in best[:3]:
    nm = (t_all.index.max()-t_all.index.min()).days/30
    sh = (t_all.mean()/t_all.std())*np.sqrt(252*6)
    sh_r = (r.mean()/r.std())*np.sqrt(252*6) if r.std()>0 else 0
    print()
    print(f'Stop {stop_p} / Target {tgt_p} ({tgt_p/stop_p:.1f}x R:R) | {len(t_all):,} trades ({len(t_all)/nm:.0f}/mo) | WR {(t_all>0).mean():.1%} | EV {t_all.mean():+.2f} | Sh {sh:+.2f}')
    print(f'  {"Year":>6} {"Trades":>7} {"WR":>7} {"EV":>8} {"Total":>9}')
    for yr in sorted(t_all.index.year.unique()):
        yt = t_all[t_all.index.year==yr]
        marker = ' *' if yt.mean()>1 else (' -' if yt.mean()<-2 else '')
        print(f'  {yr:>6} {len(yt):>7,} {(yt>0).mean():>7.1%} {yt.mean():>+8.2f} {yt.sum():>+9.0f}{marker}')
