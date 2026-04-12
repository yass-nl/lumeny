"""
Signal Anatomy
==============
For each signal (MFE>=50, hours 7-20 UTC, direction system, 72h cooldown),
compute a comprehensive picture of what happens inside the 72h window.

Metrics computed per signal:
  Price path:
    - MFE  : max favorable excursion (pips) in 72h
    - MAE  : max adverse excursion (pips) in 72h
    - Final: final 72h return (pips)
    - MFE/MAE ratio
    - Time-to-MFE, Time-to-MAE (hour index 0-71)
    - MFE before MAE? (favorable move comes first)
    - Monotonicity: fraction of hours price moved in signal direction

  Moving averages at entry:
    - MA50, MA200 level at bar 0
    - Price vs MA50, vs MA200 (above/below, distance in pips)
    - MA50 slope (last 10h)
    - MA200 slope (last 10h)
    - MA50 vs MA200 (golden/death cross state)

  Moving averages at exit (bar 72):
    - MA50, MA200 level
    - Did MA50 cross MA200 during the window?

  Session / timing:
    - Hour of entry
    - Day of week of entry
    - Which session: London open (7-9), London (9-12), London/NY overlap (12-14), NY (14-17), NY close (17-20)

  Volume:
    - Entry bar volume vs 24h avg volume
    - Avg volume during window vs avg volume before window

  Momentum at entry:
    - Return last 1h, 4h, 12h, 24h before signal
    - RSI-like: fraction of up bars in last 14h
    - Price range position in last 24h (where in the range is price?)

  Per-pair aggregated output:
    - Avg MFE, MAE, Final by pair
    - Hour-by-hour avg price path (0-71h) — shape of move
    - % of signals where MAE > 20p before MFE (tells us if wide SL needed)
    - Distribution of time-to-MFE (when to take profit)
    - Best SL level (MAE percentiles: p50, p75, p90)
    - Best TP level (MFE percentiles: p50, p75, p90)
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path

SCRIPT_DIR    = Path(__file__).parent
FEATURES_DIR  = SCRIPT_DIR / '../backend/data/features_9'
PROCESSED_DIR = SCRIPT_DIR / '../backend/data/processed'
MFE_MODEL_PATH = SCRIPT_DIR / '../backend/models_9/mfe_q50/model_1H_Q50.joblib'

START_DATE    = '2024-10-11'
MFE_THRESH    = 50.0
HOURS_ALLOWED = set(range(7, 21))
COOLDOWN_H    = 72
FWD_BARS      = 72
MA_SHORT      = 50
MA_LONG       = 200

JPY_PAIRS = {'USDJPY', 'EURJPY', 'GBPJPY', 'AUDJPY', 'CADJPY', 'CHFJPY'}
PAIRS_ALL = [
    'EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD', 'USDCAD', 'NZDUSD',
    'EURJPY', 'GBPJPY', 'EURGBP', 'EURAUD', 'AUDJPY', 'CADJPY', 'CHFJPY', 'AUDNZD',
]

# ── Load MFE model ────────────────────────────────────────────────────────────
print('Loading MFE model...')
bundle       = joblib.load(MFE_MODEL_PATH)
mfe_model    = bundle['model']
feature_cols = bundle['feature_cols']

# ── Load features_9 ───────────────────────────────────────────────────────────
print('Loading features_9...')
dfs = [pd.read_parquet(f) for f in sorted(FEATURES_DIR.glob('*_features.parquet'))]
df  = pd.concat(dfs).sort_index()
df  = df[df.index >= START_DATE].copy()

# ── Run MFE model + hour filter ───────────────────────────────────────────────
print('Running MFE model...')
X = df[feature_cols].ffill().fillna(0)
df['q50_mfe'] = mfe_model.predict(X)
df['hour']    = pd.to_datetime(df.index).hour
df = df[(df['q50_mfe'] >= MFE_THRESH) & df['hour'].isin(HOURS_ALLOWED)].copy()

# ── Direction rules (same as 01_direction_system_all_bars.py) ─────────────────
def apply_direction_rules(df):
    dirs = pd.Series(np.nan, index=df.index)
    pair = df['pair']

    dirs = dirs.where(pair != 'USDJPY', -1.0)

    m = pair == 'AUDUSD'
    lc = m & (df.get('beta_gbpusd_1w', pd.Series(np.nan, index=df.index)).gt(0.775) |
               df.get('atr_24',         pd.Series(np.nan, index=df.index)).lt(40.8))
    dirs = dirs.where(~lc, 1.0).where(~(m & ~lc), np.nan)

    m = pair == 'GBPUSD'
    lc = m & df.get('csi_usd_24h', pd.Series(np.nan, index=df.index)).lt(0.004)
    dirs = dirs.where(~lc, 1.0).where(~(m & ~lc), np.nan)

    m = pair == 'EURUSD'
    lc = m & df.get('corr_audusd_24h', pd.Series(np.nan, index=df.index)).lt(0.22)
    dirs = dirs.where(~lc, 1.0).where(~(m & ~lc), np.nan)

    m = pair == 'NZDUSD'
    lc = m & df.get('dist_5d_high', pd.Series(np.nan, index=df.index)).gt(0.35)
    dirs = dirs.where(~lc, 1.0).where(~(m & ~lc), np.nan)

    m = pair == 'USDCHF'
    lc = m & df.get('corr_eurusd_1w', pd.Series(np.nan, index=df.index)).gt(-0.60)
    dirs = dirs.where(~lc, 1.0).where(~(m & ~lc), np.nan)

    m = pair == 'CHFJPY'
    cv = df.get('corr_usdjpy_1w', pd.Series(np.nan, index=df.index))
    lc = m & cv.gt(0.40); sc = m & cv.lt(0.26)
    dirs = dirs.where(~lc, 1.0).where(~sc, -1.0).where(~(m & ~lc & ~sc), np.nan)

    m = pair == 'CADJPY'
    vt = df.get('vol_trend', pd.Series(np.nan, index=df.index))
    lc = m & vt.lt(1.15); sc = m & vt.ge(1.15)
    dirs = dirs.where(~lc, 1.0).where(~sc, -1.0)

    m = pair == 'AUDJPY'
    lc = m & df.get('beta_usdjpy_1w', pd.Series(np.nan, index=df.index)).gt(0.74)
    dirs = dirs.where(~lc, 1.0).where(~(m & ~lc), np.nan)

    m = pair == 'EURJPY'
    lc = m & df.get('beta_eurusd_1w', pd.Series(np.nan, index=df.index)).gt(0.38)
    dirs = dirs.where(~lc, 1.0).where(~(m & ~lc), np.nan)

    m = pair == 'GBPJPY'
    lc = m & df.get('beta_eurusd_1w', pd.Series(np.nan, index=df.index)).gt(0.50)
    dirs = dirs.where(~lc, 1.0).where(~(m & ~lc), np.nan)

    m = pair == 'EURAUD'
    lc = m & df.get('corr_audusd_24h', pd.Series(np.nan, index=df.index)).lt(0.22)
    dirs = dirs.where(~lc, 1.0).where(~(m & ~lc), np.nan)

    m = pair == 'AUDNZD'
    lc = m & df.get('corr_regime_audusd', pd.Series(np.nan, index=df.index)).gt(0.0)
    dirs = dirs.where(~lc, 1.0).where(~(m & ~lc), np.nan)

    m = pair == 'EURGBP'
    sc = m & df.get('csi_usd_24h', pd.Series(np.nan, index=df.index)).gt(0.004)
    dirs = dirs.where(~sc, -1.0).where(~(m & ~sc), np.nan)

    dirs = dirs.where(pair != 'USDCAD', np.nan)
    return dirs

df['direction'] = apply_direction_rules(df)
df = df[df['direction'].notna()].copy()

# ── Apply 72h cooldown per pair ───────────────────────────────────────────────
print('Applying cooldown...')
candidates = df.sort_index()
cooldown_until = {}
kept = []
for ts, row in candidates.iterrows():
    p = row['pair']
    if p in cooldown_until and ts < cooldown_until[p]:
        continue
    cooldown_until[p] = ts + pd.Timedelta(hours=COOLDOWN_H)
    kept.append(ts)
df_sig = candidates.loc[kept].copy()
print(f'  Signals: {len(df_sig):,}')

# ── Load 1H OHLCV for all pairs ───────────────────────────────────────────────
print('Loading 1H price data...')
price_data = {}
for pair in PAIRS_ALL:
    fpath = PROCESSED_DIR / f'{pair}_1H.parquet'
    if fpath.exists():
        price_data[pair] = pd.read_parquet(fpath).sort_index()

# ── Compute anatomy per signal ────────────────────────────────────────────────
print('Computing signal anatomy...')
records = []

for ts, sig in df_sig.iterrows():
    pair      = sig['pair']
    direction = int(sig['direction'])
    pip       = 0.01 if pair in JPY_PAIRS else 0.0001

    if pair not in price_data:
        continue
    bars = price_data[pair]

    # Get bars at and after signal
    future = bars[bars.index >= ts].head(FWD_BARS + 1)
    if len(future) < FWD_BARS:
        continue

    entry_bar  = future.iloc[0]
    entry_price = entry_bar['close']
    window      = future.iloc[1:FWD_BARS + 1]   # bars 1..72 after entry

    closes = window['close'].values
    highs  = window['high'].values
    lows   = window['low'].values

    # ── Price path in pips relative to entry ─────────────────────────────────
    path_pips = direction * (closes - entry_price) / pip   # positive = favorable

    # MFE / MAE
    if direction == 1:
        favorable = (highs  - entry_price) / pip
        adverse   = (entry_price - lows) / pip
    else:
        favorable = (entry_price - lows) / pip
        adverse   = (highs - entry_price) / pip

    mfe        = favorable.max()
    mae        = adverse.max()
    final_pips = path_pips[-1]
    t_mfe      = int(favorable.argmax())
    t_mae      = int(adverse.argmax())
    mfe_first  = t_mfe < t_mae   # favorable move came before adverse

    # Monotonicity: fraction of hours moving in signal direction
    hourly_moves = np.diff(np.concatenate([[entry_price], closes]))
    mono = (direction * hourly_moves > 0).mean()

    # Front-load: what fraction of final move happened in first 24h
    if abs(final_pips) > 1:
        front_load = path_pips[23] / final_pips if len(path_pips) > 23 else np.nan
    else:
        front_load = np.nan

    # ── Moving averages at entry ──────────────────────────────────────────────
    history = bars[bars.index < ts].tail(MA_LONG + 20)
    if len(history) < MA_LONG:
        ma50_entry = ma200_entry = np.nan
        ma50_slope = ma200_slope = np.nan
        price_vs_ma50 = price_vs_ma200 = np.nan
        golden_cross_entry = np.nan
    else:
        closes_hist = history['close'].values
        ma50_entry  = closes_hist[-MA_SHORT:].mean()
        ma200_entry = closes_hist[-MA_LONG:].mean()
        ma50_slope  = (closes_hist[-1] - closes_hist[-11]) / pip / 10   # pips/bar over last 10h
        ma200_slope = (closes_hist[-1] - closes_hist[-11]) / pip / 10
        price_vs_ma50  = (entry_price - ma50_entry)  / pip   # positive = price above MA50
        price_vs_ma200 = (entry_price - ma200_entry) / pip
        golden_cross_entry = 1.0 if ma50_entry > ma200_entry else -1.0  # +1 = bullish (MA50>MA200)

    # ── Moving averages at exit ───────────────────────────────────────────────
    exit_price = closes[-1]
    history_exit = bars[bars.index <= window.index[-1]].tail(MA_LONG + 20)
    if len(history_exit) < MA_LONG:
        ma50_exit = ma200_exit = np.nan
        golden_cross_exit = np.nan
    else:
        ce = history_exit['close'].values
        ma50_exit  = ce[-MA_SHORT:].mean()
        ma200_exit = ce[-MA_LONG:].mean()
        golden_cross_exit = 1.0 if ma50_exit > ma200_exit else -1.0

    ma_cross_during = np.nan
    if not (np.isnan(golden_cross_entry) or np.isnan(golden_cross_exit)):
        ma_cross_during = 1.0 if golden_cross_entry != golden_cross_exit else 0.0

    # ── Momentum at entry ─────────────────────────────────────────────────────
    hist_closes = history['close'].values if len(history) >= 24 else np.array([])
    ret_1h  = (entry_price - bars['close'].iloc[-2]) / pip if len(bars) >= 2 else np.nan

    def ret_nh(n):
        h = bars[bars.index < ts].tail(n + 1)
        if len(h) < n + 1:
            return np.nan
        return (h['close'].iloc[-1] - h['close'].iloc[0]) / pip

    ret_4h  = ret_nh(4)
    ret_12h = ret_nh(12)
    ret_24h = ret_nh(24)

    # RSI-like: fraction of up bars in last 14h
    h14 = bars[bars.index < ts].tail(15)
    if len(h14) >= 14:
        diffs   = np.diff(h14['close'].values)
        rsi_raw = (diffs > 0).mean()
    else:
        rsi_raw = np.nan

    # Range position in last 24h
    h24 = bars[bars.index < ts].tail(24)
    if len(h24) >= 24:
        hi24 = h24['high'].max()
        lo24 = h24['low'].min()
        range24 = hi24 - lo24
        range_pos_24 = (entry_price - lo24) / range24 if range24 > 0 else 0.5
    else:
        range_pos_24 = np.nan

    # ── Volume at entry ───────────────────────────────────────────────────────
    entry_vol = entry_bar.get('volume', np.nan)
    h24v = bars[bars.index < ts].tail(24)
    avg_vol_24h = h24v['volume'].mean() if len(h24v) > 0 else np.nan
    vol_ratio   = entry_vol / avg_vol_24h if avg_vol_24h and avg_vol_24h > 0 else np.nan

    # ── Session ───────────────────────────────────────────────────────────────
    h = sig['hour']
    if   7  <= h <  9: session = 'London_open'
    elif 9  <= h < 12: session = 'London'
    elif 12 <= h < 14: session = 'LN_NY_overlap'
    elif 14 <= h < 17: session = 'NY'
    else:              session = 'NY_close'

    dow = pd.Timestamp(ts).day_of_week   # 0=Mon, 4=Fri

    # ── SL/TP levels by percentile (will aggregate later) ────────────────────
    records.append({
        'ts':            ts,
        'pair':          pair,
        'direction':     direction,
        'mfe_score':     sig['q50_mfe'],
        'hour':          h,
        'session':       session,
        'dow':           dow,
        # outcome
        'mfe':           mfe,
        'mae':           mae,
        'final_pips':    final_pips,
        'correct':       1 if final_pips > 0 else 0,
        'mfe_mae_ratio': mfe / mae if mae > 0.1 else np.nan,
        't_mfe':         t_mfe,
        't_mae':         t_mae,
        'mfe_first':     int(mfe_first),
        'monotonicity':  mono,
        'front_load':    front_load,
        # MAs at entry
        'ma50_entry':    ma50_entry,
        'ma200_entry':   ma200_entry,
        'price_vs_ma50': price_vs_ma50,
        'price_vs_ma200':price_vs_ma200,
        'ma50_slope':    ma50_slope,
        'golden_cross':  golden_cross_entry,
        # MAs at exit
        'ma50_exit':     ma50_exit,
        'ma200_exit':    ma200_exit,
        'ma_cross_during': ma_cross_during,
        # momentum
        'ret_1h':        ret_1h,
        'ret_4h':        ret_4h,
        'ret_12h':       ret_12h,
        'ret_24h':       ret_24h,
        'rsi_raw':       rsi_raw,
        'range_pos_24':  range_pos_24,
        # volume
        'vol_ratio':     vol_ratio,
        # path (store for per-pair avg shape)
        '_path':         path_pips,
    })

ana = pd.DataFrame([{k: v for k, v in r.items() if k != '_path'} for r in records])
paths = [r['_path'] for r in records]

print(f'  Computed {len(ana):,} signal anatomies')

# ── Print results ─────────────────────────────────────────────────────────────
SEP = '=' * 72

def pct(x): return f'{x:.1%}'
def pip_fmt(x): return f'{x:+.1f}p' if not np.isnan(x) else 'n/a'
def fmt(x, dec=2): return f'{x:.{dec}f}' if not np.isnan(x) else 'n/a'

print(f'\n{SEP}')
print(f'  SIGNAL ANATOMY  —  {len(ana):,} signals  |  MFE>=50  |  hours 7-20 UTC  |  72h window')
print(f'{SEP}')

# ── Overall summary ───────────────────────────────────────────────────────────
print(f'\n  OVERALL')
print(f'  {"Accuracy":<30}: {pct(ana["correct"].mean())}')
print(f'  {"Avg final":<30}: {pip_fmt(ana["final_pips"].mean())}  (med {pip_fmt(ana["final_pips"].median())})')
print(f'  {"Avg MFE":<30}: {pip_fmt(ana["mfe"].mean())}  (med {pip_fmt(ana["mfe"].median())})')
print(f'  {"Avg MAE":<30}: {pip_fmt(ana["mae"].mean())}  (med {pip_fmt(ana["mae"].median())})')
print(f'  {"Avg MFE/MAE ratio":<30}: {fmt(ana["mfe_mae_ratio"].mean())}')
print(f'  {"MFE before MAE %":<30}: {pct(ana["mfe_first"].mean())}  (favorable move comes first)')
print(f'  {"Avg time to MFE (h)":<30}: {fmt(ana["t_mfe"].mean(), 1)}  (med {fmt(ana["t_mfe"].median(), 1)})')
print(f'  {"Avg time to MAE (h)":<30}: {fmt(ana["t_mae"].mean(), 1)}  (med {fmt(ana["t_mae"].median(), 1)})')
print(f'  {"Avg monotonicity":<30}: {pct(ana["monotonicity"].mean())}  (% of bars moving in signal dir)')
print(f'  {"Avg front-load (24h/72h)":<30}: {pct(ana["front_load"].dropna().mean())}  (fraction of final move in first 24h)')

print(f'\n  SL SIZING  (MAE percentiles — how far it goes against before recovering):')
for p in [25, 50, 75, 90, 95]:
    print(f'    p{p:<3}: {ana["mae"].quantile(p/100):>6.1f}p')

print(f'\n  TP SIZING  (MFE percentiles — how far it goes in our favor):')
for p in [25, 50, 75, 90, 95]:
    print(f'    p{p:<3}: {ana["mfe"].quantile(p/100):>6.1f}p')

print(f'\n  TIME TO MFE distribution  (when does the best exit occur):')
bins = [(0,12,'first 12h'), (12,24,'12-24h'), (24,48,'24-48h'), (48,72,'48-72h')]
for lo, hi, label in bins:
    n = ((ana['t_mfe'] >= lo) & (ana['t_mfe'] < hi)).sum()
    print(f'    {label:<15}: {n:>5}  ({n/len(ana):.1%})')

# ── MA context ────────────────────────────────────────────────────────────────
print(f'\n  MA CONTEXT AT ENTRY')
gc = ana['golden_cross'].dropna()
print(f'  {"Golden cross (MA50>MA200)":<35}: {pct((gc == 1).mean())}  of signals')
print(f'  {"Death cross  (MA50<MA200)":<35}: {pct((gc == -1).mean())}  of signals')
print(f'  {"Avg price vs MA50 (pips)":<35}: {pip_fmt(ana["price_vs_ma50"].mean())}  (+ = above MA50)')
print(f'  {"Avg price vs MA200 (pips)":<35}: {pip_fmt(ana["price_vs_ma200"].mean())}  (+ = above MA200)')
print(f'  {"Avg MA50 slope (p/bar)":<35}: {fmt(ana["ma50_slope"].mean(), 3)}')

# Accuracy split by MA alignment
for label, mask in [
    ('Price above MA50  ', ana['price_vs_ma50'] > 0),
    ('Price below MA50  ', ana['price_vs_ma50'] < 0),
    ('Price above MA200 ', ana['price_vs_ma200'] > 0),
    ('Price below MA200 ', ana['price_vs_ma200'] < 0),
    ('Golden cross      ', ana['golden_cross'] == 1),
    ('Death cross       ', ana['golden_cross'] == -1),
]:
    sub = ana[mask]
    if len(sub) < 20: continue
    print(f'    {label}: N={len(sub):>4}  acc={pct(sub["correct"].mean())}  avg={pip_fmt(sub["final_pips"].mean())}  MFE={pip_fmt(sub["mfe"].mean())}  MAE={pip_fmt(sub["mae"].mean())}')

# ── Momentum context ──────────────────────────────────────────────────────────
print(f'\n  MOMENTUM AT ENTRY (avg return before signal, in pip)')
print(f'  {"ret_1h":<15}: {pip_fmt(ana["ret_1h"].mean())}')
print(f'  {"ret_4h":<15}: {pip_fmt(ana["ret_4h"].mean())}')
print(f'  {"ret_12h":<15}: {pip_fmt(ana["ret_12h"].mean())}')
print(f'  {"ret_24h":<15}: {pip_fmt(ana["ret_24h"].mean())}')
print(f'  {"RSI-like (14h)":<15}: {fmt(ana["rsi_raw"].mean())}  (0.5=neutral)')
print(f'  {"Range pos 24h":<15}: {fmt(ana["range_pos_24"].mean())}  (0=at low, 1=at high)')

# Accuracy split by momentum alignment
print(f'\n  Accuracy by momentum alignment:')
ana['momentum_aligned'] = (ana['direction'] * ana['ret_4h'] > 0)
for label, mask in [
    ('4h momentum aligned   ', ana['momentum_aligned'] == True),
    ('4h momentum counter   ', ana['momentum_aligned'] == False),
    ('RSI-like > 0.5 (upward)', ana['rsi_raw'] > 0.5),
    ('RSI-like < 0.5 (downward)', ana['rsi_raw'] < 0.5),
]:
    sub = ana[mask]
    if len(sub) < 20: continue
    print(f'    {label}: N={len(sub):>4}  acc={pct(sub["correct"].mean())}  avg={pip_fmt(sub["final_pips"].mean())}')

# ── Session / timing ──────────────────────────────────────────────────────────
print(f'\n  ACCURACY BY SESSION')
for sess in ['London_open', 'London', 'LN_NY_overlap', 'NY', 'NY_close']:
    sub = ana[ana['session'] == sess]
    if len(sub) < 10: continue
    print(f'    {sess:<20}: N={len(sub):>4}  acc={pct(sub["correct"].mean())}  avg={pip_fmt(sub["final_pips"].mean())}  MFE={pip_fmt(sub["mfe"].mean())}  MAE={pip_fmt(sub["mae"].mean())}')

print(f'\n  ACCURACY BY DAY OF WEEK')
days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri']
for d in range(5):
    sub = ana[ana['dow'] == d]
    if len(sub) < 10: continue
    print(f'    {days[d]}: N={len(sub):>4}  acc={pct(sub["correct"].mean())}  avg={pip_fmt(sub["final_pips"].mean())}')

# ── Per-pair breakdown ────────────────────────────────────────────────────────
print(f'\n{SEP}')
print(f'  PER-PAIR ANATOMY')
print(f'{SEP}')

pair_path_records = {r['pair']: [] for r in records}
for r in records:
    pair_path_records[r['pair']].append(r['_path'])

for pair in sorted(ana['pair'].unique()):
    sub = ana[ana['pair'] == pair]
    if len(sub) < 5:
        continue
    print(f'\n  {pair}  N={len(sub)}  dir={int(sub["direction"].mode()[0]):+d}  acc={pct(sub["correct"].mean())}')
    print(f'  {"":4}  {"MFE":>8}  {"MAE":>8}  {"Final":>8}  {"R:R":>6}  {"t_MFE":>7}  {"t_MAE":>7}  {"MFE1st":>7}  {"Mono":>6}')
    print(f'  {"":4}  {"-"*66}')
    print(f'  {"avg":4}  {sub["mfe"].mean():>8.1f}  {sub["mae"].mean():>8.1f}  {sub["final_pips"].mean():>+8.1f}  '
          f'{sub["mfe_mae_ratio"].mean():>6.2f}  {sub["t_mfe"].mean():>7.1f}  {sub["t_mae"].mean():>7.1f}  '
          f'{sub["mfe_first"].mean():>7.1%}  {sub["monotonicity"].mean():>6.1%}')
    print(f'\n    SL (MAE percentiles): p25={sub["mae"].quantile(0.25):.1f}p  p50={sub["mae"].quantile(0.5):.1f}p  '
          f'p75={sub["mae"].quantile(0.75):.1f}p  p90={sub["mae"].quantile(0.9):.1f}p')
    print(f'    TP (MFE percentiles): p25={sub["mfe"].quantile(0.25):.1f}p  p50={sub["mfe"].quantile(0.5):.1f}p  '
          f'p75={sub["mfe"].quantile(0.75):.1f}p  p90={sub["mfe"].quantile(0.9):.1f}p')

    # MA alignment
    gc = sub['golden_cross'].dropna()
    if len(gc) > 0:
        print(f'    MA: golden={pct((gc==1).mean())}  price_vs_ma50={pip_fmt(sub["price_vs_ma50"].mean())}  '
              f'price_vs_ma200={pip_fmt(sub["price_vs_ma200"].mean())}')

    # Momentum
    print(f'    Momentum: ret_4h={pip_fmt(sub["ret_4h"].mean())}  ret_24h={pip_fmt(sub["ret_24h"].mean())}  '
          f'rsi={fmt(sub["rsi_raw"].mean())}  range_pos={fmt(sub["range_pos_24"].mean())}')

    # Hour-by-hour avg path (show every 6h)
    pair_paths = pair_path_records[pair]
    if pair_paths:
        min_len = min(len(p) for p in pair_paths)
        arr = np.array([p[:min_len] for p in pair_paths])
        avg_path = arr.mean(axis=0)
        checkpoints = [0, 5, 11, 17, 23, 29, 35, 41, 47, 53, 59, 65, 71]
        checkpoints = [c for c in checkpoints if c < len(avg_path)]
        path_str = '  '.join(f'h{c+1:02d}:{avg_path[c]:>+5.0f}p' for c in checkpoints)
        print(f'    Avg path: {path_str}')

# ── Average path shape (all pairs combined) ───────────────────────────────────
print(f'\n{SEP}')
print(f'  AVG PRICE PATH — ALL SIGNALS (pips in signal direction, h1 to h72)')
print(f'{SEP}')
min_len = min(len(p) for p in paths)
all_arr = np.array([p[:min_len] for p in paths])
avg_all = all_arr.mean(axis=0)
print(f'\n  {"Hour":<6}  {"AvgPip":>8}  Bar')
print(f'  {"-"*40}')
for h in range(0, min(72, min_len), 6):
    v   = avg_all[h]
    bar = '|' * int(abs(v) / 3)
    sign = '+' if v >= 0 else '-'
    print(f'  h{h+1:02d}    {v:>+8.1f}  {sign}{bar}')

print(f'\n{SEP}')
print(f'  DONE')
print(f'{SEP}')
