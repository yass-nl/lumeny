"""
Signal Anatomy — 8h MFE Model
==============================
For each signal (MFE>=30, all hours, 8h cooldown),
compute a comprehensive picture of what happens inside the 8h window.

No direction system — this is purely about what the MFE model identifies:
bars where price will make a large excursion (in either direction) within 8h.

Metrics computed per signal:
  Price path:
    - MFE  : max favorable excursion (pips) in 8h — max(up, down)
    - MAE  : max adverse excursion — min(up, down) (opposite side to MFE)
    - Final: net 8h return (pips, unsigned — abs)
    - MFE/MAE ratio
    - Time-to-MFE, Time-to-MAE (hour index 0-7)
    - MFE before MAE?
    - Monotonicity: fraction of hours moving toward MFE direction

  Moving averages at entry:
    - MA50, MA200 level at bar 0
    - Price vs MA50, vs MA200 (pips)
    - MA50 slope (last 10h)
    - MA50 vs MA200 (golden/death cross state)

  Session / timing:
    - Hour of entry
    - Day of week of entry
    - Session bucket

  Volume:
    - Entry bar volume vs 24h avg

  Momentum at entry:
    - Return last 1h, 4h, 8h, 24h before signal
    - RSI-like: fraction of up bars in last 14h
    - Price range position in last 24h

  Per-pair aggregated output:
    - Avg MFE, MAE, Final by pair
    - Hour-by-hour avg price path (0-7h)
    - SL sizing: MAE percentiles p25/50/75/90
    - TP sizing: MFE percentiles p25/50/75/90
    - % of signals where actual MFE >= predicted score
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path

SCRIPT_DIR    = Path(__file__).parent
FEATURES_DIR  = SCRIPT_DIR / '../backend/data/features_9'
PROCESSED_DIR = SCRIPT_DIR / '../backend/data/processed'
MFE_MODEL_PATH = SCRIPT_DIR / '../backend/models_9/mfe_q50_8h/model_1H_Q50.joblib'

START_DATE    = '2024-10-11'
MFE_THRESH    = 30.0
HOURS_ALLOWED = set(range(0, 24))   # all hours — no filter
COOLDOWN_H    = 8
FWD_BARS      = 8
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
print(f'  Model: {MFE_MODEL_PATH.name}  |  features: {len(feature_cols)}  |  cv_pinball: {bundle.get("cv_pinball", "n/a"):.3f}')

# ── Load features_9 ───────────────────────────────────────────────────────────
print('Loading features_9...')
dfs = [pd.read_parquet(f) for f in sorted(FEATURES_DIR.glob('*_features.parquet'))]
df  = pd.concat(dfs).sort_index()
df  = df[df.index >= START_DATE].copy()

# ── Run MFE model ─────────────────────────────────────────────────────────────
print('Running MFE model...')
X = df[feature_cols].ffill().fillna(0)
df['q50_mfe'] = mfe_model.predict(X)
df['hour']    = pd.to_datetime(df.index).hour
df = df[df['q50_mfe'] >= MFE_THRESH].copy()
print(f'  Bars above threshold: {len(df):,}')

# ── Apply 8h cooldown per pair ────────────────────────────────────────────────
print('Applying cooldown...')
candidates = df.sort_index().reset_index()   # 'datetime' becomes a column
cooldown_until = {}
kept_idx = []
for i, row in candidates.iterrows():
    ts = row['datetime']
    p  = row['pair']
    if p in cooldown_until and ts < cooldown_until[p]:
        continue
    cooldown_until[p] = ts + pd.Timedelta(hours=COOLDOWN_H)
    kept_idx.append(i)
df_sig = candidates.iloc[kept_idx].set_index('datetime').copy()
print(f'  Signals after cooldown: {len(df_sig):,}')

# ── Load 1H OHLCV and pre-compute rolling indicators ─────────────────────────
print('Loading 1H price data and pre-computing indicators...')
price_data    = {}
indicator_data = {}

for pair in PAIRS_ALL:
    fpath = PROCESSED_DIR / f'{pair}_1H.parquet'
    if not fpath.exists():
        continue
    bars = pd.read_parquet(fpath).sort_index()
    price_data[pair] = bars
    pip = 0.01 if pair in JPY_PAIRS else 0.0001

    c = bars['close']
    h = bars['high']
    l = bars['low']
    v = bars['volume'] if 'volume' in bars.columns else pd.Series(np.nan, index=bars.index)

    ind = pd.DataFrame(index=bars.index)
    ind['ma50']        = c.rolling(MA_SHORT).mean()
    ind['ma200']       = c.rolling(MA_LONG).mean()
    ind['ma50_slope']  = (c - c.shift(10)) / pip / 10
    ind['ret_1h']      = (c - c.shift(1)).abs() / pip
    ind['ret_4h']      = (c - c.shift(4)).abs() / pip
    ind['ret_8h']      = (c - c.shift(8)).abs() / pip
    ind['ret_24h']     = (c - c.shift(24)).abs() / pip
    # RSI-like: fraction of up bars in last 14h (shift 1 so we don't use current bar)
    ind['rsi_raw']     = c.diff().shift(1).rolling(14).apply(lambda x: (x > 0).mean(), raw=True)
    # Range position in last 24h
    hi24               = h.shift(1).rolling(24).max()
    lo24               = l.shift(1).rolling(24).min()
    rng24              = hi24 - lo24
    ind['range_pos_24']= (c - lo24) / rng24.replace(0, np.nan)
    # Volume ratio
    ind['avg_vol_24h'] = v.shift(1).rolling(24).mean()
    ind['vol_entry']   = v

    indicator_data[pair] = ind

print('  Done pre-computing indicators.')

# ── Compute anatomy per signal ────────────────────────────────────────────────
print('Computing signal anatomy...')
records = []

for i, (ts, sig) in enumerate(df_sig.iterrows()):
    if i % 5000 == 0:
        print(f'  {i:,}/{len(df_sig):,}...')
    pair = sig['pair']
    pip  = 0.01 if pair in JPY_PAIRS else 0.0001

    if pair not in price_data:
        continue
    bars = price_data[pair]
    ind  = indicator_data[pair]

    future = bars[bars.index >= ts].head(FWD_BARS + 1)
    if len(future) < FWD_BARS:
        continue

    entry_bar   = future.iloc[0]
    entry_price = entry_bar['close']
    window      = future.iloc[1:FWD_BARS + 1]   # bars 1..8

    closes = window['close'].values
    highs  = window['high'].values
    lows   = window['low'].values

    # ── MFE / MAE — unsigned, dominant side ──────────────────────────────────
    move_up   = (highs.max()  - entry_price) / pip
    move_down = (entry_price  - lows.min())  / pip

    if move_up >= move_down:
        mfe     = move_up
        mae     = move_down
        mfe_dir = 1
        t_mfe   = int(np.argmax(highs))
        t_mae   = int(np.argmin(lows))
    else:
        mfe     = move_down
        mae     = move_up
        mfe_dir = -1
        t_mfe   = int(np.argmin(lows))
        t_mae   = int(np.argmax(highs))

    mfe_first = t_mfe < t_mae
    final_abs = abs(closes[-1] - entry_price) / pip

    # Monotonicity: fraction of bars moving toward MFE direction
    hourly_moves = np.diff(np.concatenate([[entry_price], closes]))
    mono = (mfe_dir * hourly_moves > 0).mean()

    # Predicted vs actual
    predicted_score = sig['q50_mfe']
    beat_prediction = 1 if mfe >= predicted_score else 0

    # ── Indicators at entry (pre-computed, single lookup) ────────────────────
    if ts in ind.index:
        row = ind.loc[ts]
        ma50_entry    = row['ma50']
        ma200_entry   = row['ma200']
        ma50_slope    = row['ma50_slope']
        price_vs_ma50  = (entry_price - ma50_entry)  / pip if not np.isnan(ma50_entry)  else np.nan
        price_vs_ma200 = (entry_price - ma200_entry) / pip if not np.isnan(ma200_entry) else np.nan
        golden_cross_entry = 1.0 if (not np.isnan(ma50_entry) and not np.isnan(ma200_entry) and ma50_entry > ma200_entry) else (-1.0 if not np.isnan(ma50_entry) else np.nan)
        ret_1h        = row['ret_1h']
        ret_4h        = row['ret_4h']
        ret_8h        = row['ret_8h']
        ret_24h       = row['ret_24h']
        rsi_raw       = row['rsi_raw']
        range_pos_24  = row['range_pos_24']
        avg_vol_24h   = row['avg_vol_24h']
        entry_vol     = row['vol_entry']
        vol_ratio     = entry_vol / avg_vol_24h if avg_vol_24h > 0 and not np.isnan(avg_vol_24h) else np.nan
    else:
        ma50_entry = ma200_entry = ma50_slope = np.nan
        price_vs_ma50 = price_vs_ma200 = golden_cross_entry = np.nan
        ret_1h = ret_4h = ret_8h = ret_24h = rsi_raw = range_pos_24 = vol_ratio = np.nan

    # ── Session ───────────────────────────────────────────────────────────────
    hr = sig['hour']
    if   7  <= hr <  9: session = 'London_open'
    elif 9  <= hr < 12: session = 'London'
    elif 12 <= hr < 14: session = 'LN_NY_overlap'
    elif 14 <= hr < 17: session = 'NY'
    elif 17 <= hr < 21: session = 'NY_close'
    else:               session = 'Off_hours'

    dow = pd.Timestamp(ts).day_of_week

    # ── Path in dominant direction (positive = toward MFE) ───────────────────
    path_pips = mfe_dir * (closes - entry_price) / pip

    records.append({
        'ts':              ts,
        'pair':            pair,
        'mfe_score':       predicted_score,
        'hour':            hr,
        'session':         session,
        'dow':             dow,
        # outcome
        'mfe':             mfe,
        'mae':             mae,
        'final_abs':       final_abs,
        'mfe_dir':         mfe_dir,
        'mfe_mae_ratio':   mfe / mae if mae > 0.1 else np.nan,
        't_mfe':           t_mfe,
        't_mae':           t_mae,
        'mfe_first':       int(mfe_first),
        'monotonicity':    mono,
        'beat_prediction': beat_prediction,
        # MAs
        'price_vs_ma50':   price_vs_ma50,
        'price_vs_ma200':  price_vs_ma200,
        'ma50_slope':      ma50_slope,
        'golden_cross':    golden_cross_entry,
        # momentum
        'ret_1h':          ret_1h,
        'ret_4h':          ret_4h,
        'ret_8h':          ret_8h,
        'ret_24h':         ret_24h,
        'rsi_raw':         rsi_raw,
        'range_pos_24':    range_pos_24,
        # volume
        'vol_ratio':       vol_ratio,
        # path
        '_path':           path_pips,
    })

ana   = pd.DataFrame([{k: v for k, v in r.items() if k != '_path'} for r in records])
paths = [r['_path'] for r in records]
print(f'  Computed {len(ana):,} signal anatomies')

# ── Print results ─────────────────────────────────────────────────────────────
SEP = '=' * 72

def pct(x):    return f'{x:.1%}' if not np.isnan(x) else 'n/a'
def p(x):      return f'{x:+.1f}p' if not np.isnan(x) else 'n/a'
def pu(x):     return f'{x:.1f}p' if not np.isnan(x) else 'n/a'
def fmt(x, d=2): return f'{x:.{d}f}' if not np.isnan(x) else 'n/a'

print(f'\n{SEP}')
print(f'  SIGNAL ANATOMY  —  {len(ana):,} signals  |  MFE>=30  |  all hours  |  8h cooldown  |  8h window')
print(f'  Model: mfe_q50_8h  |  Period: {START_DATE} to present')
print(f'{SEP}')

# ── Model calibration ─────────────────────────────────────────────────────────
print(f'\n  MODEL CALIBRATION  (does actual MFE match the prediction?)')
print(f'  {"Avg predicted score":<35}: {pu(ana["mfe_score"].mean())}  med={pu(ana["mfe_score"].median())}')
print(f'  {"Avg actual MFE":<35}: {pu(ana["mfe"].mean())}  med={pu(ana["mfe"].median())}')
print(f'  {"% signals beating prediction":<35}: {pct(ana["beat_prediction"].mean())}')
print(f'  {"Avg over-performance":<35}: {pu((ana["mfe"] - ana["mfe_score"]).mean())}  (actual - predicted)')
print(f'\n  Calibration by score bucket:')
for lo, hi in [(30,40),(40,50),(50,60),(60,70),(70,90),(90,999)]:
    sub = ana[(ana['mfe_score'] >= lo) & (ana['mfe_score'] < hi)]
    if len(sub) < 20: continue
    beat = sub['beat_prediction'].mean()
    print(f'    score {lo:>3}-{hi:<4}: N={len(sub):>5}  pred={pu(sub["mfe_score"].mean())}  actual={pu(sub["mfe"].mean())}  beat={pct(beat)}')

# ── Overall summary ───────────────────────────────────────────────────────────
print(f'\n{SEP}')
print(f'  OVERALL PRICE ACTION IN 8H WINDOW')
print(f'{SEP}')
print(f'  {"Avg MFE (dominant side)":<35}: {pu(ana["mfe"].mean())}  med={pu(ana["mfe"].median())}')
print(f'  {"Avg MAE (opposite side)":<35}: {pu(ana["mae"].mean())}  med={pu(ana["mae"].median())}')
print(f'  {"Avg final move (abs)":<35}: {pu(ana["final_abs"].mean())}  med={pu(ana["final_abs"].median())}')
print(f'  {"Avg MFE/MAE ratio":<35}: {fmt(ana["mfe_mae_ratio"].mean())}')
print(f'  {"MFE before MAE":<35}: {pct(ana["mfe_first"].mean())}  (dominant move comes first)')
print(f'  {"Avg time to MFE (h)":<35}: {fmt(ana["t_mfe"].mean(), 1)}  med={fmt(ana["t_mfe"].median(), 1)}')
print(f'  {"Avg time to MAE (h)":<35}: {fmt(ana["t_mae"].mean(), 1)}  med={fmt(ana["t_mae"].median(), 1)}')
print(f'  {"Avg monotonicity":<35}: {pct(ana["monotonicity"].mean())}  (% of bars in dominant direction)')
print(f'  {"Up dominant":<35}: {pct((ana["mfe_dir"] == 1).mean())}  |  Down: {pct((ana["mfe_dir"] == -1).mean())}')

print(f'\n  SL SIZING  (MAE percentiles):')
for pv in [25, 50, 75, 90, 95]:
    print(f'    p{pv:<3}: {ana["mae"].quantile(pv/100):>6.1f}p')

print(f'\n  TP SIZING  (MFE percentiles):')
for pv in [25, 50, 75, 90, 95]:
    print(f'    p{pv:<3}: {ana["mfe"].quantile(pv/100):>6.1f}p')

print(f'\n  TIME TO MFE distribution:')
for lo, hi, label in [(0,2,'h0-2'),(2,4,'h2-4'),(4,6,'h4-6'),(6,8,'h6-8')]:
    n = ((ana['t_mfe'] >= lo) & (ana['t_mfe'] < hi)).sum()
    print(f'    {label:<8}: {n:>6}  ({n/len(ana):.1%})')

# ── MAE vs actual MFE ─────────────────────────────────────────────────────────
print(f'\n  MAE BUCKETS vs ACTUAL MFE  (small MAE = cleaner entry?):')
for lo, hi, label in [(0,10,'MAE<10p'),(10,20,'MAE 10-20p'),(20,40,'MAE 20-40p'),(40,999,'MAE>40p')]:
    sub = ana[(ana['mae'] >= lo) & (ana['mae'] < hi)]
    if len(sub) < 20: continue
    print(f'    {label:<14}: N={len(sub):>5}  MFE={pu(sub["mfe"].mean())}  beat_pred={pct(sub["beat_prediction"].mean())}  t_mfe={fmt(sub["t_mfe"].mean(),1)}h')

# ── MA context ────────────────────────────────────────────────────────────────
print(f'\n{SEP}')
print(f'  MA CONTEXT AT ENTRY')
print(f'{SEP}')
gc = ana['golden_cross'].dropna()
print(f'  {"Golden cross (MA50>MA200)":<35}: {pct((gc == 1).mean())}')
print(f'  {"Death cross  (MA50<MA200)":<35}: {pct((gc == -1).mean())}')
print(f'  {"Avg price vs MA50":<35}: {p(ana["price_vs_ma50"].mean())}  (+ = above MA50)')
print(f'  {"Avg price vs MA200":<35}: {p(ana["price_vs_ma200"].mean())}')
print(f'  {"Avg MA50 slope (p/bar)":<35}: {fmt(ana["ma50_slope"].mean(), 3)}')

print(f'\n  MFE by MA context:')
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
    print(f'    {label}: N={len(sub):>5}  MFE={pu(sub["mfe"].mean())}  MAE={pu(sub["mae"].mean())}  beat={pct(sub["beat_prediction"].mean())}')

# ── Momentum context ──────────────────────────────────────────────────────────
print(f'\n{SEP}')
print(f'  MOMENTUM AT ENTRY')
print(f'{SEP}')
print(f'  {"avg |ret_1h|":<20}: {pu(ana["ret_1h"].mean())}')
print(f'  {"avg |ret_4h|":<20}: {pu(ana["ret_4h"].mean())}')
print(f'  {"avg |ret_8h|":<20}: {pu(ana["ret_8h"].mean())}')
print(f'  {"avg |ret_24h|":<20}: {pu(ana["ret_24h"].mean())}')
print(f'  {"RSI-like (14h)":<20}: {fmt(ana["rsi_raw"].mean())}  (0.5=neutral)')
print(f'  {"Range pos 24h":<20}: {fmt(ana["range_pos_24"].mean())}  (0=at low, 1=at high)')

print(f'\n  MFE by momentum magnitude (|ret_4h|):')
for lo, hi, label in [(0,5,'<5p'),(5,15,'5-15p'),(15,30,'15-30p'),(30,999,'>30p')]:
    sub = ana[(ana['ret_4h'] >= lo) & (ana['ret_4h'] < hi)]
    if len(sub) < 20: continue
    print(f'    ret_4h {label:<8}: N={len(sub):>5}  MFE={pu(sub["mfe"].mean())}  MAE={pu(sub["mae"].mean())}  beat={pct(sub["beat_prediction"].mean())}')

# ── Session / timing ──────────────────────────────────────────────────────────
print(f'\n{SEP}')
print(f'  SESSION & TIMING')
print(f'{SEP}')
print(f'\n  By session:')
for sess in ['London_open','London','LN_NY_overlap','NY','NY_close','Off_hours']:
    sub = ana[ana['session'] == sess]
    if len(sub) < 10: continue
    print(f'    {sess:<18}: N={len(sub):>5}  MFE={pu(sub["mfe"].mean())}  MAE={pu(sub["mae"].mean())}  beat={pct(sub["beat_prediction"].mean())}  t_mfe={fmt(sub["t_mfe"].mean(),1)}h')

print(f'\n  By day of week:')
for d, name in enumerate(['Mon','Tue','Wed','Thu','Fri']):
    sub = ana[ana['dow'] == d]
    if len(sub) < 10: continue
    print(f'    {name}: N={len(sub):>5}  MFE={pu(sub["mfe"].mean())}  MAE={pu(sub["mae"].mean())}  beat={pct(sub["beat_prediction"].mean())}')

print(f'\n  By hour of day:')
for h in range(24):
    sub = ana[ana['hour'] == h]
    if len(sub) < 20: continue
    print(f'    h{h:02d}: N={len(sub):>5}  MFE={pu(sub["mfe"].mean())}  MAE={pu(sub["mae"].mean())}  beat={pct(sub["beat_prediction"].mean())}')

# ── Per-pair breakdown ────────────────────────────────────────────────────────
print(f'\n{SEP}')
print(f'  PER-PAIR ANATOMY')
print(f'{SEP}')

pair_path_records = {r['pair']: [] for r in records}
for r in records:
    pair_path_records[r['pair']].append(r['_path'])

for pair in sorted(ana['pair'].unique()):
    sub = ana[ana['pair'] == pair]
    if len(sub) < 10:
        continue
    print(f'\n  {pair}  N={len(sub)}  beat_pred={pct(sub["beat_prediction"].mean())}')
    print(f'    MFE={pu(sub["mfe"].mean())} (med {pu(sub["mfe"].median())})  '
          f'MAE={pu(sub["mae"].mean())} (med {pu(sub["mae"].median())})  '
          f'R:R={fmt(sub["mfe_mae_ratio"].mean())}  '
          f't_mfe={fmt(sub["t_mfe"].mean(),1)}h  MFE1st={pct(sub["mfe_first"].mean())}')
    print(f'    SL: p25={sub["mae"].quantile(0.25):.1f}p  p50={sub["mae"].quantile(0.5):.1f}p  '
          f'p75={sub["mae"].quantile(0.75):.1f}p  p90={sub["mae"].quantile(0.9):.1f}p')
    print(f'    TP: p25={sub["mfe"].quantile(0.25):.1f}p  p50={sub["mfe"].quantile(0.5):.1f}p  '
          f'p75={sub["mfe"].quantile(0.75):.1f}p  p90={sub["mfe"].quantile(0.9):.1f}p')

    # Avg path in dominant direction
    pair_paths = pair_path_records[pair]
    if pair_paths:
        min_len = min(len(p) for p in pair_paths)
        arr = np.array([p[:min_len] for p in pair_paths])
        avg_path = arr.mean(axis=0)
        checkpoints = [c for c in range(min_len)]
        path_str = '  '.join(f'h{c+1}:{avg_path[c]:>+5.0f}p' for c in checkpoints)
        print(f'    Avg path: {path_str}')

# ── Average path shape (all pairs combined) ───────────────────────────────────
print(f'\n{SEP}')
print(f'  AVG PRICE PATH — ALL SIGNALS (pips in dominant direction, h1 to h8)')
print(f'{SEP}')
min_len  = min(len(p) for p in paths)
all_arr  = np.array([p[:min_len] for p in paths])
avg_all  = all_arr.mean(axis=0)
std_all  = all_arr.std(axis=0)
print(f'\n  {"Hour":<6}  {"AvgPip":>8}  {"StdDev":>8}  Bar')
print(f'  {"-"*50}')
for h in range(min(FWD_BARS, min_len)):
    v   = avg_all[h]
    s   = std_all[h]
    bar = '|' * int(abs(v) / 2)
    sgn = '+' if v >= 0 else '-'
    print(f'  h{h+1:<4}  {v:>+8.1f}  {s:>8.1f}  {sgn}{bar}')

print(f'\n{SEP}')
print(f'  DONE')
print(f'{SEP}')
