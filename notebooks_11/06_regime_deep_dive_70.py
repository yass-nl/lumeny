"""
Regime Deep Dive
=================
Comprehensive analysis of MFE>=70 signals + directional system.
No intraday noise — purely understanding the 72h window structure.

Questions answered:
  1. MFE model accuracy alone (ignoring direction)
  2. Direction system accuracy alone (on MFE>=70 bars)
  3. Combined accuracy (MFE>=70 + direction)
  4. When does the ACTUAL MFE start? (hour within 72h window)
  5. How long does the MFE last / when does it resolve?
  6. MAE: how far does price go against before the MFE?
  7. MA50 / MA200 context at signal bar (per LONG/SHORT bias)
  8. MFE vs MAE ratio (R:R of the natural move)
  9. Session and day-of-week breakdown
  10. Monthly stability
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path

SCRIPT_DIR     = Path(__file__).parent
FEATURES_DIR   = SCRIPT_DIR / '../backend/data/features_9'
PROCESSED_DIR  = SCRIPT_DIR / '../backend/data/processed'
MFE_MODEL_PATH = SCRIPT_DIR / '../backend/models_9/mfe_q50/model_1H_Q50.joblib'

START_DATE  = '2024-10-11'
MFE_THRESH  = 70.0
HOURS_ALLOWED = set(range(7, 21))
COOLDOWN_H  = 72
WINDOW_H    = 72
MA_SHORT    = 50
MA_LONG     = 200

JPY_PAIRS = {'USDJPY','EURJPY','GBPJPY','AUDJPY','CADJPY','CHFJPY'}
PAIRS_ALL = [
    'EURUSD','GBPUSD','USDJPY','USDCHF','AUDUSD','USDCAD','NZDUSD',
    'EURJPY','GBPJPY','EURGBP','EURAUD','AUDJPY','CADJPY','CHFJPY','AUDNZD',
]

# ── Load model + features ─────────────────────────────────────────────────────
print('Loading MFE model...')
bundle       = joblib.load(MFE_MODEL_PATH)
mfe_model    = bundle['model']
feature_cols = bundle['feature_cols']

print('Loading features_9...')
dfs = [pd.read_parquet(f) for f in sorted(FEATURES_DIR.glob('*_features.parquet'))]
df9 = pd.concat(dfs).sort_index()
df9 = df9[df9.index >= START_DATE].copy()

print('Running MFE model...')
X = df9[feature_cols].ffill().fillna(0)
df9['q50_mfe'] = mfe_model.predict(X)
df9['hour']    = pd.to_datetime(df9.index).hour

# ── Direction rules ───────────────────────────────────────────────────────────
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

df9['direction'] = apply_direction_rules(df9)

# ── Two pools: MFE-only and MFE+direction ─────────────────────────────────────
df_mfe = df9[(df9['q50_mfe'] >= MFE_THRESH) & df9['hour'].isin(HOURS_ALLOWED)].copy()
df_dir = df_mfe[df_mfe['direction'].notna()].copy()

# Apply 72h cooldown to both pools separately
def apply_cooldown(df):
    cooldown = {}
    kept = []
    for ts, row in df.sort_index().iterrows():
        p = row['pair']
        if p in cooldown and ts < cooldown[p]: continue
        cooldown[p] = ts + pd.Timedelta(hours=COOLDOWN_H)
        kept.append(ts)
    return df.loc[kept].copy()

df_mfe_sig = apply_cooldown(df_mfe)
df_dir_sig = apply_cooldown(df_dir)

print(f'  MFE>=70 signals:            {len(df_mfe_sig):,}')
print(f'  MFE>=70 + direction signals: {len(df_dir_sig):,}')

# ── Load 1H price data ────────────────────────────────────────────────────────
print('Loading 1H price data...')
price_data = {}
for pair in PAIRS_ALL:
    fpath = PROCESSED_DIR / f'{pair}_1H.parquet'
    if fpath.exists():
        price_data[pair] = pd.read_parquet(fpath).sort_index()

# ── Per-signal analysis function ──────────────────────────────────────────────
def analyse_signals(df_sig, label, use_direction=False):
    print(f'\n  Processing {len(df_sig):,} signals for [{label}]...')
    records = []

    for ts, sig in df_sig.iterrows():
        pair      = sig['pair']
        pip       = 0.01 if pair in JPY_PAIRS else 0.0001
        direction = sig.get('direction', np.nan) if use_direction else np.nan

        if pair not in price_data:
            continue
        bars = price_data[pair]

        # 72h window
        future = bars[bars.index >= ts].head(WINDOW_H + 1)
        if len(future) < WINDOW_H:
            continue

        entry_price = future.iloc[0]['close']
        window      = future.iloc[1:WINDOW_H + 1]
        closes      = window['close'].values
        highs       = window['high'].values
        lows        = window['low'].values

        # ── fwd_72h ──────────────────────────────────────────────────────────
        fwd_72h = (closes[-1] - entry_price) / pip

        # ── MFE / MAE — direction-agnostic, consistent with model target ────
        # Model was trained on max(up_move, down_move) — the dominant move
        # MFE = dominant move (the bigger of up/down) — always >= MAE
        # MAE = the smaller side — always <= MFE
        up_move   = (highs.max() - entry_price) / pip
        down_move = (entry_price - lows.min())  / pip
        if up_move >= down_move:
            mfe_val   = up_move
            mae_val   = down_move
            t_mfe_val = int(np.argmax(highs))
            t_mae_val = int(np.argmin(lows))
        else:
            mfe_val   = down_move
            mae_val   = up_move
            t_mfe_val = int(np.argmin(lows))
            t_mae_val = int(np.argmax(highs))

        # ── Directional outcome (direction system only, fully separate) ───────
        if use_direction and not np.isnan(direction):
            dir_int    = int(direction)
            fwd_signed = dir_int * fwd_72h
            correct    = 1 if fwd_signed > 0 else 0
        else:
            dir_int    = np.nan
            fwd_signed = np.nan
            correct    = np.nan

        # ── MFE/MAE ratio ─────────────────────────────────────────────────────
        mfe_mae_ratio = mfe_val / mae_val if mae_val > 1 else np.nan

        # ── Did the smaller move (MAE side) come before the dominant (MFE)? ──
        mae_before_mfe = 1 if t_mae_val < t_mfe_val else 0

        # ── Moving averages at entry ──────────────────────────────────────────
        hist = bars[bars.index < ts].tail(MA_LONG + 10)
        if len(hist) >= MA_LONG:
            c_hist  = hist['close'].values
            ma50    = c_hist[-MA_SHORT:].mean()
            ma200   = c_hist[-MA_LONG:].mean()
            p_vs_50  = (entry_price - ma50)  / pip
            p_vs_200 = (entry_price - ma200) / pip
            # MA position category
            if entry_price > ma50 and entry_price > ma200:
                ma_cat = 'above_both'
            elif entry_price < ma50 and entry_price < ma200:
                ma_cat = 'below_both'
            elif entry_price > ma50 and entry_price < ma200:
                ma_cat = 'above50_below200'
            else:
                ma_cat = 'below50_above200'
            # MA alignment
            if ma50 > ma200:
                ma_align = 'golden'   # bullish structure
            else:
                ma_align = 'death'    # bearish structure
            ma50_slope  = (c_hist[-1] - c_hist[-11]) / pip / 10
            ma200_slope = (c_hist[-1] - c_hist[-MA_LONG]) / pip / MA_LONG
        else:
            ma50 = ma200 = p_vs_50 = p_vs_200 = np.nan
            ma_cat = 'unknown'
            ma_align = 'unknown'
            ma50_slope = ma200_slope = np.nan

        # ── Session ───────────────────────────────────────────────────────────
        h = pd.Timestamp(ts).hour
        if   7  <= h <  9: session = 'London_open'
        elif 9  <= h < 12: session = 'London'
        elif 12 <= h < 14: session = 'LN_NY_overlap'
        elif 14 <= h < 17: session = 'NY'
        else:              session = 'NY_close'

        records.append({
            'ts':            ts,
            'pair':          pair,
            'mfe_score':     sig['q50_mfe'],
            'direction':     dir_int,
            'hour':          h,
            'session':       session,
            'dow':           pd.Timestamp(ts).day_of_week,
            'month':         pd.Timestamp(ts).to_period('M'),
            # outcome
            'fwd_72h':       fwd_72h,
            'fwd_signed':    fwd_signed,
            'correct':       correct,
            'mfe':           mfe_val,   # max upward excursion (direction-agnostic)
            'mae':           mae_val,   # max downward excursion (direction-agnostic)
            't_mfe':         t_mfe_val,
            't_mae':         t_mae_val,
            'mae_before_mfe': mae_before_mfe,
            'mfe_mae_ratio': mfe_mae_ratio,
            # MAs
            'p_vs_ma50':     p_vs_50,
            'p_vs_ma200':    p_vs_200,
            'ma_cat':        ma_cat,
            'ma_align':      ma_align,
            'ma50_slope':    ma50_slope,
        })

    return pd.DataFrame(records)


ana_mfe = analyse_signals(df_mfe_sig, 'MFE>=70 only',  use_direction=False)
ana_dir = analyse_signals(df_dir_sig, 'MFE+Direction', use_direction=True)

# ── Output ────────────────────────────────────────────────────────────────────
SEP  = '=' * 72
SEP2 = '-' * 72
months_mfe = max((df_mfe_sig.index.max() - df_mfe_sig.index.min()).days / 30, 0.1)
months_dir = max((df_dir_sig.index.max() - df_dir_sig.index.min()).days / 30, 0.1)

def pct(x): return f'{x:.1%}' if not np.isnan(x) else 'n/a'
def p(x, d=1): return f'{x:+.{d}f}p' if not np.isnan(x) else 'n/a'
def f(x, d=1): return f'{x:.{d}f}' if not np.isnan(x) else 'n/a'

print(f'\n\n{SEP}')
print(f'  REGIME DEEP DIVE  —  MFE>=70  |  hours 7-20 UTC  |  72h cooldown')
print(f'  Period: {START_DATE} to present  |  last 18 months')
print(f'{SEP}')

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1: Accuracy overview
# ─────────────────────────────────────────────────────────────────────────────
print(f'\n{SEP}')
print(f'  1. ACCURACY OVERVIEW')
print(f'{SEP}')

# MFE-only: accuracy of fwd_72h > 0 (does price go up more than it went down)
ana_mfe['correct_long']  = (ana_mfe['fwd_72h'] > 0).astype(float)
ana_mfe['correct_short'] = (ana_mfe['fwd_72h'] < 0).astype(float)
print(f'\n  MFE>=70 alone  (N={len(ana_mfe):,}, {len(ana_mfe)/months_mfe:.0f}/mo):')
print(f'    fwd_72h > 0  (bullish): {pct(ana_mfe["correct_long"].mean())}   avg={p(ana_mfe[ana_mfe["fwd_72h"]>0]["fwd_72h"].mean())}')
print(f'    fwd_72h < 0  (bearish): {pct(ana_mfe["correct_short"].mean())}   avg={p(ana_mfe[ana_mfe["fwd_72h"]<0]["fwd_72h"].mean())}')
print(f'    avg |fwd_72h|: {p(ana_mfe["fwd_72h"].abs().mean())}')
print(f'    avg MFE (best seen):    {p(ana_mfe["mfe"].mean())}  med={p(ana_mfe["mfe"].median())}')
print(f'    avg MAE (worst seen):   {p(ana_mfe["mae"].mean())}  med={p(ana_mfe["mae"].median())}')
print(f'    avg MFE/MAE ratio:      {f(ana_mfe["mfe_mae_ratio"].mean())}')

print(f'\n  MFE>=70 + Direction  (N={len(ana_dir):,}, {len(ana_dir)/months_dir:.0f}/mo):')
print(f'    Directional accuracy:   {pct(ana_dir["correct"].mean())}')
print(f'    avg fwd_72h in dir:     {p(ana_dir["fwd_signed"].mean())}  med={p(ana_dir["fwd_signed"].median())}')
print(f'    avg MFE (in dir):       {p(ana_dir["mfe"].mean())}  med={p(ana_dir["mfe"].median())}')
print(f'    avg MAE (vs dir):       {p(ana_dir["mae"].mean())}  med={p(ana_dir["mae"].median())}')
print(f'    avg MFE/MAE ratio:      {f(ana_dir["mfe_mae_ratio"].mean())}')

# LONG vs SHORT split
for d, label in [(1, 'LONG'), (-1, 'SHORT')]:
    sub = ana_dir[ana_dir['direction'] == d]
    if len(sub) < 5: continue
    print(f'\n    {label} (N={len(sub)}):')
    print(f'      accuracy={pct(sub["correct"].mean())}  avg_dir={p(sub["fwd_signed"].mean())}')
    print(f'      MFE={p(sub["mfe"].mean())}  MAE={p(sub["mae"].mean())}  R:R={f(sub["mfe_mae_ratio"].mean())}')

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 & 3: When does MFE start and resolve?
# ─────────────────────────────────────────────────────────────────────────────
print(f'\n{SEP}')
print(f'  2 & 3. MFE TIMING  (when does the move start and peak?)')
print(f'{SEP}')

for df_a, label in [(ana_mfe, 'MFE>=70 only'), (ana_dir, 'MFE+Direction')]:
    print(f'\n  [{label}]')
    print(f'    Avg hour of highest high (h0=entry): {f(df_a["t_mfe"].mean())}h  '
          f'med={f(df_a["t_mfe"].median())}h')
    print(f'    Avg hour of lowest low  (h0=entry): {f(df_a["t_mae"].mean())}h  '
          f'med={f(df_a["t_mae"].median())}h')
    print(f'    Low comes before high:               {pct(df_a["mae_before_mfe"].mean())}  of signals')

    bins = [(0,12,'h00-12'),(12,24,'h12-24'),(24,36,'h24-36'),(36,48,'h36-48'),(48,60,'h48-60'),(60,72,'h60-72')]
    print(f'\n    Highest high timing distribution:')
    for lo, hi, bl in bins:
        n = ((df_a['t_mfe'] >= lo) & (df_a['t_mfe'] < hi)).sum()
        print(f'      {bl}: {n:>4}  ({n/len(df_a):.1%})')

    print(f'\n    Lowest low timing distribution:')
    for lo, hi, bl in bins:
        n = ((df_a['t_mae'] >= lo) & (df_a['t_mae'] < hi)).sum()
        print(f'      {bl}: {n:>4}  ({n/len(df_a):.1%})')

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4: MAE detail
# ─────────────────────────────────────────────────────────────────────────────
print(f'\n{SEP}')
print(f'  4. MFE / MAE  (unsigned excursions from entry, direction-agnostic)')
print(f'{SEP}')

for df_a, label in [(ana_mfe, 'MFE>=70 only'), (ana_dir, 'MFE+Direction')]:
    mae = df_a['mae'].dropna()
    print(f'\n  [{label}]  MAE percentiles:')
    for p_val in [10, 25, 50, 75, 90, 95]:
        print(f'    p{p_val:<3}: {mae.quantile(p_val/100):>7.1f}p')
    print(f'    Mean:  {mae.mean():>7.1f}p')
    pct_large = (mae > 30).mean()
    pct_small = (mae < 15).mean()
    print(f'    MAE < 15p (small):  {pct(pct_small)}  — tight SL workable')
    print(f'    MAE > 30p (large):  {pct(pct_large)}  — needs wide SL or no SL')

# MAE vs outcome (does small MAE = better result?)
print(f'\n  MAE vs outcome (MFE+Direction):')
for lo, hi, label in [(0,15,'MAE<15p'),(15,30,'MAE 15-30p'),(30,60,'MAE 30-60p'),(60,999,'MAE>60p')]:
    sub = ana_dir[(ana_dir['mae'] >= lo) & (ana_dir['mae'] < hi)]
    if len(sub) < 10: continue
    print(f'    {label:<15}: N={len(sub):>4}  acc={pct(sub["correct"].mean())}  '
          f'avg_dir={p(sub["fwd_signed"].mean())}  MFE={p(sub["mfe"].mean())}')

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5: MA context
# ─────────────────────────────────────────────────────────────────────────────
print(f'\n{SEP}')
print(f'  5. MA50 / MA200 CONTEXT AT SIGNAL BAR')
print(f'{SEP}')

print(f'\n  Overall MA position distribution (MFE+Direction):')
for cat in ['above_both','below_both','above50_below200','below50_above200']:
    n = (ana_dir['ma_cat'] == cat).sum()
    sub = ana_dir[ana_dir['ma_cat'] == cat]
    if len(sub) < 5: continue
    print(f'    {cat:<22}: {n:>5}  ({n/len(ana_dir):.1%})  '
          f'acc={pct(sub["correct"].mean())}  avg={p(sub["fwd_signed"].mean())}  '
          f'MFE={p(sub["mfe"].mean())}  MAE={p(sub["mae"].mean())}')

print(f'\n  MA alignment:')
for cat in ['golden','death']:
    sub = ana_dir[ana_dir['ma_align'] == cat]
    if len(sub) < 5: continue
    print(f'    {cat:<8}: N={len(sub):>4}  acc={pct(sub["correct"].mean())}  '
          f'avg={p(sub["fwd_signed"].mean())}  MFE={p(sub["mfe"].mean())}  MAE={p(sub["mae"].mean())}')

print(f'\n  Per LONG / SHORT bias:')
for d, dlabel in [(1,'LONG'), (-1,'SHORT')]:
    sub_d = ana_dir[ana_dir['direction'] == d]
    if len(sub_d) < 5: continue
    print(f'\n    {dlabel} bias (N={len(sub_d)}):')
    for cat in ['above_both','below_both','above50_below200','below50_above200']:
        sub = sub_d[sub_d['ma_cat'] == cat]
        if len(sub) < 5: continue
        print(f'      {cat:<22}: N={len(sub):>4} ({len(sub)/len(sub_d):.1%})  '
              f'acc={pct(sub["correct"].mean())}  avg={p(sub["fwd_signed"].mean())}  '
              f'MFE={p(sub["mfe"].mean())}  MAE={p(sub["mae"].mean())}')
    print(f'      avg price vs MA50 : {p(sub_d["p_vs_ma50"].mean())}')
    print(f'      avg price vs MA200: {p(sub_d["p_vs_ma200"].mean())}')
    print(f'      avg MA50 slope    : {sub_d["ma50_slope"].mean():+.3f} p/bar')

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6: Per-pair deep stats
# ─────────────────────────────────────────────────────────────────────────────
print(f'\n{SEP}')
print(f'  6. PER-PAIR SUMMARY (MFE+Direction)')
print(f'{SEP}')
print(f'\n  {"Pair":<10}  {"N":>5}  {"Acc":>7}  {"AvgDir":>8}  {"MFE":>8}  {"MAE":>8}  {"R:R":>6}  {"t_MFE":>7}  {"MAEbfr":>7}')
print(f'  {"-"*80}')
for pair in sorted(ana_dir['pair'].unique()):
    sub = ana_dir[ana_dir['pair'] == pair]
    if len(sub) < 5: continue
    print(f'  {pair:<10}  {len(sub):>5}  {pct(sub["correct"].mean()):>7}  '
          f'{p(sub["fwd_signed"].mean()):>8}  {p(sub["mfe"].mean()):>8}  '
          f'{p(sub["mae"].mean()):>8}  {f(sub["mfe_mae_ratio"].mean()):>6}  '
          f'{f(sub["t_mfe"].mean()):>7}  {pct(sub["mae_before_mfe"].mean()):>7}')

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7: Session + DoW
# ─────────────────────────────────────────────────────────────────────────────
print(f'\n{SEP}')
print(f'  7. SESSION & DAY-OF-WEEK  (MFE+Direction)')
print(f'{SEP}')
print(f'\n  By session:')
for sess in ['London_open','London','LN_NY_overlap','NY','NY_close']:
    sub = ana_dir[ana_dir['session'] == sess]
    if len(sub) < 5: continue
    print(f'    {sess:<18}: N={len(sub):>4}  acc={pct(sub["correct"].mean())}  '
          f'avg={p(sub["fwd_signed"].mean())}  MFE={p(sub["mfe"].mean())}  MAE={p(sub["mae"].mean())}')

print(f'\n  By day of week:')
days = ['Mon','Tue','Wed','Thu','Fri']
for d in range(5):
    sub = ana_dir[ana_dir['dow'] == d]
    if len(sub) < 5: continue
    print(f'    {days[d]}: N={len(sub):>4}  acc={pct(sub["correct"].mean())}  '
          f'avg={p(sub["fwd_signed"].mean())}  MFE={p(sub["mfe"].mean())}  MAE={p(sub["mae"].mean())}')

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8: Monthly stability
# ─────────────────────────────────────────────────────────────────────────────
print(f'\n{SEP}')
print(f'  8. MONTHLY STABILITY  (MFE+Direction)')
print(f'{SEP}')
print(f'\n  {"Month":<10}  {"N":>5}  {"Acc":>7}  {"AvgDir":>8}  {"MFE":>8}  {"MAE":>8}')
print(f'  {"-"*55}')
for month, sub in ana_dir.groupby('month'):
    flag = ' <<' if sub['correct'].mean() < 0.45 else (' >>' if sub['correct'].mean() > 0.65 else '')
    print(f'  {str(month):<10}  {len(sub):>5}  {pct(sub["correct"].mean()):>7}  '
          f'{p(sub["fwd_signed"].mean()):>8}  {p(sub["mfe"].mean()):>8}  '
          f'{p(sub["mae"].mean()):>8}{flag}')

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 9: MFE score distribution → accuracy lift
# ─────────────────────────────────────────────────────────────────────────────
print(f'\n{SEP}')
print(f'  9. MFE SCORE vs ACCURACY  (does higher score = better outcome?)')
print(f'{SEP}')
print(f'\n  MFE+Direction:')
bins = [(50,60),(60,70),(70,80),(80,90),(90,110),(110,130),(130,999)]
for lo, hi in bins:
    sub = ana_dir[(ana_dir['mfe_score'] >= lo) & (ana_dir['mfe_score'] < hi)]
    if len(sub) < 5: continue
    print(f'    score {lo:>3}-{hi:<4}: N={len(sub):>4}  acc={pct(sub["correct"].mean())}  '
          f'avg={p(sub["fwd_signed"].mean())}  MFE={p(sub["mfe"].mean())}')

print(f'\n{SEP}')
print(f'  DONE')
print(f'{SEP}')
