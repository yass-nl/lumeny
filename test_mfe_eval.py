"""
MFE Model Evaluation — Live Data (8h model)
=============================================
Evaluates the mfe_q50_8h model on live Polygon data.
No direction, no trades — purely model accuracy.

MFE = dominant side (max of up/down excursion over 8h)  — always >= MAE
MAE = opposite side                                      — always <= MFE
"""

import os, asyncio, sys
import pandas as pd
import numpy as np
import joblib
import httpx
import shap
from datetime import datetime, timedelta
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
API_KEY   = os.getenv('POLYGON_S3_SECRET_KEY', '')
REST_BASE = 'https://api.polygon.io'

sys.path.insert(0, str(Path(__file__).parent / 'backend'))
from features import compute_features_for_pair, PIP_SIZE
from live_features_extra import compute_momentum_calendar_features

# ── Config ────────────────────────────────────────────────────────────────────
MFE_THRESH    = 100.0
MFE_HORIZON_H = 8
FETCH_DAYS    = 220
TEST_DAYS     = 180

PAIRS = [
    'EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD', 'USDCAD', 'NZDUSD',
    'EURJPY', 'GBPJPY', 'EURGBP', 'EURAUD', 'AUDJPY', 'CADJPY', 'CHFJPY', 'AUDNZD',
]

CURRENCY_SIGN = {
    'EURUSD': {'EUR': +1, 'USD': -1}, 'GBPUSD': {'GBP': +1, 'USD': -1},
    'USDJPY': {'USD': +1, 'JPY': -1}, 'USDCHF': {'USD': +1, 'CHF': -1},
    'AUDUSD': {'AUD': +1, 'USD': -1}, 'USDCAD': {'USD': +1, 'CAD': -1},
    'NZDUSD': {'NZD': +1, 'USD': -1}, 'EURJPY': {'EUR': +1, 'JPY': -1},
    'GBPJPY': {'GBP': +1, 'JPY': -1}, 'EURGBP': {'EUR': +1, 'GBP': -1},
    'EURAUD': {'EUR': +1, 'AUD': -1}, 'AUDJPY': {'AUD': +1, 'JPY': -1},
    'CADJPY': {'CAD': +1, 'JPY': -1}, 'CHFJPY': {'CHF': +1, 'JPY': -1},
    'AUDNZD': {'AUD': +1, 'NZD': -1},
}

JPY_PAIRS = {'USDJPY', 'EURJPY', 'GBPJPY', 'AUDJPY', 'CADJPY', 'CHFJPY'}


# ── Polygon fetch ─────────────────────────────────────────────────────────────
async def fetch_bars(client, pair, from_date, to_date):
    ticker = f'C:{pair}'
    url    = f'{REST_BASE}/v2/aggs/ticker/{ticker}/range/1/minute/{from_date}/{to_date}'
    params = {'apiKey': API_KEY, 'limit': 50000, 'sort': 'asc'}
    all_results = []
    resp = await client.get(url, params=params)
    resp.raise_for_status()
    data = resp.json()
    all_results.extend(data.get('results', []))
    while 'next_url' in data:
        sep  = '&' if '?' in data['next_url'] else '?'
        resp = await client.get(f"{data['next_url']}{sep}apiKey={API_KEY}")
        resp.raise_for_status()
        data = resp.json()
        all_results.extend(data.get('results', []))
    if not all_results:
        return pd.DataFrame()
    df = pd.DataFrame(all_results)
    df['datetime'] = pd.to_datetime(df['t'], unit='ms', utc=True).dt.tz_localize(None)
    df = df.rename(columns={'o':'open','h':'high','l':'low','c':'close','v':'volume'})
    df = df.set_index('datetime')[['open','high','low','close','volume']]
    df = df.sort_index().drop_duplicates()
    df = df[~((df.index.dayofweek == 5) | ((df.index.dayofweek == 6) & (df.index.hour < 21)))]
    return df

async def fetch_all_pairs():
    now       = datetime.now()
    to_date   = now.strftime('%Y-%m-%d')
    from_date = (now - timedelta(days=FETCH_DAYS)).strftime('%Y-%m-%d')
    print(f'Fetching {from_date} -> {to_date}')
    raw = {}
    async with httpx.AsyncClient(timeout=120) as client:
        for pair in PAIRS:
            print(f'  {pair}...', end=' ', flush=True)
            df_1m = await fetch_bars(client, pair, from_date, to_date)
            await asyncio.sleep(0.3)
            if df_1m.empty:
                print('NO DATA'); continue
            df_5m  = df_1m.resample('5min').agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum'}).dropna()
            df_15m = df_1m.resample('15min').agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum'}).dropna()
            df_1h  = df_1m.resample('1h').agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum'}).dropna()
            raw[pair] = {'1m': df_1m, '5m': df_5m, '15m': df_15m, '1h': df_1h}
            print(f'{len(df_1m):,} 1m / {len(df_1h)} 1h bars')
    return raw, from_date, to_date


# ── Cross-pair features ───────────────────────────────────────────────────────
def compute_all_cross_pair_features(close_1h_all):
    returns_all = {p: np.log(c / c.shift(1)) for p, c in close_1h_all.items()}
    returns_df  = pd.DataFrame(returns_all)

    currencies = ['EUR', 'USD', 'GBP', 'JPY', 'AUD', 'NZD', 'CAD', 'CHF']
    csi = {}
    for ccy in currencies:
        comps = [CURRENCY_SIGN[p][ccy] * returns_df[p]
                 for p in PAIRS if ccy in CURRENCY_SIGN.get(p, {}) and p in returns_df]
        if comps:
            csi[f'csi_{ccy.lower()}'] = pd.concat(comps, axis=1).mean(axis=1)
    csi_df = pd.DataFrame(csi)
    csi_rolling = {}
    for col in csi_df.columns:
        csi_rolling[f'{col}_24h'] = csi_df[col].rolling(24,  min_periods=8).sum()
        csi_rolling[f'{col}_72h'] = csi_df[col].rolling(72,  min_periods=24).sum()
    csi_rolling_df = pd.DataFrame(csi_rolling)

    result = {}
    for pair in PAIRS:
        if pair not in returns_df.columns:
            continue
        r      = returns_df[pair]
        c_pair = close_1h_all[pair]
        cols   = {}
        for peer in [p for p in PAIRS if p != pair]:
            if peer not in returns_df.columns:
                continue
            p_ret  = returns_df[peer]
            c_peer = close_1h_all[peer]
            sl     = peer.lower()
            for w, lbl in [(24, '24h'), (72, '3d'), (168, '1w')]:
                cols[f'corr_{sl}_{lbl}'] = r.rolling(w, min_periods=w//2).corr(p_ret)
            cols[f'corr_regime_{sl}'] = cols[f'corr_{sl}_24h'] - cols[f'corr_{sl}_1w']
            for w, lbl in [(24, '24h'), (168, '1w')]:
                cov = r.rolling(w, min_periods=w//2).cov(p_ret)
                var = p_ret.rolling(w, min_periods=w//2).var().clip(lower=1e-12)
                cols[f'beta_{sl}_{lbl}'] = cov / var
            cols[f'relstr_{sl}_1h']    = r - p_ret
            cols[f'relstr_{sl}_4h']    = np.log(c_pair / c_pair.shift(4))  - np.log(c_peer / c_peer.shift(4))
            cols[f'relstr_{sl}_24h']   = np.log(c_pair / c_pair.shift(24)) - np.log(c_peer / c_peer.shift(24))
            cols[f'peer_{sl}_ret_1h']  = p_ret
            cols[f'peer_{sl}_ret_4h']  = np.log(c_peer / c_peer.shift(4))
            cols[f'peer_{sl}_ret_24h'] = np.log(c_peer / c_peer.shift(24))
        for col in csi_df.columns:
            cols[col]          = csi_df[col]
            cols[f'{col}_24h'] = csi_rolling_df[f'{col}_24h']
            cols[f'{col}_72h'] = csi_rolling_df[f'{col}_72h']
        result[pair] = pd.DataFrame(cols, index=r.index).astype(np.float32)
    return result


# ── Compute actual MFE + MAE + path ──────────────────────────────────────────
def compute_window_stats(df_1h, pos, pip_size):
    """
    MFE = dominant side (max of up/down) — always the bigger move
    MAE = opposite side                  — always the smaller move
    Returns dict with mfe, mae, mfe_dir, t_mfe, t_mae, mfe_first, path_pips, final_abs
    """
    highs  = df_1h['high'].values
    lows   = df_1h['low'].values
    closes = df_1h['close'].values
    index  = df_1h.index
    n      = len(df_1h)
    entry  = closes[pos]
    end    = min(pos + MFE_HORIZON_H + 1, n)
    if end <= pos + 1:
        return None

    w_highs  = highs[pos+1:end]
    w_lows   = lows[pos+1:end]
    w_closes = closes[pos+1:end]

    move_up   = (w_highs.max() - entry) / pip_size
    move_down = (entry - w_lows.min())  / pip_size

    if move_up >= move_down:
        mfe     = move_up
        mae     = move_down
        mfe_dir = 1
        t_mfe   = int(np.argmax(w_highs))
        t_mae   = int(np.argmin(w_lows))
    else:
        mfe     = move_down
        mae     = move_up
        mfe_dir = -1
        t_mfe   = int(np.argmin(w_lows))
        t_mae   = int(np.argmax(w_highs))

    path_pips = mfe_dir * (w_closes - entry) / pip_size
    final_abs = abs(w_closes[-1] - entry) / pip_size

    return {
        'mfe':        mfe,
        'mae':        mae,
        'mfe_dir':    mfe_dir,
        't_mfe':      t_mfe,
        't_mae':      t_mae,
        'mfe_first':  int(t_mfe < t_mae),
        'path':       path_pips,
        'final_abs':  final_abs,
        'entry_price': entry,
        'exit_price':  w_closes[-1],
        'resolve_ts':  index[end - 1],
    }


# ── Main ──────────────────────────────────────────────────────────────────────
async def main():
    print('Loading MFE model...')
    mfe_bundle   = joblib.load('backend/models_9/mfe_q50_8h/model_1H_Q50.joblib')
    mfe_model    = mfe_bundle['model']
    feature_cols = mfe_bundle['feature_cols']
    print(f'  {len(feature_cols)} features  |  cv_pinball={mfe_bundle.get("cv_pinball", 0):.3f}  |  {mfe_bundle["n_iters"]} iters')

    raw, from_date, to_date = await fetch_all_pairs()
    if not raw:
        print('No data fetched.'); return

    test_start = datetime.now() - timedelta(days=TEST_DAYS)
    print(f'\nTest window: {test_start.strftime("%Y-%m-%d")} -> {to_date}')

    close_1h_all = {pair: raw[pair]['1h']['close'] for pair in raw}
    print('\nComputing cross-pair features...')
    cross_features = compute_all_cross_pair_features(close_1h_all)

    print('Computing per-pair features...')
    pair_features = {}
    for pair in PAIRS:
        if pair not in raw:
            continue
        print(f'  {pair}...', end=' ', flush=True)
        df_base = compute_features_for_pair(
            pair, raw[pair]['1m'], raw[pair]['5m'],
            raw[pair]['15m'], df_1h=raw[pair]['1h']
        )
        if df_base.empty:
            print('no features'); continue
        df_extra = compute_momentum_calendar_features(raw[pair]['1h'], PIP_SIZE[pair])
        df_base  = df_base.join(df_extra.reindex(df_base.index), how='left')
        if pair in cross_features:
            df_base = df_base.join(cross_features[pair].reindex(df_base.index), how='left')
        pair_features[pair] = df_base
        print(f'{len(df_base)} rows')

    # ── Score all bars and collect stats ──────────────────────────────────────
    print('\nScoring all bars...')
    rows      = []
    shap_rows = []

    for pair in PAIRS:
        if pair not in pair_features:
            continue
        pip_size = PIP_SIZE[pair]
        df_feat  = pair_features[pair]
        df_1h    = raw[pair]['1h']

        df_test = df_feat[df_feat.index >= test_start].copy()
        if len(df_test) < 10:
            continue

        common_idx = df_test.index.intersection(df_1h.index)
        df_test    = df_test.loc[common_idx]

        for col in feature_cols:
            if col not in df_test.columns:
                df_test[col] = 0.0

        X_all = df_test[feature_cols].ffill().fillna(0)
        df_test = df_test.copy()
        df_test['q50_mfe'] = mfe_model.predict(X_all)

        df_cands       = df_test[df_test['q50_mfe'] >= MFE_THRESH]
        cooldown_until = -1

        for ts, row in df_cands.iterrows():
            try:
                pos = df_1h.index.get_loc(ts)
            except KeyError:
                continue
            if pos <= cooldown_until:
                continue
            cooldown_until = pos + MFE_HORIZON_H

            if pos + MFE_HORIZON_H >= len(df_1h):
                continue  # full 8h window not yet available — skip incomplete signals
            stats = compute_window_stats(df_1h, pos, pip_size)
            if stats is None:
                continue

            h = ts.hour
            if   7  <= h <  9: session = 'London_open'
            elif 9  <= h < 12: session = 'London'
            elif 12 <= h < 14: session = 'LN_NY_overlap'
            elif 14 <= h < 17: session = 'NY'
            elif 17 <= h < 21: session = 'NY_close'
            else:              session = 'Off_hours'

            rows.append({
                'ts':           ts,
                'pair':         pair,
                'predicted':    row['q50_mfe'],
                'mfe':          stats['mfe'],
                'mae':          stats['mae'],
                'mfe_dir':      stats['mfe_dir'],
                't_mfe':        stats['t_mfe'],
                't_mae':        stats['t_mae'],
                'mfe_first':    stats['mfe_first'],
                'final_abs':    stats['final_abs'],
                'beat':         int(stats['mfe'] >= row['q50_mfe']),
                'hour':         h,
                'session':      session,
                'dow':          ts.day_of_week,
                'month':        ts.to_period('M'),
                '_path':        stats['path'],
                'entry_price':  stats['entry_price'],
                'exit_price':   stats['exit_price'],
                'resolve_ts':   stats['resolve_ts'],
            })
            if row['q50_mfe'] >= 50:
                shap_rows.append(X_all.loc[ts])

    if not rows:
        print('No candidates found.'); return

    paths = [r.pop('_path') for r in rows]
    df    = pd.DataFrame(rows).sort_values('ts').reset_index(drop=True)
    print(f'  Total signals: {len(df):,}')

    # ── Output ────────────────────────────────────────────────────────────────
    SEP  = '=' * 72
    SEP2 = '-' * 72

    def pct(x):   return f'{x:.1%}' if not np.isnan(x) else 'n/a'
    def pu(x):    return f'{x:.1f}p' if not np.isnan(x) else 'n/a'
    def fmt(x,d=2): return f'{x:.{d}f}' if not np.isnan(x) else 'n/a'

    print(f'\n{SEP}')
    print(f'  MFE MODEL EVAL (8h)  —  {len(df):,} signals  |  MFE>=30  |  8h cooldown')
    print(f'  Model: mfe_q50_8h  |  Period: {df["ts"].min().date()} to {df["ts"].max().date()}')
    print(f'{SEP}')

    # ── 1. Model calibration ──────────────────────────────────────────────────
    print(f'\n  1. MODEL CALIBRATION')
    print(f'  {"Avg predicted":<35}: {pu(df["predicted"].mean())}  med={pu(df["predicted"].median())}')
    print(f'  {"Avg actual MFE":<35}: {pu(df["mfe"].mean())}  med={pu(df["mfe"].median())}')
    print(f'  {"Avg actual MAE":<35}: {pu(df["mae"].mean())}  med={pu(df["mae"].median())}')
    print(f'  {"% beat prediction":<35}: {pct(df["beat"].mean())}')
    print(f'  {"Avg over-performance":<35}: {pu((df["mfe"] - df["predicted"]).mean())}  (actual - predicted)')
    print(f'  {"Correlation pred/actual MFE":<35}: {fmt(df["predicted"].corr(df["mfe"]), 3)}')

    print(f'\n  Calibration by score bucket:')
    print(f'  {"Bucket":<12} {"N":>6}  {"Pred":>8}  {"Actual MFE":>10}  {"Actual MAE":>10}  {"Beat":>7}  {"R:R":>6}')
    print(f'  {SEP2}')
    for lo, hi in [(30,40),(40,50),(50,60),(60,70),(70,90),(90,999)]:
        sub = df[(df['predicted'] >= lo) & (df['predicted'] < hi)]
        if len(sub) < 5: continue
        rr = (sub['mfe'] / sub['mae'].replace(0, np.nan)).mean()
        print(f'  {lo:>3}-{hi:<4}p   {len(sub):>6}  {pu(sub["predicted"].mean()):>8}  '
              f'{pu(sub["mfe"].mean()):>10}  {pu(sub["mae"].mean()):>10}  '
              f'{pct(sub["beat"].mean()):>7}  {fmt(rr):>6}')

    # ── 2. Price action structure ─────────────────────────────────────────────
    print(f'\n{SEP}')
    print(f'  2. PRICE ACTION IN 8H WINDOW')
    print(f'{SEP}')
    print(f'  {"Avg MFE/MAE ratio":<35}: {fmt(( df["mfe"] / df["mae"].replace(0,np.nan) ).mean())}')
    print(f'  {"MFE before MAE":<35}: {pct(df["mfe_first"].mean())}  (dominant move comes first)')
    print(f'  {"Avg t_mfe":<35}: {fmt(df["t_mfe"].mean(),1)}h  med={fmt(df["t_mfe"].median(),1)}h')
    print(f'  {"Avg t_mae":<35}: {fmt(df["t_mae"].mean(),1)}h  med={fmt(df["t_mae"].median(),1)}h')
    print(f'  {"Up dominant":<35}: {pct((df["mfe_dir"]==1).mean())}  |  Down: {pct((df["mfe_dir"]==-1).mean())}')

    print(f'\n  SL sizing (MAE percentiles):')
    for pv in [25, 50, 75, 90, 95]:
        print(f'    p{pv:<3}: {df["mae"].quantile(pv/100):.1f}p')

    print(f'\n  TP sizing (MFE percentiles):')
    for pv in [25, 50, 75, 90, 95]:
        print(f'    p{pv:<3}: {df["mfe"].quantile(pv/100):.1f}p')

    print(f'\n  Time to MFE distribution:')
    for lo, hi, label in [(0,2,'h0-2'),(2,4,'h2-4'),(4,6,'h4-6'),(6,8,'h6-8')]:
        n = ((df['t_mfe'] >= lo) & (df['t_mfe'] < hi)).sum()
        print(f'    {label}: {n:>5}  ({n/len(df):.1%})')

    print(f'\n  MAE buckets vs actual MFE:')
    for lo, hi, label in [(0,10,'MAE<10p'),(10,20,'MAE 10-20p'),(20,40,'MAE 20-40p'),(40,999,'MAE>40p')]:
        sub = df[(df['mae'] >= lo) & (df['mae'] < hi)]
        if len(sub) < 10: continue
        print(f'    {label:<14}: N={len(sub):>5}  MFE={pu(sub["mfe"].mean())}  beat={pct(sub["beat"].mean())}  t_mfe={fmt(sub["t_mfe"].mean(),1)}h')

    # ── 3. Avg price path ─────────────────────────────────────────────────────
    print(f'\n{SEP}')
    print(f'  3. AVG PRICE PATH (dominant direction, h1-h8)')
    print(f'{SEP}')
    min_len = min(len(p) for p in paths)
    arr     = np.array([p[:min_len] for p in paths])
    avg_p   = arr.mean(axis=0)
    std_p   = arr.std(axis=0)
    print(f'\n  {"Hour":<6}  {"AvgPip":>8}  {"StdDev":>8}  Bar')
    print(f'  {"-"*50}')
    for h in range(min(MFE_HORIZON_H, min_len)):
        v   = avg_p[h]
        bar = '|' * int(abs(v) / 2)
        print(f'  h{h+1:<4}  {v:>+8.1f}  {std_p[h]:>8.1f}  {("+" if v>=0 else "-")}{bar}')

    # ── 4. Session & hour breakdown ───────────────────────────────────────────
    print(f'\n{SEP}')
    print(f'  4. SESSION & TIMING')
    print(f'{SEP}')
    print(f'\n  By session:')
    for sess in ['London_open','London','LN_NY_overlap','NY','NY_close','Off_hours']:
        sub = df[df['session'] == sess]
        if len(sub) < 10: continue
        print(f'    {sess:<18}: N={len(sub):>5}  MFE={pu(sub["mfe"].mean())}  MAE={pu(sub["mae"].mean())}  beat={pct(sub["beat"].mean())}  t_mfe={fmt(sub["t_mfe"].mean(),1)}h')

    print(f'\n  By day of week:')
    for d, name in enumerate(['Mon','Tue','Wed','Thu','Fri']):
        sub = df[df['dow'] == d]
        if len(sub) < 10: continue
        print(f'    {name}: N={len(sub):>5}  MFE={pu(sub["mfe"].mean())}  MAE={pu(sub["mae"].mean())}  beat={pct(sub["beat"].mean())}')

    print(f'\n  By hour:')
    for h in range(24):
        sub = df[df['hour'] == h]
        if len(sub) < 10: continue
        print(f'    h{h:02d}: N={len(sub):>5}  MFE={pu(sub["mfe"].mean())}  MAE={pu(sub["mae"].mean())}  beat={pct(sub["beat"].mean())}')

    # ── 5. Monthly stability ──────────────────────────────────────────────────
    print(f'\n{SEP}')
    print(f'  5. MONTHLY STABILITY')
    print(f'{SEP}')
    print(f'\n  {"Month":<10}  {"N":>5}  {"Pred":>8}  {"MFE":>8}  {"MAE":>8}  {"Beat":>7}  {"R:R":>6}')
    print(f'  {"-"*60}')
    for month, sub in df.groupby('month'):
        rr = (sub['mfe'] / sub['mae'].replace(0, np.nan)).mean()
        print(f'  {str(month):<10}  {len(sub):>5}  {pu(sub["predicted"].mean()):>8}  '
              f'{pu(sub["mfe"].mean()):>8}  {pu(sub["mae"].mean()):>8}  '
              f'{pct(sub["beat"].mean()):>7}  {fmt(rr):>6}')

    # ── 6. Per-pair anatomy ───────────────────────────────────────────────────
    print(f'\n{SEP}')
    print(f'  6. PER-PAIR ANATOMY')
    print(f'{SEP}')

    pair_paths = {}
    for i, r in enumerate(rows):
        p = r['pair']
        if p not in pair_paths:
            pair_paths[p] = []
        pair_paths[p].append(paths[i])

    for pair in sorted(df['pair'].unique()):
        sub = df[df['pair'] == pair]
        if len(sub) < 10: continue
        rr = (sub['mfe'] / sub['mae'].replace(0, np.nan)).mean()
        print(f'\n  {pair}  N={len(sub)}  beat={pct(sub["beat"].mean())}')
        print(f'    MFE={pu(sub["mfe"].mean())} (med {pu(sub["mfe"].median())})  '
              f'MAE={pu(sub["mae"].mean())} (med {pu(sub["mae"].median())})  '
              f'R:R={fmt(rr)}  t_mfe={fmt(sub["t_mfe"].mean(),1)}h  MFE1st={pct(sub["mfe_first"].mean())}')
        print(f'    SL: p25={sub["mae"].quantile(0.25):.1f}p  p50={sub["mae"].quantile(0.5):.1f}p  '
              f'p75={sub["mae"].quantile(0.75):.1f}p  p90={sub["mae"].quantile(0.9):.1f}p')
        print(f'    TP: p25={sub["mfe"].quantile(0.25):.1f}p  p50={sub["mfe"].quantile(0.5):.1f}p  '
              f'p75={sub["mfe"].quantile(0.75):.1f}p  p90={sub["mfe"].quantile(0.9):.1f}p')
        pp = pair_paths.get(pair, [])
        if pp:
            ml   = min(len(x) for x in pp)
            avg  = np.array([x[:ml] for x in pp]).mean(axis=0)
            path_str = '  '.join(f'h{i+1}:{avg[i]:>+.0f}p' for i in range(ml))
            print(f'    Path: {path_str}')

    # ── 7. SHAP analysis ──────────────────────────────────────────────────────
    if not shap_rows:
        print('\nNo bars for SHAP.'); return

    import shap as shap_lib
    X_shap = pd.DataFrame(shap_rows, columns=feature_cols)
    print(f'\n{SEP}')
    print(f'  7. SHAP ANALYSIS  (predicted >= 50p, N={len(X_shap)})')
    print(f'{SEP}')
    print('  Computing SHAP values...')
    explainer   = shap_lib.TreeExplainer(mfe_model)
    shap_values = explainer.shap_values(X_shap)
    shap_df     = pd.DataFrame(shap_values, columns=feature_cols)
    mean_abs    = shap_df.abs().mean().sort_values(ascending=False)

    print(f'\n  Top 30 features by mean |SHAP|:')
    print(f'  {"Feature":<45} {"Mean|SHAP|":>10}  {"Mean SHAP":>10}  {"% pos":>7}')
    print(f'  {"-"*76}')
    for feat in mean_abs.head(30).index:
        vals = shap_df[feat]
        print(f'  {feat:<45} {vals.abs().mean():>+10.3f}  {vals.mean():>+10.3f}  {(vals>0).mean():>6.1%}')

    print(f'\n  Feature category breakdown:')
    categories = {
        'rv/microstructure': [f for f in feature_cols if any(x in f for x in ['rv','bv','jump','kyle','amihud','vpin','entropy','hurst','fractal','vr_'])],
        'volume':   [f for f in feature_cols if 'vol' in f and 'vpin' not in f],
        'momentum': [f for f in feature_cols if any(x in f for x in ['ret_','rsi','range_pos','dist_','accel','momentum'])],
        'corr':     [f for f in feature_cols if 'corr' in f],
        'beta':     [f for f in feature_cols if 'beta' in f],
        'csi':      [f for f in feature_cols if 'csi' in f],
        'relstr':   [f for f in feature_cols if 'relstr' in f],
        'peer':     [f for f in feature_cols if 'peer' in f],
        'calendar': [f for f in feature_cols if any(x in f for x in ['hour','dow','sin','cos','is_','days_','month','quarter'])],
        'candle':   [f for f in feature_cols if any(x in f for x in ['body','wick','candle','consec'])],
    }
    covered = set()
    cat_scores = {}
    for cat, feats in categories.items():
        fp = [f for f in feats if f in mean_abs.index]
        if fp:
            cat_scores[cat] = (mean_abs[fp].sum(), len(fp))
            covered.update(fp)
    other = [f for f in mean_abs.index if f not in covered]
    if other:
        cat_scores['other'] = (mean_abs[other].sum(), len(other))
    total = sum(v[0] for v in cat_scores.values())
    print(f'  {"Category":<22} {"Sum|SHAP|":>10}  {"% total":>8}  {"N_feats":>8}')
    print(f'  {"-"*52}')
    for cat, (score, n) in sorted(cat_scores.items(), key=lambda x: -x[1][0]):
        print(f'  {cat:<22} {score:>10.3f}  {score/total:>8.1%}  {n:>8}')

    # ── 8. Signal log — every signal, one line each ───────────────────────────
    print(f'\n{SEP}')
    print(f'  8. SIGNAL LOG  (every signal, chronological)')
    print(f'{SEP}')
    print(f'\n  {"#":>4}  {"Pair":<8}  {"Fired (UTC)":<18}  {"Resolves (UTC)":<18}  {"Entry":>10}  {"Exit":>10}  {"Pred":>6}  {"MFE":>7}  {"MAE":>7}  {"Beat":>5}  {"Dir":>5}  {"t_MFE":>6}  {"t_MAE":>6}')
    print(f'  {"-"*120}')
    for i, (_, row) in enumerate(df.sort_values('ts').iterrows()):
        fired    = row['ts'].strftime('%Y-%m-%d %H:%M')
        resolves = row['resolve_ts'].strftime('%Y-%m-%d %H:%M')
        beat_str = 'YES' if row['beat'] else 'no'
        dir_str  = 'UP' if row['mfe_dir'] == 1 else 'DN'
        print(f'  {i+1:>4}  {row["pair"]:<8}  {fired:<18}  {resolves:<18}  '
              f'{row["entry_price"]:>10.5f}  {row["exit_price"]:>10.5f}  '
              f'{row["predicted"]:>5.1f}p  {row["mfe"]:>6.1f}p  {row["mae"]:>6.1f}p  '
              f'{beat_str:>5}  {dir_str:>5}  h{row["t_mfe"]:>4}  h{row["t_mae"]:>4}')

    print(f'\n{SEP}')
    print(f'  DONE')
    print(f'{SEP}')


asyncio.run(main())
