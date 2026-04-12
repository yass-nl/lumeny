"""
Live Signal Scan — Last 10 Days
=================================
Replicates the exact MFE>=50 + RELAXED directional system on recent data
fetched live from Polygon, so you can inspect every signal on a chart.

Output:
  - Console table: each signal with pair, timestamp (UTC), direction, MFE score,
    key feature values, and (where available) realized 72h move
  - CSV: notebooks_11/07_live_signals.csv  (for charting)

Pipeline:
  1. Fetch last FETCH_DAYS of 1M OHLCV from Polygon (needs warm-up for features)
  2. Resample to 1H, compute microstructure + cross-pair features
  3. Run MFE Q50 model — keep score >= 50
  4. Filter hours 7-20 UTC
  5. Apply RELAXED pair-specific direction rules
  6. Apply 72h cooldown per pair
  7. Compute realized fwd_72h where window is closed (>= 72h ago)
  8. Print + save
"""

import os, asyncio, sys
import pandas as pd
import numpy as np
import joblib
import httpx
from datetime import datetime, timedelta
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
API_KEY   = os.getenv('POLYGON_S3_SECRET_KEY', '')
REST_BASE = 'https://api.polygon.io'

sys.path.insert(0, str(Path(__file__).parent.parent / 'backend'))
sys.path.insert(0, str(Path(__file__).parent.parent))
from features import compute_features_for_pair, PIP_SIZE
from live_features_extra import compute_momentum_calendar_features

SCRIPT_DIR = Path(__file__).parent
OUT_CSV    = SCRIPT_DIR / '07_live_signals.csv'

# ── Config ────────────────────────────────────────────────────────────────────
FETCH_DAYS    = 60     # warm-up + scan window (features need ~200 bars look-back)
SCAN_DAYS     = 10     # only report signals in last N days
MFE_THRESH    = 50.0
HOURS_ALLOWED = set(range(7, 21))   # 7-20 UTC inclusive
COOLDOWN_H    = 72
FWD_H         = 72

JPY_PAIRS = {'USDJPY', 'EURJPY', 'GBPJPY', 'AUDJPY', 'CADJPY', 'CHFJPY'}

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
    df = df.rename(columns={'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume'})
    df = df.set_index('datetime')[['open', 'high', 'low', 'close', 'volume']]
    df = df.sort_index().drop_duplicates()
    df = df[~((df.index.dayofweek == 5) | ((df.index.dayofweek == 6) & (df.index.hour < 21)))]
    return df


async def fetch_all_pairs():
    now       = datetime.utcnow()
    to_date   = now.strftime('%Y-%m-%d')
    from_date = (now - timedelta(days=FETCH_DAYS)).strftime('%Y-%m-%d')
    print(f'Fetching {from_date} -> {to_date}  ({FETCH_DAYS} days warm-up)')
    raw = {}
    async with httpx.AsyncClient(timeout=120) as client:
        for pair in PAIRS:
            print(f'  {pair}...', end=' ', flush=True)
            df_1m = await fetch_bars(client, pair, from_date, to_date)
            await asyncio.sleep(0.3)
            if df_1m.empty:
                print('NO DATA'); continue
            df_5m  = df_1m.resample('5min').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}).dropna()
            df_15m = df_1m.resample('15min').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}).dropna()
            df_1h  = df_1m.resample('1h').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}).dropna()
            raw[pair] = {'1m': df_1m, '5m': df_5m, '15m': df_15m, '1h': df_1h}
            print(f'{len(df_1m):,} 1m / {len(df_1h)} 1h bars')
    return raw


# ── Cross-pair features ───────────────────────────────────────────────────────
def compute_cross_pair_features(close_1h_all):
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
                cols[f'corr_{sl}_{lbl}'] = r.rolling(w, min_periods=w // 2).corr(p_ret)
            cols[f'corr_regime_{sl}'] = cols[f'corr_{sl}_24h'] - cols[f'corr_{sl}_1w']
            for w, lbl in [(24, '24h'), (168, '1w')]:
                cov = r.rolling(w, min_periods=w // 2).cov(p_ret)
                var = p_ret.rolling(w, min_periods=w // 2).var().clip(lower=1e-12)
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


# ── Direction rules (RELAXED version) ────────────────────────────────────────
def apply_direction_rules(df):
    """Returns pd.Series: +1 LONG, -1 SHORT, NaN skip."""
    dirs = pd.Series(np.nan, index=df.index)
    pair = df['pair']

    def col(name):
        return df.get(name, pd.Series(np.nan, index=df.index))

    # USDJPY -> always SHORT
    dirs = dirs.where(pair != 'USDJPY', -1.0)

    # AUDUSD -> LONG if beta_gbpusd_1w > 0.775 OR atr_24 < 40.8
    m = pair == 'AUDUSD'
    lc = m & (col('beta_gbpusd_1w').gt(0.775) | col('atr_24').lt(40.8))
    dirs = dirs.where(~lc, 1.0).where(~(m & ~lc), np.nan)

    # GBPUSD -> LONG if csi_usd_24h < 0.004
    m = pair == 'GBPUSD'
    lc = m & col('csi_usd_24h').lt(0.004)
    dirs = dirs.where(~lc, 1.0).where(~(m & ~lc), np.nan)

    # EURUSD -> LONG if corr_audusd_24h < 0.22
    m = pair == 'EURUSD'
    lc = m & col('corr_audusd_24h').lt(0.22)
    dirs = dirs.where(~lc, 1.0).where(~(m & ~lc), np.nan)

    # NZDUSD -> LONG if dist_5d_high > 0.35
    m = pair == 'NZDUSD'
    lc = m & col('dist_5d_high').gt(0.35)
    dirs = dirs.where(~lc, 1.0).where(~(m & ~lc), np.nan)

    # USDCHF -> LONG if corr_eurusd_1w > -0.60
    m = pair == 'USDCHF'
    lc = m & col('corr_eurusd_1w').gt(-0.60)
    dirs = dirs.where(~lc, 1.0).where(~(m & ~lc), np.nan)

    # CHFJPY -> LONG if corr_usdjpy_1w > 0.40, SHORT if < 0.26, else NaN
    m  = pair == 'CHFJPY'
    cv = col('corr_usdjpy_1w')
    lc = m & cv.gt(0.40)
    sc = m & cv.lt(0.26)
    dirs = dirs.where(~lc, 1.0).where(~sc, -1.0).where(~(m & ~lc & ~sc), np.nan)

    # CADJPY -> LONG if vol_trend < 1.15, SHORT if >= 1.15
    m  = pair == 'CADJPY'
    vt = col('vol_trend')
    lc = m & vt.lt(1.15)
    sc = m & vt.ge(1.15)
    dirs = dirs.where(~lc, 1.0).where(~sc, -1.0)

    # AUDJPY -> LONG if beta_usdjpy_1w > 0.74
    m  = pair == 'AUDJPY'
    lc = m & col('beta_usdjpy_1w').gt(0.74)
    dirs = dirs.where(~lc, 1.0).where(~(m & ~lc), np.nan)

    # EURJPY -> LONG if beta_eurusd_1w > 0.38
    m  = pair == 'EURJPY'
    lc = m & col('beta_eurusd_1w').gt(0.38)
    dirs = dirs.where(~lc, 1.0).where(~(m & ~lc), np.nan)

    # GBPJPY -> LONG if beta_eurusd_1w > 0.50
    m  = pair == 'GBPJPY'
    lc = m & col('beta_eurusd_1w').gt(0.50)
    dirs = dirs.where(~lc, 1.0).where(~(m & ~lc), np.nan)

    # EURAUD -> LONG if corr_audusd_24h < 0.22
    m  = pair == 'EURAUD'
    lc = m & col('corr_audusd_24h').lt(0.22)
    dirs = dirs.where(~lc, 1.0).where(~(m & ~lc), np.nan)

    # AUDNZD -> LONG if corr_regime_audusd > 0.0
    m  = pair == 'AUDNZD'
    lc = m & col('corr_regime_audusd').gt(0.0)
    dirs = dirs.where(~lc, 1.0).where(~(m & ~lc), np.nan)

    # EURGBP -> SHORT if csi_usd_24h > 0.004
    m  = pair == 'EURGBP'
    sc = m & col('csi_usd_24h').gt(0.004)
    dirs = dirs.where(~sc, -1.0).where(~(m & ~sc), np.nan)

    # USDCAD -> no rule
    dirs = dirs.where(pair != 'USDCAD', np.nan)

    return dirs


# ── Main ──────────────────────────────────────────────────────────────────────
async def main():
    # Load MFE model
    print('\nLoading MFE model...')
    mfe_path   = Path(__file__).parent.parent / 'backend/models_9/mfe_q50/model_1H_Q50.joblib'
    bundle     = joblib.load(mfe_path)
    mfe_model  = bundle['model']
    feat_cols  = bundle['feature_cols']
    print(f'  {len(feat_cols)} features, {bundle["n_iters"]} iters')

    # Fetch from Polygon
    print()
    raw = await fetch_all_pairs()
    if not raw:
        print('No data fetched.'); return

    # Cross-pair features
    print('\nComputing cross-pair features...')
    close_1h_all   = {pair: raw[pair]['1h']['close'] for pair in raw}
    cross_features = compute_cross_pair_features(close_1h_all)

    # Per-pair microstructure features + combine
    print('Computing per-pair microstructure features...')
    frames = []
    for pair in PAIRS:
        if pair not in raw:
            continue
        d = raw[pair]
        try:
            feat = compute_features_for_pair(
                pair,
                d['1m'], d['5m'], d['15m'], d['1h'],
            )
        except Exception as e:
            print(f'  {pair} features error: {e}'); continue

        if pair in cross_features:
            cp = cross_features[pair].reindex(feat.index)
            feat = pd.concat([feat, cp], axis=1)

        # Extra momentum/calendar features (vol_trend, dist_5d_high, ret_1d, etc.)
        pip = 0.01 if pair in JPY_PAIRS else 0.0001
        extra = compute_momentum_calendar_features(d['1h'], pip)
        feat = pd.concat([feat, extra.reindex(feat.index)], axis=1)

        feat['pair'] = pair
        frames.append(feat)

    if not frames:
        print('No features computed.'); return

    df = pd.concat(frames).sort_index()
    print(f'  Combined: {len(df):,} rows across {len(frames)} pairs')

    # MFE model score
    print('\nScoring with MFE model...')
    missing = [c for c in feat_cols if c not in df.columns]
    if missing:
        print(f'  WARNING: {len(missing)} missing feature cols — filling with 0: {missing}')
        for c in missing:
            df[c] = 0.0
    X         = df[feat_cols].ffill().fillna(0)
    df['q50'] = mfe_model.predict(X)

    # Filter: MFE>=50 + hours 7-20
    df['hour'] = pd.to_datetime(df.index).hour
    df_filt    = df[(df['q50'] >= MFE_THRESH) & df['hour'].isin(HOURS_ALLOWED)].copy()
    print(f'  MFE>=50 + hours 7-20: {len(df_filt):,} bars')

    # Direction rules
    df_filt['direction'] = apply_direction_rules(df_filt)
    df_cands = df_filt[df_filt['direction'].notna()].sort_index().copy()
    print(f'  With direction call: {len(df_cands):,} bars')

    # 72h cooldown — track (pair, integer_pos) to avoid duplicate-index issues
    df_cands = df_cands.reset_index()
    ts_col   = df_cands.columns[0]   # the datetime column after reset_index
    cooldown_until = {}
    kept_idx = []
    for i, row in df_cands.iterrows():
        ts   = row[ts_col]
        pair = row['pair']
        if pair in cooldown_until and ts < cooldown_until[pair]:
            continue
        cooldown_until[pair] = ts + pd.Timedelta(hours=COOLDOWN_H)
        kept_idx.append(i)

    df_sig = df_cands.iloc[kept_idx].set_index(ts_col).copy()
    df_sig.index.name = None
    print(f'  After 72h cooldown: {len(df_sig):,} signals total')

    # Restrict output to SCAN_DAYS window
    now        = datetime.utcnow()
    scan_start = now - timedelta(days=SCAN_DAYS)
    df_out     = df_sig[df_sig.index >= scan_start].copy()
    print(f'  In last {SCAN_DAYS} days: {len(df_out):,} signals')

    if df_out.empty:
        print('  No signals in window.'); return

    # ── Enrich each signal ────────────────────────────────────────────────────
    # Which feature drove the direction decision for each pair
    DIR_FEATURE = {
        'USDJPY':  ('always SHORT',           None,              None),
        'AUDUSD':  ('beta_gbpusd_1w>0.775 OR atr_24<40.8', 'beta_gbpusd_1w', 'atr_24'),
        'GBPUSD':  ('csi_usd_24h<0.004',      'csi_usd_24h',     None),
        'EURUSD':  ('corr_audusd_24h<0.22',   'corr_audusd_24h', None),
        'NZDUSD':  ('dist_5d_high>0.35',      'dist_5d_high',    None),
        'USDCHF':  ('corr_eurusd_1w>-0.60',   'corr_eurusd_1w',  None),
        'CHFJPY':  ('corr_usdjpy_1w>0.40/<0.26', 'corr_usdjpy_1w', None),
        'CADJPY':  ('vol_trend</>1.15',        'vol_trend',       None),
        'AUDJPY':  ('beta_usdjpy_1w>0.74',    'beta_usdjpy_1w',  None),
        'EURJPY':  ('beta_eurusd_1w>0.38',    'beta_eurusd_1w',  None),
        'GBPJPY':  ('beta_eurusd_1w>0.50',    'beta_eurusd_1w',  None),
        'EURAUD':  ('corr_audusd_24h<0.22',   'corr_audusd_24h', None),
        'AUDNZD':  ('corr_regime_audusd>0',   'corr_regime_audusd', None),
        'EURGBP':  ('csi_usd_24h>0.004',      'csi_usd_24h',     None),
        'USDCAD':  ('no rule',                 None,              None),
    }

    records = []
    cutoff  = now - timedelta(hours=FWD_H)

    for ts, row in df_out.sort_index().iterrows():
        pair    = row['pair']
        pip     = 0.01 if pair in JPY_PAIRS else 0.0001
        dir_val = int(row['direction'])
        dir_lbl = 'LONG' if dir_val == 1 else 'SHORT'
        q50     = row['q50']

        close_series = raw[pair]['1h']['close']
        pos_start    = close_series.index.searchsorted(ts)

        entry_price  = close_series.iloc[pos_start] if pos_start < len(close_series) else np.nan
        # window_close = actual timestamp of the 72nd trading bar (skips weekends)
        pos_end_disp = pos_start + FWD_H
        if pos_end_disp < len(close_series):
            window_close = close_series.index[pos_end_disp]
        else:
            window_close = ts + timedelta(hours=FWD_H)  # fallback if data incomplete

        # Realized outcome
        fwd_pips  = np.nan
        exit_price = np.nan
        result    = 'OPEN'
        in_dir_pips = np.nan

        if ts <= cutoff:
            pos_end = pos_start + FWD_H
            if pos_end < len(close_series):
                exit_price  = close_series.iloc[pos_end]
                raw_move    = (exit_price - entry_price) / pip
                fwd_pips    = round(raw_move, 1)
                in_dir_pips = round(dir_val * raw_move, 1)
                result      = 'WIN' if in_dir_pips > 0 else 'LOSE'

        # Max favorable / max adverse within closed window
        mfe_pips = np.nan
        mae_pips = np.nan
        if ts <= cutoff and pos_start < len(close_series):
            window_highs = raw[pair]['1h']['high'].iloc[pos_start:pos_start + FWD_H + 1]
            window_lows  = raw[pair]['1h']['low'].iloc[pos_start:pos_start + FWD_H + 1]
            if len(window_highs) > 1 and not np.isnan(entry_price):
                if dir_val == 1:
                    mfe_pips = round((window_highs.max() - entry_price) / pip, 1)
                    mae_pips = round((entry_price - window_lows.min()) / pip, 1)
                else:
                    mfe_pips = round((entry_price - window_lows.min()) / pip, 1)
                    mae_pips = round((window_highs.max() - entry_price) / pip, 1)

        # Key driver feature value
        rule_desc, feat1, feat2 = DIR_FEATURE.get(pair, ('unknown', None, None))
        feat1_val = round(float(row[feat1]), 4) if feat1 and feat1 in row.index and pd.notna(row[feat1]) else None
        feat2_val = round(float(row[feat2]), 4) if feat2 and feat2 in row.index and pd.notna(row[feat2]) else None
        feat_str  = f'{feat1}={feat1_val}' if feat1_val is not None else rule_desc
        if feat2_val is not None:
            feat_str += f'  {feat2}={feat2_val}'

        records.append({
            'signal_time_utc':  str(ts)[:16],
            'window_close_utc': str(window_close)[:16],
            'pair':             pair,
            'direction':        dir_lbl,
            'mfe_q50_score':    round(q50, 1),
            'entry_price':      round(entry_price, 5) if not np.isnan(entry_price) else '',
            'exit_price_72h':   round(exit_price, 5) if not np.isnan(exit_price) else '',
            'fwd_72h_raw_pips': fwd_pips,
            'in_dir_pips':      in_dir_pips,
            'mfe_pips':         mfe_pips,
            'mae_pips':         mae_pips,
            'result':           result,
            'dir_rule':         rule_desc,
            'key_feature':      feat_str,
        })

    df_csv = pd.DataFrame(records)

    # ── Console summary ───────────────────────────────────────────────────────
    W = 110
    print(f'\n{"="*W}')
    print(f'  LIVE SIGNAL SCAN  |  Last {SCAN_DAYS} days  |  MFE>=50 + Direction + 72h cooldown')
    print(f'  Generated: {now.strftime("%Y-%m-%d %H:%M")} UTC')
    print(f'{"="*W}')
    hdr = f'  {"Signal (UTC)":<17} {"Close (UTC)":<17} {"Pair":<8} {"Dir":<6} {"Score":>5}  {"Entry":>9} {"Exit72h":>9} {"RawPips":>8} {"InDir":>7} {"MFE":>7} {"MAE":>7}  {"Result":<5}  Key feature'
    print(hdr)
    print(f'  {"-"*(W-2)}')

    for r in records:
        ep   = f'{r["entry_price"]}' if r['entry_price'] != '' else '    -    '
        xp   = f'{r["exit_price_72h"]}' if r['exit_price_72h'] != '' else '    -    '
        raw_ = f'{r["fwd_72h_raw_pips"]:+.1f}' if r['fwd_72h_raw_pips'] == r['fwd_72h_raw_pips'] else '   -  '
        ind  = f'{r["in_dir_pips"]:+.1f}' if r['in_dir_pips'] == r['in_dir_pips'] else '   -  '
        mfe_ = f'{r["mfe_pips"]:.1f}' if r['mfe_pips'] == r['mfe_pips'] else '  -  '
        mae_ = f'{r["mae_pips"]:.1f}' if r['mae_pips'] == r['mae_pips'] else '  -  '
        res  = r['result']
        kf   = r['key_feature'][:40]
        print(f'  {r["signal_time_utc"]:<17} {r["window_close_utc"]:<17} {r["pair"]:<8} {r["direction"]:<6} {r["mfe_q50_score"]:>5}  {ep:>9} {xp:>9} {raw_:>8} {ind:>7} {mfe_:>7} {mae_:>7}  {res:<5}  {kf}')

    closed = df_csv[df_csv['result'].isin(['WIN', 'LOSE'])]
    open_  = df_csv[df_csv['result'] == 'OPEN']
    print(f'\n  Total: {len(df_csv)}  |  Closed: {len(closed)}  |  Open: {len(open_)}')
    if len(closed) > 0:
        wins = (closed['result'] == 'WIN').sum()
        acc  = wins / len(closed)
        avg  = closed['in_dir_pips'].mean()
        print(f'  Closed accuracy: {acc:.1%}  ({wins}/{len(closed)})  |  avg in-dir: {avg:+.1f}p')
        print(f'  avg MFE: {closed["mfe_pips"].mean():.1f}p  |  avg MAE: {closed["mae_pips"].mean():.1f}p')

    # ── Save CSV ──────────────────────────────────────────────────────────────
    df_csv.to_csv(OUT_CSV, index=False)
    print(f'\n  Saved: {OUT_CSV}')
    print(f'{"="*W}')


if __name__ == '__main__':
    asyncio.run(main())
