"""
Episode Low Backtest — LumenY 9 Long Model
==========================================
Mirrors test_capital_sim_3.py structure exactly:
  1. Fetch fresh data from Polygon (last ~6 months, 1-minute bars)
  2. Resample to 15M / 1H / 4H
  3. Compute features_10 on-the-fly (geometric features_8 + MA context)
  4. Run episode low model (LightGBM binary classifier)
  5. Simulate capital with ATR-based lot sizing, spreads, slippage
     - Episode tracker: one trade per below-both-MAs episode
     - Entry: close of first bar with proba >= threshold
     - Exit: first close above MA200, or timeout at 48H

No lookahead: features use only past bars. MA50/MA200 computed from rolling close.
"""

import os
import asyncio
import time
import pandas as pd
import numpy as np
import joblib
import httpx
import warnings
from pathlib import Path
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')

from dotenv import load_dotenv
load_dotenv()

API_KEY   = os.getenv('POLYGON_S3_SECRET_KEY', '')
REST_BASE = 'https://api.polygon.io'

PAIRS = [
    'EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD', 'USDCAD', 'NZDUSD',
    'EURJPY', 'GBPJPY', 'EURGBP', 'EURAUD', 'AUDJPY', 'CADJPY', 'CHFJPY', 'AUDNZD',
]

PIP_SIZE = {
    'EURUSD':0.0001,'GBPUSD':0.0001,'USDCHF':0.0001,'AUDUSD':0.0001,
    'USDCAD':0.0001,'NZDUSD':0.0001,'EURGBP':0.0001,'EURAUD':0.0001,'AUDNZD':0.0001,
    'USDJPY':0.01,'EURJPY':0.01,'GBPJPY':0.01,'AUDJPY':0.01,'CADJPY':0.01,'CHFJPY':0.01,
}

JPY_PAIRS = {'USDJPY','EURJPY','GBPJPY','AUDJPY','CADJPY','CHFJPY'}

# Realistic spreads in POINTS (1 point = 0.00001 for non-JPY, 0.001 for JPY)
SPREAD_POINTS = {
    'EURUSD': 6, 'GBPUSD': 8, 'USDJPY': 10, 'USDCHF': 7,
    'AUDUSD': 6, 'USDCAD': 12, 'NZDUSD': 9,
    'EURJPY': 14, 'GBPJPY': 21, 'EURGBP': 7, 'EURAUD': 21,
    'AUDJPY': 15, 'CADJPY': 16, 'CHFJPY': 25, 'AUDNZD': 20,
}

OFFHOUR_SPREAD_MULT = {
    'EURUSD':1.1,'GBPUSD':1.1,'USDJPY':1.1,'USDCHF':1.2,
    'AUDUSD':1.0,'USDCAD':0.9,'NZDUSD':1.0,
    'EURJPY':1.2,'GBPJPY':1.2,'EURGBP':1.1,'EURAUD':1.2,
    'AUDJPY':1.1,'CADJPY':1.2,'CHFJPY':1.2,'AUDNZD':1.3,
}

SLIPPAGE_BASE_POINTS = {
    'EURUSD':3.0,'GBPUSD':4.0,'USDJPY':5.0,'USDCHF':5.0,
    'AUDUSD':3.0,'USDCAD':3.0,'NZDUSD':3.0,
    'EURGBP':3.0,'AUDNZD':6.0,'AUDJPY':4.0,'CADJPY':6.0,
    'CHFJPY':9.0,'EURAUD':6.0,'EURJPY':6.0,'GBPJPY':7.0,
}

SLIPPAGE_TIME_MULT = {
    'EURUSD':1.5,'GBPUSD':2.0,'USDJPY':2.0,'USDCHF':2.0,
    'AUDUSD':2.0,'USDCAD':2.0,'NZDUSD':2.0,
    'EURJPY':2.5,'GBPJPY':2.5,'EURGBP':2.0,'EURAUD':2.5,
    'AUDJPY':2.5,'CADJPY':2.5,'CHFJPY':3.0,'AUDNZD':2.5,
}

BASE_USD = {
    'EUR':1.15,'GBP':1.34,'AUD':0.70,'NZD':0.58,
    'USD':1.00,'CAD':0.73,'CHF':0.79,'JPY':0.0063,
}

QUOTE_USD = {
    'USD':1.00,'CHF':1.27,'CAD':0.73,'GBP':1.34,
    'AUD':0.70,'NZD':0.58,'JPY':0.0063,
}

MODELS_DIR      = Path('backend/models_10/ma_cross')
LOT_UNITS       = 100_000
LEVERAGE        = 50
STARTING_CAP    = 100_000.0
RISK_PCT        = 0.02          # 2% equity per trade
PROBA_THRESHOLD = 0.2
TIMEOUT_BARS    = 48            # hours
ATR_STOP_MULT   = 2.0           # stop loss at 2× ATR against entry
BACKTEST_DAYS   = 200           # ~6.5 months
WARMUP_DAYS     = 250           # extra bars for MA200 (200 bars) + warmup
TOTAL_FETCH_DAYS = BACKTEST_DAYS + WARMUP_DAYS


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS: spread / slippage / P&L
# ─────────────────────────────────────────────────────────────────────────────

def spread_in_price(pair, hour_utc=12):
    pts = SPREAD_POINTS.get(pair, 6)
    if hour_utc in (21, 22):
        pts = int(pts * OFFHOUR_SPREAD_MULT.get(pair, 2.0))
    if pair in JPY_PAIRS:
        return pts * 0.001
    return pts * 0.00001

def slippage_in_price(pair, hour_utc=12, lots=1.0):
    base = SLIPPAGE_BASE_POINTS.get(pair, 1.0)
    if hour_utc in (21, 22):
        base *= SLIPPAGE_TIME_MULT.get(pair, 2.0)
    size_mult = min(1.0 + 0.05 * max(lots - 1.0, 0), 3.0)
    pts = base * size_mult
    if pair in JPY_PAIRS:
        return pts * 0.001
    return pts * 0.00001

def compute_pnl_usd(pair, direction, lots, entry, exit_price):
    move = (exit_price - entry) * direction
    quote = pair[3:]
    if pair in JPY_PAIRS:
        return lots * LOT_UNITS * move / exit_price
    elif quote == 'USD':
        return lots * LOT_UNITS * move
    else:
        return lots * LOT_UNITS * move * QUOTE_USD.get(quote, 1.0)

def compute_spread_cost_usd(pair, lots, exit_price, hour_utc=12):
    sp = spread_in_price(pair, hour_utc)
    quote = pair[3:]
    if pair in JPY_PAIRS:
        return lots * LOT_UNITS * sp / exit_price
    elif quote == 'USD':
        return lots * LOT_UNITS * sp
    else:
        return lots * LOT_UNITS * sp * QUOTE_USD.get(quote, 1.0)

def compute_slippage_cost_usd(pair, lots, exit_price, hour_utc=12):
    sl = slippage_in_price(pair, hour_utc, lots)
    quote = pair[3:]
    if pair in JPY_PAIRS:
        return lots * LOT_UNITS * sl / exit_price
    elif quote == 'USD':
        return lots * LOT_UNITS * sl
    else:
        return lots * LOT_UNITS * sl * QUOTE_USD.get(quote, 1.0)

def margin_required(pair, lots):
    base_ccy = pair[:3]
    base_val = BASE_USD.get(base_ccy, 1.0)
    return lots * LOT_UNITS * base_val / LEVERAGE

def lot_size(equity, pair, atr_price, entry_price=1.0):
    """Size lots so 1 ATR move = RISK_PCT of equity. atr_price in price units."""
    quote = pair[3:]
    if pair in JPY_PAIRS:
        # P&L per lot per price unit = LOT_UNITS / price (JPY-denominated, convert to USD)
        val_per_lot = LOT_UNITS * atr_price / entry_price
    elif quote == 'USD':
        val_per_lot = LOT_UNITS * atr_price
    else:
        val_per_lot = LOT_UNITS * atr_price * QUOTE_USD.get(quote, 1.0)
    if val_per_lot < 1e-6:
        return 0.01
    lots = (equity * RISK_PCT) / val_per_lot
    return max(0.01, min(round(lots, 2), 10.0))


# ─────────────────────────────────────────────────────────────────────────────
# DATA FETCHING (same as sim_3)
# ─────────────────────────────────────────────────────────────────────────────

async def fetch_bars(pair, multiplier, timespan, from_date, to_date, limit=50000):
    ticker = f'C:{pair}'
    url = f'{REST_BASE}/v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{from_date}/{to_date}'
    params = {'apiKey': API_KEY, 'limit': limit, 'sort': 'asc'}
    all_results = []

    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.get(url, params=params)
        resp.raise_for_status()
        data = resp.json()
        all_results.extend(data.get('results', []))

        while 'next_url' in data:
            next_url = data['next_url']
            sep = '&' if '?' in next_url else '?'
            resp = await client.get(f'{next_url}{sep}apiKey={API_KEY}')
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
    now       = datetime.utcnow()
    to_date   = now.strftime('%Y-%m-%d')
    from_date = (now - timedelta(days=TOTAL_FETCH_DAYS)).strftime('%Y-%m-%d')
    print(f'\nFetching {TOTAL_FETCH_DAYS} days of data ({from_date} to {to_date})...')

    data = {}
    for pair in PAIRS:
        print(f'  {pair}...', end=' ', flush=True)
        t0 = time.time()
        df_1m = await fetch_bars(pair, 1, 'minute', from_date, to_date)
        elapsed = time.time() - t0

        if df_1m.empty:
            print('NO DATA')
            continue

        df_15m = df_1m.resample('15min').agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum'}).dropna()
        df_1h  = df_1m.resample('1h').agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum'}).dropna()
        df_4h  = df_1m.resample('4h').agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum'}).dropna()

        data[pair] = {'1m':df_1m,'15m':df_15m,'1h':df_1h,'4h':df_4h}
        print(f'{len(df_1m):,} 1m bars | {len(df_1h):,} 1h bars ({elapsed:.1f}s)')
        await asyncio.sleep(0.3)

    return data


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE COMPUTATION (features_8 geometric + MA context = features_10)
# ─────────────────────────────────────────────────────────────────────────────

def bar_shape_features(df_1h):
    """Layer 1: Single bar shape features."""
    o = df_1h['open']; h = df_1h['high']; l = df_1h['low']; c = df_1h['close']
    feat = pd.DataFrame(index=df_1h.index)
    rng  = (h - l).clip(lower=1e-10)

    feat['close_position'] = (c - l) / rng
    feat['body_ratio']     = (c - o).abs() / rng
    feat['upper_wick']     = (h - np.maximum(o, c)) / rng
    feat['lower_wick']     = (np.minimum(o, c) - l) / rng
    feat['wick_asymmetry'] = feat['upper_wick'] - feat['lower_wick']
    feat['bar_direction']  = np.sign(c - o).astype(np.float32)

    d = feat['bar_direction']
    feat['consec_bull'] = d.rolling(6, min_periods=1).apply(lambda x: (x > 0).sum(), raw=True)
    feat['consec_bear'] = d.rolling(6, min_periods=1).apply(lambda x: (x < 0).sum(), raw=True)
    feat['prev_close_pos']  = feat['close_position'].shift(1)
    feat['prev_body_ratio'] = feat['body_ratio'].shift(1)
    return feat.astype(np.float32)


def sequence_shape_features(df_1h):
    """Layer 2: Sequence shape features (slopes, envelope, residual, curvature)."""
    h = df_1h['high']; l = df_1h['low']; c = df_1h['close']
    feat = pd.DataFrame(index=df_1h.index)

    tr  = np.maximum(h - l, np.maximum((h - c.shift(1)).abs(), (l - c.shift(1)).abs()))
    atr = tr.rolling(24, min_periods=4).mean().clip(lower=1e-10)
    feat['atr_24'] = atr

    def norm_slope(series, n):
        def _slope(x):
            if np.isnan(x).any(): return np.nan
            return np.polyfit(np.arange(len(x)), x, 1)[0]
        return series.rolling(n, min_periods=max(3, n//2)).apply(_slope, raw=True) / atr

    def regression_residual(series, n):
        def _resid(x):
            if np.isnan(x).any(): return np.nan
            xi = np.arange(len(x))
            pred = np.polyval(np.polyfit(xi, x, 1), xi[-1])
            return x[-1] - pred
        return series.rolling(n, min_periods=max(3, n//2)).apply(_resid, raw=True) / atr

    for n, label in [(3,'3h'),(6,'6h'),(12,'12h'),(24,'24h')]:
        feat[f'slope_close_{label}'] = norm_slope(c, n)

    for n, label in [(6,'6h'),(12,'12h'),(24,'24h')]:
        hs = norm_slope(h, n); ls = norm_slope(l, n)
        feat[f'high_slope_{label}']       = hs
        feat[f'low_slope_{label}']        = ls
        feat[f'envelope_squeeze_{label}'] = hs - ls

    for n, label in [(6,'6h'),(12,'12h'),(24,'24h')]:
        feat[f'residual_{label}'] = regression_residual(c, n)

    feat['curvature_6h']  = feat['slope_close_6h']  - feat['slope_close_6h'].shift(3)
    feat['curvature_12h'] = feat['slope_close_12h'] - feat['slope_close_12h'].shift(6)
    feat['curvature_24h'] = feat['slope_close_24h'] - feat['slope_close_24h'].shift(12)
    return feat.astype(np.float32)


def path_geometry_features(df_1h):
    """Layer 3: Path geometry (tortuosity, slope agreement, compression, range position)."""
    h = df_1h['high']; l = df_1h['low']; c = df_1h['close']
    feat = pd.DataFrame(index=df_1h.index)

    tr  = np.maximum(h - l, np.maximum((h - c.shift(1)).abs(), (l - c.shift(1)).abs()))
    atr = tr.rolling(24, min_periods=4).mean().clip(lower=1e-10)

    def tortuosity(series, n):
        def _tort(x):
            if np.isnan(x).any(): return np.nan
            return np.abs(np.diff(x)).sum() / max(abs(x[-1] - x[0]), 1e-10)
        return series.rolling(n, min_periods=max(3, n//2)).apply(_tort, raw=True)

    feat['tortuosity_6h']  = tortuosity(c, 6)
    feat['tortuosity_12h'] = tortuosity(c, 12)
    feat['tortuosity_24h'] = tortuosity(c, 24)

    def norm_slope(series, n):
        def _slope(x):
            if np.isnan(x).any(): return np.nan
            return np.polyfit(np.arange(len(x)), x, 1)[0]
        return series.rolling(n, min_periods=max(3, n//2)).apply(_slope, raw=True)

    s3  = norm_slope(c, 3)
    s6  = norm_slope(c, 6)
    s12 = norm_slope(c, 12)
    s24 = norm_slope(c, 24)

    feat['slope_agree_3_12']  = (np.sign(s3)  == np.sign(s12)).astype(np.float32)
    feat['slope_agree_6_24']  = (np.sign(s6)  == np.sign(s24)).astype(np.float32)
    feat['slope_agree_3_24']  = (np.sign(s3)  == np.sign(s24)).astype(np.float32)
    feat['slope_alignment_3_12'] = (s3 * s12).clip(-1, 1) / atr**2
    feat['slope_alignment_6_24'] = (s6 * s24).clip(-1, 1) / atr**2

    rng = h - l
    feat['compression_6_24']  = rng.rolling(6,  min_periods=3).mean() / rng.rolling(24, min_periods=6).mean().clip(lower=1e-10)
    feat['compression_12_48'] = rng.rolling(12, min_periods=6).mean() / rng.rolling(48, min_periods=12).mean().clip(lower=1e-10)

    for n, label in [(12,'12h'),(24,'24h'),(48,'48h')]:
        h_n = h.rolling(n, min_periods=n//2).max()
        l_n = l.rolling(n, min_periods=n//2).min()
        feat[f'range_pos_{label}'] = (c - l_n) / (h_n - l_n).clip(lower=1e-10)

    return feat.astype(np.float32)


def intrabar_features(df_1h, df_15m):
    """Layer 4: Intra-hour structure from 15M bars (vectorized)."""
    df_15m = df_15m.sort_index().copy()
    df_15m['hour_ts'] = df_15m.index.floor('1h')
    valid_hours = set(df_1h.index)
    df_15m = df_15m[df_15m['hour_ts'].isin(valid_hours)]
    df_15m['bar_idx'] = df_15m.groupby('hour_ts').cumcount()

    c_pivot = df_15m.pivot_table(index='hour_ts', columns='bar_idx', values='close', aggfunc='first')
    v_pivot = df_15m.pivot_table(index='hour_ts', columns='bar_idx', values='volume', aggfunc='first')
    c_pivot.columns = [f'c{i}' for i in c_pivot.columns]
    v_pivot.columns = [f'v{i}' for i in v_pivot.columns]
    c = c_pivot.reindex(df_1h.index)
    v = v_pivot.reindex(df_1h.index)

    rng_1h = (df_1h['high'] - df_1h['low']).clip(lower=1e-10)
    feat = pd.DataFrame(index=df_1h.index)

    has_all = c[['c0','c1','c2','c3']].notna().all(axis=1) if all(f'c{i}' in c.columns for i in range(4)) else pd.Series(False, index=df_1h.index)
    slope = pd.Series(np.nan, index=df_1h.index)
    if has_all.any():
        slope[has_all] = (-3*c.loc[has_all,'c0'] - c.loc[has_all,'c1'] + c.loc[has_all,'c2'] + 3*c.loc[has_all,'c3']) / (10 * rng_1h[has_all])
    feat['intrabar_slope'] = slope

    has_c01  = c[['c0','c1']].notna().all(axis=1) if all(f'c{i}' in c.columns for i in [0,1]) else pd.Series(False, index=df_1h.index)
    has_c012 = c[['c0','c1','c2']].notna().all(axis=1) if all(f'c{i}' in c.columns for i in [0,1,2]) else pd.Series(False, index=df_1h.index)
    steps = pd.Series(0.0, index=df_1h.index)
    if has_c01.any():  steps[has_c01]  += (c['c1'] - c['c0']).abs()[has_c01]
    if has_c012.any(): steps[has_c012] += (c['c2'] - c['c1']).abs()[has_c012]
    if has_all.any():  steps[has_all]  += (c['c3'] - c['c2']).abs()[has_all]
    net = (c['c3'] - c['c0']).abs().clip(lower=1e-10) if 'c3' in c.columns and 'c0' in c.columns else pd.Series(np.nan, index=df_1h.index)
    tort = pd.Series(np.nan, index=df_1h.index)
    if has_all.any(): tort[has_all] = steps[has_all] / net[has_all]
    feat['intrabar_tortuosity'] = tort

    if 'c3' in c.columns and 'c2' in c.columns and 'c1' in c.columns:
        last_close = c['c3'].where(has_all, c['c2'].where(has_c012, c['c1'].where(has_c01, np.nan)))
    else:
        last_close = pd.Series(np.nan, index=df_1h.index)
    feat['intrabar_close_pos'] = (last_close - df_1h['low']) / rng_1h

    feat['intrabar_momentum'] = pd.Series(np.nan, index=df_1h.index)
    if has_all.any():
        feat.loc[has_all, 'intrabar_momentum'] = ((c.loc[has_all,'c3'] - c.loc[has_all,'c2']) - (c.loc[has_all,'c1'] - c.loc[has_all,'c0'])) / rng_1h[has_all]

    feat['intrabar_vol_accel'] = pd.Series(np.nan, index=df_1h.index)
    if all(f'v{i}' in v.columns for i in range(4)):
        has_v = v[['v0','v1','v2','v3']].notna().all(axis=1)
        if has_v.any():
            vol_first = ((v['v0'] + v['v1']) / 2).clip(lower=1e-10)
            vol_last  =  (v['v2'] + v['v3']) / 2
            feat.loc[has_v, 'intrabar_vol_accel'] = (vol_last / vol_first)[has_v]

    return feat.astype(np.float32)


def macro_context_features(df_1h, df_4h):
    """Layer 5: 4H macro context features."""
    feat = pd.DataFrame(index=df_1h.index)
    df_4h = df_4h.sort_index()

    h4 = df_4h['high']; l4 = df_4h['low']; c4 = df_4h['close']
    r4 = (h4 - l4).clip(lower=1e-10)
    tr4  = np.maximum(r4, np.maximum((h4 - c4.shift(1)).abs(), (l4 - c4.shift(1)).abs()))
    atr4 = tr4.rolling(12, min_periods=3).mean().clip(lower=1e-10)

    def norm_slope_4h(series, n):
        def _slope(x):
            if np.isnan(x).any(): return np.nan
            return np.polyfit(np.arange(len(x)), x, 1)[0]
        return series.rolling(n, min_periods=max(2, n//2)).apply(_slope, raw=True) / atr4

    def resid_4h(series, n):
        def _r(x):
            if np.isnan(x).any(): return np.nan
            xi = np.arange(len(x))
            pred = np.polyval(np.polyfit(xi, x, 1), xi[-1])
            return x[-1] - pred
        return series.rolling(n, min_periods=max(2, n//2)).apply(_r, raw=True) / atr4

    df_4h_feats = pd.DataFrame({
        '4h_slope_8h':    norm_slope_4h(c4, 2),
        '4h_slope_16h':   norm_slope_4h(c4, 4),
        '4h_slope_24h':   norm_slope_4h(c4, 6),
        '4h_close_pos':   (c4 - l4) / r4,
        '4h_compression': r4.rolling(3, min_periods=2).mean() / r4.rolling(12, min_periods=4).mean().clip(lower=1e-10),
        '4h_residual':    resid_4h(c4, 6),
    }, index=df_4h.index).shift(1)

    df_4h_1h = df_4h_feats.reindex(df_4h_feats.index.union(df_1h.index)).ffill().reindex(df_1h.index)
    for col in df_4h_1h.columns:
        feat[col] = df_4h_1h[col].values
    return feat.astype(np.float32)


def compute_ma_features(df_1h, pip):
    """MA context features (features_9 notebook)."""
    c = df_1h['close']; h = df_1h['high']; l = df_1h['low']
    ma50  = c.rolling(50).mean()
    ma200 = c.rolling(200).mean()

    tr    = pd.concat([h-l, (h-c.shift(1)).abs(), (l-c.shift(1)).abs()], axis=1).max(axis=1)
    atr24 = tr.rolling(24, min_periods=6).mean()

    feat = pd.DataFrame(index=df_1h.index)
    feat['dist_ma50_pips']      = (c - ma50)  / pip
    feat['dist_ma200_pips']     = (c - ma200) / pip
    feat['dist_ma50_atr']       = (c - ma50)  / atr24.clip(lower=1e-8)
    feat['dist_ma200_atr']      = (c - ma200) / atr24.clip(lower=1e-8)
    feat['ma50_ma200_gap_pips'] = (ma50 - ma200) / pip
    feat['ma50_ma200_gap_atr']  = (ma50 - ma200) / atr24.clip(lower=1e-8)
    channel = (ma200 - ma50).replace(0, np.nan)
    feat['price_in_channel']    = (c - ma50) / channel
    feat['ma50_slope_3h']       = ma50.diff(3)  / pip
    feat['ma50_slope_8h']       = ma50.diff(8)  / pip
    feat['ma50_slope_24h']      = ma50.diff(24) / pip
    feat['ma200_slope_8h']      = ma200.diff(8) / pip
    feat['ma200_slope_24h']     = ma200.diff(24) / pip
    feat['ma50_accel_3h']       = feat['ma50_slope_3h'].diff(3)
    feat['ma50_accel_8h']       = feat['ma50_slope_8h'].diff(8)
    feat['above_ma50']          = (c > ma50).astype(float)
    feat['above_ma200']         = (c > ma200).astype(float)
    feat['both_above']          = ((c > ma50) & (c > ma200)).astype(float)
    feat['both_below']          = ((c < ma50) & (c < ma200)).astype(float)
    feat['between_mas']         = (((c > ma50) & (c < ma200)) | ((c < ma50) & (c > ma200))).astype(float)

    below = (c < ma50).astype(float).values
    above_arr = (c > ma50).astype(float).values
    consec_below = np.zeros(len(c)); consec_above = np.zeros(len(c))
    for i in range(1, len(c)):
        consec_below[i] = consec_below[i-1] + 1 if below[i] else 0
        consec_above[i] = consec_above[i-1] + 1 if above_arr[i] else 0
    feat['bars_below_ma50'] = consec_below
    feat['bars_above_ma50'] = consec_above

    abs_dist_ma50 = (c - ma50).abs() / pip
    feat['min_dist_ma50_6h']   = abs_dist_ma50.rolling(6).min()
    feat['min_dist_ma50_12h']  = abs_dist_ma50.rolling(12).min()
    feat['min_dist_ma50_24h']  = abs_dist_ma50.rolling(24).min()
    feat['dist_ma50_roc_3h']   = (c - ma50).diff(3)  / pip
    feat['dist_ma50_roc_6h']   = (c - ma50).diff(6)  / pip
    feat['dist_ma50_roc_12h']  = (c - ma50).diff(12) / pip

    near_ma50  = (abs_dist_ma50 < atr24 / pip * 0.5).astype(float)
    feat['ma50_touches_24h']   = near_ma50.rolling(24).sum()
    feat['ma50_touches_48h']   = near_ma50.rolling(48).sum()

    abs_dist_ma200 = (c - ma200).abs() / pip
    near_ma200 = (abs_dist_ma200 < atr24 / pip * 0.5).astype(float)
    feat['ma200_touches_24h']  = near_ma200.rolling(24).sum()
    feat['ma200_touches_48h']  = near_ma200.rolling(48).sum()

    return feat.astype(np.float32)


def compute_features_for_pair(pair, data):
    """Compute full features_10 for a pair from raw data dict."""
    df_1h  = data[pair]['1h']
    df_15m = data[pair]['15m']
    df_4h  = data[pair]['4h']
    pip    = PIP_SIZE[pair]

    f1 = bar_shape_features(df_1h)
    f2 = sequence_shape_features(df_1h)
    f3 = path_geometry_features(df_1h)
    f4 = intrabar_features(df_1h, df_15m)
    f5 = macro_context_features(df_1h, df_4h)
    f6 = compute_ma_features(df_1h, pip)

    df_feat = pd.concat([f1, f2, f3, f4, f5, f6], axis=1)
    df_feat['pair'] = pair
    return df_feat


# ─────────────────────────────────────────────────────────────────────────────
# EPISODE TRACKER
# ─────────────────────────────────────────────────────────────────────────────

class EpisodeTracker:
    def __init__(self):
        self.in_episode = False
        self.traded     = False

    def update(self, both_below, both_above, proba):
        """Returns True if we should enter a trade now."""
        if both_below:
            if not self.in_episode:
                self.in_episode = True
                self.traded     = False
            if not self.traded and proba >= PROBA_THRESHOLD:
                self.traded = True
                return True
        elif both_above:
            self.in_episode = False
            self.traded     = False
        else:
            # Between MAs — episode continues
            if self.in_episode and not self.traded and proba >= PROBA_THRESHOLD:
                self.traded = True
                return True
        return False


# ─────────────────────────────────────────────────────────────────────────────
# CAPITAL SIMULATION
# ─────────────────────────────────────────────────────────────────────────────

def simulate_capital(all_data, model, feature_cols, backtest_start_str):
    print(f'\n{"="*70}')
    print(f'CAPITAL SIMULATION — Episode Low Long Model')
    print(f'{"="*70}')
    print(f'Proba threshold: {PROBA_THRESHOLD} | Timeout: {TIMEOUT_BARS}H | ATR stop: {ATR_STOP_MULT}× | Risk: {RISK_PCT*100:.0f}%/trade')
    print(f'Starting capital: ${STARTING_CAP:,.0f}')

    # Compute features for all pairs
    print('\nComputing features_10...')
    feat_cache = {}
    ohlc_cache = {}
    for pair in PAIRS:
        if pair not in all_data:
            continue
        feat_cache[pair] = compute_features_for_pair(pair, all_data)
        ohlc_cache[pair] = all_data[pair]['1h']
        print(f'  {pair}: {feat_cache[pair].shape}')

    # Build unified timeline from EURUSD 1H
    ref = ohlc_cache.get('EURUSD', next(iter(ohlc_cache.values())))
    backtest_start = pd.Timestamp(backtest_start_str)
    timeline = ref.index[ref.index >= backtest_start]
    print(f'\nBacktest period: {timeline[0].date()} to {timeline[-1].date()} ({len(timeline):,} bars)')

    equity      = STARTING_CAP
    peak_equity = STARTING_CAP
    max_dd      = 0.0
    max_dd_pct  = 0.0

    trades       = []
    equity_curve = []
    open_pos     = {}    # pair -> position dict
    trackers     = {pair: EpisodeTracker() for pair in PAIRS}

    for ts in timeline:
        # ── 1. Close matured positions ──────────────────────────────────────
        to_close = []
        for pair, pos in open_pos.items():
            if ts < pos['close_at']:
                continue
            # Use pre-scanned exit price — already determined at entry time
            exit_price = pos['exit']
            exit_ts    = pos['close_at']
            hour_utc   = exit_ts.hour
            raw_pnl      = compute_pnl_usd(pair, 1, pos['lots'], pos['entry'], exit_price)
            spread_cost  = compute_spread_cost_usd(pair, pos['lots'], exit_price, hour_utc)
            slip_cost    = compute_slippage_cost_usd(pair, pos['lots'], exit_price, hour_utc)
            pnl          = raw_pnl - spread_cost - slip_cost

            equity      += pnl
            peak_equity  = max(peak_equity, equity)
            dd = peak_equity - equity
            max_dd      = max(max_dd, dd)
            max_dd_pct  = max(max_dd_pct, dd / peak_equity if peak_equity > 0 else 0)

            trades.append({
                'open_ts':      pos['open_ts'],
                'close_ts':     exit_ts,
                'pair':         pair,
                'direction':    'BUY',
                'lots':         pos['lots'],
                'entry_price':  pos['entry'],
                'exit_price':   exit_price,
                'hold_h':       (exit_ts  - pos['open_ts']).total_seconds() / 3600,
                'exit_type':    pos['exit_type'],
                'pnl_pips':     (exit_price - pos['entry']) / PIP_SIZE[pair],
                'spread_cost':  spread_cost,
                'slippage_cost': slip_cost,
                'pnl_usd':      pnl,
                'equity_after': equity,
                'proba':        pos['proba'],
            })
            to_close.append(pair)

        for pair in to_close:
            del open_pos[pair]

        equity_curve.append({'ts': ts, 'equity': equity, 'n_open': len(open_pos)})

        # ── 2. Check new entries ────────────────────────────────────────────
        for pair in PAIRS:
            if pair in open_pos:
                continue
            if pair not in feat_cache or pair not in ohlc_cache:
                continue

            ohlc = ohlc_cache[pair]
            feat = feat_cache[pair]

            if ts not in ohlc.index or ts not in feat.index:
                continue

            iloc = ohlc.index.get_loc(ts)
            if iloc < 200:
                continue   # need at least 200 bars for MA200

            row = feat.loc[ts]
            both_below = bool(row.get('both_below', 0) == 1)
            both_above = bool(row.get('both_above', 0) == 1)

            # Model inference on this bar's features
            row_feat = feat.loc[[ts], feature_cols].ffill().fillna(0)
            proba = float(model.predict_proba(row_feat)[0, 1])

            signal = trackers[pair].update(both_below, both_above, proba)
            if not signal:
                continue

            # Entry
            entry_price = ohlc['close'].loc[ts]
            hour_utc    = ts.hour

            # Check margin
            locked_margin = sum(p['margin_used'] for p in open_pos.values())
            avail = equity - locked_margin
            if avail < 5_000:
                continue

            # ATR-based lot sizing (atr_24 is in price units, not pips)
            atr_price = float(row.get('atr_24', np.nan))
            pip = PIP_SIZE[pair]
            if np.isnan(atr_price) or atr_price <= 0:
                atr_price = 50.0 * pip   # fallback: 50 pips
            lots = lot_size(equity, pair, atr_price, entry_price)

            margin = margin_required(pair, lots)
            if margin > avail:
                affordable = avail * LEVERAGE / (LOT_UNITS * BASE_USD.get(pair[:3], 1.0))
                lots = max(0.01, round(min(lots, affordable), 2))
                margin = margin_required(pair, lots)
            if margin > avail:
                continue

            # Find exit: scan forward up to TIMEOUT_BARS
            # Exit on: (1) first close above MA200, (2) ATR stop hit, (3) timeout
            stop_price = entry_price - atr_price * ATR_STOP_MULT  # long stop below entry
            close_ts   = None
            exit_price = None
            exit_type  = 'TIMEOUT'

            for k in range(1, TIMEOUT_BARS + 1):
                fi = iloc + k
                if fi >= len(ohlc):
                    break
                fts = ohlc.index[fi]
                fc  = ohlc['close'].iloc[fi]
                # ATR stop hit
                if fc <= stop_price:
                    close_ts   = fts
                    exit_price = fc
                    exit_type  = 'STOP'
                    break
                # MA200 TP
                ma200_k = ohlc['close'].iloc[max(0, fi-199):fi+1].mean()
                if fc > ma200_k:
                    close_ts   = fts
                    exit_price = fc
                    exit_type  = 'TP'
                    break

            if close_ts is None:
                to_iloc    = min(iloc + TIMEOUT_BARS, len(ohlc) - 1)
                close_ts   = ohlc.index[to_iloc]
                exit_price = ohlc['close'].iloc[to_iloc]
                exit_type  = 'TIMEOUT'

            open_pos[pair] = {
                'open_ts':   ts,
                'close_at':  close_ts,
                'entry':     entry_price,
                'exit':      exit_price,
                'lots':      lots,
                'margin_used': margin,
                'exit_type': exit_type,
                'proba':     proba,
            }

    # ── Flush any still-open positions at last bar ──────────────────────────
    last_ts = timeline[-1]
    for pair, pos in open_pos.items():
        ohlc = ohlc_cache[pair]
        if last_ts in ohlc.index:
            exit_price   = ohlc['close'].loc[last_ts]
            raw_pnl      = compute_pnl_usd(pair, 1, pos['lots'], pos['entry'], exit_price)
            spread_cost  = compute_spread_cost_usd(pair, pos['lots'], exit_price, last_ts.hour)
            slip_cost    = compute_slippage_cost_usd(pair, pos['lots'], exit_price, last_ts.hour)
            pnl          = raw_pnl - spread_cost - slip_cost
            equity      += pnl
            trades.append({
                'open_ts':       pos['open_ts'],
                'close_ts':      last_ts,
                'pair':          pair,
                'direction':     'BUY',
                'lots':          pos['lots'],
                'entry_price':   pos['entry'],
                'exit_price':    exit_price,
                'hold_h':        (last_ts - pos['open_ts']).total_seconds() / 3600,
                'exit_type':     'OPEN_AT_END',
                'pnl_pips':      (exit_price - pos['entry']) / PIP_SIZE[pair],
                'spread_cost':   spread_cost,
                'slippage_cost': slip_cost,
                'pnl_usd':       pnl,
                'equity_after':  equity,
                'proba':         pos['proba'],
            })

    # ── Print results ────────────────────────────────────────────────────────
    df_trades = pd.DataFrame(trades)
    df_equity = pd.DataFrame(equity_curve).set_index('ts')

    if len(df_trades) == 0:
        print('\nNo trades taken.')
        return df_trades, df_equity

    total_trades        = len(df_trades)
    wins_mask           = df_trades['pnl_usd'] > 0
    n_wins              = wins_mask.sum()
    n_losses            = total_trades - n_wins
    win_rate            = n_wins / total_trades
    total_pnl           = df_trades['pnl_usd'].sum()
    avg_pnl             = df_trades['pnl_usd'].mean()
    avg_win             = df_trades[wins_mask]['pnl_usd'].mean()             if n_wins   > 0 else 0
    avg_loss            = df_trades[~wins_mask]['pnl_usd'].mean()            if n_losses > 0 else 0
    total_spread_cost   = df_trades['spread_cost'].sum()
    total_slippage_cost = df_trades['slippage_cost'].sum()
    gross_wins          = df_trades[wins_mask]['pnl_usd'].sum()              if n_wins   > 0 else 0
    gross_losses        = abs(df_trades[~wins_mask]['pnl_usd'].sum())        if n_losses > 0 else 0.01
    n_days              = (timeline[-1] - timeline[0]).days
    tps                 = (df_trades['exit_type'] == 'TP').sum()
    tos                 = (df_trades['exit_type'] == 'TIMEOUT').sum()
    stops               = (df_trades['exit_type'] == 'STOP').sum()

    print(f'\n--- Account Summary ---')
    print(f'Starting capital:   ${STARTING_CAP:>12,.2f}')
    print(f'Final equity:       ${equity:>12,.2f}')
    print(f'Total P&L:          ${total_pnl:>12,.2f} ({total_pnl/STARTING_CAP*100:+.2f}%)')
    print(f'Max drawdown:       ${max_dd:>12,.2f} ({max_dd_pct*100:.2f}%)')
    print(f'Total spread cost:  ${total_spread_cost:>12,.2f}')
    print(f'Total slippage:     ${total_slippage_cost:>12,.2f}')
    print(f'Total exec cost:    ${total_spread_cost + total_slippage_cost:>12,.2f}')

    print(f'\n--- Trade Statistics ---')
    print(f'Total trades:       {total_trades}')
    print(f'Trades/day:         {total_trades/max(n_days,1):.2f}')
    print(f'Win / Loss:         {n_wins} / {n_losses}')
    print(f'Win rate:           {win_rate:.1%}')
    print(f'TP exits:           {tps} ({tps/total_trades:.1%})')
    print(f'Stop exits:         {stops} ({stops/total_trades:.1%})')
    print(f'Timeout exits:      {tos} ({tos/total_trades:.1%})')
    print(f'Avg win:            ${avg_win:>10,.2f}')
    print(f'Avg loss:           ${avg_loss:>10,.2f}')
    print(f'Avg P&L/trade:      ${avg_pnl:>10,.2f}')
    print(f'Avg hold:           {df_trades["hold_h"].mean():.1f}H')
    # Net pips: convert net USD P&L back to pips per trade
    def usd_to_pips(row):
        pip = PIP_SIZE.get(row['pair'], 0.0001)
        if row['pair'] in JPY_PAIRS:
            return row['pnl_usd'] * row['exit_price'] / (row['lots'] * LOT_UNITS * pip)
        elif row['pair'][3:] == 'USD':
            return row['pnl_usd'] / (row['lots'] * LOT_UNITS * pip)
        else:
            return row['pnl_usd'] / (row['lots'] * LOT_UNITS * pip * QUOTE_USD.get(row['pair'][3:], 1.0))
    df_trades['pnl_pips_net'] = df_trades.apply(usd_to_pips, axis=1)
    print(f'Avg pips gross:     {df_trades["pnl_pips"].mean():+.1f}p')
    print(f'Avg pips net:       {df_trades["pnl_pips_net"].mean():+.1f}p')
    print(f'Profit factor:      {gross_wins/gross_losses:.2f}')

    df_trades['date'] = pd.to_datetime(df_trades['open_ts']).dt.date
    daily_pnl = df_trades.groupby('date')['pnl_usd'].sum()
    if len(daily_pnl) > 1 and daily_pnl.std() > 0:
        sharpe = (daily_pnl.mean() / daily_pnl.std()) * np.sqrt(252)
        print(f'Sharpe (annualized): {sharpe:.2f}')

    # ── Per-pair breakdown ───────────────────────────────────────────────────
    print(f'\n--- Per-Pair Breakdown ---')
    print(f'{"Pair":<10} {"Trades":>7} {"WR":>7} {"Avg P&L":>10} {"Total P&L":>12} {"Spread$":>9} {"Slip$":>9}')
    print('-' * 72)
    for pair in sorted(df_trades['pair'].unique()):
        p  = df_trades[df_trades['pair'] == pair]
        wr = (p['pnl_usd'] > 0).mean()
        flag = ' <<<' if p['pnl_usd'].sum() > 0 else ''
        print(f'{pair:<10} {len(p):>7} {wr:>6.1%} ${p["pnl_usd"].mean():>9,.2f} '
              f'${p["pnl_usd"].sum():>11,.2f} ${p["spread_cost"].sum():>8,.2f} ${p["slippage_cost"].sum():>8,.2f}{flag}')

    # ── Monthly breakdown ────────────────────────────────────────────────────
    print(f'\n--- Monthly Breakdown ---')
    df_trades['month'] = pd.to_datetime(df_trades['open_ts']).dt.to_period('M')
    print(f'{"Month":<10} {"Trades":>7} {"WR":>7} {"P&L":>12} {"Cumulative":>12}')
    print('-' * 55)
    cum = 0
    for month in sorted(df_trades['month'].unique()):
        m    = df_trades[df_trades['month'] == month]
        wr   = (m['pnl_usd'] > 0).mean()
        mpnl = m['pnl_usd'].sum()
        cum += mpnl
        flag = ' <<<' if mpnl > 0 else ''
        print(f'{str(month):<10} {len(m):>7} {wr:>6.1%} ${mpnl:>11,.2f} ${cum:>11,.2f}{flag}')

    # ── Weekly breakdown ─────────────────────────────────────────────────────
    print(f'\n--- Weekly Breakdown ---')
    df_trades['week']     = pd.to_datetime(df_trades['open_ts']).dt.isocalendar().week.values
    df_trades['year']     = pd.to_datetime(df_trades['open_ts']).dt.isocalendar().year.values
    df_trades['yearweek'] = df_trades['year'].astype(str) + '-W' + df_trades['week'].astype(str).str.zfill(2)
    print(f'{"Week":<12} {"Trades":>7} {"WR":>7} {"P&L":>12}')
    print('-' * 45)
    for week in sorted(df_trades['yearweek'].unique()):
        w    = df_trades[df_trades['yearweek'] == week]
        wr   = (w['pnl_usd'] > 0).mean()
        wpnl = w['pnl_usd'].sum()
        flag = ' <<<' if wpnl > 0 else ''
        print(f'{week:<12} {len(w):>7} {wr:>6.1%} ${wpnl:>11,.2f}{flag}')

    # ── Top wins & losses ────────────────────────────────────────────────────
    print(f'\n--- Top 5 Wins ---')
    for _, t in df_trades.nlargest(5, 'pnl_usd').iterrows():
        print(f'  {str(t["open_ts"])[:16]}  {t["pair"]:>8}  BUY  {t["lots"]} lots  '
              f'proba={t["proba"]:.3f}  pips={t["pnl_pips"]:+.1f}  P&L=${t["pnl_usd"]:>+,.2f}  [{t["exit_type"]}]')

    print(f'\n--- Top 5 Losses ---')
    for _, t in df_trades.nsmallest(5, 'pnl_usd').iterrows():
        print(f'  {str(t["open_ts"])[:16]}  {t["pair"]:>8}  BUY  {t["lots"]} lots  '
              f'proba={t["proba"]:.3f}  pips={t["pnl_pips"]:+.1f}  P&L=${t["pnl_usd"]:>+,.2f}  [{t["exit_type"]}]')

    # ── Hour-of-day analysis ─────────────────────────────────────────────────
    print(f'\n--- P&L by Hour of Day ---')
    df_trades['hour_of_day'] = pd.to_datetime(df_trades['open_ts']).dt.hour
    hourly = df_trades.groupby('hour_of_day').agg(
        trades=('pnl_usd', 'count'),
        wr=('pnl_usd', lambda x: (x > 0).mean()),
        pnl=('pnl_usd', 'sum'),
    )
    print(f'{"Hour":>6} {"Trades":>7} {"WR":>7} {"P&L":>12}')
    print('-' * 40)
    for hour_val, row in hourly.iterrows():
        flag = ' <<<' if row['pnl'] > 0 else ''
        print(f'{hour_val:>6} {int(row["trades"]):>7} {row["wr"]:>6.1%} ${row["pnl"]:>11,.2f}{flag}')

    # ── Full trade log ───────────────────────────────────────────────────────
    print(f'\n{"="*130}')
    print(f'FULL TRADE LOG')
    print(f'{"="*130}')
    print(f'{"#":>4} {"Entry Time":<20} {"Exit Time":<20} {"Pair":<8} {"Lots":>5} '
          f'{"Entry":>10} {"Exit":>10} {"Hold":>6} {"Spread$":>8} {"Slip$":>7} {"P&L":>10} {"Result":<7} {"ExitType":<10} {"Equity":>12}')
    print('-' * 130)
    for i, (_, t) in enumerate(df_trades.iterrows(), 1):
        result = 'WIN' if t['pnl_usd'] > 0 else 'LOSS'
        print(f'{i:>4} {str(t["open_ts"])[:16]:<20} {str(t["close_ts"])[:16]:<20} {t["pair"]:<8} {t["lots"]:>5} '
              f'{t["entry_price"]:>10.5f} {t["exit_price"]:>10.5f} {t["hold_h"]:>5.0f}H '
              f'${t["spread_cost"]:>7.2f} ${t["slippage_cost"]:>6.2f} ${t["pnl_usd"]:>+9.2f} {result:<7} {t["exit_type"]:<10} ${t["equity_after"]:>11,.2f}')

    df_trades.to_csv('backtest_ep_low_trades.csv', index=False)
    df_equity.to_csv('backtest_ep_low_equity.csv')
    print(f'\nTrades -> backtest_ep_low_trades.csv')
    print(f'Equity -> backtest_ep_low_equity.csv')

    return df_trades, df_equity


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

async def main():
    print('Episode Low Backtest — LumenY 9')
    print(f'  Model:      {MODELS_DIR}/model_ep_low.joblib')
    print(f'  Capital:    ${STARTING_CAP:,.0f}')
    print(f'  Risk/trade: {RISK_PCT*100:.0f}%')
    print(f'  Threshold:  {PROBA_THRESHOLD}')
    print(f'  Timeout:    {TIMEOUT_BARS}H')

    # Load model
    bundle       = joblib.load(MODELS_DIR / 'model_ep_low.joblib')
    model        = bundle['model']
    feature_cols = bundle['feature_cols']
    print(f'\nModel loaded. CV AUC={bundle.get("cv_auc", "?"):.4f}. Features: {len(feature_cols)}')

    # Fetch data
    all_data = await fetch_all_pairs()
    if not all_data:
        print('ERROR: No data fetched. Check API key.')
        return

    # Compute backtest start (exclude warmup)
    now            = datetime.utcnow()
    backtest_start = (now - timedelta(days=BACKTEST_DAYS)).strftime('%Y-%m-%d')
    print(f'\nBacktest start (after warmup): {backtest_start}')

    # Run simulation
    simulate_capital(all_data, model, feature_cols, backtest_start)


if __name__ == '__main__':
    asyncio.run(main())
