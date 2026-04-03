"""
Paper Trading — Real Bid/Ask Spreads from Polygon (last 14 days)

Same pipeline as test_capital_sim_3.py but:
- Only last 14 days of data
- At each signal hour, fetches REAL bid/ask quotes from Polygon
  around the entry timestamp to get the actual spread at trade time
- Compares real spread vs hardcoded assumption
- Reports EV per trade using real spread costs

Key output: did the edge survive real spreads?
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
from scipy import stats
from math import lgamma

warnings.filterwarnings('ignore')

from dotenv import load_dotenv
load_dotenv()

API_KEY = os.getenv('POLYGON_S3_SECRET_KEY', '')
REST_BASE = 'https://api.polygon.io'

PAIRS = [
    # Majors
    'EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD', 'USDCAD', 'NZDUSD',
    # Liquid crosses only
    'EURJPY', 'GBPJPY', 'EURGBP',
]

MODELS_DIR = Path('backend/models_6/3_quants')
META_DIR   = Path('backend/models_6/meta')

# ── Capital simulation parameters ──
STARTING_CAPITAL = 1_000_000.0
LOT_UNITS = 100_000       # 1 standard lot = 100,000 units
MIN_MARGIN_TO_TRADE = 4_000
MAX_SPREAD_POINTS = 50
RISK_PER_TRADE = 0.005    # 0.5% of equity per trade

# Market capacity caps (lots) — from BIS 2022 analysis, 50% of 1% hourly volume / 4 simultaneous
# Binding constraint is always the illiquid crosses
CAPACITY_CAPS = {
    'EURUSD': 139, 'GBPUSD': 58,  'USDJPY': 83,  'USDCHF': 13,
    'AUDUSD': 24,  'USDCAD': 18,  'NZDUSD': 6,
    'EURJPY': 11,  'GBPJPY': 8,   'EURGBP': 9,
    'EURAUD': 3,   'AUDJPY': 3,   'CADJPY': 1,   'CHFJPY': 2,   'AUDNZD': 1,
}

MAJORS = {'EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD', 'USDCAD', 'NZDUSD'}
CROSSES = {'EURJPY', 'GBPJPY', 'EURGBP', 'EURAUD', 'AUDJPY', 'CADJPY', 'CHFJPY', 'AUDNZD'}

LEVERAGE = {pair: 50 for pair in MAJORS}
LEVERAGE.update({pair: 50 for pair in CROSSES})

# Realistic spreads in POINTS — calibrated from Polygon bid/ask quotes (30-day median, liquid hours)
SPREAD_POINTS = {
    'AUDUSD': 6, 'EURUSD': 6, 'GBPUSD': 8, 'NZDUSD': 9,
    'USDCAD': 12, 'USDCHF': 7, 'USDJPY': 10,
    'EURGBP': 7, 'AUDNZD': 20, 'AUDJPY': 15, 'CADJPY': 16,
    'CHFJPY': 25, 'EURAUD': 21, 'EURJPY': 14, 'GBPJPY': 21,
}

JPY_PAIRS = {'USDJPY', 'EURJPY', 'GBPJPY', 'AUDJPY', 'CADJPY', 'CHFJPY'}

PIP_SIZE = {
    'EURUSD': 0.0001, 'GBPUSD': 0.0001, 'AUDUSD': 0.0001, 'NZDUSD': 0.0001,
    'USDCAD': 0.0001, 'USDCHF': 0.0001, 'USDJPY': 0.01,
    'EURJPY': 0.01, 'GBPJPY': 0.01, 'EURGBP': 0.0001, 'EURAUD': 0.0001,
    'AUDJPY': 0.01, 'CADJPY': 0.01, 'CHFJPY': 0.01, 'AUDNZD': 0.0001,
}

# At 21:00-23:00 UTC — off-hours multipliers calibrated from Polygon (off-hours median / liquid median)
OFFHOUR_SPREAD_MULTIPLIER = {
    # Majors
    'EURUSD': 1.1, 'GBPUSD': 1.1, 'USDJPY': 1.1, 'USDCHF': 1.2,
    'AUDUSD': 1.0, 'USDCAD': 0.9, 'NZDUSD': 1.0,
    # Crosses
    'EURJPY': 1.2, 'GBPJPY': 1.2, 'EURGBP': 1.1, 'EURAUD': 1.2,
    'AUDJPY': 1.1, 'CADJPY': 1.2, 'CHFJPY': 1.2, 'AUDNZD': 1.3,
}

def get_spread_points(pair, hour_utc):
    """Get spread in points, adjusted for time of day."""
    base = SPREAD_POINTS.get(pair, 5)
    if hour_utc in (21, 22):
        mult = OFFHOUR_SPREAD_MULTIPLIER.get(pair, 2.0)
        return int(base * mult)
    return base

def spread_in_price(pair, hour_utc=12):
    pts = get_spread_points(pair, hour_utc)
    if pair in JPY_PAIRS:
        return pts * 0.001
    else:
        return pts * 0.00001

# Approximate base currency value in USD (for margin calculation)
BASE_USD = {
    'EUR': 1.15, 'GBP': 1.34, 'AUD': 0.70, 'NZD': 0.58,
    'USD': 1.00, 'CAD': 0.73, 'CHF': 0.79, 'JPY': 0.0063,
}

# Approximate quote currency value in USD (for P&L conversion)
QUOTE_USD = {
    'USD': 1.00, 'CHF': 1.27, 'CAD': 0.73, 'GBP': 1.34,
    'AUD': 0.70, 'NZD': 0.58, 'JPY': 0.0063,
}

def margin_required(pair, lots):
    base_ccy = pair[:3]
    base_val = BASE_USD.get(base_ccy, 1.0)
    notional = lots * LOT_UNITS * base_val
    return notional / LEVERAGE[pair]

def compute_pnl_usd(pair, direction, lots, entry_price, exit_price):
    """Compute P&L in USD for a closed position."""
    if direction == 1:
        price_move = exit_price - entry_price
    else:
        price_move = entry_price - exit_price

    quote_ccy = pair[3:]
    if pair in JPY_PAIRS:
        pnl = lots * LOT_UNITS * price_move / exit_price
    elif quote_ccy == 'USD':
        pnl = lots * LOT_UNITS * price_move
    else:
        rate = QUOTE_USD.get(quote_ccy, 1.0)
        pnl = lots * LOT_UNITS * price_move * rate
    return pnl

def compute_spread_cost_usd(pair, lots, exit_price, hour_utc=12):
    """Compute spread cost in USD."""
    sp = spread_in_price(pair, hour_utc)
    quote_ccy = pair[3:]
    if pair in JPY_PAIRS:
        return lots * LOT_UNITS * sp / exit_price
    elif quote_ccy == 'USD':
        return lots * LOT_UNITS * sp
    else:
        rate = QUOTE_USD.get(quote_ccy, 1.0)
        return lots * LOT_UNITS * sp * rate

# ── Slippage model ──
# Base slippage in points (liquid hours) — calibrated from Polygon bid_std (quote instability proxy)
# Majors: ~3 pts, Crosses: ~5-9 pts
SLIPPAGE_BASE_POINTS = {
    'EURUSD': 3.0, 'GBPUSD': 4.0, 'USDJPY': 5.0, 'USDCHF': 5.0,
    'AUDUSD': 3.0, 'USDCAD': 3.0, 'NZDUSD': 3.0,
    'EURGBP': 3.0, 'AUDNZD': 6.0, 'AUDJPY': 4.0, 'CADJPY': 6.0,
    'CHFJPY': 9.0, 'EURAUD': 6.0, 'EURJPY': 6.0, 'GBPJPY': 7.0,
}

# Time multiplier: off-hours = thinner books = more slippage
SLIPPAGE_TIME_MULTIPLIER = {
    'EURUSD': 1.5, 'GBPUSD': 2.0, 'USDJPY': 2.0, 'USDCHF': 2.0,
    'AUDUSD': 2.0, 'USDCAD': 2.0, 'NZDUSD': 2.0,
    'EURJPY': 2.5, 'GBPJPY': 2.5, 'EURGBP': 2.0, 'EURAUD': 2.5,
    'AUDJPY': 2.5, 'CADJPY': 2.5, 'CHFJPY': 3.0, 'AUDNZD': 2.5,
}

def get_slippage_points(pair, hour_utc, realized_vol=None, lots=1.0):
    """Slippage = base × time_multiplier × volatility_multiplier × size_multiplier."""
    base = SLIPPAGE_BASE_POINTS.get(pair, 1.0)

    # Time multiplier (21-22 UTC = post-NY close, thinnest books)
    if hour_utc in (21, 22):
        time_mult = SLIPPAGE_TIME_MULTIPLIER.get(pair, 2.0)
    else:
        time_mult = 1.0

    # Volatility multiplier: high vol = more slippage (books thin out)
    vol_mult = 1.0
    if realized_vol is not None and realized_vol > 0:
        if realized_vol < 0.08:
            vol_mult = 0.8
        elif realized_vol < 0.15:
            vol_mult = 1.0
        elif realized_vol < 0.25:
            vol_mult = 1.5
        else:
            vol_mult = 2.0

    # Size multiplier: larger positions walk the book
    # +5% per lot above 1, capped at 3x
    size_mult = min(1.0 + 0.05 * max(lots - 1.0, 0), 3.0)

    return base * time_mult * vol_mult * size_mult

def slippage_in_price(pair, hour_utc, realized_vol=None, lots=1.0):
    pts = get_slippage_points(pair, hour_utc, realized_vol, lots)
    if pair in JPY_PAIRS:
        return pts * 0.001
    else:
        return pts * 0.00001

def compute_slippage_cost_usd(pair, lots, exit_price, hour_utc=12, realized_vol=None):
    """Compute slippage cost in USD."""
    sp = slippage_in_price(pair, hour_utc, realized_vol, lots)
    quote_ccy = pair[3:]
    if pair in JPY_PAIRS:
        return lots * LOT_UNITS * sp / exit_price
    elif quote_ccy == 'USD':
        return lots * LOT_UNITS * sp
    else:
        rate = QUOTE_USD.get(quote_ccy, 1.0)
        return lots * LOT_UNITS * sp * rate


# ── Model thresholds (same as live) ──
AVG_SPREAD = 0.00028
MIN_Q50_THRESHOLD = AVG_SPREAD * 0.7
META_THRESHOLD = 0.50

BACKTEST_DAYS = 14    # paper trading: last 2 weeks only
WARMUP_DAYS = 10
TOTAL_FETCH_DAYS = BACKTEST_DAYS + WARMUP_DAYS
DATE_OFFSET_DAYS = 0

print(f'Capital Simulation — Realistic P&L')
print(f'  Starting capital: ${STARTING_CAPITAL:,.0f}')
print(f'  Lot sizing: ATR-based (0.5% risk/trade) with per-pair capacity caps')
print(f'  Max spread: {MAX_SPREAD_POINTS} points')
print(f'  Meta threshold: {META_THRESHOLD}')
print(f'  Q50 threshold: {MIN_Q50_THRESHOLD}')
print(f'  Fetch window: {TOTAL_FETCH_DAYS} days (offset: {DATE_OFFSET_DAYS} days back)')


# ──────────────────────────────────────────────
# DATA FETCHING (same as v5.2)
# ──────────────────────────────────────────────
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
    df = df.rename(columns={'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume'})
    df = df.set_index('datetime')[['open', 'high', 'low', 'close', 'volume']]
    df = df.sort_index().drop_duplicates()
    df = df[~((df.index.dayofweek == 5) | ((df.index.dayofweek == 6) & (df.index.hour < 21)))]
    return df


async def fetch_all_pairs():
    now = datetime.utcnow() - timedelta(days=DATE_OFFSET_DAYS)
    to_date = now.strftime('%Y-%m-%d')
    from_date = (now - timedelta(days=TOTAL_FETCH_DAYS)).strftime('%Y-%m-%d')

    print(f'\nFetching data from {from_date} to {to_date}...')
    data = {}

    for pair in PAIRS:
        print(f'  {pair}...', end=' ', flush=True)
        t0 = time.time()
        df_1m = await fetch_bars(pair, 1, 'minute', from_date, to_date)
        elapsed = time.time() - t0

        if df_1m.empty:
            print(f'NO DATA')
            continue

        df_5m = df_1m.resample('5min').agg({
            'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
        }).dropna()
        df_15m = df_1m.resample('15min').agg({
            'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
        }).dropna()
        df_1h = df_1m.resample('1h').agg({
            'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
        }).dropna()

        data[pair] = {'1m': df_1m, '5m': df_5m, '15m': df_15m, '1h': df_1h}
        print(f'{len(df_1m):,} bars ({elapsed:.1f}s) | {df_1m.index.min().date()} to {df_1m.index.max().date()}')
        await asyncio.sleep(0.5)

    return data


# ──────────────────────────────────────────────
# MICROSTRUCTURE FEATURES (copied from v5.2)
# ──────────────────────────────────────────────
def realized_vol_estimators(o, h, l, c):
    n = len(o)
    if n < 5:
        return {k: np.nan for k in ['rv_close', 'rv_parkinson', 'rv_garman_klass',
                                     'rv_rogers_satchell', 'rv_yang_zhang', 'range_return_ratio']}
    log_ret = np.log(c[1:] / c[:-1])
    log_hl = np.log(h / l)
    log_co = np.log(c / o)
    log_ho = np.log(h / o)
    log_lo = np.log(l / o)
    log_hc = np.log(h / c)
    log_lc = np.log(l / c)

    rv_close = np.sum(log_ret**2)
    rv_parkinson = np.sum(log_hl**2) / (4 * n * np.log(2))
    rv_gk = np.mean(0.5 * log_hl**2 - (2 * np.log(2) - 1) * log_co**2)
    rv_rs = np.mean(log_ho * log_hc + log_lo * log_lc)

    log_oc = np.log(o[1:] / c[:-1])
    n_oc = len(log_oc)
    if n_oc > 1:
        mu_oc = np.mean(log_oc)
        sigma_oc = np.sum((log_oc - mu_oc)**2) / (n_oc - 1)
        mu_co = np.mean(log_co)
        sigma_co = np.sum((log_co - mu_co)**2) / (n - 1)
        sigma_rs = np.mean(log_ho * log_hc + log_lo * log_lc)
        k = 0.34 / (1.34 + (n + 1) / (n - 1))
        rv_yz = sigma_oc + k * sigma_co + (1 - k) * sigma_rs
    else:
        rv_yz = rv_gk

    total_range = np.sum(log_hl)
    total_abs_ret = np.sum(np.abs(log_ret))
    range_return_ratio = total_abs_ret / total_range if total_range > 1e-15 else 1.0

    return {
        'rv_close': rv_close, 'rv_parkinson': rv_parkinson, 'rv_garman_klass': rv_gk,
        'rv_rogers_satchell': rv_rs, 'rv_yang_zhang': rv_yz, 'range_return_ratio': range_return_ratio,
    }


def jump_detection(returns):
    n = len(returns)
    if n < 5:
        return {k: np.nan for k in ['rv', 'bv', 'jump_ratio', 'jump_z',
                                     'jump_intensity', 'jump_mean_size', 'jump_asymmetry']}
    rv = np.sum(returns**2)
    abs_r = np.abs(returns)
    bv = (np.pi / 2) * (n / (n - 1)) * np.sum(abs_r[1:] * abs_r[:-1])
    jump = max(rv - bv, 0)
    jump_ratio = jump / rv if rv > 1e-20 else 0

    mu_43 = 2**(2/3) * np.exp(lgamma(7/6) - lgamma(1/2))
    if n >= 4:
        tpq = n * mu_43**(-3) * (n / (n-2)) * np.sum(
            abs_r[2:]**(4/3) * abs_r[1:-1]**(4/3) * abs_r[:-2]**(4/3))
        v_const = np.pi**2/4 + np.pi - 5
        relative_qv = max(tpq / (bv**2) - 1, 0) if bv > 1e-20 else 0
        denom = np.sqrt(v_const * relative_qv / n) * bv if bv > 1e-20 else 0
        jump_z = (rv - bv) / denom if denom > 1e-20 else 0
        jump_z = np.clip(jump_z, -10, 10)
    else:
        jump_z = 0

    threshold = 3.0 * np.sqrt(max(bv / n, 1e-20))
    is_jump = np.abs(returns) > threshold
    n_jumps = is_jump.sum()
    jump_intensity = n_jumps / n

    if n_jumps > 0:
        jump_returns = returns[is_jump]
        jump_mean_size = np.mean(np.abs(jump_returns))
        n_pos = (jump_returns > 0).sum()
        n_neg = (jump_returns < 0).sum()
        jump_asymmetry = (n_pos - n_neg) / n_jumps
    else:
        jump_mean_size = 0
        jump_asymmetry = 0

    return {
        'rv': rv, 'bv': bv, 'jump_ratio': jump_ratio, 'jump_z': jump_z,
        'jump_intensity': jump_intensity, 'jump_mean_size': jump_mean_size, 'jump_asymmetry': jump_asymmetry,
    }


def order_flow_features(o, h, l, c, v, rolling_sigma=None):
    n = len(c)
    if n < 5:
        return {k: np.nan for k in ['buy_volume_frac', 'order_imbalance', 'order_imbalance_intensity',
                                     'kyle_lambda', 'kyle_lambda_r2', 'amihud_illiq', 'volume_cv']}
    log_ret = np.log(c / o)
    if rolling_sigma is not None and rolling_sigma > 1e-10:
        sigma = rolling_sigma
    else:
        sigma = max(np.std(np.diff(np.log(c))), 1e-10)

    z_scores = log_ret / sigma
    v_buy = v * stats.norm.cdf(z_scores)
    v_sell = v - v_buy
    total_v = max(v.sum(), 1)
    buy_frac = v_buy.sum() / total_v
    oi = (v_buy.sum() - v_sell.sum()) / total_v
    oi_intensity = np.abs(v_buy.sum() - v_sell.sum()) / total_v

    bar_returns = np.diff(np.log(c))
    signed_flow = np.sign(bar_returns) * np.sqrt(np.maximum(v[1:], 1))
    if len(bar_returns) > 3 and np.std(signed_flow) > 1e-10:
        slope, intercept, r_value, _, _ = stats.linregress(signed_flow, bar_returns)
        kyle_lambda = slope
        kyle_r2 = r_value**2
    else:
        kyle_lambda = 0
        kyle_r2 = 0

    v_safe = np.maximum(v, 1)
    abs_ret = np.abs(np.diff(np.log(c)))
    amihud = np.mean(abs_ret / np.log1p(v_safe[1:]))
    v_mean = v.mean()
    volume_cv = v.std() / v_mean if v_mean > 0 else 0

    return {
        'buy_volume_frac': buy_frac, 'order_imbalance': oi, 'order_imbalance_intensity': oi_intensity,
        'kyle_lambda': kyle_lambda, 'kyle_lambda_r2': kyle_r2, 'amihud_illiq': amihud, 'volume_cv': volume_cv,
    }


def information_theory_features(returns, volumes=None):
    n = len(returns)
    if n < 10:
        return {k: np.nan for k in ['entropy_returns', 'entropy_norm', 'jb_statistic',
                                     'entropy_volume_divergence', 'kl_proxy']}
    iqr_val = np.percentile(returns, 75) - np.percentile(returns, 25)
    if iqr_val > 0:
        bin_width = 2 * iqr_val * n**(-1/3)
        n_bins = max(int(np.ceil((returns.max() - returns.min()) / bin_width)), 5)
        n_bins = min(n_bins, 30)
    else:
        n_bins = 10

    hist, _ = np.histogram(returns, bins=n_bins)
    p = hist / hist.sum()
    p = p[p > 0]
    entropy = -np.sum(p * np.log(p))
    entropy_norm = entropy / np.log(n_bins)

    s = stats.skew(returns)
    k = stats.kurtosis(returns, fisher=True)
    jb = (n / 6) * (s**2 + (k**2) / 4)
    kl_proxy = jb / n

    if volumes is not None and len(volumes) == n and volumes.sum() > 0:
        bin_edges = np.linspace(returns.min() - 1e-10, returns.max() + 1e-10, n_bins + 1)
        bin_idx = np.digitize(returns, bin_edges) - 1
        bin_idx = np.clip(bin_idx, 0, n_bins - 1)
        vol_per_bin = np.zeros(n_bins)
        for i in range(n):
            vol_per_bin[bin_idx[i]] += volumes[i]
        pv = vol_per_bin / vol_per_bin.sum()
        pv = pv[pv > 0]
        entropy_vol = -np.sum(pv * np.log(pv))
        entropy_vol_norm = entropy_vol / np.log(n_bins)
        entropy_volume_divergence = entropy_norm - entropy_vol_norm
    else:
        entropy_volume_divergence = 0

    return {
        'entropy_returns': entropy, 'entropy_norm': entropy_norm, 'jb_statistic': jb,
        'entropy_volume_divergence': entropy_volume_divergence, 'kl_proxy': kl_proxy,
    }


def hurst_rs(series, min_window=10):
    n = len(series)
    if n < 30:
        return np.nan
    max_k = int(np.log2(n))
    window_sizes = [2**i for i in range(int(np.log2(min_window)), max_k + 1) if 2**i <= n // 2]
    if len(window_sizes) < 3:
        return np.nan
    rs_values = []
    for w in window_sizes:
        n_segments = n // w
        rs_list = []
        for seg in range(n_segments):
            chunk = series[seg*w:(seg+1)*w]
            m = chunk.mean()
            cumdev = np.cumsum(chunk - m)
            R = cumdev.max() - cumdev.min()
            S = chunk.std(ddof=1)
            if S > 1e-15:
                rs_list.append(R / S)
        if rs_list:
            rs_values.append((np.log(w), np.log(np.mean(rs_list))))
    if len(rs_values) < 3:
        return np.nan
    x = np.array([v[0] for v in rs_values])
    y = np.array([v[1] for v in rs_values])
    slope, _, _, _, _ = stats.linregress(x, y)
    return slope


def fractal_dimension_higuchi(series, k_max=16):
    n = len(series)
    if n < 30:
        return np.nan
    k_values = [k for k in [1, 2, 4, 8, 16, 32] if k < n // 4]
    if len(k_values) < 3:
        return np.nan
    lk_values = []
    for k in k_values:
        lm_list = []
        for m in range(1, k + 1):
            indices = np.arange(m - 1, n, k)
            if len(indices) < 2:
                continue
            diffs = np.abs(np.diff(series[indices]))
            norm = (n - 1) / (len(diffs) * k * k)
            lm_list.append(diffs.sum() * norm)
        if lm_list:
            lk_values.append((np.log(1.0/k), np.log(np.mean(lm_list))))
    if len(lk_values) < 3:
        return np.nan
    x = np.array([v[0] for v in lk_values])
    y = np.array([v[1] for v in lk_values])
    slope, _, _, _, _ = stats.linregress(x, y)
    return slope


def fractal_features(returns, prices):
    return {'fractal_dim': fractal_dimension_higuchi(prices)}


def market_efficiency_features(returns):
    n = len(returns)
    if n < 15:
        return {k: np.nan for k in ['vr_5', 'vr_10', 'vr_z5', 'runs_z',
                                     'autocorr_1', 'autocorr_2', 'autocorr_5',
                                     'autocorr_decay_halflife', 'sum_abs_autocorr', 'noise_to_signal']}
    rv1 = np.var(returns, ddof=1)

    def variance_ratio(r, q):
        if len(r) < q + 1 or rv1 < 1e-20:
            return 1.0, 0.0
        r_q = np.array([r[i:i+q].sum() for i in range(len(r) - q + 1)])
        vr = np.var(r_q, ddof=1) / (q * rv1)
        z = (vr - 1) / np.sqrt(2 * (2*q - 1) * (q - 1) / (3 * q * n))
        return vr, z

    vr5, z5 = variance_ratio(returns, 5)
    vr10, _ = variance_ratio(returns, 10)

    signs = returns[returns != 0]
    if len(signs) > 5:
        pos = (signs > 0).astype(int)
        n_pos = pos.sum()
        n_neg = len(pos) - n_pos
        runs = 1 + np.sum(np.diff(pos) != 0)
        n_total = len(pos)
        if n_pos > 0 and n_neg > 0:
            e_runs = 1 + 2 * n_pos * n_neg / n_total
            v_runs = (2*n_pos*n_neg*(2*n_pos*n_neg - n_total)) / (n_total**2 * (n_total - 1))
            runs_z = (runs - e_runs) / np.sqrt(max(v_runs, 1e-20))
        else:
            runs_z = 0
    else:
        runs_z = 0

    def autocorr(r, lag):
        if len(r) <= lag + 1:
            return 0
        r_dm = r - r.mean()
        denom = np.sum(r_dm**2)
        if denom < 1e-20:
            return 0
        return np.sum(r_dm[lag:] * r_dm[:-lag]) / denom

    ac1 = autocorr(returns, 1)
    ac2 = autocorr(returns, 2)
    ac5 = autocorr(returns, 5)

    lags = list(range(1, min(21, n//2)))
    abs_acs = [abs(autocorr(returns, k)) for k in lags]
    valid = [(k, ac) for k, ac in zip(lags, abs_acs) if ac > 1e-6]
    if len(valid) >= 3:
        x_fit = np.array([v[0] for v in valid])
        y_fit = np.log(np.array([v[1] for v in valid]))
        slope, _, _, _, _ = stats.linregress(x_fit, y_fit)
        tau = -1 / slope if slope < -0.01 else 100
        halflife = tau * np.log(2)
    else:
        halflife = 0
    sum_abs_ac = sum(abs_acs)

    noise_var = -np.mean(returns[:-1] * returns[1:])
    noise_var = max(noise_var, 0)
    bv_per_bar = np.mean(np.abs(returns[1:]) * np.abs(returns[:-1])) * (np.pi / 2)
    nsr = noise_var / bv_per_bar if bv_per_bar > 1e-20 else 0

    return {
        'vr_5': vr5, 'vr_10': vr10, 'vr_z5': z5, 'runs_z': runs_z,
        'autocorr_1': ac1, 'autocorr_2': ac2, 'autocorr_5': ac5,
        'autocorr_decay_halflife': halflife, 'sum_abs_autocorr': sum_abs_ac, 'noise_to_signal': nsr,
    }


def tail_risk_features(returns):
    n = len(returns)
    if n < 10:
        return {k: np.nan for k in ['realized_skew', 'realized_kurt', 'tail_ratio_95_5', 'hill_tail_index']}
    rv = np.sum(returns**2)
    if rv > 1e-20:
        rskew = (np.sqrt(n) * np.sum(returns**3)) / rv**1.5
        rkurt = (n * np.sum(returns**4)) / rv**2
    else:
        rskew = 0
        rkurt = 3
    p95 = np.abs(np.percentile(returns, 95))
    p5 = np.abs(np.percentile(returns, 5))
    tail_ratio = p95 / p5 if p5 > 1e-15 else 1.0

    abs_r = np.sort(np.abs(returns))[::-1]
    k = max(int(np.sqrt(n)), 3)
    if k < n and abs_r[k] > 1e-15:
        log_sum = np.sum(np.log(abs_r[:k] / abs_r[k]))
        hill = k / log_sum if log_sum > 1e-10 else np.nan
        hill = min(hill, 20.0) if hill is not np.nan else np.nan
    else:
        hill = np.nan

    return {'realized_skew': rskew, 'realized_kurt': rkurt, 'tail_ratio_95_5': tail_ratio, 'hill_tail_index': hill}


def acceleration_features(c, returns):
    n = len(returns)
    if n < 10:
        return {k: np.nan for k in ['accel_mean', 'accel_std', 'accel_skew',
                                     'momentum_shift', 'ret_concentration', 'vol_clustering_ac1']}
    velocity = np.diff(c)
    if len(velocity) > 1:
        accel = np.diff(velocity)
        accel_mean = np.mean(accel)
        accel_std = np.std(accel)
        accel_skew = stats.skew(accel) if len(accel) > 3 else 0
    else:
        accel_mean = accel_std = accel_skew = 0

    third = n // 3
    if third > 0:
        momentum_shift = returns[-third:].sum() - returns[:third].sum()
    else:
        momentum_shift = 0

    abs_ret = np.abs(returns)
    total_abs = abs_ret.sum()
    if total_abs > 1e-15:
        top_k = max(int(n * 0.1), 1)
        ret_concentration = np.sort(abs_ret)[-top_k:].sum() / total_abs
    else:
        ret_concentration = 0

    r2 = returns**2
    if len(r2) > 2:
        r2_dm = r2 - r2.mean()
        denom = np.sum(r2_dm**2)
        vol_cluster_ac1 = np.sum(r2_dm[1:] * r2_dm[:-1]) / denom if denom > 1e-20 else 0
    else:
        vol_cluster_ac1 = 0

    return {
        'accel_mean': accel_mean, 'accel_std': accel_std, 'accel_skew': accel_skew,
        'momentum_shift': momentum_shift, 'ret_concentration': ret_concentration,
        'vol_clustering_ac1': vol_cluster_ac1,
    }


def cross_timeframe_features(returns_1m, returns_5m, returns_15m):
    feat = {}
    rv_1m = np.sum(returns_1m**2) if len(returns_1m) > 0 else np.nan
    rv_5m = np.sum(returns_5m**2) if len(returns_5m) > 0 else np.nan
    rv_15m = np.sum(returns_15m**2) if len(returns_15m) > 0 else np.nan

    def safe_ratio(num, den, cap=50.0):
        if den is not None and den > 1e-20:
            return min(num / den, cap)
        return np.nan

    feat['epps_1m_5m'] = safe_ratio(rv_1m, rv_5m)
    feat['epps_1m_15m'] = safe_ratio(rv_1m, rv_15m)
    feat['epps_5m_15m'] = safe_ratio(rv_5m, rv_15m)

    if len(returns_1m) >= 10:
        mid = len(returns_1m) // 2
        rv_first = np.sum(returns_1m[:mid]**2)
        rv_last = np.sum(returns_1m[mid:]**2)
        feat['info_accel'] = min(rv_last / rv_first, 50.0) if rv_first > 1e-20 else 1.0
    else:
        feat['info_accel'] = np.nan

    if len(returns_1m) >= 30:
        block_size = len(returns_1m) // 6
        block_vols = []
        for i in range(6):
            block = returns_1m[i*block_size:(i+1)*block_size]
            if len(block) > 1:
                block_vols.append(np.std(block))
        if len(block_vols) > 1:
            mean_bv = np.mean(block_vols)
            feat['vol_of_vol'] = np.std(block_vols) / mean_bv if mean_bv > 1e-15 else 0
        else:
            feat['vol_of_vol'] = np.nan
    else:
        feat['vol_of_vol'] = np.nan

    if len(returns_5m) > 3:
        r5_dm = returns_5m - returns_5m.mean()
        denom = np.sum(r5_dm**2)
        feat['autocorr_5m_lag1'] = np.sum(r5_dm[1:] * r5_dm[:-1]) / denom if denom > 1e-20 else 0
    else:
        feat['autocorr_5m_lag1'] = np.nan

    return feat


def compute_hour_features(df_1m_hour, df_5m_hour, df_15m_hour, rolling_sigma=None):
    if len(df_1m_hour) < 5:
        return None, None
    o = df_1m_hour['open'].values
    h = df_1m_hour['high'].values
    l = df_1m_hour['low'].values
    c = df_1m_hour['close'].values
    v = df_1m_hour['volume'].values.astype(np.float64)
    returns_1m = np.diff(np.log(c))
    if len(returns_1m) < 3:
        return None, None
    returns_5m = np.diff(np.log(df_5m_hour['close'].values)) if len(df_5m_hour) > 1 else np.array([])
    returns_15m = np.diff(np.log(df_15m_hour['close'].values)) if len(df_15m_hour) > 1 else np.array([])

    features = {}
    features.update(realized_vol_estimators(o, h, l, c))
    features.update(jump_detection(returns_1m))
    features.update(order_flow_features(o, h, l, c, v, rolling_sigma))
    features.update(information_theory_features(returns_1m, v[1:]))
    features.update(fractal_features(returns_1m, c))
    features.update(market_efficiency_features(returns_1m))
    features.update(tail_risk_features(returns_1m))
    features.update(acceleration_features(c, returns_1m))
    features.update(cross_timeframe_features(returns_1m, returns_5m, returns_15m))
    return features, returns_1m


def compute_trailing_features(df_hourly, returns_1m_dict):
    n = len(df_hourly)
    hurst_values = np.full(n, np.nan)
    fractal_values = np.full(n, np.nan)
    hours = df_hourly.index.tolist()

    for i in range(6, n):
        window_hours = hours[max(0, i-6):i]
        all_returns = []
        for h in window_hours:
            if h in returns_1m_dict:
                all_returns.append(returns_1m_dict[h])
        if all_returns:
            concat_ret = np.concatenate(all_returns)
            if len(concat_ret) >= 60:
                hurst_values[i] = hurst_rs(concat_ret, min_window=8)
                fractal_values[i] = fractal_dimension_higuchi(np.cumsum(concat_ret))

    df_hourly['hurst_6h'] = hurst_values
    df_hourly['fractal_dim_6h'] = fractal_values

    if 'order_imbalance_intensity' in df_hourly.columns:
        df_hourly['vpin_4h'] = df_hourly['order_imbalance_intensity'].rolling(4, min_periods=2).mean()
        df_hourly['vpin_12h'] = df_hourly['order_imbalance_intensity'].rolling(12, min_periods=4).mean()
    if 'vr_5' in df_hourly.columns:
        df_hourly['vr_5_ma12'] = df_hourly['vr_5'].rolling(12, min_periods=4).mean()
        df_hourly['hurst_change'] = df_hourly['hurst_6h'].diff(6)
    if 'rv_close' in df_hourly.columns:
        df_hourly['rv_zscore_24'] = (
            (df_hourly['rv_close'] - df_hourly['rv_close'].rolling(24, min_periods=8).mean())
            / df_hourly['rv_close'].rolling(24, min_periods=8).std().clip(lower=1e-15))
        df_hourly['jump_ratio_ma6'] = df_hourly['jump_ratio'].rolling(6, min_periods=2).mean()
    if 'kyle_lambda' in df_hourly.columns:
        df_hourly['kyle_lambda_ma12'] = df_hourly['kyle_lambda'].rolling(12, min_periods=4).mean()
        df_hourly['kyle_lambda_change'] = (df_hourly['kyle_lambda'].rolling(6, min_periods=2).mean() -
                                            df_hourly['kyle_lambda'].rolling(24, min_periods=8).mean())
    if 'entropy_norm' in df_hourly.columns:
        df_hourly['entropy_ma12'] = df_hourly['entropy_norm'].rolling(12, min_periods=4).mean()
        df_hourly['entropy_change_6h'] = df_hourly['entropy_norm'].diff(6)

    return df_hourly


def compute_contextual_features(df_features, df_1h, pair):
    """Compute features_6 contextual features on top of microstructure features.

    Mirrors exactly: notebooks_6/01_big_move_detection.ipynb → compute_extra_features()
    """
    pip = PIP_SIZE.get(pair, 0.0001)
    feat = pd.DataFrame(index=df_features.index)

    # Align 1H OHLCV to feature index
    df_1h = df_1h.reindex(df_features.index)
    o = df_1h['open']
    h = df_1h['high']
    l = df_1h['low']
    c = df_1h['close']
    v = df_1h['volume']

    # ── ATR CONTEXT ──
    tr = np.maximum(h - l, np.maximum(np.abs(h - c.shift(1)), np.abs(l - c.shift(1))))
    feat['atr_6'] = tr.rolling(6, min_periods=3).mean() / pip
    feat['atr_24'] = tr.rolling(24, min_periods=6).mean() / pip
    feat['atr_72'] = tr.rolling(72, min_periods=24).mean() / pip
    feat['atr_ratio_6_24'] = feat['atr_6'] / feat['atr_24'].clip(lower=1e-10)
    feat['atr_ratio_6_72'] = feat['atr_6'] / feat['atr_72'].clip(lower=1e-10)

    # ── RANGE POSITION ──
    high_24 = h.rolling(24, min_periods=6).max()
    low_24 = l.rolling(24, min_periods=6).min()
    range_24 = high_24 - low_24
    feat['range_pos_24'] = (c - low_24) / range_24.clip(lower=1e-10)
    feat['range_width_24'] = range_24 / pip

    high_48 = h.rolling(48, min_periods=12).max()
    low_48 = l.rolling(48, min_periods=12).min()
    range_48 = high_48 - low_48
    feat['range_pos_48'] = (c - low_48) / range_48.clip(lower=1e-10)
    feat['range_width_48'] = range_48 / pip

    atr_price = tr.rolling(24, min_periods=6).mean()
    feat['dist_from_24h_high'] = (high_24 - c) / atr_price.clip(lower=1e-10)
    feat['dist_from_24h_low'] = (c - low_24) / atr_price.clip(lower=1e-10)

    # ── SESSION FLAGS ──
    hour = df_features.index.hour
    feat['hour_sin'] = np.sin(2 * np.pi * hour / 24)
    feat['hour_cos'] = np.cos(2 * np.pi * hour / 24)
    feat['dow_sin'] = np.sin(2 * np.pi * df_features.index.dayofweek / 5)
    feat['dow_cos'] = np.cos(2 * np.pi * df_features.index.dayofweek / 5)
    feat['is_london'] = ((hour >= 7) & (hour < 16)).astype(np.float32)
    feat['is_ny'] = ((hour >= 13) & (hour < 22)).astype(np.float32)
    feat['is_overlap'] = ((hour >= 13) & (hour < 16)).astype(np.float32)
    feat['is_asia'] = ((hour >= 0) & (hour < 7)).astype(np.float32)

    # ── CANDLE STRUCTURE ──
    body = (c - o).abs()
    full_range = (h - l).clip(lower=1e-10)
    feat['body_ratio'] = body / full_range
    feat['upper_wick_ratio'] = (h - np.maximum(o, c)) / full_range
    feat['lower_wick_ratio'] = (np.minimum(o, c) - l) / full_range
    feat['candle_direction'] = np.sign(c - o)

    direction = feat['candle_direction']
    feat['consec_bullish'] = direction.rolling(6, min_periods=1).apply(
        lambda x: (x > 0).sum(), raw=True)
    feat['consec_bearish'] = direction.rolling(6, min_periods=1).apply(
        lambda x: (x < 0).sum(), raw=True)

    # ── FEATURE MOMENTUM (deltas of key microstructure features) ──
    key_features = ['hurst_6h', 'vr_5', 'entropy_norm', 'order_imbalance',
                    'kyle_lambda', 'jump_ratio', 'rv_yang_zhang']
    for col in key_features:
        if col in df_features.columns:
            series = df_features[col]
            feat[f'{col}_delta_3h'] = series.diff(3)
            feat[f'{col}_delta_6h'] = series.diff(6)
            feat[f'{col}_delta_12h'] = series.diff(12)

    # ── VOLUME CONTEXT ──
    feat['volume_ratio_6'] = v / v.rolling(6, min_periods=2).mean().clip(lower=1)
    feat['volume_ratio_24'] = v / v.rolling(24, min_periods=6).mean().clip(lower=1)

    # Convert to float32
    for col in feat.columns:
        feat[col] = feat[col].astype(np.float32)

    return feat


def compute_features_for_pair(pair, data):
    df_1m = data[pair]['1m']
    df_5m = data[pair]['5m']
    df_15m = data[pair]['15m']
    df_1h = data[pair]['1h']

    df_1m['hour'] = df_1m.index.floor('h')
    hours = sorted(df_1m['hour'].unique())

    all_features = []
    returns_1m_dict = {}
    rolling_sigma = None

    for hour_ts in hours:
        mask_1m = df_1m['hour'] == hour_ts
        df_1m_hour = df_1m[mask_1m]

        hour_end = hour_ts + pd.Timedelta(hours=1)
        df_5m_hour = df_5m[(df_5m.index >= hour_ts) & (df_5m.index < hour_end)]
        df_15m_hour = df_15m[(df_15m.index >= hour_ts) & (df_15m.index < hour_end)]

        feat, ret_1m = compute_hour_features(df_1m_hour, df_5m_hour, df_15m_hour, rolling_sigma)

        if feat is not None:
            feat['datetime'] = hour_ts
            all_features.append(feat)
            returns_1m_dict[hour_ts] = ret_1m

            hour_sigma = np.std(ret_1m)
            if rolling_sigma is None:
                rolling_sigma = hour_sigma
            else:
                rolling_sigma = 0.95 * rolling_sigma + 0.05 * hour_sigma

    if not all_features:
        return pd.DataFrame()

    df_features = pd.DataFrame(all_features).set_index('datetime').sort_index()
    df_features = compute_trailing_features(df_features, returns_1m_dict)
    df_features['pair'] = pair

    # Add contextual features (features_6)
    df_extra = compute_contextual_features(df_features, df_1h, pair)
    df_features = df_features.join(df_extra, how='left')

    close_1h = df_1h['close'].reindex(df_features.index, method='ffill')
    close_1h_3 = df_1h['close'].shift(-3).reindex(df_features.index, method='ffill')
    df_features['label_1H'] = np.log(close_1h_3 / close_1h)
    df_features['entry_price'] = close_1h
    df_features['exit_price'] = close_1h_3

    float_cols = df_features.select_dtypes(include=[np.float64]).columns
    df_features[float_cols] = df_features[float_cols].astype(np.float32)

    return df_features


# ──────────────────────────────────────────────
# MODEL INFERENCE (no rescue model)
# ──────────────────────────────────────────────
def run_inference(df_all):
    q50_bundle = joblib.load(MODELS_DIR / 'model_1H_Q50.joblib')
    q25_bundle = joblib.load(MODELS_DIR / 'model_1H_Q25.joblib')
    q75_bundle = joblib.load(MODELS_DIR / 'model_1H_Q75.joblib')
    meta_bundle = joblib.load(META_DIR / 'meta_confidence.joblib')

    feature_cols = q50_bundle['feature_cols']
    meta_feature_cols = meta_bundle['meta_feature_cols']

    X = df_all[feature_cols].groupby(df_all['pair']).ffill().fillna(0)

    q50_pred = q50_bundle['model'].predict(X)
    q25_pred = q25_bundle['model'].predict(X)
    q75_pred = q75_bundle['model'].predict(X)

    df_all['Q50'] = q50_pred
    df_all['Q25'] = q25_pred
    df_all['Q75'] = q75_pred
    df_all['abs_Q50'] = np.abs(q50_pred)
    df_all['pred_dir'] = np.sign(q50_pred)
    df_all['actual_dir'] = np.sign(df_all['label_1H'])

    df_all['Q50_oof'] = q50_pred
    df_all['Q25_oof'] = q25_pred
    df_all['Q75_oof'] = q75_pred
    df_all['iqr'] = q75_pred - q25_pred
    df_all['conf_ratio'] = np.abs(q50_pred) / np.clip(df_all['iqr'], 1e-10, None)

    tradeable_mask = df_all['abs_Q50'] > MIN_Q50_THRESHOLD
    df_tradeable = df_all[tradeable_mask].copy()

    df_all['meta_proba'] = np.nan
    if len(df_tradeable) > 0:
        X_meta = df_tradeable[meta_feature_cols].groupby(df_tradeable['pair']).ffill().fillna(0)
        meta_proba = meta_bundle['model'].predict_proba(X_meta)[:, 1]
        df_all.loc[tradeable_mask, 'meta_proba'] = meta_proba

    meta_accepted = tradeable_mask & (df_all['meta_proba'] > META_THRESHOLD)
    print(f'  Total hours: {len(df_all):,}')
    print(f'  Tradeable (|Q50|>0.5x): {tradeable_mask.sum():,}')
    print(f'  Meta-accepted (P>{META_THRESHOLD}): {meta_accepted.sum():,}')

    # Diagnostic: hourly distribution — base model vs meta model
    print(f'\n--- Hourly Distribution: Base Model vs Meta ---')
    print(f'{"Hour UTC":<10} {"Base(Q50)":>10} {"%":>6} {"Meta P>0.5":>12} {"%":>6} {"Meta/Base":>10}')
    print('-' * 58)
    base_hours = df_all[tradeable_mask].index.hour.value_counts().sort_index()
    meta_hours = df_all[meta_accepted].index.hour.value_counts().sort_index()
    base_total = tradeable_mask.sum()
    meta_total = meta_accepted.sum()
    for h in range(24):
        b = base_hours.get(h, 0)
        m = meta_hours.get(h, 0)
        b_pct = 100 * b / base_total if base_total > 0 else 0
        m_pct = 100 * m / meta_total if meta_total > 0 else 0
        ratio = m / b if b > 0 else 0
        if b > 0 or m > 0:
            print(f'{h:02d}:00     {b:>10} {b_pct:>5.1f}% {m:>12} {m_pct:>5.1f}% {ratio:>9.1%}')

    return df_all


# ──────────────────────────────────────────────
# CAPITAL SIMULATION
# ──────────────────────────────────────────────
def simulate_capital(df_all, backtest_start):
    """Simulate trading with real capital, margin, spreads, and lot sizing."""
    df = df_all[df_all.index >= backtest_start].copy()
    df = df[df['label_1H'].notna()].copy()

    # Dynamic config: meta P>0.5 + Q50>0.7x spread, OR bypass meta if Q50>1x spread
    HIGH_CONV_THRESHOLD = AVG_SPREAD * 1.0
    meta_path = (df['meta_proba'] > META_THRESHOLD) & (df['abs_Q50'] > MIN_Q50_THRESHOLD)
    high_conv_path = df['abs_Q50'] > HIGH_CONV_THRESHOLD
    df_signals = df[meta_path | high_conv_path].copy()

    # Apply 3H cooldown per pair
    df_signals = df_signals.sort_index()
    pair_unlock_time = {}
    keep = []
    for idx, row in df_signals.iterrows():
        pair = row['pair']
        unlock = pair_unlock_time.get(pair)
        if unlock is not None and idx < unlock:
            keep.append(False)
        else:
            keep.append(True)
            pair_unlock_time[pair] = idx + pd.Timedelta(hours=3)
    df_signals = df_signals[keep].copy()

    print(f'\n{"="*80}')
    print(f'CAPITAL SIMULATION')
    print(f'{"="*80}')
    print(f'Period: {df.index.min().date()} to {df.index.max().date()}')
    n_days = (df.index.max() - df.index.min()).days
    print(f'Duration: {n_days} days')
    print(f'Signals after cooldown: {len(df_signals)}')

    # Build set of signal hours for quick lookup
    signal_index = set(df_signals.index)

    equity = STARTING_CAPITAL
    peak_equity = equity
    max_drawdown = 0.0
    max_drawdown_pct = 0.0

    trades = []
    equity_curve = []
    open_positions = []

    all_hours = sorted(df.index.unique())

    for hour in all_hours:
        # ── Close matured positions ──
        still_open = []
        for pos in open_positions:
            if hour >= pos['close_at']:
                pnl = compute_pnl_usd(pos['pair'], pos['direction'], pos['lots'],
                                       pos['entry_price'], pos['exit_price'])
                open_hour_utc = pos['open_at'].hour
                spread_cost = compute_spread_cost_usd(pos['pair'], pos['lots'], pos['exit_price'], open_hour_utc)
                slippage_cost = compute_slippage_cost_usd(pos['pair'], pos['lots'], pos['exit_price'],
                                                          open_hour_utc, pos.get('realized_vol'))
                pnl -= spread_cost
                pnl -= slippage_cost

                equity += pnl  # P&L only (margin tracked separately)

                trades.append({
                    'datetime': pos['open_at'],
                    'close_at': hour,
                    'pair': pos['pair'],
                    'direction': 'BUY' if pos['direction'] == 1 else 'SELL',
                    'lots': pos['lots'],
                    'entry_price': pos['entry_price'],
                    'exit_price': pos['exit_price'],
                    'spread_cost': spread_cost,
                    'slippage_cost': slippage_cost,
                    'pnl_usd': pnl,
                    'equity_after': equity,
                    'meta_proba': pos['meta_proba'],
                    'q50': pos['q50'],
                })
            else:
                still_open.append(pos)
        open_positions = still_open

        # ── Open new positions ──
        if hour in signal_index:
            hour_signals = df_signals[df_signals.index == hour]

            for _, sig in hour_signals.iterrows():
                pair = sig['pair']
                entry_price = sig['entry_price']
                exit_price_2h = sig['exit_price']
                q50 = sig['Q50']
                meta_p = sig['meta_proba']
                direction = int(sig['pred_dir'])

                # Spread filter (time-aware)
                hour_utc = hour.hour
                if get_spread_points(pair, hour_utc) > MAX_SPREAD_POINTS:
                    continue

                # Check available margin
                locked_margin = sum(p['margin_used'] for p in open_positions)
                available = equity - locked_margin

                if available < MIN_MARGIN_TO_TRADE:
                    continue

                # ATR-based position sizing with market capacity cap
                atr_pips = float(sig.get('atr_24', np.nan))
                pip = PIP_SIZE.get(pair, 0.0001)
                quote_ccy = pair[3:]
                if pair in JPY_PAIRS:
                    pip_value_per_lot = LOT_UNITS * pip / entry_price  # USD per pip per lot
                elif quote_ccy == 'USD':
                    pip_value_per_lot = LOT_UNITS * pip
                else:
                    pip_value_per_lot = LOT_UNITS * pip * QUOTE_USD.get(quote_ccy, 1.0)

                risk_usd = equity * RISK_PER_TRADE
                if pd.notna(atr_pips) and atr_pips > 0 and pip_value_per_lot > 0:
                    target_lots = risk_usd / (atr_pips * pip_value_per_lot)
                else:
                    target_lots = 1.0  # fallback

                # Apply market capacity cap
                cap = CAPACITY_CAPS.get(pair, 10)
                target_lots = min(target_lots, cap)
                target_lots = max(round(target_lots, 1), 0.1)

                # Check margin and scale down if needed
                margin_needed = margin_required(pair, target_lots)
                if margin_needed > available:
                    base_ccy = pair[:3]
                    base_val = BASE_USD.get(base_ccy, 1.0)
                    affordable = available * LEVERAGE[pair] / (LOT_UNITS * base_val)
                    target_lots = max(round(min(target_lots, affordable), 1), 0.1)
                    margin_needed = margin_required(pair, target_lots)
                lots = target_lots

                if margin_needed > available:
                    continue

                # Get realized vol for slippage calc (annualized from rv_close)
                rv_raw = sig.get('rv_close', np.nan)
                ann_vol = np.sqrt(rv_raw * 252 * 24) if pd.notna(rv_raw) and rv_raw > 0 else None

                open_positions.append({
                    'pair': pair,
                    'direction': direction,
                    'lots': lots,
                    'entry_price': entry_price,
                    'exit_price': exit_price_2h,
                    'margin_used': margin_needed,
                    'open_at': hour,
                    'close_at': hour + pd.Timedelta(hours=3),
                    'meta_proba': meta_p,
                    'q50': q50,
                    'realized_vol': ann_vol,
                })

        # Track equity
        locked = sum(p['margin_used'] for p in open_positions)
        equity_curve.append({
            'datetime': hour,
            'equity': equity,
            'locked_margin': locked,
            'free_margin': equity - locked,
            'n_open': len(open_positions),
        })

        if equity > peak_equity:
            peak_equity = equity
        dd = peak_equity - equity
        dd_pct = dd / peak_equity if peak_equity > 0 else 0
        max_drawdown = max(max_drawdown, dd)
        max_drawdown_pct = max(max_drawdown_pct, dd_pct)

    # ── Print results ──
    df_trades = pd.DataFrame(trades)
    df_equity = pd.DataFrame(equity_curve)

    if len(df_trades) == 0:
        print('\nNo trades taken!')
        return df_trades, df_equity

    total_trades = len(df_trades)
    wins = (df_trades['pnl_usd'] > 0).sum()
    losses = total_trades - wins
    win_rate = wins / total_trades
    total_pnl = df_trades['pnl_usd'].sum()
    avg_pnl = df_trades['pnl_usd'].mean()
    avg_win = df_trades[df_trades['pnl_usd'] > 0]['pnl_usd'].mean() if wins > 0 else 0
    avg_loss = df_trades[df_trades['pnl_usd'] <= 0]['pnl_usd'].mean() if losses > 0 else 0
    total_spread_cost = df_trades['spread_cost'].sum()
    total_slippage_cost = df_trades['slippage_cost'].sum()
    gross_wins = df_trades[df_trades['pnl_usd'] > 0]['pnl_usd'].sum() if wins > 0 else 0
    gross_losses = abs(df_trades[df_trades['pnl_usd'] <= 0]['pnl_usd'].sum()) if losses > 0 else 0.01

    print(f'\n--- Account Summary ---')
    print(f'Starting capital:   ${STARTING_CAPITAL:>12,.2f}')
    print(f'Final equity:       ${equity:>12,.2f}')
    print(f'Total P&L:          ${total_pnl:>12,.2f} ({total_pnl/STARTING_CAPITAL*100:+.2f}%)')
    print(f'Max drawdown:       ${max_drawdown:>12,.2f} ({max_drawdown_pct*100:.2f}%)')
    print(f'Total spread cost:  ${total_spread_cost:>12,.2f}')
    print(f'Total slippage:     ${total_slippage_cost:>12,.2f}')
    print(f'Total exec cost:    ${total_spread_cost + total_slippage_cost:>12,.2f}')

    print(f'\n--- Trade Statistics ---')
    print(f'Total trades:       {total_trades}')
    print(f'Trades/day:         {total_trades/max(n_days,1):.2f}')
    print(f'Win / Loss:         {wins} / {losses}')
    print(f'Win rate:           {win_rate:.1%}')
    print(f'Avg win:            ${avg_win:>10,.2f}')
    print(f'Avg loss:           ${avg_loss:>10,.2f}')
    print(f'Avg P&L/trade:      ${avg_pnl:>10,.2f}')
    print(f'Profit factor:      {gross_wins / gross_losses:.2f}')

    # Sharpe
    df_trades['date'] = pd.to_datetime(df_trades['datetime']).dt.date
    daily_pnl = df_trades.groupby('date')['pnl_usd'].sum()
    if len(daily_pnl) > 1 and daily_pnl.std() > 0:
        sharpe = (daily_pnl.mean() / daily_pnl.std()) * np.sqrt(252)
        print(f'Sharpe (annualized): {sharpe:.2f}')

    # ── Per-pair breakdown ──
    print(f'\n--- Per-Pair Breakdown ---')
    print(f'{"Pair":<10} {"Trades":>7} {"WR":>7} {"Avg P&L":>10} {"Total P&L":>12} {"Spread$":>9} {"Slip$":>9}')
    print('-' * 72)
    for pair in sorted(df_trades['pair'].unique()):
        p = df_trades[df_trades['pair'] == pair]
        wr = (p['pnl_usd'] > 0).mean()
        flag = ' <<<' if p['pnl_usd'].sum() > 0 else ''
        print(f'{pair:<10} {len(p):>7} {wr:>6.1%} ${p["pnl_usd"].mean():>9,.2f} '
              f'${p["pnl_usd"].sum():>11,.2f} ${p["spread_cost"].sum():>8,.2f} ${p["slippage_cost"].sum():>8,.2f}{flag}')

    # ── Monthly breakdown ──
    print(f'\n--- Monthly Breakdown ---')
    df_trades['month'] = pd.to_datetime(df_trades['datetime']).dt.to_period('M')
    print(f'{"Month":<10} {"Trades":>7} {"WR":>7} {"P&L":>12} {"Cumulative":>12}')
    print('-' * 55)
    cum = 0
    for month in sorted(df_trades['month'].unique()):
        m = df_trades[df_trades['month'] == month]
        wr = (m['pnl_usd'] > 0).mean()
        mpnl = m['pnl_usd'].sum()
        cum += mpnl
        flag = ' <<<' if mpnl > 0 else ''
        print(f'{str(month):<10} {len(m):>7} {wr:>6.1%} ${mpnl:>11,.2f} ${cum:>11,.2f}{flag}')

    # ── Weekly breakdown ──
    print(f'\n--- Weekly Breakdown ---')
    df_trades['week'] = pd.to_datetime(df_trades['datetime']).dt.isocalendar().week.values
    df_trades['year'] = pd.to_datetime(df_trades['datetime']).dt.isocalendar().year.values
    df_trades['yearweek'] = df_trades['year'].astype(str) + '-W' + df_trades['week'].astype(str).str.zfill(2)
    print(f'{"Week":<12} {"Trades":>7} {"WR":>7} {"P&L":>12}')
    print('-' * 45)
    for week in sorted(df_trades['yearweek'].unique()):
        w = df_trades[df_trades['yearweek'] == week]
        wr = (w['pnl_usd'] > 0).mean()
        wpnl = w['pnl_usd'].sum()
        flag = ' <<<' if wpnl > 0 else ''
        print(f'{week:<12} {len(w):>7} {wr:>6.1%} ${wpnl:>11,.2f}{flag}')

    # ── Top wins & losses ──
    print(f'\n--- Top 5 Wins ---')
    for _, t in df_trades.nlargest(5, 'pnl_usd').iterrows():
        print(f'  {t["datetime"]}  {t["pair"]:>8}  {t["direction"]}  {t["lots"]} lots  '
              f'meta={t["meta_proba"]:.2f}  q50={t["q50"]:+.5f}  P&L=${t["pnl_usd"]:>+,.2f}')

    print(f'\n--- Top 5 Losses ---')
    for _, t in df_trades.nsmallest(5, 'pnl_usd').iterrows():
        print(f'  {t["datetime"]}  {t["pair"]:>8}  {t["direction"]}  {t["lots"]} lots  '
              f'meta={t["meta_proba"]:.2f}  q50={t["q50"]:+.5f}  P&L=${t["pnl_usd"]:>+,.2f}')

    # ── Hour-of-day analysis ──
    print(f'\n--- P&L by Hour of Day ---')
    df_trades['hour_of_day'] = pd.to_datetime(df_trades['datetime']).dt.hour
    hourly = df_trades.groupby('hour_of_day').agg(
        trades=('pnl_usd', 'count'),
        wr=('pnl_usd', lambda x: (x > 0).mean()),
        pnl=('pnl_usd', 'sum'),
    )
    print(f'{"Hour":>6} {"Trades":>7} {"WR":>7} {"P&L":>12}')
    print('-' * 40)
    for hour_val, row in hourly.iterrows():
        flag = ' <<<' if row['pnl'] > 0 else ''
        print(f'{hour_val:>6} {row["trades"]:>7} {row["wr"]:>6.1%} ${row["pnl"]:>11,.2f}{flag}')

    # ── Full Trade Log ──
    print(f'\n{"="*80}')
    print(f'FULL TRADE LOG')
    print(f'{"="*80}')
    print(f'{"#":>4} {"Entry Time":<20} {"Exit Time":<20} {"Pair":<8} {"Side":<5} {"Lots":>5} '
          f'{"Entry":>10} {"Exit":>10} {"Spread$":>8} {"Slip$":>7} {"P&L":>10} {"Result":<6} {"Equity":>12}')
    print('-' * 135)
    for i, (_, t) in enumerate(df_trades.iterrows(), 1):
        result = 'WIN' if t['pnl_usd'] > 0 else 'LOSS'
        entry_str = str(t['datetime'])[:16]
        exit_str = str(t['close_at'])[:16]
        print(f'{i:>4} {entry_str:<20} {exit_str:<20} {t["pair"]:<8} {t["direction"]:<5} {t["lots"]:>5} '
              f'{t["entry_price"]:>10.5f} {t["exit_price"]:>10.5f} ${t["spread_cost"]:>7.2f} '
              f'${t["slippage_cost"]:>6.2f} ${t["pnl_usd"]:>+9.2f} {result:<6} ${t["equity_after"]:>11,.2f}')

    return df_trades, df_equity


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────
async def fetch_real_spread(pair: str, signal_dt: datetime, n_quotes: int = 100) -> dict:
    """
    Fetch real bid/ask quotes from Polygon around signal_dt.
    Returns median spread, min spread, max spread at that timestamp.
    signal_dt is the hourly bar close = entry time.
    We fetch quotes in a 60-second window ending at signal_dt.
    """
    ticker = f'C:{pair}'
    # 60-second window ending at signal timestamp
    ts_start = int(signal_dt.timestamp() * 1e9)
    ts_end   = int((signal_dt + timedelta(seconds=60)).timestamp() * 1e9)

    try:
        async with httpx.AsyncClient(timeout=15) as client:
            r = await client.get(
                f'{REST_BASE}/v3/quotes/{ticker}',
                params={
                    'apiKey':              API_KEY,
                    'timestamp.gte':       ts_start,
                    'timestamp.lte':       ts_end,
                    'limit':               n_quotes,
                    'sort':                'timestamp',
                    'order':               'desc',
                }
            )
            if r.status_code != 200:
                return None
            results = r.json().get('results', [])
            if not results:
                return None

            spreads = [q['ask_price'] - q['bid_price'] for q in results
                       if q.get('ask_price') and q.get('bid_price') and q['ask_price'] > q['bid_price']]
            if not spreads:
                return None

            mid_prices = [(q['ask_price'] + q['bid_price']) / 2 for q in results
                          if q.get('ask_price') and q.get('bid_price')]

            return {
                'spread_median': float(np.median(spreads)),
                'spread_mean':   float(np.mean(spreads)),
                'spread_min':    float(np.min(spreads)),
                'spread_max':    float(np.max(spreads)),
                'spread_p75':    float(np.percentile(spreads, 75)),
                'mid_price':     float(np.mean(mid_prices)) if mid_prices else None,
                'n_quotes':      len(spreads),
            }
    except Exception:
        return None


async def fetch_spreads_for_signals(signals: list) -> dict:
    """
    Fetch real spreads for all signals concurrently (batched to avoid rate limits).
    signals: list of (pair, datetime) tuples
    Returns dict keyed by (pair, datetime) -> spread info
    """
    results = {}
    BATCH_SIZE = 20  # concurrent requests per batch

    print(f'\nFetching real bid/ask spreads for {len(signals)} signals...')
    for i in range(0, len(signals), BATCH_SIZE):
        batch = signals[i:i + BATCH_SIZE]
        tasks = [fetch_real_spread(pair, dt) for pair, dt in batch]
        batch_results = await asyncio.gather(*tasks)
        for (pair, dt), res in zip(batch, batch_results):
            results[(pair, dt)] = res
        print(f'  {min(i + BATCH_SIZE, len(signals))}/{len(signals)} done', end='\r')
        await asyncio.sleep(0.3)  # gentle rate limiting

    print(f'  Done. Got spreads for {sum(1 for v in results.values() if v is not None)}/{len(signals)} signals.')
    return results


async def main():
    t_start = time.time()

    data = await fetch_all_pairs()

    print(f'\nComputing microstructure features...')
    all_dfs = []
    for pair in PAIRS:
        if pair not in data:
            continue
        print(f'  {pair}...', end=' ', flush=True)
        t0 = time.time()
        df_feat = compute_features_for_pair(pair, data)
        elapsed = time.time() - t0
        print(f'{len(df_feat):,} hours ({elapsed:.1f}s)')
        all_dfs.append(df_feat)

    df_all = pd.concat(all_dfs).sort_index()
    print(f'Total feature rows: {len(df_all):,}')

    print(f'\nRunning model inference...')
    df_all = run_inference(df_all)

    backtest_start = df_all.index.min() + pd.Timedelta(days=WARMUP_DAYS)
    print(f'Backtest starts: {backtest_start.date()} (after {WARMUP_DAYS} days warmup)')

    # ── Get signals (meta-accepted rows after warmup) ──
    signal_mask = (
        (df_all.index >= backtest_start) &
        (df_all['abs_Q50'] > MIN_Q50_THRESHOLD) &
        (df_all['meta_proba'] > META_THRESHOLD)
    )
    df_signals = df_all[signal_mask].copy()
    signal_list = [(row['pair'], idx) for idx, row in df_signals.iterrows()]
    print(f'Signals to evaluate: {len(signal_list)}')

    # ── Fetch real spreads from Polygon ──
    real_spreads = await fetch_spreads_for_signals(signal_list)

    # ── Build results ──
    rows = []
    for idx, row in df_signals.iterrows():
        pair     = row['pair']
        hour_utc = idx.hour
        assumed  = spread_in_price(pair, hour_utc)  # hardcoded assumption

        real = real_spreads.get((pair, idx))
        real_spread = real['spread_median'] if real else None
        real_p75    = real['spread_p75']    if real else None
        n_quotes    = real['n_quotes']      if real else 0

        actual_return = row.get('label_1H', np.nan)
        pred_dir      = row['pred_dir']
        gross_pnl     = pred_dir * actual_return if not np.isnan(actual_return) else np.nan

        rows.append({
            'datetime':       idx,
            'pair':           pair,
            'hour':           hour_utc,
            'pred_dir':       pred_dir,
            'Q50':            row['Q50'],
            'meta_proba':     row['meta_proba'],
            'actual_return':  actual_return,
            'gross_pnl':      gross_pnl,
            'assumed_spread': assumed,
            'real_spread':    real_spread,
            'real_p75':       real_p75,
            'n_quotes':       n_quotes,
            'spread_ratio':   (real_spread / assumed) if real_spread and assumed > 0 else None,
            'net_pnl_assumed': gross_pnl - assumed    if not np.isnan(actual_return) else np.nan,
            'net_pnl_real':    gross_pnl - real_spread if (real_spread and not np.isnan(actual_return)) else np.nan,
        })

    df_results = pd.DataFrame(rows)

    # ── REPORT ──
    print(f'\n{"="*70}')
    print(f'PAPER TRADING — REAL SPREAD ANALYSIS')
    print(f'{"="*70}')
    print(f'Period:  {backtest_start.date()} to {df_all.index.max().date()}')
    print(f'Signals: {len(df_results)}')

    has_real = df_results['real_spread'].notna()
    print(f'Real spread data: {has_real.sum()}/{len(df_results)} signals ({has_real.mean():.0%})')

    # Overall spread comparison
    print(f'\n--- Spread Comparison (assumed vs real) ---')
    s = df_results[has_real]
    print(f'{"":20} {"Assumed":>12} {"Real median":>12} {"Real P75":>12} {"Ratio":>8}')
    print('-' * 68)
    print(f'{"ALL signals":<20} {s["assumed_spread"].mean():>12.6f} {s["real_spread"].mean():>12.6f} {s["real_p75"].mean():>12.6f} {s["spread_ratio"].mean():>8.2f}x')

    for hour in sorted(s['hour'].unique()):
        h = s[s['hour'] == hour]
        if len(h) < 3:
            continue
        flag = ' <-- KEY' if hour in (19, 20, 21, 22) else ''
        print(f'  Hour {hour:02d}:{" "*13} {h["assumed_spread"].mean():>12.6f} {h["real_spread"].mean():>12.6f} {h["real_p75"].mean():>12.6f} {h["spread_ratio"].mean():>8.2f}x{flag}')

    # Per-pair spread comparison
    print(f'\n--- Per-Pair Spread Comparison ---')
    print(f'{"Pair":<10} {"Signals":>8} {"Assumed":>12} {"Real median":>12} {"Real P75":>12} {"Ratio":>8}')
    print('-' * 60)
    for pair in sorted(s['pair'].unique()):
        p = s[s['pair'] == pair]
        print(f'{pair:<10} {len(p):>8} {p["assumed_spread"].mean():>12.6f} {p["real_spread"].mean():>12.6f} {p["real_p75"].mean():>12.6f} {p["spread_ratio"].mean():>8.2f}x')

    # EV with assumed vs real spreads
    has_return = df_results['actual_return'].notna() & has_real
    ev = df_results[has_return]
    correct = (ev['pred_dir'] == np.sign(ev['actual_return'])).mean()

    print(f'\n--- EV Analysis (signals with actual returns) ---')
    print(f'Signals with returns: {has_return.sum()}')
    print(f'Win rate: {correct:.1%}')
    print(f'Gross PnL/trade:          {ev["gross_pnl"].mean():>10.6f}')
    print(f'Net PnL/trade (assumed):  {ev["net_pnl_assumed"].mean():>10.6f}  total={ev["net_pnl_assumed"].sum():.4f}')
    print(f'Net PnL/trade (real):     {ev["net_pnl_real"].mean():>10.6f}  total={ev["net_pnl_real"].sum():.4f}')
    print(f'Net PnL/trade (real P75): {(ev["gross_pnl"] - ev["real_p75"]).mean():>10.6f}')

    # Hours 19-22 deep dive
    print(f'\n--- Hours 19-22 Deep Dive ---')
    print(f'{"Hour":<6} {"N":>5} {"WR":>7} {"Gross":>10} {"Net(assumed)":>14} {"Net(real)":>12} {"Ratio":>8}')
    print('-' * 65)
    for hour in [19, 20, 21, 22]:
        h = ev[ev['hour'] == hour]
        if len(h) == 0:
            continue
        wr  = (h['pred_dir'] == np.sign(h['actual_return'])).mean()
        print(f'{hour:<6} {len(h):>5} {wr:>6.1%} {h["gross_pnl"].mean():>10.6f} '
              f'{h["net_pnl_assumed"].mean():>14.6f} {h["net_pnl_real"].mean():>12.6f} '
              f'{h["spread_ratio"].mean():>8.2f}x')

    # Worst spread cases
    print(f'\n--- Top 10 Worst Real Spreads (vs assumption) ---')
    worst = s.nlargest(10, 'spread_ratio')[['datetime','pair','hour','assumed_spread','real_spread','spread_ratio','meta_proba','Q50']]
    print(worst.to_string(index=False))

    # ── SPREAD PROFILE: every hour 19-00, every pair, every day ──
    # Build list of (pair, datetime) for each hour in 19-23 + 00
    # Sample: one timestamp per hour per pair per day (at HH:59 = end of bar)
    print(f'\n{"="*70}')
    print(f'SPREAD PROFILE — Hours 19:00 to 00:00 (all days, all pairs)')
    print(f'{"="*70}')

    profile_hours = [19, 20, 21, 22, 23, 0]
    backtest_dates = pd.date_range(backtest_start.date(), df_all.index.max().date(), freq='D')
    profile_samples = []
    for day in backtest_dates:
        dow = day.dayofweek
        if dow == 5:  # Saturday — skip
            continue
        for h in profile_hours:
            # Skip Sunday before 21:00
            if dow == 6 and h < 21:
                continue
            # Use HH:59 as sample point (end of hour bar)
            if h == 0:
                sample_dt = datetime(day.year, day.month, day.day, 0, 1)
            else:
                sample_dt = datetime(day.year, day.month, day.day, h, 1)
            for pair in PAIRS:
                profile_samples.append((pair, sample_dt, h))

    # Deduplicate and fetch
    unique_samples = list({(p, dt): (p, dt, h) for p, dt, h in profile_samples}.values())
    print(f'Sampling {len(unique_samples)} timestamps ({len(backtest_dates)} days × {len(profile_hours)} hours × {len(PAIRS)} pairs)...')

    profile_results = {}
    BATCH_SIZE = 20
    for i in range(0, len(unique_samples), BATCH_SIZE):
        batch = unique_samples[i:i + BATCH_SIZE]
        tasks = [fetch_real_spread(pair, dt) for pair, dt, h in batch]
        batch_res = await asyncio.gather(*tasks)
        for (pair, dt, h), res in zip(batch, batch_res):
            profile_results[(pair, dt, h)] = res
        print(f'  {min(i + BATCH_SIZE, len(unique_samples))}/{len(unique_samples)}', end='\r')
        await asyncio.sleep(0.3)
    print(f'  Done.')

    # Build profile dataframe
    prof_rows = []
    for (pair, dt, h), res in profile_results.items():
        if res is None:
            continue
        prof_rows.append({
            'pair':           pair,
            'hour':           h,
            'datetime':       dt,
            'real_spread':    res['spread_median'],
            'real_p75':       res['spread_p75'],
            'assumed_spread': spread_in_price(pair, h),
            'n_quotes':       res['n_quotes'],
        })
    df_prof = pd.DataFrame(prof_rows)
    df_prof['ratio'] = df_prof['real_spread'] / df_prof['assumed_spread'].clip(lower=1e-10)

    # Per-hour summary across all pairs
    print(f'\n--- Spread Ratio by Hour (median across all pairs & days) ---')
    print(f'{"Hour":<6} {"Samples":>8} {"Assumed(avg)":>14} {"Real median":>12} {"Real P75":>10} {"Ratio":>8}')
    print('-' * 65)
    for h in profile_hours:
        hd = df_prof[df_prof['hour'] == h]
        if len(hd) == 0:
            continue
        flag = ' <<<' if hd['ratio'].median() > 2.0 else ''
        print(f'{h:02d}:00  {len(hd):>8} {hd["assumed_spread"].mean():>14.6f} '
              f'{hd["real_spread"].median():>12.6f} {hd["real_p75"].median():>10.6f} '
              f'{hd["ratio"].median():>8.2f}x{flag}')

    # Per-pair per-hour heatmap
    print(f'\n--- Spread Ratio Heatmap: Pair × Hour ---')
    header = f'{"Pair":<10}' + ''.join(f'  {h:02d}:00' for h in profile_hours)
    print(header)
    print('-' * (10 + 8 * len(profile_hours)))
    for pair in sorted(PAIRS):
        row = f'{pair:<10}'
        for h in profile_hours:
            cell = df_prof[(df_prof['pair'] == pair) & (df_prof['hour'] == h)]
            if len(cell) == 0:
                row += f'{"  N/A":>8}'
            else:
                ratio = cell['ratio'].median()
                row += f'  {ratio:>5.1f}x'
        print(row)

    elapsed_total = time.time() - t_start
    print(f'\nTotal runtime: {elapsed_total:.0f}s')


if __name__ == '__main__':
    asyncio.run(main())
