"""
Live Backtest v6.0 — Big Move Model (15 FX pairs)

Fetches recent 1-minute data from Polygon.io,
computes microstructure + contextual features,
runs LONG/SHORT big-move model predictions,
and simulates trades hour by hour.

No leakage: each hour uses only data available at that point in time.
Models were trained on data before 2024-06-30.
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

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
from dotenv import load_dotenv
load_dotenv()

API_KEY = os.getenv('POLYGON_S3_SECRET_KEY', '')
REST_BASE = 'https://api.polygon.io'

PAIRS = [
    'EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD', 'USDCAD', 'NZDUSD',
    'EURJPY', 'GBPJPY', 'EURGBP', 'EURAUD', 'AUDJPY', 'CADJPY', 'CHFJPY', 'AUDNZD',
]

PIP_SIZE = {
    'EURUSD': 0.0001, 'GBPUSD': 0.0001, 'AUDUSD': 0.0001, 'NZDUSD': 0.0001,
    'USDCAD': 0.0001, 'USDCHF': 0.0001, 'USDJPY': 0.01,
    'EURJPY': 0.01, 'GBPJPY': 0.01, 'EURGBP': 0.0001, 'EURAUD': 0.0001,
    'AUDJPY': 0.01, 'CADJPY': 0.01, 'CHFJPY': 0.01, 'AUDNZD': 0.0001,
}

MODELS_DIR = Path('backend/models_6')

# Fetch window
BACKTEST_DAYS = 450
WARMUP_DAYS = 10
TOTAL_FETCH_DAYS = BACKTEST_DAYS + WARMUP_DAYS

# Probability thresholds to evaluate
PROB_THRESHOLDS = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]

# Spread in pips for PnL
SPREAD_PIPS = 2.8

# Cooldown after a trade (hours)
COOLDOWN_HOURS = 4

# Forward horizon for simple PnL proxy (hours)
FORWARD_HOURS = [4, 8, 12, 24]

print(f'Live Backtest v6.0 — Big Move Model (15 pairs)')
print(f'  API key: {"OK" if API_KEY else "MISSING"}')
print(f'  Pairs: {len(PAIRS)}')
print(f'  Fetch window: {TOTAL_FETCH_DAYS} days ({WARMUP_DAYS} warmup + {BACKTEST_DAYS} backtest)')


# ──────────────────────────────────────────────
# DATA FETCHING
# ──────────────────────────────────────────────
async def fetch_bars(pair, multiplier, timespan, from_date, to_date, limit=50000):
    """Fetch OHLCV bars from Polygon REST API with pagination."""
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

    # Filter weekends
    df = df[~((df.index.dayofweek == 5) | ((df.index.dayofweek == 6) & (df.index.hour < 21)))]
    return df


async def fetch_all_pairs():
    """Fetch 1-min data for all pairs."""
    now = datetime.utcnow()
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

        # Resample to 5m, 15m, 1H
        df_5m = df_1m.resample('5min').agg({
            'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
        }).dropna()
        df_15m = df_1m.resample('15min').agg({
            'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
        }).dropna()
        df_1h = df_1m.resample('1h').agg({
            'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
        }).dropna()

        data[pair] = {
            '1m': df_1m, '5m': df_5m, '15m': df_15m, '1h': df_1h
        }
        print(f'{len(df_1m):,} bars ({elapsed:.1f}s) | {df_1m.index.min().date()} to {df_1m.index.max().date()}')

        await asyncio.sleep(0.5)  # rate limit

    return data


# ──────────────────────────────────────────────
# MICROSTRUCTURE FEATURE FUNCTIONS
# (exact replica from notebooks_5/01 + notebooks_6/01)
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
    iqr = np.percentile(returns, 75) - np.percentile(returns, 25)
    if iqr > 0:
        bin_width = 2 * iqr * n**(-1/3)
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


# ──────────────────────────────────────────────
# CONTEXTUAL FEATURES (from notebooks_6/01)
# ──────────────────────────────────────────────
def compute_contextual_features(df_micro, df_1h, pair):
    """Add ATR, range, session, candle, delta, and volume context features."""
    pip = PIP_SIZE[pair]
    o = df_1h['open'].reindex(df_micro.index, method='ffill')
    h = df_1h['high'].reindex(df_micro.index, method='ffill')
    l = df_1h['low'].reindex(df_micro.index, method='ffill')
    c = df_1h['close'].reindex(df_micro.index, method='ffill')
    v = df_1h['volume'].reindex(df_micro.index, method='ffill')

    # ATR context
    tr = np.maximum(h - l, np.maximum(np.abs(h - c.shift(1)), np.abs(l - c.shift(1))))
    df_micro['atr_6'] = tr.rolling(6, min_periods=3).mean() / pip
    df_micro['atr_24'] = tr.rolling(24, min_periods=6).mean() / pip
    df_micro['atr_72'] = tr.rolling(72, min_periods=24).mean() / pip
    df_micro['atr_ratio_6_24'] = df_micro['atr_6'] / df_micro['atr_24'].clip(lower=1e-10)
    df_micro['atr_ratio_6_72'] = df_micro['atr_6'] / df_micro['atr_72'].clip(lower=1e-10)

    # Range position
    high_24 = h.rolling(24, min_periods=6).max()
    low_24 = l.rolling(24, min_periods=6).min()
    range_24 = high_24 - low_24
    df_micro['range_pos_24'] = (c - low_24) / range_24.clip(lower=1e-10)
    df_micro['range_width_24'] = range_24 / pip

    high_48 = h.rolling(48, min_periods=12).max()
    low_48 = l.rolling(48, min_periods=12).min()
    range_48 = high_48 - low_48
    df_micro['range_pos_48'] = (c - low_48) / range_48.clip(lower=1e-10)
    df_micro['range_width_48'] = range_48 / pip

    atr_price = tr.rolling(24, min_periods=6).mean()
    df_micro['dist_from_24h_high'] = (high_24 - c) / atr_price.clip(lower=1e-10)
    df_micro['dist_from_24h_low'] = (c - low_24) / atr_price.clip(lower=1e-10)

    # Session flags (UTC)
    hour = df_micro.index.hour
    df_micro['hour_sin'] = np.sin(2 * np.pi * hour / 24)
    df_micro['hour_cos'] = np.cos(2 * np.pi * hour / 24)
    df_micro['dow_sin'] = np.sin(2 * np.pi * df_micro.index.dayofweek / 5)
    df_micro['dow_cos'] = np.cos(2 * np.pi * df_micro.index.dayofweek / 5)

    df_micro['is_london'] = ((hour >= 7) & (hour < 16)).astype(np.float32)
    df_micro['is_ny'] = ((hour >= 13) & (hour < 22)).astype(np.float32)
    df_micro['is_overlap'] = ((hour >= 13) & (hour < 16)).astype(np.float32)
    df_micro['is_asia'] = ((hour >= 0) & (hour < 7)).astype(np.float32)

    # Candle structure
    body = (c - o).abs()
    full_range = (h - l).clip(lower=1e-10)
    df_micro['body_ratio'] = body / full_range
    df_micro['upper_wick_ratio'] = (h - np.maximum(o, c)) / full_range
    df_micro['lower_wick_ratio'] = (np.minimum(o, c) - l) / full_range
    df_micro['candle_direction'] = np.sign(c - o)

    direction = df_micro['candle_direction']
    df_micro['consec_bullish'] = direction.rolling(6, min_periods=1).apply(lambda x: (x > 0).sum(), raw=True)
    df_micro['consec_bearish'] = direction.rolling(6, min_periods=1).apply(lambda x: (x < 0).sum(), raw=True)

    # Feature momentum (deltas)
    key_micro_features = [
        'hurst_6h', 'vr_5', 'entropy_norm', 'order_imbalance',
        'kyle_lambda', 'jump_ratio', 'rv_yang_zhang'
    ]
    for col in key_micro_features:
        if col in df_micro.columns:
            series = df_micro[col]
            df_micro[f'{col}_delta_3h'] = series.diff(3)
            df_micro[f'{col}_delta_6h'] = series.diff(6)
            df_micro[f'{col}_delta_12h'] = series.diff(12)

    # Volume context
    df_micro['volume_ratio_6'] = v / v.rolling(6, min_periods=2).mean().clip(lower=1)
    df_micro['volume_ratio_24'] = v / v.rolling(24, min_periods=6).mean().clip(lower=1)

    return df_micro


# ──────────────────────────────────────────────
# FEATURE PIPELINE
# ──────────────────────────────────────────────
def compute_features_for_pair(pair, data):
    """Compute microstructure + contextual features for one pair."""
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

    # Add contextual features
    df_features = compute_contextual_features(df_features, df_1h, pair)

    # Store forward prices for PnL calculation
    close_1h = df_1h['close'].reindex(df_features.index, method='ffill')
    for fwd_h in FORWARD_HOURS:
        fwd_close = df_1h['close'].shift(-fwd_h).reindex(df_features.index, method='ffill')
        df_features[f'fwd_return_{fwd_h}h'] = fwd_close - close_1h
    df_features['entry_price'] = close_1h

    float_cols = df_features.select_dtypes(include=[np.float64]).columns
    df_features[float_cols] = df_features[float_cols].astype(np.float32)

    return df_features


# ──────────────────────────────────────────────
# MODEL INFERENCE
# ──────────────────────────────────────────────
def run_inference(df_all):
    """Run LONG + SHORT big-move models."""

    long_bundle = joblib.load(MODELS_DIR / 'model_long.joblib')
    short_bundle = joblib.load(MODELS_DIR / 'model_short.joblib')

    feature_cols = long_bundle['feature_cols']

    # Prepare features (per-pair ffill)
    X = df_all[feature_cols].groupby(df_all['pair']).ffill().fillna(0)

    # Predict
    p_long = long_bundle['model'].predict_proba(X)[:, 1]
    p_short = short_bundle['model'].predict_proba(X)[:, 1]

    df_all['p_long'] = p_long
    df_all['p_short'] = p_short
    df_all['best_dir'] = np.where(p_long >= p_short, 'long', 'short')
    df_all['best_prob'] = np.maximum(p_long, p_short)
    df_all['pred_dir_sign'] = np.where(df_all['best_dir'] == 'long', 1.0, -1.0)

    print(f'\n  P(long)  — mean={p_long.mean():.3f}, median={np.median(p_long):.3f}, max={p_long.max():.3f}')
    print(f'  P(short) — mean={p_short.mean():.3f}, median={np.median(p_short):.3f}, max={p_short.max():.3f}')
    print(f'  Direction split — Long: {(df_all["best_dir"] == "long").mean():.1%}, Short: {(df_all["best_dir"] == "short").mean():.1%}')

    return df_all, feature_cols


# ──────────────────────────────────────────────
# TRADE SIMULATION
# ──────────────────────────────────────────────
def apply_cooldown(df, cooldown_hours=COOLDOWN_HOURS):
    """Per-pair cooldown: once a pair trades, it's locked for N hours."""
    df = df.sort_index()
    pair_unlock_time = {}
    keep = []
    for idx, row in df.iterrows():
        pair = row['pair']
        unlock = pair_unlock_time.get(pair)
        if unlock is not None and idx < unlock:
            keep.append(False)
        else:
            keep.append(True)
            pair_unlock_time[pair] = idx + pd.Timedelta(hours=cooldown_hours)
    return df[keep].copy()


def simulate_trades(df_all, backtest_start):
    """Simulate trades and report results."""
    df = df_all[df_all.index >= backtest_start].copy()

    # Need forward returns for PnL
    has_fwd = [c for c in df.columns if c.startswith('fwd_return_')]
    df = df.dropna(subset=has_fwd, how='all')

    print(f'\n{"="*80}')
    print(f'LIVE BACKTEST RESULTS — Big Move Model v6.0')
    print(f'{"="*80}')
    print(f'Period: {df.index.min().date()} to {df.index.max().date()}')
    n_days = (df.index.max() - df.index.min()).days
    print(f'Duration: {n_days} days')
    print(f'Total hours: {len(df):,}')
    print(f'Pairs: {df["pair"].nunique()}')

    # ── Results by threshold and forward horizon ──
    for fwd_h in FORWARD_HOURS:
        fwd_col = f'fwd_return_{fwd_h}h'
        if fwd_col not in df.columns:
            continue

        df_valid = df[df[fwd_col].notna()].copy()
        if len(df_valid) == 0:
            continue

        print(f'\n--- Forward Horizon: {fwd_h}H ---')
        print(f'{"Threshold":<12} {"Trades":>8} {"Tr/day":>8} {"Win%":>8} '
              f'{"AvgPnL(pip)":>12} {"TotalPnL":>10} {"Sharpe":>8}')
        print('-' * 75)

        for thresh in PROB_THRESHOLDS:
            candidates = df_valid[df_valid['best_prob'] > thresh]
            s = apply_cooldown(candidates)
            n = len(s)
            if n < 3:
                continue

            pip = s['pair'].map(PIP_SIZE)
            fwd_pips = s[fwd_col] / pip  # convert price return to pips
            pnl_pips = s['pred_dir_sign'] * fwd_pips - SPREAD_PIPS
            wr = (pnl_pips > 0).mean()
            avg_pnl = pnl_pips.mean()
            total_pnl = pnl_pips.sum()
            sharpe = (pnl_pips.mean() / pnl_pips.std()) * np.sqrt(252 * 24 / fwd_h) if pnl_pips.std() > 0 else 0
            flag = ' <<<' if avg_pnl > 0 else ''
            print(f'P > {thresh:.2f}    {n:>8,} {n/max(n_days,1):>8.2f} {wr:>7.1%} '
                  f'{avg_pnl:>11.1f}p {total_pnl:>10.1f}p {sharpe:>8.2f}{flag}')

    # ── Per-pair breakdown at best threshold ──
    best_fwd = 4  # default to 4H
    fwd_col = f'fwd_return_{best_fwd}h'
    best_thresh = 0.70  # high confidence

    df_valid = df[df[fwd_col].notna()].copy()
    candidates = df_valid[df_valid['best_prob'] > best_thresh]
    s = apply_cooldown(candidates)

    if len(s) > 0:
        print(f'\n--- Per-Pair Breakdown (P > {best_thresh:.2f}, {best_fwd}H forward) ---')
        print(f'{"Pair":<10} {"Trades":>8} {"Tr/day":>8} {"Win%":>8} '
              f'{"AvgPnL":>10} {"TotalPnL":>10} {"Long%":>8}')
        print('-' * 70)
        for pair in sorted(s['pair'].unique()):
            p = s[s['pair'] == pair]
            pip_val = PIP_SIZE[pair]
            fwd_pips = p[fwd_col] / pip_val
            pnl_pips = p['pred_dir_sign'] * fwd_pips - SPREAD_PIPS
            wr = (pnl_pips > 0).mean()
            long_pct = (p['best_dir'] == 'long').mean()
            flag = ' <<<' if pnl_pips.mean() > 0 else ''
            print(f'{pair:<10} {len(p):>8,} {len(p)/max(n_days,1):>8.2f} {wr:>7.1%} '
                  f'{pnl_pips.mean():>9.1f}p {pnl_pips.sum():>9.1f}p {long_pct:>7.1%}{flag}')

    # ── Weekly summary ──
    if len(s) > 0:
        s_copy = s.copy()
        pip_vals = s_copy['pair'].map(PIP_SIZE)
        s_copy['pnl_pips'] = s_copy['pred_dir_sign'] * s_copy[fwd_col] / pip_vals - SPREAD_PIPS
        s_copy['date'] = s_copy.index.date
        s_copy['week'] = pd.to_datetime(s_copy['date']).dt.isocalendar().week.values
        s_copy['year'] = pd.to_datetime(s_copy['date']).dt.isocalendar().year.values
        s_copy['yearweek'] = s_copy['year'].astype(str) + '-W' + s_copy['week'].astype(str).str.zfill(2)

        print(f'\n--- Weekly Summary (P > {best_thresh:.2f}, {best_fwd}H) ---')
        print(f'{"Week":<12} {"Trades":>8} {"Win%":>8} {"PnL":>10}')
        print('-' * 42)
        weekly = s_copy.groupby('yearweek').agg(
            trades=('pnl_pips', 'count'),
            wins=('pnl_pips', lambda x: (x > 0).sum()),
            pnl=('pnl_pips', 'sum')
        )
        for week, row in weekly.iterrows():
            wr = row['wins'] / row['trades'] if row['trades'] > 0 else 0
            flag = ' <<<' if row['pnl'] > 0 else ''
            print(f'{week:<12} {row["trades"]:>8} {wr:>7.1%} {row["pnl"]:>10.1f}p{flag}')

        pos_weeks = (weekly['pnl'] > 0).sum()
        total_weeks = len(weekly)
        print(f'\nPositive weeks: {pos_weeks}/{total_weeks} ({100*pos_weeks/total_weeks:.0f}%)')

    return df


def walk_forward_validation(df_all, backtest_start):
    """Walk-forward: split OOS into 90-day blocks."""
    df = df_all[df_all.index >= backtest_start].copy()

    fwd_col = 'fwd_return_4h'
    df = df[df[fwd_col].notna()].copy()

    print(f'\n{"="*80}')
    print(f'WALK-FORWARD VALIDATION (4H forward, 90-day blocks)')
    print(f'{"="*80}')

    BLOCK_DAYS = 90
    start = df.index.min()
    end = df.index.max()

    blocks = []
    block_start = start
    while block_start < end:
        block_end = block_start + pd.Timedelta(days=BLOCK_DAYS)
        if block_end > end:
            block_end = end + pd.Timedelta(hours=1)
        block_df = df[(df.index >= block_start) & (df.index < block_end)]
        if len(block_df) > 0:
            blocks.append((block_start, block_end, block_df))
        block_start = block_end

    for thresh in [0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
        print(f'\n--- P > {thresh:.2f} ---')
        print(f'{"Block":<25} {"Days":>5} {"Trades":>7} {"Tr/day":>7} '
              f'{"Win%":>7} {"AvgPnL":>9} {"TotalPnL":>9} {"Sharpe":>7}')
        print('-' * 85)

        block_results = []
        for i, (bs, be, bdf) in enumerate(blocks):
            candidates = bdf[bdf['best_prob'] > thresh]
            s = apply_cooldown(candidates)
            n = len(s)
            n_days_block = (be - bs).days

            if n < 2:
                block_label = f'{bs.date()} - {(be - pd.Timedelta(days=1)).date()}'
                print(f'{block_label:<25} {n_days_block:>5} {n:>7}')
                block_results.append({'trades': n, 'pnl': 0, 'wr': np.nan})
                continue

            pip_vals = s['pair'].map(PIP_SIZE)
            fwd_pips = s[fwd_col] / pip_vals
            pnl_pips = s['pred_dir_sign'] * fwd_pips - SPREAD_PIPS
            wr = (pnl_pips > 0).mean()
            avg_pnl = pnl_pips.mean()
            total_pnl = pnl_pips.sum()
            sharpe = (pnl_pips.mean() / pnl_pips.std()) * np.sqrt(252 * 24 / 4) if pnl_pips.std() > 0 else 0
            flag = ' <<<' if total_pnl > 0 else ''
            block_label = f'{bs.date()} - {(be - pd.Timedelta(days=1)).date()}'
            print(f'{block_label:<25} {n_days_block:>5} {n:>7} {n/max(n_days_block,1):>7.2f} '
                  f'{wr:>6.1%} {avg_pnl:>8.1f}p {total_pnl:>8.1f}p {sharpe:>7.2f}{flag}')
            block_results.append({'trades': n, 'pnl': total_pnl, 'wr': wr})

        valid = [r for r in block_results if not np.isnan(r.get('wr', np.nan))]
        if len(valid) >= 2:
            n_pos = sum(1 for r in valid if r['pnl'] > 0)
            total_pnl = sum(r['pnl'] for r in valid)
            print(f'  AGGREGATE: {sum(r["trades"] for r in valid)} trades, '
                  f'PnL={total_pnl:.1f}p, Blocks positive: {n_pos}/{len(valid)}')


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────
async def main():
    t_start = time.time()

    # Step 1: Fetch data
    data = await fetch_all_pairs()

    # Step 2: Compute features
    print(f'\nComputing features...')
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

    # Step 3: Run inference
    print(f'\nRunning model inference...')
    df_all, feature_cols = run_inference(df_all)

    # Step 4: Determine backtest start
    backtest_start = df_all.index.min() + pd.Timedelta(days=WARMUP_DAYS)
    print(f'Backtest starts: {backtest_start.date()} (after {WARMUP_DAYS} days warmup)')

    # Step 5: Simulate and report
    simulate_trades(df_all, backtest_start)

    # Step 6: Walk-forward
    walk_forward_validation(df_all, backtest_start)

    elapsed_total = time.time() - t_start
    print(f'\nTotal runtime: {elapsed_total:.0f}s')


if __name__ == '__main__':
    asyncio.run(main())
