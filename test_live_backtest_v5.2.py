"""
Live Backtest v5.2 — 15-pair model, 4H target (7 majors + 8 crosses)

Fetches the last ~100 days of 1-minute data from Polygon.io,
computes microstructure features, runs Q50 + meta-model predictions (4H horizon),
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
    # Majors
    'EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD', 'USDCAD', 'NZDUSD',
    # Crosses
    'EURJPY', 'GBPJPY', 'EURGBP', 'EURAUD', 'AUDJPY', 'CADJPY', 'CHFJPY', 'AUDNZD',
]

MODELS_DIR = Path('backend/models_5.1/3_quants')
META_DIR   = Path('backend/models_5.1/meta')
RESCUE_DIR = Path('backend/models_5.1/rescue')

AVG_SPREAD = 0.00028
MIN_Q50_THRESHOLD = AVG_SPREAD * 0.5

# Fetch ~460 days: 450 for backtest + 10 for trailing features warmup
BACKTEST_DAYS = 450
WARMUP_DAYS = 10
TOTAL_FETCH_DAYS = BACKTEST_DAYS + WARMUP_DAYS

# Meta-model thresholds to evaluate
META_THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]

# Rescue-model thresholds to evaluate
RESCUE_THRESHOLDS = [0.50, 0.55, 0.60, 0.65]

print(f'Live Backtest v5.2 — 15-pair model (4H target)')
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

        # Resample to 5m and 15m
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
# (exact replica from notebooks_5/01_microstructure_features.ipynb)
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
# FEATURE PIPELINE
# ──────────────────────────────────────────────
def compute_features_for_pair(pair, data):
    """Compute microstructure features for one pair from fetched data."""
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

    # Compute labels from 1H closes (4H forward return)
    close_1h = df_1h['close'].reindex(df_features.index, method='ffill')
    close_1h_4 = df_1h['close'].shift(-4).reindex(df_features.index, method='ffill')
    df_features['label_1H'] = np.log(close_1h_4 / close_1h)  # named label_1H to keep downstream code working

    # Also store entry/exit prices
    df_features['entry_price'] = close_1h
    df_features['exit_price'] = close_1h_4

    float_cols = df_features.select_dtypes(include=[np.float64]).columns
    df_features[float_cols] = df_features[float_cols].astype(np.float32)

    return df_features


# ──────────────────────────────────────────────
# MODEL INFERENCE
# ──────────────────────────────────────────────
def run_inference(df_all):
    """Run Q25/Q50/Q75 + meta-model + rescue-model on feature data."""

    # Load models — all quantile models are in 3_quants/ for v5.1
    q50_bundle = joblib.load(MODELS_DIR / 'model_1H_Q50.joblib')
    q25_bundle = joblib.load(MODELS_DIR / 'model_1H_Q25.joblib')
    q75_bundle = joblib.load(MODELS_DIR / 'model_1H_Q75.joblib')
    meta_bundle = joblib.load(META_DIR / 'meta_confidence.joblib')
    rescue_bundle = joblib.load(RESCUE_DIR / 'rescue_model.joblib')

    feature_cols = q50_bundle['feature_cols']
    meta_feature_cols = meta_bundle['meta_feature_cols']
    rescue_feature_cols = rescue_bundle['rescue_feature_cols']
    pair_map = rescue_bundle['pair_map']

    # Prepare features (per-pair ffill to match live inference)
    X = df_all[feature_cols].groupby(df_all['pair']).ffill().fillna(0)

    # Q50/Q25/Q75 predictions
    q50_pred = q50_bundle['model'].predict(X)
    q25_pred = q25_bundle['model'].predict(X)
    q75_pred = q75_bundle['model'].predict(X)

    df_all['Q50'] = q50_pred
    df_all['Q25'] = q25_pred
    df_all['Q75'] = q75_pred
    df_all['abs_Q50'] = np.abs(q50_pred)
    df_all['pred_dir'] = np.sign(q50_pred)
    df_all['actual_dir'] = np.sign(df_all['label_1H'])

    # Meta-model features
    df_all['Q50_oof'] = q50_pred  # named _oof to match meta_feature_cols
    df_all['Q25_oof'] = q25_pred
    df_all['Q75_oof'] = q75_pred
    df_all['abs_Q50'] = np.abs(q50_pred)
    df_all['iqr'] = q75_pred - q25_pred
    df_all['conf_ratio'] = np.abs(q50_pred) / np.clip(df_all['iqr'], 1e-10, None)

    # Filter tradeable hours and run meta-model
    tradeable_mask = df_all['abs_Q50'] > MIN_Q50_THRESHOLD
    df_tradeable = df_all[tradeable_mask].copy()

    df_all['meta_proba'] = np.nan
    if len(df_tradeable) > 0:
        X_meta = df_tradeable[meta_feature_cols].groupby(df_tradeable['pair']).ffill().fillna(0)
        meta_proba = meta_bundle['model'].predict_proba(X_meta)[:, 1]
        df_all.loc[tradeable_mask, 'meta_proba'] = meta_proba

    # ── Rescue model: score rejected signals ──
    # Rejected = not tradeable (|Q50| < 0.5x spread) OR tradeable but meta < 0.55
    meta_accepted_mask = tradeable_mask & (df_all['meta_proba'] > 0.55)
    rejected_mask = ~meta_accepted_mask

    # Build contextual features for rescue model
    df_all['pair_id'] = df_all['pair'].map(pair_map).fillna(0).astype(int)
    df_all['hour_sin'] = np.sin(2 * np.pi * df_all.index.hour / 24)
    df_all['hour_cos'] = np.cos(2 * np.pi * df_all.index.hour / 24)
    df_all['dow_sin'] = np.sin(2 * np.pi * df_all.index.dayofweek / 5)
    df_all['dow_cos'] = np.cos(2 * np.pi * df_all.index.dayofweek / 5)

    # Cross-pair agreement: for each timestamp, how many pairs agree on direction
    ts_groups = df_all.groupby(df_all.index)
    n_positive = ts_groups['pred_dir'].transform(lambda x: (x > 0).sum())
    n_negative = ts_groups['pred_dir'].transform(lambda x: (x < 0).sum())
    n_pairs = ts_groups['pred_dir'].transform('count')
    df_all['n_positive'] = n_positive
    df_all['n_negative'] = n_negative
    df_all['n_pairs'] = n_pairs
    df_all['cross_pair_agree'] = np.where(
        df_all['pred_dir'] > 0,
        df_all['n_positive'] / df_all['n_pairs'],
        df_all['n_negative'] / df_all['n_pairs']
    )

    # is_tradeable_zone: whether |Q50| > spread threshold
    df_all['is_tradeable_zone'] = tradeable_mask.astype(int)

    # Fill meta_proba for rescue (non-tradeable rows get NaN, fill with 0.5)
    df_all['meta_proba_rescue'] = df_all['meta_proba'].fillna(0.5)

    # Score rejected signals with rescue model
    df_all['rescue_proba'] = np.nan
    df_rejected = df_all[rejected_mask].copy()

    if len(df_rejected) > 0:
        # The rescue model expects 'meta_proba' in its feature cols — use the filled version
        df_rejected['meta_proba'] = df_rejected['meta_proba_rescue']
        X_rescue = df_rejected[rescue_feature_cols].groupby(df_rejected['pair']).ffill().fillna(0)
        rescue_proba = rescue_bundle['model'].predict_proba(X_rescue)[:, 1]
        df_all.loc[rejected_mask, 'rescue_proba'] = rescue_proba

    print(f'  Meta-accepted (P>0.55 & |Q50|>0.5x): {meta_accepted_mask.sum():,}')
    print(f'  Rejected (rescue candidates): {rejected_mask.sum():,}')
    print(f'  Rescue scored: {df_all["rescue_proba"].notna().sum():,}')

    return df_all


# ──────────────────────────────────────────────
# TRADE SIMULATION
# ──────────────────────────────────────────────
def apply_4h_cooldown(df):
    """Filter trades: once a pair trades, it's locked for 4 hours.
    Model checks every hour, but skips pairs with active positions."""
    df = df.sort_index()
    pair_unlock_time = {}  # pair -> earliest datetime it can trade again
    keep = []
    for idx, row in df.iterrows():
        pair = row['pair']
        unlock = pair_unlock_time.get(pair)
        if unlock is not None and idx < unlock:
            keep.append(False)
        else:
            keep.append(True)
            pair_unlock_time[pair] = idx + pd.Timedelta(hours=4)
    return df[keep].copy()


def simulate_trades(df_all, backtest_start):
    """Simulate trades and report results."""
    # Only use backtest period (skip warmup)
    df = df_all[df_all.index >= backtest_start].copy()
    df = df[df['label_1H'].notna()].copy()

    print(f'\n{"="*80}')
    print(f'LIVE BACKTEST RESULTS')
    print(f'{"="*80}')
    print(f'Period: {df.index.min().date()} to {df.index.max().date()}')
    n_days = (df.index.max() - df.index.min()).days
    print(f'Duration: {n_days} days')
    print(f'Total hours: {len(df):,}')
    print(f'Pairs: {df["pair"].nunique()}')

    # ── Q50-only baselines ──
    print(f'\n--- Q50-Only Baselines ---')
    print(f'{"Filter":<20} {"Trades":>8} {"Tr/day":>8} {"WR":>8} {"EV/trade":>12} {"TotalPnL":>10} {"Sharpe":>8}')
    print('-' * 75)

    for name, thresh in [('|Q50|>0.5x', AVG_SPREAD*0.5), ('|Q50|>1x', AVG_SPREAD),
                         ('|Q50|>2x', AVG_SPREAD*2), ('|Q50|>3x', AVG_SPREAD*3)]:
        s = apply_4h_cooldown(df[df['abs_Q50'] > thresh])
        n = len(s)
        if n < 3:
            continue
        pnl = s['pred_dir'] * s['label_1H'] - AVG_SPREAD
        wr = (s['pred_dir'] == s['actual_dir']).mean()
        sharpe = (pnl.mean() / pnl.std()) * np.sqrt(252 * 24) if pnl.std() > 0 else 0
        print(f'{name:<20} {n:>8,} {n/max(n_days,1):>8.2f} {wr:>7.1%} {pnl.mean():>12.6f} {pnl.sum():>10.4f} {sharpe:>8.2f}')

    # ── Meta-model results ──
    df_tradeable = df[df['meta_proba'].notna()].copy()
    print(f'\n--- Meta-Model Results ---')
    print(f'Tradeable hours (|Q50|>0.5x with meta score): {len(df_tradeable):,}')
    if len(df_tradeable) > 0:
        print(f'Meta probability: mean={df_tradeable["meta_proba"].mean():.3f}, '
              f'median={df_tradeable["meta_proba"].median():.3f}')

    print(f'\n{"Threshold":<12} {"Trades":>8} {"Tr/day":>8} {"WR":>8} {"EV/trade":>12} {"TotalPnL":>10} {"Sharpe":>8}')
    print('-' * 75)

    for thresh in META_THRESHOLDS:
        s = apply_4h_cooldown(df_tradeable[df_tradeable['meta_proba'] > thresh])
        n = len(s)
        if n < 3:
            continue
        pnl = s['pred_dir'] * s['label_1H'] - AVG_SPREAD
        wr = (s['pred_dir'] == s['actual_dir']).mean()
        sharpe = (pnl.mean() / pnl.std()) * np.sqrt(252 * 24) if pnl.std() > 0 else 0
        flag = ' <<<' if wr >= 0.80 and n >= 10 else (' <<' if wr >= 0.70 else '')
        print(f'P > {thresh:.2f}    {n:>8,} {n/max(n_days,1):>8.2f} {wr:>7.1%} {pnl.mean():>12.6f} {pnl.sum():>10.4f} {sharpe:>8.2f}{flag}')

    # ── Rescue-only results ──
    df_rescued = df[df['rescue_proba'].notna()].copy()
    print(f'\n--- Rescue Model Results ---')
    print(f'Rejected hours with rescue score: {len(df_rescued):,}')

    print(f'\n{"Threshold":<12} {"Rescued":>8} {"R/day":>8} {"WR":>8} {"EV/trade":>12} {"TotalPnL":>10} {"Sharpe":>8}')
    print('-' * 75)
    for thresh in RESCUE_THRESHOLDS:
        s = apply_4h_cooldown(df_rescued[df_rescued['rescue_proba'] > thresh])
        n = len(s)
        if n < 3:
            continue
        pnl = s['pred_dir'] * s['label_1H'] - AVG_SPREAD
        wr = (s['pred_dir'] == s['actual_dir']).mean()
        sharpe = (pnl.mean() / pnl.std()) * np.sqrt(252 * 24) if pnl.std() > 0 else 0
        flag = ' <<<' if wr >= 0.65 and pnl.sum() > 0 else ''
        print(f'R > {thresh:.2f}    {n:>8,} {n/max(n_days,1):>8.2f} {wr:>7.1%} {pnl.mean():>12.6f} {pnl.sum():>10.4f} {sharpe:>8.2f}{flag}')

    # ── Combined pipeline: Meta P>0.55 + Rescue ──
    META_BEST = 0.55
    meta_trades = apply_4h_cooldown(df_tradeable[df_tradeable['meta_proba'] > META_BEST])
    meta_pnl_series = meta_trades['pred_dir'] * meta_trades['label_1H'] - AVG_SPREAD

    print(f'\n--- Combined Pipeline: Meta P>{META_BEST} + Rescue ---')
    print(f'{"Rescue Thresh":<15} {"Meta Tr":>8} {"Rescued":>8} {"Total":>8} {"Tr/day":>8} {"WR":>8} {"EV/trade":>12} {"TotalPnL":>10} {"vs Meta":>10}')
    print('-' * 100)

    for rthresh in RESCUE_THRESHOLDS:
        rescued_trades = apply_4h_cooldown(df_rescued[df_rescued['rescue_proba'] > rthresh])
        n_rescued = len(rescued_trades)
        if n_rescued < 3:
            continue
        rescued_pnl = rescued_trades['pred_dir'] * rescued_trades['label_1H'] - AVG_SPREAD

        # Combined
        combined_pnl = pd.concat([meta_pnl_series, rescued_pnl])
        total_n = len(meta_trades) + n_rescued
        total_pnl = combined_pnl.sum()
        wr_combined = (pd.concat([
            (meta_trades['pred_dir'] == meta_trades['actual_dir']),
            (rescued_trades['pred_dir'] == rescued_trades['actual_dir'])
        ])).mean()

        delta = rescued_pnl.sum()
        flag = ' <<<' if delta > 0 else ''
        print(f'R > {rthresh:.2f}       {len(meta_trades):>8,} {n_rescued:>8,} {total_n:>8,} '
              f'{total_n/max(n_days,1):>8.2f} {wr_combined:>7.1%} {combined_pnl.mean():>12.6f} '
              f'{total_pnl:>10.4f} {delta:>+10.4f}{flag}')

    # ── Per-pair breakdown for best rescue threshold ──
    RESCUE_BEST = 0.55
    rescued_best = apply_4h_cooldown(df_rescued[df_rescued['rescue_proba'] > RESCUE_BEST])
    if len(rescued_best) > 0:
        print(f'\n--- Rescued Signals at R>{RESCUE_BEST} — Per-Pair (with 4H cooldown) ---')
        print(f'{"Pair":<10} {"Rescued":>8} {"R/day":>8} {"WR":>8} {"EV/trade":>12} {"TotalPnL":>10}')
        print('-' * 60)
        for pair in sorted(rescued_best['pair'].unique()):
            p = rescued_best[rescued_best['pair'] == pair]
            pnl = p['pred_dir'] * p['label_1H'] - AVG_SPREAD
            wr = (p['pred_dir'] == p['actual_dir']).mean()
            flag = ' <<<' if pnl.sum() > 0 else ''
            print(f'{pair:<10} {len(p):>8,} {len(p)/max(n_days,1):>8.2f} {wr:>7.1%} '
                  f'{pnl.mean():>12.6f} {pnl.sum():>10.4f}{flag}')

    # ── Strategy: take all trades where meta P > 0.50 ──
    META_THRESH = 0.50
    filtered = apply_4h_cooldown(df_tradeable[df_tradeable['meta_proba'] > META_THRESH])
    if len(filtered) > 0:
        print(f'\n{"="*80}')
        print(f'STRATEGY: Q50>0.5x spread + Meta P>{META_THRESH}')
        print(f'{"="*80}')
        filtered_copy = filtered.copy()
        filtered_copy['pnl'] = filtered_copy['pred_dir'] * filtered_copy['label_1H'] - AVG_SPREAD
        filtered_copy['correct'] = (filtered_copy['pred_dir'] == filtered_copy['actual_dir']).astype(int)

        total_trades = len(filtered_copy)
        total_wins = filtered_copy['correct'].sum()
        total_pnl = filtered_copy['pnl'].sum()
        total_wr = total_wins / total_trades
        ev = filtered_copy['pnl'].mean()

        print(f'\nTotal trades:  {total_trades}')
        print(f'Wins / Losses: {total_wins} / {total_trades - total_wins}')
        print(f'Win Rate:      {total_wr:.1%}')
        print(f'EV per trade:  {ev:.6f}')
        print(f'Total PnL:     {total_pnl:.4f}')
        print(f'Trades/day:    {total_trades/max(n_days,1):.2f}')

        # ── Per-pair breakdown ──
        print(f'\n--- Per-Pair Breakdown ---')
        print(f'{"Pair":<10} {"Trades":>8} {"Tr/day":>8} {"WR":>8} {"EV/trade":>12} {"TotalPnL":>10}')
        print('-' * 60)
        for pair in sorted(filtered_copy['pair'].unique()):
            p = filtered_copy[filtered_copy['pair'] == pair]
            wr = p['correct'].mean()
            flag = ' <<<' if p['pnl'].mean() > 0 else ''
            print(f'{pair:<10} {len(p):>8,} {len(p)/max(n_days,1):>8.2f} {wr:>7.1%} '
                  f'{p["pnl"].mean():>12.6f} {p["pnl"].sum():>10.4f}{flag}')

        # ── Day-by-day ──
        print(f'\n--- Day-by-Day ---')
        print(f'{"Date":<12} {"Trades":>8} {"Wins":>6} {"WR":>8} {"PnL":>10}')
        print('-' * 50)
        filtered_copy['date'] = filtered_copy.index.date

        daily = filtered_copy.groupby('date').agg(
            trades=('pnl', 'count'),
            wins=('correct', 'sum'),
            pnl=('pnl', 'sum')
        )
        cum_pnl = 0
        for date, row in daily.iterrows():
            wr = row['wins'] / row['trades'] if row['trades'] > 0 else 0
            cum_pnl += row['pnl']
            flag = ' <<<' if row['pnl'] > 0 else ''
            print(f'{str(date):<12} {row["trades"]:>8} {row["wins"]:>6} {wr:>7.1%} {row["pnl"]:>10.4f}  cum:{cum_pnl:>10.4f}{flag}')

        # ── Weekly summary ──
        print(f'\n--- Weekly Summary ---')
        filtered_copy['week'] = pd.to_datetime(filtered_copy['date']).dt.isocalendar().week.values
        filtered_copy['year'] = pd.to_datetime(filtered_copy['date']).dt.isocalendar().year.values
        filtered_copy['yearweek'] = filtered_copy['year'].astype(str) + '-W' + filtered_copy['week'].astype(str).str.zfill(2)
        weekly = filtered_copy.groupby('yearweek').agg(
            trades=('pnl', 'count'),
            wins=('correct', 'sum'),
            pnl=('pnl', 'sum')
        )
        print(f'{"Week":<12} {"Trades":>8} {"Wins":>6} {"WR":>8} {"PnL":>10}')
        print('-' * 50)
        for week, row in weekly.iterrows():
            wr = row['wins'] / row['trades'] if row['trades'] > 0 else 0
            flag = ' <<<' if row['pnl'] > 0 else ''
            print(f'{week:<12} {row["trades"]:>8} {row["wins"]:>6} {wr:>7.1%} {row["pnl"]:>10.4f}{flag}')

    return df


def walk_forward_validation(df_all, backtest_start):
    """Purged walk-forward evaluation: split OOS into non-overlapping blocks."""
    df = df_all[df_all.index >= backtest_start].copy()
    df = df[df['label_1H'].notna()].copy()

    print(f'\n{"="*80}')
    print(f'WALK-FORWARD VALIDATION')
    print(f'{"="*80}')

    # Split into 3-month (90-day) non-overlapping blocks
    BLOCK_DAYS = 90
    start = df.index.min()
    end = df.index.max()
    total_days = (end - start).days

    blocks = []
    block_start = start
    while block_start < end:
        block_end = block_start + pd.Timedelta(days=BLOCK_DAYS)
        if block_end > end:
            block_end = end + pd.Timedelta(hours=1)  # include last hour
        block_df = df[(df.index >= block_start) & (df.index < block_end)]
        if len(block_df) > 0:
            blocks.append((block_start, block_end, block_df))
        block_start = block_end

    print(f'Period: {start.date()} to {end.date()} ({total_days} days)')
    print(f'Block size: {BLOCK_DAYS} days')
    print(f'Number of blocks: {len(blocks)}')

    # ── Test each meta threshold across all blocks ──
    for thresh in META_THRESHOLDS:
        print(f'\n--- Meta P > {thresh:.2f} ---')
        print(f'{"Block":<25} {"Days":>5} {"Trades":>7} {"Tr/day":>7} {"WR":>7} {"EV/trade":>11} {"PnL":>9} {"Sharpe":>7}')
        print('-' * 85)

        block_results = []
        for i, (bs, be, bdf) in enumerate(blocks):
            bdf_tradeable = bdf[bdf['meta_proba'].notna()]
            s = apply_4h_cooldown(bdf_tradeable[bdf_tradeable['meta_proba'] > thresh])
            n = len(s)
            n_days_block = (be - bs).days
            if n < 2:
                print(f'{str(bs.date())} - {str((be - pd.Timedelta(days=1)).date()):<11} {n_days_block:>5} {n:>7} {"":>7} {"":>7} {"":>11} {"":>9} {"":>7}')
                block_results.append({'trades': n, 'wr': np.nan, 'ev': np.nan, 'pnl': 0, 'sharpe': np.nan})
                continue
            pnl = s['pred_dir'] * s['label_1H'] - AVG_SPREAD
            wr = (s['pred_dir'] == s['actual_dir']).mean()
            sharpe = (pnl.mean() / pnl.std()) * np.sqrt(252 * 24) if pnl.std() > 0 else 0
            flag = ' <<<' if wr >= 0.70 else (' <<' if wr >= 0.60 else '')
            block_label = f'{bs.date()} - {(be - pd.Timedelta(days=1)).date()}'
            print(f'{block_label:<25} {n_days_block:>5} {n:>7} {n/max(n_days_block,1):>7.2f} {wr:>6.1%} {pnl.mean():>11.6f} {pnl.sum():>9.4f} {sharpe:>7.2f}{flag}')
            block_results.append({'trades': n, 'wr': wr, 'ev': pnl.mean(), 'pnl': pnl.sum(), 'sharpe': sharpe})

        # Summary stats
        valid = [r for r in block_results if not np.isnan(r.get('wr', np.nan))]
        if len(valid) >= 2:
            wrs = [r['wr'] for r in valid]
            evs = [r['ev'] for r in valid]
            sharpes = [r['sharpe'] for r in valid]
            pnls = [r['pnl'] for r in valid]
            n_positive = sum(1 for p in pnls if p > 0)
            total_pnl = sum(pnls)
            print(f'{"":>25} {"":>5} {"":>7} {"":>7} {"":>7} {"":>11} {"-"*9} {"":>7}')
            print(f'{"AGGREGATE":<25} {"":>5} {sum(r["trades"] for r in valid):>7} '
                  f'{"":>7} {np.mean(wrs):>6.1%} {np.mean(evs):>11.6f} {total_pnl:>9.4f} {"":>7}')
            print(f'  Blocks positive: {n_positive}/{len(valid)}')
            print(f'  WR range: {min(wrs):.1%} - {max(wrs):.1%} (std: {np.std(wrs):.1%})')
            print(f'  Sharpe range: {min(sharpes):.2f} - {max(sharpes):.2f}')
            # Block-level consistency: is the signal present in most blocks?
            n_wr_above_55 = sum(1 for w in wrs if w > 0.55)
            print(f'  Blocks with WR > 55%: {n_wr_above_55}/{len(valid)}')

    # ── Q50-only walk-forward (no meta) ──
    print(f'\n{"="*80}')
    print(f'Q50-ONLY WALK-FORWARD (no meta-model)')
    print(f'{"="*80}')
    for name, q_thresh in [('|Q50|>1x', AVG_SPREAD), ('|Q50|>2x', AVG_SPREAD*2)]:
        print(f'\n--- {name} ---')
        print(f'{"Block":<25} {"Days":>5} {"Trades":>7} {"Tr/day":>7} {"WR":>7} {"EV/trade":>11} {"PnL":>9} {"Sharpe":>7}')
        print('-' * 85)
        for i, (bs, be, bdf) in enumerate(blocks):
            s = apply_4h_cooldown(bdf[bdf['abs_Q50'] > q_thresh])
            n = len(s)
            n_days_block = (be - bs).days
            if n < 2:
                print(f'{str(bs.date())} - {str((be - pd.Timedelta(days=1)).date()):<11} {n_days_block:>5} {n:>7}')
                continue
            pnl = s['pred_dir'] * s['label_1H'] - AVG_SPREAD
            wr = (s['pred_dir'] == s['actual_dir']).mean()
            sharpe = (pnl.mean() / pnl.std()) * np.sqrt(252 * 24) if pnl.std() > 0 else 0
            flag = ' <<<' if wr >= 0.70 else (' <<' if wr >= 0.60 else '')
            block_label = f'{bs.date()} - {(be - pd.Timedelta(days=1)).date()}'
            print(f'{block_label:<25} {n_days_block:>5} {n:>7} {n/max(n_days_block,1):>7.2f} {wr:>6.1%} {pnl.mean():>11.6f} {pnl.sum():>9.4f} {sharpe:>7.2f}{flag}')

    # ── Rescue model walk-forward ──
    print(f'\n{"="*80}')
    print(f'RESCUE MODEL WALK-FORWARD')
    print(f'{"="*80}')
    for rthresh in RESCUE_THRESHOLDS:
        print(f'\n--- Rescue R > {rthresh:.2f} ---')
        print(f'{"Block":<25} {"Days":>5} {"Rescued":>8} {"R/day":>7} {"WR":>7} {"EV/trade":>11} {"PnL":>9} {"Sharpe":>7}')
        print('-' * 85)
        block_results_r = []
        for i, (bs, be, bdf) in enumerate(blocks):
            bdf_rescued = bdf[bdf['rescue_proba'].notna()]
            s = apply_4h_cooldown(bdf_rescued[bdf_rescued['rescue_proba'] > rthresh])
            n = len(s)
            n_days_block = (be - bs).days
            if n < 2:
                print(f'{str(bs.date())} - {str((be - pd.Timedelta(days=1)).date()):<11} {n_days_block:>5} {n:>8}')
                block_results_r.append({'trades': n, 'wr': np.nan, 'pnl': 0})
                continue
            pnl = s['pred_dir'] * s['label_1H'] - AVG_SPREAD
            wr = (s['pred_dir'] == s['actual_dir']).mean()
            sharpe = (pnl.mean() / pnl.std()) * np.sqrt(252 * 24) if pnl.std() > 0 else 0
            flag = ' <<<' if wr >= 0.60 and pnl.sum() > 0 else ''
            block_label = f'{bs.date()} - {(be - pd.Timedelta(days=1)).date()}'
            print(f'{block_label:<25} {n_days_block:>5} {n:>8} {n/max(n_days_block,1):>7.2f} {wr:>6.1%} {pnl.mean():>11.6f} {pnl.sum():>9.4f} {sharpe:>7.2f}{flag}')
            block_results_r.append({'trades': n, 'wr': wr, 'pnl': pnl.sum()})
        valid_r = [r for r in block_results_r if not np.isnan(r.get('wr', np.nan))]
        if len(valid_r) >= 2:
            n_pos = sum(1 for r in valid_r if r['pnl'] > 0)
            total_pnl_r = sum(r['pnl'] for r in valid_r)
            print(f'  AGGREGATE: {sum(r["trades"] for r in valid_r)} trades, PnL={total_pnl_r:.4f}, '
                  f'Blocks positive: {n_pos}/{len(valid_r)}')

    # ── Combined walk-forward: Meta P>0.55 + Rescue ──
    print(f'\n{"="*80}')
    print(f'COMBINED WALK-FORWARD: Meta P>0.55 + Rescue')
    print(f'{"="*80}')
    META_BEST_WF = 0.55
    for rthresh in RESCUE_THRESHOLDS:
        print(f'\n--- Meta P>{META_BEST_WF} + Rescue R>{rthresh:.2f} ---')
        print(f'{"Block":<25} {"Days":>5} {"Meta":>6} {"Resc":>6} {"Total":>6} {"WR":>7} {"MetaPnL":>9} {"RescPnL":>9} {"CombPnL":>9}')
        print('-' * 95)
        for i, (bs, be, bdf) in enumerate(blocks):
            n_days_block = (be - bs).days
            # Meta trades
            bdf_tradeable = bdf[bdf['meta_proba'].notna()]
            meta_s = apply_4h_cooldown(bdf_tradeable[bdf_tradeable['meta_proba'] > META_BEST_WF])
            # Rescued trades
            bdf_rescued = bdf[bdf['rescue_proba'].notna()]
            rescue_s = apply_4h_cooldown(bdf_rescued[bdf_rescued['rescue_proba'] > rthresh])

            n_meta = len(meta_s)
            n_resc = len(rescue_s)
            total_n = n_meta + n_resc

            if total_n < 2:
                print(f'{str(bs.date())} - {str((be - pd.Timedelta(days=1)).date()):<11} {n_days_block:>5} {n_meta:>6} {n_resc:>6} {total_n:>6}')
                continue

            meta_pnl = (meta_s['pred_dir'] * meta_s['label_1H'] - AVG_SPREAD).sum() if n_meta > 0 else 0
            resc_pnl = (rescue_s['pred_dir'] * rescue_s['label_1H'] - AVG_SPREAD).sum() if n_resc > 0 else 0
            comb_pnl = meta_pnl + resc_pnl

            all_correct = pd.concat([
                (meta_s['pred_dir'] == meta_s['actual_dir']) if n_meta > 0 else pd.Series(dtype=bool),
                (rescue_s['pred_dir'] == rescue_s['actual_dir']) if n_resc > 0 else pd.Series(dtype=bool),
            ])
            wr = all_correct.mean() if len(all_correct) > 0 else 0

            flag = ' <<<' if comb_pnl > meta_pnl and resc_pnl > 0 else ''
            block_label = f'{bs.date()} - {(be - pd.Timedelta(days=1)).date()}'
            print(f'{block_label:<25} {n_days_block:>5} {n_meta:>6} {n_resc:>6} {total_n:>6} {wr:>6.1%} '
                  f'{meta_pnl:>9.4f} {resc_pnl:>9.4f} {comb_pnl:>9.4f}{flag}')

    # ── Monotonicity test: does higher threshold = higher WR in EACH block? ──
    print(f'\n{"="*80}')
    print(f'MONOTONICITY TEST: WR by threshold per block')
    print(f'{"="*80}')
    print(f'Does higher meta threshold consistently produce higher WR?')
    header = f'{"Block":<25}'
    for thresh in META_THRESHOLDS:
        header += f' {"P>"+str(thresh):>7}'
    print(header)
    print('-' * (25 + 8 * len(META_THRESHOLDS)))

    mono_violations = 0
    mono_total = 0
    for i, (bs, be, bdf) in enumerate(blocks):
        bdf_tradeable = bdf[bdf['meta_proba'].notna()]
        block_label = f'{bs.date()} - {(be - pd.Timedelta(days=1)).date()}'
        row = f'{block_label:<25}'
        prev_wr = None
        for thresh in META_THRESHOLDS:
            s = apply_4h_cooldown(bdf_tradeable[bdf_tradeable['meta_proba'] > thresh])
            if len(s) < 3:
                row += f' {"---":>7}'
                prev_wr = None
                continue
            wr = (s['pred_dir'] == s['actual_dir']).mean()
            row += f' {wr:>6.1%}'
            if prev_wr is not None:
                mono_total += 1
                if wr < prev_wr - 0.02:  # allow 2% tolerance
                    mono_violations += 1
            prev_wr = wr
        print(row)

    if mono_total > 0:
        print(f'\nMonotonicity violations: {mono_violations}/{mono_total} '
              f'({mono_violations/mono_total:.0%}) — lower is better')
        if mono_violations / mono_total < 0.15:
            print(f'PASS: Signal is monotonic (higher threshold = higher quality)')
        elif mono_violations / mono_total < 0.30:
            print(f'MARGINAL: Some monotonicity but noisy')
        else:
            print(f'FAIL: No consistent monotonicity — possible overfitting')


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────
async def main():
    t_start = time.time()

    # Step 1: Fetch data
    data = await fetch_all_pairs()

    # Step 2: Compute features
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

    # Step 3: Run inference
    print(f'\nRunning model inference...')
    df_all = run_inference(df_all)

    # Step 4: Determine backtest start (skip warmup period)
    backtest_start = df_all.index.min() + pd.Timedelta(days=WARMUP_DAYS)
    print(f'Backtest starts: {backtest_start.date()} (after {WARMUP_DAYS} days warmup)')

    # Step 5: Simulate and report
    simulate_trades(df_all, backtest_start)

    # Step 6: Walk-forward validation
    walk_forward_validation(df_all, backtest_start)

    elapsed_total = time.time() - t_start
    print(f'\nTotal runtime: {elapsed_total:.0f}s')


if __name__ == '__main__':
    asyncio.run(main())
