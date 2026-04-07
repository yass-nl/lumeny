"""
Rule-Based Capital Simulation — standalone
No ML model. Signal logic:
  GATE  (AND): volume_cv > p[h]  AND  epps_1m_15m > p[h]
  DIR   (AND): |momentum_shift| > p[h]  AND  |accel_mean| > p[h]
               AND sign(momentum_shift) == sign(accel_mean)
  TRADE: direction = sign(momentum_shift), held 3H

Everything else (spreads, sizing, cooldown, P&L) mirrors test_capital_sim_3.py.
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

API_KEY  = os.getenv('POLYGON_S3_SECRET_KEY', '')
REST_BASE = 'https://api.polygon.io'

# ── Config ──
PAIRS = [
    'EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD', 'USDCAD', 'NZDUSD',
    'EURJPY', 'GBPJPY', 'EURGBP', 'EURAUD', 'AUDJPY', 'CADJPY', 'CHFJPY', 'AUDNZD',
]

STARTING_CAPITAL   = 1_000_000.0
LOT_UNITS          = 100_000
MIN_MARGIN_TO_TRADE = 4_000
MAX_SPREAD_POINTS  = 50
RISK_PER_TRADE     = 0.005
COOLDOWN_HRS       = 3
BACKTEST_DAYS      = 200
WARMUP_DAYS        = 10
DATE_OFFSET_DAYS   = 0

GATE_PCT = 70    # percentile for volume_cv, epps_1m_15m
DIR_PCT  = 70    # percentile for |momentum_shift|, |accel_mean|

CAPACITY_CAPS = {
    'EURUSD': 139, 'GBPUSD': 58,  'USDJPY': 83,  'USDCHF': 13,
    'AUDUSD': 24,  'USDCAD': 18,  'NZDUSD': 6,
    'EURJPY': 11,  'GBPJPY': 8,   'EURGBP': 9,
    'EURAUD': 3,   'AUDJPY': 3,   'CADJPY': 1,   'CHFJPY': 2,   'AUDNZD': 1,
}
LEVERAGE = {p: 50 for p in PAIRS}
SPREAD_POINTS = {
    'AUDUSD': 6,  'EURUSD': 6,  'GBPUSD': 8,  'NZDUSD': 9,
    'USDCAD': 12, 'USDCHF': 7,  'USDJPY': 10,
    'EURGBP': 7,  'AUDNZD': 20, 'AUDJPY': 15, 'CADJPY': 16,
    'CHFJPY': 25, 'EURAUD': 21, 'EURJPY': 14, 'GBPJPY': 21,
}
OFFHOUR_SPREAD_MULTIPLIER = {
    'EURUSD': 1.1, 'GBPUSD': 1.1, 'USDJPY': 1.1, 'USDCHF': 1.2,
    'AUDUSD': 1.0, 'USDCAD': 0.9, 'NZDUSD': 1.0,
    'EURJPY': 1.2, 'GBPJPY': 1.2, 'EURGBP': 1.1, 'EURAUD': 1.2,
    'AUDJPY': 1.1, 'CADJPY': 1.2, 'CHFJPY': 1.2, 'AUDNZD': 1.3,
}
JPY_PAIRS = {'USDJPY', 'EURJPY', 'GBPJPY', 'AUDJPY', 'CADJPY', 'CHFJPY'}
PIP_SIZE = {
    'EURUSD': 0.0001, 'GBPUSD': 0.0001, 'AUDUSD': 0.0001, 'NZDUSD': 0.0001,
    'USDCAD': 0.0001, 'USDCHF': 0.0001, 'USDJPY': 0.01,
    'EURJPY': 0.01, 'GBPJPY': 0.01, 'EURGBP': 0.0001, 'EURAUD': 0.0001,
    'AUDJPY': 0.01, 'CADJPY': 0.01, 'CHFJPY': 0.01, 'AUDNZD': 0.0001,
}
QUOTE_USD = {
    'USD': 1.00, 'CHF': 1.27, 'CAD': 0.73, 'GBP': 1.34,
    'AUD': 0.70, 'NZD': 0.58, 'JPY': 0.0063,
}
BASE_USD = {
    'EUR': 1.15, 'GBP': 1.34, 'AUD': 0.70, 'NZD': 0.58,
    'USD': 1.00, 'CAD': 0.73, 'CHF': 0.79, 'JPY': 0.0063,
}
SLIPPAGE_BASE_POINTS = {
    'EURUSD': 3.0, 'GBPUSD': 4.0, 'USDJPY': 5.0, 'USDCHF': 5.0,
    'AUDUSD': 3.0, 'USDCAD': 3.0, 'NZDUSD': 3.0,
    'EURGBP': 3.0, 'AUDNZD': 6.0, 'AUDJPY': 4.0, 'CADJPY': 6.0,
    'CHFJPY': 9.0, 'EURAUD': 6.0, 'EURJPY': 6.0, 'GBPJPY': 7.0,
}
SLIPPAGE_TIME_MULT = {
    'EURUSD': 1.5, 'GBPUSD': 2.0, 'USDJPY': 2.0, 'USDCHF': 2.0,
    'AUDUSD': 2.0, 'USDCAD': 2.0, 'NZDUSD': 2.0,
    'EURJPY': 2.5, 'GBPJPY': 2.5, 'EURGBP': 2.0, 'EURAUD': 2.5,
    'AUDJPY': 2.5, 'CADJPY': 2.5, 'CHFJPY': 3.0, 'AUDNZD': 2.5,
}

print('Rule-Based Capital Simulation')
print(f'  Gate percentile:      p{GATE_PCT}')
print(f'  Direction percentile: p{DIR_PCT}')
print(f'  Starting capital:     ${STARTING_CAPITAL:,.0f}')
print(f'  Fetch window:         {BACKTEST_DAYS + WARMUP_DAYS} days (offset: {DATE_OFFSET_DAYS} days back)')

# ── Helpers ──
def get_spread_points(pair, hour_utc):
    base = SPREAD_POINTS.get(pair, 5)
    if hour_utc in (21, 22):
        return int(base * OFFHOUR_SPREAD_MULTIPLIER.get(pair, 2.0))
    return base

def spread_in_price(pair, hour_utc=12):
    pts = get_spread_points(pair, hour_utc)
    return pts * 0.001 if pair in JPY_PAIRS else pts * 0.00001

def slippage_in_price(pair, hour_utc, lots=1.0):
    base = SLIPPAGE_BASE_POINTS.get(pair, 1.0)
    tmult = SLIPPAGE_TIME_MULT.get(pair, 2.0) if hour_utc in (21, 22) else 1.0
    smult = min(1.0 + 0.05 * max(lots - 1.0, 0), 3.0)
    pts = base * tmult * smult
    return pts * 0.001 if pair in JPY_PAIRS else pts * 0.00001

def margin_required(pair, lots):
    base_val = BASE_USD.get(pair[:3], 1.0)
    return lots * LOT_UNITS * base_val / LEVERAGE[pair]

def compute_pnl_usd(pair, direction, lots, entry, exit_price):
    move = (exit_price - entry) if direction == 1 else (entry - exit_price)
    quote_ccy = pair[3:]
    if pair in JPY_PAIRS:
        return lots * LOT_UNITS * move / exit_price
    elif quote_ccy == 'USD':
        return lots * LOT_UNITS * move
    else:
        return lots * LOT_UNITS * move * QUOTE_USD.get(quote_ccy, 1.0)

def cost_usd(pair, lots, price, hour_utc):
    sp = spread_in_price(pair, hour_utc)
    sl = slippage_in_price(pair, hour_utc, lots)
    total_pts = sp + sl
    quote_ccy = pair[3:]
    if pair in JPY_PAIRS:
        return lots * LOT_UNITS * total_pts / price
    elif quote_ccy == 'USD':
        return lots * LOT_UNITS * total_pts
    else:
        return lots * LOT_UNITS * total_pts * QUOTE_USD.get(quote_ccy, 1.0)


# ── Fetch ──
async def fetch_bars(pair, multiplier, timespan, from_date, to_date):
    ticker = f'C:{pair}'
    url = f'{REST_BASE}/v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{from_date}/{to_date}'
    params = {'apiKey': API_KEY, 'limit': 50000, 'sort': 'asc'}
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
    df = df.set_index('datetime')[['open', 'high', 'low', 'close', 'volume']].sort_index().drop_duplicates()
    df = df[~((df.index.dayofweek == 5) | ((df.index.dayofweek == 6) & (df.index.hour < 21)))]
    return df


# ── Feature functions (copied from test_capital_sim_3) ──
def realized_vol_estimators(o, h, l, c):
    n = len(o)
    if n < 5:
        return {k: np.nan for k in ['rv_close','rv_parkinson','rv_garman_klass','rv_rogers_satchell','rv_yang_zhang','range_return_ratio']}
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
    return {'rv_close': rv_close, 'rv_parkinson': rv_parkinson, 'rv_garman_klass': rv_gk,
            'rv_rogers_satchell': rv_rs, 'rv_yang_zhang': rv_yz, 'range_return_ratio': range_return_ratio}


def jump_detection(returns):
    n = len(returns)
    if n < 5:
        return {k: np.nan for k in ['rv','bv','jump_ratio','jump_z','jump_intensity','jump_mean_size','jump_asymmetry']}
    rv = np.sum(returns**2)
    abs_r = np.abs(returns)
    bv = (np.pi / 2) * (n / (n - 1)) * np.sum(abs_r[1:] * abs_r[:-1])
    jump = max(rv - bv, 0)
    jump_ratio = jump / rv if rv > 1e-20 else 0
    mu_43 = 2**(2/3) * np.exp(lgamma(7/6) - lgamma(1/2))
    if n >= 4:
        tpq = n * mu_43**(-3) * (n / (n-2)) * np.sum(abs_r[2:]**(4/3) * abs_r[1:-1]**(4/3) * abs_r[:-2]**(4/3))
        v_const = np.pi**2/4 + np.pi - 5
        relative_qv = max(tpq / (bv**2) - 1, 0) if bv > 1e-20 else 0
        denom = np.sqrt(v_const * relative_qv / n) * bv if bv > 1e-20 else 0
        jump_z = np.clip((rv - bv) / denom, -10, 10) if denom > 1e-20 else 0
    else:
        jump_z = 0
    threshold = 3.0 * np.sqrt(max(bv / n, 1e-20))
    is_jump = np.abs(returns) > threshold
    n_jumps = is_jump.sum()
    jump_intensity = n_jumps / n
    if n_jumps > 0:
        jr = returns[is_jump]
        jump_mean_size = np.mean(np.abs(jr))
        jump_asymmetry = (jr > 0).sum() / n_jumps - (jr < 0).sum() / n_jumps
    else:
        jump_mean_size = jump_asymmetry = 0
    return {'rv': rv, 'bv': bv, 'jump_ratio': jump_ratio, 'jump_z': jump_z,
            'jump_intensity': jump_intensity, 'jump_mean_size': jump_mean_size, 'jump_asymmetry': jump_asymmetry}


def order_flow_features(o, h, l, c, v, rolling_sigma=None):
    n = len(c)
    if n < 5:
        return {k: np.nan for k in ['buy_volume_frac','order_imbalance','order_imbalance_intensity','kyle_lambda','kyle_lambda_r2','amihud_illiq','volume_cv']}
    log_ret = np.log(c / o)
    sigma = rolling_sigma if (rolling_sigma is not None and rolling_sigma > 1e-10) else max(np.std(np.diff(np.log(c))), 1e-10)
    z_scores = log_ret / sigma
    v_buy = v * stats.norm.cdf(z_scores)
    v_sell = v - v_buy
    total_v = max(v.sum(), 1)
    oi = (v_buy.sum() - v_sell.sum()) / total_v
    bar_returns = np.diff(np.log(c))
    signed_flow = np.sign(bar_returns) * np.sqrt(np.maximum(v[1:], 1))
    if len(bar_returns) > 3 and np.std(signed_flow) > 1e-10:
        slope, _, r_value, _, _ = stats.linregress(signed_flow, bar_returns)
        kyle_lambda = slope
        kyle_r2 = r_value**2
    else:
        kyle_lambda = kyle_r2 = 0
    amihud = np.mean(np.abs(bar_returns) / np.log1p(np.maximum(v[1:], 1)))
    v_mean = v.mean()
    volume_cv = v.std() / v_mean if v_mean > 0 else 0
    return {'buy_volume_frac': v_buy.sum()/total_v, 'order_imbalance': oi,
            'order_imbalance_intensity': abs(oi), 'kyle_lambda': kyle_lambda,
            'kyle_lambda_r2': kyle_r2, 'amihud_illiq': amihud, 'volume_cv': volume_cv}


def information_theory_features(returns, volumes=None):
    n = len(returns)
    if n < 10:
        return {k: np.nan for k in ['entropy_returns','entropy_norm','jb_statistic','entropy_volume_divergence','kl_proxy']}
    iqr_val = np.percentile(returns, 75) - np.percentile(returns, 25)
    n_bins = max(int(np.ceil((returns.max()-returns.min()) / (2*iqr_val*n**(-1/3)))) if iqr_val > 0 else 10, 5)
    n_bins = min(n_bins, 30)
    hist, _ = np.histogram(returns, bins=n_bins)
    p = hist / hist.sum()
    p = p[p > 0]
    entropy = -np.sum(p * np.log(p))
    entropy_norm = entropy / np.log(n_bins)
    s = stats.skew(returns)
    k = stats.kurtosis(returns, fisher=True)
    jb = (n / 6) * (s**2 + (k**2) / 4)
    evd = 0
    if volumes is not None and len(volumes) == n and volumes.sum() > 0:
        bin_edges = np.linspace(returns.min()-1e-10, returns.max()+1e-10, n_bins+1)
        bin_idx = np.clip(np.digitize(returns, bin_edges)-1, 0, n_bins-1)
        vol_per_bin = np.zeros(n_bins)
        for i in range(n):
            vol_per_bin[bin_idx[i]] += volumes[i]
        pv = vol_per_bin / vol_per_bin.sum()
        pv = pv[pv > 0]
        evd = entropy_norm - (-np.sum(pv * np.log(pv)) / np.log(n_bins))
    return {'entropy_returns': entropy, 'entropy_norm': entropy_norm, 'jb_statistic': jb,
            'entropy_volume_divergence': evd, 'kl_proxy': jb/n}


def market_efficiency_features(returns):
    n = len(returns)
    if n < 15:
        return {k: np.nan for k in ['vr_5','vr_10','vr_z5','runs_z','autocorr_1','autocorr_2','autocorr_5','autocorr_decay_halflife','sum_abs_autocorr','noise_to_signal']}
    rv1 = np.var(returns, ddof=1)
    def variance_ratio(r, q):
        if len(r) < q+1 or rv1 < 1e-20: return 1.0, 0.0
        r_q = np.array([r[i:i+q].sum() for i in range(len(r)-q+1)])
        vr = np.var(r_q, ddof=1) / (q * rv1)
        z = (vr-1) / np.sqrt(2*(2*q-1)*(q-1)/(3*q*n))
        return vr, z
    vr5, z5 = variance_ratio(returns, 5)
    vr10, _ = variance_ratio(returns, 10)
    signs = returns[returns != 0]
    runs_z = 0
    if len(signs) > 5:
        pos = (signs > 0).astype(int)
        n_pos, n_neg = pos.sum(), len(pos)-pos.sum()
        runs = 1 + np.sum(np.diff(pos) != 0)
        n_total = len(pos)
        if n_pos > 0 and n_neg > 0:
            e_runs = 1 + 2*n_pos*n_neg/n_total
            v_runs = (2*n_pos*n_neg*(2*n_pos*n_neg-n_total)) / (n_total**2*(n_total-1))
            runs_z = (runs - e_runs) / np.sqrt(max(v_runs, 1e-20))
    def autocorr(r, lag):
        if len(r) <= lag+1: return 0
        r_dm = r - r.mean()
        denom = np.sum(r_dm**2)
        return np.sum(r_dm[lag:]*r_dm[:-lag]) / denom if denom > 1e-20 else 0
    ac1, ac2, ac5 = autocorr(returns,1), autocorr(returns,2), autocorr(returns,5)
    lags = list(range(1, min(21, n//2)))
    abs_acs = [abs(autocorr(returns, k)) for k in lags]
    valid = [(k, ac) for k, ac in zip(lags, abs_acs) if ac > 1e-6]
    halflife = 0
    if len(valid) >= 3:
        x_fit = np.array([v[0] for v in valid])
        y_fit = np.log(np.array([v[1] for v in valid]))
        slope, _, _, _, _ = stats.linregress(x_fit, y_fit)
        halflife = (-1/slope)*np.log(2) if slope < -0.01 else 100
    noise_var = max(-np.mean(returns[:-1]*returns[1:]), 0)
    bv_bar = np.mean(np.abs(returns[1:])*np.abs(returns[:-1]))*(np.pi/2)
    nsr = noise_var/bv_bar if bv_bar > 1e-20 else 0
    return {'vr_5': vr5, 'vr_10': vr10, 'vr_z5': z5, 'runs_z': runs_z,
            'autocorr_1': ac1, 'autocorr_2': ac2, 'autocorr_5': ac5,
            'autocorr_decay_halflife': halflife, 'sum_abs_autocorr': sum(abs_acs), 'noise_to_signal': nsr}


def tail_risk_features(returns):
    n = len(returns)
    if n < 10:
        return {k: np.nan for k in ['realized_skew','realized_kurt','tail_ratio_95_5','hill_tail_index']}
    rv = np.sum(returns**2)
    rskew = (np.sqrt(n)*np.sum(returns**3))/rv**1.5 if rv > 1e-20 else 0
    rkurt = (n*np.sum(returns**4))/rv**2 if rv > 1e-20 else 3
    p95, p5 = np.abs(np.percentile(returns, 95)), np.abs(np.percentile(returns, 5))
    tail_ratio = p95/p5 if p5 > 1e-15 else 1.0
    abs_r = np.sort(np.abs(returns))[::-1]
    k = max(int(np.sqrt(n)), 3)
    if k < n and abs_r[k] > 1e-15:
        log_sum = np.sum(np.log(abs_r[:k]/abs_r[k]))
        hill = min(k/log_sum, 20.0) if log_sum > 1e-10 else np.nan
    else:
        hill = np.nan
    return {'realized_skew': rskew, 'realized_kurt': rkurt, 'tail_ratio_95_5': tail_ratio, 'hill_tail_index': hill}


def acceleration_features(c, returns):
    n = len(returns)
    if n < 10:
        return {k: np.nan for k in ['accel_mean','accel_std','accel_skew','momentum_shift','ret_concentration','vol_clustering_ac1']}
    velocity = np.diff(c)
    if len(velocity) > 1:
        accel = np.diff(velocity)
        accel_mean = np.mean(accel)
        accel_std = np.std(accel)
        accel_skew = stats.skew(accel) if len(accel) > 3 else 0
    else:
        accel_mean = accel_std = accel_skew = 0
    third = n // 3
    momentum_shift = (returns[-third:].sum() - returns[:third].sum()) if third > 0 else 0
    abs_ret = np.abs(returns)
    total_abs = abs_ret.sum()
    top_k = max(int(n*0.1), 1)
    ret_concentration = np.sort(abs_ret)[-top_k:].sum()/total_abs if total_abs > 1e-15 else 0
    r2 = returns**2
    r2_dm = r2 - r2.mean()
    denom = np.sum(r2_dm**2)
    vol_cluster_ac1 = np.sum(r2_dm[1:]*r2_dm[:-1])/denom if (len(r2) > 2 and denom > 1e-20) else 0
    return {'accel_mean': accel_mean, 'accel_std': accel_std, 'accel_skew': accel_skew,
            'momentum_shift': momentum_shift, 'ret_concentration': ret_concentration, 'vol_clustering_ac1': vol_cluster_ac1}


def cross_timeframe_features(returns_1m, returns_5m, returns_15m):
    def safe_ratio(num, den, cap=50.0):
        return min(num/den, cap) if (den is not None and den > 1e-20) else np.nan
    rv_1m = np.sum(returns_1m**2) if len(returns_1m) > 0 else np.nan
    rv_5m = np.sum(returns_5m**2) if len(returns_5m) > 0 else np.nan
    rv_15m = np.sum(returns_15m**2) if len(returns_15m) > 0 else np.nan
    feat = {
        'epps_1m_5m':  safe_ratio(rv_1m, rv_5m),
        'epps_1m_15m': safe_ratio(rv_1m, rv_15m),
        'epps_5m_15m': safe_ratio(rv_5m, rv_15m),
    }
    if len(returns_1m) >= 10:
        mid = len(returns_1m) // 2
        rv_f = np.sum(returns_1m[:mid]**2)
        rv_l = np.sum(returns_1m[mid:]**2)
        feat['info_accel'] = min(rv_l/rv_f, 50.0) if rv_f > 1e-20 else 1.0
    else:
        feat['info_accel'] = np.nan
    if len(returns_1m) >= 30:
        bs = len(returns_1m) // 6
        bvols = [np.std(returns_1m[i*bs:(i+1)*bs]) for i in range(6) if len(returns_1m[i*bs:(i+1)*bs]) > 1]
        mean_bv = np.mean(bvols) if bvols else 0
        feat['vol_of_vol'] = np.std(bvols)/mean_bv if (bvols and mean_bv > 1e-15) else np.nan
    else:
        feat['vol_of_vol'] = np.nan
    if len(returns_5m) > 3:
        r5_dm = returns_5m - returns_5m.mean()
        denom = np.sum(r5_dm**2)
        feat['autocorr_5m_lag1'] = np.sum(r5_dm[1:]*r5_dm[:-1])/denom if denom > 1e-20 else 0
    else:
        feat['autocorr_5m_lag1'] = np.nan
    return feat


def hurst_rs(series, min_window=10):
    n = len(series)
    if n < 30: return np.nan
    max_k = int(np.log2(n))
    window_sizes = [2**i for i in range(int(np.log2(min_window)), max_k+1) if 2**i <= n//2]
    if len(window_sizes) < 3: return np.nan
    rs_values = []
    for w in window_sizes:
        rs_list = []
        for seg in range(n//w):
            chunk = series[seg*w:(seg+1)*w]
            m = chunk.mean()
            cumdev = np.cumsum(chunk - m)
            R = cumdev.max() - cumdev.min()
            S = chunk.std(ddof=1)
            if S > 1e-15: rs_list.append(R/S)
        if rs_list: rs_values.append((np.log(w), np.log(np.mean(rs_list))))
    if len(rs_values) < 3: return np.nan
    x = np.array([v[0] for v in rs_values])
    y = np.array([v[1] for v in rs_values])
    slope, _, _, _, _ = stats.linregress(x, y)
    return slope


def fractal_dimension_higuchi(series, k_max=16):
    n = len(series)
    if n < 30: return np.nan
    k_values = [k for k in [1,2,4,8,16,32] if k < n//4]
    if len(k_values) < 3: return np.nan
    lk_values = []
    for k in k_values:
        lm_list = []
        for m in range(1, k+1):
            indices = np.arange(m-1, n, k)
            if len(indices) < 2: continue
            diffs = np.abs(np.diff(series[indices]))
            norm = (n-1) / (len(diffs)*k*k)
            lm_list.append(diffs.sum()*norm)
        if lm_list: lk_values.append((np.log(1.0/k), np.log(np.mean(lm_list))))
    if len(lk_values) < 3: return np.nan
    x = np.array([v[0] for v in lk_values])
    y = np.array([v[1] for v in lk_values])
    slope, _, _, _, _ = stats.linregress(x, y)
    return slope


def compute_hour_features(df_1m_hour, df_5m_hour, df_15m_hour, rolling_sigma=None):
    if len(df_1m_hour) < 5: return None, None
    o = df_1m_hour['open'].values
    h = df_1m_hour['high'].values
    l = df_1m_hour['low'].values
    c = df_1m_hour['close'].values
    v = df_1m_hour['volume'].values.astype(np.float64)
    returns_1m = np.diff(np.log(c))
    if len(returns_1m) < 3: return None, None
    returns_5m  = np.diff(np.log(df_5m_hour['close'].values))  if len(df_5m_hour)  > 1 else np.array([])
    returns_15m = np.diff(np.log(df_15m_hour['close'].values)) if len(df_15m_hour) > 1 else np.array([])
    features = {}
    features.update(realized_vol_estimators(o, h, l, c))
    features.update(jump_detection(returns_1m))
    features.update(order_flow_features(o, h, l, c, v, rolling_sigma))
    features.update(information_theory_features(returns_1m, v[1:]))
    features['fractal_dim'] = fractal_dimension_higuchi(c)
    features.update(market_efficiency_features(returns_1m))
    features.update(tail_risk_features(returns_1m))
    features.update(acceleration_features(c, returns_1m))
    features.update(cross_timeframe_features(returns_1m, returns_5m, returns_15m))
    return features, returns_1m


def compute_features_for_pair(pair, data):
    df_1m  = data[pair]['1m']
    df_5m  = data[pair]['5m']
    df_15m = data[pair]['15m']
    df_1h  = data[pair]['1h']
    df_1m['hour'] = df_1m.index.floor('h')
    hours = sorted(df_1m['hour'].unique())
    all_features = []
    returns_1m_dict = {}
    rolling_sigma = None
    for hour_ts in hours:
        mask_1m = df_1m['hour'] == hour_ts
        hour_end = hour_ts + pd.Timedelta(hours=1)
        feat, ret_1m = compute_hour_features(
            df_1m[mask_1m],
            df_5m[(df_5m.index >= hour_ts) & (df_5m.index < hour_end)],
            df_15m[(df_15m.index >= hour_ts) & (df_15m.index < hour_end)],
            rolling_sigma
        )
        if feat is not None:
            feat['datetime'] = hour_ts
            all_features.append(feat)
            returns_1m_dict[hour_ts] = ret_1m
            hour_sigma = np.std(ret_1m)
            rolling_sigma = hour_sigma if rolling_sigma is None else 0.95*rolling_sigma + 0.05*hour_sigma
    if not all_features:
        return pd.DataFrame()
    df_feat = pd.DataFrame(all_features).set_index('datetime').sort_index()

    # Trailing features
    n = len(df_feat)
    hurst_vals = np.full(n, np.nan)
    for i in range(6, n):
        all_ret = [returns_1m_dict[h] for h in df_feat.index[max(0,i-6):i] if h in returns_1m_dict]
        if all_ret:
            concat_ret = np.concatenate(all_ret)
            if len(concat_ret) >= 60:
                hurst_vals[i] = hurst_rs(concat_ret, min_window=8)
    df_feat['hurst_6h'] = hurst_vals
    if 'order_imbalance_intensity' in df_feat.columns:
        df_feat['vpin_4h']  = df_feat['order_imbalance_intensity'].rolling(4, min_periods=2).mean()
        df_feat['vpin_12h'] = df_feat['order_imbalance_intensity'].rolling(12, min_periods=4).mean()
    if 'rv_close' in df_feat.columns:
        df_feat['rv_zscore_24'] = (
            (df_feat['rv_close'] - df_feat['rv_close'].rolling(24, min_periods=8).mean())
            / df_feat['rv_close'].rolling(24, min_periods=8).std().clip(lower=1e-15))
    if 'kyle_lambda' in df_feat.columns:
        df_feat['kyle_lambda_change'] = (
            df_feat['kyle_lambda'].rolling(6, min_periods=2).mean() -
            df_feat['kyle_lambda'].rolling(24, min_periods=8).mean())

    # ATR for sizing
    tr = np.maximum(
        df_1h['high'] - df_1h['low'],
        np.maximum(np.abs(df_1h['high'] - df_1h['close'].shift(1)),
                   np.abs(df_1h['low']  - df_1h['close'].shift(1)))
    )
    df_feat['atr_24h']   = tr.rolling(24, min_periods=6).mean().reindex(df_feat.index, method='ffill')
    df_feat['close']     = df_1h['close'].reindex(df_feat.index, method='ffill')
    df_feat['pair']      = pair
    return df_feat


# ── Load per-hour thresholds ──
thresholds = pd.read_parquet('backend/models_6/hour_thresholds.parquet')
print(f'  Thresholds loaded: {len(thresholds)} hours\n')


# ── Fetch data ──
async def fetch_all():
    now       = datetime.utcnow() - timedelta(days=DATE_OFFSET_DAYS)
    to_date   = now.strftime('%Y-%m-%d')
    from_date = (now - timedelta(days=BACKTEST_DAYS + WARMUP_DAYS)).strftime('%Y-%m-%d')
    print(f'Fetching {from_date} to {to_date}...')
    data = {}
    for pair in PAIRS:
        print(f'  {pair}...', end=' ', flush=True)
        t0 = time.time()
        df_1m = await fetch_bars(pair, 1, 'minute', from_date, to_date)
        if df_1m.empty:
            print('NO DATA'); continue
        df_5m  = df_1m.resample('5min').agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum'}).dropna()
        df_15m = df_1m.resample('15min').agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum'}).dropna()
        df_1h  = df_1m.resample('1h').agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum'}).dropna()
        data[pair] = {'1m': df_1m, '5m': df_5m, '15m': df_15m, '1h': df_1h}
        print(f'{len(df_1m):,} bars ({time.time()-t0:.1f}s)')
        await asyncio.sleep(0.5)
    return data

raw_data = asyncio.run(fetch_all())

# ── Build features ──
print('\nBuilding features...')
WARMUP_CUTOFF = pd.Timestamp(datetime.utcnow() - timedelta(days=DATE_OFFSET_DAYS + BACKTEST_DAYS)).tz_localize(None)
frames = []
for pair in PAIRS:
    if pair not in raw_data: continue
    df_feat = compute_features_for_pair(pair, raw_data)
    if df_feat is None or len(df_feat) == 0:
        print(f'  {pair}: NO FEATURES'); continue
    df_feat = df_feat[df_feat.index >= WARMUP_CUTOFF]
    frames.append(df_feat)
    print(f'  {pair}: {len(df_feat)} hours')

if not frames:
    print('No features built.'); exit()

df_all = pd.concat(frames).sort_index()
print(f'\nTotal rows: {len(df_all):,}')

# ── Apply rules ──
print('\nApplying rules...')
hours_arr  = df_all.index.hour
vol_cut    = thresholds.loc[hours_arr, f'volume_cv_p{GATE_PCT}'].values
epps_cut   = thresholds.loc[hours_arr, f'epps_1m_15m_p{GATE_PCT}'].values
mom_cut    = thresholds.loc[hours_arr, f'abs_momentum_shift_p{DIR_PCT}'].values
accel_cut  = thresholds.loc[hours_arr, f'abs_accel_mean_p{DIR_PCT}'].values

gate   = (df_all['volume_cv'].values > vol_cut) & (df_all['epps_1m_15m'].values > epps_cut)
dir_ok = (df_all['momentum_shift'].abs().values > mom_cut) & (df_all['accel_mean'].abs().values > accel_cut)
agree  = np.sign(df_all['momentum_shift'].values) == np.sign(df_all['accel_mean'].values)
spread_ok = np.array([get_spread_points(r['pair'], idx.hour) <= MAX_SPREAD_POINTS
                      for idx, r in df_all.iterrows()])

signal_mask = gate & dir_ok & agree & spread_ok
df_signals  = df_all[signal_mask].copy()
df_signals['signal_direction'] = np.sign(df_signals['momentum_shift']).astype(int)
print(f'Raw signals: {len(df_signals):,}')

# ── Cooldown ──
df_signals = df_signals.sort_index()
pair_unlock = {}
keep = []
for idx, row in df_signals.iterrows():
    unlock = pair_unlock.get(row['pair'])
    if unlock is not None and idx < unlock:
        keep.append(False)
    else:
        keep.append(True)
        pair_unlock[row['pair']] = idx + pd.Timedelta(hours=COOLDOWN_HRS)
df_signals = df_signals[keep]
print(f'After cooldown: {len(df_signals):,}')

# ── Per-hour signal count ──
print(f'\nSignals per hour:')
for h in range(24):
    n = (df_signals.index.hour == h).sum()
    if n > 0:
        print(f'  {h:>2}h  {n:>4}  {"#" * (n // 2)}')

# ── Simulate ──
print(f'\n{"="*70}')
print(f'SIMULATION  {df_all.index.min().date()} to {df_all.index.max().date()}')
print(f'Signals: {len(df_signals):,}')

signal_index = set(df_signals.index)
equity = STARTING_CAPITAL
peak_equity = equity
max_dd = 0.0
open_pos = {}
trade_log = []

for hour in sorted(df_all.index.unique()):
    # Close positions at 3H
    for pair in list(open_pos.keys()):
        pos = open_pos[pair]
        if hour >= pos['open_time'] + pd.Timedelta(hours=COOLDOWN_HRS):
            row = df_all[(df_all.index == hour) & (df_all['pair'] == pair)]
            exit_price = row.iloc[0]['close'] if not row.empty else pos['entry_price']
            gross = compute_pnl_usd(pair, pos['direction'], pos['lots'], pos['entry_price'], exit_price)
            costs = cost_usd(pair, pos['lots'], exit_price, hour.hour)
            net   = gross - costs - pos['entry_cost']
            equity += net
            peak_equity = max(peak_equity, equity)
            max_dd = max(max_dd, (peak_equity - equity) / peak_equity)
            trade_log.append({
                'open_time': pos['open_time'], 'close_time': hour,
                'pair': pair, 'direction': pos['direction'],
                'lots': pos['lots'], 'entry': pos['entry_price'], 'exit': exit_price,
                'gross': gross, 'costs': costs + pos['entry_cost'], 'net': net,
                'hour': pos['open_time'].hour,
            })
            del open_pos[pair]

    if hour not in signal_index:
        continue

    for _, sig in df_signals[df_signals.index == hour].iterrows():
        pair = sig['pair']
        if pair in open_pos: continue
        price = sig.get('close', np.nan)
        if pd.isna(price) or price <= 0: continue

        # ATR sizing
        atr = sig.get('atr_24h', np.nan)
        if pd.isna(atr) or atr <= 0:
            atr = price * 0.005
        pip = PIP_SIZE.get(pair, 0.0001)
        atr_pips = atr / pip
        quote_ccy = pair[3:]
        if pair in JPY_PAIRS:
            val_per_pip = LOT_UNITS * pip / price
        elif quote_ccy != 'USD':
            val_per_pip = LOT_UNITS * pip * QUOTE_USD.get(quote_ccy, 1.0)
        else:
            val_per_pip = LOT_UNITS * pip
        raw_lots = (equity * RISK_PER_TRADE) / (atr_pips * val_per_pip) if atr_pips > 0 else 0
        lots = max(0.01, min(round(raw_lots, 2), float(CAPACITY_CAPS.get(pair, 1))))

        if equity - margin_required(pair, lots) < MIN_MARGIN_TO_TRADE: continue

        entry_cost = cost_usd(pair, lots, price, hour.hour)
        open_pos[pair] = {
            'direction': sig['signal_direction'], 'entry_price': price,
            'lots': lots, 'open_time': hour, 'entry_cost': entry_cost,
        }

# ── Results ──
if not trade_log:
    print('No trades executed.'); exit()

trades = pd.DataFrame(trade_log)
n = len(trades)
wins = (trades['net'] > 0).sum()
wr = wins / n
total_pnl = trades['net'].sum()
pct_return = total_pnl / STARTING_CAPITAL * 100
daily_pnl = trades.set_index('close_time')['net'].resample('D').sum().fillna(0)
sharpe = daily_pnl.mean() / daily_pnl.std() * np.sqrt(252) if daily_pnl.std() > 0 else 0
avg_win  = trades[trades['net'] > 0]['net'].mean() if wins > 0 else 0
avg_loss = trades[trades['net'] <= 0]['net'].mean() if (n - wins) > 0 else 0
rr = abs(avg_win / avg_loss) if avg_loss != 0 else 0

print(f'\nTotal trades:   {n}')
print(f'Win rate:       {wr:.1%}  ({wins}W / {n-wins}L)')
print(f'Avg win:        ${avg_win:,.0f}')
print(f'Avg loss:       ${avg_loss:,.0f}')
print(f'Risk/reward:    {rr:.2f}')
print(f'Total P&L:      ${total_pnl:,.0f}  ({pct_return:.1f}%)')
print(f'Sharpe:         {sharpe:.2f}')
print(f'Max drawdown:   {max_dd:.1%}')

print(f'\n{"Hour":>6} {"Trades":>8} {"Win%":>7} {"Net P&L":>12} {"Avg net":>10}')
print('-' * 48)
for h in range(24):
    sub = trades[trades['hour'] == h]
    if len(sub) == 0: continue
    print(f'{h:>6} {len(sub):>8} {(sub["net"]>0).mean():>7.1%} {sub["net"].sum():>12,.0f} {sub["net"].mean():>10,.0f}')

print(f'\n{"Pair":>8} {"Trades":>8} {"Win%":>7} {"Net P&L":>12}')
print('-' * 38)
for pair in PAIRS:
    sub = trades[trades['pair'] == pair]
    if len(sub) == 0: continue
    print(f'{pair:>8} {len(sub):>8} {(sub["net"]>0).mean():>7.1%} {sub["net"].sum():>12,.0f}')
