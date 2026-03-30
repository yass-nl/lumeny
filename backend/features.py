"""
Feature engineering v6.0 — 112 features (64 microstructure + 48 contextual).
Microstructure: exact replica of notebooks_5.1/02_microstructure_features.ipynb.
Contextual: mirrors notebooks_6/01_big_move_detection.ipynb → compute_extra_features().
"""

import numpy as np
import pandas as pd
from scipy import stats
from math import lgamma


PAIRS = [
    # Majors
    'EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD', 'USDCAD', 'NZDUSD',
    # Crosses
    'EURJPY', 'GBPJPY', 'EURGBP', 'EURAUD', 'AUDJPY', 'CADJPY', 'CHFJPY', 'AUDNZD',
]

PIP_SIZE = {
    'EURUSD': 0.0001, 'GBPUSD': 0.0001, 'AUDUSD': 0.0001, 'NZDUSD': 0.0001,
    'USDCAD': 0.0001, 'USDCHF': 0.0001, 'USDJPY': 0.01,
    'EURJPY': 0.01, 'GBPJPY': 0.01, 'EURGBP': 0.0001, 'EURAUD': 0.0001,
    'AUDJPY': 0.01, 'CADJPY': 0.01, 'CHFJPY': 0.01, 'AUDNZD': 0.0001,
}


def resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    return df.resample(rule).agg({
        'open': 'first', 'high': 'max', 'low': 'min',
        'close': 'last', 'volume': 'sum',
    }).dropna()


# ── Microstructure feature functions ──────────────────────────────────────

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
    """Compute all microstructure features for one hour of 1-min data."""
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
    """Compute trailing/rolling features across multiple hours."""
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


def compute_features_for_pair(pair, df_1m, df_5m, df_15m, df_1h=None):
    """
    Compute all features for one pair from OHLCV data.
    Returns a DataFrame indexed by hour timestamps with 112 features
    (64 microstructure + 48 contextual if df_1h is provided).
    """
    df_1m = df_1m.copy()
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

    # Add contextual features (features_6) if 1H data is available
    if df_1h is not None and not df_1h.empty:
        df_extra = compute_contextual_features(df_features, df_1h, pair)
        df_features = df_features.join(df_extra, how='left')

    float_cols = df_features.select_dtypes(include=[np.float64]).columns
    df_features[float_cols] = df_features[float_cols].astype(np.float32)

    return df_features
