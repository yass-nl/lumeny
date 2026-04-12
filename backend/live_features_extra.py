"""
Extra features added in features_7/8 pipeline (between features_6 and features_9).
These are multi-day momentum, range position, volatility regime, and calendar features.
Formulas reverse-engineered from features_9 parquet via correlation analysis.
"""

import numpy as np
import pandas as pd


def compute_momentum_calendar_features(df_1h, pip_size):
    """
    Compute the 17 extra features added between features_6 and features_9.

    Input: df_1h — 1H OHLCV DataFrame (datetime index)
    Returns: DataFrame with extra columns aligned to df_1h.index
    """
    feat = pd.DataFrame(index=df_1h.index)
    c = df_1h['close']
    h = df_1h['high']
    l = df_1h['low']

    # ── Multi-day log returns (verified corr > 0.99) ─────────────────────────
    feat['ret_1d']  = np.log(c / c.shift(24))
    feat['ret_3d']  = np.log(c / c.shift(72))
    feat['ret_1w']  = np.log(c / c.shift(168))
    feat['ret_2w']  = np.log(c / c.shift(336))

    # ── 5/10-day range features ───────────────────────────────────────────────
    h5  = h.rolling(120, min_periods=24).max()
    l5  = l.rolling(120, min_periods=24).min()
    h10 = h.rolling(240, min_periods=48).max()
    l10 = l.rolling(240, min_periods=48).min()

    rng5  = (h5  - l5).clip(lower=1e-10)
    rng10 = (h10 - l10).clip(lower=1e-10)

    # range_pos_5d / range_pos_10d: (close - low) / range  (corr 0.989 / 0.992)
    feat['range_pos_5d']  = (c - l5)  / rng5
    feat['range_pos_10d'] = (c - l10) / rng10

    # dist_5d_high / dist_5d_low: position within 5d range (corr 0.989)
    feat['dist_5d_high'] = (h5 - c) / rng5
    feat['dist_5d_low']  = (c - l5) / rng5

    # range_width_5d: absolute range in price (corr 0.956)
    feat['range_width_5d'] = h5 - l5

    # ── Volatility regime features (rolling std ratios — verified corr > 0.965) ─
    log_ret = np.log(c / c.shift(1))
    std24  = log_ret.rolling(24,  min_periods=6).std()
    std120 = log_ret.rolling(120, min_periods=30).std()
    std240 = log_ret.rolling(240, min_periods=60).std()

    # vol_regime_5d = std(24h) / std(120h)  — corr 0.968
    feat['vol_regime_5d']  = std24  / std120.clip(lower=1e-10)
    # vol_regime_10d = std(24h) / std(240h) — corr 0.967
    feat['vol_regime_10d'] = std24  / std240.clip(lower=1e-10)
    # vol_trend = std(120h) / std(240h)     — corr 0.965
    feat['vol_trend']      = std120 / std240.clip(lower=1e-10)

    # ── Calendar features (verified corr = 1.0) ───────────────────────────────
    idx = df_1h.index

    # is_month_end: pandas month-end flag (corr 1.0)
    feat['is_month_end'] = idx.is_month_end.astype(np.float32)

    # days_to_friday: (4 - dayofweek) % 7 (corr 1.0)
    feat['days_to_friday'] = ((4 - idx.dayofweek) % 7).astype(np.float32)

    # is_quarter_end: last day of Q (corr 1.0)
    feat['is_quarter_end'] = (idx.month.isin([3, 6, 9, 12]) & idx.is_month_end).astype(np.float32)

    # is_month_end_3d: last 3 calendar days of month (corr 0.93 — best found)
    days_in_month = idx.to_series().dt.days_in_month.values
    feat['is_month_end_3d'] = ((days_in_month - idx.day) < 3).astype(np.float32)

    return feat.astype(np.float32)
