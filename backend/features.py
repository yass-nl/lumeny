"""
Feature engineering — exact replica of notebook 02.
Computes 208 features from multi-timeframe OHLCV data.
"""

import pandas as pd
import numpy as np
import pandas_ta as ta


PAIRS = ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD', 'USDCAD', 'NZDUSD']

TIMEFRAMES_RESAMPLE = {
    '5m':  '5min',
    '15m': '15min',
    '1H':  '1h',
    '4H':  '4h',
    '1D':  '1D',
}


def resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    return df.resample(rule).agg({
        'open': 'first', 'high': 'max', 'low': 'min',
        'close': 'last', 'volume': 'sum',
    }).dropna()


def compute_features(df: pd.DataFrame, tf: str) -> pd.DataFrame:
    """
    Compute all TA features for a given OHLCV DataFrame.
    Returns a DataFrame with columns suffixed with _{tf}.
    Exact replica of notebook 02 compute_features().
    """
    feat = pd.DataFrame(index=df.index)
    o, h, l, c = df['open'], df['high'], df['low'], df['close']

    # ── RETURNS & MOMENTUM ──────────────────────────────────────
    for n in [1, 3, 6, 12, 24, 48]:
        feat[f'log_ret_{n}'] = np.log(c / c.shift(n))

    rsi_14 = ta.rsi(c, length=14)
    rsi_28 = ta.rsi(c, length=28)
    feat['rsi_14'] = rsi_14 if rsi_14 is not None else np.nan
    feat['rsi_28'] = rsi_28 if rsi_28 is not None else np.nan
    feat['rsi_slope'] = feat['rsi_14'] - feat['rsi_14'].shift(3)

    price_higher = (c > c.shift(5)).astype(int)
    rsi_higher = (feat['rsi_14'] > feat['rsi_14'].shift(5)).astype(int)
    feat['rsi_divergence'] = (price_higher != rsi_higher).astype(int)

    macd = ta.macd(c, fast=12, slow=26, signal=9)
    if macd is not None:
        feat['macd'] = macd.iloc[:, 0]
        feat['macd_signal'] = macd.iloc[:, 2]
        feat['macd_hist'] = macd.iloc[:, 1]

    for n in [20, 50, 200]:
        ma = ta.sma(c, length=n)
        if ma is not None:
            feat[f'dist_ma_{n}'] = (c - ma) / c
        else:
            feat[f'dist_ma_{n}'] = np.nan

    ema_fast = ta.ema(c, length=9)
    ema_slow = ta.ema(c, length=21)
    if ema_fast is not None and ema_slow is not None:
        feat['ema_cross'] = (ema_fast > ema_slow).astype(int)
        feat['ema_dist'] = (ema_fast - ema_slow) / c
    else:
        feat['ema_cross'] = np.nan
        feat['ema_dist'] = np.nan

    # ── VOLATILITY ──────────────────────────────────────────────
    atr_14 = ta.atr(h, l, c, length=14)
    atr_28 = ta.atr(h, l, c, length=28)
    if atr_14 is not None:
        feat['atr_14_norm'] = atr_14 / c
    else:
        feat['atr_14_norm'] = np.nan
    if atr_28 is not None:
        feat['atr_28_norm'] = atr_28 / c
    else:
        feat['atr_28_norm'] = np.nan
    if atr_14 is not None and atr_28 is not None:
        feat['atr_ratio'] = atr_14 / atr_28
    else:
        feat['atr_ratio'] = np.nan

    log_ret = np.log(c / c.shift(1))
    feat['rvol_12'] = log_ret.rolling(12).std()
    feat['rvol_24'] = log_ret.rolling(24).std()
    feat['rvol_48'] = log_ret.rolling(48).std()
    feat['rvol_ratio'] = feat['rvol_12'] / feat['rvol_48']

    feat['hl_range'] = (h - l) / c

    bb = ta.bbands(c, length=20, std=2)
    if bb is not None:
        bb_upper = bb.iloc[:, 0]
        bb_mid = bb.iloc[:, 1]
        bb_lower = bb.iloc[:, 2]
        feat['bb_width'] = (bb_upper - bb_lower) / bb_mid
        feat['bb_position'] = (c - bb_lower) / (bb_upper - bb_lower + 1e-10)

    # ── MARKET STRUCTURE ────────────────────────────────────────
    for n in [20, 50]:
        feat[f'dist_high_{n}'] = (c - h.rolling(n).max()) / c
        feat[f'dist_low_{n}'] = (c - l.rolling(n).min()) / c

    feat['breakout_20'] = (c > h.shift(1).rolling(20).max()).astype(int)
    feat['breakdown_20'] = (c < l.shift(1).rolling(20).min()).astype(int)

    def rolling_slope(series, n):
        slopes = series.copy() * np.nan
        x = np.arange(n)
        for i in range(n, len(series)):
            y = series.iloc[i - n:i].values
            if not np.any(np.isnan(y)):
                slopes.iloc[i] = np.polyfit(x, y, 1)[0] / series.iloc[i]
        return slopes

    feat['trend_slope_20'] = rolling_slope(c, 20)

    body = abs(c - o)
    range_ = h - l + 1e-10
    feat['body_ratio'] = body / range_
    feat['upper_wick_ratio'] = (h - pd.concat([c, o], axis=1).max(axis=1)) / range_
    feat['lower_wick_ratio'] = (pd.concat([c, o], axis=1).min(axis=1) - l) / range_

    adx = ta.adx(h, l, c, length=14)
    if adx is not None:
        feat['adx'] = adx.iloc[:, 0]

    feat.columns = [f'{col}_{tf}' for col in feat.columns]
    return feat


def compute_time_features(df: pd.DataFrame) -> pd.DataFrame:
    feat = pd.DataFrame(index=df.index)
    feat['hour'] = df.index.hour
    feat['session_asian'] = ((df.index.hour >= 0) & (df.index.hour < 8)).astype(int)
    feat['session_london'] = ((df.index.hour >= 8) & (df.index.hour < 16)).astype(int)
    feat['session_ny'] = ((df.index.hour >= 13) & (df.index.hour < 21)).astype(int)
    feat['session_overlap'] = ((df.index.hour >= 13) & (df.index.hour < 16)).astype(int)
    feat['day_of_week'] = df.index.dayofweek
    feat['is_monday'] = (df.index.dayofweek == 0).astype(int)
    feat['is_friday'] = (df.index.dayofweek == 4).astype(int)
    feat['month'] = df.index.month
    return feat


def compute_cross_tf_features(
    feat_1h: pd.DataFrame,
    feat_4h: pd.DataFrame,
    feat_1d: pd.DataFrame,
) -> pd.DataFrame:
    feat = pd.DataFrame(index=feat_1h.index)

    if 'ema_cross_1H' in feat_1h.columns and 'ema_cross_4H' in feat_4h.columns:
        feat['trend_align_1h_4h'] = (feat_1h['ema_cross_1H'] == feat_4h['ema_cross_4H']).astype(int)
    if 'ema_cross_4H' in feat_4h.columns and 'ema_cross_1D' in feat_1d.columns:
        feat['trend_align_4h_1d'] = (feat_4h['ema_cross_4H'] == feat_1d['ema_cross_1D']).astype(int)
    if all(c in feat.columns for c in ['trend_align_1h_4h', 'trend_align_4h_1d']):
        feat['full_confluence'] = (feat['trend_align_1h_4h'] & feat['trend_align_4h_1d']).astype(int)
    if 'atr_ratio_1H' in feat_1h.columns and 'atr_ratio_1D' in feat_1d.columns:
        feat['vol_expansion'] = (feat_1h['atr_ratio_1H'] > feat_1d['atr_ratio_1D']).astype(int)
    if 'rsi_14_1H' in feat_1h.columns and 'rsi_14_4H' in feat_4h.columns:
        feat['rsi_align_1h_4h'] = (np.sign(feat_1h['rsi_14_1H'] - 50) == np.sign(feat_4h['rsi_14_4H'] - 50)).astype(int)

    scores = []
    for col, df_ in [('rsi_14_1H', feat_1h), ('rsi_14_4H', feat_4h), ('rsi_14_1D', feat_1d)]:
        if col in df_.columns:
            scores.append((df_[col] > 50).astype(int))
    if scores:
        feat['momentum_confluence'] = sum(scores)

    return feat


def compute_cross_pair_correlations(
    closes_1h: dict[str, pd.Series],
    pair: str,
    index: pd.DatetimeIndex,
) -> pd.DataFrame:
    """
    Compute rolling cross-pair correlations and PCA share features.
    Produces all 19 features expected by the model:
      - corr_{OTHER}_{24H|1W} for each other pair (12 features)
      - corr_{SELF}_{24H|1W}  self-correlation = 1.0 (2 features)
      - corr_mean_abs_{24H|1W} (2 features)
      - corr_std_{24H} (1 feature)
      - pc1_share_{24H|1W} (2 features)
    """
    close_df = pd.DataFrame(closes_1h)
    feat = pd.DataFrame(index=index)

    for window, w_name in [(24, '24H'), (168, '1W')]:
        corr_vals = []

        if len(close_df.columns) >= 2:
            rolling_corr = close_df.rolling(window).corr()

            for other_pair in PAIRS:
                col_name = f'corr_{other_pair}_{w_name}'
                if other_pair == pair:
                    # Training skipped self-correlation (left as NaN)
                    feat[col_name] = np.nan
                    continue
                try:
                    corr_series = rolling_corr.xs(pair, level=1)[other_pair]
                    feat[col_name] = corr_series.reindex(index, method='ffill')
                    corr_vals.append(feat[col_name])
                except Exception:
                    feat[col_name] = np.nan
        else:
            # Not enough pairs loaded — fill with NaN
            for other_pair in PAIRS:
                col_name = f'corr_{other_pair}_{w_name}'
                feat[col_name] = np.nan

        if corr_vals:
            corr_matrix = pd.concat(corr_vals, axis=1)
            feat[f'corr_mean_abs_{w_name}'] = corr_matrix.abs().mean(axis=1)
            if w_name == '24H':
                feat[f'corr_std_{w_name}'] = corr_matrix.std(axis=1)
        else:
            feat[f'corr_mean_abs_{w_name}'] = np.nan
            if w_name == '24H':
                feat[f'corr_std_{w_name}'] = np.nan

    # PCA share features — PCA on rolling correlation matrix of raw close prices
    from sklearn.decomposition import PCA as _PCA
    for window, w_name in [(24, '24H'), (168, '1W')]:
        if len(close_df.columns) >= 2:
            # Only compute for the tail — we just need the latest value (ffill handles the rest)
            n_compute = min(10, len(close_df) - window + 1)
            if n_compute <= 0:
                feat[f'pc1_share_{w_name}'] = np.nan
                continue
            tail_indices = close_df.index[-(n_compute):]
            rolling_corr = close_df.tail(window + n_compute).rolling(window).corr()
            pc1_shares = pd.Series(np.nan, index=close_df.index)
            for ts in tail_indices:
                try:
                    corr_mat = rolling_corr.xs(ts, level=0).values
                    if corr_mat.shape[0] >= 2 and not np.any(np.isnan(corr_mat)):
                        pca = _PCA(n_components=1)
                        pca.fit(corr_mat)
                        pc1_shares.loc[ts] = pca.explained_variance_ratio_[0]
                except Exception:
                    pass
            feat[f'pc1_share_{w_name}'] = pc1_shares.reindex(index, method='ffill')
        else:
            feat[f'pc1_share_{w_name}'] = np.nan

    return feat


def build_feature_row(
    ohlcv_by_tf: dict[str, pd.DataFrame],
    closes_1h_all_pairs: dict[str, pd.Series],
    pair: str,
    pair_id: int,
    expected_cols: list[str] | None = None,
) -> pd.DataFrame:
    """
    Build the full feature matrix for a pair, aligned to 1H index.

    If expected_cols is provided (from the model bundle), the output is
    filtered and reordered to match exactly — no extra columns, no missing
    columns, correct order.
    """
    base = ohlcv_by_tf['1H'].copy()

    feat_5m = compute_features(ohlcv_by_tf['5m'], '5m') if '5m' in ohlcv_by_tf else None
    feat_15m = compute_features(ohlcv_by_tf['15m'], '15m') if '15m' in ohlcv_by_tf else None
    feat_1h = compute_features(ohlcv_by_tf['1H'], '1H')
    feat_4h = compute_features(ohlcv_by_tf['4H'], '4H') if '4H' in ohlcv_by_tf else None
    feat_1d = compute_features(ohlcv_by_tf['1D'], '1D') if '1D' in ohlcv_by_tf else None

    feat_time = compute_time_features(base)

    all_features = feat_1h.copy()

    if feat_5m is not None:
        all_features = all_features.join(feat_5m.resample('1h').last(), how='left')
    if feat_15m is not None:
        all_features = all_features.join(feat_15m.resample('1h').last(), how='left')
    if feat_4h is not None:
        all_features = all_features.join(feat_4h.reindex(all_features.index, method='ffill'), how='left')
    if feat_1d is not None:
        all_features = all_features.join(feat_1d.reindex(all_features.index, method='ffill'), how='left')

    if feat_4h is not None and feat_1d is not None:
        feat_4h_aligned = feat_4h.reindex(all_features.index, method='ffill')
        feat_1d_aligned = feat_1d.reindex(all_features.index, method='ffill')
        feat_cross = compute_cross_tf_features(feat_1h, feat_4h_aligned, feat_1d_aligned)
        all_features = all_features.join(feat_cross, how='left')

    all_features = all_features.join(feat_time, how='left')

    if closes_1h_all_pairs:
        feat_corr = compute_cross_pair_correlations(closes_1h_all_pairs, pair, all_features.index)
        all_features = all_features.join(feat_corr, how='left')

    # Defragment to fix PerformanceWarning
    all_features = all_features.copy()

    all_features['pair_id'] = pair_id

    # Align to model's expected columns — exact set, exact order
    if expected_cols is not None:
        for col in expected_cols:
            if col not in all_features.columns:
                all_features[col] = np.nan
        all_features = all_features[expected_cols]

    return all_features
