"""
Verify predictions using notebook logic (same as test_models2_live.py)
but for the current last closed 1H candle only.
Uses backend/models/ (same as Railway).
"""
import os, sys, asyncio, logging
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from features import (
    PAIRS, compute_features, compute_time_features,
    compute_cross_tf_features, compute_cross_pair_correlations,
    resample_ohlcv
)
from data_service import fetch_historical_bars

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

MODELS_DIR = Path('backend/models')
HORIZONS = ['1H', '4H', '1D']
QUANTILES = [0.10, 0.25, 0.50, 0.75, 0.90]
QUANTILE_NAMES = ['Q10', 'Q25', 'Q50', 'Q75', 'Q90']


def derive_p_down(q_vals):
    qs = np.array(QUANTILES)
    vals = q_vals
    if vals[0] <= 0 <= vals[-1]:
        return float(np.interp(0, vals, qs))
    elif vals[-1] < 0:
        slope = (qs[-1] - qs[-2]) / (vals[-1] - vals[-2] + 1e-10)
        p_down = qs[-1] + slope * (0 - vals[-1])
        return float(np.clip(p_down, 0.90, 0.999))
    else:
        slope = (qs[1] - qs[0]) / (vals[1] - vals[0] + 1e-10)
        p_down = qs[0] + slope * (0 - vals[0])
        return float(np.clip(p_down, 0.001, 0.10))


async def fetch_all_data():
    """Fetch data exactly like test_models2_live.py."""
    now = datetime.now(timezone.utc)
    to_date = now.strftime('%Y-%m-%d')

    tf_fetch_config = {
        '5m':  (5,   'minute', 20),
        '15m': (15,  'minute', 20),
        '1H':  (1,   'hour',   850),
    }

    buffers = {}
    tf_durations_min = {'5m': 5, '15m': 15, '1H': 60}

    for pair in PAIRS:
        logger.info(f'Fetching {pair}...')
        buffers[pair] = {}

        for tf_name, (multiplier, timespan, lookback_days) in tf_fetch_config.items():
            from_date = (now - timedelta(days=lookback_days)).strftime('%Y-%m-%d')
            await asyncio.sleep(0.3)
            try:
                df = await fetch_historical_bars(pair, multiplier, timespan, from_date, to_date)
                if not df.empty:
                    df = df[~((df.index.dayofweek == 5) |
                              ((df.index.dayofweek == 6) & (df.index.hour < 21)))]
                    bar_dur = timedelta(minutes=tf_durations_min[tf_name])
                    now_naive = now.replace(tzinfo=None)
                    df = df[df.index + bar_dur <= now_naive]
                    buffers[pair][tf_name] = df
                    logger.info(f'  {pair} {tf_name}: {len(df)} bars')
            except Exception as e:
                logger.warning(f'  {pair} {tf_name} error: {e}')

        if '1H' in buffers[pair] and not buffers[pair]['1H'].empty:
            df_1h = buffers[pair]['1H']
            for tf_name, rule in [('4H', '4h'), ('1D', '1D')]:
                try:
                    df_r = resample_ohlcv(df_1h, rule)
                    if not df_r.empty:
                        buffers[pair][tf_name] = df_r
                        logger.info(f'  {pair} {tf_name}: {len(df_r)} bars (resampled)')
                except Exception as e:
                    logger.warning(f'  {pair} {tf_name} resample error: {e}')

    return buffers


def build_features_notebook_style(buffers, pair, pair_id, expected_cols):
    """Exact same feature building as test_models2_live.py (notebook logic)."""
    ohlcv = buffers[pair]
    base = ohlcv['1H'].copy()

    feat_5m = compute_features(ohlcv['5m'], '5m') if '5m' in ohlcv else None
    feat_15m = compute_features(ohlcv['15m'], '15m') if '15m' in ohlcv else None
    feat_1h = compute_features(ohlcv['1H'], '1H')
    feat_4h = compute_features(ohlcv['4H'], '4H') if '4H' in ohlcv else None
    feat_1d = compute_features(ohlcv['1D'], '1D') if '1D' in ohlcv else None

    feat_time = compute_time_features(base)

    all_features = feat_1h.copy()

    if feat_5m is not None:
        all_features = all_features.join(feat_5m.resample('1h').last(), how='left')
    if feat_15m is not None:
        all_features = all_features.join(feat_15m.resample('1h').last(), how='left')
    # NO shift(1) — same as training
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

    closes_1h = {}
    for p in PAIRS:
        if p in buffers and '1H' in buffers[p]:
            closes_1h[p] = buffers[p]['1H']['close']
    if closes_1h:
        feat_corr = compute_cross_pair_correlations(closes_1h, pair, all_features.index)
        all_features = all_features.join(feat_corr, how='left')

    all_features = all_features.copy()
    all_features['pair_id'] = pair_id

    if expected_cols is not None:
        for col in expected_cols:
            if col not in all_features.columns:
                all_features[col] = np.nan
        all_features = all_features[expected_cols]

    return all_features


async def main():
    logger.info("=== Fetching data from Polygon (notebook style) ===")
    buffers = await fetch_all_data()

    logger.info("=== Loading models from backend/models/ ===")
    models = {}
    feature_cols = None
    for horizon in HORIZONS:
        for q, q_name in zip(QUANTILES, QUANTILE_NAMES):
            path = MODELS_DIR / f'model_{horizon}_Q{int(q * 100)}.joblib'
            bundle = joblib.load(path)
            models[(horizon, q_name)] = bundle['model']
            if feature_cols is None:
                feature_cols = bundle['feature_cols']
    logger.info(f"Loaded {len(models)} models")

    print("\n" + "=" * 70)
    print("PREDICTIONS — notebook logic, backend/models/, last closed 1H candle")
    print("=" * 70)

    for pair_idx, pair in enumerate(PAIRS):
        if '1H' not in buffers[pair]:
            print(f"\n{pair}: NO DATA")
            continue

        features_df = build_features_notebook_style(buffers, pair, pair_idx, feature_cols)
        last_ts = features_df.index[-1]
        entry_price = float(buffers[pair]['1H']['close'].iloc[-1])

        X = features_df[feature_cols].ffill().bfill()
        row = X.iloc[[-1]]

        print(f"\n{pair}  |  T={last_ts}  |  entry={entry_price}")

        for horizon in HORIZONS:
            q_preds = []
            for q, q_name in zip(QUANTILES, QUANTILE_NAMES):
                model = models[(horizon, q_name)]
                q_preds.append(float(model.predict(row)[0]))

            q_vals = np.sort(np.array(q_preds))
            p_down = derive_p_down(q_vals)
            p_up = 1.0 - p_down

            if p_down > p_up:
                direction = 'bearish'
                prob = p_down
            else:
                direction = 'bullish'
                prob = p_up

            print(f"  {horizon}: {direction:>8s}  prob={prob:.4f}  "
                  f"Q10={q_vals[0]*100:.4f}  Q25={q_vals[1]*100:.4f}  "
                  f"Q50={q_vals[2]*100:.4f}  Q75={q_vals[3]*100:.4f}  "
                  f"Q90={q_vals[4]*100:.4f}")

    print("\n" + "=" * 70)
    print("Compare with Railway predictions:")
    print("  - Same direction? Same probability tier?")
    print("  - Small quantile diffs from 5m/15m fetch timing are expected.")
    print("=" * 70)


if __name__ == '__main__':
    asyncio.run(main())
