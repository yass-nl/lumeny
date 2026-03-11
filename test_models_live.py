"""
Test models_2 (old models with lookahead) on real live Polygon data from the last 2 weeks.
"""
import os, sys, asyncio, logging
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv

load_dotenv()

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from features import (
    PAIRS, compute_features, compute_time_features,
    compute_cross_tf_features, compute_cross_pair_correlations,
    resample_ohlcv
)
from data_service import fetch_historical_bars

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

MODELS_DIR = Path('backend/models_2')
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
    """Fetch 5m, 15m, 1H for all pairs, resample 4H/1D from 1H."""
    now = datetime.now(timezone.utc)
    to_date = now.strftime('%Y-%m-%d')

    tf_fetch_config = {
        '5m':  (5,   'minute', 20),     # need ~2 weeks + buffer for indicators
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
                    # Weekend filter
                    df = df[~((df.index.dayofweek == 5) |
                              ((df.index.dayofweek == 6) & (df.index.hour < 21)))]
                    # Drop incomplete bars
                    bar_dur = timedelta(minutes=tf_durations_min[tf_name])
                    now_naive = now.replace(tzinfo=None)
                    df = df[df.index + bar_dur <= now_naive]
                    buffers[pair][tf_name] = df
                    logger.info(f'  {pair} {tf_name}: {len(df)} bars')
                else:
                    logger.warning(f'  {pair} {tf_name}: no data')
            except Exception as e:
                logger.warning(f'  {pair} {tf_name} error: {e}')

        # Resample 4H and 1D from 1H
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


def build_features_for_pair(buffers, pair, pair_id, expected_cols):
    """Build features using the same logic as build_feature_row but WITHOUT shift(1) on 4H/1D."""
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
    # NO shift(1) on 4H/1D — old way, matching models_2 training
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

    # Cross-pair correlations
    closes_1h = {}
    for p in PAIRS:
        if p in buffers and '1H' in buffers[p]:
            closes_1h[p] = buffers[p]['1H']['close']
    if closes_1h:
        feat_corr = compute_cross_pair_correlations(closes_1h, pair, all_features.index)
        all_features = all_features.join(feat_corr, how='left')

    all_features = all_features.copy()
    all_features['pair_id'] = pair_id

    # Align to expected columns
    if expected_cols is not None:
        for col in expected_cols:
            if col not in all_features.columns:
                all_features[col] = np.nan
        all_features = all_features[expected_cols]

    return all_features


def load_models():
    """Load all models from models_2."""
    models = {}
    feature_cols = None
    for horizon in HORIZONS:
        for q, q_name in zip(QUANTILES, QUANTILE_NAMES):
            path = MODELS_DIR / f'model_{horizon}_Q{int(q * 100)}.joblib'
            bundle = joblib.load(path)
            models[(horizon, q_name)] = bundle['model']
            if feature_cols is None:
                feature_cols = bundle['feature_cols']
    return models, feature_cols


def run_predictions(all_features_df, models, feature_cols, close_1h):
    """
    For each 1H timestamp in the last 2 weeks, run predictions and compute actual returns.
    """
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    two_weeks_ago = now - timedelta(days=14)

    # Filter to last 2 weeks
    mask = all_features_df.index >= two_weeks_ago
    eval_timestamps = all_features_df.index[mask]

    # Horizon in number of 1H bars
    horizon_bars = {'1H': 1, '4H': 4, '1D': 24}

    results = []

    X_full = all_features_df[feature_cols].ffill().bfill()

    for ts in eval_timestamps:
        if ts not in X_full.index:
            continue
        row = X_full.loc[[ts]]

        for horizon in HORIZONS:
            n_bars = horizon_bars[horizon]
            # Find the close at T and T+n
            try:
                ts_idx = close_1h.index.get_loc(ts)
            except KeyError:
                continue

            if ts_idx + n_bars >= len(close_1h):
                continue  # can't compute actual return

            close_t = close_1h.iloc[ts_idx]
            close_t_n = close_1h.iloc[ts_idx + n_bars]
            actual_ret = np.log(close_t_n / close_t)

            # Run 5 quantile models
            q_preds = []
            for q, q_name in zip(QUANTILES, QUANTILE_NAMES):
                model = models[(horizon, q_name)]
                q_preds.append(float(model.predict(row)[0]))

            q_vals = np.sort(np.array(q_preds))
            p_down = derive_p_down(q_vals)
            p_up = 1.0 - p_down

            if p_down > p_up:
                pred_dir = 'bearish'
                confidence = p_down
            else:
                pred_dir = 'bullish'
                confidence = p_up

            actual_dir = 'bearish' if actual_ret < 0 else 'bullish'
            correct = (pred_dir == actual_dir)

            results.append({
                'timestamp': ts,
                'horizon': horizon,
                'pred_dir': pred_dir,
                'confidence': confidence,
                'p_down': p_down,
                'actual_ret': actual_ret,
                'actual_dir': actual_dir,
                'correct': correct,
                'q50': q_vals[2],
            })

    return pd.DataFrame(results)


async def main():
    logger.info("=== Fetching data from Polygon ===")
    buffers = await fetch_all_data()

    logger.info("=== Loading models_2 ===")
    models, feature_cols = load_models()

    logger.info("=== Computing features and running predictions ===")
    all_results = []

    for pair_idx, pair in enumerate(PAIRS):
        logger.info(f"Processing {pair}...")
        if '1H' not in buffers[pair]:
            logger.warning(f"  Skipping {pair} — no 1H data")
            continue

        features_df = build_features_for_pair(buffers, pair, pair_idx, feature_cols)
        close_1h = buffers[pair]['1H']['close']

        results_df = run_predictions(features_df, models, feature_cols, close_1h)
        if not results_df.empty:
            results_df['pair'] = pair
            all_results.append(results_df)
            logger.info(f"  {pair}: {len(results_df)} predictions")

    if not all_results:
        logger.error("No results!")
        return

    df = pd.concat(all_results, ignore_index=True)

    # ── Report ──
    print("\n" + "=" * 80)
    print("MODELS_2 LIVE TEST RESULTS — Last 2 Weeks of Polygon Data")
    print("=" * 80)

    print(f"\nTotal predictions: {len(df)}")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"Pairs: {df['pair'].nunique()}")

    # Overall accuracy by horizon
    print("\n--- Overall Accuracy by Horizon ---")
    for horizon in HORIZONS:
        hdf = df[df['horizon'] == horizon]
        if len(hdf) == 0:
            continue
        acc = hdf['correct'].mean()
        print(f"  {horizon}: {acc:.4f} ({hdf['correct'].sum()}/{len(hdf)})")

    # Accuracy at different confidence thresholds
    thresholds = [0.55, 0.60, 0.65, 0.70, 0.80]
    print("\n--- Accuracy by Confidence Threshold ---")
    print(f"{'Threshold':<12} {'Horizon':<8} {'Accuracy':<12} {'Count':<8} {'% of Total':<12}")
    print("-" * 52)
    for threshold in thresholds:
        for horizon in HORIZONS:
            hdf = df[(df['horizon'] == horizon) & (df['confidence'] >= threshold)]
            if len(hdf) == 0:
                print(f"  >= {threshold:.0%}     {horizon:<8} {'N/A':<12} {0:<8} {'0.0%':<12}")
                continue
            total_h = len(df[df['horizon'] == horizon])
            acc = hdf['correct'].mean()
            pct = len(hdf) / total_h * 100
            print(f"  >= {threshold:.0%}     {horizon:<8} {acc:.4f}       {len(hdf):<8} {pct:.1f}%")

    # Accuracy by pair and horizon
    print("\n--- Accuracy by Pair and Horizon ---")
    print(f"{'Pair':<10}", end="")
    for h in HORIZONS:
        print(f"  {h:<18}", end="")
    print()
    print("-" * 64)
    for pair in PAIRS:
        pdf = df[df['pair'] == pair]
        print(f"{pair:<10}", end="")
        for h in HORIZONS:
            hpdf = pdf[pdf['horizon'] == h]
            if len(hpdf) == 0:
                print(f"  {'N/A':<18}", end="")
            else:
                acc = hpdf['correct'].mean()
                print(f"  {acc:.4f} ({len(hpdf):>3})     ", end="")
        print()

    # Directional bias check
    print("\n--- Directional Bias (% Bullish Predictions) ---")
    for horizon in HORIZONS:
        hdf = df[df['horizon'] == horizon]
        bull_pct = (hdf['pred_dir'] == 'bullish').mean() * 100
        print(f"  {horizon}: {bull_pct:.1f}% bullish, {100-bull_pct:.1f}% bearish")

    # Actual direction distribution
    print("\n--- Actual Direction Distribution ---")
    for horizon in HORIZONS:
        hdf = df[df['horizon'] == horizon]
        bull_pct = (hdf['actual_dir'] == 'bullish').mean() * 100
        print(f"  {horizon}: {bull_pct:.1f}% up, {100-bull_pct:.1f}% down")

    # Mean confidence
    print("\n--- Mean Confidence ---")
    for horizon in HORIZONS:
        hdf = df[df['horizon'] == horizon]
        print(f"  {horizon}: {hdf['confidence'].mean():.4f}")

    # High confidence accuracy (>=65%) by pair
    print("\n--- High Confidence (>=65%) Accuracy by Pair ---")
    print(f"{'Pair':<10}", end="")
    for h in HORIZONS:
        print(f"  {h:<18}", end="")
    print()
    print("-" * 64)
    for pair in PAIRS:
        pdf = df[(df['pair'] == pair) & (df['confidence'] >= 0.65)]
        print(f"{pair:<10}", end="")
        for h in HORIZONS:
            hpdf = pdf[pdf['horizon'] == h]
            if len(hpdf) == 0:
                print(f"  {'N/A':<18}", end="")
            else:
                acc = hpdf['correct'].mean()
                print(f"  {acc:.4f} ({len(hpdf):>3})     ", end="")
        print()

    print("\n" + "=" * 80)


if __name__ == '__main__':
    asyncio.run(main())
