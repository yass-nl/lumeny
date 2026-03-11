"""
Fast 2-week test for models_3 — 1H predictions.
Models_3 were trained with shift(1) on 4H/1D features, so we do the same here.
Fetch once, compute once, evaluate all timestamps.
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

MODELS_DIR = Path('backend/models_3')
QUANTILES = [0.10, 0.25, 0.50, 0.75, 0.90]
QUANTILE_NAMES = ['Q10', 'Q25', 'Q50', 'Q75', 'Q90']
TEST_DAYS = 14


def derive_p_down(q_vals):
    qs = np.array(QUANTILES)
    if q_vals[0] <= 0 <= q_vals[-1]:
        return float(np.interp(0, q_vals, qs))
    elif q_vals[-1] < 0:
        slope = (qs[-1] - qs[-2]) / (q_vals[-1] - q_vals[-2] + 1e-10)
        return float(np.clip(qs[-1] + slope * (0 - q_vals[-1]), 0.90, 0.999))
    else:
        slope = (qs[1] - qs[0]) / (q_vals[1] - q_vals[0] + 1e-10)
        return float(np.clip(qs[0] + slope * (0 - q_vals[0]), 0.001, 0.10))


async def fetch_all_data():
    now = datetime.now(timezone.utc)
    to_date = now.strftime('%Y-%m-%d')

    tf_fetch_config = {
        '5m':  (5,   'minute', TEST_DAYS + 15),
        '15m': (15,  'minute', TEST_DAYS + 15),
        '1H':  (1,   'hour',   850),
    }

    buffers = {}
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
                    buffers[pair][tf_name] = df
                    logger.info(f'  {pair} {tf_name}: {len(df)} bars')
            except Exception as e:
                logger.warning(f'  {pair} {tf_name} error: {e}')

        # Resample 4H/1D from 1H
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
    """Build features with shift(1) on 4H/1D — matching models_3 training."""
    ohlcv = buffers[pair]

    feat_5m = compute_features(ohlcv['5m'], '5m') if '5m' in ohlcv else None
    feat_15m = compute_features(ohlcv['15m'], '15m') if '15m' in ohlcv else None
    feat_1h = compute_features(ohlcv['1H'], '1H')
    feat_4h = compute_features(ohlcv['4H'], '4H') if '4H' in ohlcv else None
    feat_1d = compute_features(ohlcv['1D'], '1D') if '1D' in ohlcv else None
    feat_time = compute_time_features(ohlcv['1H'])

    all_features = feat_1h.copy()

    if feat_5m is not None:
        all_features = all_features.join(feat_5m.resample('1h').last(), how='left')
    if feat_15m is not None:
        all_features = all_features.join(feat_15m.resample('1h').last(), how='left')

    # shift(1) on 4H/1D — use previous completed bar, not current
    if feat_4h is not None:
        feat_4h_shifted = feat_4h.shift(1)
        all_features = all_features.join(feat_4h_shifted.reindex(all_features.index, method='ffill'), how='left')
    if feat_1d is not None:
        feat_1d_shifted = feat_1d.shift(1)
        all_features = all_features.join(feat_1d_shifted.reindex(all_features.index, method='ffill'), how='left')

    if feat_4h is not None and feat_1d is not None:
        feat_4h_aligned = feat_4h.shift(1).reindex(all_features.index, method='ffill')
        feat_1d_aligned = feat_1d.shift(1).reindex(all_features.index, method='ffill')
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

    if expected_cols is not None:
        for col in expected_cols:
            if col not in all_features.columns:
                all_features[col] = np.nan
        all_features = all_features[expected_cols]

    return all_features


async def main():
    logger.info("=== Fetching data from Polygon ===")
    buffers = await fetch_all_data()

    logger.info("=== Loading models_3 (1H only) ===")
    models = {}
    feature_cols = None
    for q, q_name in zip(QUANTILES, QUANTILE_NAMES):
        path = MODELS_DIR / f'model_1H_Q{int(q * 100)}.joblib'
        bundle = joblib.load(path)
        models[('1H', q_name)] = bundle['model']
        if feature_cols is None:
            feature_cols = bundle['feature_cols']
    logger.info(f"Loaded {len(models)} models, {len(feature_cols)} features")

    now = datetime.now(timezone.utc).replace(tzinfo=None)
    two_weeks_ago = now - timedelta(days=TEST_DAYS)

    results = []

    for pair_idx, pair in enumerate(PAIRS):
        logger.info(f"Processing {pair}...")
        if '1H' not in buffers[pair]:
            continue

        features_df = build_features_for_pair(buffers, pair, pair_idx, feature_cols)
        close_1h = buffers[pair]['1H']['close']

        X_full = features_df[feature_cols].ffill().bfill()
        eval_timestamps = X_full.index[X_full.index >= two_weeks_ago]

        for ts in eval_timestamps:
            try:
                ts_idx = close_1h.index.get_loc(ts)
            except KeyError:
                continue
            if ts_idx + 1 >= len(close_1h):
                continue

            close_t = close_1h.iloc[ts_idx]
            close_t1 = close_1h.iloc[ts_idx + 1]
            actual_ret = np.log(close_t1 / close_t)

            row = X_full.loc[[ts]]

            q_preds = []
            for q, q_name in zip(QUANTILES, QUANTILE_NAMES):
                model = models[('1H', q_name)]
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
                'pair': pair,
                'pred_dir': pred_dir,
                'confidence': confidence,
                'p_down': p_down,
                'actual_ret': actual_ret,
                'actual_ret_pct': actual_ret * 100,
                'actual_dir': actual_dir,
                'correct': correct,
                'entry_price': close_t,
                'exit_price': close_t1,
                'q10': q_vals[0] * 100,
                'q25': q_vals[1] * 100,
                'q50': q_vals[2] * 100,
                'q75': q_vals[3] * 100,
                'q90': q_vals[4] * 100,
            })

        logger.info(f"  {pair}: {sum(1 for r in results if r['pair'] == pair)} predictions")

    if not results:
        logger.error("No results!")
        return

    df = pd.DataFrame(results)
    df.to_csv('models3_2week_results.csv', index=False)

    print("\n" + "=" * 80)
    print("MODELS_3 TEST — 1H predictions, last 2 weeks (shift(1) on 4H/1D)")
    print("=" * 80)

    print(f"\nTotal predictions: {len(df)}")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"Pairs: {df['pair'].nunique()}")

    print("\n--- Overall Accuracy ---")
    acc = df['correct'].mean()
    print(f"  All: {acc:.4f} ({df['correct'].sum()}/{len(df)})")

    print("\n--- Accuracy by Confidence Threshold ---")
    print(f"{'Threshold':<12} {'Accuracy':<12} {'Count':<8} {'% of Total':<12}")
    print("-" * 44)
    for threshold in [0.55, 0.60, 0.65, 0.70, 0.80]:
        hdf = df[df['confidence'] >= threshold]
        if len(hdf) == 0:
            print(f"  >= {threshold:.0%}     {'N/A':<12} {0:<8} {'0.0%':<12}")
        else:
            a = hdf['correct'].mean()
            pct = len(hdf) / len(df) * 100
            print(f"  >= {threshold:.0%}     {a:.4f}       {len(hdf):<8} {pct:.1f}%")

    print("\n--- Accuracy by Pair ---")
    print(f"{'Pair':<10} {'All':>12} {'>=65%':>16} {'>=70%':>16}")
    print("-" * 58)
    for pair in PAIRS:
        pdf = df[df['pair'] == pair]
        all_acc = f"{pdf['correct'].mean():.3f} ({len(pdf)})"

        hc65 = pdf[pdf['confidence'] >= 0.65]
        hc65_s = f"{hc65['correct'].mean():.3f} ({len(hc65)})" if len(hc65) > 0 else "N/A"

        hc70 = pdf[pdf['confidence'] >= 0.70]
        hc70_s = f"{hc70['correct'].mean():.3f} ({len(hc70)})" if len(hc70) > 0 else "N/A"

        print(f"{pair:<10} {all_acc:>12} {hc65_s:>16} {hc70_s:>16}")

    print("\n--- Directional Bias ---")
    bull_pct = (df['pred_dir'] == 'bullish').mean() * 100
    print(f"  {bull_pct:.1f}% bullish, {100-bull_pct:.1f}% bearish")

    print(f"\n--- Mean Confidence: {df['confidence'].mean():.4f} ---")

    print("\n--- Simple PnL (1 unit per trade, no spread) ---")
    for threshold in [0.55, 0.60, 0.65, 0.70]:
        hdf = df[df['confidence'] >= threshold]
        if len(hdf) == 0:
            continue
        pnl = hdf.apply(lambda r: r['actual_ret_pct'] if r['pred_dir'] == 'bullish' else -r['actual_ret_pct'], axis=1)
        print(f"  >= {threshold:.0%}: total={pnl.sum():+.4f}%, mean={pnl.mean():+.6f}%, trades={len(hdf)}")

    print("\n" + "=" * 80)
    print(f"Results saved to models3_2week_results.csv")
    print("=" * 80)


if __name__ == '__main__':
    asyncio.run(main())
