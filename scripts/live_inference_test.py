"""
Live inference test — fetches current data from Polygon REST, runs inference,
and reports:
  1. What data came back (dates, freshness)
  2. What the model predicts RIGHT NOW
  3. Whether the data is stale (lag detected)

Run from lumeny/ root:
    python scripts/live_inference_test.py
"""

import sys
sys.path.insert(0, 'backend')

import asyncio
import os
from pathlib import Path
from datetime import datetime, timezone, timedelta

# Load .env
env_path = Path(__file__).parent.parent / '.env'
if env_path.exists():
    for line in env_path.read_text().splitlines():
        if '=' in line and not line.startswith('#'):
            k, v = line.split('=', 1)
            os.environ.setdefault(k.strip(), v.strip())

import pandas as pd
import numpy as np

from data_service import fetch_historical_bars, CandleBuffer
from features import build_feature_row, resample_ohlcv, PAIRS
from inference import Predictor

# Override models dir to use local v2 models
import inference as _inf
from pathlib import Path as _Path
_inf.MODELS_DIR = _Path('backend/models_2')

NOW = datetime.now(timezone.utc)
print(f"Test started at: {NOW.isoformat()}")
print(f"Today (UTC): {NOW.strftime('%Y-%m-%d')}")


async def fetch_pair_data(pair: str) -> dict[str, pd.DataFrame]:
    """Fetch all timeframes for one pair, same logic as CandleBuffer.initialize()."""
    to_date = NOW.strftime('%Y-%m-%d')
    tf_fetch_config = {
        '5m':  (5,   'minute', 3),
        '15m': (15,  'minute', 7),
        '1H':  (1,   'hour',   850),
    }
    buffers = {}
    for tf_name, (multiplier, timespan, lookback_days) in tf_fetch_config.items():
        from_date = (NOW - timedelta(days=lookback_days)).strftime('%Y-%m-%d')
        try:
            df = await fetch_historical_bars(pair, multiplier, timespan, from_date, to_date)
            if not df.empty:
                df = df[df.index.dayofweek < 5]
                if len(df) > 1:
                    df = df.iloc[:-1]
                buffers[tf_name] = df
        except Exception as e:
            print(f"  ERROR fetching {pair} {tf_name}: {e}")

    if '1H' in buffers and not buffers['1H'].empty:
        df_1h = buffers['1H']
        for tf_name, rule in [('4H', '4h'), ('1D', '1D')]:
            df_r = resample_ohlcv(df_1h, rule)
            if not df_r.empty:
                if len(df_r) > 1:
                    df_r = df_r.iloc[:-1]
                buffers[tf_name] = df_r
    return buffers


async def main():
    print("=" * 70)
    print("LIVE INFERENCE TEST")
    print("=" * 70)

    # ── Step 1: Fetch data for all pairs ──
    print(f"\n[1] Fetching data from Polygon REST...")
    all_buffers = {}
    for pair in PAIRS:
        print(f"  {pair}...", end=' ', flush=True)
        buf = await fetch_pair_data(pair)
        all_buffers[pair] = buf
        if '1H' in buf and not buf['1H'].empty:
            last_ts = buf['1H'].index[-1]
            lag_hours = (NOW - last_ts.tz_localize('UTC')).total_seconds() / 3600
            status = "OK" if lag_hours < 3 else f"STALE! lag={lag_hours:.1f}h"
            print(f"{len(buf['1H'])} 1H bars, last={last_ts}, lag={lag_hours:.1f}h [{status}]")
        else:
            print("NO 1H DATA")

    # ── Step 2: Check data freshness ──
    print(f"\n[2] Data freshness summary:")
    stale_pairs = []
    for pair in PAIRS:
        if '1H' not in all_buffers.get(pair, {}):
            continue
        last_ts = all_buffers[pair]['1H'].index[-1]
        lag_hours = (NOW - last_ts.tz_localize('UTC')).total_seconds() / 3600
        if lag_hours > 3:
            stale_pairs.append((pair, lag_hours, last_ts))

    if stale_pairs:
        print(f"  STALE DATA DETECTED for {len(stale_pairs)} pairs:")
        for pair, lag, ts in stale_pairs:
            print(f"    {pair}: last candle={ts}, lag={lag:.1f}h")
        print(f"\n  This confirms the Polygon REST API is returning delayed forex data.")
        print(f"  Predictions made with this data would be based on stale market conditions.")
    else:
        print(f"  All pairs have fresh data (lag < 3h). Data is current.")

    # ── Step 3: Run inference anyway (to see what model produces) ──
    print(f"\n[3] Running inference with available data (may be stale)...")

    closes_1h = {}
    for pair in PAIRS:
        if '1H' in all_buffers.get(pair, {}):
            closes_1h[pair] = all_buffers[pair]['1H']['close']

    predictor = Predictor()
    PAIR_IDS = {pair: i for i, pair in enumerate(PAIRS)}

    print(f"\n  {'Pair':<10} {'Last 1H bar':<22} {'Dir 1H':<12} {'Prob':<8} {'Q50%':<10} {'Lag (h)':<10} {'Conv'}")
    print(f"  {'-'*85}")

    results = {}
    for pair in PAIRS:
        buf = all_buffers.get(pair, {})
        if '1H' not in buf or buf['1H'].empty:
            print(f"  {pair:<10} NO DATA")
            continue

        last_ts = buf['1H'].index[-1]
        lag_hours = (NOW - last_ts.tz_localize('UTC')).total_seconds() / 3600
        last_close = buf['1H']['close'].iloc[-1]

        try:
            features_df = build_feature_row(
                buf, closes_1h, pair, PAIR_IDS[pair],
                expected_cols=predictor.feature_cols,
            )
            result = predictor.predict(features_df, pair)
            results[pair] = result

            h = result['horizons']['1H']
            conv = "LOW" if h['low_conviction'] else "OK"
            lag_flag = "STALE" if lag_hours > 3 else ""
            print(f"  {pair:<10} {str(last_ts):<22} {h['direction']:<12} {h['probability']:<8.4f} {h['expected_move_pct']:<10.4f} {lag_hours:<10.1f} {conv} {lag_flag}")
        except Exception as e:
            print(f"  {pair:<10} ERROR: {e}")

    # ── Step 4: Detailed output for EURUSD ──
    print(f"\n[4] Detailed output — EURUSD 1H:")
    if 'EURUSD' in results:
        h = results['EURUSD']['horizons']['1H']
        buf = all_buffers['EURUSD']
        last_ts = buf['1H'].index[-1]
        lag_hours = (NOW - last_ts.tz_localize('UTC')).total_seconds() / 3600
        print(f"  Last 1H candle : {last_ts}")
        print(f"  Lag from now   : {lag_hours:.1f} hours")
        print(f"  Last close     : {buf['1H']['close'].iloc[-1]:.5f}")
        print(f"  Direction      : {h['direction']}")
        print(f"  Probability    : {h['probability']}")
        print(f"  Q10/Q50/Q90    : {h['quantiles']['Q10']} / {h['quantiles']['Q50']} / {h['quantiles']['Q90']}")
        print(f"  Low conviction : {h['low_conviction']}")
        print(f"  Signal         : {h['signal_strength']}")
        if lag_hours > 3:
            print(f"\n  *** WARNING: Data is {lag_hours:.1f}h stale. This prediction is based on")
            print(f"  *** market conditions from {last_ts}, not current conditions.")
            print(f"  *** This is the root cause of poor live accuracy.")
    else:
        print("  No result for EURUSD.")

    # ── Step 5: Compare 5m data freshness vs 1H ──
    print(f"\n[5] 5m data freshness (should match 1H lag):")
    pair = 'EURUSD'
    if '5m' in all_buffers.get(pair, {}):
        df5 = all_buffers[pair]['5m']
        last_5m = df5.index[-1]
        lag_5m = (NOW - last_5m.tz_localize('UTC')).total_seconds() / 3600
        print(f"  Last 5m bar: {last_5m} (lag={lag_5m:.1f}h)")
    else:
        print("  No 5m data for EURUSD.")

    print(f"\n{'=' * 70}")
    print("DIAGNOSIS COMPLETE")
    print(f"{'=' * 70}")
    if stale_pairs:
        print(f"CONFIRMED: Polygon REST returns stale forex data.")
        print(f"The {len(stale_pairs)} pairs all have data lagging by hours.")
        print(f"Fix: switch to a data source with current data (S3 flat files,")
        print(f"     WebSocket, or another provider like TwelveData).")
    else:
        print(f"Data is fresh. If accuracy is still poor, the issue is elsewhere.")


if __name__ == '__main__':
    asyncio.run(main())
