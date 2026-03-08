"""
Single-Cycle Forward Test
=========================
Simulates exactly one Railway prediction cycle at a specific past hour,
resolves against real Polygon prices, and reports accuracy / MCE / calibration.

Usage:
    cd /c/Users/noual/lumeny/backend
    POLYGON_S3_SECRET_KEY=... python ../scripts/single_cycle_test.py
    POLYGON_S3_SECRET_KEY=... python ../scripts/single_cycle_test.py --candle 2026-02-14T20:00:00

The candle time is the last CLOSED 1H bar (T).
Predictions are resolved at T+1H, T+4H, T+24H.
"""

import argparse
import asyncio
import os
import sys
import warnings
from datetime import datetime, timezone, timedelta
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

BACKEND_DIR = Path(__file__).parent.parent / 'backend'
sys.path.insert(0, str(BACKEND_DIR))

PAIRS = ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD', 'USDCAD', 'NZDUSD']
HORIZONS = ['1H', '4H', '1D']
HORIZON_HOURS = {'1H': 1, '4H': 4, '1D': 24}
PAIR_IDS = {pair: i for i, pair in enumerate(PAIRS)}

# ── Polygon helpers ───────────────────────────────────────────────────────────

API_KEY = os.environ.get('POLYGON_API_KEY', os.environ.get('POLYGON_S3_SECRET_KEY', ''))


async def fetch_1h_bars(pair: str, from_dt: datetime, to_dt: datetime) -> pd.DataFrame:
    """Fetch 1H OHLCV bars from Polygon for a given window."""
    ticker = f'C:{pair}'
    from_date = (from_dt - timedelta(days=1)).strftime('%Y-%m-%d')
    to_date = (to_dt + timedelta(days=1)).strftime('%Y-%m-%d')
    url = f'https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/hour/{from_date}/{to_date}'
    params = {'apiKey': API_KEY, 'limit': 50000, 'sort': 'asc'}

    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get(url, params=params)
        resp.raise_for_status()
        data = resp.json()

    results = data.get('results', [])
    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results)
    df['timestamp'] = pd.to_datetime(df['t'], unit='ms', utc=True).dt.tz_localize(None)
    df = df.set_index('timestamp').rename(columns={'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume'})
    df = df[['open', 'high', 'low', 'close', 'volume']]
    # Drop weekends
    df = df[df.index.dayofweek < 5]
    return df


async def fetch_close_at(pair: str, at_time: datetime) -> float | None:
    """Fetch close price of the 1H bar whose open <= at_time. Exact same logic as resolve_outcomes()."""
    ticker = f'C:{pair}'
    from_date = (at_time - timedelta(days=1)).strftime('%Y-%m-%d')
    to_date = (at_time + timedelta(days=1)).strftime('%Y-%m-%d')
    url = f'https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/hour/{from_date}/{to_date}'
    params = {'apiKey': API_KEY, 'limit': 50000, 'sort': 'asc'}

    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.get(url, params=params)
        resp.raise_for_status()
        data = resp.json()

    results = data.get('results', [])
    if not results:
        return None

    target_ts = int(at_time.timestamp() * 1000)
    best = None
    for bar in results:
        if bar['t'] <= target_ts:
            best = bar
        else:
            break

    if best is not None:
        return float(best['c'])
    return float(results[0]['c'])


# ── Build buffer at candle T ──────────────────────────────────────────────────

def resample_ohlcv(df_1h: pd.DataFrame, rule: str) -> pd.DataFrame:
    return df_1h.resample(rule, label='left', closed='left').agg({
        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum',
    }).dropna(subset=['open'])


async def build_buffers_at(candle_time: datetime) -> dict:
    """
    Build OHLCV buffers exactly as CandleBuffer.initialize() does at candle_time T.
    - Fetches ~90 days of history ending at T (inclusive)
    - Applies iloc[:-1] to 1H to drop the incomplete bar (T+1 which would be open)
      Wait — at candle_time T, bar T is the LAST CLOSED bar. We include up to T.
      CandleBuffer drops the last bar of whatever it fetches (the incomplete open bar).
      So we fetch up to T+1H open (exclusive), meaning we fetch bars up to and including T,
      then iloc[:-1] drops T, leaving T-1 as last.

      Actually: CandleBuffer fetches "now" worth of data, and the last bar is the currently
      OPEN bar (incomplete). It drops that. So at wall-clock time T+HH:05,
      last complete bar = T. We want to simulate being at wall-clock T+00:05,
      so we fetch bars up to T+1H (exclusive), iloc[:-1] drops the T+1H open bar,
      leaving T as the last complete bar. Correct.
    - Resamples 4H/1D from 1H (no extra drop — fixed bug)
    """
    # Fetch 30 days of history ending just after candle_time.
    # Polygon free tier covers ~45 days from today, so keep the window tight.
    fetch_end = candle_time + timedelta(hours=2)
    fetch_start = candle_time - timedelta(days=30)

    print(f'Fetching 90d of 1H data for all pairs up to {candle_time} UTC...')

    buffers = {}
    for pair in PAIRS:
        df = await fetch_1h_bars(pair, fetch_start, fetch_end)
        if df.empty:
            print(f'  {pair}: no data')
            continue

        # Keep only bars up to and including candle_time (index is tz-naive).
        # We manually control the window so there's no incomplete bar — no iloc[:-1] needed.
        candle_naive = candle_time.replace(tzinfo=None)
        df_1h = df[df.index <= candle_naive]

        if len(df_1h) < 2:
            print(f'  {pair}: insufficient data')
            continue

        last = df_1h.index[-1]
        print(f'  {pair}: {len(df_1h)} 1H bars, last closed = {last}')

        buffers[pair] = {'1H': df_1h}

        # Resample 4H/1D from 1H — no extra drop (fixed)
        for tf_name, rule in [('4H', '4h'), ('1D', '1D')]:
            df_resampled = resample_ohlcv(df_1h, rule)
            if not df_resampled.empty:
                buffers[pair][tf_name] = df_resampled
        # 5m/15m not included — build_feature_row handles missing TFs gracefully

    return buffers


# ── Run inference ─────────────────────────────────────────────────────────────

def run_inference(buffers: dict) -> dict:
    import inference as _inf_module
    _inf_module.MODELS_DIR = BACKEND_DIR / 'models_2'

    from inference import Predictor
    from features import build_feature_row

    predictor = Predictor()

    # Build cross-pair closes (needed for correlation features)
    closes_1h = {pair: buffers[pair]['1H']['close'] for pair in buffers if '1H' in buffers[pair]}

    predictions = {}
    for pair in PAIRS:
        if pair not in buffers:
            continue
        try:
            features_df = build_feature_row(
                buffers[pair], closes_1h, pair, PAIR_IDS[pair],
                expected_cols=predictor.feature_cols,
            )
            if features_df.empty:
                print(f'  {pair}: empty features')
                continue
            result = predictor.predict(features_df, pair)
            predictions[pair] = result
        except Exception as e:
            print(f'  {pair}: inference error — {e}')

    return predictions


# ── Resolve ───────────────────────────────────────────────────────────────────

async def resolve_predictions(predictions: dict, candle_time: datetime, entry_prices: dict) -> list[dict]:
    """Fetch exit prices and compute actual_return + correct for each prediction."""
    rows = []
    for pair, result in predictions.items():
        entry_price = entry_prices.get(pair)
        if entry_price is None:
            continue

        for horizon in HORIZONS:
            if horizon not in result['horizons']:
                continue
            h = result['horizons'][horizon]
            matures_at = candle_time + timedelta(hours=HORIZON_HOURS[horizon])

            exit_price = await fetch_close_at(pair, matures_at)
            if exit_price is None:
                print(f'  {pair}/{horizon}: no exit price at {matures_at}')
                continue

            actual_return = (exit_price - entry_price) / entry_price * 100

            prob = h['probability']
            direction = h['direction']
            if prob < 0.60:
                correct = None
            elif direction == 'bullish':
                correct = int(actual_return > 0)
            elif direction == 'bearish':
                correct = int(actual_return < 0)
            else:
                correct = None

            rows.append({
                'pair': pair,
                'horizon': horizon,
                'direction': direction,
                'probability': prob,
                'p_down': h['calibrated_p_down'],
                'p_up': h['calibrated_p_up'],
                'entry_price': entry_price,
                'exit_price': exit_price,
                'matures_at': matures_at,
                'actual_return': actual_return,
                'correct': correct,
            })
            status = 'OK' if correct == 1 else ('NO' if correct == 0 else '--')
            print(f'  {pair}/{horizon}: {direction} {prob:.1%} -> actual {actual_return:+.3f}% {status}')

    return rows


# ── Metrics ───────────────────────────────────────────────────────────────────

def print_metrics(rows: list[dict], candle_time: datetime):
    df = pd.DataFrame(rows)
    df_scored = df[df['correct'].notna()]

    print(f'\n{"="*60}')
    print(f'SINGLE-CYCLE TEST  --  candle T = {candle_time} UTC')
    print(f'{"="*60}')
    print(f'Total predictions : {len(df)}')
    print(f'Scored (prob>=60%): {len(df_scored)}')

    print(f'\n{"-"*60}')
    print(f'{"Horizon":<8} {"N":>4} {"Acc":>7} {">=70% N":>7} {">=70% Acc":>9} {"Avg Prob":>9}')
    print(f'{"-"*60}')

    for horizon in HORIZONS:
        h = df_scored[df_scored['horizon'] == horizon]
        h70 = h[h['probability'] >= 0.70]
        if len(h) == 0:
            continue
        acc = h['correct'].mean()
        acc70 = h70['correct'].mean() if len(h70) > 0 else float('nan')
        n70 = len(h70)
        avg_prob = h['probability'].mean()
        print(f'{horizon:<8} {len(h):>4} {acc:>7.1%} {n70:>7} {acc70:>9.1%} {avg_prob:>9.3f}')

    # Calibration — with only 7 pairs, bins will be sparse. Show raw scatter instead.
    print(f'\n{"-"*60}')
    print('CALIBRATION (p_down vs actual_down) — all horizons combined')
    print(f'{"-"*60}')
    df_all = df.copy()  # use all resolved rows (include prob<0.60 for calibration)
    df_all = df_all[df_all['actual_return'].notna()]
    df_all['actual_down'] = (df_all['actual_return'] < 0).astype(int)

    bins = np.linspace(0, 1, 6)  # 5 bins: 0-20, 20-40, 40-60, 60-80, 80-100%
    cal_points = []
    print(f'{"Bin":>12} {"N":>4} {"Pred p_down":>11} {"Actual down%":>13}')
    for i in range(len(bins) - 1):
        mask = (df_all['p_down'] >= bins[i]) & (df_all['p_down'] < bins[i + 1])
        n = mask.sum()
        if n == 0:
            continue
        pred = df_all.loc[mask, 'p_down'].mean()
        actual = df_all.loc[mask, 'actual_down'].mean()
        cal_points.append({'pred': pred, 'actual': actual, 'n': n})
        print(f'{bins[i]:.0%}-{bins[i+1]:.0%}      {n:>4}      {pred:.3f}       {actual:.3f}')

    if cal_points:
        mce = np.mean([abs(p['pred'] - p['actual']) for p in cal_points])
        print(f'\nMCE (mean cal. error): {mce:.4f}')
        print(f'NOTE: With only {len(df_all)} total predictions, MCE is highly noisy.')
        print(f'      In live trading, MCE will stabilize after ~200+ resolved predictions.')

    # Per-horizon calibration
    print(f'\n{"-"*60}')
    print('PER-HORIZON CALIBRATION')
    print(f'{"-"*60}')
    for horizon in HORIZONS:
        h_df = df_all[df_all['horizon'] == horizon]
        if len(h_df) < 3:
            continue
        # With 7 points, just show raw p_down vs actual_down
        mce_h = np.mean(np.abs(h_df['p_down'].values - h_df['actual_down'].values))
        print(f'{horizon}: raw MCE={mce_h:.4f}  (n={len(h_df)}, '
              f'avg p_down={h_df["p_down"].mean():.3f}, '
              f'actual down%={h_df["actual_down"].mean():.3f})')

    print(f'\n{"-"*60}')
    print('RAW PREDICTIONS')
    print(f'{"-"*60}')
    print(f'{"Pair":<10} {"H":<5} {"Dir":<10} {"Prob":>6} {"p_down":>7} {"Return":>8} {"OK":>3}')
    for _, row in df.sort_values(['horizon', 'pair']).iterrows():
        ok = 'OK' if row['correct'] == 1 else ('NO' if row['correct'] == 0 else '--')
        print(f'{row["pair"]:<10} {row["horizon"]:<5} {row["direction"]:<10} '
              f'{row["probability"]:>6.1%} {row["p_down"]:>7.3f} '
              f'{row["actual_return"]:>8.3f}% {ok:>3}')


# ── Main ──────────────────────────────────────────────────────────────────────

async def main():
    parser = argparse.ArgumentParser(description='Single-cycle forward test')
    parser.add_argument(
        '--candle',
        default='2026-02-28T20:00:00',
        help='Last closed 1H candle time in UTC ISO format (default: 2026-02-28T20:00:00)',
    )
    args = parser.parse_args()

    candle_time = datetime.fromisoformat(args.candle).replace(tzinfo=timezone.utc)
    print(f'Requested candle T = {candle_time} UTC\n')

    # 1. Build buffers
    buffers = await build_buffers_at(candle_time)
    if not buffers:
        print('ERROR: No buffers built. Check API key and candle time.')
        return

    # 2. Entry prices and effective candle time from the actual last bar in buffer
    # (Polygon may not have data up to candle_time exactly — use what we got)
    entry_prices = {}
    effective_candle_time = candle_time  # fallback
    for pair in buffers:
        if '1H' in buffers[pair] and not buffers[pair]['1H'].empty:
            entry_prices[pair] = float(buffers[pair]['1H']['close'].iloc[-1])
            last_bar = buffers[pair]['1H'].index[-1].to_pydatetime()
            if last_bar.tzinfo is None:
                last_bar = last_bar.replace(tzinfo=timezone.utc)
            effective_candle_time = last_bar  # use EURUSD or last pair's time

    print(f'Effective candle T (from buffer) = {effective_candle_time} UTC')
    print(f'Resolving at: T+1H={effective_candle_time + timedelta(hours=1)}, '
          f'T+4H={effective_candle_time + timedelta(hours=4)}, '
          f'T+24H={effective_candle_time + timedelta(hours=24)}\n')

    # 3. Run inference
    print('Running inference...')
    predictions = run_inference(buffers)
    print(f'  Got predictions for {len(predictions)} pairs\n')

    # 4. Resolve
    print('Resolving predictions...')
    rows = await resolve_predictions(predictions, effective_candle_time, entry_prices)

    # 5. Print metrics
    if not rows:
        print('\nERROR: No predictions resolved. Check inference errors above.')
        return
    print_metrics(rows, effective_candle_time)


if __name__ == '__main__':
    asyncio.run(main())
