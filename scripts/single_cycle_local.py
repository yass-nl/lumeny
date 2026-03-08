"""
Single-Cycle Local Test
=======================
Same as single_cycle_test.py but uses local 1-min parquet files instead of
fetching from Polygon. All 2025 data is fully unseen (training cutoff 2024-06-30).

Usage:
    cd /c/Users/noual/lumeny/backend
    python ../scripts/single_cycle_local.py
    python ../scripts/single_cycle_local.py --candle 2025-06-11T14:00:00
    python ../scripts/single_cycle_local.py --candle 2025-06-11T14:00:00 --lookback 30

The candle time is the last CLOSED 1H bar (T).
Exit prices are read directly from local parquet (no API needed).
"""

import argparse
import os
import sys
import warnings
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

BACKEND_DIR = Path(__file__).parent.parent / 'backend'
DATA_DIR = BACKEND_DIR / 'data' / 'raw'
sys.path.insert(0, str(BACKEND_DIR))

PAIRS = ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD', 'USDCAD', 'NZDUSD']
HORIZONS = ['1H', '4H', '1D']
HORIZON_HOURS = {'1H': 1, '4H': 4, '1D': 24}
PAIR_IDS = {pair: i for i, pair in enumerate(PAIRS)}


# ── Local data helpers ────────────────────────────────────────────────────────

def load_1min(pair: str, years: list[int]) -> pd.DataFrame:
    """Load and concatenate 1-min parquet files for given years."""
    frames = []
    for year in years:
        path = DATA_DIR / f'{pair}_1min_{year}.parquet'
        if path.exists():
            df = pd.read_parquet(path)
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames)
    # Set datetime column as index if not already
    if 'datetime' in df.columns:
        df = df.set_index('datetime')
    df = df.sort_index()
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    return df[['open', 'high', 'low', 'close', 'volume']]


def resample_to_1h(df_1min: pd.DataFrame) -> pd.DataFrame:
    """Resample 1-min OHLCV to 1H."""
    df = df_1min.resample('1h', label='left', closed='left').agg({
        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum',
    }).dropna(subset=['open'])
    # Drop weekends
    df = df[df.index.dayofweek < 5]
    return df


def resample_ohlcv(df_1h: pd.DataFrame, rule: str) -> pd.DataFrame:
    return df_1h.resample(rule, label='left', closed='left').agg({
        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum',
    }).dropna(subset=['open'])


def get_close_at(df_1min: pd.DataFrame, at_time: datetime) -> float | None:
    """
    Get the close price of the 1H bar whose open <= at_time.
    Exact same logic as resolve_outcomes() / _fetch_close_price().
    """
    at_naive = at_time.replace(tzinfo=None) if at_time.tzinfo else at_time
    # Get 1H bars around at_time
    window = df_1min[
        (df_1min.index >= at_naive - timedelta(hours=2)) &
        (df_1min.index <= at_naive + timedelta(hours=2))
    ]
    if window.empty:
        return None
    df_1h = resample_to_1h(window)
    if df_1h.empty:
        return None
    # Find bar whose open <= at_time (last such bar)
    candidates = df_1h[df_1h.index <= at_naive]
    if candidates.empty:
        return float(df_1h['close'].iloc[0])
    return float(candidates['close'].iloc[-1])


# ── Build buffers ─────────────────────────────────────────────────────────────

def build_buffers_at(candle_time: datetime, lookback_days: int = 30) -> dict:
    """
    Build OHLCV buffers from local parquet, exactly as CandleBuffer would at candle_time T.
    Uses lookback_days of 1H history ending at candle_time (inclusive).
    """
    candle_naive = candle_time.replace(tzinfo=None)
    fetch_start = candle_naive - timedelta(days=lookback_days)

    # Which years do we need?
    years = sorted(set(range(fetch_start.year, candle_naive.year + 1)))

    print(f'Loading local parquet data ({fetch_start.date()} to {candle_naive.date()})...')

    buffers = {}
    raw_1min = {}  # keep for resolution later

    for pair in PAIRS:
        df_1min = load_1min(pair, years)
        if df_1min.empty:
            print(f'  {pair}: no local data')
            continue

        # Trim to window
        df_1min = df_1min[(df_1min.index >= fetch_start) & (df_1min.index <= candle_naive + timedelta(hours=2))]
        if df_1min.empty:
            print(f'  {pair}: no data in window')
            continue

        raw_1min[pair] = df_1min  # store for exit price lookup

        # Resample to 1H
        df_1h = resample_to_1h(df_1min)

        # Keep bars up to and including candle_time
        df_1h = df_1h[df_1h.index <= candle_naive]

        if len(df_1h) < 2:
            print(f'  {pair}: insufficient 1H bars')
            continue

        last = df_1h.index[-1]
        print(f'  {pair}: {len(df_1h)} 1H bars, last closed = {last}')

        buffers[pair] = {'1H': df_1h}

        # Resample 4H/1D — no extra drop (fixed)
        for tf_name, rule in [('4H', '4h'), ('1D', '1D')]:
            df_resampled = resample_ohlcv(df_1h, rule)
            if not df_resampled.empty:
                buffers[pair][tf_name] = df_resampled

    return buffers, raw_1min


# ── Run inference ─────────────────────────────────────────────────────────────

def run_inference(buffers: dict) -> dict:
    import inference as _inf_module
    _inf_module.MODELS_DIR = BACKEND_DIR / 'models_2'

    from inference import Predictor
    from features import build_feature_row

    predictor = Predictor()
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
            print(f'  {pair}: inference error -- {e}')

    return predictions


# ── Resolve ───────────────────────────────────────────────────────────────────

def resolve_predictions(predictions: dict, candle_time: datetime,
                        entry_prices: dict, raw_1min: dict) -> list[dict]:
    """Resolve using local parquet exit prices."""
    rows = []
    for pair, result in predictions.items():
        entry_price = entry_prices.get(pair)
        if entry_price is None or pair not in raw_1min:
            continue

        # Extend 1min data to cover exit times (load extra years if needed)
        df_1min = raw_1min[pair]
        candle_naive = candle_time.replace(tzinfo=None)

        for horizon in HORIZONS:
            if horizon not in result['horizons']:
                continue
            h = result['horizons'][horizon]
            matures_at = candle_time + timedelta(hours=HORIZON_HOURS[horizon])
            matures_naive = matures_at.replace(tzinfo=None)

            # If 1min data doesn't cover matures_at, load more
            if df_1min.index.max() < matures_naive - timedelta(minutes=30):
                extra_years = list(range(candle_naive.year, matures_naive.year + 1))
                df_ext = load_1min(pair, extra_years)
                if not df_ext.empty:
                    df_1min_ext = df_ext[df_ext.index <= matures_naive + timedelta(hours=2)]
                else:
                    df_1min_ext = df_1min
            else:
                df_1min_ext = df_1min

            exit_price = get_close_at(df_1min_ext, matures_at)
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
    print(f'SINGLE-CYCLE LOCAL TEST  --  candle T = {candle_time} UTC')
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

    print(f'\n{"-"*60}')
    print('CALIBRATION (p_down vs actual_down) -- all horizons combined')
    print(f'{"-"*60}')
    df_all = df[df['actual_return'].notna()].copy()
    df_all['actual_down'] = (df_all['actual_return'] < 0).astype(int)

    bins = np.linspace(0, 1, 6)
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

    print(f'\n{"-"*60}')
    print('PER-HORIZON CALIBRATION')
    print(f'{"-"*60}')
    for horizon in HORIZONS:
        h_df = df_all[df_all['horizon'] == horizon]
        if len(h_df) < 3:
            continue
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

def main():
    parser = argparse.ArgumentParser(description='Single-cycle local test (no API needed)')
    parser.add_argument(
        '--candle',
        default='2025-06-11T14:00:00',
        help='Last closed 1H candle time in UTC ISO format (default: 2025-06-11T14:00:00)',
    )
    parser.add_argument(
        '--lookback',
        type=int,
        default=30,
        help='Days of 1H history to use as context (default: 30)',
    )
    args = parser.parse_args()

    candle_time = datetime.fromisoformat(args.candle).replace(tzinfo=timezone.utc)
    print(f'Requested candle T = {candle_time} UTC\n')

    # 1. Build buffers from local data
    buffers, raw_1min = build_buffers_at(candle_time, args.lookback)
    if not buffers:
        print('ERROR: No buffers built. Check DATA_DIR and candle time.')
        return

    # 2. Effective candle time + entry prices
    entry_prices = {}
    effective_candle_time = candle_time
    for pair in buffers:
        if '1H' in buffers[pair] and not buffers[pair]['1H'].empty:
            entry_prices[pair] = float(buffers[pair]['1H']['close'].iloc[-1])
            last_bar = buffers[pair]['1H'].index[-1].to_pydatetime()
            if last_bar.tzinfo is None:
                last_bar = last_bar.replace(tzinfo=timezone.utc)
            effective_candle_time = last_bar

    print(f'\nEffective candle T (from buffer) = {effective_candle_time} UTC')
    print(f'Resolving at: T+1H={effective_candle_time + timedelta(hours=1)}, '
          f'T+4H={effective_candle_time + timedelta(hours=4)}, '
          f'T+24H={effective_candle_time + timedelta(hours=24)}\n')

    # 3. Run inference
    print('Running inference...')
    predictions = run_inference(buffers)
    print(f'  Got predictions for {len(predictions)} pairs\n')

    # 4. Resolve using local data
    print('Resolving predictions...')
    rows = resolve_predictions(predictions, effective_candle_time, entry_prices, raw_1min)

    # 5. Print metrics
    if not rows:
        print('\nERROR: No predictions resolved.')
        return
    print_metrics(rows, effective_candle_time)


if __name__ == '__main__':
    main()
