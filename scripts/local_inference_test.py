"""
Local inference test — replicates notebook pipeline on fresh Polygon data.

Fetches recent 1H bars from Polygon, builds features identically to the
notebook's build_feature_matrix(), runs inference using local models,
computes accuracy/MCE/EV on resolved predictions, and prints results.

Usage:
    cd backend
    python ../scripts/local_inference_test.py

Compare p_down values against Railway DB to verify inference parity.
"""

import sys
import os
import asyncio
import warnings
import numpy as np
import pandas as pd
import joblib
import httpx
from datetime import datetime, timedelta, timezone
from pathlib import Path

warnings.filterwarnings('ignore')

# ── Paths ────────────────────────────────────────────────────────────────────
BACKEND_DIR = Path(__file__).parent.parent / 'backend'
MODELS_DIR  = BACKEND_DIR / 'models_2'
sys.path.insert(0, str(BACKEND_DIR))

from features import PAIRS, resample_ohlcv, build_feature_row

# ── Config ───────────────────────────────────────────────────────────────────
API_KEY    = os.getenv('POLYGON_S3_SECRET_KEY', os.getenv('POLYGON_API_KEY', ''))
REST_BASE  = 'https://api.polygon.io'
HORIZONS   = ['1H', '4H', '1D']
QUANTILES  = [0.10, 0.25, 0.50, 0.75, 0.90]
QNAMES     = ['Q10', 'Q25', 'Q50', 'Q75', 'Q90']
AVG_SPREAD = 0.00028

# How many recent closed 1H bars to evaluate on (labels need the next bar)
EVAL_BARS  = 24   # last 24 closed 1H bars = ~1 trading day
# How many days to fetch for feature computation (needs 850 days for dist_ma_200)
LOOKBACK_DAYS = 850

# ── Polygon fetch ─────────────────────────────────────────────────────────────

async def fetch_bars(pair: str, multiplier: int, timespan: str,
                     from_date: str, to_date: str) -> pd.DataFrame:
    ticker = f'C:{pair}'
    url    = f'{REST_BASE}/v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{from_date}/{to_date}'
    params = {'apiKey': API_KEY, 'limit': 50000, 'sort': 'asc'}
    all_results = []

    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.get(url, params=params)
        resp.raise_for_status()
        data = resp.json()
        all_results.extend(data.get('results', []))
        while 'next_url' in data:
            nxt = data['next_url']
            sep = '&' if '?' in nxt else '?'
            resp = await client.get(f'{nxt}{sep}apiKey={API_KEY}')
            resp.raise_for_status()
            data = resp.json()
            all_results.extend(data.get('results', []))

    if not all_results:
        return pd.DataFrame()

    df = pd.DataFrame(all_results)
    df['datetime'] = pd.to_datetime(df['t'], unit='ms', utc=True).dt.tz_localize(None)
    df = df.rename(columns={'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume'})
    df = df.set_index('datetime')[['open', 'high', 'low', 'close', 'volume']]
    return df.sort_index().drop_duplicates()


async def fetch_all_pairs() -> dict:
    now      = datetime.now(timezone.utc)
    to_date  = now.strftime('%Y-%m-%d')
    from_1h  = (now - timedelta(days=LOOKBACK_DAYS)).strftime('%Y-%m-%d')
    from_5m  = (now - timedelta(days=3)).strftime('%Y-%m-%d')
    from_15m = (now - timedelta(days=7)).strftime('%Y-%m-%d')

    buffers = {}
    for pair in PAIRS:
        print(f'  Fetching {pair}...', end=' ', flush=True)
        buffers[pair] = {}

        df_1h = await fetch_bars(pair, 1, 'hour', from_1h, to_date)
        await asyncio.sleep(0.3)
        df_5m = await fetch_bars(pair, 5, 'minute', from_5m, to_date)
        await asyncio.sleep(0.3)
        df_15m = await fetch_bars(pair, 15, 'minute', from_15m, to_date)
        await asyncio.sleep(0.3)

        for tf, df in [('1H', df_1h), ('5m', df_5m), ('15m', df_15m)]:
            if not df.empty:
                df = df[df.index.dayofweek < 5]
                # RAILWAY MODE: drop last bar of each TF (same as CandleBuffer.initialize())
                if len(df) > 1:
                    df = df.iloc[:-1]
                buffers[pair][tf] = df

        if '1H' in buffers[pair]:
            df_1h_clean = buffers[pair]['1H']
            for tf_name, rule in [('4H', '4h'), ('1D', '1D')]:
                df_resampled = resample_ohlcv(df_1h_clean, rule)
                if not df_resampled.empty:
                    buffers[pair][tf_name] = df_resampled

        n = len(buffers[pair].get('1H', []))
        print(f'{n} 1H bars')

    return buffers


# ── Feature matrix (notebook style) ─────────────────────────────────────────

def build_features(buffers: dict) -> pd.DataFrame:
    """Build full feature matrix for all pairs, aligned to 1H index."""
    # Collect all 1H closes for cross-pair correlations
    closes_1h = {p: buffers[p]['1H']['close'] for p in PAIRS if '1H' in buffers[p]}

    # Load expected feature cols from any model bundle
    bundle = joblib.load(MODELS_DIR / 'model_1H_Q50.joblib')
    expected_cols = bundle['feature_cols']
    pair_ids      = {p: i for i, p in enumerate(PAIRS)}

    all_dfs = []
    for pair in PAIRS:
        if '1H' not in buffers[pair]:
            continue
        ohlcv = buffers[pair]

        # RAILWAY MODE: use build_feature_row() exactly as paper_trading.py does
        all_feat = build_feature_row(
            ohlcv, closes_1h, pair, pair_ids[pair],
            expected_cols=expected_cols,
        )
        all_feat['pair']  = pair
        all_feat['close'] = ohlcv['1H']['close']
        all_feat = all_feat[expected_cols + ['pair', 'close']]

        all_dfs.append(all_feat)

    df = pd.concat(all_dfs).sort_index()
    # Trim to last EVAL_BARS per pair for evaluation (full history kept for feature computation above)
    df = df.groupby('pair', group_keys=False).apply(lambda x: x.iloc[-(EVAL_BARS + 25):])
    print(f'  Feature matrix (trimmed to last {EVAL_BARS+25} bars/pair): {len(df):,} rows, {len(expected_cols)} features')
    return df, expected_cols


# ── Inference (notebook style) ───────────────────────────────────────────────

def derive_p_down(q_vals: np.ndarray) -> float:
    qs   = np.array(QUANTILES)
    vals = np.sort(q_vals)
    if vals[0] <= 0 <= vals[-1]:
        return float(np.interp(0, vals, qs))
    elif vals[-1] < 0:
        slope = (qs[-1] - qs[-2]) / (vals[-1] - vals[-2] + 1e-10)
        return float(np.clip(qs[-1] + slope * (0 - vals[-1]), 0.90, 0.999))
    else:
        slope = (qs[1] - qs[0]) / (vals[1] - vals[0] + 1e-10)
        return float(np.clip(qs[0] + slope * (0 - vals[0]), 0.001, 0.10))


def run_inference(df: pd.DataFrame, expected_cols: list) -> pd.DataFrame:
    """Run inference on all rows, return predictions with labels."""
    models = {}
    for horizon in HORIZONS:
        for q, qn in zip(QUANTILES, QNAMES):
            b = joblib.load(MODELS_DIR / f'model_{horizon}_Q{int(q*100)}.joblib')
            models[(horizon, qn)] = b['model']

    records = []
    for pair in PAIRS:
        pair_df = df[df['pair'] == pair].copy()
        if len(pair_df) < 10:
            continue

        X = pair_df[expected_cols].ffill().bfill()

        for horizon in HORIZONS:
            n_bars = {'1H': 1, '4H': 4, '1D': 24}[horizon]

            # Label: forward log return (same as training)
            close = pair_df['close']
            label = np.log(close.shift(-n_bars) / close)

            q_preds_all = {}
            for qn in QNAMES:
                q_preds_all[qn] = models[(horizon, qn)].predict(X)

            for i in range(len(X) - n_bars):  # exclude last n_bars (no label yet)
                row_time = X.index[i]
                q_vals   = np.array([q_preds_all[qn][i] for qn in QNAMES])
                q_vals   = np.sort(q_vals)

                p_down   = derive_p_down(q_vals)
                p_up     = 1.0 - p_down
                prob     = max(p_down, p_up)
                direction = 'bearish' if p_down > p_up else 'bullish'
                if prob < 0.55:
                    direction = 'neutral'

                y = label.iloc[i]
                if pd.isna(y):
                    correct = None
                elif prob < 0.60 or direction == 'neutral':
                    correct = None
                elif direction == 'bullish':
                    correct = int(y > 0)
                else:
                    correct = int(y < 0)

                records.append({
                    'pair':          pair,
                    'horizon':       horizon,
                    'candle_time':   row_time,
                    'matures_at':    row_time + pd.Timedelta(hours=n_bars),
                    'p_down':        round(p_down, 4),
                    'probability':   round(prob, 4),
                    'direction':     direction,
                    'actual_return': round(float(y) * 100, 6) if not pd.isna(y) else None,
                    'correct':       correct,
                })

    return pd.DataFrame(records)


# ── Metrics ──────────────────────────────────────────────────────────────────

def print_metrics(preds: pd.DataFrame):
    print('\n' + '=' * 65)
    print('ACCURACY BY HORIZON & THRESHOLD')
    print('=' * 65)
    print(f'{"Horizon":<8} {"Threshold":<12} {"Accuracy":<12} {"Freq":<10} {"N"}')
    print('-' * 56)

    for horizon in HORIZONS:
        h = preds[(preds['horizon'] == horizon) & preds['correct'].notna()].copy()
        if len(h) < 5:
            print(f'{horizon:<8} insufficient data (n={len(h)})')
            continue
        p_dom = np.where(h['p_down'] > 0.5, h['p_down'], 1 - h['p_down'])
        correct = h['correct'].values.astype(float)
        for thresh in [0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
            mask = p_dom >= thresh
            if mask.sum() < 3:
                continue
            acc  = correct[mask].mean()
            freq = mask.mean()
            print(f'{horizon:<8} >={thresh:.0%}      {acc:<12.1%} {freq:<10.1%} {mask.sum()}')
        print()

    print('=' * 65)
    print('PAIR-BY-PAIR ACCURACY (>=70%)')
    print('=' * 65)
    for horizon in HORIZONS:
        h = preds[(preds['horizon'] == horizon) & preds['correct'].notna()].copy()
        if len(h) < 5:
            continue
        p_dom = np.where(h['p_down'].values > 0.5, h['p_down'].values, 1 - h['p_down'].values)
        print(f'\nHorizon: {horizon}')
        print(f'  {"Pair":<10} {"Overall":<12} {">=70% acc":<12} {"N"}')
        for pair in PAIRS:
            pm = h['pair'] == pair
            if pm.sum() < 3:
                continue
            overall = h.loc[pm, 'correct'].mean()
            mask70  = (p_dom[pm.values] >= 0.70)
            acc70   = h.loc[pm, 'correct'].values[mask70].mean() if mask70.sum() >= 3 else float('nan')
            print(f'  {pair:<10} {overall:<12.1%} {acc70:<12.1%} {mask70.sum()}')

    print('\n' + '=' * 65)
    print('CALIBRATION (MCE)')
    print('=' * 65)
    for horizon in HORIZONS:
        h = preds[preds['horizon'] == horizon].copy()
        h = h[h['actual_return'].notna()]
        if len(h) < 10:
            print(f'{horizon}: insufficient data (n={len(h)})')
            continue
        bins = np.linspace(0, 1, 16)
        bp, ba = [], []
        for i in range(len(bins) - 1):
            mask = (h['p_down'] >= bins[i]) & (h['p_down'] < bins[i+1])
            if mask.sum() >= 3:
                bp.append(h.loc[mask, 'p_down'].mean())
                ba.append((h.loc[mask, 'actual_return'] < 0).mean())
        if bp:
            mce = np.mean(np.abs(np.array(bp) - np.array(ba)))
            print(f'{horizon}: MCE={mce:.3f}  n={len(h)}')
        else:
            print(f'{horizon}: not enough populated bins')

    print('\n' + '=' * 65)
    print('ECONOMIC VALUE (p>=65%)')
    print('=' * 65)
    print(f'{"Horizon":<8} {"Win Rate":<12} {"Avg Move":<12} {"EV%":<10} {"N"}')
    for horizon in HORIZONS:
        h = preds[
            (preds['horizon'] == horizon) &
            (preds['probability'] >= 0.65) &
            preds['correct'].notna() &
            preds['actual_return'].notna()
        ].copy()
        if len(h) < 5:
            continue
        win_rate = h['correct'].mean()
        avg_move = h['actual_return'].abs().mean()
        ev = (win_rate * avg_move) - ((1 - win_rate) * avg_move) - (AVG_SPREAD * 100)
        print(f'{horizon:<8} {win_rate:<12.1%} {avg_move:<12.4f} {ev:<10.4f} {len(h)}')

    print('\n' + '=' * 65)
    print('SAMPLE PREDICTIONS (last 3 candles, EURUSD)')
    print('=' * 65)
    sample = preds[(preds['pair'] == 'EURUSD')].sort_values('candle_time').tail(9)
    print(sample[['candle_time', 'horizon', 'direction', 'probability', 'p_down', 'actual_return', 'correct']].to_string(index=False))

    cutoff = preds['candle_time'].max() - pd.Timedelta(hours=24)
    recent = preds[preds['candle_time'] > cutoff].sort_values(['pair', 'horizon', 'candle_time'])
    return preds, recent


# ── Railway comparison ────────────────────────────────────────────────────────

async def fetch_railway_predictions(railway_url: str, password: str) -> pd.DataFrame:
    """Login to Railway monitor then fetch last 200 predictions."""
    base = railway_url.rstrip('/')
    async with httpx.AsyncClient(timeout=30) as client:
        # Step 1: login to get session cookie
        login_resp = await client.post(f'{base}/auth/login', json={'password': password})
        login_resp.raise_for_status()
        # Step 2: fetch predictions using the session cookie
        resp = await client.get(f'{base}/api/monitor/predictions?limit=200')
        resp.raise_for_status()
        data = resp.json()
    rows = data.get('predictions', [])
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df['logged_at'] = pd.to_datetime(df['logged_at'])
    return df


def print_comparison(local: pd.DataFrame, railway: pd.DataFrame):
    """Print side-by-side p_down comparison for the last 24h."""
    cutoff = datetime.now(timezone.utc) - timedelta(hours=24)
    rw = railway[railway['logged_at'] >= cutoff].copy()

    # Normalize logged_at to tz-naive UTC
    rw['logged_at_naive'] = rw['logged_at'].dt.tz_convert('UTC').dt.tz_localize(None)
    # Railway stores matures_at as ISO string — derive candle_time from it
    horizon_hours = {'1H': 1, '4H': 4, '1D': 24}
    rw['matures_at_dt'] = pd.to_datetime(rw['matures_at'])
    if rw['matures_at_dt'].dt.tz is not None:
        rw['matures_at_dt'] = rw['matures_at_dt'].dt.tz_convert('UTC').dt.tz_localize(None)
    rw['candle_time'] = rw.apply(
        lambda r: r['matures_at_dt'] - pd.Timedelta(hours=horizon_hours.get(r['horizon'], 1)),
        axis=1,
    )

    print(f'\nRailway predictions (last 24h): {len(rw)} rows')
    print(rw[['pair', 'horizon', 'candle_time', 'p_down']].sort_values(['pair', 'horizon']).to_string(index=False))

    print('\n' + '=' * 80)
    print('SIDE-BY-SIDE COMPARISON — Railway candles vs closest local candle')
    print(f'{"Pair":<10} {"Horizon":<8} {"RW Candle":<20} {"Local Candle":<20} {"Local p_down":<14} {"RW p_down":<12} {"Match"}')
    print('-' * 90)

    matched = total = 0
    for _, rw_row in rw.iterrows():
        pair    = rw_row['pair']
        horizon = rw_row['horizon']
        rw_ct   = rw_row['candle_time']
        rw_p    = round(float(rw_row['p_down']), 4)

        # Find closest local row for same pair+horizon within ±2h
        local_ph = local[(local['pair'] == pair) & (local['horizon'] == horizon)].copy()
        if local_ph.empty:
            print(f'{pair:<10} {horizon:<8} {str(rw_ct):<20} {"N/A":<20} {"N/A":<14} {rw_p:<12} ?')
            continue

        local_ph = local_ph.copy()
        local_ph['tdiff'] = (local_ph['candle_time'] - rw_ct).abs()
        closest = local_ph.sort_values('tdiff').iloc[0]
        local_ct = closest['candle_time']
        local_p  = round(float(closest['p_down']), 4)
        tdiff_h  = closest['tdiff'].total_seconds() / 3600

        if tdiff_h > 2:
            print(f'{pair:<10} {horizon:<8} {str(rw_ct):<20} {str(local_ct):<20} {local_p:<14} {rw_p:<12} TOO_FAR ({tdiff_h:.1f}h)')
            continue

        diff = abs(local_p - rw_p)
        match_str = 'OK' if diff < 0.005 else f'DIFF {diff:.4f} ({tdiff_h:.1f}h apart)'
        if diff < 0.005:
            matched += 1
        total += 1
        print(f'{pair:<10} {horizon:<8} {str(rw_ct):<20} {str(local_ct):<20} {local_p:<14} {rw_p:<12} {match_str}')

    print('-' * 90)
    if total > 0:
        print(f'Match rate: {matched}/{total} ({matched/total:.0%}) — threshold <0.005')
    print('=' * 90)


# ── Main ─────────────────────────────────────────────────────────────────────

async def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--compare', nargs=2, metavar=('RAILWAY_URL', 'MONITOR_PASSWORD'),
                        help='Fetch Railway predictions and compare side-by-side')
    args = parser.parse_args()

    if not API_KEY:
        print('ERROR: POLYGON_S3_SECRET_KEY or POLYGON_API_KEY not set.')
        sys.exit(1)

    print(f'Fetching data from Polygon (lookback={LOOKBACK_DAYS} days)...')
    buffers = await fetch_all_pairs()

    print('\nBuilding feature matrix...')
    df, expected_cols = build_features(buffers)

    print('\nRunning inference...')
    preds = run_inference(df, expected_cols)
    print(f'  Generated {len(preds):,} predictions ({preds["correct"].notna().sum()} with correct label)')

    _, recent_local = print_metrics(preds)

    print('\n' + '=' * 65)
    print('LOCAL p_down — LAST 24H')
    print('=' * 65)
    print(recent_local[['pair', 'horizon', 'candle_time', 'direction', 'probability', 'p_down']].to_string(index=False))

    if args.compare:
        railway_url, token = args.compare
        print(f'\nFetching Railway predictions from {railway_url}...')
        railway_df = await fetch_railway_predictions(railway_url, token)
        if railway_df.empty:
            print('ERROR: no predictions returned from Railway.')
        else:
            print(f'  Got {len(railway_df)} Railway predictions.')
            print_comparison(recent_local, railway_df)
    else:
        print('\nTip: run with --compare RAILWAY_URL TOKEN to compare against Railway live data.')


if __name__ == '__main__':
    asyncio.run(main())
