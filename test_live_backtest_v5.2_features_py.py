"""
Live Backtest v5.2 — USING features.py DIRECTLY
Same as test_live_backtest_v5.2.py but imports compute_features_for_pair
from backend/features.py instead of using its own copy.
This validates that live inference produces identical results.
"""

import os
import sys
import asyncio
import time
import pandas as pd
import numpy as np
import joblib
import httpx
import warnings
from pathlib import Path
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8')

# Add backend to path so we can import features.py
sys.path.insert(0, str(Path(__file__).parent / 'backend'))

from dotenv import load_dotenv
load_dotenv()

from features import compute_features_for_pair as features_py_compute, resample_ohlcv

API_KEY = os.getenv('POLYGON_S3_SECRET_KEY', '')
REST_BASE = 'https://api.polygon.io'

PAIRS = [
    'EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD', 'USDCAD', 'NZDUSD',
    'EURJPY', 'GBPJPY', 'EURGBP', 'EURAUD', 'AUDJPY', 'CADJPY', 'CHFJPY', 'AUDNZD',
]

MODELS_DIR = Path('backend/models_5.1/3_quants')
META_DIR   = Path('backend/models_5.1/meta')

AVG_SPREAD = 0.00028
MIN_Q50_THRESHOLD = AVG_SPREAD * 0.5

BACKTEST_DAYS = 450
WARMUP_DAYS = 10
TOTAL_FETCH_DAYS = BACKTEST_DAYS + WARMUP_DAYS

META_THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]

print(f'Live Backtest v5.2 -- USING features.py DIRECTLY')
print(f'  API key: {"OK" if API_KEY else "MISSING"}')
print(f'  Pairs: {len(PAIRS)}')
print(f'  Fetch window: {TOTAL_FETCH_DAYS} days ({WARMUP_DAYS} warmup + {BACKTEST_DAYS} backtest)')


# ---- DATA FETCHING (same as original) ----
async def fetch_bars(pair, multiplier, timespan, from_date, to_date, limit=50000):
    ticker = f'C:{pair}'
    url = f'{REST_BASE}/v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{from_date}/{to_date}'
    params = {'apiKey': API_KEY, 'limit': limit, 'sort': 'asc'}
    all_results = []

    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.get(url, params=params)
        resp.raise_for_status()
        data = resp.json()
        all_results.extend(data.get('results', []))

        while 'next_url' in data:
            next_url = data['next_url']
            sep = '&' if '?' in next_url else '?'
            resp = await client.get(f'{next_url}{sep}apiKey={API_KEY}')
            resp.raise_for_status()
            data = resp.json()
            all_results.extend(data.get('results', []))

    if not all_results:
        return pd.DataFrame()

    df = pd.DataFrame(all_results)
    df['datetime'] = pd.to_datetime(df['t'], unit='ms', utc=True).dt.tz_localize(None)
    df = df.rename(columns={'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume'})
    df = df.set_index('datetime')[['open', 'high', 'low', 'close', 'volume']]
    df = df.sort_index().drop_duplicates()

    # Filter weekends
    df = df[~((df.index.dayofweek == 5) | ((df.index.dayofweek == 6) & (df.index.hour < 21)))]
    return df


async def fetch_all_pairs():
    now = datetime.utcnow()
    to_date = now.strftime('%Y-%m-%d')
    from_date = (now - timedelta(days=TOTAL_FETCH_DAYS)).strftime('%Y-%m-%d')

    print(f'\nFetching data from {from_date} to {to_date}...')
    data = {}

    for pair in PAIRS:
        print(f'  {pair}...', end=' ', flush=True)
        t0 = time.time()
        df_1m = await fetch_bars(pair, 1, 'minute', from_date, to_date)
        elapsed = time.time() - t0

        if df_1m.empty:
            print(f'NO DATA')
            continue

        # Resample using resample_ohlcv from features.py (same as live)
        df_5m = resample_ohlcv(df_1m, '5min')
        df_15m = resample_ohlcv(df_1m, '15min')
        df_1h = resample_ohlcv(df_1m, '1h')

        data[pair] = {
            '1m': df_1m, '5m': df_5m, '15m': df_15m, '1h': df_1h
        }
        print(f'{len(df_1m):,} bars ({elapsed:.1f}s) | {df_1m.index.min().date()} to {df_1m.index.max().date()}')

        await asyncio.sleep(0.5)

    return data


# ---- FEATURE COMPUTATION (using features.py) ----
def compute_features_for_pair_wrapper(pair, data):
    """
    Wrapper that calls features.py's compute_features_for_pair,
    then adds labels from 1H data (same as original backtest).
    """
    df_1m = data[pair]['1m']
    df_5m = data[pair]['5m']
    df_15m = data[pair]['15m']
    df_1h = data[pair]['1h']

    # Use features.py's compute_features_for_pair
    df_features = features_py_compute(pair, df_1m, df_5m, df_15m)

    if df_features.empty:
        return df_features

    # Add labels (same as original backtest)
    close_1h = df_1h['close'].reindex(df_features.index, method='ffill')
    close_1h_4 = df_1h['close'].shift(-4).reindex(df_features.index, method='ffill')
    df_features['label_1H'] = np.log(close_1h_4 / close_1h)

    df_features['entry_price'] = close_1h
    df_features['exit_price'] = close_1h_4

    return df_features


# ---- MODEL INFERENCE (same as original) ----
def run_inference(df_all):
    q50_bundle = joblib.load(MODELS_DIR / 'model_1H_Q50.joblib')
    q25_bundle = joblib.load(MODELS_DIR / 'model_1H_Q25.joblib')
    q75_bundle = joblib.load(MODELS_DIR / 'model_1H_Q75.joblib')
    meta_bundle = joblib.load(META_DIR / 'meta_confidence.joblib')

    feature_cols = q50_bundle['feature_cols']
    meta_feature_cols = meta_bundle['meta_feature_cols']

    X = df_all[feature_cols].groupby(df_all['pair']).ffill().fillna(0)

    q50_pred = q50_bundle['model'].predict(X)
    q25_pred = q25_bundle['model'].predict(X)
    q75_pred = q75_bundle['model'].predict(X)

    df_all['Q50'] = q50_pred
    df_all['Q25'] = q25_pred
    df_all['Q75'] = q75_pred
    df_all['abs_Q50'] = np.abs(q50_pred)
    df_all['pred_dir'] = np.sign(q50_pred)
    df_all['actual_dir'] = np.sign(df_all['label_1H'])

    df_all['Q50_oof'] = q50_pred
    df_all['Q25_oof'] = q25_pred
    df_all['Q75_oof'] = q75_pred
    df_all['abs_Q50'] = np.abs(q50_pred)
    df_all['iqr'] = q75_pred - q25_pred
    df_all['conf_ratio'] = np.abs(q50_pred) / np.clip(df_all['iqr'], 1e-10, None)

    tradeable_mask = df_all['abs_Q50'] > MIN_Q50_THRESHOLD
    df_tradeable = df_all[tradeable_mask].copy()

    if len(df_tradeable) > 0:
        X_meta = df_tradeable[meta_feature_cols].groupby(df_tradeable['pair']).ffill().fillna(0)
        meta_proba = meta_bundle['model'].predict_proba(X_meta)[:, 1]
        df_all.loc[tradeable_mask, 'meta_proba'] = meta_proba
    else:
        df_all['meta_proba'] = np.nan

    return df_all


# ---- TRADE SIMULATION (same as original) ----
def apply_4h_cooldown(df):
    df = df.sort_index()
    pair_unlock_time = {}
    keep = []
    for idx, row in df.iterrows():
        pair = row['pair']
        unlock = pair_unlock_time.get(pair)
        if unlock is not None and idx < unlock:
            keep.append(False)
        else:
            keep.append(True)
            pair_unlock_time[pair] = idx + pd.Timedelta(hours=4)
    return df[keep].copy()


def simulate_trades(df_all, backtest_start):
    df = df_all[df_all.index >= backtest_start].copy()
    df = df[df['label_1H'].notna()].copy()

    print(f'\n{"="*80}')
    print(f'LIVE BACKTEST RESULTS (features.py)')
    print(f'{"="*80}')
    print(f'Period: {df.index.min().date()} to {df.index.max().date()}')
    n_days = (df.index.max() - df.index.min()).days
    print(f'Duration: {n_days} days')
    print(f'Total hours: {len(df):,}')
    print(f'Pairs: {df["pair"].nunique()}')

    # Q50-only baselines
    print(f'\n--- Q50-Only Baselines ---')
    print(f'{"Filter":<20} {"Trades":>8} {"Tr/day":>8} {"WR":>8} {"EV/trade":>12} {"TotalPnL":>10} {"Sharpe":>8}')
    print('-' * 75)

    for name, thresh in [('|Q50|>0.5x', AVG_SPREAD*0.5), ('|Q50|>1x', AVG_SPREAD),
                         ('|Q50|>2x', AVG_SPREAD*2), ('|Q50|>3x', AVG_SPREAD*3)]:
        s = apply_4h_cooldown(df[df['abs_Q50'] > thresh])
        n = len(s)
        if n < 3:
            continue
        pnl = s['pred_dir'] * s['label_1H'] - AVG_SPREAD
        wr = (s['pred_dir'] == s['actual_dir']).mean()
        sharpe = (pnl.mean() / pnl.std()) * np.sqrt(252 * 24) if pnl.std() > 0 else 0
        print(f'{name:<20} {n:>8,} {n/max(n_days,1):>8.2f} {wr:>7.1%} {pnl.mean():>12.6f} {pnl.sum():>10.4f} {sharpe:>8.2f}')

    # Meta-model results
    df_tradeable = df[df['meta_proba'].notna()].copy()
    print(f'\n--- Meta-Model Results ---')
    print(f'Tradeable hours (|Q50|>0.5x with meta score): {len(df_tradeable):,}')
    if len(df_tradeable) > 0:
        print(f'Meta probability: mean={df_tradeable["meta_proba"].mean():.3f}, '
              f'median={df_tradeable["meta_proba"].median():.3f}')

    print(f'\n{"Threshold":<12} {"Trades":>8} {"Tr/day":>8} {"WR":>8} {"EV/trade":>12} {"TotalPnL":>10} {"Sharpe":>8}')
    print('-' * 75)

    for thresh in META_THRESHOLDS:
        s = apply_4h_cooldown(df_tradeable[df_tradeable['meta_proba'] > thresh])
        n = len(s)
        if n < 3:
            continue
        pnl = s['pred_dir'] * s['label_1H'] - AVG_SPREAD
        wr = (s['pred_dir'] == s['actual_dir']).mean()
        sharpe = (pnl.mean() / pnl.std()) * np.sqrt(252 * 24) if pnl.std() > 0 else 0
        flag = ' <<<' if wr >= 0.80 and n >= 10 else (' <<' if wr >= 0.70 else '')
        print(f'P > {thresh:.2f}    {n:>8,} {n/max(n_days,1):>8.2f} {wr:>7.1%} {pnl.mean():>12.6f} {pnl.sum():>10.4f} {sharpe:>8.2f}{flag}')

    # Strategy: P > 0.55
    META_THRESH = 0.55
    filtered = apply_4h_cooldown(df_tradeable[df_tradeable['meta_proba'] > META_THRESH])
    if len(filtered) > 0:
        print(f'\n{"="*80}')
        print(f'STRATEGY: Q50>0.5x spread + Meta P>{META_THRESH}')
        print(f'{"="*80}')
        filtered_copy = filtered.copy()
        filtered_copy['pnl'] = filtered_copy['pred_dir'] * filtered_copy['label_1H'] - AVG_SPREAD
        filtered_copy['correct'] = (filtered_copy['pred_dir'] == filtered_copy['actual_dir']).astype(int)

        total_trades = len(filtered_copy)
        total_wins = filtered_copy['correct'].sum()
        total_pnl = filtered_copy['pnl'].sum()
        total_wr = total_wins / total_trades
        ev = filtered_copy['pnl'].mean()

        print(f'\nTotal trades:  {total_trades}')
        print(f'Wins / Losses: {total_wins} / {total_trades - total_wins}')
        print(f'Win Rate:      {total_wr:.1%}')
        print(f'EV per trade:  {ev:.6f}')
        print(f'Total PnL:     {total_pnl:.4f}')
        print(f'Trades/day:    {total_trades/max(n_days,1):.2f}')

        # Per-pair breakdown
        print(f'\n--- Per-Pair Breakdown ---')
        print(f'{"Pair":<10} {"Trades":>8} {"Tr/day":>8} {"WR":>8} {"EV/trade":>12} {"TotalPnL":>10}')
        print('-' * 60)
        for pair in sorted(filtered_copy['pair'].unique()):
            p = filtered_copy[filtered_copy['pair'] == pair]
            wr = p['correct'].mean()
            flag = ' <<<' if p['pnl'].mean() > 0 else ''
            print(f'{pair:<10} {len(p):>8,} {len(p)/max(n_days,1):>8.2f} {wr:>7.1%} '
                  f'{p["pnl"].mean():>12.6f} {p["pnl"].sum():>10.4f}{flag}')

    return df


def walk_forward_validation(df_all, backtest_start):
    df = df_all[df_all.index >= backtest_start].copy()
    df = df[df['label_1H'].notna()].copy()

    print(f'\n{"="*80}')
    print(f'WALK-FORWARD VALIDATION')
    print(f'{"="*80}')

    BLOCK_DAYS = 90
    start = df.index.min()
    end = df.index.max()
    total_days = (end - start).days

    blocks = []
    block_start = start
    while block_start < end:
        block_end = block_start + pd.Timedelta(days=BLOCK_DAYS)
        if block_end > end:
            block_end = end + pd.Timedelta(hours=1)
        block_df = df[(df.index >= block_start) & (df.index < block_end)]
        if len(block_df) > 0:
            blocks.append((block_start, block_end, block_df))
        block_start = block_end

    print(f'Period: {start.date()} to {end.date()} ({total_days} days)')
    print(f'Block size: {BLOCK_DAYS} days')
    print(f'Number of blocks: {len(blocks)}')

    for thresh in [0.55]:
        print(f'\n--- Meta P > {thresh:.2f} ---')
        print(f'{"Block":<25} {"Days":>5} {"Trades":>7} {"Tr/day":>7} {"WR":>7} {"EV/trade":>11} {"PnL":>9} {"Sharpe":>7}')
        print('-' * 85)

        block_results = []
        for i, (bs, be, bdf) in enumerate(blocks):
            bdf_tradeable = bdf[bdf['meta_proba'].notna()]
            s = apply_4h_cooldown(bdf_tradeable[bdf_tradeable['meta_proba'] > thresh])
            n = len(s)
            n_days_block = (be - bs).days
            if n < 2:
                print(f'{str(bs.date())} - {str((be - pd.Timedelta(days=1)).date()):<11} {n_days_block:>5} {n:>7}')
                block_results.append({'trades': n, 'wr': np.nan, 'ev': np.nan, 'pnl': 0, 'sharpe': np.nan})
                continue
            pnl = s['pred_dir'] * s['label_1H'] - AVG_SPREAD
            wr = (s['pred_dir'] == s['actual_dir']).mean()
            sharpe = (pnl.mean() / pnl.std()) * np.sqrt(252 * 24) if pnl.std() > 0 else 0
            flag = ' <<<' if wr >= 0.70 else (' <<' if wr >= 0.60 else '')
            block_label = f'{bs.date()} - {(be - pd.Timedelta(days=1)).date()}'
            print(f'{block_label:<25} {n_days_block:>5} {n:>7} {n/max(n_days_block,1):>7.2f} {wr:>6.1%} {pnl.mean():>11.6f} {pnl.sum():>9.4f} {sharpe:>7.2f}{flag}')
            block_results.append({'trades': n, 'wr': wr, 'ev': pnl.mean(), 'pnl': pnl.sum(), 'sharpe': sharpe})

        valid = [r for r in block_results if not np.isnan(r.get('wr', np.nan))]
        if len(valid) >= 2:
            wrs = [r['wr'] for r in valid]
            pnls = [r['pnl'] for r in valid]
            n_positive = sum(1 for p in pnls if p > 0)
            total_pnl = sum(pnls)
            print(f'{"":>25} {"":>5} {"":>7} {"":>7} {"":>7} {"":>11} {"-"*9} {"":>7}')
            print(f'{"AGGREGATE":<25} {"":>5} {sum(r["trades"] for r in valid):>7} '
                  f'{"":>7} {np.mean(wrs):>6.1%} {np.mean([r["ev"] for r in valid]):>11.6f} {total_pnl:>9.4f} {"":>7}')
            print(f'  Blocks positive: {n_positive}/{len(valid)}')
            print(f'  WR range: {min(wrs):.1%} - {max(wrs):.1%} (std: {np.std(wrs):.1%})')


# ---- MAIN ----
async def main():
    t_start = time.time()

    data = await fetch_all_pairs()

    print(f'\nComputing features using features.py...')
    all_dfs = []
    for pair in PAIRS:
        if pair not in data:
            continue
        print(f'  {pair}...', end=' ', flush=True)
        t0 = time.time()
        df_feat = compute_features_for_pair_wrapper(pair, data)
        elapsed = time.time() - t0
        print(f'{len(df_feat):,} hours ({elapsed:.1f}s)')
        all_dfs.append(df_feat)

    df_all = pd.concat(all_dfs).sort_index()
    print(f'Total feature rows: {len(df_all):,}')

    print(f'\nRunning model inference...')
    df_all = run_inference(df_all)

    backtest_start = df_all.index.min() + pd.Timedelta(days=WARMUP_DAYS)
    print(f'Backtest starts: {backtest_start.date()} (after {WARMUP_DAYS} days warmup)')

    simulate_trades(df_all, backtest_start)
    walk_forward_validation(df_all, backtest_start)

    elapsed_total = time.time() - t_start
    print(f'\nTotal runtime: {elapsed_total:.0f}s')


if __name__ == '__main__':
    asyncio.run(main())
