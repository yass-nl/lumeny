"""
Analyze trade timing patterns -- which hours and days of the week
generate tradeable signals, and are they correct?

Uses last 40 days of Polygon 1m data (30 backtest + 10 warmup).
"""

import os
import sys
sys.stdout.reconfigure(encoding='utf-8')
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

from dotenv import load_dotenv
load_dotenv()

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
META_THRESHOLD = 0.55

BACKTEST_DAYS = 30
WARMUP_DAYS = 10
TOTAL_FETCH_DAYS = BACKTEST_DAYS + WARMUP_DAYS

DAY_NAMES = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']


# -- Data fetching (same as backtest) --
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
    df = df[~((df.index.dayofweek == 5) | ((df.index.dayofweek == 6) & (df.index.hour < 21)))]
    return df


async def fetch_all_pairs():
    now = datetime.utcnow()
    to_date = now.strftime('%Y-%m-%d')
    from_date = (now - timedelta(days=TOTAL_FETCH_DAYS)).strftime('%Y-%m-%d')

    print(f'Fetching data from {from_date} to {to_date}...')
    data = {}

    for pair in PAIRS:
        print(f'  {pair}...', end=' ', flush=True)
        t0 = time.time()
        df_1m = await fetch_bars(pair, 1, 'minute', from_date, to_date)
        elapsed = time.time() - t0

        if df_1m.empty:
            print(f'NO DATA')
            continue

        df_5m = df_1m.resample('5min').agg({
            'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
        }).dropna()
        df_15m = df_1m.resample('15min').agg({
            'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
        }).dropna()
        df_1h = df_1m.resample('1h').agg({
            'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
        }).dropna()

        data[pair] = {'1m': df_1m, '5m': df_5m, '15m': df_15m, '1h': df_1h}
        print(f'{len(df_1m):,} bars ({elapsed:.1f}s)')
        await asyncio.sleep(0.5)

    return data


# -- Import feature computation from the backtest script --
import importlib.util
spec = importlib.util.spec_from_file_location("backtest", "test_live_backtest_v5.2.py")
backtest_mod = importlib.util.module_from_spec(spec)

# Suppress the module's top-level prints
import io, sys
old_stdout = sys.stdout
sys.stdout = io.StringIO()
spec.loader.exec_module(backtest_mod)
sys.stdout = old_stdout

compute_features_for_pair = backtest_mod.compute_features_for_pair
apply_4h_cooldown = backtest_mod.apply_4h_cooldown


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


def analyze_timing(df_all, backtest_start):
    df = df_all[df_all.index >= backtest_start].copy()
    df = df[df['label_1H'].notna()].copy()

    # Filter to tradeable signals (P > 0.55)
    df_tradeable = df[df['meta_proba'].notna()].copy()
    trades = apply_4h_cooldown(df_tradeable[df_tradeable['meta_proba'] > META_THRESHOLD])

    trades['pnl'] = trades['pred_dir'] * trades['label_1H'] - AVG_SPREAD
    trades['correct'] = (trades['pred_dir'] == trades['actual_dir']).astype(int)
    trades['hour'] = trades.index.hour
    trades['dow'] = trades.index.dayofweek  # 0=Mon
    trades['day_name'] = trades['dow'].map(lambda d: DAY_NAMES[d])
    trades['date'] = trades.index.date

    n_days = (trades.index.max() - trades.index.min()).days

    print(f'\n{"="*90}')
    print(f'TRADE TIMING ANALYSIS — Last {BACKTEST_DAYS} days (P > {META_THRESHOLD})')
    print(f'{"="*90}')
    print(f'Period: {trades.index.min().date()} to {trades.index.max().date()} ({n_days} days)')
    print(f'Total trades: {len(trades)}  |  WR: {trades["correct"].mean():.1%}  |  PnL: {trades["pnl"].sum():.4f}')

    # -- 1. Hour of Day breakdown --
    print(f'\n{"-"*90}')
    print(f'TRADES BY HOUR OF DAY (UTC)')
    print(f'{"-"*90}')
    print(f'{"Hour":>6} {"Trades":>8} {"Wins":>6} {"WR":>8} {"PnL":>10} {"AvgMeta":>9} {"Bar":>20}')
    print(f'{"-"*70}')

    for hour in range(24):
        h = trades[trades['hour'] == hour]
        if len(h) == 0:
            print(f'{hour:>4}:00 {0:>8}')
            continue
        wr = h['correct'].mean()
        bar = '#' * len(h)
        color = ''
        print(f'{hour:>4}:00 {len(h):>8} {h["correct"].sum():>6} {wr:>7.1%} '
              f'{h["pnl"].sum():>10.4f} {h["meta_proba"].mean():>9.4f} {bar}')

    # -- 2. Day of Week breakdown --
    print(f'\n{"-"*90}')
    print(f'TRADES BY DAY OF WEEK')
    print(f'{"-"*90}')
    print(f'{"Day":>6} {"Trades":>8} {"Wins":>6} {"WR":>8} {"PnL":>10} {"AvgMeta":>9} {"Tr/wk":>8}')
    print(f'{"-"*60}')

    n_weeks = max(n_days / 7, 1)
    for dow in range(7):
        d = trades[trades['dow'] == dow]
        if len(d) == 0:
            print(f'{DAY_NAMES[dow]:>6} {0:>8}')
            continue
        wr = d['correct'].mean()
        print(f'{DAY_NAMES[dow]:>6} {len(d):>8} {d["correct"].sum():>6} {wr:>7.1%} '
              f'{d["pnl"].sum():>10.4f} {d["meta_proba"].mean():>9.4f} {len(d)/n_weeks:>8.1f}')

    # -- 3. Hour x Day heatmap --
    print(f'\n{"-"*90}')
    print(f'TRADE COUNT HEATMAP (Hour x Day)')
    print(f'{"-"*90}')
    header = f'{"Hour":>6}'
    for d in DAY_NAMES[:5]:  # Mon-Fri only
        header += f' {d:>5}'
    header += f' {"Total":>6}'
    print(header)
    print('-' * 45)

    for hour in range(24):
        row = f'{hour:>4}:00'
        total = 0
        for dow in range(5):
            n = len(trades[(trades['hour'] == hour) & (trades['dow'] == dow)])
            total += n
            row += f' {n:>5}' if n > 0 else f' {".":>5}'
        row += f' {total:>6}'
        if total > 0:
            print(row)

    # -- 4. Win Rate heatmap --
    print(f'\n{"-"*90}')
    print(f'WIN RATE HEATMAP (Hour x Day) — blank = no trades')
    print(f'{"-"*90}')
    header = f'{"Hour":>6}'
    for d in DAY_NAMES[:5]:
        header += f' {d:>6}'
    print(header)
    print('-' * 45)

    for hour in range(24):
        row = f'{hour:>4}:00'
        has_trades = False
        for dow in range(5):
            h = trades[(trades['hour'] == hour) & (trades['dow'] == dow)]
            if len(h) == 0:
                row += f' {".":>6}'
            else:
                has_trades = True
                wr = h['correct'].mean()
                row += f' {wr:>5.0%}'
        if has_trades:
            print(row)

    # -- 5. Per-pair x hour breakdown --
    print(f'\n{"-"*90}')
    print(f'TRADES BY PAIR x HOUR (top combinations)')
    print(f'{"-"*90}')
    pair_hour = trades.groupby(['pair', 'hour']).agg(
        trades=('pnl', 'count'),
        wins=('correct', 'sum'),
        wr=('correct', 'mean'),
        pnl=('pnl', 'sum'),
        avg_meta=('meta_proba', 'mean')
    ).sort_values('trades', ascending=False)

    print(f'{"Pair":>8} {"Hour":>6} {"Trades":>8} {"Wins":>6} {"WR":>8} {"PnL":>10} {"AvgMeta":>9}')
    print('-' * 60)
    for (pair, hour), row in pair_hour.head(30).iterrows():
        print(f'{pair:>8} {hour:>4}:00 {row["trades"]:>8} {row["wins"]:>6} '
              f'{row["wr"]:>7.1%} {row["pnl"]:>10.4f} {row["avg_meta"]:>9.4f}')

    # -- 6. Session analysis --
    print(f'\n{"-"*90}')
    print(f'TRADES BY SESSION')
    print(f'{"-"*90}')

    sessions = {
        'Asian (00-08 UTC)':  trades[(trades['hour'] >= 0) & (trades['hour'] < 8)],
        'London (08-16 UTC)': trades[(trades['hour'] >= 8) & (trades['hour'] < 16)],
        'NY (13-21 UTC)':     trades[(trades['hour'] >= 13) & (trades['hour'] < 21)],
        'Late NY (21-00 UTC)': trades[(trades['hour'] >= 21) | (trades['hour'] < 0)],
    }

    print(f'{"Session":>25} {"Trades":>8} {"Wins":>6} {"WR":>8} {"PnL":>10} {"AvgMeta":>9}')
    print('-' * 70)
    for name, s in sessions.items():
        if len(s) == 0:
            print(f'{name:>25} {0:>8}')
            continue
        wr = s['correct'].mean()
        print(f'{name:>25} {len(s):>8} {s["correct"].sum():>6} {wr:>7.1%} '
              f'{s["pnl"].sum():>10.4f} {s["meta_proba"].mean():>9.4f}')

    # -- 7. Individual trades list --
    print(f'\n{"-"*90}')
    print(f'ALL INDIVIDUAL TRADES')
    print(f'{"-"*90}')
    print(f'{"Datetime (UTC)":>20} {"Pair":>8} {"Dir":>8} {"Meta":>7} {"Q50":>10} {"Correct":>8} {"PnL":>10}')
    print('-' * 80)
    for idx, row in trades.sort_index().iterrows():
        dir_str = 'LONG' if row['pred_dir'] > 0 else 'SHORT'
        correct_str = 'WIN' if row['correct'] else 'LOSS'
        print(f'{str(idx):>20} {row["pair"]:>8} {dir_str:>8} {row["meta_proba"]:>7.4f} '
              f'{row["Q50"]:>+10.6f} {correct_str:>8} {row["pnl"]:>+10.6f}')

    return trades


async def main():
    t_start = time.time()

    data = await fetch_all_pairs()

    print(f'\nComputing features...')
    all_dfs = []
    for pair in PAIRS:
        if pair not in data:
            continue
        print(f'  {pair}...', end=' ', flush=True)
        t0 = time.time()
        df_feat = compute_features_for_pair(pair, data)
        elapsed = time.time() - t0
        print(f'{len(df_feat):,} hours ({elapsed:.1f}s)')
        all_dfs.append(df_feat)

    df_all = pd.concat(all_dfs).sort_index()
    print(f'Total feature rows: {len(df_all):,}')

    print(f'\nRunning inference...')
    df_all = run_inference(df_all)

    backtest_start = df_all.index.min() + pd.Timedelta(days=WARMUP_DAYS)
    print(f'Analysis starts: {backtest_start.date()}')

    analyze_timing(df_all, backtest_start)

    print(f'\nRuntime: {time.time() - t_start:.0f}s')


if __name__ == '__main__':
    asyncio.run(main())
