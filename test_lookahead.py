"""
Lookahead Bias Test
====================
Tests whether any computed feature at bar `t` leaks information about
future price moves (bar t+1 onwards).

Method:
  For each feature, compute:
    - corr(feature[t], future_move[t])   where future_move = close[t+1] - close[t]
    - corr(feature[t], past_move[t])     where past_move  = close[t]   - close[t-1]

  A clean feature may correlate with past_move (it's built from past data),
  but should NOT correlate with future_move more than noise (~0).

  If any feature shows |corr_future| >> |corr_past| that's a red flag.

  We also run the "shift test": compute features normally, then shift the
  feature DataFrame forward by 1 bar. If the shifted features STILL predict
  future moves about as well, the features don't contain lookahead.
  If shifting destroys predictive power, features are genuinely look-back only.

Usage:
  python test_lookahead.py
"""

import asyncio, sys
import numpy as np
import pandas as pd
import httpx
import os
from datetime import datetime, timedelta
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
API_KEY   = os.getenv('POLYGON_S3_SECRET_KEY', '')
REST_BASE = 'https://api.polygon.io'

sys.path.insert(0, str(Path(__file__).parent / 'backend'))
from features import compute_features_for_pair, PIP_SIZE
from live_features_extra import compute_momentum_calendar_features

TEST_PAIR  = 'CHFJPY'   # pick the most-traded pair
FETCH_DAYS = 60         # enough bars for a meaningful test
TOP_N      = 20         # report top N suspicious features


async def fetch_pair(pair):
    now       = datetime.now()
    to_date   = now.strftime('%Y-%m-%d')
    from_date = (now - timedelta(days=FETCH_DAYS)).strftime('%Y-%m-%d')
    print(f'Fetching {pair}  {from_date} -> {to_date} ...')
    ticker = f'C:{pair}'
    url    = f'{REST_BASE}/v2/aggs/ticker/{ticker}/range/1/minute/{from_date}/{to_date}'
    params = {'apiKey': API_KEY, 'limit': 50000, 'sort': 'asc'}
    all_results = []
    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.get(url, params=params)
        resp.raise_for_status()
        data = resp.json()
        all_results.extend(data.get('results', []))
        while 'next_url' in data:
            sep  = '&' if '?' in data['next_url'] else '?'
            resp = await client.get(f"{data['next_url']}{sep}apiKey={API_KEY}")
            resp.raise_for_status()
            data = resp.json()
            all_results.extend(data.get('results', []))
    df = pd.DataFrame(all_results)
    df['datetime'] = pd.to_datetime(df['t'], unit='ms', utc=True).dt.tz_localize(None)
    df = df.rename(columns={'o':'open','h':'high','l':'low','c':'close','v':'volume'})
    df = df.set_index('datetime')[['open','high','low','close','volume']]
    df = df.sort_index().drop_duplicates()
    df = df[~((df.index.dayofweek == 5) | ((df.index.dayofweek == 6) & (df.index.hour < 21)))]
    print(f'  {len(df):,} 1m bars')
    return df


def safe_corr(a, b):
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 30:
        return np.nan
    return np.corrcoef(a[mask], b[mask])[0, 1]


async def main():
    df_1m  = await fetch_pair(TEST_PAIR)
    df_5m  = df_1m.resample('5min').agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum'}).dropna()
    df_15m = df_1m.resample('15min').agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum'}).dropna()
    df_1h  = df_1m.resample('1h').agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum'}).dropna()

    pip = PIP_SIZE[TEST_PAIR]

    print('Computing features...')
    df_feat = compute_features_for_pair(TEST_PAIR, df_1m, df_5m, df_15m, df_1h=df_1h)
    df_extra = compute_momentum_calendar_features(df_1h, pip)
    df_feat  = df_feat.join(df_extra.reindex(df_feat.index), how='left')

    # Align to 1h
    df_feat = df_feat.join(df_1h[['close']], how='inner')
    df_feat = df_feat.sort_index()

    n = len(df_feat)
    print(f'  {n} 1h feature bars, {df_feat.shape[1]} features')

    close = df_feat['close'].values

    # Moves in pips
    future_move = np.full(n, np.nan)
    past_move   = np.full(n, np.nan)
    future_move[:-1] = (close[1:] - close[:-1]) / pip   # close[t+1] - close[t]
    past_move[1:]    = (close[1:] - close[:-1]) / pip   # close[t]   - close[t-1]

    feature_cols = [c for c in df_feat.columns if c not in ('close', 'pair')]

    results = []
    for col in feature_cols:
        vals = df_feat[col].values.astype(float)
        cf = safe_corr(vals, future_move)
        cp = safe_corr(vals, past_move)
        if np.isfinite(cf):
            results.append({'feature': col, 'corr_future': cf, 'corr_past': cp,
                            'abs_future': abs(cf), 'suspicious': abs(cf) > 0.10})

    res = pd.DataFrame(results).sort_values('abs_future', ascending=False)

    print()
    print('=' * 70)
    print(f'  LOOKAHEAD TEST — {TEST_PAIR}  ({n} bars)')
    print('=' * 70)
    print(f'  Total features tested : {len(res)}')
    suspicious = res[res['suspicious']]
    print(f'  |corr_future| > 0.10  : {len(suspicious)}  (potential lookahead)')
    print()
    print(f'  Top {TOP_N} features by |corr_future|:')
    print(f'  {"Feature":<40}  {"corr_future":>12}  {"corr_past":>10}')
    print(f'  {"-"*40}  {"-"*12}  {"-"*10}')
    for _, row in res.head(TOP_N).iterrows():
        flag = ' <-- SUSPICIOUS' if row['suspicious'] else ''
        print(f'  {row["feature"]:<40}  {row["corr_future"]:>+12.4f}  {row["corr_past"]:>+10.4f}{flag}')

    print()
    print('=' * 70)
    print('  SHIFT TEST')
    print('  Shift features forward 1 bar and recompute correlations with future_move.')
    print('  If features are clean (no lookahead), shifting should INCREASE corr_future')
    print('  (because shifted[t] = original[t-1], which is closer to "past" of t+1).')
    print('  If lookahead exists, shifting DECREASES corr_future significantly.')
    print('=' * 70)

    # Shift features by +1 (i.e. at position t, use features from t-1)
    shifted_results = []
    for col in feature_cols:
        vals          = df_feat[col].values.astype(float)
        vals_shifted  = np.roll(vals, 1); vals_shifted[0] = np.nan   # shift forward by 1
        cf_orig       = safe_corr(vals, future_move)
        cf_shifted    = safe_corr(vals_shifted, future_move)
        if np.isfinite(cf_orig) and np.isfinite(cf_shifted):
            # If lookahead: cf_orig > cf_shifted (original has future info, shifted loses it)
            # If clean:     cf_orig ≈ cf_shifted (no future info to lose)
            delta = abs(cf_orig) - abs(cf_shifted)
            shifted_results.append({
                'feature':    col,
                'corr_orig':  cf_orig,
                'corr_shift': cf_shifted,
                'delta':      delta,   # positive = original was "better" at predicting future
            })

    sres = pd.DataFrame(shifted_results).sort_values('delta', ascending=False)
    print(f'\n  Top {TOP_N} features where shifting MOST reduces corr_future')
    print(f'  (large positive delta = strong lookahead signal):')
    print(f'  {"Feature":<40}  {"corr_orig":>10}  {"corr_shift":>10}  {"delta":>8}')
    print(f'  {"-"*40}  {"-"*10}  {"-"*10}  {"-"*8}')
    for _, row in sres.head(TOP_N).iterrows():
        flag = ' <-- LOOKAHEAD?' if row['delta'] > 0.05 else ''
        print(f'  {row["feature"]:<40}  {row["corr_orig"]:>+10.4f}  {row["corr_shift"]:>+10.4f}  {row["delta"]:>+8.4f}{flag}')

    print()
    n_clean    = (sres['delta'] < 0.02).sum()
    n_suspect  = (sres['delta'] > 0.05).sum()
    print(f'  Summary: {n_clean}/{len(sres)} features have delta < 0.02 (clean)')
    print(f'           {n_suspect}/{len(sres)} features have delta > 0.05 (suspicious)')
    if n_suspect == 0:
        print('\n  VERDICT: No lookahead bias detected.')
    else:
        print(f'\n  VERDICT: {n_suspect} features may have lookahead — review above.')


if __name__ == '__main__':
    asyncio.run(main())
