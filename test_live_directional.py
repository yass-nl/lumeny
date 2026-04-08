"""
Live Directional Test — Last 6 months via Polygon
===================================================
Tests the C1 directional filter on fresh data without the Q50 model.

Directional conditions (C1 short signal):
  - is_ny: hour UTC in 13..20
  - range_return_ratio < 0.16  (low body relative to range = indecisive candle)
  - autocorr_1 < -0.15         (mean-reverting returns)

Trailing stop: 1.5 x ATR(24) — exact same as label computation
ATR proxy filter: only trade when ATR(24) > median ATR (proxy for Q50 > 30 filter)

No Q50 model required.
"""

import os, asyncio, time
import pandas as pd
import numpy as np
import httpx
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()
API_KEY  = os.getenv('POLYGON_S3_SECRET_KEY', '')
REST_BASE = 'https://api.polygon.io'

PAIRS = [
    'EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD', 'USDCAD', 'NZDUSD',
    'EURJPY', 'GBPJPY', 'EURGBP', 'EURAUD', 'AUDJPY', 'CADJPY', 'CHFJPY', 'AUDNZD',
]

PIP = {
    'EURUSD':0.0001,'GBPUSD':0.0001,'AUDUSD':0.0001,'NZDUSD':0.0001,'USDCAD':0.0001,
    'USDCHF':0.0001,'USDJPY':0.01,'EURJPY':0.01,'GBPJPY':0.01,'AUDJPY':0.01,
    'CADJPY':0.01,'CHFJPY':0.01,'EURGBP':0.0001,'EURAUD':0.0001,'AUDNZD':0.0001,
}
SPREAD_PRICE = 0.00028

# C1 thresholds
RRR_THRESH    = 0.16
AUTOCORR_THRESH = -0.15
AUTOCORR_WINDOW = 20    # bars for autocorr computation
ATR_WINDOW    = 24
TRAIL_MULT    = 1.5
TIMEOUT_H     = 24

# Fetch window
FETCH_DAYS    = 210     # ~7 months to have warm-up buffer
TEST_START    = datetime.utcnow() - timedelta(days=180)  # last 6 months for actual test

# ── Fetch ──────────────────────────────────────────────────────────────────────
async def fetch_bars(pair, from_date, to_date):
    ticker = f'C:{pair}'
    url = f'{REST_BASE}/v2/aggs/ticker/{ticker}/range/1/hour/{from_date}/{to_date}'
    params = {'apiKey': API_KEY, 'limit': 50000, 'sort': 'asc'}
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
    df = df.rename(columns={'o':'open','h':'high','l':'low','c':'close','v':'volume'})
    df = df.set_index('datetime')[['open','high','low','close','volume']]
    df = df.sort_index().drop_duplicates()
    # drop weekend gaps
    df = df[~((df.index.dayofweek == 5) | ((df.index.dayofweek == 6) & (df.index.hour < 21)))]
    return df


async def fetch_all():
    now = datetime.utcnow()
    to_date   = now.strftime('%Y-%m-%d')
    from_date = (now - timedelta(days=FETCH_DAYS)).strftime('%Y-%m-%d')
    print(f'Fetching 1H OHLCV from {from_date} to {to_date}...')
    data = {}
    for pair in PAIRS:
        print(f'  {pair}...', end=' ', flush=True)
        df = await fetch_bars(pair, from_date, to_date)
        if df.empty:
            print('NO DATA')
        else:
            data[pair] = df
            print(f'{len(df)} bars')
        await asyncio.sleep(0.2)
    return data


# ── Feature computation ────────────────────────────────────────────────────────
def compute_features(df, pip_size):
    """Compute directional features + ATR trailing stop on 1H OHLCV."""
    df = df.copy()

    # ATR(24)
    tr = np.maximum(
        df['high'] - df['low'],
        np.maximum(
            (df['high'] - df['close'].shift(1)).abs(),
            (df['low']  - df['close'].shift(1)).abs()
        )
    )
    df['atr_24'] = tr.rolling(ATR_WINDOW, min_periods=6).mean()
    df['trail_stop_price'] = TRAIL_MULT * df['atr_24']
    df['trail_stop_pips']  = df['trail_stop_price'] / pip_size

    # range_return_ratio = abs(close-open) / (high-low)
    body  = (df['close'] - df['open']).abs()
    rng   = df['high'] - df['low']
    df['range_return_ratio'] = (body / rng.replace(0, np.nan)).fillna(0)

    # autocorr_1: 1-lag autocorrelation of log returns over rolling window
    log_ret = np.log(df['close'] / df['close'].shift(1))
    df['autocorr_1'] = log_ret.rolling(AUTOCORR_WINDOW, min_periods=10).apply(
        lambda x: pd.Series(x).autocorr(lag=1), raw=False
    )

    # is_ny: 13-20 UTC inclusive
    df['is_ny'] = df.index.hour.isin(range(13, 21)).astype(int)

    # ATR filter proxy for Q50>30: ATR > rolling median (selects volatile bars)
    df['atr_median'] = df['atr_24'].rolling(24*7, min_periods=48).median()
    df['atr_filter'] = (df['atr_24'] > df['atr_median']).astype(int)

    return df


# ── Trailing stop simulation ───────────────────────────────────────────────────
def simulate_trailing_stop(df, entry_idx, direction, trail_price, pip_size, max_bars=TIMEOUT_H):
    """
    Walk forward bar-by-bar from entry_idx+1, tracking trailing stop.
    Returns (mfe_pips, duration_bars, exit_type)
    direction: +1 = long, -1 = short
    """
    entry_price = df['close'].iloc[entry_idx]
    highs  = df['high'].values
    lows   = df['low'].values
    n      = len(df)

    if direction == 1:  # long
        running_max = entry_price
        for k in range(1, min(max_bars + 1, n - entry_idx)):
            idx = entry_idx + k
            running_max = max(running_max, highs[idx])
            if running_max - lows[idx] >= trail_price:
                mfe_pips = (running_max - entry_price) / pip_size
                return mfe_pips, k, 'trail'
        mfe_pips = (running_max - entry_price) / pip_size
        return mfe_pips, min(max_bars, n - entry_idx - 1), 'timeout'
    else:  # short
        running_min = entry_price
        for k in range(1, min(max_bars + 1, n - entry_idx)):
            idx = entry_idx + k
            running_min = min(running_min, lows[idx])
            if highs[idx] - running_min >= trail_price:
                mfe_pips = (entry_price - running_min) / pip_size
                return mfe_pips, k, 'trail'
        mfe_pips = (entry_price - running_min) / pip_size
        return mfe_pips, min(max_bars, n - entry_idx - 1), 'timeout'


# ── Main simulation ────────────────────────────────────────────────────────────
def run_simulation(pair_data):
    all_trades = []

    for pair, df_raw in pair_data.items():
        pip_size    = PIP[pair]
        spread_pips = SPREAD_PRICE / pip_size

        df = compute_features(df_raw, pip_size)

        # Restrict to test period (last 6 months) — warm-up already included
        df_test = df[df.index >= TEST_START].copy()
        df_test = df_test.dropna(subset=['atr_24', 'autocorr_1', 'range_return_ratio'])

        # Re-index to integer positions for fast forward walk
        df_full = df.copy()  # keep full df for forward simulation
        values   = df_full.index
        test_positions = [df_full.index.get_loc(ts) for ts in df_test.index if ts in df_full.index]

        open_until_pos = -1

        for pos in test_positions:
            if pos <= open_until_pos:
                continue

            row = df_full.iloc[pos]

            # ATR filter (proxy for Q50 > 30)
            if row['atr_filter'] != 1:
                continue

            # C1 short signal
            short_signal = (
                row['is_ny'] == 1 and
                row['range_return_ratio'] < RRR_THRESH and
                row['autocorr_1'] < AUTOCORR_THRESH
            )
            # C1 long signal (nzd corr not available — skip longs, short-only test)
            # This is conservative — matches the fact that longs are only ~5% of trades

            if not short_signal:
                continue

            trail_price = row['trail_stop_price']
            if pd.isna(trail_price) or trail_price <= 0:
                continue

            mfe_pips, duration, exit_type = simulate_trailing_stop(
                df_full, pos, direction=-1, trail_price=trail_price, pip_size=pip_size
            )

            result_pips = mfe_pips - row['trail_stop_pips'] - spread_pips

            all_trades.append({
                'pair':        pair,
                'open_ts':     df_full.index[pos],
                'direction':   'short',
                'mfe_pips':    mfe_pips,
                'trail_pips':  row['trail_stop_pips'],
                'result_pips': result_pips,
                'duration':    duration,
                'exit_type':   exit_type,
                'rrr':         row['range_return_ratio'],
                'autocorr':    row['autocorr_1'],
                'atr_24':      row['atr_24'],
            })

            open_until_pos = pos + int(duration)

    return pd.DataFrame(all_trades)


# ── Report ─────────────────────────────────────────────────────────────────────
def print_report(tr):
    if len(tr) == 0:
        print('No trades generated.')
        return

    tr = tr.sort_values('open_ts')
    wins   = tr[tr['result_pips'] > 0]
    losses = tr[tr['result_pips'] <= 0]
    n      = len(tr)
    months = 6

    pf_val = wins['result_pips'].sum() / abs(losses['result_pips'].sum()) if len(losses) > 0 else 999
    tr['month'] = tr['open_ts'].dt.to_period('M')
    monthly = tr.groupby('month')['result_pips'].sum()
    monthly_sharpe = monthly.mean() / monthly.std() * np.sqrt(12) if monthly.std() > 0 else 0
    cum = tr['result_pips'].cumsum()
    max_dd = (cum - cum.cummax()).min()

    print(f'\n{"="*65}')
    print(f'  LIVE TEST: C1 SHORT FILTER — Last 6 months (Polygon)')
    print(f'  Test from: {TEST_START.strftime("%Y-%m-%d")} to now')
    print(f'{"="*65}')
    print(f'  Total trades:     {n:,} (~{n//months}/month)')
    print(f'  Short only:       yes (longs skipped — corr_nzdusd not computed)')
    print(f'  ATR filter:       ATR > rolling 7-day median (proxy for Q50>30)')
    print()
    print(f'  Win rate:         {len(wins)/n:.1%}')
    print(f'  Avg win:          +{wins["result_pips"].mean():.1f} pips')
    print(f'  Avg loss:         {losses["result_pips"].mean():.1f} pips')
    print(f'  Win/Loss ratio:   {abs(wins["result_pips"].mean()/losses["result_pips"].mean()):.2f}x')
    print(f'  EV / trade:       {tr["result_pips"].mean():.2f} pips')
    print(f'  Profit factor:    {pf_val:.3f}')
    print()
    print(f'  Total PnL:        {tr["result_pips"].sum():,.0f} pips')
    print(f'  PnL / month:      {tr["result_pips"].sum()/months:,.0f} pips')
    print(f'  Max drawdown:     {max_dd:,.0f} pips')
    print(f'  Sharpe (monthly): {monthly_sharpe:.2f}')
    print(f'  Profitable months:{(monthly>0).sum()}/{len(monthly)}')
    print()
    print(f'  Exit breakdown:')
    print(f'  {tr["exit_type"].value_counts().to_dict()}')
    print()
    print(f'  Monthly PnL:')
    for month, pnl in monthly.items():
        bar = '#' * int(abs(pnl) / 100)
        sign = '+' if pnl > 0 else '-'
        print(f'    {month}  {sign}{abs(pnl):>7.0f} pips  {sign}{bar}')
    print()
    print(f'  Per-pair breakdown:')
    print(f'  {"Pair":<10} {"Trades":>7} {"WR":>7} {"EV":>8} {"Total":>9}')
    print(f'  {"-"*45}')
    for pair in sorted(tr["pair"].unique()):
        s = tr[tr["pair"]==pair]
        w = (s["result_pips"]>0).mean()
        ev = s["result_pips"].mean()
        tot = s["result_pips"].sum()
        print(f'  {pair:<10} {len(s):>7,} {w:>7.1%} {ev:>8.1f} {tot:>9.0f}')

    # Comparison context vs backtest H2 (Jan-Dec 2025)
    print(f'\n{"="*65}')
    print(f'  CONTEXT vs historical backtest (C1, short only):')
    print(f'  H1 2024: WR~66%, EV~86 pips, PnL~6,444/month')
    print(f'  H2 2025: WR~49%, EV~15 pips, PnL~926/month')
    print(f'  Live:    WR {len(wins)/n:.1%},  EV {tr["result_pips"].mean():.1f} pips')
    print(f'{"="*65}')


async def main():
    pair_data = await fetch_all()
    if not pair_data:
        print('No data fetched.')
        return
    print(f'\nRunning simulation...')
    tr = run_simulation(pair_data)
    print_report(tr)

if __name__ == '__main__':
    asyncio.run(main())
