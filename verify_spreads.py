"""
Spread & Slippage Verification — Compare sim assumptions vs real Polygon bid/ask quotes.

Fetches real bid/ask quotes at each hour boundary for 30 days across all 15 pairs.
Compares actual spreads to the simulation's hardcoded SPREAD_POINTS + time multipliers.
Estimates slippage proxy from quote volatility within short windows.
"""

import os
import asyncio
import time
import pandas as pd
import numpy as np
import httpx
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv('POLYGON_S3_SECRET_KEY', '')
REST_BASE = 'https://api.polygon.io'

PAIRS = [
    'EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD', 'USDCAD', 'NZDUSD',
    'EURJPY', 'GBPJPY', 'EURGBP', 'EURAUD', 'AUDJPY', 'CADJPY', 'CHFJPY', 'AUDNZD',
]

JPY_PAIRS = {'USDJPY', 'EURJPY', 'GBPJPY', 'AUDJPY', 'CADJPY', 'CHFJPY'}

# ── Sim assumptions (copied from test_capital_sim.py) ──
SPREAD_POINTS_SIM = {
    'AUDUSD': 1, 'EURUSD': 1, 'GBPUSD': 1, 'NZDUSD': 1,
    'USDCAD': 2, 'USDCHF': 1, 'USDJPY': 3,
    'EURGBP': 1, 'AUDNZD': 5, 'AUDJPY': 7, 'CADJPY': 7,
    'CHFJPY': 11, 'EURAUD': 5, 'EURJPY': 5, 'GBPJPY': 8,
}

OFFHOUR_SPREAD_MULTIPLIER = {
    'EURUSD': 1.5, 'GBPUSD': 2.0, 'USDJPY': 2.0, 'USDCHF': 2.0,
    'AUDUSD': 2.0, 'USDCAD': 2.0, 'NZDUSD': 2.0,
    'EURJPY': 3.0, 'GBPJPY': 3.0, 'EURGBP': 2.5, 'EURAUD': 3.0,
    'AUDJPY': 3.0, 'CADJPY': 3.0, 'CHFJPY': 3.0, 'AUDNZD': 3.0,
}

SLIPPAGE_BASE_POINTS = {
    'EURUSD': 0.3, 'GBPUSD': 0.3, 'USDJPY': 0.5, 'USDCHF': 0.5,
    'AUDUSD': 0.5, 'USDCAD': 0.5, 'NZDUSD': 0.5,
    'EURGBP': 0.5, 'AUDNZD': 1.0, 'AUDJPY': 1.5, 'CADJPY': 1.5,
    'CHFJPY': 2.0, 'EURAUD': 1.0, 'EURJPY': 1.0, 'GBPJPY': 1.5,
}

SLIPPAGE_TIME_MULTIPLIER = {
    'EURUSD': 1.5, 'GBPUSD': 2.0, 'USDJPY': 2.0, 'USDCHF': 2.0,
    'AUDUSD': 2.0, 'USDCAD': 2.0, 'NZDUSD': 2.0,
    'EURJPY': 2.5, 'GBPJPY': 2.5, 'EURGBP': 2.0, 'EURAUD': 2.5,
    'AUDJPY': 2.5, 'CADJPY': 2.5, 'CHFJPY': 3.0, 'AUDNZD': 2.5,
}

FETCH_DAYS = 30


def sim_spread_points(pair, hour_utc):
    base = SPREAD_POINTS_SIM.get(pair, 5)
    if hour_utc in (21, 22):
        mult = OFFHOUR_SPREAD_MULTIPLIER.get(pair, 2.0)
        return int(base * mult)
    return base


def spread_price_to_points(pair, spread_price):
    """Convert a price-level spread to points."""
    if pair in JPY_PAIRS:
        return spread_price / 0.001
    else:
        return spread_price / 0.00001


async def fetch_quotes_for_hour(client, pair, timestamp_ms, limit=10):
    """Fetch a small batch of quotes around a specific timestamp."""
    ticker = f'C:{pair}'
    url = f'{REST_BASE}/v3/quotes/{ticker}'
    # Get quotes starting from this timestamp
    params = {
        'apiKey': API_KEY,
        'timestamp.gte': timestamp_ms * 1_000_000,  # nanoseconds
        'timestamp.lt': (timestamp_ms + 60_000) * 1_000_000,  # +1 minute window
        'limit': limit,
        'order': 'asc',
        'sort': 'timestamp',
    }
    try:
        resp = await client.get(url, params=params)
        resp.raise_for_status()
        data = resp.json()
        results = data.get('results', [])
        return results
    except Exception as e:
        return []


async def fetch_pair_spreads(pair):
    """Fetch hourly quote samples for one pair over FETCH_DAYS days."""
    now = datetime.utcnow()
    end = now.replace(minute=0, second=0, microsecond=0)
    start = end - timedelta(days=FETCH_DAYS)

    # Build list of hourly timestamps (skip weekends)
    hours = []
    current = start
    while current < end:
        dow = current.weekday()
        # Skip Saturday and Sunday before 21:00
        if dow != 5 and not (dow == 6 and current.hour < 21):
            hours.append(current)
        current += timedelta(hours=1)

    records = []
    async with httpx.AsyncClient(timeout=30) as client:
        # Process in batches to respect rate limits
        batch_size = 20
        for i in range(0, len(hours), batch_size):
            batch = hours[i:i + batch_size]
            tasks = []
            for h in batch:
                ts_ms = int(h.timestamp() * 1000)
                tasks.append(fetch_quotes_for_hour(client, pair, ts_ms))

            results = await asyncio.gather(*tasks)

            for h, quotes in zip(batch, results):
                if not quotes:
                    continue
                # Compute spread from each quote, take median
                spreads = []
                bid_moves = []
                for q in quotes:
                    bid = q.get('bid_price', 0)
                    ask = q.get('ask_price', 0)
                    if bid > 0 and ask > 0 and ask > bid:
                        spreads.append(ask - bid)

                # Quote instability (slippage proxy): std of bid prices in the window
                bids = [q.get('bid_price', 0) for q in quotes if q.get('bid_price', 0) > 0]
                bid_std = np.std(bids) if len(bids) > 1 else 0.0

                if spreads:
                    records.append({
                        'pair': pair,
                        'datetime': h,
                        'hour_utc': h.hour,
                        'dow': h.weekday(),
                        'median_spread_price': np.median(spreads),
                        'mean_spread_price': np.mean(spreads),
                        'max_spread_price': np.max(spreads),
                        'n_quotes': len(spreads),
                        'bid_std': bid_std,  # slippage proxy
                    })

            # Small delay between batches to avoid rate limits
            if i + batch_size < len(hours):
                await asyncio.sleep(0.5)

    return records


async def main():
    print(f'Spread & Slippage Verification')
    print(f'  Period: last {FETCH_DAYS} days')
    print(f'  Pairs: {len(PAIRS)}')
    print(f'  Sampling: 1 quote batch per hour boundary\n')

    all_records = []
    for pair in PAIRS:
        print(f'  Fetching {pair}...', end=' ', flush=True)
        t0 = time.time()
        records = await fetch_pair_spreads(pair)
        elapsed = time.time() - t0
        print(f'{len(records)} samples in {elapsed:.1f}s')
        all_records.extend(records)

    if not all_records:
        print('\nNo data fetched. Check API key and plan.')
        return

    df = pd.DataFrame(all_records)

    # Convert spreads to points
    df['median_spread_pts'] = df.apply(
        lambda r: spread_price_to_points(r['pair'], r['median_spread_price']), axis=1)
    df['mean_spread_pts'] = df.apply(
        lambda r: spread_price_to_points(r['pair'], r['mean_spread_price']), axis=1)
    df['max_spread_pts'] = df.apply(
        lambda r: spread_price_to_points(r['pair'], r['max_spread_price']), axis=1)
    df['bid_std_pts'] = df.apply(
        lambda r: spread_price_to_points(r['pair'], r['bid_std']), axis=1)

    # Add sim assumed spread
    df['sim_spread_pts'] = df.apply(
        lambda r: sim_spread_points(r['pair'], r['hour_utc']), axis=1)

    # ── SPREAD COMPARISON ──
    print(f'\n{"="*90}')
    print(f'SPREAD COMPARISON: Simulation vs Real (in points)')
    print(f'{"="*90}')

    # Overall per-pair
    print(f'\n--- Overall (all hours) ---')
    print(f'{"Pair":<10} {"Sim":>6} {"Real Med":>9} {"Real Mean":>10} {"Real P95":>9} '
          f'{"Diff%":>7} {"Verdict":>14} {"Samples":>8}')
    print('-' * 80)

    for pair in PAIRS:
        p = df[df['pair'] == pair]
        if p.empty:
            continue
        sim_avg = p['sim_spread_pts'].mean()
        real_med = p['median_spread_pts'].median()
        real_mean = p['mean_spread_pts'].mean()
        real_p95 = p['median_spread_pts'].quantile(0.95)
        diff_pct = ((sim_avg - real_med) / real_med * 100) if real_med > 0 else 0

        if diff_pct > 15:
            verdict = 'OVERESTIMATE'
        elif diff_pct < -15:
            verdict = 'UNDERESTIMATE'
        else:
            verdict = 'ACCURATE'

        print(f'{pair:<10} {sim_avg:>6.1f} {real_med:>9.1f} {real_mean:>10.1f} {real_p95:>9.1f} '
              f'{diff_pct:>+6.0f}% {verdict:>14} {len(p):>8}')

    # Liquid hours only (8-17 UTC — London+NY overlap)
    liquid = df[df['hour_utc'].between(8, 17)]
    print(f'\n--- Liquid Hours Only (08:00-17:00 UTC) ---')
    print(f'{"Pair":<10} {"Sim":>6} {"Real Med":>9} {"Real Mean":>10} {"Real P95":>9} '
          f'{"Diff%":>7} {"Verdict":>14}')
    print('-' * 72)

    for pair in PAIRS:
        p = liquid[liquid['pair'] == pair]
        if p.empty:
            continue
        sim_base = SPREAD_POINTS_SIM.get(pair, 5)
        real_med = p['median_spread_pts'].median()
        real_mean = p['mean_spread_pts'].mean()
        real_p95 = p['median_spread_pts'].quantile(0.95)
        diff_pct = ((sim_base - real_med) / real_med * 100) if real_med > 0 else 0

        if diff_pct > 15:
            verdict = 'OVERESTIMATE'
        elif diff_pct < -15:
            verdict = 'UNDERESTIMATE'
        else:
            verdict = 'ACCURATE'

        print(f'{pair:<10} {sim_base:>6.1f} {real_med:>9.1f} {real_mean:>10.1f} {real_p95:>9.1f} '
              f'{diff_pct:>+6.0f}% {verdict:>14}')

    # Off-hours (21-22 UTC)
    offhours = df[df['hour_utc'].isin([21, 22])]
    if not offhours.empty:
        print(f'\n--- Off-Hours (21:00-22:00 UTC) ---')
        print(f'{"Pair":<10} {"Sim":>6} {"Real Med":>9} {"Real Mean":>10} {"Real P95":>9} '
              f'{"Diff%":>7} {"Verdict":>14}')
        print('-' * 72)

        for pair in PAIRS:
            p = offhours[offhours['pair'] == pair]
            if p.empty:
                continue
            sim_off = sim_spread_points(pair, 21)
            real_med = p['median_spread_pts'].median()
            real_mean = p['mean_spread_pts'].mean()
            real_p95 = p['median_spread_pts'].quantile(0.95)
            diff_pct = ((sim_off - real_med) / real_med * 100) if real_med > 0 else 0

            if diff_pct > 15:
                verdict = 'OVERESTIMATE'
            elif diff_pct < -15:
                verdict = 'UNDERESTIMATE'
            else:
                verdict = 'ACCURATE'

            print(f'{pair:<10} {sim_off:>6.1f} {real_med:>9.1f} {real_mean:>10.1f} {real_p95:>9.1f} '
                  f'{diff_pct:>+6.0f}% {verdict:>14}')

    # ── SLIPPAGE PROXY ──
    # bid_std (quote instability in a 1-min window) as a proxy for execution slippage
    print(f'\n{"="*90}')
    print(f'SLIPPAGE PROXY: Quote Instability vs Sim Assumptions (in points)')
    print(f'{"="*90}')
    print(f'  bid_std = std of bid prices within 1-min window at each hour')
    print(f'  This proxies how much the price moves during execution\n')

    print(f'{"Pair":<10} {"Sim Slip":>9} {"Real bid_std":>13} {"Ratio":>7} {"Verdict":>14}')
    print('-' * 60)

    for pair in PAIRS:
        p = liquid[liquid['pair'] == pair]
        if p.empty:
            continue
        sim_slip = SLIPPAGE_BASE_POINTS.get(pair, 1.0)
        real_std = p['bid_std_pts'].median()
        ratio = (sim_slip / real_std) if real_std > 0 else float('inf')

        if ratio > 2.0:
            verdict = 'OVERESTIMATE'
        elif ratio < 0.5:
            verdict = 'UNDERESTIMATE'
        else:
            verdict = 'REASONABLE'

        print(f'{pair:<10} {sim_slip:>9.1f} {real_std:>13.2f} {ratio:>7.1f}x {verdict:>14}')

    # ── HOURLY HEATMAP ──
    print(f'\n{"="*90}')
    print(f'SPREAD BY HOUR (median points) — Top 5 widest pairs')
    print(f'{"="*90}')

    # Pick 5 pairs with highest average real spread
    pair_avg = df.groupby('pair')['median_spread_pts'].median().nlargest(5)
    wide_pairs = pair_avg.index.tolist()

    hours_range = list(range(0, 24))
    header = f'{"Pair":<10} ' + ' '.join(f'{h:>4}' for h in hours_range)
    print(header)
    print('-' * (10 + 5 * 24))

    for pair in wide_pairs:
        p = df[df['pair'] == pair]
        vals = []
        for h in hours_range:
            ph = p[p['hour_utc'] == h]
            if ph.empty:
                vals.append('   -')
            else:
                vals.append(f'{ph["median_spread_pts"].median():>4.0f}')
        print(f'{pair:<10} ' + ' '.join(vals))

    # ── SUMMARY ──
    print(f'\n{"="*90}')
    print(f'SUMMARY')
    print(f'{"="*90}')

    over = 0
    under = 0
    accurate = 0
    for pair in PAIRS:
        p = df[df['pair'] == pair]
        if p.empty:
            continue
        sim_avg = p['sim_spread_pts'].mean()
        real_med = p['median_spread_pts'].median()
        diff_pct = ((sim_avg - real_med) / real_med * 100) if real_med > 0 else 0
        if diff_pct > 15:
            over += 1
        elif diff_pct < -15:
            under += 1
        else:
            accurate += 1

    total_sim_cost = df['sim_spread_pts'].sum()
    total_real_cost = df['median_spread_pts'].sum()
    overall_diff = ((total_sim_cost - total_real_cost) / total_real_cost * 100) if total_real_cost > 0 else 0

    print(f'  Pairs overestimating spread:  {over}')
    print(f'  Pairs underestimating spread: {under}')
    print(f'  Pairs accurate (±15%):        {accurate}')
    print(f'  Overall spread bias:          {overall_diff:+.1f}%')
    print(f'  -> Positive = sim is conservative (good), Negative = sim is optimistic (risky)')


if __name__ == '__main__':
    asyncio.run(main())
