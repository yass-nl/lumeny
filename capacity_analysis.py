"""
Market Capacity Analysis — Maximum position sizing before market impact.

Uses BIS Triennial Survey (2022) daily average turnover data + academic
time-of-day volume distributions to estimate capacity at 21-23 UTC.

Sources:
- BIS Triennial Central Bank Survey 2022 (daily turnover by pair)
- Breedon & Ranaldo (2013), Chaboud et al. (2014) — intraday volume curves
- 1-2% of hourly volume = conservative market impact threshold
"""

import pandas as pd
import numpy as np

# ── BIS 2022 Triennial Survey: Daily Average Turnover (USD billions) ──
# Source: https://www.bis.org/statistics/rpfx22.htm
# These are spot + forwards + swaps combined, April 2022 averages.
# Spot-only is roughly 28% of total for most pairs.
SPOT_FRACTION = 0.28

BIS_DAILY_TOTAL_USD_BN = {
    'EURUSD': 1706,
    'USDJPY': 1015,
    'GBPUSD': 714,
    'AUDUSD': 294,
    'USDCAD': 218,
    'USDCHF': 164,
    'NZDUSD': 82,
    'EURJPY': 114,
    'GBPJPY': 81,
    'EURGBP': 108,
    'EURAUD': 28,
    'AUDJPY': 34,
    'CADJPY': 14,
    'CHFJPY': 18,
    'AUDNZD': 7,
}

# ── Intraday Volume Distribution (% of daily volume per hour UTC) ──
# Based on Chaboud et al. and empirical studies.
# FX volume follows a bimodal pattern: Asian open, London/NY overlap.
# Percentages are approximate and pair-dependent.

# Generic distribution (majors)
HOURLY_VOL_PCT_MAJOR = {
    0: 2.5, 1: 3.0, 2: 3.5, 3: 3.5, 4: 3.0, 5: 3.0,
    6: 4.0, 7: 5.5, 8: 6.5, 9: 6.5, 10: 6.0, 11: 5.5,
    12: 6.0, 13: 7.0, 14: 7.5, 15: 7.0, 16: 5.5, 17: 4.0,
    18: 3.0, 19: 2.5, 20: 2.5, 21: 2.0, 22: 2.5, 23: 2.5,
}

# JPY crosses: more volume during Asian session
HOURLY_VOL_PCT_JPY_CROSS = {
    0: 4.0, 1: 4.5, 2: 5.0, 3: 5.0, 4: 4.5, 5: 4.0,
    6: 4.0, 7: 5.0, 8: 5.5, 9: 5.5, 10: 5.0, 11: 4.5,
    12: 5.0, 13: 5.5, 14: 6.0, 15: 5.5, 16: 4.5, 17: 3.5,
    18: 3.0, 19: 2.5, 20: 2.5, 21: 2.5, 22: 3.5, 23: 4.0,
}

# AUD/NZD crosses: volume shifted toward Asian/early London
HOURLY_VOL_PCT_AUDNZD = {
    0: 4.5, 1: 5.0, 2: 5.5, 3: 5.0, 4: 4.5, 5: 4.0,
    6: 4.0, 7: 5.0, 8: 5.5, 9: 5.0, 10: 4.5, 11: 4.0,
    12: 4.5, 13: 5.0, 14: 5.5, 15: 5.0, 16: 4.0, 17: 3.5,
    18: 3.0, 19: 2.5, 20: 2.5, 21: 2.5, 22: 3.5, 23: 4.5,
}

JPY_CROSSES = {'EURJPY', 'GBPJPY', 'AUDJPY', 'CADJPY', 'CHFJPY'}
AUD_NZD_CROSSES = {'AUDNZD', 'EURAUD'}

PAIRS = [
    'EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD', 'USDCAD', 'NZDUSD',
    'EURJPY', 'GBPJPY', 'EURGBP', 'EURAUD', 'AUDJPY', 'CADJPY', 'CHFJPY', 'AUDNZD',
]

# Your trading config
TRADING_HOURS = [20, 21, 22]  # main trading window
IMPACT_THRESHOLD_PCT = 0.01   # 1% of hourly volume (conservative)
IMPACT_THRESHOLD_AGG = 0.02   # 2% (aggressive)
AVG_SIMULTANEOUS_POSITIONS = 4  # from trade log analysis
LOT_UNITS = 100_000


def get_hourly_pct(pair, hour):
    if pair in JPY_CROSSES:
        return HOURLY_VOL_PCT_JPY_CROSS[hour] / 100
    elif pair in AUD_NZD_CROSSES:
        return HOURLY_VOL_PCT_AUDNZD[hour] / 100
    else:
        return HOURLY_VOL_PCT_MAJOR[hour] / 100


def main():
    print('=' * 95)
    print('MARKET CAPACITY ANALYSIS')
    print('=' * 95)
    print(f'Source: BIS Triennial Survey 2022 + academic intraday volume distributions')
    print(f'Impact threshold: {IMPACT_THRESHOLD_PCT*100:.0f}% (conservative) / {IMPACT_THRESHOLD_AGG*100:.0f}% (aggressive) of hourly volume')
    print(f'Trading window: {TRADING_HOURS[0]:02d}:00 - {TRADING_HOURS[-1]+1:02d}:00 UTC')
    print(f'Avg simultaneous positions: {AVG_SIMULTANEOUS_POSITIONS}')
    print()

    # ── Per-Pair Analysis ──
    print(f'{"Pair":<10} {"Daily Vol":>12} {"Spot Daily":>12} {"21-22 Hourly":>14} '
          f'{"1% Ceiling":>12} {"2% Ceiling":>12} {"Max Lots(1%)":>13}')
    print('-' * 95)

    pair_ceilings = {}

    for pair in PAIRS:
        daily_total = BIS_DAILY_TOTAL_USD_BN[pair]
        daily_spot = daily_total * SPOT_FRACTION

        # Average hourly volume during 21-22 UTC window
        avg_hourly_pct = np.mean([get_hourly_pct(pair, h) for h in TRADING_HOURS])
        hourly_vol = daily_spot * avg_hourly_pct  # in USD billions

        ceiling_1pct = hourly_vol * IMPACT_THRESHOLD_PCT  # USD billions
        ceiling_2pct = hourly_vol * IMPACT_THRESHOLD_AGG

        # Convert to lots (1 lot = 100,000 units of base currency, ~$100K notional)
        max_lots_1pct = ceiling_1pct * 1e9 / LOT_UNITS

        pair_ceilings[pair] = {
            'daily_total': daily_total,
            'daily_spot': daily_spot,
            'hourly_vol': hourly_vol,
            'ceiling_1pct': ceiling_1pct,
            'ceiling_2pct': ceiling_2pct,
            'max_lots_1pct': max_lots_1pct,
        }

        print(f'{pair:<10} ${daily_total:>9,.0f}B ${daily_spot:>9,.1f}B '
              f'${hourly_vol*1000:>11,.0f}M '
              f'${ceiling_1pct*1000:>9,.1f}M '
              f'${ceiling_2pct*1000:>9,.1f}M '
              f'{max_lots_1pct:>11,.0f}')

    # ── Portfolio-Level Analysis ──
    print(f'\n{"=" * 95}')
    print('PORTFOLIO CAPACITY (based on your actual trade distribution)')
    print(f'{"=" * 95}')

    # Weight by trade frequency from 3h sim results
    trade_counts = {
        'AUDJPY': 40, 'AUDNZD': 138, 'AUDUSD': 39, 'CADJPY': 46,
        'CHFJPY': 71, 'EURAUD': 22, 'EURGBP': 62, 'EURJPY': 39,
        'EURUSD': 20, 'GBPJPY': 34, 'GBPUSD': 26, 'NZDUSD': 51,
        'USDCAD': 25, 'USDCHF': 42, 'USDJPY': 30,
    }
    total_trades = sum(trade_counts.values())

    print(f'\nTrade frequency weights (from 3h hold sim):')

    # The binding constraint is the pair with the smallest ceiling
    # relative to how often you trade it
    print(f'\n{"Pair":<10} {"Trades":>8} {"Weight":>8} {"1% Ceiling":>12} '
          f'{"Per-Trade Max":>14} {"Bottleneck?":>12}')
    print('-' * 75)

    bottleneck_pair = None
    bottleneck_lots = float('inf')

    for pair in PAIRS:
        trades = trade_counts.get(pair, 0)
        weight = trades / total_trades
        ceiling = pair_ceilings[pair]['ceiling_1pct'] * 1000  # in $M
        # Per-trade max assuming you might have AVG_SIMULTANEOUS_POSITIONS at once
        per_trade_max_usd = (ceiling / AVG_SIMULTANEOUS_POSITIONS)  # $M per position
        per_trade_lots = per_trade_max_usd * 1e6 / LOT_UNITS

        is_bottleneck = ''
        if trades > 20 and per_trade_lots < bottleneck_lots:
            bottleneck_lots = per_trade_lots
            bottleneck_pair = pair
            is_bottleneck = '<-- binding'

        print(f'{pair:<10} {trades:>8} {weight:>7.1%} ${ceiling:>9,.1f}M '
              f'{per_trade_lots:>11,.0f} lots {is_bottleneck}')

    # ── Scaling Analysis ──
    print(f'\n{"=" * 95}')
    print('SCALING SCENARIOS')
    print(f'{"=" * 95}')
    print(f'Bottleneck pair: {bottleneck_pair} '
          f'(1% ceiling = ${pair_ceilings[bottleneck_pair]["ceiling_1pct"]*1000:,.1f}M/hr, '
          f'{AVG_SIMULTANEOUS_POSITIONS} simultaneous positions)')

    # Current config
    current_lots = 1.0
    current_notional = current_lots * LOT_UNITS

    scenarios = [
        ('Current (1 lot)', 1.0),
        ('5 lots', 5.0),
        ('10 lots', 10.0),
        ('25 lots', 25.0),
        ('50 lots', 50.0),
        ('100 lots', 100.0),
        ('500 lots', 500.0),
    ]

    print(f'\n{"Scenario":<20} {"Notional":>14} {"% of Bottleneck":>16} {"Impact Risk":>14}')
    print('-' * 70)

    for name, lots in scenarios:
        notional = lots * LOT_UNITS
        # With simultaneous positions
        total_exposure = notional * AVG_SIMULTANEOUS_POSITIONS
        bn_ceiling = pair_ceilings[bottleneck_pair]['ceiling_1pct'] * 1e9
        pct_of_ceiling = (notional / bn_ceiling) * 100  # single position vs ceiling

        if pct_of_ceiling < 5:
            risk = 'NEGLIGIBLE'
        elif pct_of_ceiling < 20:
            risk = 'LOW'
        elif pct_of_ceiling < 50:
            risk = 'MODERATE'
        elif pct_of_ceiling < 100:
            risk = 'HIGH'
        else:
            risk = 'EXCEEDS LIMIT'

        print(f'{name:<20} ${notional/1e6:>11,.1f}M {pct_of_ceiling:>14.1f}% {risk:>14}')

    # ── Summary ──
    print(f'\n{"=" * 95}')
    print('SUMMARY')
    print(f'{"=" * 95}')

    # Find max lots where all pairs stay under 50% of their 1% ceiling
    max_safe_lots = float('inf')
    for pair in PAIRS:
        if trade_counts.get(pair, 0) < 10:
            continue
        ceiling_usd = pair_ceilings[pair]['ceiling_1pct'] * 1e9
        # 50% of ceiling, divided by simultaneous positions
        safe_per_position = (ceiling_usd * 0.5) / AVG_SIMULTANEOUS_POSITIONS
        safe_lots = safe_per_position / LOT_UNITS
        if safe_lots < max_safe_lots:
            max_safe_lots = safe_lots
            limiting_pair = pair

    max_capital_per_trade = max_safe_lots * LOT_UNITS
    max_total_capital = max_capital_per_trade * AVG_SIMULTANEOUS_POSITIONS

    # With 50:1 leverage, margin needed
    margin_per_trade = max_capital_per_trade / 50
    total_margin = margin_per_trade * AVG_SIMULTANEOUS_POSITIONS

    print(f'  Bottleneck pair:           {limiting_pair}')
    print(f'  Safe position size:        {max_safe_lots:,.0f} lots (${max_capital_per_trade/1e6:,.1f}M notional)')
    print(f'  Total exposure (x{AVG_SIMULTANEOUS_POSITIONS}):      ${max_total_capital/1e6:,.1f}M')
    print(f'  Required margin (50:1):    ${total_margin/1e6:,.1f}M')
    print(f'  Required account size:     ${total_margin/1e6:,.1f}M (at 50:1 leverage)')
    print()
    print(f'  Conservative ceiling:      Stay under {max_safe_lots:,.0f} lots per trade')
    print(f'  This keeps you below 50% of the 1% hourly volume threshold')
    print(f'  on your most illiquid actively-traded pair ({limiting_pair}).')
    print()
    print(f'  NOTE: BIS volumes are from April 2022. Actual volumes fluctuate')
    print(f'  and may be lower during holidays, news events, or thin markets.')
    print(f'  Cross pairs (especially AUDNZD, CADJPY) have significantly less')
    print(f'  depth than majors -- these are your binding constraints.')


if __name__ == '__main__':
    main()
