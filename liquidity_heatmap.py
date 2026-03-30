"""
Liquidity Heatmap — Hourly spot volume and capacity per pair, 24h UTC.

Uses BIS 2022 Triennial Survey daily turnover + academic intraday volume curves.
Shows estimated spot volume ($M) and max safe lots (1% of hourly volume) for every
hour of the day, for each pair.
"""

import numpy as np

# ── BIS 2022 Daily Turnover (USD billions, total instruments) ──
SPOT_FRACTION = 0.28

BIS_DAILY_TOTAL_USD_BN = {
    'EURUSD': 1706, 'USDJPY': 1015, 'GBPUSD': 714, 'AUDUSD': 294,
    'USDCAD': 218,  'USDCHF': 164, 'NZDUSD': 82,  'EURJPY': 114,
    'GBPJPY': 81,   'EURGBP': 108, 'EURAUD': 28,  'AUDJPY': 34,
    'CADJPY': 14,   'CHFJPY': 18,  'AUDNZD': 7,
}

# ── Intraday Volume Distributions (% of daily per hour UTC) ──
HOURLY_VOL_PCT_MAJOR = {
    0: 2.5, 1: 3.0, 2: 3.5, 3: 3.5, 4: 3.0, 5: 3.0,
    6: 4.0, 7: 5.5, 8: 6.5, 9: 6.5, 10: 6.0, 11: 5.5,
    12: 6.0, 13: 7.0, 14: 7.5, 15: 7.0, 16: 5.5, 17: 4.0,
    18: 3.0, 19: 2.5, 20: 2.5, 21: 2.0, 22: 2.5, 23: 2.5,
}

HOURLY_VOL_PCT_JPY_CROSS = {
    0: 4.0, 1: 4.5, 2: 5.0, 3: 5.0, 4: 4.5, 5: 4.0,
    6: 4.0, 7: 5.0, 8: 5.5, 9: 5.5, 10: 5.0, 11: 4.5,
    12: 5.0, 13: 5.5, 14: 6.0, 15: 5.5, 16: 4.5, 17: 3.5,
    18: 3.0, 19: 2.5, 20: 2.5, 21: 2.5, 22: 3.5, 23: 4.0,
}

HOURLY_VOL_PCT_AUDNZD = {
    0: 4.5, 1: 5.0, 2: 5.5, 3: 5.0, 4: 4.5, 5: 4.0,
    6: 4.0, 7: 5.0, 8: 5.5, 9: 5.0, 10: 4.5, 11: 4.0,
    12: 4.5, 13: 5.0, 14: 5.5, 15: 5.0, 16: 4.0, 17: 3.5,
    18: 3.0, 19: 2.5, 20: 2.5, 21: 2.5, 22: 3.5, 23: 4.5,
}

JPY_CROSSES = {'EURJPY', 'GBPJPY', 'AUDJPY', 'CADJPY', 'CHFJPY'}
AUD_NZD_CROSSES = {'AUDNZD', 'EURAUD'}

LOT_UNITS = 100_000
IMPACT_PCT = 0.01  # 1% conservative threshold

PAIRS_LIQUID = ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD', 'USDCAD', 'NZDUSD',
                'EURJPY', 'GBPJPY', 'EURGBP']
PAIRS_ILLIQUID = ['EURAUD', 'AUDJPY', 'CADJPY', 'CHFJPY', 'AUDNZD']
ALL_PAIRS = PAIRS_LIQUID + PAIRS_ILLIQUID


def get_hourly_pct(pair, hour):
    if pair in JPY_CROSSES:
        return HOURLY_VOL_PCT_JPY_CROSS[hour] / 100
    elif pair in AUD_NZD_CROSSES:
        return HOURLY_VOL_PCT_AUDNZD[hour] / 100
    else:
        return HOURLY_VOL_PCT_MAJOR[hour] / 100


def hourly_spot_volume_M(pair, hour):
    """Estimated hourly spot volume in $M."""
    daily_spot = BIS_DAILY_TOTAL_USD_BN[pair] * SPOT_FRACTION  # $B
    return daily_spot * get_hourly_pct(pair, hour) * 1000  # $M


def max_lots_at_hour(pair, hour):
    """Max lots at 1% of hourly spot volume."""
    vol_usd = hourly_spot_volume_M(pair, hour) * 1e6  # to $
    return (vol_usd * IMPACT_PCT) / LOT_UNITS


def session_label(hour):
    """Trading session active at this hour."""
    if 0 <= hour <= 8:
        return 'Asia'
    elif 7 <= hour <= 16:
        return 'London'
    elif 12 <= hour <= 21:
        return 'NY'
    elif hour >= 22 or hour <= 1:
        return 'Asia'
    return ''


def session_tag(hour):
    tags = []
    if 0 <= hour <= 8: tags.append('TKY')
    if 7 <= hour <= 16: tags.append('LDN')
    if 12 <= hour <= 21: tags.append('NY')
    if hour >= 22 or hour <= 1: tags.append('TKY')
    return '/'.join(tags) if tags else ''


def main():
    # ═══════════════════════════════════════════════════════════════
    # SECTION 1: Hourly Spot Volume Heatmap ($M)
    # ═══════════════════════════════════════════════════════════════
    print('=' * 130)
    print('HOURLY SPOT VOLUME BY PAIR ($M) — Based on BIS 2022 Triennial Survey')
    print('=' * 130)

    # Header
    header = f'{"Hour":>6} {"Session":<10}'
    for pair in ALL_PAIRS:
        header += f' {pair:>8}'
    header += f' {"TOTAL":>10}'
    print(header)
    print('-' * 130)

    hourly_totals = []
    for hour in range(24):
        tag = session_tag(hour)
        row = f'{hour:>4}:00 {tag:<10}'
        total = 0
        for pair in ALL_PAIRS:
            vol = hourly_spot_volume_M(pair, hour)
            total += vol
            row += f' {vol:>8,.0f}'
        row += f' {total:>10,.0f}'
        hourly_totals.append(total)

        # Highlight your trading hours
        marker = '  <-- YOUR WINDOW' if hour in [19, 20, 21, 22] else ''
        print(row + marker)

    print('-' * 130)
    # Daily totals
    row_total = f'{"DAILY":>6} {"":>10}'
    for pair in ALL_PAIRS:
        daily = BIS_DAILY_TOTAL_USD_BN[pair] * SPOT_FRACTION * 1000  # $M
        row_total += f' {daily:>8,.0f}'
    row_total += f' {sum(BIS_DAILY_TOTAL_USD_BN[p] for p in ALL_PAIRS) * SPOT_FRACTION * 1000:>10,.0f}'
    print(row_total)

    # ═══════════════════════════════════════════════════════════════
    # SECTION 2: Max Safe Lots (1% threshold) Heatmap
    # ═══════════════════════════════════════════════════════════════
    print(f'\n{"=" * 130}')
    print('MAX SAFE LOTS PER TRADE (1% of hourly spot volume)')
    print(f'{"=" * 130}')

    header = f'{"Hour":>6} {"Session":<10}'
    for pair in ALL_PAIRS:
        header += f' {pair:>8}'
    print(header)
    print('-' * 130)

    for hour in range(24):
        tag = session_tag(hour)
        row = f'{hour:>4}:00 {tag:<10}'
        for pair in ALL_PAIRS:
            lots = max_lots_at_hour(pair, hour)
            if lots >= 1000:
                row += f' {lots:>7,.0f}L'
            elif lots >= 100:
                row += f' {lots:>7,.0f}L'
            else:
                row += f' {lots:>7,.0f}L'
        marker = '  <-- YOUR WINDOW' if hour in [19, 20, 21, 22] else ''
        print(row + marker)

    # ═══════════════════════════════════════════════════════════════
    # SECTION 3: Your Trading Window Deep Dive
    # ═══════════════════════════════════════════════════════════════
    print(f'\n{"=" * 130}')
    print('YOUR TRADING WINDOW DEEP DIVE (19:00 - 23:00 UTC)')
    print(f'{"=" * 130}')

    window_hours = [19, 20, 21, 22, 23]

    for pair in ALL_PAIRS:
        daily_spot_M = BIS_DAILY_TOTAL_USD_BN[pair] * SPOT_FRACTION * 1000
        print(f'\n  {pair}  (Daily spot: ${daily_spot_M:,.0f}M)')
        print(f'  {"Hour":>6}  {"Vol ($M)":>10}  {"% of Daily":>10}  {"1% Cap (lots)":>14}  {"1% Cap ($M)":>12}  {"Bar":}')
        for hour in window_hours:
            vol = hourly_spot_volume_M(pair, hour)
            pct = get_hourly_pct(pair, hour) * 100
            lots = max_lots_at_hour(pair, hour)
            cap_M = lots * LOT_UNITS / 1e6
            bar_len = int(vol / 200)  # scale bar
            bar = '#' * max(bar_len, 1)
            print(f'  {hour:>4}:00  ${vol:>9,.0f}M  {pct:>9.1f}%  {lots:>13,.0f}L  ${cap_M:>10,.1f}M  {bar}')

    # ═══════════════════════════════════════════════════════════════
    # SECTION 4: Capacity at Different Equity Levels — Per Hour
    # ═══════════════════════════════════════════════════════════════
    equity_levels = [100_000, 500_000, 1_000_000, 3_000_000, 5_000_000, 7_000_000, 10_000_000, 20_000_000]
    window_hours_cap = [19, 20, 21, 22, 23, 0]  # 19:00 - 00:00 UTC

    for hour in window_hours_cap:
        tag = session_tag(hour)
        print(f'\n{"=" * 130}')
        print(f'CAPACITY CHECK AT {hour:02d}:00 UTC ({tag})')
        print(f'{"=" * 130}')

        # Show hourly volume for context
        vol_row = f'  Spot vol ($M):    '
        for pair in ALL_PAIRS:
            vol_row += f'  {hourly_spot_volume_M(pair, hour):>7,.0f}'
        print(vol_row)
        cap_row = f'  1% cap (lots):    '
        for pair in ALL_PAIRS:
            cap_row += f'  {max_lots_at_hour(pair, hour):>7,.0f}'
        print(cap_row)

        print(f'\n{"Equity":>12}  {"~Lots":>7}', end='')
        for pair in ALL_PAIRS:
            print(f'  {pair:>7}', end='')
        print(f'  {"Constrained":>12}')
        print('-' * (21 + len(ALL_PAIRS) * 9 + 14))

        for equity in equity_levels:
            risk = equity * 0.005
            approx_lots = risk / 500  # rough estimate
            row = f'${equity/1e6:>10.1f}M  {approx_lots:>5.0f}L'
            n_capped = 0
            n_tight = 0
            for pair in ALL_PAIRS:
                cap = max_lots_at_hour(pair, hour)
                if approx_lots > cap:
                    row += f'  {"CAPPED":>7}'
                    n_capped += 1
                elif approx_lots > cap * 0.5:
                    row += f'  {"~tight":>7}'
                    n_tight += 1
                else:
                    row += f'  {"ok":>7}'
            summary = f'{n_capped} capped'
            if n_tight:
                summary += f', {n_tight} tight'
            row += f'  {summary:>12}'
            print(row)

    # ═══════════════════════════════════════════════════════════════
    # SECTION 5: Session Summary
    # ═══════════════════════════════════════════════════════════════
    print(f'\n{"=" * 130}')
    print('SESSION VOLUME COMPARISON (% of daily volume)')
    print(f'{"=" * 130}')

    sessions = {
        'Asia (0-7 UTC)': range(0, 8),
        'London (7-16 UTC)': range(7, 17),
        'NY (12-21 UTC)': range(12, 22),
        'Your Window (19-23 UTC)': range(19, 24),
        'Thinnest (21-22 UTC)': range(21, 23),
    }

    print(f'\n{"Session":<25}', end='')
    for pair in ALL_PAIRS:
        print(f' {pair:>7}', end='')
    print()
    print('-' * (25 + len(ALL_PAIRS) * 8))

    for name, hours in sessions.items():
        row = f'{name:<25}'
        for pair in ALL_PAIRS:
            total_pct = sum(get_hourly_pct(pair, h) for h in hours) * 100
            row += f' {total_pct:>6.1f}%'
        print(row)


if __name__ == '__main__':
    main()
