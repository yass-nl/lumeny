"""
Scaling Estimate — Project PnL at different capital levels.

Takes $1M simulation results and scales them, applying capacity caps
at the worst trading hour (21:00 UTC).
"""

# $1M simulation results
SIM_RESULTS = {
    'AUDUSD': {'trades': 153, 'wr': 0.562, 'avg_pnl': 628.65, 'total_pnl': 96_183.74},
    'EURGBP': {'trades': 140, 'wr': 0.664, 'avg_pnl': 265.02, 'total_pnl': 37_103.22},
    'EURJPY': {'trades': 100, 'wr': 0.630, 'avg_pnl': 159.36, 'total_pnl': 15_936.24},
    'EURUSD': {'trades':  54, 'wr': 0.611, 'avg_pnl': 1910.29, 'total_pnl': 103_155.54},
    'GBPJPY': {'trades': 117, 'wr': 0.564, 'avg_pnl':  99.55, 'total_pnl': 11_647.20},
    'GBPUSD': {'trades':  65, 'wr': 0.631, 'avg_pnl': 1877.79, 'total_pnl': 122_056.19},
    'NZDUSD': {'trades': 193, 'wr': 0.622, 'avg_pnl': 123.63, 'total_pnl': 23_859.90},
    'USDCAD': {'trades':  75, 'wr': 0.600, 'avg_pnl': 401.46, 'total_pnl': 30_109.82},
    'USDCHF': {'trades': 146, 'wr': 0.452, 'avg_pnl': -16.48, 'total_pnl': -2_405.73},
    'USDJPY': {'trades':  81, 'wr': 0.593, 'avg_pnl': 881.16, 'total_pnl': 71_373.72},
}

TOTAL_PNL_1M = sum(v['total_pnl'] for v in SIM_RESULTS.values())

# 1% of hourly spot volume at 21:00 UTC (worst hour) — in lots
# From BIS 2022 data
CAPACITY_LOTS_21H = {
    'EURUSD': 955, 'GBPUSD': 400, 'USDJPY': 568, 'USDCHF': 92,
    'AUDUSD': 165, 'USDCAD': 122, 'NZDUSD': 46,
    'EURJPY': 80,  'GBPJPY': 57,  'EURGBP': 60,
}

# Average capacity across 19-23 UTC (more realistic since not all trades at 21)
CAPACITY_LOTS_AVG = {
    'EURUSD': 1146, 'GBPUSD': 480, 'USDJPY': 682, 'USDCHF': 112,
    'AUDUSD': 198,  'USDCAD': 147, 'NZDUSD': 55,
    'EURJPY': 100,  'GBPJPY': 71,  'EURGBP': 73,
}

# At $1M with 0.5% risk and avg ATR ~50 pips: ~10 lots per trade
BASE_LOTS_AT_1M = 10

def estimate_scaling(equity_M, use_avg_cap=True):
    """
    Scale each pair's PnL from $1M baseline.
    Position size scales linearly with equity, but is capped at capacity.
    """
    scale = equity_M  # e.g. 10 for $10M
    target_lots = BASE_LOTS_AT_1M * scale

    caps = CAPACITY_LOTS_AVG if use_avg_cap else CAPACITY_LOTS_21H

    results = {}
    for pair, data in SIM_RESULTS.items():
        cap = caps[pair]
        effective_lots = min(target_lots, cap)
        # How much of the desired size can we actually deploy?
        utilization = effective_lots / target_lots
        # PnL scales with position size
        scaled_pnl = data['total_pnl'] * (effective_lots / BASE_LOTS_AT_1M)
        results[pair] = {
            'target_lots': target_lots,
            'effective_lots': effective_lots,
            'utilization': utilization,
            'capped': effective_lots < target_lots,
            'scaled_pnl': scaled_pnl,
            'base_pnl': data['total_pnl'],
        }
    return results


def main():
    equity_levels = [1, 3, 5, 7, 10, 15, 20, 30, 40]

    print('=' * 120)
    print(f'SCALING PROJECTIONS — Based on $1M simulation (total PnL: ${TOTAL_PNL_1M:,.0f})')
    print('=' * 120)

    # ── Overview Table ──
    print(f'\n{"Equity":>10} {"Target Lots":>12} {"Net PnL (21h)":>15} {"Return (21h)":>13} '
          f'{"Net PnL (avg)":>15} {"Return (avg)":>13} {"Pairs Capped":>13}')
    print('-' * 100)

    for eq in equity_levels:
        r21 = estimate_scaling(eq, use_avg_cap=False)
        ravg = estimate_scaling(eq, use_avg_cap=True)

        pnl_21 = sum(v['scaled_pnl'] for v in r21.values())
        pnl_avg = sum(v['scaled_pnl'] for v in ravg.values())
        n_capped = sum(1 for v in ravg.values() if v['capped'])

        print(f'${eq:>8}M {eq * BASE_LOTS_AT_1M:>10}L '
              f'${pnl_21:>13,.0f} {pnl_21/(eq*1e6)*100:>11.1f}% '
              f'${pnl_avg:>13,.0f} {pnl_avg/(eq*1e6)*100:>11.1f}% '
              f'{n_capped:>13}')

    # ── Detailed $30M Breakdown ──
    print(f'\n{"=" * 120}')
    print('DETAILED BREAKDOWN AT $30M')
    print(f'{"=" * 120}')

    eq = 30
    target = eq * BASE_LOTS_AT_1M

    print(f'\nTarget lots per trade: {target}')
    print(f'\nUsing AVERAGE capacity across 19:00-23:00 UTC window:')
    print(f'\n{"Pair":<10} {"Base PnL":>12} {"Target":>8} {"Cap":>8} {"Effective":>10} '
          f'{"Util%":>8} {"Scaled PnL":>14} {"Status":>10}')
    print('-' * 90)

    ravg = estimate_scaling(eq, use_avg_cap=True)
    total_scaled = 0
    total_uncapped = 0

    for pair in sorted(ravg.keys()):
        v = ravg[pair]
        status = 'CAPPED' if v['capped'] else 'ok'
        total_scaled += v['scaled_pnl']
        if not v['capped']:
            total_uncapped += v['scaled_pnl']
        print(f'{pair:<10} ${v["base_pnl"]:>10,.0f} {v["target_lots"]:>7,.0f}L '
              f'{CAPACITY_LOTS_AVG[pair]:>7,.0f}L {v["effective_lots"]:>9,.0f}L '
              f'{v["utilization"]*100:>7.1f}% ${v["scaled_pnl"]:>12,.0f} {status:>10}')

    print('-' * 90)
    print(f'{"TOTAL":<10} ${TOTAL_PNL_1M:>10,.0f} {"":>8} {"":>8} {"":>10} '
          f'{"":>8} ${total_scaled:>12,.0f}')

    pnl_loss = (1 - total_scaled / (TOTAL_PNL_1M * eq)) * 100
    print(f'\nProjected PnL at $30M:  ${total_scaled:,.0f}')
    print(f'Return:                 {total_scaled/(eq*1e6)*100:.1f}%')
    print(f'PnL lost to caps:       {pnl_loss:.1f}%')
    print(f'vs linear scaling:      ${TOTAL_PNL_1M * eq:,.0f} (if no caps)')

    # ── Same for worst case (21h only) ──
    print(f'\nUsing WORST-CASE capacity (21:00 UTC only):')
    print(f'\n{"Pair":<10} {"Target":>8} {"Cap":>8} {"Effective":>10} '
          f'{"Util%":>8} {"Scaled PnL":>14} {"Status":>10}')
    print('-' * 70)

    r21 = estimate_scaling(eq, use_avg_cap=False)
    total_21 = 0
    for pair in sorted(r21.keys()):
        v = r21[pair]
        status = 'CAPPED' if v['capped'] else 'ok'
        total_21 += v['scaled_pnl']
        print(f'{pair:<10} {v["target_lots"]:>7,.0f}L '
              f'{CAPACITY_LOTS_21H[pair]:>7,.0f}L {v["effective_lots"]:>9,.0f}L '
              f'{v["utilization"]*100:>7.1f}% ${v["scaled_pnl"]:>12,.0f} {status:>10}')

    print('-' * 70)
    print(f'{"TOTAL":<10} {"":>8} {"":>8} {"":>10} {"":>8} ${total_21:>12,.0f}')
    print(f'\nWorst-case PnL at $30M: ${total_21:,.0f}  ({total_21/(eq*1e6)*100:.1f}%)')
    print(f'Realistic PnL at $30M:  ${total_scaled:,.0f}  ({total_scaled/(eq*1e6)*100:.1f}%)')
    print(f'\nReality is somewhere between these two numbers.')


if __name__ == '__main__':
    main()
