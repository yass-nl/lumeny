"""
Multi-Period Backtester — Runs test_capital_sim_3.py across multiple
non-overlapping windows to get a statistically meaningful performance estimate.

Training cutoff: 2024-06-30 (all windows are out-of-sample).

Windows (each ~200 days / 6.5 months):
  Period 1: Jul 2024 – Jan 2025   (DATE_OFFSET ~630)
  Period 2: Jan 2025 – Aug 2025   (DATE_OFFSET ~430)
  Period 3: Feb 2025 – Sep 2025   (DATE_OFFSET ~200)  [already tested]
  Period 4: Sep 2025 – Mar 2026   (DATE_OFFSET ~0)    [already tested]

Since Period 2 and Period 3 overlap heavily, we use 3 clean windows:
  Window A: Jul 1 2024 – Jan 27 2025   (offset ~425)
  Window B: Jan 27 2025 – Sep 14 2025  (offset ~195)  — ~matches previous Period 1
  Window C: Sep 14 2025 – Mar 28 2026  (offset ~0)    — ~matches previous Period 2

Actually, with BACKTEST_DAYS=200, each window covers ~200 days.
The offset is days-back from today (2026-03-28):
  Window C end: 2026-03-28, start: ~2025-09-09  → offset 0
  Window B end: 2025-09-09, start: ~2025-02-20  → offset 200
  Window A end: 2025-02-20, start: ~2024-08-04  → offset 400
  Window 0 end: 2024-08-04, start: ~2024-01-17  → offset 600 (partly in-sample!)

So 3 clean out-of-sample windows: offset 0, 200, 400
Window at offset 600 starts Jan 2024 which is IN-SAMPLE — skip it.
"""

import subprocess
import sys
import re
import os

# Each tuple: (label, DATE_OFFSET_DAYS)
WINDOWS = [
    ("Aug 2024 – Feb 2025", 400),
    ("Feb 2025 – Sep 2025", 200),
    ("Sep 2025 – Mar 2026", 0),
]

SCRIPT = "test_capital_sim_3.py"


def patch_offset(offset_days):
    """Temporarily patch DATE_OFFSET_DAYS in the script."""
    with open(SCRIPT, 'r') as f:
        content = f.read()

    patched = re.sub(
        r'^DATE_OFFSET_DAYS\s*=\s*\d+',
        f'DATE_OFFSET_DAYS = {offset_days}',
        content,
        flags=re.MULTILINE
    )

    with open(SCRIPT, 'w') as f:
        f.write(patched)


def extract_results(output):
    """Parse key metrics from simulation output."""
    metrics = {}

    # Total P&L:          $  509,019.83 (+50.90%)
    # pnl_usd is always NET (spread + slippage already deducted at lines 1078-1079)
    # Try multiple patterns — '&' can cause encoding issues on Windows
    m = re.search(r'Total P.L:\s+\$([\s+-]?[\d,]+(?:\.\d+)?)\s+\(([+-]?[\d.]+)%\)', output)
    if not m:
        # Fallback: parse Final equity and compute PnL from starting capital ($1M)
        m2 = re.search(r'Final equity:\s+\$([\s\d,]+(?:\.\d+)?)', output)
        if m2:
            final_eq = float(m2.group(1).replace(',', '').strip())
            pnl = final_eq - 1_000_000
            metrics['net_pnl'] = pnl
            metrics['return_pct'] = (pnl / 1_000_000) * 100
    else:
        metrics['net_pnl'] = float(m.group(1).replace(',', '').strip())
        metrics['return_pct'] = float(m.group(2))

    # Last-resort fallback: extract final equity from last trade line
    # Format: ... $1,653,374.39\n
    if 'net_pnl' not in metrics:
        all_equity = re.findall(r'\$\s*([\d,]+\.\d{2})\s*$', output, re.MULTILINE)
        if all_equity:
            final_eq = float(all_equity[-1].replace(',', ''))
            if final_eq > 500_000:  # sanity check
                metrics['net_pnl'] = final_eq - 1_000_000
                metrics['return_pct'] = ((final_eq - 1_000_000) / 1_000_000) * 100

    # Win rate:           58.8%
    m = re.search(r'Win rate:\s+([\d.]+)%', output)
    if m:
        metrics['win_rate'] = float(m.group(1))

    # Total trades:       1124
    m = re.search(r'Total trades:\s+([\d,]+)', output)
    if m:
        metrics['total_trades'] = int(m.group(1).replace(',', ''))

    # Sharpe (annualized): 4.24
    m = re.search(r'Sharpe \(annualized\):\s+([\d.]+)', output)
    if m:
        metrics['sharpe'] = float(m.group(1))

    # Max drawdown:       $  12,345.67 (1.23%)
    m = re.search(r'Max drawdown:.*?\(([\d.]+)%\)', output)
    if m:
        metrics['max_dd'] = float(m.group(1))

    # Profit factor:      1.85
    m = re.search(r'Profit factor:\s+([\d.]+)', output)
    if m:
        metrics['profit_factor'] = float(m.group(1))

    # Backtest starts: 2025-09-18
    m = re.search(r'Backtest starts:\s+(\d{4}-\d{2}-\d{2})', output)
    if m:
        metrics['start_date'] = m.group(1)

    return metrics


def main():
    results = []
    original_offset = None

    # Save original offset
    with open(SCRIPT, 'r') as f:
        m = re.search(r'^DATE_OFFSET_DAYS\s*=\s*(\d+)', f.read(), re.MULTILINE)
        if m:
            original_offset = int(m.group(1))

    print('=' * 100)
    print('MULTI-PERIOD BACKTEST')
    print('Training cutoff: 2024-06-30 — all windows are out-of-sample')
    print('=' * 100)

    for label, offset in WINDOWS:
        print(f'\n{"=" * 100}')
        print(f'RUNNING: {label} (DATE_OFFSET={offset})')
        print(f'{"=" * 100}\n')

        patch_offset(offset)

        try:
            result = subprocess.run(
                [sys.executable, SCRIPT],
                capture_output=True,
                text=True,
                timeout=600,  # 10 min max per window
                cwd=os.path.dirname(os.path.abspath(__file__)) or '.',
            )
            output = result.stdout + result.stderr
            # Save full output for debugging
            with open(f'sim_output_offset{offset}.txt', 'w', encoding='utf-8') as f:
                f.write(output)
            print(output[-3000:] if len(output) > 3000 else output)  # tail of output

            metrics = extract_results(output)
            metrics['label'] = label
            metrics['offset'] = offset
            results.append(metrics)

        except subprocess.TimeoutExpired:
            print(f'  TIMEOUT after 600s — skipping')
            results.append({'label': label, 'offset': offset, 'error': 'TIMEOUT'})
        except Exception as e:
            print(f'  ERROR: {e}')
            results.append({'label': label, 'offset': offset, 'error': str(e)})

    # Restore original offset
    if original_offset is not None:
        patch_offset(original_offset)

    # ── Summary ──
    print(f'\n\n{"=" * 100}')
    print('MULTI-PERIOD SUMMARY')
    print(f'{"=" * 100}')
    print(f'\n{"Period":<25} {"Trades":>8} {"WR%":>7} {"Net PnL":>14} {"Return":>9} {"Sharpe":>8} {"MaxDD":>8} {"PF":>7}')
    print('-' * 90)

    total_pnl = 0
    total_trades = 0
    valid = 0

    for r in results:
        if 'error' in r:
            print(f'{r["label"]:<25} {"ERROR: " + r["error"]}')
            continue

        valid += 1
        pnl = r.get('net_pnl', 0)
        trades = r.get('total_trades', 0)
        total_pnl += pnl
        total_trades += trades

        print(f'{r["label"]:<25} {trades:>8,} {r.get("win_rate", 0):>6.1f}% '
              f'${pnl:>+12,.0f} {r.get("return_pct", 0):>+8.1f}% '
              f'{r.get("sharpe", 0):>7.2f} {r.get("max_dd", 0):>7.1f}% '
              f'{r.get("profit_factor", 0):>6.2f}')

    if valid > 0:
        print('-' * 90)
        avg_return = sum(r.get('return_pct', 0) for r in results if 'error' not in r) / valid
        avg_sharpe = sum(r.get('sharpe', 0) for r in results if 'error' not in r) / valid
        avg_wr = sum(r.get('win_rate', 0) for r in results if 'error' not in r) / valid
        print(f'{"AVERAGE":<25} {total_trades//valid:>8,} {avg_wr:>6.1f}% '
              f'${total_pnl/valid:>+12,.0f} {avg_return:>+8.1f}% '
              f'{avg_sharpe:>7.2f}')
        print(f'{"TOTAL":<25} {total_trades:>8,} {"":>7} '
              f'${total_pnl:>+12,.0f}')

        print(f'\nAnnualized estimate (from {valid} x ~6.5-month windows):')
        months = valid * 6.5
        annual_return = (total_pnl / WINDOWS[0][1] if False else avg_return * (12 / 6.5))
        print(f'  Avg return per window:   {avg_return:+.1f}%')
        print(f'  Annualized (x1.85):      {avg_return * 1.85:+.1f}%')
        print(f'  Avg Sharpe per window:   {avg_sharpe:.2f}')


if __name__ == '__main__':
    main()
