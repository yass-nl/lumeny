import json

cells = []

# ── 0. Title ──────────────────────────────────────────────────────────────────
cells.append({
    "cell_type": "markdown", "metadata": {},
    "source": (
        "# LumenY 10 — System 2: Volatility-Driven Mean Reversion\n\n"
        "**Thesis:** When realized volatility spikes above its 24H norm, price is stretched far from "
        "its short-term trend, informed flow has recently surged, the return distribution shows no "
        "skew bias in the current direction, AND the market is in a confirmed mean-reverting regime "
        "(variance ratio < 30th pct) — the next 4H strongly reverts.\n\n"
        "**No training. Pure rules. Full history valid.**\n\n"
        "| Layer | Feature | Condition | Rationale |\n"
        "|-------|---------|-----------|----------|\n"
        "| Flow | `kyle_lambda_delta_3h` | top 30% | Informed flow just spiked |\n"
        "| Structure | `residual_12h` | bottom 25% (long) / top 25% (short) | Price stretched from 12H trend |\n"
        "| Volatility | `rv_zscore_24` | top 85% | Vol above 24H norm |\n"
        "| Distribution | `realized_skew` | bottom 30% (long) / top 30% (short) | No skew bias in current direction |\n"
        "| Regime | `vr_5` | bottom 30% | Confirmed mean-reverting regime |\n\n"
        "**Exit:** Fixed 4H hold\n\n"
        "**Active pairs:** AUDJPY, AUDNZD, AUDUSD, CADJPY, CHFJPY, EURAUD, EURGBP, "
        "EURJPY, EURUSD, NZDUSD, USDCAD  \n"
        "**Excluded:** GBPUSD (negative gross EV), USDJPY (zero margin), GBPJPY, USDCHF (thin margin)"
    )
})

# ── 1. Imports & config ───────────────────────────────────────────────────────
cells.append({
    "cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
    "source": """\
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path

FEATURES_6_DIR = Path('../backend/data/features_6')
FEATURES_8_DIR = Path('../backend/data/features_8')
PROCESSED_DIR  = Path('../backend/data/processed')

HOLD_H = 4

PIP_SIZE = {
    'AUDJPY':0.01, 'AUDNZD':0.0001, 'AUDUSD':0.0001,
    'CADJPY':0.01, 'CHFJPY':0.01,
    'EURAUD':0.0001, 'EURGBP':0.0001, 'EURJPY':0.01, 'EURUSD':0.0001,
    'NZDUSD':0.0001, 'USDCAD':0.0001,
}
SPREAD_PIPS = {
    'AUDJPY':3.0, 'AUDNZD':3.0, 'AUDUSD':1.5,
    'CADJPY':3.0, 'CHFJPY':3.0,
    'EURAUD':3.0, 'EURGBP':1.5, 'EURJPY':2.0, 'EURUSD':1.0,
    'NZDUSD':2.0, 'USDCAD':2.0,
}
MAX_1H_MOVE = {
    'AUDJPY':250, 'AUDNZD':80,  'AUDUSD':120,
    'CADJPY':250, 'CHFJPY':250,
    'EURAUD':150, 'EURGBP':100, 'EURJPY':300, 'EURUSD':150,
    'NZDUSD':100, 'USDCAD':150,
}
PAIRS = list(PIP_SIZE.keys())
print(f'System 2 | {len(PAIRS)} pairs | {HOLD_H}H hold')
print('Pairs:', ', '.join(PAIRS))
"""
})

# ── 2. Load data ──────────────────────────────────────────────────────────────
cells.append({
    "cell_type": "markdown", "metadata": {},
    "source": "## 1. Load features_6 + features_8 + 1H closes"
})
cells.append({
    "cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
    "source": """\
dfs = []
for pair, pip in PIP_SIZE.items():
    df6 = pd.read_parquet(FEATURES_6_DIR / f'{pair}_features.parquet')
    df8 = pd.read_parquet(FEATURES_8_DIR / f'{pair}_geometric.parquet')
    df8['pair'] = pair
    df6r = df6.reset_index()
    df8r = df8.reset_index()
    idx = df6r.columns[0]
    drop_dup = [c for c in df8r.columns if c in df6r.columns and c not in [idx, 'pair']]
    df8r = df8r.drop(columns=drop_dup, errors='ignore')
    df = pd.merge(df6r, df8r, on=[idx, 'pair'], how='inner').set_index(idx).sort_index()

    df1h = pd.read_parquet(PROCESSED_DIR / f'{pair}_1H.parquet')
    if 'datetime' in df1h.columns:
        df1h = df1h.set_index('datetime')
    df1h.index = pd.to_datetime(df1h.index)
    close = df1h['close'].reindex(df.index)

    fwd = (close.shift(-HOLD_H) - close) / pip
    ret_1h = ((close.shift(-1) - close) / pip).abs()
    holiday = ~(
        ((df.index.month == 12) & (df.index.day.isin([24, 25, 26, 31]))) |
        ((df.index.month == 1)  & (df.index.day.isin([1, 2])))
    )
    clean = (ret_1h.reindex(df.index) <= MAX_1H_MOVE[pair]) & holiday
    liquid = (df.index.hour >= 7) & (df.index.hour <= 21)
    df = df[clean & liquid].copy()
    df['fwd_pips'] = fwd
    df['spread']   = SPREAD_PIPS[pair]
    dfs.append(df)

data = pd.concat(dfs).sort_index()
data = data.dropna(subset=['fwd_pips', 'realized_skew', 'residual_12h',
                            'rv_zscore_24', 'kyle_lambda_delta_3h', 'vr_5'])
print(f'Loaded: {data.shape}')
print(f'Period: {data.index.min().date()} -> {data.index.max().date()}')
print(f'Pairs:  {sorted(data["pair"].unique())}')
"""
})

# ── 3. Rank features per pair ─────────────────────────────────────────────────
cells.append({
    "cell_type": "markdown", "metadata": {},
    "source": "## 2. Per-pair percentile ranks"
})
cells.append({
    "cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
    "source": """\
RANK_COLS = ['kyle_lambda_delta_3h', 'residual_12h', 'rv_zscore_24', 'realized_skew', 'vr_5']
for col in RANK_COLS:
    data[f'{col}_r'] = data.groupby('pair')[col].rank(pct=True)

print('Rank features computed.')
print()
print('Feature summary (median per rank bucket should be monotone):')
for col in RANK_COLS:
    r = data[f'{col}_r']
    q25 = data.loc[r < 0.25, col].median()
    q50 = data.loc[(r >= 0.25) & (r < 0.75), col].median()
    q75 = data.loc[r >= 0.75, col].median()
    print(f'  {col:<30} Q25:{q25:>10.4f}  Q50:{q50:>10.4f}  Q75:{q75:>10.4f}')
"""
})

# ── 4. Apply system ───────────────────────────────────────────────────────────
cells.append({
    "cell_type": "markdown", "metadata": {},
    "source": "## 3. System 2 signal"
})
cells.append({
    "cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
    "source": """\
def apply_system2(df):
    \"\"\"
    System 2: Volatility-Driven Mean Reversion
    +1 = long (revert up), -1 = short (revert down), 0 = no trade
    \"\"\"
    long_signal = (
        (df['kyle_lambda_delta_3h_r'] > 0.70) &  # flow spike
        (df['residual_12h_r']         < 0.25) &  # price below 12H trend
        (df['rv_zscore_24_r']         > 0.85) &  # vol elevated
        (df['realized_skew_r']        < 0.30) &  # no upward skew bias
        (df['vr_5_r']                 < 0.30)    # mean-reverting regime
    )
    short_signal = (
        (df['kyle_lambda_delta_3h_r'] < 0.30) &  # flow collapse
        (df['residual_12h_r']         > 0.75) &  # price above 12H trend
        (df['rv_zscore_24_r']         > 0.85) &  # vol elevated
        (df['realized_skew_r']        > 0.70) &  # no downward skew bias
        (df['vr_5_r']                 < 0.30)    # mean-reverting regime
    )
    sig = pd.Series(0, index=df.index)
    sig[long_signal]  =  1
    sig[short_signal] = -1
    return sig

data['signal'] = apply_system2(data)
trades = data[data['signal'] != 0].copy()
trades['pnl'] = trades['signal'] * trades['fwd_pips'] - trades['spread']

print(f'Total signals : {len(trades):,}')
print(f'  Long        : {(trades["signal"]==1).sum():,}')
print(f'  Short       : {(trades["signal"]==-1).sum():,}')
nm = (trades.index.max() - trades.index.min()).days / 30
print(f'Trades/month  : {len(trades)/nm:.1f}')
"""
})

# ── 5. Full history results ───────────────────────────────────────────────────
cells.append({
    "cell_type": "markdown", "metadata": {},
    "source": "## 4. Results"
})
cells.append({
    "cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
    "source": """\
pnl = trades['pnl'].dropna()
wins   = pnl[pnl > 0]
losses = pnl[pnl <= 0]
nm     = (pnl.index.max() - pnl.index.min()).days / 30
sharpe = (pnl.mean() / pnl.std()) * np.sqrt(252 * 24 / HOLD_H)
cut18  = pnl.index.max() - pd.DateOffset(months=18)
r18    = pnl[pnl.index >= cut18]
sh18   = (r18.mean() / r18.std()) * np.sqrt(252 * 24 / HOLD_H) if r18.std() > 0 else 0

print('=' * 58)
print('SYSTEM 2 — FULL HISTORY')
print('=' * 58)
print(f'Period        {pnl.index.min().date()} -> {pnl.index.max().date()}')
print(f'Trades        {len(pnl):,}  ({len(pnl)/nm:.0f}/mo)')
print(f'Win rate      {(pnl>0).mean():.1%}')
print(f'Avg win       {wins.mean():.1f} pips')
print(f'Avg loss      {losses.mean():.1f} pips')
print(f'Win/Loss      {abs(wins.mean()/losses.mean()):.2f}x')
print(f'EV/trade      {pnl.mean():+.2f} pips')
print(f'Profit factor {wins.sum()/abs(losses.sum()):.2f}')
print(f'Total PnL     {pnl.sum():+,.0f} pips')
print(f'Sharpe        {sharpe:+.2f}')
print()
print(f'--- Last 18 months ({cut18.date()} -> {pnl.index.max().date()}) ---')
print(f'Trades        {len(r18):,}  ({len(r18)/18:.0f}/mo)')
print(f'Win rate      {(r18>0).mean():.1%}')
print(f'EV/trade      {r18.mean():+.2f} pips')
print(f'Total PnL     {r18.sum():+,.0f} pips')
print(f'Sharpe        {sh18:+.2f}')
print()

# Year by year
print('Year-by-year:')
print(f'  {"Year":>6} {"Trades":>7} {"WR":>7} {"EV":>8} {"Total":>9}')
print(f'  {"-"*42}')
for yr in sorted(pnl.index.year.unique()):
    yt = pnl[pnl.index.year == yr]
    marker = ' *' if yt.mean() > 5 else (' -' if yt.mean() < -2 else '')
    print(f'  {yr:>6} {len(yt):>7,} {(yt>0).mean():>7.1%} {yt.mean():>+8.2f} {yt.sum():>+9.0f}{marker}')
"""
})

# ── 6. Per-pair breakdown ─────────────────────────────────────────────────────
cells.append({
    "cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
    "source": """\
print('Per-pair breakdown (full history + last 18m):')
print(f'  {"Pair":<10} {"Trades":>7} {"WR":>7} {"EV":>8} {"Total":>9} | {"18mTrd":>6} {"18mWR":>7} {"18mEV":>8} {"18mSh":>7}')
print(f'  {"-"*78}')
for pair in sorted(trades['pair'].unique()):
    p = trades[trades['pair'] == pair]['pnl'].dropna()
    r = p[p.index >= cut18]
    sh_r = (r.mean()/r.std())*np.sqrt(252*24/HOLD_H) if len(r) > 3 and r.std() > 0 else 0
    flag = ' <<<' if r.mean() > 10 else (' <<' if r.mean() > 5 else '')
    print(f'  {pair:<10} {len(p):>7,} {(p>0).mean():>7.1%} {p.mean():>+8.2f} {p.sum():>+9.0f} | {len(r):>6,} {(r>0).mean():>7.1%} {r.mean():>+8.2f} {sh_r:>+7.2f}{flag}')
"""
})

# ── 7. Spread sensitivity ─────────────────────────────────────────────────────
cells.append({
    "cell_type": "markdown", "metadata": {},
    "source": "## 5. Spread sensitivity"
})
cells.append({
    "cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
    "source": """\
spread_scenarios = {
    'ECN tight':      {'AUDJPY':2.0,'AUDNZD':2.0,'AUDUSD':0.8,'CADJPY':2.0,'CHFJPY':2.0,
                       'EURAUD':1.5,'EURGBP':0.8,'EURJPY':1.2,'EURUSD':0.5,'NZDUSD':1.2,'USDCAD':1.0},
    'Retail (base)':  SPREAD_PIPS,
    'Wide/news':      {'AUDJPY':7.0,'AUDNZD':7.0,'AUDUSD':3.0,'CADJPY':7.0,'CHFJPY':7.0,
                       'EURAUD':6.0,'EURGBP':3.5,'EURJPY':5.0,'EURUSD':2.5,'NZDUSD':4.0,'USDCAD':4.0},
    'Worst case 3x':  {p: SPREAD_PIPS[p]*3 for p in PAIRS},
}

raw = trades['signal'] * trades['fwd_pips']
print(f'Gross EV/trade (zero spread): {raw.mean():.2f} pips')
print(f'Break-even spread: {raw.mean():.1f} pips\n')
print(f'{"Scenario":<18} {"EV":>8} {"WR":>8} {"Sharpe":>8} | {"18m EV":>8} {"18m Sh":>8}')
print('-' * 72)
for name, sp in spread_scenarios.items():
    sp_s = trades['pair'].map(sp)
    p = trades['signal'] * trades['fwd_pips'] - sp_s
    sh = (p.mean()/p.std())*np.sqrt(252*24/HOLD_H)
    r = p[p.index >= cut18]
    sh_r = (r.mean()/r.std())*np.sqrt(252*24/HOLD_H) if r.std()>0 else 0
    print(f'{name:<18} {p.mean():>+8.2f} {(p>0).mean():>8.1%} {sh:>+8.2f} | {r.mean():>+8.2f} {sh_r:>+8.2f}')
"""
})

# ── 8. Equity curves ──────────────────────────────────────────────────────────
cells.append({
    "cell_type": "markdown", "metadata": {},
    "source": "## 6. Equity curves"
})
cells.append({
    "cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
    "source": """\
fig = plt.figure(figsize=(18, 12))
fig.patch.set_facecolor('#080c14')
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)

def style_ax(ax, title):
    ax.set_facecolor('#080c14')
    ax.tick_params(colors='#aaaaaa', labelsize=8)
    ax.set_title(title, color='white', fontsize=10, pad=6)
    for sp in ax.spines.values():
        sp.set_edgecolor('#1a2332')
    ax.axhline(0, color='#333344', linewidth=0.8)

# Full history equity
ax0 = fig.add_subplot(gs[0, :2])
cum = trades.sort_index()['pnl'].cumsum()
ax0.plot(cum.index, cum.values, color='#4fc3f7', linewidth=1.2)
ax0.fill_between(cum.index, cum.values, 0, where=cum.values >= 0,
                  alpha=0.15, color='#4fc3f7')
ax0.fill_between(cum.index, cum.values, 0, where=cum.values < 0,
                  alpha=0.2, color='#ff4757')
style_ax(ax0, f'Full history equity — {len(trades):,} trades  |  Sharpe {sharpe:.2f}')
ax0.set_ylabel('Cumulative pips', color='#aaaaaa', fontsize=8)

# Per-pair equity (last 18m)
ax1 = fig.add_subplot(gs[0, 2])
colors = plt.cm.tab20(np.linspace(0, 1, len(PAIRS)))
for i, pair in enumerate(sorted(trades['pair'].unique())):
    pp = trades[trades['pair'] == pair].sort_index()
    pp18 = pp[pp.index >= cut18]['pnl'].cumsum()
    if len(pp18) > 0:
        ax1.plot(pp18.index, pp18.values, linewidth=0.9, alpha=0.8,
                 label=pair, color=colors[i])
style_ax(ax1, 'Per-pair equity (last 18m)')
ax1.legend(fontsize=6, facecolor='#0d1421', labelcolor='white',
           ncol=2, loc='upper left')

# Last 18m equity
ax2 = fig.add_subplot(gs[1, :2])
cum18 = trades[trades.index >= cut18].sort_index()['pnl'].cumsum()
ax2.plot(cum18.index, cum18.values, color='#2ecc71', linewidth=1.4)
ax2.fill_between(cum18.index, cum18.values, 0, where=cum18.values >= 0,
                  alpha=0.15, color='#2ecc71')
ax2.fill_between(cum18.index, cum18.values, 0, where=cum18.values < 0,
                  alpha=0.2, color='#ff4757')
style_ax(ax2, f'Last 18 months — {len(cum18):,} trades  |  Sharpe {sh18:.2f}')
ax2.set_ylabel('Cumulative pips', color='#aaaaaa', fontsize=8)

# Annual EV bar chart
ax3 = fig.add_subplot(gs[1, 2])
years = sorted(pnl.index.year.unique())
ev_by_year = [pnl[pnl.index.year == y].mean() for y in years]
bar_colors = ['#2ecc71' if v > 0 else '#ff4757' for v in ev_by_year]
ax3.bar(years, ev_by_year, color=bar_colors, alpha=0.85, width=0.7)
ax3.set_xticks(years)
ax3.set_xticklabels([str(y)[2:] for y in years], rotation=45, fontsize=7)
style_ax(ax3, 'EV/trade by year (pips)')
ax3.set_ylabel('pips', color='#aaaaaa', fontsize=8)

plt.suptitle('System 2: Volatility-Driven Mean Reversion', color='white',
             fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
plt.show()
"""
})

# ── 9. Condition contribution ─────────────────────────────────────────────────
cells.append({
    "cell_type": "markdown", "metadata": {},
    "source": "## 7. Condition contribution (ablation)\n\nRemove one condition at a time and measure the impact."
})
cells.append({
    "cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
    "source": """\
def run_variant(name, lmask, smask):
    sig = pd.Series(0, index=data.index)
    sig[lmask] = 1; sig[smask] = -1
    p = sig * data['fwd_pips'] - data['spread'] * sig.abs()
    t = p[sig != 0].dropna()
    if len(t) < 10:
        print(f'{name:<40} insufficient trades')
        return
    nm_ = (t.index.max() - t.index.min()).days / 30
    sh_ = (t.mean()/t.std())*np.sqrt(252*24/HOLD_H)
    r_ = t[t.index >= cut18]
    sh_r_ = (r_.mean()/r_.std())*np.sqrt(252*24/HOLD_H) if len(r_)>5 and r_.std()>0 else 0
    delta_ev = t.mean() - pnl.mean()
    print(f'{name:<40} {len(t):>7,} ({len(t)/nm_:>3.0f}/mo)  WR:{(t>0).mean():>5.1%}  EV:{t.mean():>+6.2f}  Sh:{sh_:>+5.2f}  dEV:{delta_ev:>+6.2f} | 18m:{r_.mean():>+6.2f} Sh:{sh_r_:>+5.2f}')

L = data['kyle_lambda_delta_3h_r'] > 0.70
R = data['residual_12h_r'] < 0.25
V = data['rv_zscore_24_r'] > 0.85
S = data['realized_skew_r'] < 0.30
VR = data['vr_5_r'] < 0.30

Ls = data['kyle_lambda_delta_3h_r'] < 0.30
Rs = data['residual_12h_r'] > 0.75
Ss = data['realized_skew_r'] > 0.70

print(f'{"Variant":<40} {"Trades":>7}         {"WR":>5}   {"EV":>6}   {"Sh":>5}  {"dEV":>6} | {"18m EV":>6} {"Sh":>5}')
print('-' * 100)
run_variant('Full system (baseline)',       L & R & V & S & VR,      Ls & Rs & V & Ss & VR)
print()
run_variant('Remove kyle_lambda_delta_3h', R & V & S & VR,           Rs & V & Ss & VR)
run_variant('Remove residual_12h',         L & V & S & VR,           Ls & V & Ss & VR)
run_variant('Remove rv_zscore_24',         L & R & S & VR,           Ls & Rs & Ss & VR)
run_variant('Remove realized_skew',        L & R & V & VR,           Ls & Rs & V & VR)
run_variant('Remove vr_5 (regime)',        L & R & V & S,            Ls & Rs & V & Ss)
print()
run_variant('Only vr_5 + residual',        VR & R,                   VR & Rs)
run_variant('Only vr_5 + rv_z',            VR & V,                   VR & V)
run_variant('Only kyle + residual + rv_z', L & R & V,                Ls & Rs & V)
"""
})

# ── Build notebook ─────────────────────────────────────────────────────────────
nb = {
    "nbformat": 4, "nbformat_minor": 4,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.13.0"}
    },
    "cells": cells
}

out = 'notebooks_10/03_system2_vol_mean_reversion.ipynb'
with open(out, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print(f'Written {len(cells)} cells -> {out}')
