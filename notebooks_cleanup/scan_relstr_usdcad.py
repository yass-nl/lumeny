"""
Scan: cross-pair relative strength divergence as directional signal.

Idea: pairs sharing USD as common currency move together. When one pair
diverges from its peers, it tends to catch up (mean reversion) or lead
(momentum). We test both directions.

Signals tested:
1. Relative strength vs peers (divergence = mean revert or follow?)
2. Lead-lag between correlated pairs
3. USD index proxy divergence
"""
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path

PRICE_DIR = Path('backend/data/processed')
TRAIN_END = '2024-06-30'
MAJORS    = ['EURUSD', 'GBPUSD', 'AUDUSD', 'NZDUSD', 'USDJPY', 'USDCHF', 'USDCAD']

# USD-positive (USD weak = price rises): EURUSD, GBPUSD, AUDUSD, NZDUSD
# USD-negative (USD weak = price falls): USDJPY, USDCHF, USDCAD
USD_POS = ['EURUSD', 'GBPUSD', 'AUDUSD', 'NZDUSD']
USD_NEG = ['USDJPY', 'USDCHF', 'USDCAD']

# ── Load all 1H closes ────────────────────────────────────────────────────────
print('Loading data...')
closes = {}
for pair in MAJORS:
    df = pd.read_parquet(PRICE_DIR / f'{pair}_1H.parquet')
    df = df[df.index <= TRAIN_END]
    closes[pair] = df['close']

idx = closes[MAJORS[0]].index
for pair in MAJORS:
    idx = idx.intersection(closes[pair].index)
print(f'Common bars: {len(idx):,}')

rets = {}
for pair in MAJORS:
    c = closes[pair].reindex(idx)
    rets[pair] = np.log(c / c.shift(1))
df_rets = pd.DataFrame(rets).dropna()

# Normalize direction: all series positive = USD weak
df_usd = pd.DataFrame(index=df_rets.index)
for pair in USD_POS:
    df_usd[pair] = df_rets[pair]
for pair in USD_NEG:
    df_usd[pair] = -df_rets[pair]

# ── Signal 1: Relative strength divergence ───────────────────────────────────
print('\n--- SIGNAL 1: Pair vs peers divergence over N bars ---')
print('Corr < 0 = mean reversion  |  Corr > 0 = momentum')
print()
print(f'{"Pair":<10} {"Win":>4} {"Corr_4H":>9} {"Pos->up%":>10} {"Neg->dn%":>10} {"Edge":>8}')
print('-' * 56)

results = []
for window in [4, 8, 12, 24]:
    for pair in MAJORS:
        if pair in USD_POS:
            peers = [p for p in USD_POS if p != pair]
            sign  = 1
        else:
            peers = [p for p in USD_NEG if p != pair]
            sign  = -1

        if len(peers) < 2:
            continue

        cum_pair  = df_rets[pair].rolling(window).sum()
        cum_peers = pd.concat([df_usd[p] for p in peers], axis=1).rolling(window).sum().mean(axis=1)
        cum_peers_adj = cum_peers * sign
        divergence = cum_pair - cum_peers_adj

        fwd_4h = df_rets[pair].rolling(4).sum().shift(-4)
        valid  = pd.DataFrame({'div': divergence, 'fwd': fwd_4h}).dropna()
        if len(valid) < 500: continue

        corr   = np.corrcoef(valid['div'].values, valid['fwd'].values)[0, 1]
        pos    = valid[valid['div'] > 0]['fwd']
        neg    = valid[valid['div'] < 0]['fwd']
        pos_up = (pos > 0).mean() if len(pos) > 50 else np.nan
        neg_dn = (neg < 0).mean() if len(neg) > 50 else np.nan
        edge   = (pos_up + neg_dn) / 2 - 0.5 if not np.isnan(pos_up) else np.nan

        results.append({'pair': pair, 'window': window, 'corr': corr,
                        'pos_up': pos_up, 'neg_dn': neg_dn, 'edge': edge, 'n': len(valid)})

for r in sorted(results, key=lambda x: abs(x['corr']), reverse=True)[:20]:
    flag = ' ***' if abs(r['corr']) > 0.03 else (' *' if abs(r['corr']) > 0.015 else '')
    print(f'{r["pair"]:<10} {r["window"]:>4} {r["corr"]:>+9.4f} {r["pos_up"]:>10.1%} {r["neg_dn"]:>10.1%} {r["edge"]:>+8.3f}{flag}')

# ── Signal 2: Lead-lag between correlated pairs ───────────────────────────────
print('\n--- SIGNAL 2: Lead-lag (does pair A predict pair B N bars later?) ---')
print()
print(f'{"Leader":<10} {"Follower":<10} {"Lag":>4} {"Corr":>9} {"Pos->up%":>10} {"Neg->dn%":>10} {"Edge":>8}')
print('-' * 60)

LEAD_PAIRS = [
    ('EURUSD', 'GBPUSD'), ('GBPUSD', 'EURUSD'),
    ('EURUSD', 'AUDUSD'), ('AUDUSD', 'EURUSD'),
    ('AUDUSD', 'NZDUSD'), ('NZDUSD', 'AUDUSD'),
    ('USDJPY', 'USDCHF'), ('USDCHF', 'USDJPY'),
    ('EURUSD', 'USDCHF'), ('USDJPY', 'EURUSD'),
]

for leader, follower in LEAD_PAIRS:
    for lag in [1, 2, 4]:
        x     = df_rets[leader]
        fwd   = df_rets[follower].shift(-lag)
        valid = pd.DataFrame({'x': x, 'fwd': fwd}).dropna()
        if len(valid) < 500: continue
        corr  = np.corrcoef(valid['x'].values, valid['fwd'].values)[0, 1]
        pos   = valid[valid['x'] > 0]['fwd']
        neg   = valid[valid['x'] < 0]['fwd']
        pos_up = (pos > 0).mean() if len(pos) > 50 else np.nan
        neg_dn = (neg < 0).mean() if len(neg) > 50 else np.nan
        edge   = (pos_up + neg_dn) / 2 - 0.5
        flag   = ' ***' if abs(corr) > 0.02 else ''
        print(f'{leader:<10} {follower:<10} {lag:>4} {corr:>+9.4f} {pos_up:>10.1%} {neg_dn:>10.1%} {edge:>+8.3f}{flag}')
    print()

# ── Signal 3: USD index proxy divergence ─────────────────────────────────────
print('\n--- SIGNAL 3: Individual pair vs USD index proxy ---')
print()
print(f'{"Pair":<10} {"Win":>4} {"Corr_4H":>9} {"Pos->up%":>10} {"Neg->dn%":>10} {"Edge":>8}')
print('-' * 56)

usd_index = df_usd.mean(axis=1)

idx_results = []
for pair in MAJORS:
    sign = 1 if pair in USD_POS else -1
    for window in [4, 8, 12, 24]:
        cum_pair  = df_rets[pair].rolling(window).sum()
        cum_index = (usd_index * sign).rolling(window).sum()
        divergence = cum_pair - cum_index

        fwd_4h = df_rets[pair].rolling(4).sum().shift(-4)
        valid  = pd.DataFrame({'div': divergence, 'fwd': fwd_4h}).dropna()
        if len(valid) < 500: continue

        corr   = np.corrcoef(valid['div'].values, valid['fwd'].values)[0, 1]
        pos    = valid[valid['div'] > 0]['fwd']
        neg    = valid[valid['div'] < 0]['fwd']
        pos_up = (pos > 0).mean() if len(pos) > 50 else np.nan
        neg_dn = (neg < 0).mean() if len(neg) > 50 else np.nan
        edge   = (pos_up + neg_dn) / 2 - 0.5 if not np.isnan(pos_up) else np.nan
        idx_results.append({'pair': pair, 'window': window, 'corr': corr,
                             'pos_up': pos_up, 'neg_dn': neg_dn, 'edge': edge})

for r in sorted(idx_results, key=lambda x: abs(x['corr']), reverse=True)[:20]:
    flag = ' ***' if abs(r['corr']) > 0.03 else (' *' if abs(r['corr']) > 0.015 else '')
    print(f'{r["pair"]:<10} {r["window"]:>4} {r["corr"]:>+9.4f} {r["pos_up"]:>10.1%} {r["neg_dn"]:>10.1%} {r["edge"]:>+8.3f}{flag}')
