"""
Scan: does 15M slope / momentum predict 1H and 4H forward price direction?
Fully vectorized — no rolling.apply().
"""
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path

PRICE_DIR = Path('backend/data/processed')
MAJORS    = ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'USDCAD', 'AUDUSD', 'NZDUSD']
TRAIN_END = '2024-06-30'

def fast_slope(c, n, atr):
    """
    Vectorized OLS slope over rolling window of n bars.
    Uses the closed-form formula for slope of (0..n-1, y).
    xi = [0,1,...,n-1], mean_x = (n-1)/2
    slope = sum((xi - mean_x) * (yi - mean_y)) / sum((xi - mean_x)^2)
    """
    xi   = np.arange(n, dtype=np.float64)
    mx   = xi.mean()
    dxi  = xi - mx                          # shape (n,)
    denom = (dxi ** 2).sum()                # scalar

    # rolling dot product of dxi with the window of c
    # = sum_i dxi[i] * c[t-n+1+i]  for each t
    # implemented as weighted rolling sum
    num = c.rolling(n).apply(lambda x: np.dot(x, dxi), raw=True)
    slope = num / denom
    return slope / atr.clip(lower=1e-10)

all_rows = []

for pair in MAJORS:
    print(f'  {pair}...', flush=True)

    df_1h  = pd.read_parquet(PRICE_DIR / f'{pair}_1H.parquet')
    df_15m = pd.read_parquet(PRICE_DIR / f'{pair}_15m.parquet')

    df_1h  = df_1h[df_1h.index  <= TRAIN_END]
    df_15m = df_15m[df_15m.index <= TRAIN_END]

    c15 = df_15m['close']
    h15 = df_15m['high']
    l15 = df_15m['low']
    v15 = df_15m['volume']

    tr15  = np.maximum(h15 - l15,
            np.maximum((h15 - c15.shift(1)).abs(),
                       (l15 - c15.shift(1)).abs()))
    atr15 = tr15.rolling(96, min_periods=16).mean()

    feat = pd.DataFrame(index=df_15m.index)

    # Momentum: (close - close_n_bars_ago) / ATR  — fully vectorized
    for n, label in [(2,'30m'), (4,'1h'), (8,'2h'), (12,'3h'), (16,'4h')]:
        feat[f'mom_{label}'] = (c15 - c15.shift(n)) / atr15.clip(lower=1e-10)

    # Slope via fast_slope (still uses apply but with raw=True which is faster)
    for n, label in [(4,'1h'), (8,'2h'), (16,'4h')]:
        feat[f'slope_{label}'] = fast_slope(c15, n, atr15)

    # Acceleration: momentum change
    feat['accel_1h'] = feat['mom_1h'] - feat['mom_1h'].shift(4)
    feat['accel_2h'] = feat['mom_2h'] - feat['mom_2h'].shift(8)

    # Volume surge
    feat['vol_surge'] = v15.rolling(4).mean() / v15.rolling(96).mean().clip(lower=1e-10) - 1

    # Resample to 1H: last 15M bar of each hour
    feat_1h = feat.resample('1h').last().reindex(df_1h.index)

    c1h    = df_1h['close']
    fwd_1h = np.log(c1h.shift(-1) / c1h)
    fwd_4h = np.log(c1h.shift(-4) / c1h)

    combined = feat_1h.copy()
    combined['fwd_1h'] = fwd_1h
    combined['fwd_4h'] = fwd_4h
    combined['pair']   = pair
    combined = combined.dropna(subset=['fwd_1h', 'fwd_4h'])
    all_rows.append(combined)

df = pd.concat(all_rows).reset_index(drop=True)
feat_cols = [c for c in df.columns if c not in ('pair', 'fwd_1h', 'fwd_4h')]

print(f'\nTotal bars: {len(df):,}  |  Features: {len(feat_cols)}')

# ── Pearson correlations ──────────────────────────────────────────────────────
print(f'\n{"Feature":<18} {"Corr_1H":>10} {"Corr_4H":>10} {"|max|":>8}')
print('=' * 52)

results = []
for col in feat_cols:
    valid = df[[col, 'fwd_1h', 'fwd_4h']].dropna()
    if len(valid) < 500: continue
    x  = valid[col].values
    p1 = np.corrcoef(x, valid['fwd_1h'].values)[0, 1]
    p4 = np.corrcoef(x, valid['fwd_4h'].values)[0, 1]
    results.append({'feat': col, 'p1h': p1, 'p4h': p4, 'abs_max': max(abs(p1), abs(p4))})

results = sorted(results, key=lambda r: r['abs_max'], reverse=True)
for r in results:
    flag = ' ***' if r['abs_max'] > 0.02 else (' *' if r['abs_max'] > 0.01 else '')
    print(f'{r["feat"]:<18} {r["p1h"]:>+10.4f} {r["p4h"]:>+10.4f} {r["abs_max"]:>8.4f}{flag}')

# ── Decile analysis for top 4 features ───────────────────────────────────────
print(f'\nDECILE ANALYSIS (vs fwd_4h):')
for r in results[:4]:
    col   = r['feat']
    valid = df[[col, 'fwd_4h']].dropna().copy()
    valid['decile'] = pd.qcut(valid[col], 10, labels=False, duplicates='drop')
    d_stats = valid.groupby('decile')['fwd_4h'].agg(['mean', lambda x: (x > 0).mean(), 'count'])
    d_stats.columns = ['avg_ret', 'up_pct', 'n']
    print(f'\n  {col}  (corr_4H={r["p4h"]:+.4f}):')
    print(f'  {"D":<4} {"N":>6} {"AvgRet":>10} {"Up%":>8}')
    for d, row in d_stats.iterrows():
        print(f'  {int(d):<4} {int(row["n"]):>6,} {row["avg_ret"]:>+10.6f} {row["up_pct"]:>8.1%}')

# ── Direction consistency ─────────────────────────────────────────────────────
print(f'\nDIRECTION CHECK — does positive feature predict positive 4H return?')
print(f'{"Feature":<18} {"Pos->up%":>10} {"Neg->dn%":>10} {"Edge":>8}')
print('-' * 50)
for r in results[:8]:
    col   = r['feat']
    valid = df[[col, 'fwd_4h']].dropna()
    pos   = valid[valid[col] > 0]['fwd_4h']
    neg   = valid[valid[col] < 0]['fwd_4h']
    if len(pos) < 100 or len(neg) < 100: continue
    pos_up = (pos > 0).mean()
    neg_dn = (neg < 0).mean()
    edge   = (pos_up + neg_dn) / 2 - 0.5
    print(f'{col:<18} {pos_up:>10.1%} {neg_dn:>10.1%} {edge:>+8.3f}')
