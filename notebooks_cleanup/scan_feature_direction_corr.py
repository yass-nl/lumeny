import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path

DATA_DIR  = Path('backend/data/features_combined')
PRICE_DIR = Path('backend/data/processed')

MAJORS = ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'USDCAD', 'AUDUSD', 'NZDUSD']
TRAIN_END = '2024-06-30'

COLS_DROP_SUFFIX = '_f9'
COLS_DROP_SUBSTR = '_1W'
LOOKAHEAD_COLS   = {
    'mfe_long_pips', 'mfe_short_pips',
    'trail_long_bars', 'trail_short_bars',
    'trail_stop_pips', 'mfe_atr_24',
}

# ── Pass 1: collect union of feature columns ──────────────────────────────────
print('Pass 1: collecting feature columns...')
feature_cols = None
for pair in MAJORS:
    df = pd.read_parquet(DATA_DIR / f'{pair}_combined.parquet', columns=None)
    drop = {c for c in df.columns if (
        c.endswith(COLS_DROP_SUFFIX) or COLS_DROP_SUBSTR in c or
        c in LOOKAHEAD_COLS or c == 'pair'
    )}
    cols = [c for c in df.columns if c not in drop]
    if feature_cols is None:
        feature_cols = cols
    else:
        # keep only columns present in ALL pairs (intersection)
        fc_set = set(feature_cols)
        feature_cols = [c for c in feature_cols if c in set(cols)]
    del df

F = len(feature_cols)
print(f'Common feature columns: {F}')

# ── Pass 2: accumulate statistics one pair at a time ─────────────────────────
n_total  = 0
sx       = np.zeros(F, dtype=np.float64)
sx2      = np.zeros(F, dtype=np.float64)
sxy      = np.zeros(F, dtype=np.float64)
sy       = 0.0
sy2      = 0.0
sum_up   = np.zeros(F, dtype=np.float64)
sum_dn   = np.zeros(F, dtype=np.float64)
n_up     = 0
n_dn     = 0
hour_acc = {h: {'n': 0, 'n_up': 0, 'sum_ret': 0.0} for h in range(24)}

for pair in MAJORS:
    print(f'  {pair}...', flush=True)

    df = pd.read_parquet(DATA_DIR / f'{pair}_combined.parquet')
    drop = [c for c in df.columns if (
        c.endswith(COLS_DROP_SUFFIX) or COLS_DROP_SUBSTR in c or
        c in LOOKAHEAD_COLS or c == 'pair'
    )]
    df.drop(columns=drop, errors='ignore', inplace=True)
    df = df[df.index <= TRAIN_END]

    close   = pd.read_parquet(PRICE_DIR / f'{pair}_1H.parquet')['close']
    fwd_ret = np.log(close.shift(-4) / close).reindex(df.index)
    fwd_dir = np.sign(fwd_ret)

    valid = fwd_dir.notna() & (fwd_dir != 0)
    df    = df[valid][feature_cols].ffill().fillna(0)
    y     = fwd_dir[valid].values.astype(np.float64)
    ret   = fwd_ret[valid].values.astype(np.float64)
    hours = df.index.hour
    X     = df.values.astype(np.float64)
    del df

    n = len(y)
    n_total += n

    # hour stats
    for h in range(24):
        mask = hours == h
        if mask.sum() == 0:
            continue
        hour_acc[h]['n']       += int(mask.sum())
        hour_acc[h]['n_up']    += int((y[mask] == 1).sum())
        hour_acc[h]['sum_ret'] += float(ret[mask].sum())

    # correlation accumulators
    sy  += y.sum()
    sy2 += (y ** 2).sum()
    sx  += X.sum(axis=0)
    sx2 += (X ** 2).sum(axis=0)
    sxy += (X * y[:, None]).sum(axis=0)

    up_mask = y == 1
    dn_mask = y == -1
    sum_up += X[up_mask].sum(axis=0)
    sum_dn += X[dn_mask].sum(axis=0)
    n_up   += int(up_mask.sum())
    n_dn   += int(dn_mask.sum())

    del X, y, ret

# ── Compute Pearson correlations ──────────────────────────────────────────────
print(f'\nComputing correlations (total rows: {n_total:,})...')
n = n_total
num = n * sxy - sx * sy
den = np.sqrt(np.maximum((n * sx2 - sx**2) * (n * sy2 - sy**2), 0.0))
pearson = np.where(den > 1e-10, num / den, 0.0)

mean_up = np.where(n_up > 0, sum_up / n_up, 0.0)
mean_dn = np.where(n_dn > 0, sum_dn / n_dn, 0.0)
diff    = mean_up - mean_dn

order = np.argsort(np.abs(pearson))[::-1]

print(f'\nTop 40 features by |correlation| with 4H forward direction:')
print(f'{"Feature":<40} {"Pearson":>8} {"Mean(up)":>12} {"Mean(dn)":>12} {"Diff":>12}')
print('=' * 90)
for i in order[:40]:
    print(f'{feature_cols[i]:<40} {pearson[i]:>+8.4f}   {mean_up[i]:>12.4f}   {mean_dn[i]:>12.4f}   {diff[i]:>+12.4f}')

print(f'\nBottom 10 (near-zero correlation):')
for i in order[-10:]:
    print(f'{feature_cols[i]:<40} {pearson[i]:>+8.4f}')

# ── Hour of day bias ──────────────────────────────────────────────────────────
print(f'\nHOUR OF DAY direction bias:')
print(f'  {"Hour":>5} {"N":>7} {"Up%":>7} {"AvgRet":>10}')
for h in range(24):
    ha = hour_acc[h]
    if ha['n'] == 0:
        continue
    up_pct  = ha['n_up'] / ha['n']
    avg_ret = ha['sum_ret'] / ha['n']
    flag = ' *' if abs(up_pct - 0.5) > 0.02 else ''
    print(f'  {h:>5}H  {ha["n"]:>7,} {up_pct:>7.1%} {avg_ret:>+10.6f}{flag}')
