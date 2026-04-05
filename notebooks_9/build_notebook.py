import json
from pathlib import Path

nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11.0"}
    },
    "cells": []
}

def md(src): return {"cell_type": "markdown", "metadata": {}, "source": src, "id": "md"}
def code(src): return {"cell_type": "code", "metadata": {}, "source": src, "outputs": [], "execution_count": None, "id": "code"}

cells = [

md("# LumenY 9 — Feature Engineering\n\nLoads **features_8** (geometric), computes **MA context features** from raw OHLCV, appends them, saves to `backend/data/features_10/`.\n\nMA features capture price position relative to MA50/MA200 — regime-agnostic structural context."),

code("""\
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path

PROCESSED_DIR  = Path('../backend/data/processed')
FEATURES_8_DIR = Path('../backend/data/features_8')
OUTPUT_DIR     = Path('../backend/data/features_10')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PAIRS = [
    'EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD', 'USDCAD', 'NZDUSD',
    'EURJPY', 'GBPJPY', 'EURGBP', 'EURAUD', 'AUDJPY', 'CADJPY', 'CHFJPY', 'AUDNZD',
]
PIP_SIZE = {
    'EURUSD':0.0001,'GBPUSD':0.0001,'USDCHF':0.0001,'AUDUSD':0.0001,
    'USDCAD':0.0001,'NZDUSD':0.0001,'EURGBP':0.0001,'EURAUD':0.0001,'AUDNZD':0.0001,
    'USDJPY':0.01,'EURJPY':0.01,'GBPJPY':0.01,'AUDJPY':0.01,'CADJPY':0.01,'CHFJPY':0.01,
}

print('Ready.')
print(f'Output: {OUTPUT_DIR.resolve()}')
"""),

md("## 1. Compute MA Context Features"),

code("""\
def compute_ma_features(ohlc, pip):
    c   = ohlc['close']
    h   = ohlc['high']
    l   = ohlc['low']

    ma50  = c.rolling(50).mean()
    ma200 = c.rolling(200).mean()

    # ATR for normalization
    tr   = pd.concat([h - l,
                      (h - c.shift(1)).abs(),
                      (l - c.shift(1)).abs()], axis=1).max(axis=1)
    atr24 = tr.rolling(24, min_periods=6).mean()

    feat = pd.DataFrame(index=ohlc.index)

    # ── Distance features ──
    feat['dist_ma50_pips']      = (c - ma50)  / pip          # signed: neg=below, pos=above
    feat['dist_ma200_pips']     = (c - ma200) / pip          # signed
    feat['dist_ma50_atr']       = (c - ma50)  / atr24.clip(lower=1e-8)   # ATR-normalized signed
    feat['dist_ma200_atr']      = (c - ma200) / atr24.clip(lower=1e-8)

    # ── Gap between MAs ──
    feat['ma50_ma200_gap_pips'] = (ma50 - ma200) / pip       # pos = MA50 above MA200
    feat['ma50_ma200_gap_atr']  = (ma50 - ma200) / atr24.clip(lower=1e-8)

    # ── Price position within MA channel ──
    # 0 = at MA50, 1 = at MA200 (extrapolates outside channel)
    channel = (ma200 - ma50).replace(0, np.nan)
    feat['price_in_channel']    = (c - ma50) / channel

    # ── MA slopes ──
    feat['ma50_slope_3h']       = ma50.diff(3)  / pip
    feat['ma50_slope_8h']       = ma50.diff(8)  / pip
    feat['ma50_slope_24h']      = ma50.diff(24) / pip
    feat['ma200_slope_8h']      = ma200.diff(8) / pip
    feat['ma200_slope_24h']     = ma200.diff(24) / pip

    # ── MA acceleration (slope of slope) ──
    feat['ma50_accel_3h']       = feat['ma50_slope_3h'].diff(3)
    feat['ma50_accel_8h']       = feat['ma50_slope_8h'].diff(8)

    # ── Position flags ──
    feat['above_ma50']          = (c > ma50).astype(float)
    feat['above_ma200']         = (c > ma200).astype(float)
    feat['both_above']          = ((c > ma50) & (c > ma200)).astype(float)
    feat['both_below']          = ((c < ma50) & (c < ma200)).astype(float)
    feat['between_mas']         = (
        ((c > ma50) & (c < ma200)) | ((c < ma50) & (c > ma200))
    ).astype(float)

    # ── Consecutive bars below/above MA50 ──
    below = (c < ma50).astype(float).values
    above = (c > ma50).astype(float).values
    consec_below = np.zeros(len(c))
    consec_above = np.zeros(len(c))
    for i in range(1, len(c)):
        consec_below[i] = consec_below[i-1] + 1 if below[i] else 0
        consec_above[i] = consec_above[i-1] + 1 if above[i] else 0
    feat['bars_below_ma50']     = consec_below
    feat['bars_above_ma50']     = consec_above

    # ── Min distance to MA50 in last 6h / 12h / 24h (how close did it get recently) ──
    abs_dist_ma50 = (c - ma50).abs() / pip
    feat['min_dist_ma50_6h']    = abs_dist_ma50.rolling(6).min()
    feat['min_dist_ma50_12h']   = abs_dist_ma50.rolling(12).min()
    feat['min_dist_ma50_24h']   = abs_dist_ma50.rolling(24).min()

    # ── Rate of change of distance (is price approaching or leaving MA50?) ──
    feat['dist_ma50_roc_3h']    = (c - ma50).diff(3)  / pip   # neg = approaching from below
    feat['dist_ma50_roc_6h']    = (c - ma50).diff(6)  / pip
    feat['dist_ma50_roc_12h']   = (c - ma50).diff(12) / pip

    # ── MA50 / MA200 touch count (bounces in last 24h / 48h) ──
    # A touch = abs distance < 0.5 ATR
    near_ma50  = (abs_dist_ma50 < atr24 / pip * 0.5).astype(float)
    feat['ma50_touches_24h']    = near_ma50.rolling(24).sum()
    feat['ma50_touches_48h']    = near_ma50.rolling(48).sum()

    abs_dist_ma200 = (c - ma200).abs() / pip
    near_ma200 = (abs_dist_ma200 < atr24 / pip * 0.5).astype(float)
    feat['ma200_touches_24h']   = near_ma200.rolling(24).sum()
    feat['ma200_touches_48h']   = near_ma200.rolling(48).sum()

    return feat

print('compute_ma_features() defined.')
"""),

md("## 2. Process All Pairs"),

code("""\
for pair in PAIRS:
    pip = PIP_SIZE[pair]

    # Load features_8
    f8 = pd.read_parquet(FEATURES_8_DIR / f'{pair}_geometric.parquet')
    # drop label leak and duplicate pair col
    f8 = f8.drop(columns=[c for c in f8.columns if c.startswith('label_') or c == 'pair'], errors='ignore')

    # Load OHLCV
    ohlc = pd.read_parquet(PROCESSED_DIR / f'{pair}_1H.parquet')[['open','high','low','close','volume']]
    ohlc.index = pd.to_datetime(ohlc.index)

    # Compute MA features
    ma_feat = compute_ma_features(ohlc, pip)

    # Join: inner on timestamp (features_8 may have slightly different range)
    combined = f8.join(ma_feat, how='inner')
    combined['pair'] = pair

    # Save
    out_path = OUTPUT_DIR / f'{pair}_features.parquet'
    combined.to_parquet(out_path)
    print(f'{pair}: {combined.shape}  -> {out_path.name}')

print(f'\\nDone. All pairs saved to {OUTPUT_DIR.resolve()}')
"""),

md("## 3. Sanity Check"),

code("""\
df = pd.read_parquet(OUTPUT_DIR / 'EURUSD_features.parquet')
print(f'EURUSD shape: {df.shape}')
print(f'Date range: {df.index.min().date()} -> {df.index.max().date()}')

geom_cols = [c for c in df.columns if c not in ['pair'] and not c.startswith(('dist_','ma','above_','both_','between_','bars_','min_','price_in'))]
ma_cols   = [c for c in df.columns if c.startswith(('dist_','ma','above_','both_','between_','bars_','min_','price_in'))]
print(f'\\nGeometric features:  {len(geom_cols)}')
print(f'MA context features: {len(ma_cols)}')
print(f'Total features:      {len(df.columns)-1}  (+pair)')

print(f'\\nMA features:')
for c in sorted(ma_cols):
    print(f'  {c}')

print(f'\\nSample stats (EURUSD, last 1000 rows):')
print(df[ma_cols].tail(1000).describe().round(3))
"""),

]

nb['cells'] = cells

out = Path('notebooks_9/01_features.ipynb')
with open(out, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print(f'Written: {out}')
