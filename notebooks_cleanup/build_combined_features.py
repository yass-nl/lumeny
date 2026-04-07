import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

MAJORS = ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'USDCAD', 'AUDUSD', 'NZDUSD']

F6_DIR  = Path('backend/data/features_6')
F9_DIR  = Path('backend/data/features_9')
FG_DIR  = Path('backend/data/features')
OUT_DIR = Path('backend/data/features_combined')
OUT_DIR.mkdir(parents=True, exist_ok=True)

combined_pairs = []

for pair in MAJORS:
    print(f'{pair}...', flush=True)

    f6 = pd.read_parquet(F6_DIR / f'{pair}_features.parquet')
    f9 = pd.read_parquet(F9_DIR / f'{pair}_features.parquet')
    fg = pd.read_parquet(FG_DIR / f'{pair}_features.parquet')

    # drop pair col from any that have it
    for df in [f6, f9, fg]:
        if 'pair' in df.columns:
            df.drop(columns=['pair'], inplace=True)

    # join on index — f6 as base, avoid duplicate col names
    merged = f6.join(f9, how='outer', rsuffix='_f9')
    merged = merged.join(fg, how='outer', rsuffix='_fg')
    merged['pair'] = pair

    print(f'  f6={f6.shape[1]} + f9={f9.shape[1]} + fg={fg.shape[1]} -> {merged.shape[1]} cols, {len(merged):,} rows')

    out_path = OUT_DIR / f'{pair}_combined.parquet'
    merged.to_parquet(out_path)
    print(f'  saved -> {out_path}')

    combined_pairs.append(merged)

print('\nConcatenating all pairs...')
all_pairs = pd.concat(combined_pairs).sort_index()
print(f'Shape: {all_pairs.shape}')
print(f'Date range: {all_pairs.index.min()} to {all_pairs.index.max()}')

nan_pct = all_pairs.isnull().mean()
print(f'\nNaN summary:')
print(f'  Cols with 0% NaN:   {(nan_pct == 0).sum()}')
print(f'  Cols with <5% NaN:  {(nan_pct < 0.05).sum()}')
print(f'  Cols with >20% NaN: {(nan_pct > 0.20).sum()}')

out_all = OUT_DIR / 'all_pairs.parquet'
all_pairs.to_parquet(out_all)
print(f'\nSaved -> {out_all}  ({out_all.stat().st_size/1024/1024:.1f} MB)')
