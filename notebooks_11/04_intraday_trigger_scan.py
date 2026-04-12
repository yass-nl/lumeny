"""
Intraday Entry Trigger Scan
============================
Within each 72h directional window (MFE>=50, direction system, 7-20 UTC),
we have up to 72 individual 1h bars. At each bar, we know the regime direction.

Question: which features identify the best 1h bars to enter a SHORT-TERM trade
(4h, 6h, 8h forward) in the regime direction?

Method:
  - For each signal, take all 1h bars WITHIN the 72h window (hours 7-20 UTC only)
  - At each bar, compute: did price move in regime direction over next 4h / 6h / 8h?
  - Correlate every universal feature against that short-term outcome
  - Find features that cleanly split good entry bars from bad ones
  - Show threshold buckets: accuracy and avg pip at each level

This gives us the entry trigger for intraday trades within the regime window.
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from scipy import stats

SCRIPT_DIR     = Path(__file__).parent
FEATURES_DIR   = SCRIPT_DIR / '../backend/data/features_9'
GEOM_DIR       = SCRIPT_DIR / '../backend/data/features_8'
PROCESSED_DIR  = SCRIPT_DIR / '../backend/data/processed'
MFE_MODEL_PATH = SCRIPT_DIR / '../backend/models_9/mfe_q50/model_1H_Q50.joblib'

START_DATE     = '2024-10-11'
MFE_THRESH     = 50.0
HOURS_ALLOWED  = set(range(7, 21))
COOLDOWN_H     = 72
WINDOW_H       = 72
HORIZONS       = [4, 6, 8]   # short-term forward returns to test
TOP_N          = 25
MIN_N          = 200          # min bars to include a feature

JPY_PAIRS = {'USDJPY', 'EURJPY', 'GBPJPY', 'AUDJPY', 'CADJPY', 'CHFJPY'}
PAIRS_ALL = [
    'EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD', 'USDCAD', 'NZDUSD',
    'EURJPY', 'GBPJPY', 'EURGBP', 'EURAUD', 'AUDJPY', 'CADJPY', 'CHFJPY', 'AUDNZD',
]

# ── Load MFE model ────────────────────────────────────────────────────────────
print('Loading MFE model...')
bundle       = joblib.load(MFE_MODEL_PATH)
mfe_model    = bundle['model']
feature_cols = bundle['feature_cols']

# ── Load features_9 ───────────────────────────────────────────────────────────
print('Loading features_9...')
dfs = [pd.read_parquet(f) for f in sorted(FEATURES_DIR.glob('*_features.parquet'))]
df9 = pd.concat(dfs).sort_index()
df9 = df9[df9.index >= START_DATE].copy()

# ── Run MFE model + hour filter to get signals ───────────────────────────────
print('Running MFE model...')
X = df9[feature_cols].ffill().fillna(0)
df9['q50_mfe'] = mfe_model.predict(X)
df9['hour']    = pd.to_datetime(df9.index).hour
df_filtered    = df9[(df9['q50_mfe'] >= MFE_THRESH) & df9['hour'].isin(HOURS_ALLOWED)].copy()

# ── Direction rules ───────────────────────────────────────────────────────────
def apply_direction_rules(df):
    dirs = pd.Series(np.nan, index=df.index)
    pair = df['pair']
    dirs = dirs.where(pair != 'USDJPY', -1.0)
    m = pair == 'AUDUSD'
    lc = m & (df.get('beta_gbpusd_1w', pd.Series(np.nan, index=df.index)).gt(0.775) |
               df.get('atr_24', pd.Series(np.nan, index=df.index)).lt(40.8))
    dirs = dirs.where(~lc, 1.0).where(~(m & ~lc), np.nan)
    m = pair == 'GBPUSD'
    lc = m & df.get('csi_usd_24h', pd.Series(np.nan, index=df.index)).lt(0.004)
    dirs = dirs.where(~lc, 1.0).where(~(m & ~lc), np.nan)
    m = pair == 'EURUSD'
    lc = m & df.get('corr_audusd_24h', pd.Series(np.nan, index=df.index)).lt(0.22)
    dirs = dirs.where(~lc, 1.0).where(~(m & ~lc), np.nan)
    m = pair == 'NZDUSD'
    lc = m & df.get('dist_5d_high', pd.Series(np.nan, index=df.index)).gt(0.35)
    dirs = dirs.where(~lc, 1.0).where(~(m & ~lc), np.nan)
    m = pair == 'USDCHF'
    lc = m & df.get('corr_eurusd_1w', pd.Series(np.nan, index=df.index)).gt(-0.60)
    dirs = dirs.where(~lc, 1.0).where(~(m & ~lc), np.nan)
    m = pair == 'CHFJPY'
    cv = df.get('corr_usdjpy_1w', pd.Series(np.nan, index=df.index))
    lc = m & cv.gt(0.40); sc = m & cv.lt(0.26)
    dirs = dirs.where(~lc, 1.0).where(~sc, -1.0).where(~(m & ~lc & ~sc), np.nan)
    m = pair == 'CADJPY'
    vt = df.get('vol_trend', pd.Series(np.nan, index=df.index))
    lc = m & vt.lt(1.15); sc = m & vt.ge(1.15)
    dirs = dirs.where(~lc, 1.0).where(~sc, -1.0)
    m = pair == 'AUDJPY'
    lc = m & df.get('beta_usdjpy_1w', pd.Series(np.nan, index=df.index)).gt(0.74)
    dirs = dirs.where(~lc, 1.0).where(~(m & ~lc), np.nan)
    m = pair == 'EURJPY'
    lc = m & df.get('beta_eurusd_1w', pd.Series(np.nan, index=df.index)).gt(0.38)
    dirs = dirs.where(~lc, 1.0).where(~(m & ~lc), np.nan)
    m = pair == 'GBPJPY'
    lc = m & df.get('beta_eurusd_1w', pd.Series(np.nan, index=df.index)).gt(0.50)
    dirs = dirs.where(~lc, 1.0).where(~(m & ~lc), np.nan)
    m = pair == 'EURAUD'
    lc = m & df.get('corr_audusd_24h', pd.Series(np.nan, index=df.index)).lt(0.22)
    dirs = dirs.where(~lc, 1.0).where(~(m & ~lc), np.nan)
    m = pair == 'AUDNZD'
    lc = m & df.get('corr_regime_audusd', pd.Series(np.nan, index=df.index)).gt(0.0)
    dirs = dirs.where(~lc, 1.0).where(~(m & ~lc), np.nan)
    m = pair == 'EURGBP'
    sc = m & df.get('csi_usd_24h', pd.Series(np.nan, index=df.index)).gt(0.004)
    dirs = dirs.where(~sc, -1.0).where(~(m & ~sc), np.nan)
    dirs = dirs.where(pair != 'USDCAD', np.nan)
    return dirs

df_filtered['direction'] = apply_direction_rules(df_filtered)
df_filtered = df_filtered[df_filtered['direction'].notna()].copy()

# ── Apply 72h cooldown to get signal start bars ───────────────────────────────
print('Applying cooldown to get signal windows...')
cooldown_until = {}
signal_list = []
for ts, row in df_filtered.sort_index().iterrows():
    p = row['pair']
    if p in cooldown_until and ts < cooldown_until[p]:
        continue
    cooldown_until[p] = ts + pd.Timedelta(hours=COOLDOWN_H)
    signal_list.append({'start_ts': ts, 'pair': p, 'direction': row['direction']})

print(f'  Signal windows: {len(signal_list):,}')

# ── Load geometric features_8 ────────────────────────────────────────────────
print('Loading features_8 (geometric)...')
g8_files = [f for f in sorted(GEOM_DIR.glob('*_geometric.parquet')) if 'all_pairs' not in f.name]
dg = pd.concat([pd.read_parquet(f) for f in g8_files]).sort_index()
dg = dg[dg.index >= START_DATE]
geom_cols = [c for c in dg.columns if c not in ('pair', 'label_1H')]

# ── Load 1H price data ────────────────────────────────────────────────────────
print('Loading 1H price data...')
price_data = {}
for pair in PAIRS_ALL:
    fpath = PROCESSED_DIR / f'{pair}_1H.parquet'
    if fpath.exists():
        price_data[pair] = pd.read_parquet(fpath).sort_index()

# ── Build intraday bar dataset ────────────────────────────────────────────────
# For each signal window, collect all 1h bars within (hours 7-20 only)
# and compute short-term fwd returns at each bar
print('Building intraday bar dataset...')

EXCLUDE_PREFIXES = ('peer_', 'corr_', 'beta_', 'relstr_', 'csi_', 'is_',
                    'dow_', 'hour_')

rows = []
for sig in signal_list:
    start_ts  = sig['start_ts']
    pair      = sig['pair']
    direction = sig['direction']
    end_ts    = start_ts + pd.Timedelta(hours=WINDOW_H)
    pip       = 0.01 if pair in JPY_PAIRS else 0.0001

    if pair not in price_data:
        continue
    bars = price_data[pair]

    # All bars within the 72h window (hours 7-20 only)
    window_bars = bars[(bars.index >= start_ts) & (bars.index < end_ts)]
    window_bars = window_bars[pd.to_datetime(window_bars.index).hour.isin(HOURS_ALLOWED)]

    for ts, bar in window_bars.iterrows():
        hour_of_day = pd.Timestamp(ts).hour

        # Short-term forward returns
        fwd = {}
        valid = True
        for h in HORIZONS:
            future = bars[bars.index > ts].head(h)
            if len(future) < h:
                valid = False
                break
            fwd[h] = direction * (future['close'].iloc[-1] - bar['close']) / pip
        if not valid:
            continue

        row = {
            'ts':        ts,
            'pair':      pair,
            'direction': direction,
            'hour':      hour_of_day,
            'bar_in_window': int((ts - start_ts).total_seconds() / 3600),
        }
        for h in HORIZONS:
            row[f'fwd_{h}h']     = fwd[h]
            row[f'correct_{h}h'] = 1 if fwd[h] > 0 else 0

        rows.append(row)

df_bars = pd.DataFrame(rows).set_index('ts')
print(f'  Total intraday bars: {len(df_bars):,}  across {len(signal_list):,} windows')
for h in HORIZONS:
    print(f'  Baseline accuracy @{h}h: {df_bars[f"correct_{h}h"].mean():.1%}')

# ── Join features onto intraday bars ─────────────────────────────────────────
print('Joining features...')

# features_9 — filter to universal only
f9_universal = [c for c in feature_cols
                if not any(c.startswith(p) for p in EXCLUDE_PREFIXES)]

df9_feat = df9[f9_universal + ['pair']].copy()
df9_feat.index.name = 'ts'

# features_8
dg_feat = dg[geom_cols + ['pair']].copy()
dg_feat.index.name = 'ts'

# Merge both onto df_bars using (ts, pair)
df_bars = df_bars.reset_index()
df9_r   = df9_feat.reset_index()
dg_r    = dg_feat.reset_index()

df_bars = df_bars.merge(df9_r, on=['ts', 'pair'], how='left')
df_bars = df_bars.merge(dg_r,  on=['ts', 'pair'], how='left', suffixes=('', '_g8'))
df_bars = df_bars.set_index('ts')
df_bars.index.name = None

all_feat_cols = (
    [('f9', c) for c in f9_universal] +
    [('f8', c) for c in geom_cols]
)
print(f'  Features to scan: {len(all_feat_cols)}  ({len(f9_universal)} f9 + {len(geom_cols)} f8)')

# ── Correlation scan ──────────────────────────────────────────────────────────
def scan(df, feat_list, target_col, min_n=MIN_N):
    y = df[target_col].values.astype(float)
    rows = []
    for src, col in feat_list:
        if col not in df.columns:
            continue
        x = df[col].values.astype(float)
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() < min_n:
            continue
        xm, ym = x[mask], y[mask]
        if target_col.startswith('correct'):
            r, p = stats.pointbiserialr(ym > 0, xm)
        else:
            r, p = stats.pearsonr(xm, ym)
        rows.append({'src': src, 'feature': col, 'corr': r,
                     'abs_corr': abs(r), 'p': p, 'n': mask.sum()})
    return pd.DataFrame(rows).sort_values('abs_corr', ascending=False)

print(f'\nScanning features against intraday accuracy...')
results = {}
for h in HORIZONS:
    print(f'  Horizon {h}h...')
    results[h] = scan(df_bars, all_feat_cols, f'correct_{h}h')

SEP = '=' * 72

# ── Top features per horizon ──────────────────────────────────────────────────
for h in HORIZONS:
    print(f'\n{SEP}')
    print(f'  HORIZON {h}H — TOP {TOP_N} FEATURES  (N={len(df_bars):,} intraday bars)')
    print(f'{SEP}')
    res = results[h]
    print(f'  {"Src":<4}  {"Feature":<42}  {"corr":>7}  {"p":>10}  {"N":>6}')
    print(f'  {"-"*72}')
    for _, row in res.head(TOP_N).iterrows():
        sig = '***' if row['p'] < 0.001 else ('** ' if row['p'] < 0.01 else ('*  ' if row['p'] < 0.05 else '   '))
        print(f'  {row["src"]:<4}  {row["feature"]:<42}  {row["corr"]:>+7.4f}  {sig}  {row["n"]:>6}')

# ── Consistent features across all horizons ───────────────────────────────────
print(f'\n{SEP}')
print(f'  CONSISTENT FEATURES  (top 30 at ALL three horizons: 4h, 6h, 8h)')
print(f'{SEP}')
sets = [set(results[h].head(30)['feature']) for h in HORIZONS]
consistent = sets[0] & sets[1] & sets[2]

if consistent:
    r_ref = results[6].set_index('feature')
    rows_c = []
    for f in consistent:
        corrs = [results[h].set_index('feature').loc[f, 'corr']
                 if f in results[h].set_index('feature').index else np.nan
                 for h in HORIZONS]
        src = r_ref.loc[f, 'src'] if f in r_ref.index else '?'
        rows_c.append((f, corrs, src, sum(abs(c) for c in corrs if not np.isnan(c))))
    rows_c.sort(key=lambda x: -x[3])

    print(f'\n  {"Feature":<42}  {"4h":>7}  {"6h":>7}  {"8h":>7}  Src')
    print(f'  {"-"*72}')
    for f, corrs, src, _ in rows_c:
        print(f'  {f:<42}  {corrs[0]:>+7.4f}  {corrs[1]:>+7.4f}  {corrs[2]:>+7.4f}  {src}')
else:
    print('  No features consistent across all 3 horizons at top 30.')
    rows_c = []

# ── Threshold tests ───────────────────────────────────────────────────────────
print(f'\n{SEP}')
print(f'  THRESHOLD TESTS — TOP FEATURES')
print(f'{SEP}')

# Pick top features to test: consistent ones + top from each horizon
test_feats = [r[0] for r in rows_c[:6]]
for h in HORIZONS:
    for f in results[h].head(5)['feature'].tolist():
        if f not in test_feats:
            test_feats.append(f)
test_feats = test_feats[:15]

for feat in test_feats:
    if feat not in df_bars.columns:
        continue
    vals = df_bars[feat].dropna()
    if len(vals) < MIN_N:
        continue
    p25, p50, p75 = vals.quantile([0.25, 0.5, 0.75])
    print(f'\n  {feat}  (p25={p25:.4f}  p50={p50:.4f}  p75={p75:.4f})')
    print(f'  {"Bucket":<22}  {"N":>6}', end='')
    for h in HORIZONS:
        print(f'  {"Acc@"+str(h)+"h":>8}  {"Pip@"+str(h)+"h":>9}', end='')
    print()
    print(f'  {"-"*80}')
    buckets = [
        (f'< p25 ({p25:.3f})',  df_bars[df_bars[feat] <  p25]),
        ('p25-p50',             df_bars[(df_bars[feat] >= p25) & (df_bars[feat] < p50)]),
        ('p50-p75',             df_bars[(df_bars[feat] >= p50) & (df_bars[feat] < p75)]),
        (f'> p75 ({p75:.3f})',  df_bars[df_bars[feat] >= p75]),
    ]
    for label, bucket in buckets:
        if len(bucket) < 20:
            continue
        print(f'  {label:<22}  {len(bucket):>6}', end='')
        for h in HORIZONS:
            acc = bucket[f'correct_{h}h'].mean()
            pip = bucket[f'fwd_{h}h'].mean()
            print(f'  {acc:>8.1%}  {pip:>+9.1f}p', end='')
        print()

# ── Hour-of-day accuracy within windows ──────────────────────────────────────
print(f'\n{SEP}')
print(f'  ACCURACY BY HOUR OF DAY  (within regime windows)')
print(f'{SEP}')
print(f'\n  {"Hour":<6}  {"N":>6}', end='')
for h in HORIZONS:
    print(f'  {"Acc@"+str(h)+"h":>8}  {"Pip@"+str(h)+"h":>9}', end='')
print()
print(f'  {"-"*70}')
for hr in range(7, 21):
    sub = df_bars[df_bars['hour'] == hr]
    if len(sub) < 20:
        continue
    print(f'  {hr:<6}  {len(sub):>6}', end='')
    for h in HORIZONS:
        acc = sub[f'correct_{h}h'].mean()
        pip = sub[f'fwd_{h}h'].mean()
        print(f'  {acc:>8.1%}  {pip:>+9.1f}p', end='')
    print()

# ── Bar position within window ────────────────────────────────────────────────
print(f'\n{SEP}')
print(f'  ACCURACY BY POSITION WITHIN 72H WINDOW')
print(f'{SEP}')
print(f'\n  {"Bar h":<8}  {"N":>6}', end='')
for h in HORIZONS:
    print(f'  {"Acc@"+str(h)+"h":>8}  {"Pip@"+str(h)+"h":>9}', end='')
print()
print(f'  {"-"*70}')
bins = [(0,12,'h0-12'), (12,24,'h12-24'), (24,36,'h24-36'), (36,48,'h36-48'), (48,60,'h48-60'), (60,72,'h60-72')]
for lo, hi, label in bins:
    sub = df_bars[(df_bars['bar_in_window'] >= lo) & (df_bars['bar_in_window'] < hi)]
    if len(sub) < 20:
        continue
    print(f'  {label:<8}  {len(sub):>6}', end='')
    for h in HORIZONS:
        acc = sub[f'correct_{h}h'].mean()
        pip = sub[f'fwd_{h}h'].mean()
        print(f'  {acc:>8.1%}  {pip:>+9.1f}p', end='')
    print()


# ── Combination scan ──────────────────────────────────────────────────────────
# Take top N single features, binarise each at p25/p50/p75,
# then exhaustively test all 2-way and 3-way AND combinations.
# Report combos that beat baseline by most, with sufficient N.
print(f'\n{SEP}')
print(f'  COMBINATION SCAN  (2-way and 3-way AND combos of top features)')
print(f'  Target horizon: 6h  |  Min combo bucket size: 300 bars')
print(f'{SEP}')

COMBO_HORIZON  = 6       # focus on 6h as the primary intraday target
COMBO_MIN_N    = 300     # minimum bars in a combo bucket to be valid
COMBO_TOP_FEAT = 20      # how many top single features to combine

baseline_acc = df_bars[f'correct_{COMBO_HORIZON}h'].mean()
baseline_pip = df_bars[f'fwd_{COMBO_HORIZON}h'].mean()
print(f'\n  Baseline @{COMBO_HORIZON}h: acc={baseline_acc:.1%}  pip={baseline_pip:+.1f}p')

# Build candidate features — top from single scan + consistent ones
cand_feats = []
for h in HORIZONS:
    cand_feats += results[h].head(COMBO_TOP_FEAT)['feature'].tolist()
cand_feats = list(dict.fromkeys(cand_feats))   # deduplicate, preserve order
cand_feats = [f for f in cand_feats if f in df_bars.columns]
print(f'  Candidate features: {len(cand_feats)}')

# Pre-binarise each feature at p25 and p75
# Each feature gets two binary columns: feat_lo (< p25) and feat_hi (> p75)
binary_cols = []
for feat in cand_feats:
    vals = df_bars[feat].dropna()
    if len(vals) < COMBO_MIN_N:
        continue
    p25 = vals.quantile(0.25)
    p75 = vals.quantile(0.75)
    corr_sign = results[COMBO_HORIZON].set_index('feature').loc[feat, 'corr'] \
                if feat in results[COMBO_HORIZON].set_index('feature').index else 0

    # Only keep the direction that the single-feature analysis says is better
    # corr < 0 means low values are better -> use _lo
    # corr > 0 means high values are better -> use _hi
    lo_col = f'{feat}__lo'
    hi_col = f'{feat}__hi'
    df_bars[lo_col] = df_bars[feat] < p25
    df_bars[hi_col] = df_bars[feat] > p75

    if corr_sign <= 0:
        binary_cols.append((feat, lo_col, f'{feat}<p25'))
    else:
        binary_cols.append((feat, hi_col, f'{feat}>p75'))

print(f'  Binary conditions: {len(binary_cols)}')

target_col = f'correct_{COMBO_HORIZON}h'
pip_col    = f'fwd_{COMBO_HORIZON}h'
y_acc = df_bars[target_col].values
y_pip = df_bars[pip_col].values

import itertools

# ── 2-way combos ─────────────────────────────────────────────────────────────
print(f'\n  Scanning 2-way combos...')
combo2_results = []
for (f1, c1, l1), (f2, c2, l2) in itertools.combinations(binary_cols, 2):
    mask = (df_bars[c1] & df_bars[c2]).values
    n = mask.sum()
    if n < COMBO_MIN_N:
        continue
    acc = y_acc[mask].mean()
    pip = y_pip[mask].mean()
    combo2_results.append({
        'combo': f'{l1}  AND  {l2}',
        'n': n, 'acc': acc, 'pip': pip,
        'lift': acc - baseline_acc,
    })

combo2_df = pd.DataFrame(combo2_results).sort_values('acc', ascending=False)
print(f'  Valid 2-way combos: {len(combo2_df):,}')

print(f'\n  TOP 20 TWO-WAY COMBINATIONS')
print(f'  {"N":>6}  {"Acc":>7}  {"Lift":>7}  {"Pip":>8}  Combination')
print(f'  {"-"*90}')
for _, row in combo2_df.head(20).iterrows():
    print(f'  {int(row["n"]):>6}  {row["acc"]:>7.1%}  {row["lift"]:>+7.1%}  {row["pip"]:>+8.1f}p  {row["combo"]}')

print(f'\n  BOTTOM 5 (worst — avoid these combos):')
for _, row in combo2_df.tail(5).iterrows():
    print(f'  {int(row["n"]):>6}  {row["acc"]:>7.1%}  {row["lift"]:>+7.1%}  {row["pip"]:>+8.1f}p  {row["combo"]}')

# ── 3-way combos (only from top 15 single features to keep runtime manageable)
print(f'\n  Scanning 3-way combos (top 15 features)...')
binary_cols_3 = binary_cols[:15]
combo3_results = []
for (f1,c1,l1), (f2,c2,l2), (f3,c3,l3) in itertools.combinations(binary_cols_3, 3):
    mask = (df_bars[c1] & df_bars[c2] & df_bars[c3]).values
    n = mask.sum()
    if n < COMBO_MIN_N:
        continue
    acc = y_acc[mask].mean()
    pip = y_pip[mask].mean()
    combo3_results.append({
        'combo': f'{l1}  AND  {l2}  AND  {l3}',
        'n': n, 'acc': acc, 'pip': pip,
        'lift': acc - baseline_acc,
    })

combo3_df = pd.DataFrame(combo3_results).sort_values('acc', ascending=False) if combo3_results else pd.DataFrame()
print(f'  Valid 3-way combos: {len(combo3_df):,}')

if not combo3_df.empty:
    print(f'\n  TOP 20 THREE-WAY COMBINATIONS')
    print(f'  {"N":>6}  {"Acc":>7}  {"Lift":>7}  {"Pip":>8}  Combination')
    print(f'  {"-"*100}')
    for _, row in combo3_df.head(20).iterrows():
        print(f'  {int(row["n"]):>6}  {row["acc"]:>7.1%}  {row["lift"]:>+7.1%}  {row["pip"]:>+8.1f}p  {row["combo"]}')

# ── Also test combos at 4h and 8h for the best 2-way and 3-way combos ────────
print(f'\n{SEP}')
print(f'  BEST COMBOS VERIFIED ACROSS ALL HORIZONS (4h / 6h / 8h)')
print(f'{SEP}')

top_combos_2 = combo2_df.head(10)
top_combos_3 = combo3_df.head(5) if not combo3_df.empty else pd.DataFrame()

def eval_combo_all_horizons(condition_mask):
    row = {}
    for h in HORIZONS:
        y_a = df_bars[f'correct_{h}h'].values
        y_p = df_bars[f'fwd_{h}h'].values
        n   = condition_mask.sum()
        row[f'acc_{h}h'] = y_a[condition_mask].mean() if n > 0 else np.nan
        row[f'pip_{h}h'] = y_p[condition_mask].mean() if n > 0 else np.nan
        row[f'lift_{h}h'] = row[f'acc_{h}h'] - df_bars[f'correct_{h}h'].mean()
    row['n'] = n
    return row

print(f'\n  {"N":>6}  {"Acc4h":>7} {"Lift4h":>7}  {"Acc6h":>7} {"Lift6h":>7}  {"Acc8h":>7} {"Lift8h":>7}  Combination')
print(f'  {"-"*110}')

# Rebuild condition masks for best combos
for combo_df, tag in [(top_combos_2, '2-way'), (top_combos_3, '3-way')]:
    if combo_df.empty:
        continue
    print(f'\n  --- {tag} ---')
    for _, row in combo_df.iterrows():
        # Parse combo string back to condition columns
        parts = row['combo'].split('  AND  ')
        masks = []
        for p in parts:
            # Find matching binary col
            for feat, col, label in binary_cols:
                if label == p.strip():
                    masks.append(df_bars[col].values)
                    break
        if len(masks) != len(parts):
            continue
        combined = np.ones(len(df_bars), dtype=bool)
        for m in masks:
            combined = combined & m
        stats_row = eval_combo_all_horizons(combined)
        print(f'  {stats_row["n"]:>6}  '
              f'{stats_row["acc_4h"]:>7.1%} {stats_row["lift_4h"]:>+7.1%}  '
              f'{stats_row["acc_6h"]:>7.1%} {stats_row["lift_6h"]:>+7.1%}  '
              f'{stats_row["acc_8h"]:>7.1%} {stats_row["lift_8h"]:>+7.1%}  '
              f'{row["combo"]}')


# ── Per-window coverage analysis ──────────────────────────────────────────────
# Key question: does each combo fire at least once per 72h window?
# If not, we're killing volume below the minimum acceptable level.
print(f'\n{SEP}')
print(f'  PER-WINDOW COVERAGE  (how many entries per 72h window does each combo produce?)')
print(f'  We need >= 1 entry per window minimum, ideally 2-4 per day = 6-12 per window')
print(f'{SEP}')

# Add window_id to df_bars so we can group by window
window_map = {}
for i, sig in enumerate(signal_list):
    window_map[(sig['start_ts'], sig['pair'])] = i

df_bars['window_id'] = [
    window_map.get((
        # find the window this bar belongs to: walk back to find the signal start
        # stored as bar_in_window offset from signal start
        pd.Timestamp(idx) - pd.Timedelta(hours=int(df_bars.loc[idx, 'bar_in_window']))
            if not isinstance(df_bars.loc[idx, 'bar_in_window'], pd.Series)
            else pd.Timestamp(idx) - pd.Timedelta(hours=int(df_bars.loc[idx, 'bar_in_window'].iloc[0])),
        df_bars.loc[idx, 'pair']
            if not isinstance(df_bars.loc[idx, 'pair'], pd.Series)
            else df_bars.loc[idx, 'pair'].iloc[0]
    ), -1)
    for idx in df_bars.index
]

n_windows = len(signal_list)
print(f'\n  Total windows: {n_windows:,}  |  Total intraday bars: {len(df_bars):,}')
print(f'  Avg bars per window (hours 7-20 in 72h): {len(df_bars)/n_windows:.1f}')

print(f'\n  {"Combination":<65}  {"TotalN":>7}  {"Windows":>8}  {"Cov%":>6}  {"Avg/win":>8}  {"Acc@6h":>8}')
print(f'  {"-"*115}')

def coverage_stats(condition_mask, label):
    sub = df_bars[condition_mask]
    if len(sub) == 0:
        return
    n_covered = sub['window_id'].nunique()
    avg_per_win = len(sub) / n_windows
    acc = sub[f'correct_{COMBO_HORIZON}h'].mean()
    cov_pct = n_covered / n_windows
    print(f'  {label:<65}  {len(sub):>7,}  {n_covered:>8,}  {cov_pct:>6.1%}  {avg_per_win:>8.1f}  {acc:>8.1%}')

# Baseline: all bars
coverage_stats(np.ones(len(df_bars), dtype=bool), 'ALL BARS (no filter)')

# Single best features
for feat, col, label in binary_cols[:10]:
    coverage_stats(df_bars[col].values, label)

print()
# Best 2-way combos
for _, row in combo2_df.head(15).iterrows():
    parts = row['combo'].split('  AND  ')
    masks = []
    for p in parts:
        for feat, col, label in binary_cols:
            if label == p.strip():
                masks.append(df_bars[col].values); break
    if len(masks) != len(parts): continue
    combined = np.ones(len(df_bars), dtype=bool)
    for m in masks: combined = combined & m
    coverage_stats(combined, row['combo'])

print()
# Best 3-way combos
for _, row in combo3_df.head(10).iterrows():
    parts = row['combo'].split('  AND  ')
    masks = []
    for p in parts:
        for feat, col, label in binary_cols:
            if label == p.strip():
                masks.append(df_bars[col].values); break
    if len(masks) != len(parts): continue
    combined = np.ones(len(df_bars), dtype=bool)
    for m in masks: combined = combined & m
    coverage_stats(combined, row['combo'])

# ── What threshold gives >= 1 trade per window on average? ───────────────────
print(f'\n{SEP}')
print(f'  MINIMUM FILTER: what single-feature threshold gives ~1+ trade/window?')
print(f'  (i.e. fire on at least 1/30 bars = top ~3% most favorable bars per window)')
print(f'{SEP}')
print(f'\n  At 30 bars/window average, to get 1 trade/window we need to fire on ~3% of bars')
print(f'  = {int(len(df_bars)*0.03):,} bars total minimum\n')

for feat, col, label in binary_cols:
    sub = df_bars[df_bars[col]]
    if len(sub) < 400: continue
    n_covered = sub['window_id'].nunique()
    avg_per_win = len(sub) / n_windows
    acc = sub[f'correct_{COMBO_HORIZON}h'].mean()
    if avg_per_win >= 0.8:   # at least fires in 80% of windows
        lift = acc - baseline_acc
        print(f'  {label:<42}  N={len(sub):>5,}  avg/win={avg_per_win:.1f}  cov={n_covered/n_windows:.0%}  acc={acc:.1%}  lift={lift:+.1%}')

print(f'\n{SEP}')
print(f'  DONE')
print(f'{SEP}')
