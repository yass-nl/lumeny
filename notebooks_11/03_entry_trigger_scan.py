"""
Entry Trigger Scan
==================
We already have direction (regime layer) and quality gate (MFE >= 50).
This script answers: which features from features_9 (microstructure)
and features_8 (geometric) tell us WHEN to enter within a regime?

Method:
  - Start from the same 2,117 signals (MFE>=50, hours 7-20, 72h cooldown)
  - For each signal bar, compute short-term forward returns: 6h, 12h, 24h
  - Correlate every feature (f9 + f8) against:
      (a) whether the short-term move is in the signal direction (accuracy)
      (b) the magnitude of the short-term move in signal direction (pips)
  - Also test threshold buckets (p25/p50/p75) to find clean cut-offs
  - Goal: find 1-3 features per set that reliably say "enter NOW" vs "wait"

Output:
  - Top 20 features from each set by |corr| with short-term direction
  - Threshold tests on top features
  - Combined score: features that work at BOTH 12h and 24h horizons
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
TOP_N          = 20
MIN_N          = 500  # must fire on majority of signals — no narrow pair-specific features

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

# ── Run MFE model + hour filter ───────────────────────────────────────────────
print('Running MFE model...')
X = df9[feature_cols].ffill().fillna(0)
df9['q50_mfe'] = mfe_model.predict(X)
df9['hour']    = pd.to_datetime(df9.index).hour
df9 = df9[(df9['q50_mfe'] >= MFE_THRESH) & df9['hour'].isin(HOURS_ALLOWED)].copy()

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

df9['direction'] = apply_direction_rules(df9)
df9 = df9[df9['direction'].notna()].copy()

# ── Apply 72h cooldown ────────────────────────────────────────────────────────
print('Applying cooldown...')
cooldown_until = {}
kept = []
for ts, row in df9.sort_index().iterrows():
    p = row['pair']
    if p in cooldown_until and ts < cooldown_until[p]:
        continue
    cooldown_until[p] = ts + pd.Timedelta(hours=COOLDOWN_H)
    kept.append(ts)
df_sig = df9.loc[kept].copy()
print(f'  Signals: {len(df_sig):,}')

# ── Load geometric features_8 and join ───────────────────────────────────────
print('Loading features_8 (geometric)...')
g8_files = sorted(GEOM_DIR.glob('*_geometric.parquet'))
g8_files = [f for f in g8_files if 'all_pairs' not in f.name]
dg = pd.concat([pd.read_parquet(f) for f in g8_files]).sort_index()
dg = dg[dg.index >= START_DATE]
geom_cols = [c for c in dg.columns if c not in ('pair', 'label_1H')]
print(f'  Geometric features: {len(geom_cols)}')

df_sig = df_sig.reset_index().rename(columns={df_sig.reset_index().columns[0]: 'datetime'})
dg_reset = dg[geom_cols + ['pair']].reset_index().rename(columns={dg.reset_index().columns[0]: 'datetime'})
df_sig = df_sig.merge(dg_reset, on=['datetime', 'pair'], how='left', suffixes=('', '_g8'))
df_sig = df_sig.set_index('datetime')
df_sig.index.name = None
print(f'  After join: {len(df_sig):,} rows')

# ── Compute short-term forward returns ───────────────────────────────────────
print('Computing short-term forward returns...')
close_all = {}
for pair in PAIRS_ALL:
    fpath = PROCESSED_DIR / f'{pair}_1H.parquet'
    if fpath.exists():
        close_all[pair] = pd.read_parquet(fpath)['close'].sort_index()

for horizon in [6, 12, 24]:
    fwd_rows = []
    for pair in PAIRS_ALL:
        if pair not in close_all:
            continue
        c   = close_all[pair]
        pip = 0.01 if pair in JPY_PAIRS else 0.0001
        fwd = (c.shift(-horizon) - c) / pip
        fwd_rows.append(pd.DataFrame({f'fwd_{horizon}h': fwd, 'pair': pair}))
    fwd_df = pd.concat(fwd_rows)
    fwd_df.index.name = 'datetime'
    sig_reset = df_sig.reset_index().rename(columns={df_sig.reset_index().columns[0]: 'datetime'})
    sig_reset = sig_reset.merge(fwd_df.reset_index(), on=['datetime', 'pair'], how='left')
    df_sig[f'fwd_{horizon}h'] = sig_reset[f'fwd_{horizon}h'].values

# Signed returns in signal direction
for h in [6, 12, 24]:
    df_sig[f'signed_{h}h'] = df_sig['direction'] * df_sig[f'fwd_{h}h']
    df_sig[f'correct_{h}h'] = (df_sig[f'signed_{h}h'] > 0).astype(float)

print(f'  Accuracy @ 6h:  {df_sig["correct_6h"].mean():.1%}')
print(f'  Accuracy @ 12h: {df_sig["correct_12h"].mean():.1%}')
print(f'  Accuracy @ 24h: {df_sig["correct_24h"].mean():.1%}')

# ── Feature correlation scan ──────────────────────────────────────────────────
# Only keep universal features — exclude pair-specific peer/corr/beta/relstr/csi
# which only exist for a subset of pairs (narrow N)
EXCLUDE_PREFIXES = ('peer_', 'corr_', 'beta_', 'relstr_', 'csi_', 'is_')
def is_universal(col):
    return not any(col.startswith(p) for p in EXCLUDE_PREFIXES)

f9_cols  = [c for c in feature_cols if c in df_sig.columns and is_universal(c)]
f8_cols  = [c for c in geom_cols if c in df_sig.columns and is_universal(c)]
all_feat = [('f9', c) for c in f9_cols] + [('f8', c) for c in f8_cols]
print(f'  Universal f9 features: {len(f9_cols)}  |  f8 features: {len(f8_cols)}')

print(f'\nScanning {len(f9_cols)} f9 features + {len(f8_cols)} f8 features...')

def scan_features(df, feat_list, target_col, min_n=MIN_N):
    rows = []
    y = df[target_col].values.astype(float)
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
        rows.append({
            'src': src, 'feature': col,
            'corr': r, 'abs_corr': abs(r), 'p': p, 'n': mask.sum(),
        })
    return pd.DataFrame(rows).sort_values('abs_corr', ascending=False)

# Run scan for each horizon
results = {}
for h in [6, 12, 24]:
    print(f'  Scanning horizon {h}h...')
    res_acc = scan_features(df_sig, all_feat, f'correct_{h}h')
    res_pip = scan_features(df_sig, all_feat, f'signed_{h}h')
    results[h] = {'acc': res_acc, 'pip': res_pip}

# ── Print results ─────────────────────────────────────────────────────────────
SEP = '=' * 72

for h in [6, 12, 24]:
    print(f'\n{SEP}')
    print(f'  HORIZON {h}H — TOP {TOP_N} FEATURES BY |CORR| WITH DIRECTIONAL ACCURACY')
    print(f'{SEP}')
    res = results[h]['acc']
    print(f'\n  {"Src":<4}  {"Feature":<40}  {"corr":>7}  {"p":>10}  {"N":>6}')
    print(f'  {"-"*70}')
    for _, row in res.head(TOP_N).iterrows():
        sig = '***' if row['p'] < 0.001 else ('** ' if row['p'] < 0.01 else ('*  ' if row['p'] < 0.05 else '   '))
        print(f'  {row["src"]:<4}  {row["feature"]:<40}  {row["corr"]:>+7.4f}  {sig}  {row["n"]:>6}')

# ── Features consistent across ALL horizons ───────────────────────────────────
print(f'\n{SEP}')
print(f'  CONSISTENT FEATURES  (appear in top 30 at BOTH 12h AND 24h)')
print(f'{SEP}')
top30_12 = set(results[12]['acc'].head(30)['feature'])
top30_24 = set(results[24]['acc'].head(30)['feature'])
consistent = top30_12 & top30_24
if consistent:
    # Merge stats
    r12 = results[12]['acc'].set_index('feature')
    r24 = results[24]['acc'].set_index('feature')
    print(f'\n  {"Feature":<42}  {"corr_12h":>9}  {"corr_24h":>9}  {"Src"}')
    print(f'  {"-"*68}')
    rows_c = []
    for f in consistent:
        c12 = r12.loc[f, 'corr'] if f in r12.index else np.nan
        c24 = r24.loc[f, 'corr'] if f in r24.index else np.nan
        src = r12.loc[f, 'src'] if f in r12.index else '?'
        rows_c.append((f, c12, c24, src, abs(c12)+abs(c24)))
    rows_c.sort(key=lambda x: -x[4])
    for f, c12, c24, src, _ in rows_c:
        print(f'  {f:<42}  {c12:>+9.4f}  {c24:>+9.4f}  {src}')

# ── Threshold test on top 5 consistent features ──────────────────────────────
print(f'\n{SEP}')
print(f'  THRESHOLD TESTS — TOP CONSISTENT FEATURES')
print(f'  (how accuracy changes across quartile buckets)')
print(f'{SEP}')

top_feats = [r[0] for r in rows_c[:8]] if rows_c else []
# Also add top f8 and f9 individually if not already in consistent
for src_filter, res_key in [('f8', 24), ('f9', 24)]:
    top_src = results[res_key]['acc'][results[res_key]['acc']['src'] == src_filter].head(3)['feature'].tolist()
    for f in top_src:
        if f not in top_feats:
            top_feats.append(f)

for feat in top_feats[:12]:
    if feat not in df_sig.columns:
        continue
    vals = df_sig[feat].dropna()
    if len(vals) < MIN_N:
        continue
    p25, p50, p75 = vals.quantile([0.25, 0.5, 0.75])
    print(f'\n  {feat}  (p25={p25:.4f}  p50={p50:.4f}  p75={p75:.4f})')
    print(f'  {"Bucket":<22}  {"N":>5}  {"Acc@12h":>8}  {"Acc@24h":>8}  {"AvgPip@12h":>11}  {"AvgPip@24h":>11}')
    print(f'  {"-"*72}')
    buckets = [
        (f'< p25 ({p25:.3f})',  df_sig[df_sig[feat] <  p25]),
        ('p25-p50',             df_sig[(df_sig[feat] >= p25) & (df_sig[feat] < p50)]),
        ('p50-p75',             df_sig[(df_sig[feat] >= p50) & (df_sig[feat] < p75)]),
        (f'> p75 ({p75:.3f})',  df_sig[df_sig[feat] >= p75]),
    ]
    for label, bucket in buckets:
        bucket = bucket.dropna(subset=['correct_12h', 'correct_24h'])
        if len(bucket) < 5:
            continue
        a12 = bucket['correct_12h'].mean()
        a24 = bucket['correct_24h'].mean()
        p12 = bucket['signed_12h'].mean()
        p24 = bucket['signed_24h'].mean()
        print(f'  {label:<22}  {len(bucket):>5}  {a12:>8.1%}  {a24:>8.1%}  {p12:>+11.1f}p  {p24:>+11.1f}p')

# ── Quick summary by feature source ──────────────────────────────────────────
print(f'\n{SEP}')
print(f'  SUMMARY BY SOURCE')
print(f'{SEP}')
for h in [12, 24]:
    res = results[h]['acc']
    top_f9 = res[res['src'] == 'f9'].head(5)[['feature','corr','p']].values
    top_f8 = res[res['src'] == 'f8'].head(5)[['feature','corr','p']].values
    print(f'\n  Horizon {h}h:')
    print(f'  Top f9 (microstructure):')
    for feat, corr, p in top_f9:
        print(f'    {feat:<40}  corr={corr:+.4f}  p={p:.4f}')
    print(f'  Top f8 (geometric):')
    for feat, corr, p in top_f8:
        print(f'    {feat:<40}  corr={corr:+.4f}  p={p:.4f}')

print(f'\n{SEP}')
print(f'  DONE')
print(f'{SEP}')
