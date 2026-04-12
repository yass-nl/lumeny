"""
Model Signal Comparison — mfe_q50_8h vs mfe_q50 (with vol features)
=====================================================================
Uses local parquet data only (no API fetch).
Cross-pair features set to 0 (not available in per-pair parquets).
Compares signal overlap, timing, quality at thresholds 50 and 70.
Actual MFE measured at 3 horizons: 8h, 24h, 72h.
Cooldown: 8h per pair for both models.
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
FEATURES_DIR  = Path('backend/data/features_9')
PROCESSED_DIR = Path('backend/data/processed')
MODEL_8H  = Path('backend/models_9/mfe_q50_8h/model_1H_Q50.joblib')
MODEL_Q50 = Path('backend/models_9/mfe_q50/model_1H_Q50.joblib')

TRAIN_END  = '2024-06-30'   # use test set only (post train cutoff)
COOLDOWN_H = 8              # per pair, same for both models
HORIZONS   = [8, 24, 72]   # actual MFE measured at each horizon
THRESHOLDS = [50, 70]

PAIRS = [
    'EURUSD','GBPUSD','USDJPY','USDCHF','AUDUSD','USDCAD','NZDUSD',
    'EURJPY','GBPJPY','EURGBP','EURAUD','AUDJPY','CADJPY','CHFJPY','AUDNZD',
]

PIP_SIZE = {
    'EURUSD':0.0001,'GBPUSD':0.0001,'USDJPY':0.01,'USDCHF':0.0001,
    'AUDUSD':0.0001,'USDCAD':0.0001,'NZDUSD':0.0001,'EURJPY':0.01,
    'GBPJPY':0.01,'EURGBP':0.0001,'EURAUD':0.0001,'AUDJPY':0.01,
    'CADJPY':0.01,'CHFJPY':0.01,'AUDNZD':0.0001,
}

# ── Load models ───────────────────────────────────────────────────────────────
print('Loading models...')
b8h  = joblib.load(MODEL_8H)
bq50 = joblib.load(MODEL_Q50)
fc8h = b8h['feature_cols']
fcq  = bq50['feature_cols']
print(f'  mfe_q50_8h : {len(fc8h)} features, {b8h["n_iters"]} iters, cv_pinball={b8h["cv_pinball"]:.3f}')
print(f'  mfe_q50    : {len(fcq)} features, {bq50["n_iters"]} iters, cv_pinball={bq50["cv_pinball"]:.3f}')
vol_extra = sorted(set(fcq) - set(fc8h))
print(f'  Extra vol features in mfe_q50: {vol_extra}')
print(f'  Cooldown: {COOLDOWN_H}h per pair | Horizons evaluated: {HORIZONS}h')

# ── Score all pairs ───────────────────────────────────────────────────────────
print('\nScoring pairs on test set...')
all_rows = []

for pair in PAIRS:
    feat_path = FEATURES_DIR / f'{pair}_features.parquet'
    ohlc_path = PROCESSED_DIR / f'{pair}_1H.parquet'
    if not feat_path.exists() or not ohlc_path.exists():
        print(f'  {pair}: missing data, skip')
        continue

    df_feat = pd.read_parquet(feat_path)
    df_1h   = pd.read_parquet(ohlc_path)[['high','low','close']].sort_index()
    pip     = PIP_SIZE[pair]

    # Test set only
    df_feat = df_feat[df_feat.index > TRAIN_END].copy()
    if len(df_feat) < 100:
        continue

    # Fill missing features with 0
    for c in fc8h:
        if c not in df_feat.columns: df_feat[c] = 0.0
    for c in fcq:
        if c not in df_feat.columns: df_feat[c] = 0.0

    X8 = df_feat[fc8h].ffill().fillna(0)
    Xq = df_feat[fcq].ffill().fillna(0)
    df_feat = df_feat.copy()
    df_feat['pred_8h']  = b8h['model'].predict(X8)
    df_feat['pred_q50'] = bq50['model'].predict(Xq)

    # Compute actual MFE at each horizon
    highs  = df_1h['high'].reindex(df_feat.index)
    lows   = df_1h['low'].reindex(df_feat.index)
    closes = df_1h['close'].reindex(df_feat.index)

    idx_arr   = df_feat.index
    close_arr = closes.values
    high_arr  = highs.values
    low_arr   = lows.values
    n         = len(df_feat)

    max_h = max(HORIZONS)
    for hz in HORIZONS:
        df_feat[f'mfe_long_{hz}h']  = np.nan
        df_feat[f'mfe_short_{hz}h'] = np.nan

    for i in range(n - max_h - 1):
        entry = close_arr[i]
        if np.isnan(entry): continue
        h_full = high_arr[i+1:i+1+max_h]
        l_full = low_arr[i+1:i+1+max_h]
        if np.any(np.isnan(h_full)): continue
        for hz in HORIZONS:
            h_sl = h_full[:hz]
            l_sl = l_full[:hz]
            df_feat.at[idx_arr[i], f'mfe_long_{hz}h']  = (h_sl.max() - entry) / pip
            df_feat.at[idx_arr[i], f'mfe_short_{hz}h'] = (entry - l_sl.min()) / pip

    for hz in HORIZONS:
        df_feat[f'mfe_best_{hz}h'] = df_feat[[f'mfe_long_{hz}h', f'mfe_short_{hz}h']].max(axis=1)

    df_feat['pair'] = pair
    keep_cols = ['pair','pred_8h','pred_q50'] + \
                [f'mfe_long_{hz}h'  for hz in HORIZONS] + \
                [f'mfe_short_{hz}h' for hz in HORIZONS] + \
                [f'mfe_best_{hz}h'  for hz in HORIZONS]
    all_rows.append(df_feat[keep_cols].dropna())
    print(f'  {pair}: {len(df_feat):,} rows scored')

df = pd.concat(all_rows).sort_index()
print(f'\nTotal test rows scored: {len(df):,}')
print(f'Test period: {df.index.min().date()} -> {df.index.max().date()}')

# ── Apply cooldown per pair and collect signals ────────────────────────────────
def apply_cooldown(df_pair, pred_col, threshold):
    """Return boolean mask of bars that pass threshold + cooldown."""
    mask   = np.zeros(len(df_pair), dtype=bool)
    locked = -1
    for i, val in enumerate(df_pair[pred_col].values):
        if i <= locked: continue
        if val >= threshold:
            mask[i] = True
            locked  = i + COOLDOWN_H
    return mask

print('\nApplying cooldown and collecting signals...')
sig_rows = []
for pair, grp in df.groupby('pair'):
    grp = grp.sort_index()
    for thresh in THRESHOLDS:
        m8  = apply_cooldown(grp, 'pred_8h',  thresh)
        mq  = apply_cooldown(grp, 'pred_q50', thresh)
        for i, (ts, row) in enumerate(grp.iterrows()):
            if m8[i] or mq[i]:
                sig_rows.append({
                    'ts':        ts,
                    'pair':      pair,
                    'threshold': thresh,
                    'fire_8h':   bool(m8[i]),
                    'fire_q50':  bool(mq[i]),
                    'pred_8h':   row['pred_8h'],
                    'pred_q50':  row['pred_q50'],
                    **{f'best_{hz}h': row[f'mfe_best_{hz}h'] for hz in HORIZONS},
                    **{f'long_{hz}h': row[f'mfe_long_{hz}h'] for hz in HORIZONS},
                    **{f'short_{hz}h': row[f'mfe_short_{hz}h'] for hz in HORIZONS},
                })

sigs = pd.DataFrame(sig_rows)
n_months = (pd.to_datetime(df.index.max()) - pd.to_datetime(df.index.min())).days / 30.4

# ── Report ────────────────────────────────────────────────────────────────────
def quality(sub, label, thresh):
    if len(sub) == 0:
        print(f'  {label:<20} N=0')
        return
    p8 = sub['pred_8h'].mean()
    pq = sub['pred_q50'].mean()
    parts = [f'  {label:<20} N={len(sub):>4}  pred_8h={p8:>+6.1f}p  pred_q50={pq:>+6.1f}p']
    for hz in HORIZONS:
        b  = sub[f'best_{hz}h'].dropna()
        l  = sub[f'long_{hz}h'].dropna()
        sh = sub[f'short_{hz}h'].dropna()
        hit = (b >= thresh).mean()
        parts.append(
            f'    [{hz:>2}h] best={b.mean():>+7.1f}p  long={l.mean():>+6.1f}p  short={sh.mean():>+6.1f}p  hit>={thresh}p:{hit:.1%}'
        )
    print('\n'.join(parts))

for thresh in THRESHOLDS:
    s = sigs[sigs['threshold'] == thresh]
    only_8h  = s[ s['fire_8h'] & ~s['fire_q50']]
    only_q50 = s[~s['fire_8h'] &  s['fire_q50']]
    both     = s[ s['fire_8h'] &  s['fire_q50']]

    print(f'\n{"="*80}')
    print(f'  THRESHOLD >= {thresh}p  |  test period ~{n_months:.0f} months  |  cooldown={COOLDOWN_H}h')
    print(f'{"="*80}')

    print(f'\n--- Signal counts ---')
    print(f'  mfe_q50_8h only : {len(only_8h):>5,}  ({len(only_8h)/n_months:.1f}/mo)')
    print(f'  mfe_q50 only    : {len(only_q50):>5,}  ({len(only_q50)/n_months:.1f}/mo)')
    print(f'  Both fire       : {len(both):>5,}  ({len(both)/n_months:.1f}/mo)')
    total_8h  = len(only_8h) + len(both)
    total_q50 = len(only_q50) + len(both)
    if total_8h and total_q50:
        overlap_pct_8h  = len(both) / total_8h  * 100
        overlap_pct_q50 = len(both) / total_q50 * 100
        print(f'  Overlap: {overlap_pct_8h:.1f}% of 8h signals also fired by q50 | {overlap_pct_q50:.1f}% of q50 signals also fired by 8h')

    print(f'\n--- Signal quality (actual MFE at 8h / 24h / 72h horizons) ---')
    quality(only_8h,  '8h only',  thresh)
    quality(only_q50, 'q50 only', thresh)
    quality(both,     'both',     thresh)

    # Correlation between predictions on bars where BOTH fire
    if len(both) > 10:
        corr = both['pred_8h'].corr(both['pred_q50'])
        print(f'\n  Pred correlation on joint bars: {corr:.3f}')

    # Per-pair breakdown (8h actual, same as before for brevity)
    print(f'\n--- Per-pair signal split (>= {thresh}p) ---')
    print(f'  {"Pair":<10} {"8h_only":>8} {"q50_only":>9} {"both":>6} {"total_8h":>9} {"total_q50":>10}  overlap_8h')
    print(f'  {"-"*70}')
    for pair in sorted(s['pair'].unique()):
        sp    = s[s['pair'] == pair]
        o8    = sp[ sp['fire_8h'] & ~sp['fire_q50']]
        oq    = sp[~sp['fire_8h'] &  sp['fire_q50']]
        bo    = sp[ sp['fire_8h'] &  sp['fire_q50']]
        t8    = len(o8) + len(bo)
        tq    = len(oq) + len(bo)
        ov    = len(bo) / t8 * 100 if t8 else 0
        print(f'  {pair:<10} {len(o8):>8} {len(oq):>9} {len(bo):>6} {t8:>9} {tq:>10}  {ov:.0f}%')

    # Year-by-year with all horizons
    print(f'\n--- Year-by-year hit rates (>= {thresh}p) ---')
    s2 = s.copy(); s2['year'] = s2['ts'].dt.year
    only_8h2  = only_8h.copy();  only_8h2['year']  = only_8h2['ts'].dt.year
    only_q502 = only_q50.copy(); only_q502['year'] = only_q502['ts'].dt.year
    both2     = both.copy();     both2['year']     = both2['ts'].dt.year
    for yr in sorted(s2['year'].unique()):
        s8 = only_8h2[only_8h2['year']==yr]; sq = only_q502[only_q502['year']==yr]; sb = both2[both2['year']==yr]
        t8 = pd.concat([s8, sb]); tq = pd.concat([sq, sb])
        print(f'  Year {yr}  8h_N={len(t8):>4}  q50_N={len(tq):>4}  both_N={len(sb):>3}')
        for hz in HORIZONS:
            h8 = (t8[f'best_{hz}h'] >= thresh).mean() if len(t8) else float('nan')
            hq = (tq[f'best_{hz}h'] >= thresh).mean() if len(tq) else float('nan')
            print(f'    [{hz:>2}h]  8h_hit={h8:.1%}  q50_hit={hq:.1%}')

print(f'\n{"="*80}')
print('Done.')
