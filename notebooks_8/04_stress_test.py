"""
Stress Test — MFE + Directional Model Combined
================================================
Uses parquet test set (post 2024-06-30) only.

Pipeline:
  1. MFE Q50 >= MFE_THRESH          -> candidate bar
  2. direction = sign(Q50_dir)       -> no confidence threshold, sign only
  3. cooldown per pair               -> skip bar if a trade is still open on that pair
  4. result = mfe_in_direction - trail_stop_pips - spread

Compares against:
  - Always LONG  on same bars (same cooldown applied)
  - Always SHORT on same bars (same cooldown applied)
  - Random direction on same bars (same cooldown applied)
"""

import sys, joblib
import numpy as np
import pandas as pd
from pathlib import Path

FEATURES_DIR   = Path('../backend/data/features_9')
MFE_MODEL_PATH = Path('../backend/models_9/mfe_q50_8h/model_1H_Q50.joblib')
DIR_Q50_PATH = Path('../backend/models_9/dir_q50_8h/model_1H_Q50.joblib')
DIR_Q25_PATH = Path('../backend/models_9/dir_q50_8h/model_1H_Q25.joblib')
DIR_Q75_PATH = Path('../backend/models_9/dir_q50_8h/model_1H_Q75.joblib')

TRAIN_END  = '2024-06-30'
MFE_THRESH = 70.0
COOLDOWN_H = 8      # hours to lock out a pair after entry (matches mfe_8h horizon)

SPREAD_PIPS = {
    'EURUSD': 0.6, 'GBPUSD': 0.8, 'USDJPY': 1.0, 'USDCHF': 0.7,
    'AUDUSD': 0.6, 'USDCAD': 1.2, 'NZDUSD': 0.9,
    'EURJPY': 1.4, 'GBPJPY': 2.1, 'EURGBP': 0.7,
    'EURAUD': 2.1, 'AUDJPY': 1.5, 'CADJPY': 1.6, 'CHFJPY': 2.5, 'AUDNZD': 2.0,
}

# ── Load models ───────────────────────────────────────────────────────────────
print('Loading models...')
mfe_bundle   = joblib.load(MFE_MODEL_PATH)
mfe_model    = mfe_bundle['model']
feature_cols = mfe_bundle['feature_cols']
dir_bundle   = joblib.load(DIR_Q50_PATH)
dir_model    = dir_bundle['model']
dir_q25      = joblib.load(DIR_Q25_PATH)['model']
dir_q75      = joblib.load(DIR_Q75_PATH)['model']
print(f'  MFE: {mfe_bundle["n_iters"]} iters | Dir Q50: {dir_bundle["n_iters"]} iters')

# ── Load + predict ────────────────────────────────────────────────────────────
print('\nLoading features_9...')
dfs = [pd.read_parquet(f) for f in sorted(FEATURES_DIR.glob('*_features.parquet'))]
df  = pd.concat(dfs).sort_index()

df_test = df[df.index > TRAIN_END].copy()
print(f'  Test rows: {len(df_test):,}')

print('Running models...')
X = df_test[feature_cols].ffill().fillna(0)
df_test['q50_mfe'] = mfe_model.predict(X)
df_test['q50_dir'] = dir_model.predict(X)
df_test['q25_dir'] = dir_q25.predict(X)
df_test['q75_dir'] = dir_q75.predict(X)
df_test['spread_7525'] = df_test['q75_dir'] - df_test['q25_dir']   # total IQR width
df_test['upside_skew']  = df_test['q75_dir'] - df_test['q50_dir']  # upside above median
df_test['downside_skew']= df_test['q50_dir'] - df_test['q25_dir']  # downside below median

# ── Filter to valid MFE candidates ───────────────────────────────────────────
df_cands = df_test[
    (df_test['q50_mfe'] >= MFE_THRESH) &
    df_test['mfe_long_pips'].notna() &
    df_test['mfe_short_pips'].notna() &
    df_test['trail_stop_pips'].notna()
].copy()
df_cands = df_cands.copy()
df_cands['dir_signal']    = np.where(df_cands['q50_dir'] >= 0, 1, -1)
# q25/q75 are already columns in df_test which was used to build df_cands — just predict directly
X_cands = df_cands[feature_cols].ffill().fillna(0)
df_cands['q25_dir']       = dir_q25.predict(X_cands)
df_cands['q75_dir']       = dir_q75.predict(X_cands)
df_cands['upside_skew']   = df_cands['q75_dir'] - df_cands['q50_dir']
df_cands['downside_skew'] = df_cands['q50_dir'] - df_cands['q25_dir']
print(f'  Candidate bars: {len(df_cands):,}')

# ── Per-pair cooldown simulation ──────────────────────────────────────────────
# For each strategy variant, walk through candidates chronologically per pair.
# After taking a trade, lock that pair out for COOLDOWN_H hours.

def simulate(df_cands, direction_col, label, seed=42):
    """
    direction_col: column name with +1/-1 direction, or 'long'/'short'/'random'
    Returns DataFrame of taken trades.
    """
    rng = np.random.default_rng(seed)
    cooldown_until = {}   # pair -> timestamp when cooldown expires
    trades = []

    for ts, row in df_cands.iterrows():
        pair = row['pair']

        # Check cooldown
        if pair in cooldown_until and ts < cooldown_until[pair]:
            continue

        # Determine direction
        if direction_col == 'long':
            direction = 1
        elif direction_col == 'short':
            direction = -1
        elif direction_col == 'random':
            direction = 1 if rng.random() >= 0.5 else -1
        else:
            direction = int(row[direction_col])

        sp    = SPREAD_PIPS.get(pair, 1.5)
        trail = row['trail_stop_pips']
        mfe   = row['mfe_long_pips'] if direction == 1 else row['mfe_short_pips']
        result = mfe - trail - sp

        # Set cooldown
        cooldown_until[pair] = ts + pd.Timedelta(hours=COOLDOWN_H)

        trades.append({
            'ts':      ts,
            'pair':    pair,
            'dir':     direction,
            'q50_mfe': row['q50_mfe'],
            'q50_dir': row['q50_dir'],
            'result':  result,
            'win':     result > 0,
            'year':    ts.year,
            'month':   ts.to_period('M'),
            'hour':    ts.hour,
        })

    return pd.DataFrame(trades)


print('\nSimulating with cooldown...')
tr_dir    = simulate(df_cands, 'dir_signal', 'Dir model (sign only)')
tr_long   = simulate(df_cands, 'long',       'Always LONG')
tr_short  = simulate(df_cands, 'short',      'Always SHORT')
tr_random = simulate(df_cands, 'random',     'Random direction')

print(f'  Dir model trades  : {len(tr_dir):,}')
print(f'  Always LONG trades: {len(tr_long):,}')
print(f'  Random trades     : {len(tr_random):,}')


# ── Helper ────────────────────────────────────────────────────────────────────
def stats(tr, label=''):
    r  = tr['result']
    wr = r.gt(0).mean()
    avg = r.mean()
    wins = r[r > 0]; losses = r[r <= 0]
    pf = wins.sum() / abs(losses.sum()) if len(losses) > 0 else np.nan
    months = (tr['ts'].max() - tr['ts'].min()).days / 30
    per_m  = len(tr) / months if months > 0 else 0
    print(f'  {label:<48} N={len(tr):>5,} ({per_m:>4.0f}/mo)  WR={wr:.1%}  avg={avg:>+7.1f}p  PF={pf:.3f}')


# ── Baseline comparison ───────────────────────────────────────────────────────
print(f'\n{"="*80}')
print(f'  BASELINE COMPARISON  (MFE >= {MFE_THRESH}, cooldown={COOLDOWN_H}h, all 15 pairs)')
print(f'{"="*80}')
stats(tr_long,   'Always LONG')
stats(tr_short,  'Always SHORT')
stats(tr_random, 'Random direction')
stats(tr_dir,    'Dir model — sign of Q50_dir, no threshold')


# ── Per-pair ──────────────────────────────────────────────────────────────────
print(f'\n{"="*80}')
print(f'  PER-PAIR — Dir model vs Random  (all pairs, no exclusions)')
print(f'{"="*80}')
print(f'  {"Pair":<10} {"N_dir":>6} {"WR_dir":>8} {"Avg_dir":>9} {"PF_dir":>7}  |  {"N_rnd":>6} {"WR_rnd":>8} {"Avg_rnd":>9}  delta_avg')
print(f'  {"-"*80}')
for pair in sorted(tr_dir['pair'].unique()):
    d = tr_dir[tr_dir['pair'] == pair]['result']
    r = tr_random[tr_random['pair'] == pair]['result']
    wr_d = d.gt(0).mean(); avg_d = d.mean()
    wr_r = r.gt(0).mean(); avg_r = r.mean()
    wins = d[d>0]; losses = d[d<=0]
    pf = wins.sum()/abs(losses.sum()) if len(losses)>0 else np.nan
    delta = avg_d - avg_r
    flag = ' ***' if delta > 5 else (' ---' if delta < -5 else '')
    print(f'  {pair:<10} {len(d):>6,} {wr_d:>8.1%} {avg_d:>+9.2f} {pf:>7.3f}  |  '
          f'{len(r):>6,} {wr_r:>8.1%} {avg_r:>+9.2f}  {delta:>+9.2f}{flag}')


# ── Year-by-year ──────────────────────────────────────────────────────────────
print(f'\n{"="*80}')
print(f'  YEAR-BY-YEAR — Dir model')
print(f'{"="*80}')
print(f'  {"Year":<6} {"N":>6} {"WR":>7} {"Avg_pips":>10} {"Total_pips":>12}')
print(f'  {"-"*46}')
for year in sorted(tr_dir['year'].unique()):
    s = tr_dir[tr_dir['year'] == year]['result']
    print(f'  {year:<6} {len(s):>6,} {s.gt(0).mean():>7.1%} {s.mean():>+10.2f} {s.sum():>12.0f}')


# ── Month-by-month ────────────────────────────────────────────────────────────
print(f'\n{"="*80}')
print(f'  MONTH-BY-MONTH — Dir model vs Random')
print(f'{"="*80}')
print(f'  {"Month":<10} {"N":>5} {"WR":>7} {"Avg_dir":>9} {"Avg_rnd":>9} {"Cumul_pips":>12}')
print(f'  {"-"*58}')
cumul = 0
rnd_by_month = tr_random.groupby('month')['result'].mean()
for month in sorted(tr_dir['month'].unique()):
    s    = tr_dir[tr_dir['month'] == month]['result']
    cumul += s.sum()
    avg_r = rnd_by_month.get(month, np.nan)
    print(f'  {str(month):<10} {len(s):>5,} {s.gt(0).mean():>7.1%} {s.mean():>+9.2f} {avg_r:>+9.2f} {cumul:>12.0f}')


# ── Hour-by-hour ──────────────────────────────────────────────────────────────
print(f'\n{"="*80}')
print(f'  HOUR-BY-HOUR — Dir model vs Random  (sign accuracy = WR_dir vs WR_rnd)')
print(f'{"="*80}')
print(f'  {"Hour":>5} {"N":>5} {"WR_dir":>8} {"Avg_dir":>9} {"WR_rnd":>8} {"Avg_rnd":>9}  delta')
print(f'  {"-"*60}')
rnd_by_hour = tr_random.groupby('hour').agg(wr=('win','mean'), avg=('result','mean'))
for h in range(24):
    s = tr_dir[tr_dir['hour'] == h]
    if len(s) < 10: continue
    r     = s['result']
    wr_r  = rnd_by_hour.loc[h, 'wr']  if h in rnd_by_hour.index else np.nan
    avg_r = rnd_by_hour.loc[h, 'avg'] if h in rnd_by_hour.index else np.nan
    delta = r.mean() - avg_r
    print(f'  {h:>5} {len(s):>5,} {r.gt(0).mean():>8.1%} {r.mean():>+9.2f} {wr_r:>8.1%} {avg_r:>+9.2f}  {delta:>+.2f}')


# ── MFE threshold sweep ───────────────────────────────────────────────────────
print(f'\n{"="*80}')
print(f'  MFE THRESHOLD SWEEP — sign only, with cooldown')
print(f'{"="*80}')
print(f'  {"MFE_thresh":>10} {"N":>6} {"WR":>7} {"Avg_pips":>10} {"PF":>8}')
print(f'  {"-"*48}')
for mfe_t in [20, 30, 40, 50, 70]:
    sub = tr_dir[tr_dir['q50_mfe'] >= mfe_t]
    if len(sub) < 20: continue
    r  = sub['result']
    wr = r.gt(0).mean()
    wins = r[r>0]; losses = r[r<=0]
    pf = wins.sum()/abs(losses.sum()) if len(losses)>0 else np.nan
    print(f'  {mfe_t:>10} {len(sub):>6,} {wr:>7.1%} {r.mean():>+10.2f} {pf:>8.3f}')


# ── Always SHORT deep dive ────────────────────────────────────────────────────
print(f'\n{"="*80}')
print(f'  ALWAYS SHORT DEEP DIVE  (MFE >= {MFE_THRESH}, cooldown={COOLDOWN_H}h)')
print(f'{"="*80}')

print(f'\n--- Per-pair ---')
print(f'  {"Pair":<10} {"N":>6} {"WR":>7} {"Avg_pips":>10} {"PF":>8} {"Total_pips":>12}')
print(f'  {"-"*58}')
for pair in sorted(tr_short['pair'].unique()):
    s = tr_short[tr_short['pair'] == pair]['result']
    if len(s) < 3: continue
    wr = s.gt(0).mean(); avg = s.mean()
    wins = s[s>0]; losses = s[s<=0]
    pf = wins.sum()/abs(losses.sum()) if len(losses)>0 else np.nan
    print(f'  {pair:<10} {len(s):>6,} {wr:>7.1%} {avg:>+10.2f} {pf:>8.3f} {s.sum():>12.0f}')

print(f'\n--- Year-by-year ---')
print(f'  {"Year":<6} {"N":>6} {"WR":>7} {"Avg_pips":>10} {"Total_pips":>12}')
print(f'  {"-"*46}')
for year in sorted(tr_short['year'].unique()):
    s = tr_short[tr_short['year'] == year]['result']
    print(f'  {year:<6} {len(s):>6,} {s.gt(0).mean():>7.1%} {s.mean():>+10.2f} {s.sum():>12.0f}')

print(f'\n--- Month-by-month ---')
print(f'  {"Month":<10} {"N":>5} {"WR":>7} {"Avg_pips":>9} {"Cumul_pips":>12}')
print(f'  {"-"*48}')
cumul = 0
for month in sorted(tr_short['month'].unique()):
    s = tr_short[tr_short['month'] == month]['result']
    cumul += s.sum()
    print(f'  {str(month):<10} {len(s):>5,} {s.gt(0).mean():>7.1%} {s.mean():>+9.2f} {cumul:>12.0f}')

print(f'\n--- MFE threshold sweep (always short, with cooldown) ---')
print(f'  {"MFE_thresh":>10} {"N":>6} {"WR":>7} {"Avg_pips":>10} {"PF":>8}')
print(f'  {"-"*48}')
for mfe_t in [20, 30, 40, 50, 70, 100]:
    sub = df_cands[df_cands['q50_mfe'] >= mfe_t]
    if len(sub) < 20: continue
    r_list = []
    cooldown_until = {}
    for ts, row in sub.iterrows():
        pair = row['pair']
        if pair in cooldown_until and ts < cooldown_until[pair]: continue
        sp = SPREAD_PIPS.get(pair, 1.5)
        result = row['mfe_short_pips'] - row['trail_stop_pips'] - sp
        cooldown_until[pair] = ts + pd.Timedelta(hours=COOLDOWN_H)
        r_list.append(result)
    r = pd.Series(r_list).dropna()
    wr = r.gt(0).mean(); wins = r[r>0]; losses = r[r<=0]
    pf = wins.sum()/abs(losses.sum()) if len(losses)>0 else np.nan
    print(f'  {mfe_t:>10} {len(r):>6,} {wr:>7.1%} {r.mean():>+10.2f} {pf:>8.3f}')

# ── Asymmetric strategy: default SHORT, override LONG when dir model confident ─
print(f'\n{"="*80}')
print(f'  ASYMMETRIC STRATEGY: default=SHORT, override=LONG when Q50_dir >= threshold')
print(f'  (no cooldown conflict — same trade universe, just direction changes)')
print(f'{"="*80}')

print(f'\n--- Long threshold sweep ---')
print(f'  {"L_thresh_bps":>13} {"N_long":>7} {"N_short":>8} {"WR":>7} {"Avg_pips":>10} {"PF":>8}  vs_always_short')
print(f'  {"-"*72}')

base_short_avg = tr_short['result'].mean()

for long_thresh_bps in [0, 2, 4, 6, 8, 10, 15, 20, 30, 50]:
    long_thresh = long_thresh_bps / 10000
    # direction: LONG if q50_dir >= long_thresh, else SHORT
    results = []
    cooldown_until = {}
    rng2 = np.random.default_rng(0)
    for ts, row in df_cands.iterrows():
        pair = row['pair']
        if pair in cooldown_until and ts < cooldown_until[pair]: continue
        direction = 1 if row['q50_dir'] >= long_thresh else -1
        sp    = SPREAD_PIPS.get(pair, 1.5)
        trail = row['trail_stop_pips']
        mfe   = row['mfe_long_pips'] if direction == 1 else row['mfe_short_pips']
        result = mfe - trail - sp
        cooldown_until[pair] = ts + pd.Timedelta(hours=COOLDOWN_H)
        results.append({'dir': direction, 'result': result})
    res = pd.DataFrame(results)
    r = res['result']
    n_long  = (res['dir'] == 1).sum()
    n_short = (res['dir'] == -1).sum()
    wr  = r.gt(0).mean(); avg = r.mean()
    wins = r[r>0]; losses = r[r<=0]
    pf  = wins.sum()/abs(losses.sum()) if len(losses)>0 else np.nan
    vs  = avg - base_short_avg
    flag = ' ***' if vs > 5 else (' ---' if vs < -5 else '')
    print(f'  {long_thresh_bps:>13} {n_long:>7,} {n_short:>8,} {wr:>7.1%} {avg:>+10.2f} {pf:>8.3f}  {vs:>+.2f}{flag}')

print(f'\n  (always SHORT baseline: avg={base_short_avg:+.2f}p, WR={tr_short["result"].gt(0).mean():.1%})')

print(f'\n--- Per-pair at best long threshold (10 bps) ---')
long_thresh = 10 / 10000
print(f'  {"Pair":<10} {"N_L":>5} {"N_S":>5} {"WR":>7} {"Avg":>9}  vs_short')
print(f'  {"-"*52}')
cooldown_until = {}
pair_rows = {}
for ts, row in df_cands.iterrows():
    pair = row['pair']
    if pair in cooldown_until and ts < cooldown_until[pair]: continue
    direction = 1 if row['q50_dir'] >= long_thresh else -1
    sp    = SPREAD_PIPS.get(pair, 1.5)
    mfe   = row['mfe_long_pips'] if direction == 1 else row['mfe_short_pips']
    result = mfe - row['trail_stop_pips'] - sp
    cooldown_until[pair] = ts + pd.Timedelta(hours=COOLDOWN_H)
    if pair not in pair_rows: pair_rows[pair] = []
    pair_rows[pair].append({'dir': direction, 'result': result})

short_avg_by_pair = tr_short.groupby('pair')['result'].mean()
for pair in sorted(pair_rows.keys()):
    df_p = pd.DataFrame(pair_rows[pair])
    r = df_p['result']
    nl = (df_p['dir']==1).sum(); ns = (df_p['dir']==-1).sum()
    vs = r.mean() - short_avg_by_pair.get(pair, 0)
    flag = ' ***' if vs > 5 else (' ---' if vs < -5 else '')
    print(f'  {pair:<10} {nl:>5} {ns:>5} {r.gt(0).mean():>7.1%} {r.mean():>+9.2f}  {vs:>+.2f}{flag}')

# ── Q75-Q50 upside skew sweep ─────────────────────────────────────────────────
# Logic: default=SHORT, override to LONG only when:
#   Q50_dir >= 0  AND  (Q75_dir - Q50_dir) >= upside_thresh
# i.e. median is positive AND there's meaningful upside above the median
print(f'\n{"="*80}')
print(f'  Q75-Q50 UPSIDE SKEW SWEEP')
print(f'  default=SHORT, LONG when Q50_dir>=0 AND (Q75-Q50)>=upside_thresh')
print(f'{"="*80}')
print(f'  {"upside_thresh_bps":>18} {"N_long":>7} {"N_short":>8} {"WR":>7} {"Avg_pips":>10} {"PF":>8}  vs_short')
print(f'  {"-"*75}')

for upside_bps in [0, 2, 4, 6, 8, 10, 15, 20, 30, 50]:
    upside_thresh = upside_bps / 10000
    results = []
    cooldown_until = {}
    for ts, row in df_cands.iterrows():
        pair = row['pair']
        if pair in cooldown_until and ts < cooldown_until[pair]: continue
        go_long = (row['q50_dir'] >= 0) and (row['upside_skew'] >= upside_thresh)
        direction = 1 if go_long else -1
        sp    = SPREAD_PIPS.get(pair, 1.5)
        mfe   = row['mfe_long_pips'] if direction == 1 else row['mfe_short_pips']
        result = mfe - row['trail_stop_pips'] - sp
        cooldown_until[pair] = ts + pd.Timedelta(hours=COOLDOWN_H)
        results.append({'dir': direction, 'result': result})
    res = pd.DataFrame(results)
    r = res['result']
    n_long  = (res['dir'] == 1).sum()
    n_short = (res['dir'] == -1).sum()
    wr   = r.gt(0).mean()
    wins = r[r > 0]; losses = r[r <= 0]
    pf   = wins.sum() / abs(losses.sum()) if len(losses) > 0 else np.nan
    vs   = r.mean() - base_short_avg
    flag = ' ***' if vs > 5 else (' ---' if vs < -5 else '')
    print(f'  {upside_bps:>18} {n_long:>7,} {n_short:>8,} {wr:>7.1%} {r.mean():>+10.2f} {pf:>8.3f}  {vs:>+.2f}{flag}')

# ── Q50-Q25 downside skew sweep (short confidence) ───────────────────────────
# Logic: default=SHORT, override to LONG when Q50>=0 AND downside_skew < ds_thresh
# i.e. go long only when the downside risk is small (tight Q25-Q50 gap)
print(f'\n{"="*80}')
print(f'  Q50-Q25 DOWNSIDE SKEW SWEEP')
print(f'  default=SHORT, LONG when Q50_dir>=0 AND (Q50-Q25) < downside_thresh')
print(f'  (small downside gap = model confident the move stays positive)')
print(f'{"="*80}')
print(f'  {"ds_thresh_bps":>14} {"N_long":>7} {"N_short":>8} {"WR":>7} {"Avg_pips":>10} {"PF":>8}  vs_short')
print(f'  {"-"*72}')

for ds_bps in [5, 10, 15, 20, 30, 50, 100]:
    ds_thresh = ds_bps / 10000
    results = []
    cooldown_until = {}
    for ts, row in df_cands.iterrows():
        pair = row['pair']
        if pair in cooldown_until and ts < cooldown_until[pair]: continue
        go_long = (row['q50_dir'] >= 0) and (row['downside_skew'] < ds_thresh)
        direction = 1 if go_long else -1
        sp    = SPREAD_PIPS.get(pair, 1.5)
        mfe   = row['mfe_long_pips'] if direction == 1 else row['mfe_short_pips']
        result = mfe - row['trail_stop_pips'] - sp
        cooldown_until[pair] = ts + pd.Timedelta(hours=COOLDOWN_H)
        results.append({'dir': direction, 'result': result})
    res = pd.DataFrame(results)
    r = res['result']
    n_long  = (res['dir'] == 1).sum()
    n_short = (res['dir'] == -1).sum()
    wr   = r.gt(0).mean()
    wins = r[r > 0]; losses = r[r <= 0]
    pf   = wins.sum() / abs(losses.sum()) if len(losses) > 0 else np.nan
    vs   = r.mean() - base_short_avg
    flag = ' ***' if vs > 5 else (' ---' if vs < -5 else '')
    print(f'  {ds_bps:>14} {n_long:>7,} {n_short:>8,} {wr:>7.1%} {r.mean():>+10.2f} {pf:>8.3f}  {vs:>+.2f}{flag}')

# ── Combined: Q50>=thresh AND upside_skew>=thresh ────────────────────────────
print(f'\n{"="*80}')
print(f'  COMBINED SWEEP: Q50_dir >= q50_thresh AND upside_skew >= upside_thresh')
print(f'  default=SHORT, LONG only when both conditions met')
print(f'{"="*80}')
print(f'  {"q50_bps":>8} {"up_bps":>7} {"N_long":>7} {"N_short":>8} {"WR":>7} {"Avg_pips":>10} {"PF":>8}  vs_short')
print(f'  {"-"*72}')

for q50_bps, up_bps in [(0,5),(0,10),(3,5),(3,10),(6,5),(6,10),(6,15),(10,10),(10,15)]:
    q50_thresh = q50_bps / 10000
    up_thresh  = up_bps  / 10000
    results = []
    cooldown_until = {}
    for ts, row in df_cands.iterrows():
        pair = row['pair']
        if pair in cooldown_until and ts < cooldown_until[pair]: continue
        go_long = (row['q50_dir'] >= q50_thresh) and (row['upside_skew'] >= up_thresh)
        direction = 1 if go_long else -1
        sp    = SPREAD_PIPS.get(pair, 1.5)
        mfe   = row['mfe_long_pips'] if direction == 1 else row['mfe_short_pips']
        result = mfe - row['trail_stop_pips'] - sp
        cooldown_until[pair] = ts + pd.Timedelta(hours=COOLDOWN_H)
        results.append({'dir': direction, 'result': result})
    res = pd.DataFrame(results)
    r = res['result']
    n_long  = (res['dir'] == 1).sum()
    n_short = (res['dir'] == -1).sum()
    wr   = r.gt(0).mean()
    wins = r[r > 0]; losses = r[r <= 0]
    pf   = wins.sum() / abs(losses.sum()) if len(losses) > 0 else np.nan
    vs   = r.mean() - base_short_avg
    flag = ' ***' if vs > 5 else (' ---' if vs < -5 else '')
    print(f'  {q50_bps:>8} {up_bps:>7} {n_long:>7,} {n_short:>8,} {wr:>7.1%} {r.mean():>+10.2f} {pf:>8.3f}  {vs:>+.2f}{flag}')

print(f'  (always SHORT baseline: avg={base_short_avg:+.2f}p, WR={tr_short["result"].gt(0).mean():.1%})')

# ── Final summary ─────────────────────────────────────────────────────────────
print(f'\n{"="*80}')
print(f'  SUMMARY')
print(f'{"="*80}')
print(f'  MFE threshold : Q50 >= {MFE_THRESH} pips')
print(f'  Dir threshold : sign only (no |Q50_dir| cutoff)')
print(f'  Cooldown      : {COOLDOWN_H}h per pair after entry')
print(f'  Test period   : {tr_dir["ts"].min().date()} -> {tr_dir["ts"].max().date()}')
print()
stats(tr_long,   'Baseline: always LONG')
stats(tr_short,  'Baseline: always SHORT')
stats(tr_random, 'Baseline: random direction')
stats(tr_dir,    'Dir model: sign of Q50_dir, no threshold')
