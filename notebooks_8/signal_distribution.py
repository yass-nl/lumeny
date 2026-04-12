"""
MFE Signal Distribution
========================
Raw count of bars where q50_mfe >= 70, broken down by pair.
No cooldown, no direction, no simulation — just how often the model fires per pair.
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path

SCRIPT_DIR     = Path(__file__).parent
FEATURES_DIR   = SCRIPT_DIR / '../backend/data/features_9'
MFE_MODEL_PATH = SCRIPT_DIR / '../backend/models_9/mfe_q50/model_1H_Q50.joblib'

TRAIN_END  = '2024-06-30'
MFE_THRESH = 70.0
COOLDOWN_H = 72

# ── Load model + features ─────────────────────────────────────────────────────
print('Loading MFE model...')
bundle       = joblib.load(MFE_MODEL_PATH)
mfe_model    = bundle['model']
feature_cols = bundle['feature_cols']

print('Loading features_9...')
dfs     = [pd.read_parquet(f) for f in sorted(FEATURES_DIR.glob('*_features.parquet'))]
df      = pd.concat(dfs).sort_index()
df_test = df[df.index > TRAIN_END].copy()
print(f'  Test rows: {len(df_test):,}  |  period: {df_test.index.min().date()} to {df_test.index.max().date()}')

print('Running MFE model...')
X = df_test[feature_cols].ffill().fillna(0)
df_test['q50_mfe'] = mfe_model.predict(X)

# ── Apply cooldown per pair ───────────────────────────────────────────────────
candidates = df_test[df_test['q50_mfe'] >= MFE_THRESH].sort_index()
cooldown_until = {}
kept = []
for ts, row in candidates.iterrows():
    pair = row['pair']
    if pair in cooldown_until and ts < cooldown_until[pair]:
        continue
    cooldown_until[pair] = ts + pd.Timedelta(hours=COOLDOWN_H)
    kept.append(ts)

df_sig = candidates.loc[kept].copy()
total  = len(df_sig)
months = max((df_test.index.max() - df_test.index.min()).days / 30, 0.1)

print(f'\n  Total signal bars: {total:,}  ({total/months:.0f}/mo across all pairs)  [cooldown={COOLDOWN_H}h per pair]\n')

# ── Per-pair summary ──────────────────────────────────────────────────────────
summary = (
    df_sig.groupby('pair')
    .agg(
        N        = ('q50_mfe', 'count'),
        avg_score= ('q50_mfe', 'mean'),
        pct_80   = ('q50_mfe', lambda x: (x >= 80).mean()),
        pct_90   = ('q50_mfe', lambda x: (x >= 90).mean()),
    )
    .sort_values('N', ascending=False)
    .reset_index()
)
summary['pct_total'] = summary['N'] / total
summary['per_mo']    = summary['N'] / months

print(f'  {"Pair":<10} {"N":>6}  {"N/mo":>6}  {"% total":>8}  {"AvgScore":>9}  {">80%":>6}  {">90%":>6}  {"Bar chart"}')
print(f'  {"-"*80}')
for _, row in summary.iterrows():
    bar = '|' * int(row['pct_total'] * 100)
    print(f'  {row["pair"]:<10} {int(row["N"]):>6}  {row["per_mo"]:>6.1f}  '
          f'{row["pct_total"]:>8.1%}  {row["avg_score"]:>9.1f}  '
          f'{row["pct_80"]:>6.1%}  {row["pct_90"]:>6.1%}  {bar}')

# ── Monthly breakdown per pair ────────────────────────────────────────────────
print(f'\n  Monthly signal count per pair (raw):')
df_sig['month'] = pd.to_datetime(df_sig.index).to_period('M')
monthly = df_sig.groupby(['month', 'pair']).size().unstack(fill_value=0)
# Only show pairs with meaningful signal
top_pairs = summary.head(10)['pair'].tolist()
monthly_top = monthly[[p for p in top_pairs if p in monthly.columns]]

print(f'\n  {"Month":<10}', end='')
for p in monthly_top.columns:
    print(f'  {p:>8}', end='')
print(f'  {"TOTAL":>7}')
print(f'  {"-"*10}', end='')
for _ in monthly_top.columns:
    print(f'  {"--------":>8}', end='')
print(f'  {"-------":>7}')

for month, row in monthly_top.iterrows():
    total_month = monthly.loc[month].sum()
    print(f'  {str(month):<10}', end='')
    for p in monthly_top.columns:
        print(f'  {int(row[p]):>8}', end='')
    print(f'  {int(total_month):>7}')

# ── Hour of day distribution ─────────────────────────────────────────────────
print(f'\n  Signal count by hour of day (UTC):')
df_sig['hour'] = pd.to_datetime(df_sig.index).hour
df_sig_no21 = df_sig[~df_sig['hour'].isin([20, 21, 22])]
hourly = df_sig.groupby('hour').size()
hourly_pct = hourly / hourly.sum()

print(f'\n  {"Hour":>5}  {"N":>5}  {"N/mo":>6}  {"% total":>8}  Bar')
print(f'  {"-"*55}')
for h in range(24):
    n   = hourly.get(h, 0)
    pct = hourly_pct.get(h, 0.0)
    bar = '|' * int(pct * 200)
    session = ''
    if 7 <= h < 9:   session = ' <- London open'
    elif 9 <= h < 12: session = ' <- London'
    elif 12 <= h < 14: session = ' <- London/NY overlap'
    elif 14 <= h < 17: session = ' <- NY'
    elif 0 <= h < 3:  session = ' <- Tokyo'
    print(f'  {h:>5}  {n:>5}  {n/months:>6.1f}  {pct:>8.1%}  {bar}{session}')

# ── Hour distribution per pair (top 5) ───────────────────────────────────────
print(f'\n  Hour distribution by pair (top 6 pairs):')
top6 = summary.head(6)['pair'].tolist()
hour_pair = df_sig[df_sig['pair'].isin(top6)].groupby(['hour','pair']).size().unstack(fill_value=0)
hour_pair = hour_pair[[p for p in top6 if p in hour_pair.columns]]

print(f'\n  {"Hour":>5}', end='')
for p in hour_pair.columns:
    print(f'  {p:>8}', end='')
print()
print(f'  {"-----":>5}', end='')
for _ in hour_pair.columns:
    print(f'  {"--------":>8}', end='')
print()
for h in range(24):
    if h not in hour_pair.index:
        continue
    row = hour_pair.loc[h]
    if row.sum() == 0:
        continue
    print(f'  {h:>5}', end='')
    for p in hour_pair.columns:
        print(f'  {int(row[p]):>8}', end='')
    print()

# ── Without hours 20-22 ──────────────────────────────────────────────────────
n2 = len(df_sig_no21)
print(f'\n  Without hours 20-22 UTC: {n2:,} signals ({n2/months:.0f}/mo) — {n2/total:.1%} of total remaining\n')
summary2 = (
    df_sig_no21.groupby('pair')
    .agg(N=('q50_mfe','count'), avg_score=('q50_mfe','mean'))
    .sort_values('N', ascending=False)
    .reset_index()
)
summary2['pct'] = summary2['N'] / n2
print(f'  {"Pair":<10} {"N":>5}  {"N/mo":>6}  {"% total":>8}  {"AvgScore":>9}')
print(f'  {"-"*48}')
for _, row in summary2.iterrows():
    print(f'  {row["pair"]:<10} {int(row["N"]):>5}  {row["N"]/months:>6.1f}  {row["pct"]:>8.1%}  {row["avg_score"]:>9.1f}')

# ── Score distribution per pair ───────────────────────────────────────────────
print(f'\n  Score distribution (% of signals in each bucket):')
print(f'  {"Pair":<10}  {"70-75":>7}  {"75-80":>7}  {"80-85":>7}  {"85-90":>7}  {">90":>7}')
print(f'  {"-"*52}')
for _, row in summary.iterrows():
    pair = row['pair']
    s = df_sig[df_sig['pair'] == pair]['q50_mfe']
    b1 = ((s >= 70) & (s < 75)).mean()
    b2 = ((s >= 75) & (s < 80)).mean()
    b3 = ((s >= 80) & (s < 85)).mean()
    b4 = ((s >= 85) & (s < 90)).mean()
    b5 = (s >= 90).mean()
    print(f'  {pair:<10}  {b1:>7.1%}  {b2:>7.1%}  {b3:>7.1%}  {b4:>7.1%}  {b5:>7.1%}')
