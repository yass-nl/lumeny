import pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')
from pathlib import Path

PIP_SIZE = {'AUDJPY':0.01,'AUDNZD':0.0001,'AUDUSD':0.0001,'CADJPY':0.01,'CHFJPY':0.01,
    'EURAUD':0.0001,'EURGBP':0.0001,'EURJPY':0.01,'EURUSD':0.0001,'NZDUSD':0.0001,'USDCAD':0.0001}
SPREAD_PIPS = {'AUDJPY':3.0,'AUDNZD':3.0,'AUDUSD':1.5,'CADJPY':3.0,'CHFJPY':3.0,
    'EURAUD':3.0,'EURGBP':1.5,'EURJPY':2.0,'EURUSD':1.0,'NZDUSD':2.0,'USDCAD':2.0}
MAX_1H_MOVE = {'AUDJPY':250,'AUDNZD':80,'AUDUSD':120,'CADJPY':250,'CHFJPY':250,
    'EURAUD':150,'EURGBP':100,'EURJPY':300,'EURUSD':150,'NZDUSD':100,'USDCAD':150}
F6 = Path('backend/data/features_6'); F8 = Path('backend/data/features_8')

# Load one slice per pair (keeps index unique within each slice)
pair_data = {}
for pair, pip in PIP_SIZE.items():
    df6 = pd.read_parquet(F6/f'{pair}_features.parquet')
    df8 = pd.read_parquet(F8/f'{pair}_geometric.parquet'); df8['pair'] = pair
    df6r = df6.reset_index(); df8r = df8.reset_index(); idx = df6r.columns[0]
    df8r = df8r.drop(columns=[c for c in df8r.columns if c in df6r.columns and c not in [idx,'pair']],errors='ignore')
    df = pd.merge(df6r,df8r,on=[idx,'pair'],how='inner').set_index(idx).sort_index()
    df1h = pd.read_parquet(f'backend/data/processed/{pair}_1H.parquet')
    if 'datetime' in df1h.columns: df1h = df1h.set_index('datetime')
    df1h.index = pd.to_datetime(df1h.index)
    df['close'] = df1h['close'].reindex(df.index)
    ret_1h = ((df1h['close'].shift(-1) - df1h['close']) / pip).abs().reindex(df.index)
    holiday = ~(((df.index.month==12)&(df.index.day.isin([24,25,26,31])))|((df.index.month==1)&(df.index.day.isin([1,2]))))
    df = df[holiday & (ret_1h <= MAX_1H_MOVE[pair])].copy()
    df = df.dropna(subset=['realized_skew','residual_12h','rv_zscore_24','kyle_lambda_delta_3h','vr_5','close'])
    for col in ['kyle_lambda_delta_3h','residual_12h','rv_zscore_24','realized_skew','vr_5']:
        df[f'{col}_r'] = df[col].rank(pct=True)
    pair_data[pair] = df

print(f'Loaded {len(pair_data)} pairs.')


def make_signal(d):
    sig_l = (d['kyle_lambda_delta_3h_r']>0.70)&(d['residual_12h_r']<0.25)&\
            (d['rv_zscore_24_r']>0.85)&(d['realized_skew_r']<0.30)&(d['vr_5_r']<0.30)
    sig_s = (d['kyle_lambda_delta_3h_r']<0.30)&(d['residual_12h_r']>0.75)&\
            (d['rv_zscore_24_r']>0.85)&(d['realized_skew_r']>0.70)&(d['vr_5_r']<0.30)
    sig = pd.Series(0, index=d.index)
    sig[sig_l] = 1; sig[sig_s] = -1
    return sig


def test_session(hold_h, entry_start, entry_end):
    max_entry = min(entry_end, 19 - hold_h)
    if max_entry < entry_start:
        return None, None

    rows = []
    for pair, pip in PIP_SIZE.items():
        sp = SPREAD_PIPS[pair]
        d  = pair_data[pair]
        d  = d[(d.index.hour >= entry_start) & (d.index.hour <= max_entry)].copy()
        if len(d) == 0: continue
        fwd = (d['close'].shift(-hold_h) - d['close']) / pip
        sig = make_signal(d)
        pnl = sig * fwd - sp * sig.abs()
        rows.append(pnl[sig != 0].dropna())

    if not rows: return None, None
    t = pd.concat(rows).sort_index()
    if len(t) < 20: return None, None
    nm   = (t.index.max()-t.index.min()).days/30
    sh   = (t.mean()/t.std())*np.sqrt(252*24/hold_h)
    cut  = t.index.max()-pd.DateOffset(months=18)
    r    = t[t.index>=cut]
    sh_r = (r.mean()/r.std())*np.sqrt(252*24/hold_h) if len(r)>5 and r.std()>0 else 0
    stats = (len(t), len(t)/nm, (t>0).mean(), t.mean(), sh,
             len(r), (r>0).mean(), r.mean(), sh_r, r.sum())
    return stats, t


print()
print('LONDON + NY SCAN  |  entry window / hold / exit all before 20 UTC')
print()
print(f'{"Hold":>5} {"Window":>8} {"Trades":>7} {"/mo":>5} {"WR":>7} {"EV":>8} {"Sh":>7} | {"18mTrd":>7} {"18mWR":>7} {"18mEV":>8} {"18mSh":>7} {"18mTot":>9}')
print('-'*102)

best = []
for hold_h in [1, 2, 3, 4, 6]:
    for es, ee, label in [
        (5, 16, '5-16'),
        (7, 16, '7-16'),
        (7, 12, '7-12'),
        (12, 16, '12-16'),
        (5,  9, '5-9'),
        (9, 16, '9-16'),
    ]:
        res, t_all = test_session(hold_h, es, ee)
        if res is None:
            continue
        n, npm, wr, ev, sh, nr, rwr, rev, rsh, rtot = res
        flag = ' <<<' if rev > 1.0 else ''
        print(f'{hold_h:>5}H {label:>8} {n:>7,} {npm:>5.0f} {wr:>7.1%} {ev:>+8.2f} {sh:>+7.2f} | {nr:>7,} {rwr:>7.1%} {rev:>+8.2f} {rsh:>+7.2f} {rtot:>+9.0f}{flag}')
        best.append((rev, hold_h, es, ee, label, res, t_all))

print()
print('='*102)
print('YEAR-BY-YEAR — top 5 by 18m EV')
best.sort(reverse=True)
for rev, hold_h, es, ee, label, res, t_all in best[:5]:
    n, npm, wr, ev, sh, nr, rwr, rrev, rsh, rtot = res
    print()
    print(f'Hold {hold_h}H | Entry {label} UTC | {n:,} trades ({npm:.0f}/mo) | WR {wr:.1%} | EV {ev:+.2f} | Sh {sh:+.2f} | 18m EV {rrev:+.2f} Sh {rsh:+.2f}')
    print(f'  {"Year":>6} {"Trades":>7} {"WR":>7} {"EV":>8} {"Total":>9}')
    for yr in sorted(t_all.index.year.unique()):
        yt = t_all[t_all.index.year==yr]
        marker = ' *' if yt.mean()>2 else (' -' if yt.mean()<-2 else '')
        print(f'  {yr:>6} {len(yt):>7,} {(yt>0).mean():>7.1%} {yt.mean():>+8.2f} {yt.sum():>+9.0f}{marker}')
