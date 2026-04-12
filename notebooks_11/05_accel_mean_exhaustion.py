"""
Accel Mean Exhaustion Setups
=============================
Within each 72h directional window (MFE>=50, direction system, 7-20 UTC),
test all relevant exhaustion patterns using accel_mean.

Core idea: in a SHORT bias window, a counter-trend bounce shows as accel_mean
rising (positive). When that bounce exhausts, price resumes the short direction.
We try to catch the exhaustion point. Symmetric for LONG windows.

In all patterns, accel_mean is first SIGNED relative to direction:
  signed_accel = direction * accel_mean
  - signed_accel > 0  means price accelerating AGAINST the regime (counter-trend)
  - signed_accel < 0  means price accelerating WITH the regime

We look for counter-trend exhaustion: signed_accel was positive/growing, now fading.

Setups tested:
  A1: rises N bars -> turns down (1-bar reversal)
  A2: rises N bars -> flat 1 bar -> turns down
  A3: rises N bars -> turns down 2 consecutive bars
  B1: rises N bars -> goes flat (plateau = momentum died)
  B2: rises N bars -> flat M bars -> turns down
  C1: rising but each increment smaller than prev (decelerating rise)
  C2: 2nd derivative (delta of delta) turns negative while still rising
  D1: signed_accel crosses above rolling p75 -> overshoot -> entry
  D2: signed_accel zscore > threshold -> mean reversion entry
  E1: signed_accel positive for N+ consecutive bars -> duration exhaustion
  F1: A1/A2 pattern AND signed_accel above rolling median (strong enough)
  F2: A1/A2 pattern AND volume_ratio_24 below median (quiet exhaustion)

Parameters swept per setup:
  N (buildup bars): 2, 3, 4
  Flat tolerance: small delta relative to std
  Thresholds: p50, p75 of rolling accel distribution
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path

SCRIPT_DIR     = Path(__file__).parent
FEATURES_DIR   = SCRIPT_DIR / '../backend/data/features_9'
PROCESSED_DIR  = SCRIPT_DIR / '../backend/data/processed'
MFE_MODEL_PATH = SCRIPT_DIR / '../backend/models_9/mfe_q50/model_1H_Q50.joblib'

START_DATE     = '2024-10-11'
MFE_THRESH     = 50.0
HOURS_ALLOWED  = set(range(7, 21))
COOLDOWN_H     = 72
WINDOW_H       = 72
HORIZONS       = [4, 6, 8]
MIN_WIN_N      = 10   # min entries across windows to report a setup

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

# ── Load features_9 ──────────────────────────────────────────────────────────
print('Loading features_9...')
dfs = [pd.read_parquet(f) for f in sorted(FEATURES_DIR.glob('*_features.parquet'))]
df9 = pd.concat(dfs).sort_index()
df9 = df9[df9.index >= START_DATE].copy()

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

# ── Apply cooldown ────────────────────────────────────────────────────────────
print('Applying cooldown...')
cooldown_until = {}
signal_list = []
for ts, row in df9.sort_index().iterrows():
    p = row['pair']
    if p in cooldown_until and ts < cooldown_until[p]: continue
    cooldown_until[p] = ts + pd.Timedelta(hours=COOLDOWN_H)
    signal_list.append({'start_ts': ts, 'pair': p, 'direction': row['direction']})
print(f'  Signal windows: {len(signal_list):,}')

# ── Load 1H price data ────────────────────────────────────────────────────────
print('Loading 1H price data...')
price_data = {}
for pair in PAIRS_ALL:
    fpath = PROCESSED_DIR / f'{pair}_1H.parquet'
    if fpath.exists():
        price_data[pair] = pd.read_parquet(fpath).sort_index()

# ── Build per-window bar sequences with accel_mean ───────────────────────────
print('Building window bar sequences...')

# We need accel_mean at every bar, including bars NOT in features_9 filtered set
# So reload full df9 (before MFE/hour filter) for accel_mean lookup
dfs_full = [pd.read_parquet(f) for f in sorted(FEATURES_DIR.glob('*_features.parquet'))]
df_full  = pd.concat(dfs_full).sort_index()
df_full  = df_full[df_full.index >= START_DATE][['pair', 'accel_mean', 'volume_ratio_24']].copy()

# Build per-window sequences
windows = []   # list of dicts: each is one window with bar-by-bar arrays

for sig in signal_list:
    start_ts  = sig['start_ts']
    pair      = sig['pair']
    direction = sig['direction']
    end_ts    = start_ts + pd.Timedelta(hours=WINDOW_H)
    pip       = 0.01 if pair in JPY_PAIRS else 0.0001

    if pair not in price_data:
        continue

    bars = price_data[pair]
    # All bars within window (hours 7-20 only)
    window_bars = bars[(bars.index >= start_ts) & (bars.index < end_ts)]
    window_bars = window_bars[pd.to_datetime(window_bars.index).hour.isin(HOURS_ALLOWED)]

    if len(window_bars) < 3:
        continue

    # Get accel_mean for this pair at these timestamps
    feat_pair = df_full[df_full['pair'] == pair]

    bar_data = []
    for ts, bar in window_bars.iterrows():
        # accel_mean at this bar
        if ts in feat_pair.index:
            accel = feat_pair.loc[ts, 'accel_mean']
            if isinstance(accel, pd.Series): accel = accel.iloc[0]
            vol_ratio = feat_pair.loc[ts, 'volume_ratio_24']
            if isinstance(vol_ratio, pd.Series): vol_ratio = vol_ratio.iloc[0]
        else:
            accel = np.nan
            vol_ratio = np.nan

        # Forward returns at each horizon
        fwds = {}
        for h in HORIZONS:
            future = bars[bars.index > ts].head(h)
            if len(future) == h:
                fwds[h] = direction * (future['close'].iloc[-1] - bar['close']) / pip
            else:
                fwds[h] = np.nan

        bar_data.append({
            'ts':         ts,
            'accel':      float(accel) if not isinstance(accel, float) else accel,
            'vol_ratio':  float(vol_ratio) if not isinstance(vol_ratio, float) else vol_ratio,
            **{f'fwd_{h}h': fwds[h] for h in HORIZONS},
        })

    windows.append({
        'start_ts':  start_ts,
        'pair':      pair,
        'direction': direction,
        'bars':      bar_data,
    })

print(f'  Windows built: {len(windows):,}')
total_bars = sum(len(w['bars']) for w in windows)
print(f'  Total bars: {total_bars:,}')

# ── Setup detection functions ─────────────────────────────────────────────────
# Each function takes a list of bar dicts (the window's bars up to and including
# the current bar index i) and returns True if the setup fires at bar i.
# accel is already raw — we sign it per direction inside each function.
# Counter-trend = direction * accel < 0  (accel going against regime)
# Wait: we want to fade counter-trend exhaustion.
# Counter-trend bounce = signed_accel = -direction * accel > 0
# i.e. in SHORT window (dir=-1): accel > 0 = counter-trend bounce
#      in LONG  window (dir=+1): accel < 0 = counter-trend dip
# signed_ct = -direction * accel  (positive = counter-trend move)

def get_ct(bars, i, direction):
    """Counter-trend accel at bar i (positive = price moving against regime)."""
    a = bars[i]['accel']
    if np.isnan(a): return np.nan
    return -direction * a   # positive means counter-trend

def get_ct_series(bars, i, direction, n):
    """Last n counter-trend values ending at bar i (inclusive)."""
    vals = []
    for j in range(max(0, i-n+1), i+1):
        vals.append(get_ct(bars, j, direction))
    return vals

def is_rising(vals):
    """All consecutive differences positive."""
    return all(vals[k+1] > vals[k] for k in range(len(vals)-1))

def is_flat(v1, v2, tol):
    """Two values are within tolerance of each other."""
    return abs(v1 - v2) <= tol

def rolling_std(bars, i, n=20):
    """Rolling std of accel over last n bars."""
    vals = [bars[j]['accel'] for j in range(max(0, i-n+1), i+1) if not np.isnan(bars[j]['accel'])]
    return np.std(vals) if len(vals) > 3 else 1e-6

def rolling_percentile(bars, i, pct, n=20):
    """Rolling percentile of counter-trend accel over last n bars."""
    vals = []
    for j in range(max(0, i-n+1), i+1):
        v = bars[j]['accel']
        if not np.isnan(v): vals.append(abs(v))
    return np.percentile(vals, pct) if len(vals) > 3 else 0.0

def rolling_zscore(bars, i, direction, n=20):
    """Z-score of current counter-trend accel vs last n bars."""
    ct_vals = []
    for j in range(max(0, i-n+1), i+1):
        v = get_ct(bars, j, direction)
        if not np.isnan(v): ct_vals.append(v)
    if len(ct_vals) < 4: return np.nan
    mu, sig = np.mean(ct_vals[:-1]), np.std(ct_vals[:-1])
    if sig < 1e-10: return 0.0
    return (ct_vals[-1] - mu) / sig

# ── All setup detectors ───────────────────────────────────────────────────────
# Returns True/False at bar i given window bars list and direction

def setup_A1(bars, i, direction, n_up):
    """Rises n_up bars -> turns down (1-bar reversal)."""
    if i < n_up: return False
    ct = [get_ct(bars, j, direction) for j in range(i-n_up, i+1)]
    if any(np.isnan(v) for v in ct): return False
    rising_phase = is_rising(ct[:-1])     # first n_up bars rising
    reversal     = ct[-1] < ct[-2]        # last bar turned down
    in_ct_zone   = ct[-2] > 0             # was in counter-trend territory
    return rising_phase and reversal and in_ct_zone

def setup_A2(bars, i, direction, n_up, tol_factor=0.3):
    """Rises n_up bars -> flat 1 bar -> turns down."""
    if i < n_up + 1: return False
    ct = [get_ct(bars, j, direction) for j in range(i-n_up-1, i+1)]
    if any(np.isnan(v) for v in ct): return False
    std = rolling_std(bars, i)
    tol = std * tol_factor
    rising_phase = is_rising(ct[:-2])
    flat_bar     = is_flat(ct[-2], ct[-3], tol)
    reversal     = ct[-1] < ct[-2]
    in_ct_zone   = ct[-2] > 0
    return rising_phase and flat_bar and reversal and in_ct_zone

def setup_A3(bars, i, direction, n_up):
    """Rises n_up bars -> turns down 2 consecutive bars."""
    if i < n_up + 1: return False
    ct = [get_ct(bars, j, direction) for j in range(i-n_up-1, i+1)]
    if any(np.isnan(v) for v in ct): return False
    rising_phase   = is_rising(ct[:-2])
    two_down       = ct[-1] < ct[-2] < ct[-3]
    in_ct_zone     = ct[-3] > 0
    return rising_phase and two_down and in_ct_zone

def setup_B1(bars, i, direction, n_up, tol_factor=0.3):
    """Rises n_up bars -> flat (plateau = momentum died). Entry at first flat bar."""
    if i < n_up: return False
    ct = [get_ct(bars, j, direction) for j in range(i-n_up, i+1)]
    if any(np.isnan(v) for v in ct): return False
    std = rolling_std(bars, i)
    tol = std * tol_factor
    rising_phase = is_rising(ct[:-1])
    plateau      = is_flat(ct[-1], ct[-2], tol)
    in_ct_zone   = ct[-1] > 0
    return rising_phase and plateau and in_ct_zone

def setup_B2(bars, i, direction, n_up, n_flat, tol_factor=0.3):
    """Rises n_up bars -> flat n_flat bars -> turns down."""
    needed = n_up + n_flat + 1
    if i < needed - 1: return False
    ct = [get_ct(bars, j, direction) for j in range(i-needed+1, i+1)]
    if any(np.isnan(v) for v in ct): return False
    std = rolling_std(bars, i)
    tol = std * tol_factor
    rising_phase = is_rising(ct[:n_up])
    flat_phase   = all(is_flat(ct[n_up+k], ct[n_up+k-1], tol) for k in range(1, n_flat+1))
    reversal     = ct[-1] < ct[-2]
    in_ct_zone   = ct[n_up] > 0
    return rising_phase and flat_phase and reversal and in_ct_zone

def setup_C1(bars, i, direction, n_up):
    """Rising but each increment smaller than previous (decelerating rise). Entry when decel detected."""
    if i < n_up: return False
    ct = [get_ct(bars, j, direction) for j in range(i-n_up, i+1)]
    if any(np.isnan(v) for v in ct): return False
    if not all(v > 0 for v in ct): return False   # must stay in counter-trend zone
    diffs = [ct[k+1] - ct[k] for k in range(len(ct)-1)]
    decelerating = all(diffs[k] > 0 and diffs[k] < diffs[k-1] for k in range(1, len(diffs)))
    still_rising = diffs[-1] > 0   # still going up but slower
    return decelerating and still_rising and len(diffs) >= 2

def setup_C2(bars, i, direction):
    """2nd derivative (delta of delta) turns negative while accel still rising. Entry immediately."""
    if i < 2: return False
    ct = [get_ct(bars, j, direction) for j in range(i-2, i+1)]
    if any(np.isnan(v) for v in ct): return False
    d1 = ct[1] - ct[0]   # first delta
    d2 = ct[2] - ct[1]   # second delta
    still_rising = d2 > 0 and ct[-1] > 0   # still in counter-trend territory
    decel        = d2 < d1                  # but acceleration slowing
    return still_rising and decel and d1 > 0

def setup_D1(bars, i, direction, pct_thresh=75):
    """Counter-trend accel crosses above rolling percentile threshold -> overshoot."""
    if i < 2: return False
    ct_prev = get_ct(bars, i-1, direction)
    ct_curr = get_ct(bars, i,   direction)
    if np.isnan(ct_prev) or np.isnan(ct_curr): return False
    threshold = rolling_percentile(bars, i, pct_thresh)
    crossed_above = ct_prev <= threshold and ct_curr > threshold
    return crossed_above

def setup_D2(bars, i, direction, z_thresh=1.5):
    """Counter-trend accel zscore exceeds threshold (overshoot -> mean reversion)."""
    if i < 5: return False
    z = rolling_zscore(bars, i, direction)
    if np.isnan(z): return False
    ct_curr = get_ct(bars, i, direction)
    return z > z_thresh and ct_curr > 0

def setup_E1(bars, i, direction, n_consec):
    """Counter-trend accel has been positive for n_consec consecutive bars -> duration exhaustion."""
    if i < n_consec - 1: return False
    ct = [get_ct(bars, j, direction) for j in range(i-n_consec+1, i+1)]
    if any(np.isnan(v) for v in ct): return False
    return all(v > 0 for v in ct)

def setup_F1(bars, i, direction, n_up):
    """A1 pattern AND current ct above rolling median (strong enough move)."""
    if not setup_A1(bars, i, direction, n_up): return False
    ct_curr = get_ct(bars, i-1, direction)  # peak bar
    threshold = rolling_percentile(bars, i, 50)
    return ct_curr > threshold

def setup_F2(bars, i, direction, n_up):
    """A1 pattern AND volume_ratio_24 below median (quiet exhaustion)."""
    if not setup_A1(bars, i, direction, n_up): return False
    vol = bars[i].get('vol_ratio', np.nan)
    if np.isnan(vol): return False
    # compute rolling median of volume
    vols = [bars[j]['vol_ratio'] for j in range(max(0,i-19), i+1) if not np.isnan(bars[j].get('vol_ratio', np.nan))]
    if len(vols) < 4: return False
    return vol < np.median(vols)

# ── Define all setups to test ─────────────────────────────────────────────────
SETUPS = []

for n in [2, 3, 4]:
    SETUPS.append((f'A1_n{n}',  lambda bars, i, d, n=n: setup_A1(bars, i, d, n)))
    SETUPS.append((f'A2_n{n}',  lambda bars, i, d, n=n: setup_A2(bars, i, d, n)))
    SETUPS.append((f'A3_n{n}',  lambda bars, i, d, n=n: setup_A3(bars, i, d, n)))
    SETUPS.append((f'B1_n{n}',  lambda bars, i, d, n=n: setup_B1(bars, i, d, n)))
    SETUPS.append((f'E1_n{n}',  lambda bars, i, d, n=n: setup_E1(bars, i, d, n)))
    SETUPS.append((f'F1_n{n}',  lambda bars, i, d, n=n: setup_F1(bars, i, d, n)))
    SETUPS.append((f'F2_n{n}',  lambda bars, i, d, n=n: setup_F2(bars, i, d, n)))

for n in [2, 3]:
    for m in [1, 2]:
        SETUPS.append((f'B2_n{n}_m{m}', lambda bars, i, d, n=n, m=m: setup_B2(bars, i, d, n, m)))

SETUPS.append(('C1_n3', lambda bars, i, d: setup_C1(bars, i, d, 3)))
SETUPS.append(('C1_n4', lambda bars, i, d: setup_C1(bars, i, d, 4)))
SETUPS.append(('C2',    lambda bars, i, d: setup_C2(bars, i, d)))
SETUPS.append(('D1_p75', lambda bars, i, d: setup_D1(bars, i, d, 75)))
SETUPS.append(('D1_p90', lambda bars, i, d: setup_D1(bars, i, d, 90)))
SETUPS.append(('D2_z15', lambda bars, i, d: setup_D2(bars, i, d, 1.5)))
SETUPS.append(('D2_z20', lambda bars, i, d: setup_D2(bars, i, d, 2.0)))

print(f'  Total setups to test: {len(SETUPS)}')

# ── Run all setups across all windows ────────────────────────────────────────
print('Running setup scan...')
results = {name: {'entries': [], 'windows_hit': set()} for name, _ in SETUPS}

for w_idx, win in enumerate(windows):
    bars      = win['bars']
    direction = win['direction']
    n_bars    = len(bars)

    for i in range(2, n_bars):   # need at least 2 prior bars
        for name, detector in SETUPS:
            try:
                fired = detector(bars, i, direction)
            except Exception:
                fired = False
            if fired:
                bar = bars[i]
                # Only record if we have valid forward returns
                entry = {f'fwd_{h}h': bar[f'fwd_{h}h'] for h in HORIZONS}
                entry['window'] = w_idx
                if not any(np.isnan(v) for v in entry.values()):
                    results[name]['entries'].append(entry)
                    results[name]['windows_hit'].add(w_idx)

# ── Print results ──────────────────────────────────────────────────────────────
SEP = '=' * 80
n_windows    = len(windows)
baseline     = {h: np.mean([b[f'fwd_{h}h'] for w in windows for b in w['bars']
                             if not np.isnan(b[f'fwd_{h}h'])]) for h in HORIZONS}
baseline_acc = {h: np.mean([1 if b[f'fwd_{h}h'] > 0 else 0
                             for w in windows for b in w['bars']
                             if not np.isnan(b[f'fwd_{h}h'])]) for h in HORIZONS}

print(f'\n{SEP}')
print(f'  ACCEL MEAN EXHAUSTION SETUPS  —  {n_windows} windows  |  MFE>=50  |  7-20 UTC')
print(f'{SEP}')
print(f'\n  Baselines (all intraday bars):')
for h in HORIZONS:
    print(f'    @{h}h: acc={baseline_acc[h]:.1%}  avg_pip={baseline[h]:+.2f}p')

print(f'\n  {"Setup":<16}  {"N":>5}  {"Win":>5}  {"Cov":>5}', end='')
for h in HORIZONS:
    print(f'  {"Acc@"+str(h)+"h":>7}  {"Pip@"+str(h)+"h":>8}', end='')
print(f'  {"Lift@6h":>8}')
print(f'  {"-"*100}')

setup_rows = []
for name, _ in SETUPS:
    entries = results[name]['entries']
    n       = len(entries)
    if n < MIN_WIN_N:
        continue
    n_win   = len(results[name]['windows_hit'])
    cov     = n_win / n_windows

    accs, pips = {}, {}
    for h in HORIZONS:
        fwds   = [e[f'fwd_{h}h'] for e in entries]
        accs[h] = np.mean([1 if f > 0 else 0 for f in fwds])
        pips[h] = np.mean(fwds)

    lift6 = accs[6] - baseline_acc[6]
    setup_rows.append((name, n, n_win, cov, accs, pips, lift6))

# Sort by accuracy @6h descending
setup_rows.sort(key=lambda x: -x[4][6])

for name, n, n_win, cov, accs, pips, lift6 in setup_rows:
    print(f'  {name:<16}  {n:>5}  {n_win:>5}  {cov:>5.1%}', end='')
    for h in HORIZONS:
        print(f'  {accs[h]:>7.1%}  {pips[h]:>+8.1f}p', end='')
    print(f'  {lift6:>+8.1%}')

# ── Per-horizon ranking ──────────────────────────────────────────────────────
print(f'\n{SEP}')
print(f'  BEST SETUPS BY CONSISTENCY (positive lift at ALL 3 horizons)')
print(f'{SEP}')
consistent = [(name, n, n_win, cov, accs, pips, lift6)
              for name, n, n_win, cov, accs, pips, lift6 in setup_rows
              if all(accs[h] > baseline_acc[h] for h in HORIZONS)]

print(f'\n  {"Setup":<16}  {"N":>5}  {"Cov":>5}', end='')
for h in HORIZONS:
    print(f'  {"Lift@"+str(h)+"h":>9}', end='')
print()
print(f'  {"-"*70}')
for name, n, n_win, cov, accs, pips, lift6 in consistent:
    print(f'  {name:<16}  {n:>5}  {cov:>5.1%}', end='')
    for h in HORIZONS:
        print(f'  {accs[h]-baseline_acc[h]:>+9.1%}', end='')
    print()

# ── Volume check: entries per window ────────────────────────────────────────
print(f'\n{SEP}')
print(f'  VOLUME CHECK  (avg entries per window, need >= 1 minimum)')
print(f'{SEP}')
print(f'\n  {"Setup":<16}  {"Total N":>8}  {"Avg/win":>8}  {"Max/win":>8}  {"0-entry wins":>13}  {"Acc@6h":>8}')
print(f'  {"-"*75}')

for name, n, n_win, cov, accs, pips, lift6 in setup_rows:
    per_win = [0] * n_windows
    for e in results[name]['entries']:
        per_win[e['window']] += 1
    avg_pw  = np.mean(per_win)
    max_pw  = max(per_win)
    zero_pw = sum(1 for x in per_win if x == 0)
    print(f'  {name:<16}  {n:>8}  {avg_pw:>8.1f}  {max_pw:>8}  {zero_pw:>10} ({zero_pw/n_windows:.0%})  {accs[6]:>8.1%}')


# ── Distribution analysis for best setups ────────────────────────────────────
# For each best setup, show the FULL distribution of outcomes:
# percentiles, winner avg, loser avg, MFE proxy (best of 4/6/8h), MAE proxy
print(f'\n{SEP}')
print(f'  DISTRIBUTION ANALYSIS — best setups vs baseline')
print(f'  (Does the edge come from bigger winners, smaller losers, or just noise?)')
print(f'{SEP}')

# Baseline distribution at 6h
all_fwd6 = [b[f'fwd_6h'] for w in windows for b in w['bars'] if not np.isnan(b['fwd_6h'])]
all_fwd6 = np.array(all_fwd6)

def dist_stats(fwds):
    fwds = np.array(fwds)
    wins  = fwds[fwds > 0]
    loses = fwds[fwds < 0]
    return {
        'n':        len(fwds),
        'acc':      (fwds > 0).mean(),
        'mean':     fwds.mean(),
        'p10':      np.percentile(fwds, 10),
        'p25':      np.percentile(fwds, 25),
        'p50':      np.percentile(fwds, 50),
        'p75':      np.percentile(fwds, 75),
        'p90':      np.percentile(fwds, 90),
        'win_avg':  wins.mean()  if len(wins)  > 0 else np.nan,
        'los_avg':  loses.mean() if len(loses) > 0 else np.nan,
        'rr':       abs(wins.mean() / loses.mean()) if len(wins) > 0 and len(loses) > 0 else np.nan,
    }

def print_dist(label, stats):
    print(f'\n  {label}')
    print(f'    N={stats["n"]:,}  acc={stats["acc"]:.1%}  mean={stats["mean"]:+.1f}p')
    print(f'    Percentiles: p10={stats["p10"]:+.1f}  p25={stats["p25"]:+.1f}  '
          f'p50={stats["p50"]:+.1f}  p75={stats["p75"]:+.1f}  p90={stats["p90"]:+.1f}')
    print(f'    Winners avg: {stats["win_avg"]:+.1f}p  |  Losers avg: {stats["los_avg"]:+.1f}p  |  R:R={stats["rr"]:.2f}')

print_dist('BASELINE (all bars @6h)', dist_stats(all_fwd6))

# Best setups
best_setups = ['D1_p75', 'D1_p90', 'D2_z20', 'E1_n4', 'F2_n2', 'E1_n3', 'D2_z15']
for name in best_setups:
    if name not in results or len(results[name]['entries']) < MIN_WIN_N:
        continue
    fwds = [e['fwd_6h'] for e in results[name]['entries']]
    print_dist(f'{name}  @6h', dist_stats(fwds))

# Also show at 4h for the best ones (tighter horizon = cleaner signal?)
print(f'\n  --- Same setups at 4h horizon ---')
for name in ['D1_p90', 'D2_z20', 'E1_n4']:
    if name not in results or len(results[name]['entries']) < MIN_WIN_N:
        continue
    fwds = [e['fwd_4h'] for e in results[name]['entries']]
    print_dist(f'{name}  @4h', dist_stats(fwds))

# ── Spread-adjusted P&L ───────────────────────────────────────────────────────
print(f'\n{SEP}')
print(f'  SPREAD-ADJUSTED  (entry + exit spread cost per pair)')
print(f'  Using realistic spreads: JPY pairs 2.0p avg, others 0.8p avg')
print(f'{SEP}')

SPREAD_AVG_JPY   = 2.0
SPREAD_AVG_OTHER = 0.8

# Per-entry spread cost
def get_spread(pair):
    JPY = {'USDJPY','EURJPY','GBPJPY','AUDJPY','CADJPY','CHFJPY'}
    return SPREAD_AVG_JPY if pair in JPY else SPREAD_AVG_OTHER

for name in ['D1_p75', 'D1_p90', 'D2_z20', 'E1_n4']:
    if name not in results or len(results[name]['entries']) < MIN_WIN_N:
        continue
    entries = results[name]['entries']
    for h in HORIZONS:
        net = []
        for e in entries:
            win_pair = windows[e['window']]['pair']
            sp = get_spread(win_pair)
            net.append(e[f'fwd_{h}h'] - sp)
        net = np.array(net)
        acc_net = (net > 0).mean()
        mean_net = net.mean()
        if h == 6:
            print(f'  {name} @{h}h net of spread:  acc={acc_net:.1%}  mean={mean_net:+.2f}p  '
                  f'(gross acc={dist_stats([e[f"fwd_{h}h"] for e in entries])["acc"]:.1%}  '
                  f'mean={dist_stats([e[f"fwd_{h}h"] for e in entries])["mean"]:+.2f}p)')

print(f'\n{SEP}')
print(f'  DONE')
print(f'{SEP}')
