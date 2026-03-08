"""
DIAGNOSTICS PART 3: The most critical check.
We check the actual 4H feature alignment at the parquet level.
The 4H features are produced by resampling from 1H data.
A serious form of leakage: if the 4H features at time T include
data from WITHIN the current 4H window (not just the last CLOSED bar),
that would mean features[T=15:00] contain price action from 12-15:00,
but the label is the move from 15:00-16:00.
The 12-15 move is partly explained by factors that also drive 15-16,
making the feature correlated with the label even without "future" data.
This is NOT traditional leakage, but it inflates correlation.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

FEATURES_DIR  = Path('backend/data/features')
MODELS_DIR    = Path('backend/models_2')
PROCESSED_DIR = Path('backend/data/processed')

# ===================================================================
# Check: What exactly is in log_ret_1_4H at each 1H bar?
# ===================================================================
print("="*65)
print("CHECK: 4H feature value at each 1H bar within a 4H window")
print("="*65)

df_4h = pd.read_parquet(PROCESSED_DIR / 'EURUSD_4H.parquet')
df_1h = pd.read_parquet(PROCESSED_DIR / 'EURUSD_1H.parquet')

# Compute log_ret_1 in 4H space
log_ret_1_4h_raw = np.log(df_4h['close'] / df_4h['close'].shift(1))

# Forward-fill to 1H index
log_ret_1_4h_on_1h = log_ret_1_4h_raw.reindex(df_1h.index, method='ffill')

# Load stored feature
df_all = pd.read_parquet(FEATURES_DIR / 'all_pairs_features_labels.parquet')
stored_feat = df_all[df_all['pair'] == 'EURUSD']['log_ret_1_4H']

# Compare stored vs our reconstruction
common = log_ret_1_4h_on_1h.index.intersection(stored_feat.index)
diff = (log_ret_1_4h_on_1h.loc[common] - stored_feat.loc[common]).abs()
print(f"\nReconstruction from raw 4H reindex+ffill vs stored:")
print(f"  Max diff: {diff.max():.2e}")
print(f"  Mean diff: {diff.mean():.2e}")
print(f"  Match: {'YES' if diff.max() < 1e-8 else 'NO - different computation!'}")

# Show a week of data to see the pattern
sample_week = stored_feat['2024-08-05':'2024-08-09']
close_1h_sample = df_1h['close']['2024-08-05':'2024-08-09']
close_4h_sample = df_4h['close']['2024-08-05':'2024-08-09']
label_1h_sample = df_all[df_all['pair'] == 'EURUSD']['label_1H']['2024-08-05':'2024-08-09']

print(f"\nSample week 2024-08-05 to 2024-08-09 (EURUSD):")
print(f"{'Time':<20} {'4H_close':>10} {'log_ret_1_4H':>14} {'1H_label':>10}")
print(f"{'-'*60}")
for ts in sample_week.index[:48]:
    feat_val = float(sample_week.get(ts, float('nan')))
    close_1h_val = float(close_1h_sample.get(ts, float('nan')))
    label_val = float(label_1h_sample.get(ts, float('nan'))) if ts in label_1h_sample.index else float('nan')
    is_4h_bar = ts in close_4h_sample.index
    marker = " <4H" if is_4h_bar else ""
    print(f"{str(ts):<20} {close_1h_val:>10.5f} {feat_val:>14.6f} {label_val:>10.6f}{marker}")

print()
print("KEY: The log_ret_1_4H value should be IDENTICAL for all 1H bars within a 4H window.")
print("If it changes at every 1H bar, the feature is NOT using reindex+ffill from 4H data,")
print("but rather the running 4H intrabar return (which WOULD be a form of lookahead).")

# ===================================================================
# Check: Is the 5m/15m feature engineering using resample('1h').last()?
# resample('1h').last() on 5m data: at 1H bar '15:00', this takes
# the last 5m bar in the 14:00-15:00 interval, which is 14:55.
# That's correct (past data). But if it uses '1H' convention='end',
# it could grab the bar at 15:00, which closes at 15:05.
# ===================================================================
print()
print("="*65)
print("CHECK: 5m/15m resample to 1H — is it using past data?")
print("="*65)

df_5m = pd.read_parquet(PROCESSED_DIR / 'EURUSD_5m.parquet') if (PROCESSED_DIR / 'EURUSD_5m.parquet').exists() else None
if df_5m is None:
    # Try alternative naming
    import glob
    candidates = list(Path('backend/data/processed').glob('EURUSD*.parquet'))
    print(f"Available EURUSD processed files: {[c.name for c in candidates]}")
else:
    log_ret_5m = np.log(df_5m['close'] / df_5m['close'].shift(1))
    resampled = log_ret_5m.resample('1h').last()
    print(f"\n5m log_ret resampled to 1H: first few bars")
    print(f"{'1H bar time':<22} {'5m bar time used':<22} {'value'}")
    print(f"{'-'*60}")
    # Check which 5m bar feeds each 1H bar
    for h_ts in resampled.index[:10]:
        val = float(resampled.loc[h_ts])
        # Find which 5m bar has this value
        window = log_ret_5m[h_ts - pd.Timedelta(hours=1):h_ts]
        if len(window) > 0 and not np.isnan(val):
            matches = window[abs(window - val) < 1e-10]
            source_ts = matches.index[-1] if len(matches) > 0 else 'unknown'
            print(f"{str(h_ts):<22} {str(source_ts):<22} {val:.6f}")

print()

# ===================================================================
# Check: Does the parquet have intrabar data from the SAME 1H bar
# (i.e., features for bar T include data from within bar T)?
# This is the subtle form: if features[T] include the 1H bar T's own OHLC,
# they're using information that's only known at the END of bar T,
# which coincides with the start of label_1H[T] = return from T to T+1.
# Actually this is fine — bar T closes at T, features[T] use bar T's close,
# and label[T] = return from T_close to T+1_close. No overlap.
# ===================================================================
print("="*65)
print("CHECK: Is the label correctly aligned? (close[T] vs close[T+1])")
print("="*65)

close_1h = df_1h['close']
stored_label = df_all[df_all['pair'] == 'EURUSD']['label_1H']

print("\nSample: close prices and labels around 2024-08-06:")
sample_ts = close_1h['2024-08-06 10:00':'2024-08-06 16:00']
print(f"{'Time':<22} {'close':>10} {'stored_label':>14} {'manual_label':>14} {'match':>6}")
print(f"{'-'*70}")
for ts in sample_ts.index:
    c = float(close_1h[ts])
    if ts + pd.Timedelta(hours=1) in close_1h.index:
        c_next = float(close_1h[ts + pd.Timedelta(hours=1)])
        manual = np.log(c_next / c)
    else:
        manual = float('nan')
    stored = float(stored_label[ts]) if ts in stored_label.index else float('nan')
    match = 'YES' if abs(manual - stored) < 1e-10 else 'NO'
    print(f"{str(ts):<22} {c:>10.5f} {stored:>14.6f} {manual:>14.6f} {match:>6}")

print()
print("SUMMARY: If close[T] is the bar that closes at timestamp T,")
print("then label[T] = log(close[T+1]/close[T]) = the return you'd earn")
print("if you entered at T_close and exited at (T+1)_close.")
print("Features[T] can legitimately use close[T] without leakage.")
