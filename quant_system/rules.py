"""
Rule-Based Signal Engine
Per-hour normalized thresholds for gate + direction features.
No model involved — pure feature rules.
"""

import pandas as pd
import numpy as np
from pathlib import Path

THRESHOLD_PATH = Path('backend/models_6/hour_thresholds.parquet')

# Which percentile level to use as cutoff
# Can be overridden at runtime for parameter sweeps
DEFAULT_GATE_PERCENTILE      = 70  # volume_cv, epps_1m_15m
DEFAULT_DIRECTION_PERCENTILE = 70  # abs_momentum_shift, abs_accel_mean


def load_thresholds(path: Path = THRESHOLD_PATH) -> pd.DataFrame:
    """Load the precomputed per-hour threshold table."""
    return pd.read_parquet(path)


def apply_rules(
    df: pd.DataFrame,
    thresholds: pd.DataFrame,
    gate_pct: int = DEFAULT_GATE_PERCENTILE,
    dir_pct: int  = DEFAULT_DIRECTION_PERCENTILE,
) -> pd.DataFrame:
    """
    Apply per-hour rules to a feature DataFrame.

    Parameters
    ----------
    df : DataFrame with DatetimeIndex (hour granularity), must contain:
         volume_cv, epps_1m_15m, momentum_shift, accel_mean
    thresholds : output of load_thresholds()
    gate_pct : percentile level for gate features (e.g. 70)
    dir_pct  : percentile level for direction features (e.g. 70)

    Returns
    -------
    DataFrame with only signal rows, plus added columns:
        signal_direction : +1 (long) or -1 (short)
        rule_score       : number of conditions met (2 gate + 2 direction = max 4)
    """
    hours = df.index.hour

    # Pull per-hour cutoffs for each row
    vol_cv_cut   = thresholds.loc[hours, f'volume_cv_p{gate_pct}'].values
    epps_cut     = thresholds.loc[hours, f'epps_1m_15m_p{gate_pct}'].values
    mom_cut      = thresholds.loc[hours, f'abs_momentum_shift_p{dir_pct}'].values
    accel_cut    = thresholds.loc[hours, f'abs_accel_mean_p{dir_pct}'].values

    # Gate conditions
    gate_vol  = df['volume_cv'].values   > vol_cv_cut
    gate_epps = df['epps_1m_15m'].values > epps_cut

    # Direction conditions (magnitude)
    dir_mom   = df['momentum_shift'].abs().values > mom_cut
    dir_accel = df['accel_mean'].abs().values      > accel_cut

    # Direction agreement: both features must point the same way
    mom_sign   = np.sign(df['momentum_shift'].values)
    accel_sign = np.sign(df['accel_mean'].values)
    signs_agree = mom_sign == accel_sign

    # Full signal mask
    signal_mask = gate_vol & gate_epps & dir_mom & dir_accel & signs_agree

    out = df[signal_mask].copy()
    out['signal_direction'] = mom_sign[signal_mask].astype(int)
    out['rule_score'] = (
        gate_vol[signal_mask].astype(int) +
        gate_epps[signal_mask].astype(int) +
        dir_mom[signal_mask].astype(int) +
        dir_accel[signal_mask].astype(int)
    )

    return out
