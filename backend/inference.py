"""
Model inference v8.0 — MFE Q50 LightGBM quantile model + rule-based direction system.

The MFE model predicts the dominant move (max of up/down excursion) over 8h,
completely direction-agnostic.  Direction is determined separately by pair-specific
rule-based thresholds applied on top.

Two outputs per bar:
  - is_signal : mfe_q50_pips >= MFE_THRESH (30 pips)
  - direction : +1 LONG / -1 SHORT / NaN if rule has no clear call for this bar

Both are logged regardless — dashboard shows all signals, direction-unconfirmed too.
"""

import numpy as np
import joblib
import pandas as pd
from pathlib import Path

MODELS_DIR = Path("/app/models")
MFE_THRESH = 30.0   # pips (model output is already in pips)


# ---------------------------------------------------------------------------
# Direction rules (pair-specific, rule-based — completely separate from MFE)
# Returns: +1 LONG, -1 SHORT, or NaN (no clear call)
# ---------------------------------------------------------------------------

def _direction_for_row(pair: str, row: pd.Series) -> float:
    """Apply pair-specific directional rule to a single feature row."""

    def get(col):
        v = row.get(col, np.nan)
        return float(v) if not pd.isna(v) else np.nan

    if pair == 'USDJPY':
        return -1.0

    if pair == 'AUDUSD':
        beta   = get('beta_gbpusd_1w')
        atr_24 = get('atr_24')
        if (not np.isnan(beta) and beta > 0.775) or (not np.isnan(atr_24) and atr_24 < 40.8):
            return 1.0
        return np.nan

    if pair == 'GBPUSD':
        csi = get('csi_usd_24h')
        if not np.isnan(csi) and csi < 0.004:
            return 1.0
        return np.nan

    if pair == 'EURUSD':
        corr = get('corr_audusd_24h')
        if not np.isnan(corr) and corr < 0.22:
            return 1.0
        return np.nan

    if pair == 'NZDUSD':
        dist = get('dist_5d_high')
        if not np.isnan(dist) and dist > 0.35:
            return 1.0
        return np.nan

    if pair == 'USDCHF':
        corr = get('corr_eurusd_1w')
        if not np.isnan(corr) and corr > -0.60:
            return 1.0
        return np.nan

    if pair == 'CHFJPY':
        corr = get('corr_usdjpy_1w')
        if np.isnan(corr):
            return np.nan
        if corr > 0.40:
            return 1.0
        if corr < 0.26:
            return -1.0
        return np.nan

    if pair == 'CADJPY':
        vt = get('vol_trend')
        if np.isnan(vt):
            return np.nan
        return 1.0 if vt < 1.15 else -1.0

    if pair == 'AUDJPY':
        beta = get('beta_usdjpy_1w')
        if not np.isnan(beta) and beta > 0.74:
            return 1.0
        return np.nan

    if pair == 'EURJPY':
        beta = get('beta_eurusd_1w')
        if not np.isnan(beta) and beta > 0.38:
            return 1.0
        return np.nan

    if pair == 'GBPJPY':
        beta = get('beta_eurusd_1w')
        if not np.isnan(beta) and beta > 0.50:
            return 1.0
        return np.nan

    if pair == 'EURAUD':
        corr = get('corr_audusd_24h')
        if not np.isnan(corr) and corr < 0.22:
            return 1.0
        return np.nan

    if pair == 'AUDNZD':
        corr = get('corr_regime_audusd')
        if not np.isnan(corr) and corr > 0.0:
            return 1.0
        return np.nan

    if pair == 'EURGBP':
        csi = get('csi_usd_24h')
        if not np.isnan(csi) and csi > 0.004:
            return -1.0
        return np.nan

    if pair == 'USDCAD':
        return np.nan   # no rule defined yet

    return np.nan


# ---------------------------------------------------------------------------
# Predictor
# ---------------------------------------------------------------------------

class Predictor:
    """
    Loads MFE Q50 model once at startup.  Runs per-pair prediction on demand.

    cross_features must be passed in from outside (computed across all pairs
    simultaneously) — see _build_cross_features() in paper_trading.py.
    """

    def __init__(self):
        self.mfe_model    = None
        self.feature_cols = None
        self._load_models()

    def _load_models(self):
        path   = MODELS_DIR / 'model_1H_Q50.joblib'
        bundle = joblib.load(path)
        self.mfe_model    = bundle['model']
        self.feature_cols = bundle['feature_cols']

    def predict(self, features_df: pd.DataFrame, pair: str) -> dict:
        """
        Run MFE prediction + direction rule for one pair.

        Args:
            features_df : DataFrame with all features, at least 1 row.
                          Must already include momentum/calendar cols
                          (from live_features_extra) and cross-pair cols.
                          Uses the LAST row (most recent closed hour).
            pair        : e.g. 'EURUSD'

        Returns dict:
            pair                  str
            mfe_q50_pips          float  — predicted dominant move in pips
            is_signal             bool   — mfe_q50_pips >= 70
            direction             int|None  — +1 LONG / -1 SHORT / None
            direction_label       str    — 'LONG' / 'SHORT' / 'NO_RULE'
        """
        # Align features to model columns
        X = features_df.reindex(columns=self.feature_cols).ffill().fillna(0)
        latest_row = X.iloc[-1]

        # Model output is already in pips (trained with pip-denominated target)
        mfe_pips = float(self.mfe_model.predict(latest_row.values.reshape(1, -1))[0])

        # Direction (rule-based, completely separate from MFE model)
        feat_row  = features_df.iloc[-1]
        direction = _direction_for_row(pair, feat_row)

        if np.isnan(direction):
            dir_int   = None
            dir_label = 'NO_RULE'
        else:
            dir_int   = int(direction)
            dir_label = 'LONG' if dir_int == 1 else 'SHORT'

        return {
            'pair':            pair,
            'mfe_q50_pips':    round(mfe_pips, 1),
            'is_signal':       bool(mfe_pips >= MFE_THRESH),
            'direction':       dir_int,
            'direction_label': dir_label,
        }
