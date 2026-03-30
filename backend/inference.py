"""
Model inference v6.0 — 3 LightGBM quantile models (Q25/Q50/Q75)
+ 1 meta-model binary classifier for trade quality filtering.

Direction = sign(Q50), trade quality = meta_proba.
Two-path signal: meta path (P>0.50 + Q50>0.7x spread) OR high conviction bypass (Q50>1.0x spread).
"""

import joblib
from pathlib import Path

MODELS_DIR = Path("/app/models")

AVG_SPREAD = 0.00028        # ~2.8 pips average spread
MIN_Q50_THRESHOLD = AVG_SPREAD * 0.7   # 0.000196
META_THRESHOLD = 0.50
HIGH_CONV_THRESHOLD = AVG_SPREAD * 1.0  # 0.00028 — bypass meta if Q50 exceeds this


class Predictor:
    """Loads quantile + meta models once, runs inference on demand."""

    def __init__(self):
        self.quant_models = {}   # {'Q25': model, 'Q50': model, 'Q75': model}
        self.meta_model = None
        self.feature_cols = None
        self.meta_feature_cols = None
        self._load_models()

    def _load_models(self):
        # Load 3 quantile models
        for q_name in ['Q25', 'Q50', 'Q75']:
            q_int = int(q_name[1:])
            path = MODELS_DIR / f'model_1H_Q{q_int}.joblib'
            bundle = joblib.load(path)
            self.quant_models[q_name] = bundle['model']
            if self.feature_cols is None:
                self.feature_cols = bundle['feature_cols']

        # Load meta-model
        meta_path = MODELS_DIR / 'meta_confidence.joblib'
        meta_bundle = joblib.load(meta_path)
        self.meta_model = meta_bundle['model']
        self.meta_feature_cols = meta_bundle['meta_feature_cols']

    def predict(self, features_df, pair: str) -> dict:
        """
        Run inference for one pair: 3 quantile predictions + meta probability.

        Replicates the backtest logic exactly:
        1. Run Q25/Q50/Q75 on microstructure features
        2. Build derived columns (Q50_oof, Q25_oof, Q75_oof, abs_Q50, iqr, conf_ratio)
        3. Run meta-model on microstructure features + derived columns

        Args:
            features_df: DataFrame with microstructure features, at least 1 row.
                         Uses the last row (most recent hour).
            pair: e.g. "EURUSD"

        Returns:
            dict with quantile predictions, direction, meta_proba, and trade signal.
        """
        X = features_df[self.feature_cols].ffill().fillna(0)
        latest = X.iloc[[-1]]

        # Run 3 quantile models (no force-sort — matches backtest exactly)
        q_preds = {}
        for q_name in ['Q25', 'Q50', 'Q75']:
            model = self.quant_models[q_name]
            q_preds[q_name] = float(model.predict(latest)[0])

        q50 = q_preds['Q50']
        q25 = q_preds['Q25']
        q75 = q_preds['Q75']

        # Direction from sign of Q50
        if q50 > 0:
            direction = 'bullish'
        elif q50 < 0:
            direction = 'bearish'
        else:
            direction = 'neutral'

        abs_q50 = abs(q50)
        iqr = q75 - q25

        # Build meta-model input: microstructure features + derived quantile columns
        # This replicates the backtest's run_inference() logic exactly
        meta_row = latest.copy()
        meta_row['Q50_oof'] = q50
        meta_row['Q25_oof'] = q25
        meta_row['Q75_oof'] = q75
        meta_row['abs_Q50'] = abs_q50
        meta_row['iqr'] = iqr
        meta_row['conf_ratio'] = abs_q50 / max(iqr, 1e-10)

        X_meta = meta_row[self.meta_feature_cols].fillna(0)
        meta_proba = float(self.meta_model.predict_proba(X_meta)[0, 1])

        # Two-path signal (matches simulation config exactly):
        # Path 1 (meta): meta_proba > 0.50 AND |Q50| > 0.7x spread
        # Path 2 (bypass): |Q50| > 1.0x spread — fires regardless of meta
        meta_path = meta_proba > META_THRESHOLD and abs_q50 > MIN_Q50_THRESHOLD
        high_conv_path = abs_q50 > HIGH_CONV_THRESHOLD
        is_tradeable = meta_path or high_conv_path

        return {
            'pair': pair,
            'direction': direction,
            'q25': round(q25, 8),
            'q50': round(q50, 8),
            'q75': round(q75, 8),
            'meta_proba': round(meta_proba, 4),
            'is_tradeable': is_tradeable,
            'abs_q50': round(abs_q50, 8),
            'spread': round(iqr, 8),
        }
