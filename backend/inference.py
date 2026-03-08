"""
Model inference — loads 15 quantile models (v2, no calibrators),
runs predictions, and derives raw probabilities.
v2: trained on data <= 2024-06-30, no data leakage, no calibrators needed.
"""

import numpy as np
import joblib
from pathlib import Path

MODELS_DIR = Path("/app/models")

HORIZONS = ['1H', '4H', '1D']
QUANTILES = [0.10, 0.25, 0.50, 0.75, 0.90]
QUANTILE_NAMES = ['Q10', 'Q25', 'Q50', 'Q75', 'Q90']

class Predictor:
    """Loads all models once, runs inference on demand."""

    def __init__(self):
        self.models = {}       # {(horizon, quantile_name): model}
        self.feature_cols = None
        self._load_models()

    def _load_models(self):
        for horizon in HORIZONS:
            for q, q_name in zip(QUANTILES, QUANTILE_NAMES):
                path = MODELS_DIR / f'model_{horizon}_Q{int(q * 100)}.joblib'
                bundle = joblib.load(path)
                self.models[(horizon, q_name)] = bundle['model']
                if self.feature_cols is None:
                    self.feature_cols = bundle['feature_cols']

    @staticmethod
    def _derive_p_down(q_vals: np.ndarray) -> float:
        """
        Derive P(down) from 5 quantile predictions.
        Exact replica of notebook 04 derive_p_down().
        """
        qs = np.array(QUANTILES)
        vals = q_vals

        if vals[0] <= 0 <= vals[-1]:
            return float(np.interp(0, vals, qs))
        elif vals[-1] < 0:
            slope = (qs[-1] - qs[-2]) / (vals[-1] - vals[-2] + 1e-10)
            p_down = qs[-1] + slope * (0 - vals[-1])
            return float(np.clip(p_down, 0.90, 0.999))
        else:
            slope = (qs[1] - qs[0]) / (vals[1] - vals[0] + 1e-10)
            p_down = qs[0] + slope * (0 - vals[0])
            return float(np.clip(p_down, 0.001, 0.10))

    def predict(self, features_df, pair: str) -> dict:
        """
        Run full inference for one pair across all horizons.

        Args:
            features_df: DataFrame with all 208 features, at least 1 row.
                         Uses the last row (most recent candle).
            pair: e.g. "EURUSD"

        Returns:
            dict with per-horizon predictions including calibrated probabilities.
        """
        X = features_df[self.feature_cols].ffill().bfill()
        latest = X.iloc[[-1]]

        result = {'pair': pair, 'horizons': {}}

        for horizon in HORIZONS:
            # Run 5 quantile models
            q_preds = {}
            for q, q_name in zip(QUANTILES, QUANTILE_NAMES):
                model = self.models[(horizon, q_name)]
                q_preds[q_name] = float(model.predict(latest)[0])

            q_vals_raw = np.array([q_preds[n] for n in QUANTILE_NAMES])

            # Step 1: Force-sort to fix quantile crossing
            q_vals = np.sort(q_vals_raw)
            was_crossed = not np.array_equal(q_vals, q_vals_raw)

            # Update q_preds with sorted values
            for i, q_name in enumerate(QUANTILE_NAMES):
                q_preds[q_name] = float(q_vals[i])

            spread = float(q_vals[-1] - q_vals[0])

            # Derive P(down) — v2 models are well-calibrated, no calibrator needed
            cal_p_down = self._derive_p_down(q_vals)
            cal_p_up = 1.0 - cal_p_down

            # Direction and probability
            if cal_p_down > cal_p_up:
                direction = 'bearish'
                probability = cal_p_down
            else:
                direction = 'bullish'
                probability = cal_p_up

            if probability < 0.55:
                direction = 'neutral'

            # Signal strength
            if probability >= 0.80:
                signal_strength = 'very_strong'
            elif probability >= 0.70:
                signal_strength = 'strong'
            elif probability >= 0.60:
                signal_strength = 'moderate'
            else:
                signal_strength = 'weak'

            q50 = q_preds['Q50']

            result['horizons'][horizon] = {
                'direction': direction,
                'probability': round(probability, 4),
                'calibrated_p_down': round(cal_p_down, 4),
                'calibrated_p_up': round(cal_p_up, 4),
                'raw_p_down': round(cal_p_down, 4),
                'signal_strength': signal_strength,
                'expected_move_pct': round(q50 * 100, 4),
                'quantile_spread': round(spread * 100, 6),
                'quantiles': {
                    'Q10': round(q_preds['Q10'] * 100, 4),
                    'Q25': round(q_preds['Q25'] * 100, 4),
                    'Q50': round(q_preds['Q50'] * 100, 4),
                    'Q75': round(q_preds['Q75'] * 100, 4),
                    'Q90': round(q_preds['Q90'] * 100, 4),
                },
                'cone': {
                    'inner': [round(q_preds['Q25'] * 100, 4), round(q_preds['Q75'] * 100, 4)],
                    'outer': [round(q_preds['Q10'] * 100, 4), round(q_preds['Q90'] * 100, 4)],
                    'center': round(q50 * 100, 4),
                },
            }

        # Horizon alignment (3 horizons: 1H, 4H, 1D)
        directions = [result['horizons'][h]['direction'] for h in HORIZONS]
        bullish_count = sum(1 for d in directions if d == 'bullish')
        bearish_count = sum(1 for d in directions if d == 'bearish')
        if bullish_count == 3:
            alignment = 'strong_bullish'
        elif bearish_count == 3:
            alignment = 'strong_bearish'
        elif bullish_count >= 2 and bearish_count == 0:
            alignment = 'bullish'
        elif bearish_count >= 2 and bullish_count == 0:
            alignment = 'bearish'
        elif bullish_count > 0 and bearish_count > 0:
            alignment = 'conflict'
        else:
            alignment = 'neutral'
        result['horizon_alignment'] = alignment

        return result
