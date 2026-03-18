"""
Diagnostic: compare features computed two ways for AUDNZD at the latest hour.
1. "Backtest-style": fetch 40 days of 1m, resample, compute features
2. "Live-style": fetch 7 days of 1m, resample, compute features
Compare the feature values for the same hour.
"""

import asyncio
import os
import sys
import time
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

warnings.filterwarnings('ignore')

sys.stdout.reconfigure(encoding='utf-8')

BACKEND_DIR = Path(__file__).parent / 'backend'
sys.path.insert(0, str(BACKEND_DIR))

from dotenv import load_dotenv
load_dotenv()

from features import PAIRS, compute_features_for_pair, resample_ohlcv
from data_service import fetch_historical_bars

MODELS_DIR = BACKEND_DIR / 'models_5.1'

# Load models
q50_bundle = joblib.load(MODELS_DIR / '3_quants' / 'model_1H_Q50.joblib')
q25_bundle = joblib.load(MODELS_DIR / '3_quants' / 'model_1H_Q25.joblib')
q75_bundle = joblib.load(MODELS_DIR / '3_quants' / 'model_1H_Q75.joblib')
meta_bundle = joblib.load(MODELS_DIR / 'meta' / 'meta_confidence.joblib')

feature_cols = q50_bundle['feature_cols']
meta_feature_cols = meta_bundle['meta_feature_cols']

AVG_SPREAD = 0.00028
MIN_Q50_THRESHOLD = AVG_SPREAD * 0.5


def predict_single(features_df, pair):
    X = features_df[feature_cols].ffill().fillna(0)
    latest = X.iloc[[-1]]

    q50 = float(q50_bundle['model'].predict(latest)[0])
    q25 = float(q25_bundle['model'].predict(latest)[0])
    q75 = float(q75_bundle['model'].predict(latest)[0])

    abs_q50 = abs(q50)
    iqr = q75 - q25

    meta_row = latest.copy()
    meta_row['Q50_oof'] = q50
    meta_row['Q25_oof'] = q25
    meta_row['Q75_oof'] = q75
    meta_row['abs_Q50'] = abs_q50
    meta_row['iqr'] = iqr
    meta_row['conf_ratio'] = abs_q50 / max(iqr, 1e-10)

    X_meta = meta_row[meta_feature_cols].fillna(0)
    meta_proba = float(meta_bundle['model'].predict_proba(X_meta)[0, 1])

    return {
        'q50': q50, 'q25': q25, 'q75': q75,
        'meta_proba': meta_proba, 'abs_q50': abs_q50, 'iqr': iqr,
    }


async def main():
    now = datetime.utcnow()
    to_date = now.strftime('%Y-%m-%d')

    test_pairs = ['AUDNZD', 'CHFJPY', 'EURJPY', 'EURGBP']

    for pair in test_pairs:
        print(f'\n{"="*80}')
        print(f'PAIR: {pair}')
        print(f'{"="*80}')

        # --- Method 1: Live-style (7 days) ---
        from_7d = (now - timedelta(days=7)).strftime('%Y-%m-%d')
        await asyncio.sleep(0.3)
        df_1m_7d = await fetch_historical_bars(pair, 1, 'minute', from_7d, to_date)
        df_1m_7d = df_1m_7d[~((df_1m_7d.index.dayofweek == 5) |
                              ((df_1m_7d.index.dayofweek == 6) & (df_1m_7d.index.hour < 21)))]
        # Drop incomplete bars
        now_naive = now.replace(tzinfo=None) if now.tzinfo else now
        df_1m_7d = df_1m_7d[df_1m_7d.index + timedelta(minutes=1) <= now_naive]

        df_5m_7d = resample_ohlcv(df_1m_7d, '5min')
        df_15m_7d = resample_ohlcv(df_1m_7d, '15min')
        df_5m_7d = df_5m_7d[df_5m_7d.index + timedelta(minutes=5) <= now_naive]
        df_15m_7d = df_15m_7d[df_15m_7d.index + timedelta(minutes=15) <= now_naive]

        feat_7d = compute_features_for_pair(pair, df_1m_7d, df_5m_7d, df_15m_7d)

        # --- Method 2: Backtest-style (40 days) ---
        from_40d = (now - timedelta(days=40)).strftime('%Y-%m-%d')
        await asyncio.sleep(0.3)
        df_1m_40d = await fetch_historical_bars(pair, 1, 'minute', from_40d, to_date)
        df_1m_40d = df_1m_40d[~((df_1m_40d.index.dayofweek == 5) |
                                ((df_1m_40d.index.dayofweek == 6) & (df_1m_40d.index.hour < 21)))]
        df_1m_40d = df_1m_40d[df_1m_40d.index + timedelta(minutes=1) <= now_naive]

        df_5m_40d = resample_ohlcv(df_1m_40d, '5min')
        df_15m_40d = resample_ohlcv(df_1m_40d, '15min')
        df_5m_40d = df_5m_40d[df_5m_40d.index + timedelta(minutes=5) <= now_naive]
        df_15m_40d = df_15m_40d[df_15m_40d.index + timedelta(minutes=15) <= now_naive]

        feat_40d = compute_features_for_pair(pair, df_1m_40d, df_5m_40d, df_15m_40d)

        # --- Compare last few hours ---
        # Find common hours
        common_hours = feat_7d.index.intersection(feat_40d.index)
        if len(common_hours) == 0:
            print("No common hours!")
            continue

        # Compare last 3 common hours
        last_hours = sorted(common_hours)[-3:]

        for hour in last_hours:
            row_7d = feat_7d.loc[hour]
            row_40d = feat_40d.loc[hour]

            # Predict with each
            # For prediction we need the full feature history up to this hour
            feat_7d_up_to = feat_7d[feat_7d.index <= hour]
            feat_40d_up_to = feat_40d[feat_40d.index <= hour]

            pred_7d = predict_single(feat_7d_up_to, pair)
            pred_40d = predict_single(feat_40d_up_to, pair)

            print(f'\n  Hour: {hour}')
            print(f'  {"":>25} {"7-day (live)":>15} {"40-day (BT)":>15} {"Diff":>12}')
            print(f'  {"-"*70}')
            print(f'  {"meta_proba":>25} {pred_7d["meta_proba"]:>15.4f} {pred_40d["meta_proba"]:>15.4f} {pred_40d["meta_proba"]-pred_7d["meta_proba"]:>+12.4f}')
            print(f'  {"Q50":>25} {pred_7d["q50"]:>15.8f} {pred_40d["q50"]:>15.8f} {pred_40d["q50"]-pred_7d["q50"]:>+12.8f}')
            print(f'  {"abs_Q50":>25} {pred_7d["abs_q50"]:>15.8f} {pred_40d["abs_q50"]:>15.8f}')
            print(f'  {"IQR":>25} {pred_7d["iqr"]:>15.8f} {pred_40d["iqr"]:>15.8f}')

            # Show features that differ most
            diffs = []
            for col in feature_cols:
                if col in row_7d.index and col in row_40d.index:
                    v7 = float(row_7d[col]) if not pd.isna(row_7d[col]) else 0
                    v40 = float(row_40d[col]) if not pd.isna(row_40d[col]) else 0
                    if abs(v7) + abs(v40) > 1e-10:
                        rel_diff = abs(v7 - v40) / (max(abs(v7), abs(v40), 1e-10))
                        diffs.append((col, v7, v40, rel_diff))

            diffs.sort(key=lambda x: x[3], reverse=True)
            if diffs:
                print(f'\n  Top feature differences:')
                print(f'  {"Feature":>30} {"7-day":>15} {"40-day":>15} {"RelDiff":>10}')
                for col, v7, v40, rd in diffs[:10]:
                    print(f'  {col:>30} {v7:>15.6f} {v40:>15.6f} {rd:>10.4f}')


if __name__ == '__main__':
    asyncio.run(main())
