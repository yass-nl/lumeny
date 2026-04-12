"""
MFE Model Evaluation — Live Data
==================================
Strictly evaluates MFE model predictive accuracy on live data.
No direction, no capital, no trades.

For every bar where q50_mfe >= threshold:
  - What did the model predict?
  - What actually happened (max favorable move in 8h, both directions)?
  - Distribution of predicted vs actual across pairs and MFE buckets
"""

import os, asyncio, sys
import pandas as pd
import numpy as np
import joblib
import httpx
import shap
from datetime import datetime, timedelta
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
API_KEY   = os.getenv('POLYGON_S3_SECRET_KEY', '')
REST_BASE = 'https://api.polygon.io'

sys.path.insert(0, str(Path(__file__).parent / 'backend'))
from features import compute_features_for_pair, PIP_SIZE
from live_features_extra import compute_momentum_calendar_features

# ── Config ────────────────────────────────────────────────────────────────────
MFE_THRESH    = 30.0    # include all bars above this so we can slice by bucket
MFE_HORIZON_H = 72      # hours — original model uses uncapped trail stop (up to 72 bars)
FETCH_DAYS    = 220
TEST_DAYS     = 180

PAIRS = [
    'EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD', 'USDCAD', 'NZDUSD',
    'EURJPY', 'GBPJPY', 'EURGBP', 'EURAUD', 'AUDJPY', 'CADJPY', 'CHFJPY', 'AUDNZD',
]

CURRENCY_SIGN = {
    'EURUSD': {'EUR': +1, 'USD': -1}, 'GBPUSD': {'GBP': +1, 'USD': -1},
    'USDJPY': {'USD': +1, 'JPY': -1}, 'USDCHF': {'USD': +1, 'CHF': -1},
    'AUDUSD': {'AUD': +1, 'USD': -1}, 'USDCAD': {'USD': +1, 'CAD': -1},
    'NZDUSD': {'NZD': +1, 'USD': -1}, 'EURJPY': {'EUR': +1, 'JPY': -1},
    'GBPJPY': {'GBP': +1, 'JPY': -1}, 'EURGBP': {'EUR': +1, 'GBP': -1},
    'EURAUD': {'EUR': +1, 'AUD': -1}, 'AUDJPY': {'AUD': +1, 'JPY': -1},
    'CADJPY': {'CAD': +1, 'JPY': -1}, 'CHFJPY': {'CHF': +1, 'JPY': -1},
    'AUDNZD': {'AUD': +1, 'NZD': -1},
}


# ── Polygon fetch ─────────────────────────────────────────────────────────────
async def fetch_bars(client, pair, from_date, to_date):
    ticker = f'C:{pair}'
    url    = f'{REST_BASE}/v2/aggs/ticker/{ticker}/range/1/minute/{from_date}/{to_date}'
    params = {'apiKey': API_KEY, 'limit': 50000, 'sort': 'asc'}
    all_results = []
    resp = await client.get(url, params=params)
    resp.raise_for_status()
    data = resp.json()
    all_results.extend(data.get('results', []))
    while 'next_url' in data:
        sep  = '&' if '?' in data['next_url'] else '?'
        resp = await client.get(f"{data['next_url']}{sep}apiKey={API_KEY}")
        resp.raise_for_status()
        data = resp.json()
        all_results.extend(data.get('results', []))
    if not all_results:
        return pd.DataFrame()
    df = pd.DataFrame(all_results)
    df['datetime'] = pd.to_datetime(df['t'], unit='ms', utc=True).dt.tz_localize(None)
    df = df.rename(columns={'o':'open','h':'high','l':'low','c':'close','v':'volume'})
    df = df.set_index('datetime')[['open','high','low','close','volume']]
    df = df.sort_index().drop_duplicates()
    df = df[~((df.index.dayofweek == 5) | ((df.index.dayofweek == 6) & (df.index.hour < 21)))]
    return df

async def fetch_all_pairs():
    now       = datetime.now()
    to_date   = now.strftime('%Y-%m-%d')
    from_date = (now - timedelta(days=FETCH_DAYS)).strftime('%Y-%m-%d')
    print(f'Fetching {from_date} -> {to_date}')
    raw = {}
    async with httpx.AsyncClient(timeout=120) as client:
        for pair in PAIRS:
            print(f'  {pair}...', end=' ', flush=True)
            df_1m = await fetch_bars(client, pair, from_date, to_date)
            await asyncio.sleep(0.3)
            if df_1m.empty:
                print('NO DATA'); continue
            df_5m  = df_1m.resample('5min').agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum'}).dropna()
            df_15m = df_1m.resample('15min').agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum'}).dropna()
            df_1h  = df_1m.resample('1h').agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum'}).dropna()
            raw[pair] = {'1m': df_1m, '5m': df_5m, '15m': df_15m, '1h': df_1h}
            print(f'{len(df_1m):,} 1m / {len(df_1h)} 1h bars')
    return raw, from_date, to_date


# ── Cross-pair features ───────────────────────────────────────────────────────
def compute_all_cross_pair_features(close_1h_all):
    returns_all = {p: np.log(c / c.shift(1)) for p, c in close_1h_all.items()}
    returns_df  = pd.DataFrame(returns_all)

    currencies = ['EUR', 'USD', 'GBP', 'JPY', 'AUD', 'NZD', 'CAD', 'CHF']
    csi = {}
    for ccy in currencies:
        comps = [CURRENCY_SIGN[p][ccy] * returns_df[p]
                 for p in PAIRS if ccy in CURRENCY_SIGN.get(p, {}) and p in returns_df]
        if comps:
            csi[f'csi_{ccy.lower()}'] = pd.concat(comps, axis=1).mean(axis=1)
    csi_df = pd.DataFrame(csi)
    csi_rolling = {}
    for col in csi_df.columns:
        csi_rolling[f'{col}_24h'] = csi_df[col].rolling(24,  min_periods=8).sum()
        csi_rolling[f'{col}_72h'] = csi_df[col].rolling(72,  min_periods=24).sum()
    csi_rolling_df = pd.DataFrame(csi_rolling)

    result = {}
    for pair in PAIRS:
        if pair not in returns_df.columns:
            continue
        r      = returns_df[pair]
        c_pair = close_1h_all[pair]
        cols   = {}
        for peer in [p for p in PAIRS if p != pair]:
            if peer not in returns_df.columns:
                continue
            p_ret  = returns_df[peer]
            c_peer = close_1h_all[peer]
            sl     = peer.lower()
            for w, lbl in [(24, '24h'), (72, '3d'), (168, '1w')]:
                cols[f'corr_{sl}_{lbl}'] = r.rolling(w, min_periods=w//2).corr(p_ret)
            cols[f'corr_regime_{sl}'] = cols[f'corr_{sl}_24h'] - cols[f'corr_{sl}_1w']
            for w, lbl in [(24, '24h'), (168, '1w')]:
                cov = r.rolling(w, min_periods=w//2).cov(p_ret)
                var = p_ret.rolling(w, min_periods=w//2).var().clip(lower=1e-12)
                cols[f'beta_{sl}_{lbl}'] = cov / var
            cols[f'relstr_{sl}_1h']    = r - p_ret
            cols[f'relstr_{sl}_4h']    = np.log(c_pair / c_pair.shift(4))  - np.log(c_peer / c_peer.shift(4))
            cols[f'relstr_{sl}_24h']   = np.log(c_pair / c_pair.shift(24)) - np.log(c_peer / c_peer.shift(24))
            cols[f'peer_{sl}_ret_1h']  = p_ret
            cols[f'peer_{sl}_ret_4h']  = np.log(c_peer / c_peer.shift(4))
            cols[f'peer_{sl}_ret_24h'] = np.log(c_peer / c_peer.shift(24))
        for col in csi_df.columns:
            cols[col]          = csi_df[col]
            cols[f'{col}_24h'] = csi_rolling_df[f'{col}_24h']
            cols[f'{col}_72h'] = csi_rolling_df[f'{col}_72h']
        result[pair] = pd.DataFrame(cols, index=r.index).astype(np.float32)
    return result


# ── Compute actual MFE (direction-agnostic, best of LONG/SHORT over 8h) ───────
def compute_actual_mfe(df_1h, pos, pip_size):
    """
    Max favorable move in either direction over MFE_HORIZON_H bars.
    Returns (mfe_long_pips, mfe_short_pips) — both positive values.
    mfe_long  = max(highs[pos+1..pos+8]) - close[pos]
    mfe_short = close[pos] - min(lows[pos+1..pos+8])
    """
    highs  = df_1h['high'].values
    lows   = df_1h['low'].values
    closes = df_1h['close'].values
    n      = len(df_1h)
    entry  = closes[pos]
    end    = min(pos + MFE_HORIZON_H + 1, n)
    if end <= pos + 1:
        return np.nan, np.nan
    mfe_long  = (highs[pos+1:end].max() - entry) / pip_size
    mfe_short = (entry - lows[pos+1:end].min()) / pip_size
    return mfe_long, mfe_short


# ── Main ──────────────────────────────────────────────────────────────────────
async def main():
    print('Loading MFE model...')
    mfe_bundle   = joblib.load('backend/models_9/mfe_q50/model_1H_Q50.joblib')
    mfe_model    = mfe_bundle['model']
    feature_cols = mfe_bundle['feature_cols']
    print(f'  {len(feature_cols)} features, {mfe_bundle["n_iters"]} iters')

    raw, from_date, to_date = await fetch_all_pairs()
    if not raw:
        print('No data fetched.'); return

    test_start = datetime.now() - timedelta(days=TEST_DAYS)
    print(f'\nTest window: {test_start.strftime("%Y-%m-%d")} -> {to_date}')

    close_1h_all = {pair: raw[pair]['1h']['close'] for pair in raw}
    print('\nComputing cross-pair features...')
    cross_features = compute_all_cross_pair_features(close_1h_all)

    print('Computing per-pair features...')
    pair_features = {}
    for pair in PAIRS:
        if pair not in raw:
            continue
        print(f'  {pair}...', end=' ', flush=True)
        df_base = compute_features_for_pair(
            pair, raw[pair]['1m'], raw[pair]['5m'],
            raw[pair]['15m'], df_1h=raw[pair]['1h']
        )
        if df_base.empty:
            print('no features'); continue
        df_extra = compute_momentum_calendar_features(raw[pair]['1h'], PIP_SIZE[pair])
        df_base  = df_base.join(df_extra.reindex(df_base.index), how='left')
        if pair in cross_features:
            df_base = df_base.join(cross_features[pair].reindex(df_base.index), how='left')
        pair_features[pair] = df_base
        print(f'{len(df_base)} rows')

    # ── Collect all candidate bars ─────────────────────────────────────────────
    print('\nScoring all bars...')
    rows       = []
    shap_X_70  = []   # feature rows for q50_mfe >= 70 bars (for SHAP)

    for pair in PAIRS:
        if pair not in pair_features:
            continue
        pip_size   = PIP_SIZE[pair]
        df_feat    = pair_features[pair]
        df_1h      = raw[pair]['1h']

        df_test = df_feat[df_feat.index >= test_start].copy()
        if len(df_test) < 10:
            continue

        common_idx = df_test.index.intersection(df_1h.index)
        df_test    = df_test.loc[common_idx]

        for col in feature_cols:
            if col not in df_test.columns:
                df_test[col] = 0.0

        X_all = df_test[feature_cols].ffill().fillna(0)
        df_test = df_test.copy()
        preds = mfe_model.predict(X_all)
        df_test['q50_mfe'] = preds

        # Only keep bars above minimum threshold
        df_cands = df_test[df_test['q50_mfe'] >= MFE_THRESH]

        cooldown_until = -1  # integer position in df_1h

        for ts, row in df_cands.iterrows():
            try:
                pos = df_1h.index.get_loc(ts)
            except KeyError:
                continue
            if pos <= cooldown_until:
                continue
            cooldown_until = pos + MFE_HORIZON_H  # lock out next 8 bars on this pair
            mfe_long, mfe_short = compute_actual_mfe(df_1h, pos, pip_size)
            if np.isnan(mfe_long):
                continue
            actual_mfe_best = max(mfe_long, mfe_short)   # best possible direction
            rows.append({
                'ts':               ts,
                'pair':             pair,
                'q50_mfe':          row['q50_mfe'],
                'actual_mfe_long':  mfe_long,
                'actual_mfe_short': mfe_short,
                'actual_mfe_best':  actual_mfe_best,
            })
            if row['q50_mfe'] >= 70:
                shap_X_70.append(X_all.loc[ts])

    if not rows:
        print('No candidates found.'); return

    df = pd.DataFrame(rows).sort_values('ts').reset_index(drop=True)
    print(f'  Total candidate bars: {len(df):,}')

    # ── Analysis ──────────────────────────────────────────────────────────────
    print(f'\n{"="*80}')
    print(f'  MFE MODEL EVALUATION — LIVE DATA')
    print(f'  Predicted Q50 >= {MFE_THRESH}p | Actual MFE over {MFE_HORIZON_H}h (uncapped, both dirs) | No direction filter')
    print(f'  Period: {df["ts"].min().date()} to {df["ts"].max().date()}  |  N={len(df):,}')
    print(f'{"="*80}')

    # Overall stats
    pred  = df['q50_mfe']
    best  = df['actual_mfe_best']
    long_ = df['actual_mfe_long']
    short_= df['actual_mfe_short']

    print(f'\n--- Overall (all bars q50_mfe >= {MFE_THRESH}) ---')
    print(f'  N                         : {len(df):,}')
    print(f'  Predicted Q50 avg         : {pred.mean():>+7.1f}p   median={pred.median():>+7.1f}p')
    print(f'  Actual MFE (best dir) avg : {best.mean():>+7.1f}p   median={best.median():>+7.1f}p')
    print(f'  Actual MFE (long)    avg  : {long_.mean():>+7.1f}p   median={long_.median():>+7.1f}p')
    print(f'  Actual MFE (short)   avg  : {short_.mean():>+7.1f}p   median={short_.median():>+7.1f}p')
    print(f'  Pred >= actual (best)     : {(pred <= best).mean():.1%}   (model conservative?)')
    print(f'  Actual >= 50p (best)      : {(best >= 50).mean():.1%}')
    print(f'  Actual >= 70p (best)      : {(best >= 70).mean():.1%}')
    print(f'  Actual >= 100p (best)     : {(best >= 100).mean():.1%}')

    # By prediction bucket
    print(f'\n--- By prediction bucket ---')
    print(f'  {"Bucket":<18} {"N":>5}  {"Pred_avg":>9}  {"Act_best":>9}  {"Act_long":>9}  {"Act_short":>10}  {">=50p":>6}  {">=70p":>6}  {">=100p":>7}')
    print(f'  {"-"*90}')
    buckets = [(30,50,'30-50p'),(50,70,'50-70p'),(70,100,'70-100p'),(100,150,'100-150p'),(150,9999,'150p+')]
    for lo, hi, label in buckets:
        sub = df[(df['q50_mfe'] >= lo) & (df['q50_mfe'] < hi)]
        if len(sub) == 0: continue
        p  = sub['q50_mfe']
        b  = sub['actual_mfe_best']
        l  = sub['actual_mfe_long']
        s  = sub['actual_mfe_short']
        print(f'  {label:<18} {len(sub):>5}  {p.mean():>+9.1f}  {b.mean():>+9.1f}  {l.mean():>+9.1f}  {s.mean():>+10.1f}  '
              f'{(b>=50).mean():>6.1%}  {(b>=70).mean():>6.1%}  {(b>=100).mean():>7.1%}')

    # Focus: q50_mfe >= 70 (main operating threshold)
    df70 = df[df['q50_mfe'] >= 70]
    if len(df70):
        print(f'\n--- Deep dive: q50_mfe >= 70 (N={len(df70)}) ---')
        b = df70['actual_mfe_best']
        l = df70['actual_mfe_long']
        s = df70['actual_mfe_short']
        p = df70['q50_mfe']
        print(f'  Pred avg / median         : {p.mean():>+7.1f}p  /  {p.median():>+7.1f}p')
        print(f'  Actual best avg / median  : {b.mean():>+7.1f}p  /  {b.median():>+7.1f}p')
        print(f'  Actual long avg / median  : {l.mean():>+7.1f}p  /  {l.median():>+7.1f}p')
        print(f'  Actual short avg / median : {s.mean():>+7.1f}p  /  {s.median():>+7.1f}p')
        print(f'  Model is conservative     : {(p <= b).mean():.1%}  (pred < actual_best)')
        print(f'  Actual best >= pred       : {(b >= p).mean():.1%}')
        print(f'  Actual best >= 70p        : {(b >= 70).mean():.1%}')
        print(f'  Actual best >= 100p       : {(b >= 100).mean():.1%}')
        print(f'  Actual best >= 150p       : {(b >= 150).mean():.1%}')
        # Percentiles of actual_mfe_best
        pcts = [10, 25, 50, 75, 90, 95]
        pct_vals = np.percentile(b, pcts)
        print(f'  Actual best percentiles   : ' + '  '.join(f'p{p}={v:.0f}' for p, v in zip(pcts, pct_vals)))

    # Per-pair breakdown
    print(f'\n--- Per-pair (q50_mfe >= 70) ---')
    print(f'  {"Pair":<10} {"N":>5}  {"Pred_avg":>9}  {"Act_best":>9}  {"Act_long":>9}  {"Act_short":>10}  {">=70p%":>7}')
    print(f'  {"-"*70}')
    for pair in sorted(df['pair'].unique()):
        sub = df[(df['pair'] == pair) & (df['q50_mfe'] >= 70)]
        if len(sub) == 0: continue
        p = sub['q50_mfe']
        b = sub['actual_mfe_best']
        l = sub['actual_mfe_long']
        s = sub['actual_mfe_short']
        print(f'  {pair:<10} {len(sub):>5}  {p.mean():>+9.1f}  {b.mean():>+9.1f}  {l.mean():>+9.1f}  {s.mean():>+10.1f}  {(b>=70).mean():>7.1%}')

    # Calibration: is the model biased?
    print(f'\n--- Calibration: predicted vs actual (q50_mfe >= 70) ---')
    if len(df70):
        bias_best  = (df70['q50_mfe'] - df70['actual_mfe_best']).mean()
        bias_long  = (df70['q50_mfe'] - df70['actual_mfe_long']).mean()
        bias_short = (df70['q50_mfe'] - df70['actual_mfe_short']).mean()
        corr_best  = df70['q50_mfe'].corr(df70['actual_mfe_best'])
        corr_long  = df70['q50_mfe'].corr(df70['actual_mfe_long'])
        print(f'  Avg (pred - actual_best)  : {bias_best:>+7.1f}p  (+ = model overestimates)')
        print(f'  Avg (pred - actual_long)  : {bias_long:>+7.1f}p')
        print(f'  Avg (pred - actual_short) : {bias_short:>+7.1f}p')
        print(f'  Correlation pred/act_best : {corr_best:>+7.3f}')
        print(f'  Correlation pred/act_long : {corr_long:>+7.3f}')

    # Monthly trend — is the model degrading over time?
    print(f'\n--- Monthly calibration (q50_mfe >= 70) ---')
    print(f'  {"Month":<8} {"N":>5}  {"Pred_avg":>9}  {"Act_best":>9}  {"Bias":>8}  {">=70p%":>7}')
    print(f'  {"-"*55}')
    df70 = df70.copy()
    df70['month'] = df70['ts'].dt.to_period('M')
    for month in sorted(df70['month'].unique()):
        sub = df70[df70['month'] == month]
        p = sub['q50_mfe'].mean()
        b = sub['actual_mfe_best'].mean()
        hit = (sub['actual_mfe_best'] >= 70).mean()
        print(f'  {str(month):<8} {len(sub):>5}  {p:>+9.1f}  {b:>+9.1f}  {p-b:>+8.1f}  {hit:>7.1%}')

    print(f'\n{"="*80}')

    # ── SHAP analysis on q50_mfe >= 70 bars ───────────────────────────────────
    if not shap_X_70:
        print('No bars for SHAP analysis.'); return

    import shap
    X_shap = pd.DataFrame(shap_X_70, columns=feature_cols)
    print(f'\n{"="*80}')
    print(f'  SHAP ANALYSIS — what drives high MFE predictions (q50_mfe >= 70, N={len(X_shap)})')
    print(f'{"="*80}')

    print('Computing SHAP values...')
    explainer   = shap.TreeExplainer(mfe_model)
    shap_values = explainer.shap_values(X_shap)   # shape (N, n_features)

    shap_df = pd.DataFrame(shap_values, columns=feature_cols)

    # Mean absolute SHAP — overall importance
    mean_abs = shap_df.abs().mean().sort_values(ascending=False)

    print(f'\n--- Top 30 features by mean |SHAP| ---')
    print(f'  {"Feature":<45} {"Mean|SHAP|":>10}  {"Mean SHAP":>10}  {"% positive":>11}')
    print(f'  {"-"*80}')
    for feat in mean_abs.head(30).index:
        vals     = shap_df[feat]
        mean_abs_val = vals.abs().mean()
        mean_val     = vals.mean()
        pct_pos      = (vals > 0).mean()
        print(f'  {feat:<45} {mean_abs_val:>+10.3f}  {mean_val:>+10.3f}  {pct_pos:>10.1%}')

    # Group by feature prefix to understand categories
    print(f'\n--- Feature category breakdown (sum of mean |SHAP|) ---')
    categories = {
        'atr':      [f for f in feature_cols if 'atr' in f],
        'spread':   [f for f in feature_cols if 'spread' in f],
        'volume':   [f for f in feature_cols if 'vol' in f],
        'momentum': [f for f in feature_cols if any(x in f for x in ['mom','ret','rsi','macd'])],
        'corr':     [f for f in feature_cols if 'corr' in f],
        'csi':      [f for f in feature_cols if 'csi' in f],
        'beta':     [f for f in feature_cols if 'beta' in f],
        'relstr':   [f for f in feature_cols if 'relstr' in f],
        'peer':     [f for f in feature_cols if 'peer' in f],
        'calendar': [f for f in feature_cols if any(x in f for x in ['hour','day','week','month','session'])],
    }
    covered = set()
    cat_scores = {}
    for cat, feats in categories.items():
        feats_present = [f for f in feats if f in mean_abs.index]
        if feats_present:
            score = mean_abs[feats_present].sum()
            cat_scores[cat] = (score, len(feats_present))
            covered.update(feats_present)
    other_feats = [f for f in mean_abs.index if f not in covered]
    if other_feats:
        cat_scores['other'] = (mean_abs[other_feats].sum(), len(other_feats))

    total_shap = sum(v[0] for v in cat_scores.values())
    print(f'  {"Category":<12} {"Sum|SHAP|":>10}  {"% total":>8}  {"N_feats":>8}')
    print(f'  {"-"*45}')
    for cat, (score, n) in sorted(cat_scores.items(), key=lambda x: -x[1][0]):
        print(f'  {cat:<12} {score:>10.3f}  {score/total_shap:>8.1%}  {n:>8}')

    # Top features that always push prediction UP (consistently high MFE)
    print(f'\n--- Features that consistently INCREASE predicted MFE (mean SHAP > 0, top 15) ---')
    print(f'  {"Feature":<45} {"Mean SHAP":>10}  {"% positive":>11}  {"Raw avg value":>14}')
    print(f'  {"-"*85}')
    positive_drivers = shap_df.mean().sort_values(ascending=False).head(15)
    for feat in positive_drivers.index:
        mean_shap = shap_df[feat].mean()
        pct_pos   = (shap_df[feat] > 0).mean()
        raw_avg   = X_shap[feat].mean()
        print(f'  {feat:<45} {mean_shap:>+10.3f}  {pct_pos:>10.1%}  {raw_avg:>14.4f}')

    print(f'\n{"="*80}')


asyncio.run(main())
