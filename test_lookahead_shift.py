"""
Live Full Test — Capital Simulation
=====================================
MFE model (Q50 >= 70 pips) + Dir prob model:
  - dir_pred > 0.5  -> LONG
  - dir_pred <= 0.5 -> SHORT
  Always single side — highest probability direction wins.

Pipeline:
1. Fetch 1M OHLCV for all 15 pairs (FETCH_DAYS for warm-up)
2. Compute all features (microstructure + momentum/calendar + cross-pair)
3. MFE Q50 >= 70 filter
4. Direction: dir_prob model selects LONG / SHORT / dual-side
5. Cooldown: no re-entry on a pair while trade is open
6. Trail stop: TRAIL_MULT * atr_24, timeout at 8h
7. Capital simulation: $100k, 0.5% risk/trade, trail-based sizing
"""

import os, asyncio, sys
import pandas as pd
import numpy as np
import joblib
import httpx
from datetime import datetime, timedelta
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
API_KEY   = os.getenv('POLYGON_S3_SECRET_KEY', '')
REST_BASE = 'https://api.polygon.io'

sys.path.insert(0, str(Path(__file__).parent / 'backend'))
from features import compute_features_for_pair, PIP_SIZE
from live_features_extra import compute_momentum_calendar_features

# ── Strategy config ───────────────────────────────────────────────────────────
MFE_THRESH     = 70.0        # MFE Q50 >= this to consider a bar
COOLDOWN_H     = 8           # hours to lock pair after entry
TRAIL_MULT     = 1.5         # trail stop = TRAIL_MULT * atr_24 from peak/trough
TIMEOUT_H      = 8           # max trade duration hours (aligned with MFE_HORIZON_H)
DIR_CONF_THRESH = 0.65       # reserved for reporting — trading always takes side with dir_pred > 0.5
MFE_HORIZON_H  = 8           # hours over which model's MFE target was defined

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

JPY_PAIRS = {'USDJPY', 'EURJPY', 'GBPJPY', 'AUDJPY', 'CADJPY', 'CHFJPY'}

# ── Capital simulation parameters ─────────────────────────────────────────────
STARTING_CAPITAL = 100_000.0
LOT_UNITS        = 100_000
RISK_PER_TRADE   = 0.005       # 0.5% of equity per trade
LEVERAGE         = 50

SPREAD_PIPS = {
    'EURUSD': 0.6, 'GBPUSD': 0.8, 'USDJPY': 1.0, 'USDCHF': 0.7,
    'AUDUSD': 0.6, 'USDCAD': 1.2, 'NZDUSD': 0.9,
    'EURJPY': 1.4, 'GBPJPY': 2.1, 'EURGBP': 0.7,
    'EURAUD': 2.1, 'AUDJPY': 1.5, 'CADJPY': 1.6, 'CHFJPY': 2.5, 'AUDNZD': 2.0,
}

QUOTE_USD = {
    'USD': 1.00, 'CHF': 1.10, 'CAD': 0.73, 'GBP': 1.27,
    'AUD': 0.65, 'NZD': 0.59, 'JPY': 0.0067,
}
BASE_USD = {
    'EUR': 1.08, 'GBP': 1.27, 'AUD': 0.65, 'NZD': 0.59,
    'USD': 1.00, 'CAD': 0.73, 'CHF': 1.10,
}

FETCH_DAYS = 220
TEST_DAYS  = 180


# ── Capital helpers ───────────────────────────────────────────────────────────
def pip_value_per_lot(pair, entry_price):
    pip   = PIP_SIZE[pair]
    quote = pair[3:]
    if pair in JPY_PAIRS:
        return LOT_UNITS * pip / entry_price
    elif quote == 'USD':
        return LOT_UNITS * pip
    else:
        return LOT_UNITS * pip * QUOTE_USD.get(quote, 1.0)

def margin_per_lot(pair, entry_price):
    base = pair[:3]
    return LOT_UNITS * BASE_USD.get(base, 1.0) / LEVERAGE

def compute_pnl_usd(pair, direction, lots, entry_price, exit_price):
    move  = (exit_price - entry_price) if direction == 1 else (entry_price - exit_price)
    quote = pair[3:]
    if pair in JPY_PAIRS:
        return lots * LOT_UNITS * move / exit_price
    elif quote == 'USD':
        return lots * LOT_UNITS * move
    else:
        return lots * LOT_UNITS * move * QUOTE_USD.get(quote, 1.0)


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


# ── Trailing stop simulation ──────────────────────────────────────────────────
def simulate_trade(df_1h, entry_pos, direction, trail_price, pip_size):
    """
    Pure trailing stop exit with timeout fallback.
    Trail starts from entry immediately — TRAIL_MULT * atr_24 from peak/trough.
    Returns (actual_mfe_pips, mae_pips, exit_price, duration_bars, exit_type)
    exit_type: 'trail' or 'timeout'
    """
    highs  = df_1h['high'].values
    lows   = df_1h['low'].values
    closes = df_1h['close'].values
    n      = len(df_1h)
    entry  = closes[entry_pos]
    mfe_horizon = min(MFE_HORIZON_H, n - entry_pos - 1)

    if direction == 1:
        running_max = entry
        running_mae = 0.0
        exit_price  = None
        duration    = 0
        exit_type   = 'timeout'
        for k in range(1, min(TIMEOUT_H + 1, n - entry_pos)):
            idx = entry_pos + k
            # Update peak with this bar's high FIRST
            running_max = max(running_max, highs[idx])
            running_mae = max(running_mae, (entry - lows[idx]) / pip_size)
            # Trail fires only if: peak is above entry (moved in our favor)
            # AND the NEXT bar's open (i.e. this bar's close) is below peak - trail
            # We use closes[idx] as the retrace check — not lows[idx] — to avoid
            # intra-bar high-to-low range triggering the trail within the same bar
            if running_max - closes[idx] >= trail_price:
                exit_price = running_max - trail_price
                duration   = k
                exit_type  = 'trail'
                break
        if exit_price is None:
            duration   = min(TIMEOUT_H, n - entry_pos - 1)
            exit_price = closes[entry_pos + duration]
        mfe_max = entry
        for k in range(1, mfe_horizon + 1):
            mfe_max = max(mfe_max, highs[entry_pos + k])
        actual_mfe = (mfe_max - entry) / pip_size
        return actual_mfe, running_mae, exit_price, duration, exit_type
    else:
        running_min = entry
        running_mae = 0.0
        exit_price  = None
        duration    = 0
        exit_type   = 'timeout'
        for k in range(1, min(TIMEOUT_H + 1, n - entry_pos)):
            idx = entry_pos + k
            # Update trough with this bar's low FIRST
            running_min = min(running_min, lows[idx])
            running_mae = max(running_mae, (highs[idx] - entry) / pip_size)
            # Trail fires only if: trough is below entry (moved in our favor)
            # AND close retraces trail_price above that trough
            if closes[idx] - running_min >= trail_price:
                exit_price = running_min + trail_price
                duration   = k
                exit_type  = 'trail'
                break
        if exit_price is None:
            duration   = min(TIMEOUT_H, n - entry_pos - 1)
            exit_price = closes[entry_pos + duration]
        mfe_min = entry
        for k in range(1, mfe_horizon + 1):
            mfe_min = min(mfe_min, lows[entry_pos + k])
        actual_mfe = (entry - mfe_min) / pip_size
        return actual_mfe, running_mae, exit_price, duration, exit_type


# ── Main ──────────────────────────────────────────────────────────────────────
async def main():
    # ── Load models ───────────────────────────────────────────────────────────
    print('Loading models...')
    mfe_bundle   = joblib.load('backend/models_9/mfe_q50_8h/model_1H_Q50.joblib')
    mfe_model    = mfe_bundle['model']
    feature_cols = mfe_bundle['feature_cols']

    dir_bundle   = joblib.load('backend/models_9/dir_prob_8h/model_1H_dir_prob.joblib')
    dir_model    = dir_bundle['model']
    dir_cols     = dir_bundle['feature_cols']

    print(f'  MFE model  : {len(feature_cols)} features, {mfe_bundle["n_iters"]} iters')
    print(f'  Dir model  : {dir_bundle["n_iters"]} iters, {len(dir_cols)} features')
    print(f'  Dir target : {dir_bundle["target"]}')
    print(f'\nStrategy config:')
    print(f'  MFE threshold  : Q50 >= {MFE_THRESH} pips')
    print(f'  Direction      : dir_pred > 0.5 -> LONG | <= 0.5 -> SHORT (always single side)')
    print(f'  Cooldown       : no re-entry on pair while trade open (up to {TIMEOUT_H}h)')
    print(f'  Trail stop     : {TRAIL_MULT}x atr_24 from peak/trough  |  Timeout: {TIMEOUT_H}h')

    # ── Fetch data ─────────────────────────────────────────────────────────────
    raw, from_date, to_date = await fetch_all_pairs()
    if not raw:
        print('No data fetched.'); return

    test_start = datetime.now() - timedelta(days=TEST_DAYS)
    print(f'\nTest window: {test_start.strftime("%Y-%m-%d")} -> {to_date}')

    # ── Compute features ───────────────────────────────────────────────────────
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
        df_base['pair'] = pair
        pair_features[pair] = df_base
        print(f'{len(df_base)} rows, {df_base.shape[1]} cols')

    if not pair_features:
        print('No features computed.'); return

    # ── Pass 1: collect all signals ────────────────────────────────────────────
    print('\nCollecting signals...')
    all_signals = []
    diag_mfe_pass = 0
    diag_cooldown_skip = 0

    for pair in PAIRS:
        if pair not in pair_features:
            continue
        pip_size = PIP_SIZE[pair]
        sp_pips  = SPREAD_PIPS.get(pair, 2.0)
        df_feat  = pair_features[pair]
        df_1h    = raw[pair]['1h']

        # Test window only
        df_test_pair = df_feat[df_feat.index >= test_start].copy()
        if len(df_test_pair) < 10:
            continue

        # Align to 1h bars
        common_idx   = df_test_pair.index.intersection(df_1h.index)
        df_test_pair = df_test_pair.loc[common_idx]

        # Ensure all feature columns exist
        for col in feature_cols:
            if col not in df_test_pair.columns:
                df_test_pair[col] = 0.0
        for col in dir_cols:
            if col not in df_test_pair.columns:
                df_test_pair[col] = 0.0

        # Run both models
        df_test_pair = df_test_pair.copy()
        X_mfe = df_test_pair[feature_cols].shift(1).ffill().fillna(0)
        X_dir = df_test_pair[dir_cols].shift(1).ffill().fillna(0)
        df_test_pair['q50_mfe']  = mfe_model.predict(X_mfe)
        df_test_pair['dir_pred'] = dir_model.predict(X_dir)

        mfe_pass = df_test_pair[df_test_pair['q50_mfe'] >= MFE_THRESH]
        diag_mfe_pass += len(mfe_pass)
        if len(mfe_pass):
            n_long_dir  = (mfe_pass['dir_pred'] > DIR_CONF_THRESH).sum()
            n_short_dir = (mfe_pass['dir_pred'] < 1 - DIR_CONF_THRESH).sum()
            n_dual_dir  = len(mfe_pass) - n_long_dir - n_short_dir
            print(f'  {pair}: MFE>={MFE_THRESH} N={len(mfe_pass)}  (L={n_long_dir} S={n_short_dir} dual={n_dual_dir})')

        df_1h_full     = df_1h.copy()
        cooldown_until = -1   # integer position — no re-entry while trade open

        for i, (ts, row) in enumerate(df_test_pair.iterrows()):
            # MFE filter
            if row['q50_mfe'] < MFE_THRESH:
                continue

            # ATR required for trail stop
            atr_24 = row.get('atr_24', np.nan)
            if np.isnan(atr_24) or atr_24 <= 0:
                continue

            # Get position in 1h bar array
            try:
                pos = df_1h_full.index.get_loc(ts)
            except KeyError:
                continue

            # Cooldown check
            if pos <= cooldown_until:
                diag_cooldown_skip += 1
                continue

            entry_price    = df_1h_full['close'].iloc[pos]
            trail_pips_raw = TRAIL_MULT * atr_24        # pips
            trail_price    = trail_pips_raw * pip_size  # price units
            pvl            = pip_value_per_lot(pair, entry_price)
            move_8h        = (df_1h_full['close'].iloc[min(pos + 8, len(df_1h_full) - 1)] - entry_price) / pip_size

            # Determine direction from dir_prob model — always single side
            dir_pred   = row['dir_pred']
            directions = (1,) if dir_pred > 0.5 else (-1,)

            max_duration = 0
            for direction in directions:
                actual_mfe, mae_pips, exit_price, duration, exit_type = simulate_trade(
                    df_1h_full, pos, direction, trail_price, pip_size
                )
                if direction == 1:
                    result_pips = (exit_price - entry_price) / pip_size - sp_pips
                else:
                    result_pips = (entry_price - exit_price) / pip_size - sp_pips
                max_duration = max(max_duration, duration)

                all_signals.append({
                    'ts':          ts,
                    'pair':        pair,
                    'direction':   direction,
                    'entry_price': entry_price,
                    'exit_price':  exit_price,
                    'atr_24':      atr_24,
                    'trail_price': trail_price,
                    'trail_pips':  trail_pips_raw,
                    'result_pips': result_pips,
                    'actual_mfe':  actual_mfe,
                    'mae_pips':    mae_pips,
                    'exit_type':   exit_type,
                    'duration':    duration,
                    'sp_pips':     sp_pips,
                    'pvl':         pvl,
                    'q50_mfe':     row['q50_mfe'],
                    'dir_pred':    dir_pred,
                    'move_8h':     move_8h,
                })

            # Cooldown based on the longer of the trades fired
            cooldown_until = pos + max_duration

    print(f'\n--- Signal Funnel ---')
    print(f'  MFE>={MFE_THRESH} pass           : {diag_mfe_pass}')
    print(f'  Cooldown skipped          : {diag_cooldown_skip}')
    n_long_sigs  = sum(1 for s in all_signals if s['direction'] ==  1)
    n_short_sigs = sum(1 for s in all_signals if s['direction'] == -1)
    print(f'  Total leg signals         : {len(all_signals)}  (L={n_long_sigs} S={n_short_sigs})')
    if not all_signals:
        print('No signals generated.'); return

    # ── Pass 2: replay chronologically with single equity pool ─────────────────
    print('Running capital simulation...')
    all_signals.sort(key=lambda x: x['ts'])

    equity      = STARTING_CAPITAL
    peak_equity = equity
    max_dd      = 0.0
    max_dd_pct  = 0.0
    trades      = []

    for sig in all_signals:
        pair        = sig['pair']
        direction   = sig['direction']
        entry_price = sig['entry_price']
        exit_price  = sig['exit_price']
        pvl         = sig['pvl']
        sp_pips     = sig['sp_pips']
        pip_size    = PIP_SIZE[pair]
        atr_24      = sig['atr_24']

        # Position sizing: risk RISK_PER_TRADE on trail stop distance
        trail_pips = sig['trail_pips']   # already in pips
        risk_usd  = equity * RISK_PER_TRADE
        lots      = risk_usd / (trail_pips * pvl) if (trail_pips * pvl) > 0 else 0.01
        lots      = max(0.01, round(lots, 2))

        pnl_usd  = compute_pnl_usd(pair, direction, lots, entry_price, exit_price)
        sp_cost  = lots * pvl * sp_pips
        pnl_usd -= sp_cost
        win      = pnl_usd > 0

        equity += pnl_usd
        if equity > peak_equity:
            peak_equity = equity
        dd     = peak_equity - equity
        dd_pct = dd / peak_equity if peak_equity > 0 else 0
        max_dd     = max(max_dd, dd)
        max_dd_pct = max(max_dd_pct, dd_pct)

        trades.append({
            'ts':          sig['ts'],
            'pair':        pair,
            'direction':   'LONG' if direction == 1 else 'SHORT',
            'lots':        lots,
            'entry':       entry_price,
            'exit':        exit_price,
            'atr_24':      atr_24,
            'trail_price': sig['trail_price'],
            'result_pips': sig['result_pips'],
            'actual_mfe':  sig['actual_mfe'],
            'mae_pips':    sig['mae_pips'],
            'exit_type':   sig['exit_type'],
            'duration':    sig['duration'],
            'sp_cost':     sp_cost,
            'pnl_usd':     pnl_usd,
            'equity':      equity,
            'q50_mfe':     sig['q50_mfe'],
            'dir_pred':    sig['dir_pred'],
            'move_8h':     sig['move_8h'],
            'win':         win,
        })

    # ── Report ─────────────────────────────────────────────────────────────────
    tr = pd.DataFrame(trades).sort_values('ts')
    tr['month'] = tr['ts'].dt.to_period('M')
    tr['year']  = tr['ts'].dt.isocalendar().year.values
    tr['week']  = tr['ts'].dt.isocalendar().week.values
    tr['yw']    = tr['year'].astype(str) + '-W' + tr['week'].astype(str).str.zfill(2)
    tr['hour']  = tr['ts'].dt.hour

    n         = len(tr)
    wins      = tr[tr['win']]
    losses    = tr[~tr['win']]
    months    = TEST_DAYS / 30
    total_pnl = tr['pnl_usd'].sum()
    avg_pnl   = tr['pnl_usd'].mean()
    avg_win   = wins['pnl_usd'].mean()   if len(wins)   else 0
    avg_loss  = losses['pnl_usd'].mean() if len(losses) else 0
    pf        = wins['pnl_usd'].sum() / abs(losses['pnl_usd'].sum()) if len(losses) else 999

    daily_pnl = tr.groupby(tr['ts'].dt.date)['pnl_usd'].sum()
    sharpe    = (daily_pnl.mean() / daily_pnl.std() * np.sqrt(252)) if daily_pnl.std() > 0 else 0

    n_long  = (tr['direction'] == 'LONG').sum()
    n_short = (tr['direction'] == 'SHORT').sum()

    print(f'\n{"="*80}')
    print(f'  CAPITAL SIMULATION — MFE Q50>={MFE_THRESH} + DIR PROB (conf={DIR_CONF_THRESH})')
    print(f'  Trail: {TRAIL_MULT}x atr_24  |  Timeout: {TIMEOUT_H}h')
    print(f'  Period: {tr["ts"].min().date()} to {tr["ts"].max().date()}')
    print(f'{"="*80}')

    print(f'\n--- Account Summary ---')
    print(f'  Starting capital  : ${STARTING_CAPITAL:>12,.2f}')
    print(f'  Final equity      : ${equity:>12,.2f}')
    print(f'  Total P&L         : ${total_pnl:>12,.2f}  ({total_pnl/STARTING_CAPITAL*100:+.2f}%)')
    print(f'  P&L / month       : ${total_pnl/months:>12,.2f}')
    print(f'  Max drawdown      : ${max_dd:>12,.2f}  ({max_dd_pct*100:.2f}%)')
    print(f'  Sharpe (annual)   :  {sharpe:.2f}')
    print(f'  Total spread cost : ${tr["sp_cost"].sum():>12,.2f}')

    n_trail    = (tr['exit_type'] == 'trail').sum()
    n_timeout  = (tr['exit_type'] == 'timeout').sum()
    trail_wr     = tr[tr['exit_type'] == 'trail']['win'].mean()   if n_trail   else float('nan')
    timeout_wr   = tr[tr['exit_type'] == 'timeout']['win'].mean() if n_timeout else float('nan')
    trail_avg    = tr[tr['exit_type'] == 'trail']['result_pips'].mean()   if n_trail   else 0
    timeout_avg  = tr[tr['exit_type'] == 'timeout']['result_pips'].mean() if n_timeout else 0
    trail_pnl    = tr[tr['exit_type'] == 'trail']['pnl_usd'].sum()
    timeout_pnl  = tr[tr['exit_type'] == 'timeout']['pnl_usd'].sum()

    avg_win_pips  = wins['result_pips'].mean()  if len(wins)   else 0
    avg_loss_pips = losses['result_pips'].mean() if len(losses) else 0
    avg_hold      = tr['duration'].mean()

    print(f'\n--- Trade Statistics ---')
    print(f'  Total trades      : {n:,}  (~{n/months:.0f}/month)')
    print(f'  Longs / Shorts    : {n_long} / {n_short}')
    print(f'  Win / Loss        : {len(wins)} / {len(losses)}')
    print(f'  Win rate          : {tr["win"].mean():.1%}')
    print(f'  Avg win  ($)      : ${avg_win:>10,.2f}    Avg win  (pips): {avg_win_pips:>+.1f}')
    print(f'  Avg loss ($)      : ${avg_loss:>10,.2f}    Avg loss (pips): {avg_loss_pips:>+.1f}')
    print(f'  Avg P&L / trade   : ${avg_pnl:>10,.2f}    Avg pips:        {tr["result_pips"].mean():>+.1f}')
    print(f'  Profit factor     : {pf:.3f}')
    print(f'  Avg hold (bars)   : {avg_hold:.1f}h')

    print(f'\n--- MFE / MAE Analysis ---')
    print(f'  {"Metric":<30} {"Wins":>10} {"Losses":>10} {"Overall":>10}')
    print(f'  {"-"*62}')
    print(f'  {"Actual MFE (pips)":<30} {wins["actual_mfe"].mean():>+10.1f} {losses["actual_mfe"].mean():>+10.1f} {tr["actual_mfe"].mean():>+10.1f}')
    print(f'  {"MAE (pips)":<30} {wins["mae_pips"].mean():>+10.1f} {losses["mae_pips"].mean():>+10.1f} {tr["mae_pips"].mean():>+10.1f}')
    print(f'  {"Predicted MFE Q50 (pips)":<30} {wins["q50_mfe"].mean():>+10.1f} {losses["q50_mfe"].mean():>+10.1f} {tr["q50_mfe"].mean():>+10.1f}')
    print(f'  {"Result pips":<30} {avg_win_pips:>+10.1f} {avg_loss_pips:>+10.1f} {tr["result_pips"].mean():>+10.1f}')
    print(f'  Trail: {TRAIL_MULT}x atr_24 | Timeout: {TIMEOUT_H}h | MFE: first {MFE_HORIZON_H}h')

    print(f'\n--- Exit Type Breakdown ---')
    print(f'  {"Type":<13} {"N":>5} {"WR":>7} {"Avg_pips":>9} {"Total_$":>12}')
    print(f'  {"-"*50}')
    print(f'  {"trail":<13} {n_trail:>5} {trail_wr:>7.1%} {trail_avg:>+9.1f} ${trail_pnl:>11,.0f}')
    print(f'  {"timeout":<13} {n_timeout:>5} {timeout_wr:>7.1%} {timeout_avg:>+9.1f} ${timeout_pnl:>11,.0f}')

    # ── Dir mode breakdown ────────────────────────────────────────────────────
    tr['dir_label'] = np.where(tr['dir_pred'] > DIR_CONF_THRESH, 'LONG',
                      np.where(tr['dir_pred'] < 1 - DIR_CONF_THRESH, 'SHORT', 'DUAL'))
    print(f'\n--- Direction Mode Breakdown ---')
    print(f'  {"Mode":<6} {"N":>6} {"WR":>7} {"Avg_pips":>9} {"Total_$":>12}')
    print(f'  {"-"*44}')
    for mode in ['LONG', 'SHORT', 'DUAL']:
        s = tr[tr['dir_label'] == mode]
        if len(s) == 0: continue
        sw = s[s['win']]; sl = s[~s['win']]
        spf = sw['pnl_usd'].sum() / abs(sl['pnl_usd'].sum()) if len(sl) > 0 else 999
        print(f'  {mode:<6} {len(s):>6} {s["win"].mean():>7.1%} {s["result_pips"].mean():>+9.1f} ${s["pnl_usd"].sum():>11,.0f}  PF={spf:.3f}')

    # Long vs Short breakdown
    print(f'\n--- Long vs Short ---')
    for side in ['LONG', 'SHORT']:
        s  = tr[tr['direction'] == side]
        sw = s[s['win']]; sl = s[~s['win']]
        spf = sw['pnl_usd'].sum() / abs(sl['pnl_usd'].sum()) if len(sl) > 0 else 999
        print(f'  {side:<5}: N={len(s):>4}  WR={s["win"].mean():.1%}  '
              f'avg_pips={s["result_pips"].mean():>+.1f}  '
              f'avg_win=${sw["pnl_usd"].mean():>+,.0f}  '
              f'avg_loss=${sl["pnl_usd"].mean():>+,.0f}  PF={spf:.3f}')

    # Per-pair holding period
    print(f'\n--- Avg Holding Period per Pair ---')
    print(f'  {"Pair":<10} {"N":>5} {"Avg_hold(h)":>12} {"WR":>7} {"Avg_pips":>9} {"Avg_MFE":>9} {"Avg_MAE":>9}')
    print(f'  {"-"*70}')
    for pair in sorted(tr['pair'].unique()):
        s = tr[tr['pair'] == pair]
        print(f'  {pair:<10} {len(s):>5} {s["duration"].mean():>12.1f} '
              f'{s["win"].mean():>7.1%} {s["result_pips"].mean():>+9.1f} '
              f'{s["actual_mfe"].mean():>+9.1f} {s["mae_pips"].mean():>+9.1f}')

    # Per-pair
    print(f'\n--- Per-Pair Breakdown ---')
    print(f'  {"Pair":<10} {"N":>5} {"L/S":>7} {"WR":>7} {"Avg_pips":>9} {"Total_$":>10}')
    print(f'  {"-"*55}')
    for pair in sorted(tr['pair'].unique()):
        s  = tr[tr['pair'] == pair]
        nl = (s['direction'] == 'LONG').sum()
        ns = (s['direction'] == 'SHORT').sum()
        flag = ' <<<' if s['pnl_usd'].sum() > 0 else ''
        print(f'  {pair:<10} {len(s):>5} {str(nl)+"/"+str(ns):>7} '
              f'{s["win"].mean():>7.1%} {s["result_pips"].mean():>+9.1f} '
              f'${s["pnl_usd"].sum():>9,.0f}{flag}')

    # Monthly
    print(f'\n--- Monthly Breakdown ---')
    print(f'  {"Month":<10} {"N":>5} {"WR":>7} {"Pips":>8} {"P&L":>12} {"Cumul $":>12}')
    print(f'  {"-"*60}')
    cum = 0
    for month in sorted(tr['month'].unique()):
        s    = tr[tr['month'] == month]
        cum += s['pnl_usd'].sum()
        flag = ' <<<' if s['pnl_usd'].sum() > 0 else ''
        print(f'  {str(month):<10} {len(s):>5} {s["win"].mean():>7.1%} '
              f'{s["result_pips"].sum():>8.0f} ${s["pnl_usd"].sum():>11,.0f} '
              f'${cum:>11,.0f}{flag}')

    # Hourly
    print(f'\n--- Hourly Distribution ---')
    print(f'  {"Hour":>5} {"N":>5} {"L/S":>7} {"WR":>7} {"Avg_pips":>9}')
    print(f'  {"-"*38}')
    for h in range(24):
        s = tr[tr['hour'] == h]
        if len(s) == 0: continue
        nl = (s['direction'] == 'LONG').sum()
        ns = (s['direction'] == 'SHORT').sum()
        print(f'  {h:>5} {len(s):>5} {str(nl)+"/"+str(ns):>7} '
              f'{s["win"].mean():>7.1%} {s["result_pips"].mean():>+9.1f}')

    # Top wins / losses
    print(f'\n--- Top 5 Wins ---')
    for _, t in tr.nlargest(5, 'pnl_usd').iterrows():
        print(f'  {str(t["ts"])[:16]}  {t["pair"]:<8} {t["direction"]:<5} '
              f'{t["lots"]}L  {t["result_pips"]:+.0f}pips  ${t["pnl_usd"]:>+,.0f}')

    print(f'\n--- Top 5 Losses ---')
    for _, t in tr.nsmallest(5, 'pnl_usd').iterrows():
        print(f'  {str(t["ts"])[:16]}  {t["pair"]:<8} {t["direction"]:<5} '
              f'{t["lots"]}L  {t["result_pips"]:+.0f}pips  ${t["pnl_usd"]:>+,.0f}')

    # Full trade log
    print(f'\n{"="*80}')
    print(f'FULL TRADE LOG')
    print(f'{"="*80}')
    print(f'  {"#":>4} {"Time":<17} {"Pair":<8} {"Side":<5} {"Lots":>5} '
          f'{"Res_pips":>9} {"ActMFE":>7} {"MAE":>6} {"Exit":>7} {"P&L":>10} {"Equity":>12} {"MFE_q50":>9} {"dir_pred":>9}')
    print(f'  {"-"*112}')
    for i, (_, t) in enumerate(tr.iterrows(), 1):
        result = 'WIN' if t['win'] else 'LOSS'
        print(f'  {i:>4} {str(t["ts"])[:16]:<17} {t["pair"]:<8} {t["direction"]:<5} '
              f'{t["lots"]:>5.2f} {t["result_pips"]:>+9.1f} {t["actual_mfe"]:>+7.1f} {t["mae_pips"]:>+6.1f} {result:>7} '
              f'${t["pnl_usd"]:>+9,.0f} ${t["equity"]:>11,.0f} '
              f'{t["q50_mfe"]:>9.1f} {t["dir_pred"]:>9.4f}')

    # ── Directional accuracy (dir prob model vs actual 8h move) ─────────────
    # Use one row per signal bar — take the first direction fired per (ts, pair)
    tr_signal = tr.groupby(['ts', 'pair']).first().reset_index()
    tr_signal['dir_label']   = np.where(tr_signal['dir_pred'] > DIR_CONF_THRESH, 'LONG',
                               np.where(tr_signal['dir_pred'] < 1 - DIR_CONF_THRESH, 'SHORT', 'DUAL'))
    tr_signal['dir_correct'] = ((tr_signal['dir_label'] == 'LONG')  & (tr_signal['move_8h'] > 0)) | \
                               ((tr_signal['dir_label'] == 'SHORT') & (tr_signal['move_8h'] < 0)) | \
                               (tr_signal['dir_label'] == 'DUAL')    # dual: always skip for accuracy
    tr_signal_dir = tr_signal[tr_signal['dir_label'] != 'DUAL'].copy()
    tr_signal['abs_move_8h']     = tr_signal['move_8h'].abs()
    tr_signal_dir['abs_move_8h'] = tr_signal_dir['move_8h'].abs()

    print(f'\n{"="*80}')
    print(f'  DIR MODEL ACCURACY — dir_pred vs t+8h move (one row per signal bar)')
    print(f'{"="*80}')
    n_sig_long  = (tr_signal['dir_label'] == 'LONG').sum()
    n_sig_short = (tr_signal['dir_label'] == 'SHORT').sum()
    n_sig_dual  = (tr_signal['dir_label'] == 'DUAL').sum()
    print(f'  Signal bars: {len(tr_signal)}  (LONG={n_sig_long} SHORT={n_sig_short} DUAL={n_sig_dual})')

    if len(tr_signal_dir) > 0:
        print(f'\n--- Directional accuracy (LONG/SHORT only, N={len(tr_signal_dir)}) ---')
        print(f'  Overall: {tr_signal_dir["dir_correct"].mean():.1%}  '
              f'Avg |move|: {tr_signal_dir["abs_move_8h"].mean():.1f}p  '
              f'Avg move: {tr_signal_dir["move_8h"].mean():+.1f}p')

        for thresh_p in [5, 10, 20, 30]:
            sub = tr_signal_dir[tr_signal_dir['abs_move_8h'] >= thresh_p]
            if len(sub) == 0: continue
            print(f'  |move| >= {thresh_p:>2}p : N={len(sub):>4}  correct={sub["dir_correct"].mean():.1%}  avg_move={sub["move_8h"].mean():+.1f}p')

        print(f'\n--- By prediction label ---')
        for label in ['LONG', 'SHORT']:
            s = tr_signal_dir[tr_signal_dir['dir_label'] == label]
            if len(s) == 0: continue
            print(f'  {label:<5}: N={len(s):>4}  correct={s["dir_correct"].mean():.1%}  '
                  f'avg_move_8h={s["move_8h"].mean():+.1f}p  avg|move|={s["abs_move_8h"].mean():.1f}p')

        print(f'\n--- By pair (dir model accuracy, directional only) ---')
        print(f'  {"Pair":<10} {"N":>5} {"Label_L/S":>10} {"Correct%":>9} {"Avg_move":>9}')
        print(f'  {"-"*52}')
        for pair in sorted(tr_signal_dir['pair'].unique()):
            s  = tr_signal_dir[tr_signal_dir['pair'] == pair]
            nl = (s['dir_label'] == 'LONG').sum()
            ns = (s['dir_label'] == 'SHORT').sum()
            print(f'  {pair:<10} {len(s):>5} {str(nl)+"/"+str(ns):>10} '
                  f'{s["dir_correct"].mean():>9.1%} {s["move_8h"].mean():>+9.1f}p')

    print(f'\n--- t+8h move distribution ---')
    print(f'  {"Bucket":<15} {"N":>5}')
    print(f'  {"-"*24}')
    bins   = [(-9999,-50),(-50,-20),(-20,-5),(-5,5),(5,20),(20,50),(50,9999)]
    labels = ['<-50p','-50 to -20p','-20 to -5p','-5 to +5p','+5 to +20p','+20 to +50p','>+50p']
    for (lo, hi), lbl in zip(bins, labels):
        sub = tr_signal[(tr_signal['move_8h'] >= lo) & (tr_signal['move_8h'] < hi)]
        if len(sub) == 0: continue
        print(f'  {lbl:<15} {len(sub):>5}')

    print(f'\n{"="*80}')
    print(f'  STRATEGY : MFE Q50>={MFE_THRESH} | DIR PROB conf={DIR_CONF_THRESH} | Trail={TRAIL_MULT}x atr_24 | Timeout={TIMEOUT_H}h')
    print(f'  RESULT   : WR {tr["win"].mean():.1%} | EV {tr["result_pips"].mean():+.1f}pips | '
          f'PF {pf:.3f} | ${total_pnl/months:+,.0f}/month')
    print(f'{"="*80}')


if __name__ == '__main__':
    asyncio.run(main())
