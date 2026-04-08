"""
Live Full Test — Capital Simulation
=====================================
MFE model (Q50 >= 70 pips) + Asymmetric directional model:
  - default direction: SHORT
  - override to LONG only when dir model Q50_dir >= LONG_THRESH (6 bps)

Pipeline:
1. Fetch 1M OHLCV for all 15 pairs (FETCH_DAYS for warm-up)
2. Compute all features (microstructure + momentum/calendar + cross-pair)
3. MFE Q50 >= 70 filter
4. Direction: LONG if q50_dir >= 0.0006, else SHORT
5. Cooldown: no re-entry on a pair while trade is open (COOLDOWN_H = 8h)
6. Trail stop: 1.5x ATR (matches parquet trail_stop_pips generation)
7. Capital simulation: $100k, 0.5% risk/trade, ATR-based sizing
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
MFE_THRESH   = 70.0          # MFE Q50 >= this to consider a bar
LONG_THRESH  = 6 / 10000     # dir model Q50_dir >= this -> LONG, else SHORT
COOLDOWN_H   = 8             # hours to lock pair after entry
TRAIL_MULT   = 1.5           # trail stop = 1.5 x atr_24 (matches parquet labels)
TIMEOUT_H    = 24            # max trade duration hours

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
def simulate_trailing_stop(df_1h, entry_pos, direction, trail_price, pip_size):
    """
    Simulate trailing stop from entry_pos.
    Returns (mfe_pips, exit_price, duration_bars, exit_type)
    - trail exit_price = exact stop level (running_max - trail or running_min + trail)
    - timeout exit_price = bar close at timeout
    trail_price: absolute price distance for the trailing stop
    """
    highs  = df_1h['high'].values
    lows   = df_1h['low'].values
    closes = df_1h['close'].values
    n      = len(df_1h)
    entry  = closes[entry_pos]

    if direction == 1:
        running_max = entry
        for k in range(1, min(TIMEOUT_H + 1, n - entry_pos)):
            idx = entry_pos + k
            running_max = max(running_max, highs[idx])
            if running_max - lows[idx] >= trail_price:
                exit_price = running_max - trail_price  # exact stop level
                return (running_max - entry) / pip_size, exit_price, k, 'trail'
        exit_price = closes[entry_pos + min(TIMEOUT_H, n - entry_pos - 1)]
        return (running_max - entry) / pip_size, exit_price, min(TIMEOUT_H, n - entry_pos - 1), 'timeout'
    else:
        running_min = entry
        for k in range(1, min(TIMEOUT_H + 1, n - entry_pos)):
            idx = entry_pos + k
            running_min = min(running_min, lows[idx])
            if highs[idx] - running_min >= trail_price:
                exit_price = running_min + trail_price  # exact stop level
                return (entry - running_min) / pip_size, exit_price, k, 'trail'
        exit_price = closes[entry_pos + min(TIMEOUT_H, n - entry_pos - 1)]
        return (entry - running_min) / pip_size, exit_price, min(TIMEOUT_H, n - entry_pos - 1), 'timeout'


# ── Main ──────────────────────────────────────────────────────────────────────
async def main():
    # ── Load models ───────────────────────────────────────────────────────────
    print('Loading models...')
    mfe_bundle   = joblib.load('backend/models_9/mfe_q50_8h/model_1H_Q50.joblib')
    mfe_model    = mfe_bundle['model']
    feature_cols = mfe_bundle['feature_cols']

    dir_bundle = joblib.load('backend/models_9/dir_q50_8h/model_1H_Q50.joblib')
    dir_model  = dir_bundle['model']

    print(f'  MFE model : {len(feature_cols)} features, {mfe_bundle["n_iters"]} iters')
    print(f'  Dir model : {dir_bundle["n_iters"]} iters, trained on MFE>={dir_bundle["mfe_thresh"]}')
    print(f'\nStrategy config:')
    print(f'  MFE threshold  : Q50 >= {MFE_THRESH} pips')
    print(f'  Long threshold : Q50_dir >= {LONG_THRESH*10000:.0f} bps  (else SHORT)')
    print(f'  Cooldown       : {COOLDOWN_H}h per pair')
    print(f'  Trail stop     : {TRAIL_MULT}x ATR_24')

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

        # Run both models
        X        = df_test_pair[feature_cols].ffill().fillna(0)
        q50_mfe  = mfe_model.predict(X)
        q50_dir  = dir_model.predict(X)
        df_test_pair = df_test_pair.copy()
        df_test_pair['q50_mfe'] = q50_mfe
        df_test_pair['q50_dir'] = q50_dir

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
                continue

            # Direction: LONG if q50_dir >= LONG_THRESH, else SHORT
            direction = 1 if row['q50_dir'] >= LONG_THRESH else -1

            # Trail stop price distance (atr_24 is in pips from feature vector, convert back to price)
            trail_price = TRAIL_MULT * atr_24 * pip_size

            # Simulate trailing stop on 1h bars
            mfe_pips, exit_price, duration, exit_type = simulate_trailing_stop(
                df_1h_full, pos, direction, trail_price, pip_size
            )

            entry_price = df_1h_full['close'].iloc[pos]

            result_pips = mfe_pips - (trail_price / pip_size) - sp_pips
            win         = result_pips > 0

            # Set cooldown to end of this trade
            cooldown_until = pos + duration

            pvl = pip_value_per_lot(pair, entry_price)

            all_signals.append({
                'ts':          ts,
                'pair':        pair,
                'direction':   direction,
                'entry_price': entry_price,
                'exit_price':  exit_price,
                'atr_24':      atr_24,
                'trail_price': trail_price,
                'result_pips': result_pips,
                'mfe_pips':    mfe_pips,
                'exit_type':   exit_type,
                'duration':    duration,
                'sp_pips':     sp_pips,
                'pvl':         pvl,
                'win':         win,
                'q50_mfe':     row['q50_mfe'],
                'q50_dir':     row['q50_dir'],
            })

    print(f'  Total signals: {len(all_signals)}')
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

        # Position sizing: risk RISK_PER_TRADE of equity on trail stop distance
        trail_pips   = sig['trail_price'] / pip_size
        risk_usd     = equity * RISK_PER_TRADE
        lots         = risk_usd / (trail_pips * pvl) if (trail_pips * pvl) > 0 else 0.01
        lots         = max(0.01, round(lots, 2))

        pnl_usd  = compute_pnl_usd(pair, direction, lots, entry_price, exit_price)
        sp_cost  = lots * pvl * sp_pips
        pnl_usd -= sp_cost

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
            'result_pips': sig['result_pips'],
            'mfe_pips':    sig['mfe_pips'],
            'exit_type':   sig['exit_type'],
            'duration':    sig['duration'],
            'sp_cost':     sp_cost,
            'pnl_usd':     pnl_usd,
            'equity':      equity,
            'q50_mfe':     sig['q50_mfe'],
            'q50_dir':     sig['q50_dir'],
            'win':         sig['win'],
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
    print(f'  CAPITAL SIMULATION — MFE Q50>={MFE_THRESH} + ASYMMETRIC DIR MODEL')
    print(f'  DEFAULT SHORT, LONG when Q50_dir >= {LONG_THRESH*10000:.0f} bps')
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

    print(f'\n--- Trade Statistics ---')
    print(f'  Total trades      : {n:,}  (~{n/months:.0f}/month)')
    print(f'  Longs / Shorts    : {n_long} / {n_short}')
    print(f'  Win / Loss        : {len(wins)} / {len(losses)}')
    print(f'  Win rate          : {tr["win"].mean():.1%}')
    print(f'  Avg win           : ${avg_win:>10,.2f}')
    print(f'  Avg loss          : ${avg_loss:>10,.2f}')
    print(f'  Avg P&L / trade   : ${avg_pnl:>10,.2f}')
    print(f'  Profit factor     : {pf:.3f}')
    print(f'  Avg pips / trade  : {tr["result_pips"].mean():+.1f}')

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
          f'{"Res_pips":>9} {"Exit":>7} {"P&L":>10} {"Equity":>12} {"MFE_q50":>9} {"Dir_q50":>9}')
    print(f'  {"-"*100}')
    for i, (_, t) in enumerate(tr.iterrows(), 1):
        result = 'WIN' if t['win'] else 'LOSS'
        print(f'  {i:>4} {str(t["ts"])[:16]:<17} {t["pair"]:<8} {t["direction"]:<5} '
              f'{t["lots"]:>5.2f} {t["result_pips"]:>+9.1f} {result:>7} '
              f'${t["pnl_usd"]:>+9,.0f} ${t["equity"]:>11,.0f} '
              f'{t["q50_mfe"]:>9.1f} {t["q50_dir"]*10000:>+9.2f}bps')

    print(f'\n{"="*80}')
    print(f'  STRATEGY : MFE Q50>={MFE_THRESH} | SHORT default | LONG if Q50_dir>={LONG_THRESH*10000:.0f}bps')
    print(f'  RESULT   : WR {tr["win"].mean():.1%} | EV {tr["result_pips"].mean():+.1f}pips | '
          f'PF {pf:.3f} | ${total_pnl/months:+,.0f}/month')
    print(f'{"="*80}')


if __name__ == '__main__':
    asyncio.run(main())
