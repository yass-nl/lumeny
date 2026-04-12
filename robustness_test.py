"""
Robustness test - 4 variants on same data/signals:

  BASELINE : current logic (high/low peak tracking, exit at trail level)
  A_CLOSE  : exit at closes[idx] instead of synthetic trail price
  B_LAGGED : running_max uses highs[idx-1] (peak definitely in past)
  C_RANDOM : same timestamps/ATR, random entry price from that bar's OHLC
"""

import os, asyncio, sys, random
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

MFE_THRESH    = 70.0
TRAIL_MULT    = 1.5
TIMEOUT_H     = 24
MFE_HORIZON_H = 8

PAIRS = [
    'EURUSD','GBPUSD','USDJPY','USDCHF','AUDUSD','USDCAD','NZDUSD',
    'EURJPY','GBPJPY','EURGBP','EURAUD','AUDJPY','CADJPY','CHFJPY','AUDNZD',
]
CURRENCY_SIGN = {
    'EURUSD':{'EUR':+1,'USD':-1},'GBPUSD':{'GBP':+1,'USD':-1},
    'USDJPY':{'USD':+1,'JPY':-1},'USDCHF':{'USD':+1,'CHF':-1},
    'AUDUSD':{'AUD':+1,'USD':-1},'USDCAD':{'USD':+1,'CAD':-1},
    'NZDUSD':{'NZD':+1,'USD':-1},'EURJPY':{'EUR':+1,'JPY':-1},
    'GBPJPY':{'GBP':+1,'JPY':-1},'EURGBP':{'EUR':+1,'GBP':-1},
    'EURAUD':{'EUR':+1,'AUD':-1},'AUDJPY':{'AUD':+1,'JPY':-1},
    'CADJPY':{'CAD':+1,'JPY':-1},'CHFJPY':{'CHF':+1,'JPY':-1},
    'AUDNZD':{'AUD':+1,'NZD':-1},
}
JPY_PAIRS        = {'USDJPY','EURJPY','GBPJPY','AUDJPY','CADJPY','CHFJPY'}
STARTING_CAPITAL = 100_000.0
LOT_UNITS        = 100_000
RISK_PER_TRADE   = 0.005
LEVERAGE         = 50
SPREAD_PIPS = {
    'EURUSD':0.6,'GBPUSD':0.8,'USDJPY':1.0,'USDCHF':0.7,
    'AUDUSD':0.6,'USDCAD':1.2,'NZDUSD':0.9,
    'EURJPY':1.4,'GBPJPY':2.1,'EURGBP':0.7,
    'EURAUD':2.1,'AUDJPY':1.5,'CADJPY':1.6,'CHFJPY':2.5,'AUDNZD':2.0,
}
QUOTE_USD = {'USD':1.00,'CHF':1.10,'CAD':0.73,'GBP':1.27,'AUD':0.65,'NZD':0.59,'JPY':0.0067}
BASE_USD  = {'EUR':1.08,'GBP':1.27,'AUD':0.65,'NZD':0.59,'USD':1.00,'CAD':0.73,'CHF':1.10}
FETCH_DAYS = 220
TEST_DAYS  = 180


def pip_value_per_lot(pair, entry_price):
    pip   = PIP_SIZE[pair]
    quote = pair[3:]
    if pair in JPY_PAIRS:
        return LOT_UNITS * pip / entry_price
    elif quote == 'USD':
        return LOT_UNITS * pip
    else:
        return LOT_UNITS * pip * QUOTE_USD.get(quote, 1.0)

def compute_pnl_usd(pair, direction, lots, entry_price, exit_price):
    move  = (exit_price - entry_price) if direction == 1 else (entry_price - exit_price)
    quote = pair[3:]
    if pair in JPY_PAIRS:
        return lots * LOT_UNITS * move / exit_price
    elif quote == 'USD':
        return lots * LOT_UNITS * move
    else:
        return lots * LOT_UNITS * move * QUOTE_USD.get(quote, 1.0)


# - 4 simulate_trade variants -

def simulate_baseline(df_1h, entry_pos, direction, trail_price, pip_size):
    """Current logic: peak via high/low, exit at synthetic trail level."""
    highs  = df_1h['high'].values
    lows   = df_1h['low'].values
    closes = df_1h['close'].values
    n      = len(df_1h)
    entry  = closes[entry_pos]
    if direction == 1:
        running_max = entry
        exit_price  = None; duration = 0; exit_type = 'timeout'
        for k in range(1, min(TIMEOUT_H+1, n-entry_pos)):
            idx = entry_pos + k
            running_max = max(running_max, highs[idx])
            if running_max - closes[idx] >= trail_price:
                exit_price = running_max - trail_price; duration = k; exit_type = 'trail'; break
        if exit_price is None:
            duration = min(TIMEOUT_H, n-entry_pos-1); exit_price = closes[entry_pos+duration]
        result_pips = (exit_price - entry) / pip_size
    else:
        running_min = entry
        exit_price  = None; duration = 0; exit_type = 'timeout'
        for k in range(1, min(TIMEOUT_H+1, n-entry_pos)):
            idx = entry_pos + k
            running_min = min(running_min, lows[idx])
            if closes[idx] - running_min >= trail_price:
                exit_price = running_min + trail_price; duration = k; exit_type = 'trail'; break
        if exit_price is None:
            duration = min(TIMEOUT_H, n-entry_pos-1); exit_price = closes[entry_pos+duration]
        result_pips = (entry - exit_price) / pip_size
    return result_pips, exit_price, duration, exit_type


def simulate_A_close(df_1h, entry_pos, direction, trail_price, pip_size):
    """Variant A: exit at closes[idx] - no synthetic price."""
    highs  = df_1h['high'].values
    lows   = df_1h['low'].values
    closes = df_1h['close'].values
    n      = len(df_1h)
    entry  = closes[entry_pos]
    if direction == 1:
        running_max = entry
        exit_price  = None; duration = 0; exit_type = 'timeout'
        for k in range(1, min(TIMEOUT_H+1, n-entry_pos)):
            idx = entry_pos + k
            running_max = max(running_max, highs[idx])
            if running_max - closes[idx] >= trail_price:
                exit_price = closes[idx]; duration = k; exit_type = 'trail'; break  # exit at close
        if exit_price is None:
            duration = min(TIMEOUT_H, n-entry_pos-1); exit_price = closes[entry_pos+duration]
        result_pips = (exit_price - entry) / pip_size
    else:
        running_min = entry
        exit_price  = None; duration = 0; exit_type = 'timeout'
        for k in range(1, min(TIMEOUT_H+1, n-entry_pos)):
            idx = entry_pos + k
            running_min = min(running_min, lows[idx])
            if closes[idx] - running_min >= trail_price:
                exit_price = closes[idx]; duration = k; exit_type = 'trail'; break  # exit at close
        if exit_price is None:
            duration = min(TIMEOUT_H, n-entry_pos-1); exit_price = closes[entry_pos+duration]
        result_pips = (entry - exit_price) / pip_size
    return result_pips, exit_price, duration, exit_type


def simulate_B_lagged(df_1h, entry_pos, direction, trail_price, pip_size):
    """Variant B: running_max uses highs[idx-1] - peak definitely in past."""
    highs  = df_1h['high'].values
    lows   = df_1h['low'].values
    closes = df_1h['close'].values
    n      = len(df_1h)
    entry  = closes[entry_pos]
    if direction == 1:
        running_max = entry
        exit_price  = None; duration = 0; exit_type = 'timeout'
        for k in range(1, min(TIMEOUT_H+1, n-entry_pos)):
            idx = entry_pos + k
            # Use previous bar's high to update peak - definitely in the past
            if k >= 2:
                running_max = max(running_max, highs[entry_pos + k - 1])
            if running_max - closes[idx] >= trail_price:
                exit_price = running_max - trail_price; duration = k; exit_type = 'trail'; break
        if exit_price is None:
            duration = min(TIMEOUT_H, n-entry_pos-1); exit_price = closes[entry_pos+duration]
        result_pips = (exit_price - entry) / pip_size
    else:
        running_min = entry
        exit_price  = None; duration = 0; exit_type = 'timeout'
        for k in range(1, min(TIMEOUT_H+1, n-entry_pos)):
            idx = entry_pos + k
            if k >= 2:
                running_min = min(running_min, lows[entry_pos + k - 1])
            if closes[idx] - running_min >= trail_price:
                exit_price = running_min + trail_price; duration = k; exit_type = 'trail'; break
        if exit_price is None:
            duration = min(TIMEOUT_H, n-entry_pos-1); exit_price = closes[entry_pos+duration]
        result_pips = (entry - exit_price) / pip_size
    return result_pips, exit_price, duration, exit_type


def simulate_random(df_1h, entry_pos, direction, trail_price, pip_size, random_entry):
    """Variant C: same trail logic as baseline but random entry price from signal bar OHLC."""
    highs  = df_1h['high'].values
    lows   = df_1h['low'].values
    closes = df_1h['close'].values
    n      = len(df_1h)
    entry  = random_entry  # random price instead of close
    if direction == 1:
        running_max = entry
        exit_price  = None; duration = 0; exit_type = 'timeout'
        for k in range(1, min(TIMEOUT_H+1, n-entry_pos)):
            idx = entry_pos + k
            running_max = max(running_max, highs[idx])
            if running_max - closes[idx] >= trail_price:
                exit_price = running_max - trail_price; duration = k; exit_type = 'trail'; break
        if exit_price is None:
            duration = min(TIMEOUT_H, n-entry_pos-1); exit_price = closes[entry_pos+duration]
        result_pips = (exit_price - entry) / pip_size
    else:
        running_min = entry
        exit_price  = None; duration = 0; exit_type = 'timeout'
        for k in range(1, min(TIMEOUT_H+1, n-entry_pos)):
            idx = entry_pos + k
            running_min = min(running_min, lows[idx])
            if closes[idx] - running_min >= trail_price:
                exit_price = running_min + trail_price; duration = k; exit_type = 'trail'; break
        if exit_price is None:
            duration = min(TIMEOUT_H, n-entry_pos-1); exit_price = closes[entry_pos+duration]
        result_pips = (entry - exit_price) / pip_size
    return result_pips, exit_price, duration, exit_type


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
    df = df[~((df.index.dayofweek==5)|((df.index.dayofweek==6)&(df.index.hour<21)))]
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
            raw[pair] = {'1m':df_1m,'5m':df_5m,'15m':df_15m,'1h':df_1h}
            print(f'{len(df_1m):,} 1m / {len(df_1h)} 1h bars')
    return raw, from_date, to_date

def compute_all_cross_pair_features(close_1h_all):
    returns_all = {p: np.log(c/c.shift(1)) for p,c in close_1h_all.items()}
    returns_df  = pd.DataFrame(returns_all)
    currencies  = ['EUR','USD','GBP','JPY','AUD','NZD','CAD','CHF']
    csi = {}
    for ccy in currencies:
        comps = [CURRENCY_SIGN[p][ccy]*returns_df[p]
                 for p in PAIRS if ccy in CURRENCY_SIGN.get(p,{}) and p in returns_df]
        if comps:
            csi[f'csi_{ccy.lower()}'] = pd.concat(comps,axis=1).mean(axis=1)
    csi_df = pd.DataFrame(csi)
    csi_rolling = {}
    for col in csi_df.columns:
        csi_rolling[f'{col}_24h'] = csi_df[col].rolling(24, min_periods=8).sum()
        csi_rolling[f'{col}_72h'] = csi_df[col].rolling(72, min_periods=24).sum()
    csi_rolling_df = pd.DataFrame(csi_rolling)
    result = {}
    for pair in PAIRS:
        if pair not in returns_df.columns: continue
        r = returns_df[pair]; c_pair = close_1h_all[pair]
        cols = {}
        for peer in [p for p in PAIRS if p != pair]:
            if peer not in returns_df.columns: continue
            p_ret = returns_df[peer]; c_peer = close_1h_all[peer]; sl = peer.lower()
            for w, lbl in [(24,'24h'),(72,'3d'),(168,'1w')]:
                cols[f'corr_{sl}_{lbl}'] = r.rolling(w, min_periods=w//2).corr(p_ret)
            cols[f'corr_regime_{sl}'] = cols[f'corr_{sl}_24h'] - cols[f'corr_{sl}_1w']
            for w, lbl in [(24,'24h'),(168,'1w')]:
                cov = r.rolling(w,min_periods=w//2).cov(p_ret)
                var = p_ret.rolling(w,min_periods=w//2).var().clip(lower=1e-12)
                cols[f'beta_{sl}_{lbl}'] = cov/var
            cols[f'relstr_{sl}_1h']  = r - p_ret
            cols[f'relstr_{sl}_4h']  = np.log(c_pair/c_pair.shift(4))  - np.log(c_peer/c_peer.shift(4))
            cols[f'relstr_{sl}_24h'] = np.log(c_pair/c_pair.shift(24)) - np.log(c_peer/c_peer.shift(24))
            cols[f'peer_{sl}_ret_1h']  = p_ret
            cols[f'peer_{sl}_ret_4h']  = np.log(c_peer/c_peer.shift(4))
            cols[f'peer_{sl}_ret_24h'] = np.log(c_peer/c_peer.shift(24))
        for col in csi_df.columns:
            cols[col]          = csi_df[col]
            cols[f'{col}_24h'] = csi_rolling_df[f'{col}_24h']
            cols[f'{col}_72h'] = csi_rolling_df[f'{col}_72h']
        result[pair] = pd.DataFrame(cols, index=r.index).astype(np.float32)
    return result


def run_capital_sim(signal_list):
    """Replay signals chronologically, return summary stats."""
    signals = sorted(signal_list, key=lambda x: x['ts'])
    equity = STARTING_CAPITAL; peak_equity = equity; max_dd_pct = 0.0
    pnls = []
    for sig in signals:
        trail_pips = sig['trail_pips']
        pvl        = sig['pvl']
        risk_usd   = equity * RISK_PER_TRADE
        lots       = risk_usd / (trail_pips * pvl) if (trail_pips * pvl) > 0 else 0.01
        lots       = max(0.01, round(lots, 2))
        pnl = compute_pnl_usd(sig['pair'], sig['direction'], lots, sig['entry_price'], sig['exit_price'])
        pnl -= lots * pvl * sig['sp_pips']
        equity += pnl
        if equity > peak_equity: peak_equity = equity
        dd_pct = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0
        max_dd_pct = max(max_dd_pct, dd_pct)
        pnls.append(pnl)
    total_pnl = equity - STARTING_CAPITAL
    months    = TEST_DAYS / 30
    wr        = np.mean([p > 0 for p in pnls])
    wins      = [p for p in pnls if p > 0]
    losses    = [p for p in pnls if p <= 0]
    pf        = sum(wins) / abs(sum(losses)) if losses else 999
    daily     = pd.Series(pnls)  # approximate
    sharpe    = (daily.mean() / daily.std() * np.sqrt(252*24/11.5)) if daily.std() > 0 else 0  # rough
    return {
        'total_pnl':  total_pnl,
        'pct':        total_pnl / STARTING_CAPITAL * 100,
        'per_month':  total_pnl / months,
        'max_dd_pct': max_dd_pct * 100,
        'wr':         wr * 100,
        'pf':         pf,
        'n_trades':   len(pnls),
    }


async def main():
    random.seed(42)
    np.random.seed(42)

    print('Loading MFE model...')
    mfe_bundle   = joblib.load('backend/models_9/mfe_q50_8h/model_1H_Q50.joblib')
    mfe_model    = mfe_bundle['model']
    feature_cols = mfe_bundle['feature_cols']
    dir_bundle   = joblib.load('backend/models_9/dir_q50_8h/model_1H_Q50.joblib')
    dir_model    = dir_bundle['model']
    dir_v1_cols  = dir_bundle['feature_cols']

    raw, from_date, to_date = await fetch_all_pairs()
    if not raw: print('No data.'); return

    test_start = datetime.now() - timedelta(days=TEST_DAYS)

    close_1h_all = {pair: raw[pair]['1h']['close'] for pair in raw}
    print('\nComputing cross-pair features...')
    cross_features = compute_all_cross_pair_features(close_1h_all)

    print('Computing per-pair features...')
    pair_features = {}
    for pair in PAIRS:
        if pair not in raw: continue
        df_base = compute_features_for_pair(pair, raw[pair]['1m'], raw[pair]['5m'], raw[pair]['15m'], df_1h=raw[pair]['1h'])
        if df_base.empty: continue
        df_extra = compute_momentum_calendar_features(raw[pair]['1h'], PIP_SIZE[pair])
        df_base  = df_base.join(df_extra.reindex(df_base.index), how='left')
        if pair in cross_features:
            df_base = df_base.join(cross_features[pair].reindex(df_base.index), how='left')
        pair_features[pair] = df_base

    # Collect signal bars (timestamps + metadata), same cooldown logic as main test
    print('\nCollecting signal bars...')
    signal_bars = []  # one entry per signal bar (not per direction)
    for pair in PAIRS:
        if pair not in pair_features: continue
        pip_size = PIP_SIZE[pair]
        sp_pips  = SPREAD_PIPS.get(pair, 2.0)
        df_feat  = pair_features[pair]
        df_1h    = raw[pair]['1h']
        df_test  = df_feat[df_feat.index >= test_start].copy()
        if len(df_test) < 10: continue
        common_idx = df_test.index.intersection(df_1h.index)
        df_test    = df_test.loc[common_idx]
        for col in feature_cols:
            if col not in df_test.columns: df_test[col] = 0.0
        for col in dir_v1_cols:
            if col not in df_test.columns: df_test[col] = 0.0
        df_test = df_test.copy()
        df_test['q50_mfe'] = mfe_model.predict(df_test[feature_cols].ffill().fillna(0))
        df_test['q50_dir'] = dir_model.predict(df_test[dir_v1_cols].ffill().fillna(0))
        df_1h_full     = df_1h.copy()
        cooldown_until = -1
        for ts, row in df_test.iterrows():
            if row['q50_mfe'] < MFE_THRESH: continue
            atr_24 = row.get('atr_24', np.nan)
            if np.isnan(atr_24) or atr_24 <= 0: continue
            try:
                pos = df_1h_full.index.get_loc(ts)
            except KeyError:
                continue
            if pos <= cooldown_until: continue
            entry_price    = df_1h_full['close'].iloc[pos]
            trail_pips_raw = TRAIL_MULT * atr_24
            trail_price    = trail_pips_raw * pip_size
            pvl            = pip_value_per_lot(pair, entry_price)
            bar_o = df_1h_full['open'].iloc[pos]
            bar_h = df_1h_full['high'].iloc[pos]
            bar_l = df_1h_full['low'].iloc[pos]
            bar_c = df_1h_full['close'].iloc[pos]
            # Random entry: uniform within bar's range
            rand_entry = bar_l + random.random() * (bar_h - bar_l)
            signal_bars.append({
                'ts': ts, 'pair': pair, 'pos': pos,
                'entry_price': entry_price, 'rand_entry': rand_entry,
                'trail_pips': trail_pips_raw, 'trail_price': trail_price,
                'sp_pips': sp_pips, 'pvl': pvl, 'pip_size': pip_size,
                'df_1h': df_1h_full,
            })
            # Cooldown: use max duration across both directions (baseline logic)
            max_dur = 0
            for direction in (1, -1):
                _, _, dur, _ = simulate_baseline(df_1h_full, pos, direction, trail_price, pip_size)
                max_dur = max(max_dur, dur)
            cooldown_until = pos + max_dur

    print(f'  {len(signal_bars)} signal bars collected')

    # ── D_RANDOM_BARS: same pairs, same hours, same N, random bar selection ──
    # Match exact pair distribution and trade count from signal_bars
    from collections import Counter
    pair_counts = Counter(sb['pair'] for sb in signal_bars)
    signal_hours = sorted(set(sb['ts'].hour for sb in signal_bars))

    random_bars = []
    for pair, count in pair_counts.items():
        if pair not in pair_features: continue
        pip_size = PIP_SIZE[pair]
        sp_pips  = SPREAD_PIPS.get(pair, 2.0)
        df_1h    = raw[pair]['1h']
        df_feat  = pair_features[pair]
        # Candidate bars: test window, same hours as signal bars, atr_24 valid
        df_test  = df_feat[df_feat.index >= test_start].copy()
        common   = df_test.index.intersection(df_1h.index)
        df_test  = df_test.loc[common]
        df_test  = df_test[df_test.index.hour.isin(signal_hours)]
        df_test  = df_test[df_test['atr_24'].notna() & (df_test['atr_24'] > 0)]
        if len(df_test) < count: continue
        # Sample without replacement, respecting cooldown
        df_1h_full     = df_1h.copy()
        candidates = df_test.index.tolist()
        random.shuffle(candidates)
        selected   = []
        for ts in candidates:
            if len(selected) >= count: break
            try:
                pos = df_1h_full.index.get_loc(ts)
            except KeyError:
                continue
            atr_24         = df_test.loc[ts, 'atr_24']
            trail_pips_raw = TRAIL_MULT * atr_24
            trail_price    = trail_pips_raw * pip_size
            entry_price    = df_1h_full['close'].iloc[pos]
            pvl            = pip_value_per_lot(pair, entry_price)
            selected.append({
                'ts': ts, 'pair': pair, 'pos': pos,
                'entry_price': entry_price, 'rand_entry': entry_price,
                'trail_pips': trail_pips_raw, 'trail_price': trail_price,
                'sp_pips': sp_pips, 'pvl': pvl, 'pip_size': pip_size,
                'df_1h': df_1h_full,
            })
        random_bars.extend(selected)

    print(f'  {len(random_bars)} random bars collected (target: {sum(pair_counts.values())})')

    # Build signal lists for each variant
    variants = {'BASELINE': [], 'A_CLOSE': [], 'B_LAGGED': [], 'C_RANDOM': [], 'D_RAND_BARS': []}

    for sb in signal_bars:
        pos         = sb['pos']
        pair        = sb['pair']
        pip_size    = sb['pip_size']
        trail_price = sb['trail_price']
        df_1h       = sb['df_1h']
        entry_price = sb['entry_price']
        rand_entry  = sb['rand_entry']
        sp_pips     = sb['sp_pips']

        for direction in (1, -1):
            # BASELINE
            res_pips, exit_px, dur, et = simulate_baseline(df_1h, pos, direction, trail_price, pip_size)
            variants['BASELINE'].append({**sb, 'direction': direction, 'exit_price': exit_px})

            # A: close exit
            res_pips, exit_px, dur, et = simulate_A_close(df_1h, pos, direction, trail_price, pip_size)
            variants['A_CLOSE'].append({**sb, 'direction': direction, 'exit_price': exit_px})

            # B: lagged high
            res_pips, exit_px, dur, et = simulate_B_lagged(df_1h, pos, direction, trail_price, pip_size)
            variants['B_LAGGED'].append({**sb, 'direction': direction, 'exit_price': exit_px})

            # C: random entry, baseline exit logic
            res_pips, exit_px, dur, et = simulate_random(df_1h, pos, direction, trail_price, pip_size, rand_entry)
            variants['C_RANDOM'].append({**sb, 'direction': direction,
                                          'entry_price': rand_entry, 'exit_price': exit_px})

    # D: random bars
    for sb in random_bars:
        pos         = sb['pos']
        pip_size    = sb['pip_size']
        trail_price = sb['trail_price']
        df_1h       = sb['df_1h']
        for direction in (1, -1):
            res_pips, exit_px, dur, et = simulate_baseline(df_1h, pos, direction, trail_price, pip_size)
            variants['D_RAND_BARS'].append({**sb, 'direction': direction, 'exit_price': exit_px})

    # Print comparison
    print(f'\n{"="*75}')
    print(f'  ROBUSTNESS TEST - {len(signal_bars)} signal bars - 2 sides = {len(signal_bars)*2} legs each')
    print(f'{"="*75}')
    print(f'  {"Variant":<12} {"Total_PnL":>12} {"Pct%":>8} {"$/month":>10} {"MaxDD%":>8} {"WR%":>7} {"PF":>6} {"N":>5}')
    print(f'  {"-"*72}')
    for name, sigs in variants.items():
        r = run_capital_sim(sigs)
        print(f'  {name:<12} ${r["total_pnl"]:>11,.0f} {r["pct"]:>7.1f}% ${r["per_month"]:>9,.0f} '
              f'{r["max_dd_pct"]:>7.1f}% {r["wr"]:>6.1f}% {r["pf"]:>6.3f} {r["n_trades"]:>5}')
    print(f'{"="*75}')
    print()
    print('  INTERPRETATION:')
    print('  - B_LAGGED similar to BASELINE => no lookahead in peak tracking')
    print('  - C_RANDOM similar to BASELINE => entry price within bar does not matter')
    print('  - D_RAND_BARS: if similar to BASELINE => MFE model adds no value')
    print('  - D_RAND_BARS: if collapses       => MFE model genuinely selects special bars')


if __name__ == '__main__':
    asyncio.run(main())
