"""
LumenY — Paper Trading Tracker

Two jobs:
  1. log_predictions()  — runs every hour, logs model predictions to SQLite
  2. resolve_outcomes() — runs every hour, checks past predictions and records actual outcomes

Usage:
  python paper_trading.py log      → log predictions right now
  python paper_trading.py resolve  → resolve any mature predictions
  python paper_trading.py report   → print calibration report
  python paper_trading.py run      → run both jobs in a loop (every hour)

Setup (run once):
  python paper_trading.py setup
"""

import argparse
import asyncio
import logging
import os
import sqlite3
import time
import warnings
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

# ── Paths ────────────────────────────────────────────────────────────────────

BACKEND_DIR  = Path(__file__).parent
VOLUME_DIR   = Path(os.environ.get('VOLUME_PATH', str(BACKEND_DIR)))
DB_PATH      = VOLUME_DIR / 'paper_trading.db'
LOG_PATH     = BACKEND_DIR / 'paper_trading.log'

# ── Logging ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[
        logging.FileHandler(LOG_PATH),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

HORIZONS = ['1H', '4H', '1D']

# How many hours until each horizon's prediction matures.
# label_1H = log(close[T+1] / close[T]) where T is the last closed 1H bar.
# close[T]   = price at last_candle_time + 1H  (= entry price, bar T close)
# close[T+1] = price at last_candle_time + 2H  (= exit price, bar T+1 close)
# _fetch_close_price fetches bar with open < matures_at and returns its close.
# So matures_at must point to the close time of the exit bar,
# and we wait until that time has passed before resolving.
# We set matures_at = last_candle_time + (n+1)H so the resolve query only fires
# after the exit bar has closed:
#   1H:  T + 2H  (exit bar opens T+1H, closes T+2H)
#   4H:  T + 5H  (exit bar opens T+4H, closes T+5H)
#   1D:  T + 25H (exit bar opens T+24H, closes T+25H)
HORIZON_HOURS = {
    '1H':  2,
    '4H':  5,
    '1D':  25,
}

# ── Database ─────────────────────────────────────────────────────────────────

def get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def setup_db():
    """Create tables if they don't exist."""
    conn = get_db()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS predictions (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            logged_at        TEXT NOT NULL,          -- UTC ISO timestamp when logged
            pair             TEXT NOT NULL,
            horizon          TEXT NOT NULL,
            direction        TEXT NOT NULL,          -- bullish / bearish / neutral
            probability      REAL NOT NULL,          -- calibrated dominant probability
            p_down           REAL NOT NULL,          -- calibrated P(down)
            p_up             REAL NOT NULL,          -- calibrated P(up)
            raw_p_down       REAL NOT NULL,
            signal_strength  TEXT NOT NULL,
            expected_move    REAL NOT NULL,          -- Q50 in %
            q10              REAL NOT NULL,
            q25              REAL NOT NULL,
            q50              REAL NOT NULL,
            q75              REAL NOT NULL,
            q90              REAL NOT NULL,
            cone_inner_low   REAL NOT NULL,
            cone_inner_high  REAL NOT NULL,
            cone_outer_low   REAL NOT NULL,
            cone_outer_high  REAL NOT NULL,
            low_conviction   INTEGER NOT NULL,       -- 0/1
            quantile_spread  REAL NOT NULL,
            entry_price      REAL,                   -- price at time of prediction
            matures_at       TEXT NOT NULL,          -- UTC ISO timestamp when to resolve
            -- Filled in after resolution:
            resolved_at      TEXT,
            exit_price       REAL,
            actual_return    REAL,                   -- log return at horizon
            correct          INTEGER,                -- 1 if direction was right, 0 if not
            notes            TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_matures_at   ON predictions(matures_at);
        CREATE INDEX IF NOT EXISTS idx_pair_horizon ON predictions(pair, horizon);
        CREATE INDEX IF NOT EXISTS idx_resolved     ON predictions(resolved_at);

        CREATE TABLE IF NOT EXISTS hourly_log (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            logged_at   TEXT NOT NULL,
            pairs_logged INTEGER NOT NULL,
            errors      TEXT
        );
    """)
    conn.commit()
    conn.close()
    logger.info(f'Database ready at {DB_PATH}')


# ── Inference ─────────────────────────────────────────────────────────────────

async def _init_buffer():
    """Initialize a CandleBuffer with fresh data from Polygon."""
    import sys
    sys.path.insert(0, str(BACKEND_DIR))

    from dotenv import load_dotenv
    load_dotenv(BACKEND_DIR.parent / '.env')

    from data_service import CandleBuffer
    buf = CandleBuffer()
    await buf.initialize()
    return buf


def _run_inference_sync(buf) -> tuple[dict, dict]:
    """
    Run inference for all pairs using an existing CandleBuffer.
    Returns (predictions_by_pair, prices_by_pair)
    Synchronous — call via asyncio.to_thread() to avoid blocking the event loop.
    """
    import sys
    sys.path.insert(0, str(BACKEND_DIR))

    from data_service import PAIRS
    from inference import Predictor
    from features import build_feature_row

    PAIR_IDS = {pair: i for i, pair in enumerate(PAIRS)}

    predictor = Predictor()

    closes_1h = buf.get_all_1h_closes()
    predictions = {}
    prices = {}

    for pair in PAIRS:
        try:
            ohlcv_raw = buf.get_ohlcv(pair)
            if not ohlcv_raw or '1H' not in ohlcv_raw:
                logger.warning(f'No 1H data for {pair}')
                continue

            # CandleBuffer.initialize() only keeps fully-closed bars for fetched
            # timeframes (5m, 15m, 1H).  For 4H and 1D (resampled from 1H),
            # the trailing bar may be partial — but the training pipeline also
            # used the current-day 1D bar (with future data baked in), so using
            # the partial bar is the closest live approximation.
            ohlcv = dict(ohlcv_raw)

            features_df = build_feature_row(
                ohlcv, closes_1h, pair, PAIR_IDS[pair],
                expected_cols=predictor.feature_cols,
            )

            if features_df.empty:
                continue

            result = predictor.predict(features_df, pair)
            predictions[pair] = result

            # Entry price = close of last complete 1H candle (after trim)
            latest_close = float(ohlcv['1H']['close'].iloc[-1])
            prices[pair] = latest_close

        except Exception as e:
            logger.error(f'Inference error for {pair}: {e}', exc_info=True)

    return predictions, prices


async def _run_inference(buf) -> tuple[dict, dict]:
    """Run inference in a thread so it doesn't block the event loop / dashboard."""
    return await asyncio.to_thread(_run_inference_sync, buf)


# ── Log predictions ───────────────────────────────────────────────────────────

async def log_predictions(buf):
    """Fetch predictions for all pairs and log them to the database."""
    # Guard: skip during weekend market closure.
    # Market closed: all day Saturday, Sunday before 21:00 UTC (Sydney open).
    now = datetime.now(timezone.utc)
    if now.weekday() == 5 or (now.weekday() == 6 and now.hour < 21):
        logger.info(f'Market closed (weekend) — skipping inference.')
        return 0

    logger.info('Running inference...')
    predictions, prices = await _run_inference(buf)

    now = datetime.now(timezone.utc)
    conn = get_db()
    logged = 0
    errors = []

    for pair, result in predictions.items():
        entry_price = prices.get(pair)

        # Get the timestamp of the last complete 1H candle — this is "T" in the
        # backtest sense.  CandleBuffer already dropped the incomplete bar, so
        # index[-1] is the last fully-closed candle.
        ohlcv = buf.get_ohlcv(pair)
        if ohlcv and '1H' in ohlcv and not ohlcv['1H'].empty:
            last_candle_time = ohlcv['1H'].index[-1].to_pydatetime()
            if last_candle_time.tzinfo is None:
                last_candle_time = last_candle_time.replace(tzinfo=timezone.utc)
        else:
            last_candle_time = now  # fallback

        for horizon in HORIZONS:
            if horizon not in result['horizons']:
                continue

            # Frequency guard: 4H logs every 4 hours, 1D logs every 24 hours.
            # Cycle runs at XX:05 — check the current hour.
            if horizon == '4H' and now.hour % 4 != 0:
                continue
            if horizon == '1D' and now.hour != 0:
                continue

            h = result['horizons'][horizon]
            # last_candle_time = bar OPEN time e.g. T=03:00 (bar closes at 04:00).
            # Entry price = close of that bar = price at 04:00.
            # Label = log(close[T+1] / close[T]):
            #   close[T]   = entry price @ 04:00
            #   close[T+1] = exit price  @ 05:00 (bar T+1 opens 04:00, closes 05:00)
            # _fetch_close_price finds bar with open < matures_at → returns close.
            # matures_at = 03:00 + 2H = 05:00:
            #   → finds bar opening at 04:00 (last bar with t < 05:00)
            #   → returns its close = 05:00 price ✓
            #   → resolve query fires only after 05:00 ✓
            matures_at = last_candle_time + timedelta(hours=HORIZON_HOURS[horizon])

            try:
                # Skip if a prediction for this exact candle already exists (resolved or not)
                existing = conn.execute("""
                    SELECT id FROM predictions
                    WHERE pair = ? AND horizon = ? AND matures_at = ?
                """, (pair, horizon, matures_at.isoformat())).fetchone()
                if existing:
                    continue


                conn.execute("""
                    INSERT INTO predictions (
                        logged_at, pair, horizon, direction, probability,
                        p_down, p_up, raw_p_down, signal_strength,
                        expected_move, q10, q25, q50, q75, q90,
                        cone_inner_low, cone_inner_high,
                        cone_outer_low, cone_outer_high,
                        low_conviction, quantile_spread,
                        entry_price, matures_at
                    ) VALUES (
                        ?, ?, ?, ?, ?,
                        ?, ?, ?, ?,
                        ?, ?, ?, ?, ?, ?,
                        ?, ?,
                        ?, ?,
                        ?, ?,
                        ?, ?
                    )
                """, (
                    now.isoformat(),
                    pair,
                    horizon,
                    h['direction'],
                    h['probability'],
                    h['calibrated_p_down'],
                    h['calibrated_p_up'],
                    h['raw_p_down'],
                    h['signal_strength'],
                    h['expected_move_pct'],
                    h['quantiles']['Q10'],
                    h['quantiles']['Q25'],
                    h['quantiles']['Q50'],
                    h['quantiles']['Q75'],
                    h['quantiles']['Q90'],
                    h['cone']['inner'][0],
                    h['cone']['inner'][1],
                    h['cone']['outer'][0],
                    h['cone']['outer'][1],
                    0,
                    h['quantile_spread'],
                    entry_price,
                    matures_at.isoformat(),
                ))
                logged += 1

            except Exception as e:
                msg = f'{pair}/{horizon}: {e}'
                errors.append(msg)
                logger.error(msg)

    conn.execute(
        "INSERT INTO hourly_log (logged_at, pairs_logged, errors) VALUES (?, ?, ?)",
        (now.isoformat(), logged, '; '.join(errors) if errors else None)
    )
    conn.commit()
    conn.close()

    logger.info(f'Logged {logged} predictions for {len(predictions)} pairs. (candle T={last_candle_time.isoformat()})')
    return logged


# ── Resolve outcomes ──────────────────────────────────────────────────────────

async def _fetch_close_price(pair: str, at_time: datetime) -> float | None:
    """
    Fetch the 1H candle close price at a specific time directly from Polygon REST.
    No iloc[:-1] — this is just a price lookup for resolution, not feature computation.
    """
    import httpx

    api_key = os.environ.get('POLYGON_API_KEY', os.environ.get('POLYGON_S3_SECRET_KEY', ''))
    ticker = f'C:{pair}'

    from_date = (at_time - timedelta(days=1)).strftime('%Y-%m-%d')
    to_date = (at_time + timedelta(days=1)).strftime('%Y-%m-%d')

    url = f'https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/hour/{from_date}/{to_date}'
    params = {'apiKey': api_key, 'limit': 50000, 'sort': 'asc'}

    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()

        results = data.get('results', [])
        logger.info(f'Polygon {pair} at {at_time}: {len(results)} bars returned (status={data.get("status")})')
        if not results:
            return None

        # Polygon bar['t'] = bar open time (ms).  A 1H bar starting at 22:00
        # covers 22:00-23:00, its close = price at 23:00.
        # We want the bar whose CLOSE corresponds to at_time, i.e. the bar
        # that OPENED before at_time.  Use strict < so that at_time=05:00
        # picks the bar opening at 04:00 (close = 05:00), not 05:00 (close = 06:00).
        target_ts = int(at_time.timestamp() * 1000)
        best = None
        for bar in results:
            if bar['t'] < target_ts:
                best = bar
            else:
                break  # results are sorted asc, no need to continue

        if best is not None:
            return float(best['c'])

        # Fallback: at_time is before all bars (shouldn't happen with ±2h window)
        return float(results[0]['c'])

    except Exception as e:
        logger.warning(f'Failed to fetch close price for {pair} at {at_time}: {e}')
        return None


async def resolve_outcomes():
    """
    Find predictions that have matured (matures_at <= now) and not yet resolved.
    Fetches actual prices directly from Polygon REST (not from buffer).
    """
    now = datetime.now(timezone.utc)
    conn = get_db()

    pending = conn.execute("""
        SELECT * FROM predictions
        WHERE resolved_at IS NULL
          AND matures_at <= ?
    """, (now.isoformat(),)).fetchall()

    if not pending:
        logger.info('No pending predictions to resolve.')
        conn.close()
        return 0

    logger.info(f'Resolving {len(pending)} matured predictions...')

    resolved = 0

    for row in pending:
        pair    = row['pair']
        horizon = row['horizon']

        try:
            matures_at = datetime.fromisoformat(row['matures_at'])

            exit_price = await _fetch_close_price(pair, matures_at)
            if exit_price is None:
                logger.warning(f'No candle yet for {pair}/{horizon} matured at {matures_at}')
                continue

            entry_price  = row['entry_price']

            if entry_price is None or entry_price == 0:
                logger.warning(f'No entry price for prediction {row["id"]}')
                continue

            actual_return = np.log(exit_price / entry_price) * 100  # log return in %, matches training labels

            # Was the directional prediction correct?
            # Predictions below 60% confidence are non-signals — still resolved
            # (exit_price + actual_return recorded) but correct is left NULL.
            direction = row['direction']
            probability = row['probability']
            if probability < 0.60:
                correct = None
            elif direction == 'bullish':
                correct = int(actual_return > 0)
            elif direction == 'bearish':
                correct = int(actual_return < 0)
            else:
                correct = None

            conn.execute("""
                UPDATE predictions
                SET resolved_at  = ?,
                    exit_price   = ?,
                    actual_return = ?,
                    correct      = ?
                WHERE id = ?
            """, (
                now.isoformat(),
                exit_price,
                round(actual_return, 6),
                correct,
                row['id'],
            ))
            resolved += 1

        except Exception as e:
            logger.error(f'Resolution error for prediction {row["id"]}: {e}', exc_info=True)

    conn.commit()
    conn.close()
    logger.info(f'Resolved {resolved} predictions.')
    return resolved


# ── Report ────────────────────────────────────────────────────────────────────

def print_report():
    """Print calibration + accuracy report on all resolved predictions."""
    conn = get_db()

    total = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
    resolved = conn.execute("SELECT COUNT(*) FROM predictions WHERE resolved_at IS NOT NULL").fetchone()[0]
    pending = total - resolved

    print('\n' + '=' * 65)
    print('LUMENY PAPER TRADING REPORT')
    print('=' * 65)
    print(f'Total predictions logged : {total:,}')
    print(f'Resolved                 : {resolved:,}')
    print(f'Pending                  : {pending:,}')

    if resolved == 0:
        print('\nNo resolved predictions yet. Come back later.')
        conn.close()
        return

    df = pd.read_sql("""
        SELECT * FROM predictions WHERE resolved_at IS NOT NULL AND correct IS NOT NULL
    """, conn)
    df_all_resolved = pd.read_sql(
        "SELECT * FROM predictions WHERE resolved_at IS NOT NULL", conn
    )

    if len(df) == 0:
        print('\nNo scored predictions yet. Come back later.')
        conn.close()
        return

    print(f'\nDate range: {df["logged_at"].min()[:10]} → {df["logged_at"].max()[:10]}')

    # ── Accuracy by horizon + confidence bucket ───────────────────
    print('\n' + '-' * 65)
    print('ACCURACY BY HORIZON & CONFIDENCE BUCKET')
    print('-' * 65)

    buckets = [(0.50, 0.55), (0.55, 0.60), (0.60, 0.65),
               (0.65, 0.70), (0.70, 0.80), (0.80, 1.00)]

    for horizon in HORIZONS:
        h_df = df[df['horizon'] == horizon]
        if len(h_df) == 0:
            continue

        overall_acc = h_df['correct'].mean()
        print(f'\nHorizon: {horizon}  (n={len(h_df):,}, overall acc={overall_acc:.1%})')
        print(f'  {"Bucket":<12} {"Pred prob":<12} {"Actual acc":<13} {"Count":<8} {"Status"}')
        print(f'  {"-"*52}')

        for low, high in buckets:
            mask = (h_df['probability'] >= low) & (h_df['probability'] < high)
            sub  = h_df[mask]
            if len(sub) < 10:
                continue
            pred_prob  = sub['probability'].mean()
            actual_acc = sub['correct'].mean()
            gap        = actual_acc - pred_prob
            status     = 'OK' if abs(gap) < 0.05 else 'DRIFT'
            print(f'  {low:.0%}–{high:.0%}      {pred_prob:.1%}        {actual_acc:.1%}         {len(sub):<8} {status}  (gap: {gap:+.3f})')

    # ── MCE by horizon ────────────────────────────────────────────
    print('\n' + '-' * 65)
    print('CALIBRATION (MCE) BY HORIZON')
    print('-' * 65)
    print(f'  {"Horizon":<10} {"MCE":<10} {"n"}')
    print(f'  {"-"*30}')

    for horizon in HORIZONS:
        h_df_r = df_all_resolved[df_all_resolved['horizon'] == horizon]
        if len(h_df_r) < 20:
            print(f'  {horizon:<10} {"insufficient data":<10}')
            continue

        # Match notebook MCE: bin on p_down (0–1 scale), y_binary = actual_return < 0
        h_df_r = h_df_r[h_df_r['actual_return'].notna()]
        bins = np.linspace(0, 1, 16)
        bp, ba = [], []
        for i in range(len(bins) - 1):
            mask = (h_df_r['p_down'] >= bins[i]) & (h_df_r['p_down'] < bins[i+1])
            if mask.sum() > 5:
                bp.append(h_df_r.loc[mask, 'p_down'].mean())
                ba.append((h_df_r.loc[mask, 'actual_return'] < 0).mean())
        if bp:
            mce = np.mean(np.abs(np.array(bp) - np.array(ba)))
            print(f'  {horizon:<10} {mce:<10.3f} {len(h_df_r):,}')

    # ── Economic significance ─────────────────────────────────────
    print('\n' + '-' * 65)
    print('ECONOMIC SIGNIFICANCE (high conviction only, >=65%)')
    print('-' * 65)
    print(f'  {"Horizon":<10} {"Avg |move|":<14} {"Win rate":<12} {"EV (est)"}')
    print(f'  {"-"*50}')

    AVG_SPREAD = 0.00028  # ~2.8 pips average

    for horizon in HORIZONS:
        h_df = df[(df['horizon'] == horizon) & (df['probability'] >= 0.65)]
        if len(h_df) < 10:
            continue
        avg_move = h_df['actual_return'].abs().mean()
        win_rate = h_df['correct'].mean()
        ev = (win_rate * avg_move) - ((1 - win_rate) * avg_move) - (AVG_SPREAD * 100)
        print(f'  {horizon:<10} {avg_move:.4f}%       {win_rate:.1%}        EV: {ev:+.4f}%')

    # ── Conviction flag check ─────────────────────────────────────
    print('\n' + '-' * 65)
    print('LOW CONVICTION FLAG ACCURACY')
    print('-' * 65)
    normal = df[df['low_conviction'] == 0]
    low    = df[df['low_conviction'] == 1]
    if len(normal) > 0:
        print(f'  Normal conviction (n={len(normal):,}): acc={normal["correct"].mean():.1%}')
    if len(low) > 0:
        print(f'  Low conviction    (n={len(low):,}):  acc={low["correct"].mean():.1%}')
        print(f'  → Low conviction should have lower accuracy than normal.')

    # ── Weekly summary ────────────────────────────────────────────
    print('\n' + '-' * 65)
    print('WEEKLY SUMMARY')
    print('-' * 65)
    df['week'] = pd.to_datetime(df['logged_at']).dt.isocalendar().week
    df['year'] = pd.to_datetime(df['logged_at']).dt.isocalendar().year

    weekly = df.groupby(['year', 'week']).agg(
        n=('correct', 'count'),
        acc=('correct', 'mean'),
        mce_proxy=('probability', lambda x: abs(x.mean() - df.loc[x.index, 'correct'].mean()))
    ).reset_index()

    print(f'  {"Year":<6} {"Week":<6} {"n":<8} {"Accuracy"}')
    print(f'  {"-"*30}')
    for _, row in weekly.iterrows():
        print(f'  {int(row["year"]):<6} {int(row["week"]):<6} {int(row["n"]):<8} {row["acc"]:.1%}')

    print('\n' + '=' * 65)
    conn.close()


# ── Continuous loop ───────────────────────────────────────────────────────────

async def run_loop():
    """Run log + resolve every hour indefinitely."""
    logger.info('Starting paper trading loop (every 60 minutes)...')
    while True:
        try:
            # Fetch data once, reuse for both logging and resolving
            buf = await _init_buffer()
            await resolve_outcomes()
            await log_predictions(buf)
        except Exception as e:
            logger.error(f'Loop error: {e}', exc_info=True)

        # Sleep until XX:05 of the next hour to ensure cycle always starts
        # within the first few minutes of a new candle period.
        now = datetime.now(timezone.utc)
        next_run = (now + timedelta(hours=1)).replace(minute=5, second=0, microsecond=0)
        sleep_secs = (next_run - now).total_seconds()
        logger.info(f'Sleeping until {next_run.strftime("%H:%M")} UTC ({sleep_secs/60:.1f} min)...')
        await asyncio.sleep(sleep_secs)


# ── Monitoring API ────────────────────────────────────────────────────────────

from fastapi import FastAPI, Query as Q, Depends, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import jwt

monitor_app = FastAPI(title='LumenY Paper Trading Monitor', version='1.0.0')
monitor_app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_methods=['*'], allow_headers=['*'])

# ── Auth ──────────────────────────────────────────────────────────────────────

_JWT_SECRET   = os.environ.get('JWT_SECRET', 'change-me-in-production')
_MONITOR_PASS = os.environ.get('MONITOR_PASSWORD', '')
_TOKEN_TTL_H  = 6


def _require_auth(request: Request):
    token = request.cookies.get('lumeny_session')
    if not token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail='Missing token')
    try:
        jwt.decode(token, _JWT_SECRET, algorithms=['HS256'])
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail='Token expired')
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail='Invalid token')


@monitor_app.post('/auth/login')
async def login(body: dict):
    password = body.get('password', '')
    if not _MONITOR_PASS or password != _MONITOR_PASS:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail='Invalid password')
    import datetime as _dt
    token = jwt.encode(
        {'exp': _dt.datetime.now(_dt.timezone.utc) + _dt.timedelta(hours=_TOKEN_TTL_H)},
        _JWT_SECRET,
        algorithm='HS256',
    )
    response = JSONResponse({'ok': True, 'expires_in_hours': _TOKEN_TTL_H})
    response.set_cookie(
        key='lumeny_session',
        value=token,
        httponly=True,
        max_age=_TOKEN_TTL_H * 3600,
        samesite='lax',
    )
    return response


def _get_report_data() -> dict:
    """Generate the full report as a JSON-serializable dict."""
    conn = get_db()
    total = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
    resolved_count = conn.execute("SELECT COUNT(*) FROM predictions WHERE resolved_at IS NOT NULL").fetchone()[0]
    pending_count = total - resolved_count

    report = {
        'total_predictions': total,
        'resolved': resolved_count,
        'pending': pending_count,
        'horizons': {},
        'calibration': {},
        'economic': {},
        'conviction': {},
        'weekly': [],
    }

    if resolved_count == 0:
        conn.close()
        return report

    df = pd.read_sql("SELECT * FROM predictions WHERE resolved_at IS NOT NULL AND correct IS NOT NULL", conn)
    df_all_resolved = pd.read_sql("SELECT * FROM predictions WHERE resolved_at IS NOT NULL", conn)
    if len(df) == 0:
        conn.close()
        return report
    report['date_range'] = {
        'from': df['logged_at'].min()[:10],
        'to': df['logged_at'].max()[:10],
    }

    buckets = [(0.50, 0.55), (0.55, 0.60), (0.60, 0.65),
               (0.65, 0.70), (0.70, 0.80), (0.80, 1.00)]

    for horizon in HORIZONS:
        h_df = df[df['horizon'] == horizon]
        if len(h_df) == 0:
            continue

        h_report = {
            'n': int(len(h_df)),
            'overall_accuracy': round(float(h_df['correct'].mean()), 4),
            'buckets': [],
        }

        for low, high in buckets:
            mask = (h_df['probability'] >= low) & (h_df['probability'] < high)
            sub = h_df[mask]
            if len(sub) < 3:
                continue
            pred_prob = float(sub['probability'].mean())
            actual_acc = float(sub['correct'].mean())
            gap = actual_acc - pred_prob
            h_report['buckets'].append({
                'range': f'{low:.0%}-{high:.0%}',
                'predicted_prob': round(pred_prob, 4),
                'actual_accuracy': round(actual_acc, 4),
                'count': int(len(sub)),
                'gap': round(gap, 4),
                'status': 'OK' if abs(gap) < 0.05 else 'DRIFT',
            })

        report['horizons'][horizon] = h_report

    # MCE — match notebook: bin on p_down (0–1), y_binary = actual_return < 0, all resolved rows
    for horizon in HORIZONS:
        h_df_r = df_all_resolved[df_all_resolved['horizon'] == horizon]
        if len(h_df_r) < 20:
            report['calibration'][horizon] = {'mce': None, 'n': int(len(h_df_r)), 'status': 'insufficient_data'}
            continue
        h_df_r = h_df_r[h_df_r['actual_return'].notna()]
        bins = np.linspace(0, 1, 16)
        bp, ba = [], []
        for i in range(len(bins) - 1):
            mask = (h_df_r['p_down'] >= bins[i]) & (h_df_r['p_down'] < bins[i+1])
            if mask.sum() > 5:
                bp.append(float(h_df_r.loc[mask, 'p_down'].mean()))
                ba.append(float((h_df_r.loc[mask, 'actual_return'] < 0).mean()))
        if bp:
            mce = float(np.mean(np.abs(np.array(bp) - np.array(ba))))
            report['calibration'][horizon] = {'mce': round(mce, 4), 'n': int(len(h_df_r))}

    # Economic significance
    AVG_SPREAD = 0.00028
    for horizon in HORIZONS:
        h_df = df[(df['horizon'] == horizon) & (df['probability'] >= 0.65)]
        if len(h_df) < 5:
            continue
        avg_move = float(h_df['actual_return'].abs().mean())
        win_rate = float(h_df['correct'].mean())
        ev = (win_rate * avg_move) - ((1 - win_rate) * avg_move) - (AVG_SPREAD * 100)
        report['economic'][horizon] = {
            'avg_move_pct': round(avg_move, 4),
            'win_rate': round(win_rate, 4),
            'ev_pct': round(ev, 4),
            'n': int(len(h_df)),
        }

    # Accuracy by probability threshold (p>=70% matches backtest key metric)
    for horizon in HORIZONS:
        h_df = df[df['horizon'] == horizon]
        h_df_70 = h_df[h_df['probability'] >= 0.70]
        if len(h_df_70) > 0:
            report['conviction'][horizon] = {'n': int(len(h_df_70)), 'accuracy': round(float(h_df_70['correct'].mean()), 4)}

    # Weekly
    df['logged_dt'] = pd.to_datetime(df['logged_at'])
    df['week'] = df['logged_dt'].dt.isocalendar().week.astype(int)
    df['year'] = df['logged_dt'].dt.isocalendar().year.astype(int)
    weekly = df.groupby(['year', 'week']).agg(n=('correct', 'count'), acc=('correct', 'mean')).reset_index()
    for _, row in weekly.iterrows():
        report['weekly'].append({'year': int(row['year']), 'week': int(row['week']), 'n': int(row['n']), 'accuracy': round(float(row['acc']), 4)})

    conn.close()
    return report


@monitor_app.get('/api/monitor/report')
async def api_report(_: None = Depends(_require_auth)):
    """Full calibration & accuracy report."""
    return _get_report_data()


@monitor_app.get('/api/monitor/predictions')
async def api_predictions(
    pair: str = Q(default=None),
    horizon: str = Q(default=None),
    resolved: bool = Q(default=None),
    limit: int = Q(default=100),
    _: None = Depends(_require_auth),
):
    """Browse raw predictions."""
    conn = get_db()
    query = "SELECT * FROM predictions WHERE 1=1"
    params = []
    if pair:
        query += " AND pair = ?"
        params.append(pair.upper())
    if horizon:
        query += " AND horizon = ?"
        params.append(horizon)
    if resolved is not None:
        if resolved:
            query += " AND resolved_at IS NOT NULL"
        else:
            query += " AND resolved_at IS NULL"
    query += " ORDER BY logged_at DESC LIMIT ?"
    params.append(limit)

    rows = conn.execute(query, params).fetchall()
    conn.close()
    return {'predictions': [dict(r) for r in rows], 'count': len(rows)}


@monitor_app.get('/api/monitor/summary')
async def api_summary(_: None = Depends(_require_auth)):
    """Quick summary: total, resolved, pending, accuracy per horizon."""
    conn = get_db()
    total = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
    resolved = conn.execute("SELECT COUNT(*) FROM predictions WHERE resolved_at IS NOT NULL").fetchone()[0]

    summary = {
        'total': total,
        'resolved': resolved,
        'pending': total - resolved,
        'horizons': {},
    }

    if resolved > 0:
        for horizon in HORIZONS:
            row = conn.execute(
                "SELECT COUNT(*) as n, AVG(correct) as acc FROM predictions WHERE horizon = ? AND resolved_at IS NOT NULL AND correct IS NOT NULL",
                (horizon,)
            ).fetchone()
            if row['n'] > 0:
                summary['horizons'][horizon] = {'n': row['n'], 'accuracy': round(float(row['acc']), 4)}

    conn.close()
    return summary


@monitor_app.get('/api/monitor/debug')
async def api_debug():
    """Last 20 resolved 1H predictions — no auth required for debugging."""
    conn = get_db()
    rows = conn.execute("""
        SELECT id, pair, horizon, direction, probability, q50,
               q10, q25, q75, q90, p_down, p_up,
               entry_price, exit_price, actual_return, correct,
               logged_at, matures_at, resolved_at
        FROM predictions
        WHERE horizon = '1H' AND resolved_at IS NOT NULL
        ORDER BY id DESC LIMIT 20
    """).fetchall()
    conn.close()
    return {'predictions': [dict(r) for r in rows], 'count': len(rows)}


@monitor_app.get('/api/monitor/live-snapshot')
async def api_live_snapshot():
    """
    Fetch last 3 days of 1H OHLCV from Polygon for each pair and return
    candle times + prices — lightweight check, no full inference.
    No auth required, for debugging.
    """
    import httpx
    from data_service import PAIRS

    api_key = os.environ.get('POLYGON_API_KEY', os.environ.get('POLYGON_S3_SECRET_KEY', ''))
    now = datetime.now(timezone.utc)
    from_date = (now - timedelta(days=3)).strftime('%Y-%m-%d')
    to_date = now.strftime('%Y-%m-%d')

    snapshot = {'generated_at': now.isoformat(), 'pairs': {}}

    async with httpx.AsyncClient(timeout=20) as client:
        for pair in PAIRS:
            ticker = f'C:{pair}'
            url = f'https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/hour/{from_date}/{to_date}'
            try:
                resp = await client.get(url, params={'apiKey': api_key, 'limit': 200, 'sort': 'asc'})
                resp.raise_for_status()
                data = resp.json()
                results = data.get('results', [])

                if not results:
                    snapshot['pairs'][pair] = {'error': 'no data', 'status': data.get('status')}
                    continue

                # Last 3 bars (before the current open bar — bar['t'] < current_hour_start)
                current_hour_ms = int(now.replace(minute=0, second=0, microsecond=0).timestamp() * 1000)
                closed_bars = [b for b in results if b['t'] < current_hour_ms]
                last_closed = closed_bars[-1] if closed_bars else None
                last_open = results[-1]  # most recent bar (may be open)

                snapshot['pairs'][pair] = {
                    'total_bars_returned': len(results),
                    'closed_bars_count': len(closed_bars),
                    'last_closed_bar': {
                        'open_time_utc': datetime.fromtimestamp(last_closed['t'] / 1000, tz=timezone.utc).isoformat() if last_closed else None,
                        'open': last_closed['o'] if last_closed else None,
                        'close': last_closed['c'] if last_closed else None,
                        'high': last_closed['h'] if last_closed else None,
                        'low': last_closed['l'] if last_closed else None,
                    },
                    'most_recent_bar': {
                        'open_time_utc': datetime.fromtimestamp(last_open['t'] / 1000, tz=timezone.utc).isoformat(),
                        'open': last_open['o'],
                        'close': last_open['c'],
                        'is_open_bar': last_open['t'] >= current_hour_ms,
                    },
                    'expected_last_candle_time': datetime.fromtimestamp(last_closed['t'] / 1000, tz=timezone.utc).isoformat() if last_closed else None,
                    'expected_entry_price': last_closed['c'] if last_closed else None,
                    'expected_matures_at_1H': datetime.fromtimestamp((last_closed['t'] + 3600_000) / 1000, tz=timezone.utc).isoformat() if last_closed else None,
                }
            except Exception as e:
                snapshot['pairs'][pair] = {'error': str(e)}

    return snapshot


@monitor_app.get('/api/monitor/health')
async def api_health(_: None = Depends(_require_auth)):
    """Check if the paper trading system is running."""
    conn = get_db()
    last_log = conn.execute("SELECT * FROM hourly_log ORDER BY id DESC LIMIT 1").fetchone()
    conn.close()
    return {
        'status': 'ok',
        'db_path': str(DB_PATH),
        'last_log': dict(last_log) if last_log else None,
    }


@monitor_app.get('/api/monitor/dashboard-data')
async def api_dashboard_data(_: None = Depends(_require_auth)):
    """All data needed for the monitoring dashboard in one call."""
    conn = get_db()
    total = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
    resolved_count = conn.execute("SELECT COUNT(*) FROM predictions WHERE resolved_at IS NOT NULL").fetchone()[0]

    data = {
        'total': total,
        'resolved': resolved_count,
        'pending': total - resolved_count,
        'horizons': {},
        'pairs': {},
        'calibration_curve': {},
        'confidence_distribution': {},
        'daily_accuracy': [],
        'weekly': [],
        'conviction': {},
        'economic': {},
        'last_log': None,
    }

    last_log = conn.execute("SELECT * FROM hourly_log ORDER BY id DESC LIMIT 1").fetchone()
    if last_log:
        data['last_log'] = dict(last_log)

    if resolved_count == 0:
        conn.close()
        return data

    df = pd.read_sql("SELECT * FROM predictions WHERE resolved_at IS NOT NULL AND correct IS NOT NULL", conn)
    df_all_resolved = pd.read_sql("SELECT * FROM predictions WHERE resolved_at IS NOT NULL", conn)
    all_df = pd.read_sql("SELECT logged_at, pair, horizon, probability, direction, low_conviction FROM predictions", conn)
    conn.close()

    data['date_range'] = {
        'from': df['logged_at'].min()[:10],
        'to': df['logged_at'].max()[:10],
    }

    # ── Per-horizon stats ──
    for horizon in HORIZONS:
        h_df = df[df['horizon'] == horizon]
        if len(h_df) == 0:
            continue
        data['horizons'][horizon] = {
            'n': int(len(h_df)),
            'accuracy': round(float(h_df['correct'].mean()), 4),
            'avg_probability': round(float(h_df['probability'].mean()), 4),
            'avg_actual_return': round(float(h_df['actual_return'].mean()), 4),
        }

    # ── Per-pair stats ──
    for pair in df['pair'].unique():
        p_df = df[df['pair'] == pair]
        pair_data = {'n': int(len(p_df)), 'accuracy': round(float(p_df['correct'].mean()), 4), 'horizons': {}}
        for horizon in HORIZONS:
            ph_df = p_df[p_df['horizon'] == horizon]
            if len(ph_df) >= 3:
                pair_data['horizons'][horizon] = {
                    'n': int(len(ph_df)),
                    'accuracy': round(float(ph_df['correct'].mean()), 4),
                }
        data['pairs'][pair] = pair_data

    # ── Calibration curve (reliability diagram data) ──
    # Matches notebook: bin on p_down (0–1), y_binary = actual_return < 0
    for horizon in HORIZONS:
        h_df_r = df_all_resolved[df_all_resolved['horizon'] == horizon]
        if len(h_df_r) < 10:
            continue
        bins = np.linspace(0, 1, 16)
        points = []
        for i in range(len(bins) - 1):
            mask = (h_df_r['p_down'] >= bins[i]) & (h_df_r['p_down'] < bins[i + 1])
            if mask.sum() >= 3:
                points.append({
                    'bin_center': round(float((bins[i] + bins[i + 1]) / 2), 3),
                    'predicted': round(float(h_df_r.loc[mask, 'p_down'].mean()), 4),
                    'actual': round(float((h_df_r.loc[mask, 'actual_return'] < 0).mean()), 4),
                    'count': int(mask.sum()),
                })
        if points:
            mce = round(float(np.mean([abs(p['predicted'] - p['actual']) for p in points])), 4)
            data['calibration_curve'][horizon] = {'points': points, 'mce': mce}

    # ── Confidence distribution (all predictions, not just resolved) ──
    for horizon in HORIZONS:
        h_all = all_df[all_df['horizon'] == horizon]
        if len(h_all) == 0:
            continue
        bins = np.linspace(0.5, 1.0, 11)
        hist = []
        for i in range(len(bins) - 1):
            mask = (h_all['probability'] >= bins[i]) & (h_all['probability'] < bins[i + 1])
            hist.append({
                'range': f'{bins[i]:.0%}-{bins[i+1]:.0%}',
                'count': int(mask.sum()),
            })
        data['confidence_distribution'][horizon] = hist

    # ── Daily accuracy trend ──
    df['logged_dt'] = pd.to_datetime(df['logged_at'])
    df['date'] = df['logged_dt'].dt.date.astype(str)
    daily = df.groupby('date').agg(n=('correct', 'count'), acc=('correct', 'mean'), avg_prob=('probability', 'mean')).reset_index()
    for _, row in daily.iterrows():
        data['daily_accuracy'].append({
            'date': row['date'],
            'n': int(row['n']),
            'accuracy': round(float(row['acc']), 4),
            'avg_probability': round(float(row['avg_prob']), 4),
        })

    # ── Weekly ──
    df['week'] = df['logged_dt'].dt.isocalendar().week.astype(int)
    df['year'] = df['logged_dt'].dt.isocalendar().year.astype(int)
    weekly = df.groupby(['year', 'week']).agg(n=('correct', 'count'), acc=('correct', 'mean')).reset_index()
    for _, row in weekly.iterrows():
        data['weekly'].append({'year': int(row['year']), 'week': int(row['week']), 'n': int(row['n']), 'accuracy': round(float(row['acc']), 4)})

    # ── Accuracy at p>=70% threshold (matches backtest key metric) ──
    for horizon in HORIZONS:
        h_df = df[df['horizon'] == horizon]
        h_df_70 = h_df[h_df['probability'] >= 0.70]
        if len(h_df_70) > 0:
            data['conviction'][horizon] = {'n': int(len(h_df_70)), 'accuracy': round(float(h_df_70['correct'].mean()), 4)}

    # ── Economic significance ──
    AVG_SPREAD = 0.00028
    for horizon in HORIZONS:
        h_df = df[(df['horizon'] == horizon) & (df['probability'] >= 0.65)]
        if len(h_df) < 5:
            continue
        avg_move = float(h_df['actual_return'].abs().mean())
        win_rate = float(h_df['correct'].mean())
        ev = (win_rate * avg_move) - ((1 - win_rate) * avg_move) - (AVG_SPREAD * 100)
        data['economic'][horizon] = {
            'avg_move_pct': round(avg_move, 4),
            'win_rate': round(win_rate, 4),
            'ev_pct': round(ev, 4),
            'n': int(len(h_df)),
        }

    return data


from fastapi.responses import HTMLResponse

@monitor_app.get('/', response_class=HTMLResponse)
@monitor_app.get('/dashboard', response_class=HTMLResponse)
async def dashboard():
    """Self-contained monitoring dashboard."""
    return DASHBOARD_HTML


DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>LumenY Paper Trading Monitor</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #0a0e17; color: #e0e0e0; }
  .header { background: linear-gradient(135deg, #1a1f2e 0%, #0d1117 100%); padding: 24px 32px; border-bottom: 1px solid #21262d; }
  .header h1 { font-size: 22px; font-weight: 600; color: #f0f0f0; }
  .header .subtitle { font-size: 13px; color: #7d8590; margin-top: 4px; }
  .status-bar { display: flex; gap: 24px; margin-top: 12px; flex-wrap: wrap; }
  .status-item { font-size: 13px; }
  .status-item .label { color: #7d8590; }
  .status-item .value { color: #58a6ff; font-weight: 600; }
  .status-item .value.good { color: #3fb950; }
  .status-item .value.warn { color: #d29922; }
  .status-item .value.bad { color: #f85149; }
  .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(420px, 1fr)); gap: 16px; padding: 20px 32px; }
  .card { background: #161b22; border: 1px solid #21262d; border-radius: 8px; padding: 20px; }
  .card h2 { font-size: 15px; font-weight: 600; color: #c9d1d9; margin-bottom: 14px; border-bottom: 1px solid #21262d; padding-bottom: 8px; }
  .card.full-width { grid-column: 1 / -1; }
  table { width: 100%; border-collapse: collapse; font-size: 13px; }
  th { text-align: left; color: #7d8590; font-weight: 500; padding: 6px 8px; border-bottom: 1px solid #21262d; }
  td { padding: 6px 8px; border-bottom: 1px solid #21262d30; }
  .good { color: #3fb950; }
  .warn { color: #d29922; }
  .bad { color: #f85149; }
  .neutral { color: #7d8590; }
  .chart-container { position: relative; height: 260px; }
  .kpi-row { display: flex; gap: 16px; flex-wrap: wrap; padding: 0 32px 4px; }
  .kpi { background: #161b22; border: 1px solid #21262d; border-radius: 8px; padding: 16px 20px; flex: 1; min-width: 140px; }
  .kpi .kpi-value { font-size: 28px; font-weight: 700; }
  .kpi .kpi-label { font-size: 12px; color: #7d8590; margin-top: 2px; }
  .refresh-btn { background: #21262d; color: #c9d1d9; border: 1px solid #30363d; border-radius: 6px; padding: 6px 14px; cursor: pointer; font-size: 13px; }
  .refresh-btn:hover { background: #30363d; }
  .loading { text-align: center; padding: 60px; color: #7d8590; font-size: 16px; }
  .verdict { font-size: 14px; padding: 12px; border-radius: 6px; margin-bottom: 12px; }
  .verdict.pass { background: #0d1f0d; border: 1px solid #238636; color: #3fb950; }
  .verdict.fail { background: #1f0d0d; border: 1px solid #da3633; color: #f85149; }
  .verdict.wait { background: #1f1a0d; border: 1px solid #9e6a03; color: #d29922; }
</style>
</head>
<body>

<div class="header">
  <div style="display:flex; justify-content:space-between; align-items:center;">
    <div>
      <h1>LumenY Paper Trading Monitor</h1>
      <div class="subtitle">Live model validation &mdash; does the model match backtest?</div>
    </div>
    <button class="refresh-btn" onclick="loadData()">Refresh</button>
  </div>
  <div class="status-bar" id="statusBar"></div>
</div>

<div class="kpi-row" id="kpiRow" style="margin-top:20px;"></div>

<div class="grid" id="grid">
  <div class="loading">Loading data...</div>
</div>

<script>
const HORIZONS = ['1H', '4H', '1D'];
const COLORS = { '1H': '#58a6ff', '4H': '#bc8cff', '1D': '#f0883e' };
let charts = {};

function showLogin(error) {
  document.body.innerHTML = `
    <div style="display:flex;align-items:center;justify-content:center;min-height:100vh;background:#0a0e17;">
      <div style="background:#0d1420;border:1px solid #1e2d45;border-radius:12px;padding:40px;width:340px;text-align:center;">
        <div style="font-size:22px;font-weight:700;color:#58a6ff;margin-bottom:8px;">LumenY</div>
        <div style="color:#7d8fa3;margin-bottom:28px;font-size:13px;">Paper Trading Monitor</div>
        ${error ? '<div style="color:#f85149;margin-bottom:16px;font-size:13px;">' + error + '</div>' : ''}
        <input id="pw" type="password" placeholder="Password" autofocus
          style="width:100%;padding:10px 14px;border-radius:8px;border:1px solid #1e2d45;background:#111827;color:#e0e0e0;font-size:14px;margin-bottom:14px;outline:none;" />
        <button onclick="doLogin()"
          style="width:100%;padding:10px;border-radius:8px;border:none;background:#58a6ff;color:#0a0e17;font-weight:700;font-size:14px;cursor:pointer;">
          Login
        </button>
      </div>
    </div>`;
  document.getElementById('pw').addEventListener('keydown', e => { if (e.key === 'Enter') doLogin(); });
}

async function doLogin() {
  const password = document.getElementById('pw').value;
  try {
    const res = await fetch('/auth/login', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ password }),
    });
    if (!res.ok) { showLogin('Invalid password.'); return; }
    location.reload();
  } catch(e) {
    showLogin('Login failed: ' + e.message);
  }
}

function fmt(v, d=1) { return v != null ? (v * 100).toFixed(d) + '%' : 'N/A'; }
function accClass(v) { return v >= 0.55 ? 'good' : v >= 0.50 ? 'warn' : 'bad'; }
function mceClass(v) { return v <= 0.03 ? 'good' : v <= 0.06 ? 'warn' : 'bad'; }

async function loadData() {
  try {
    const res = await fetch('/api/monitor/dashboard-data');
    if (res.status === 401) { showLogin(); return; }
    const d = await res.json();
    render(d);
  } catch(e) {
    document.getElementById('grid').innerHTML = '<div class="loading">Error loading data: ' + e.message + '</div>';
  }
}

function render(d) {
  // Status bar
  const sb = document.getElementById('statusBar');
  const lastLog = d.last_log ? d.last_log.logged_at : 'never';
  sb.innerHTML = `
    <div class="status-item"><span class="label">Last run:</span> <span class="value">${lastLog}</span></div>
    <div class="status-item"><span class="label">Total:</span> <span class="value">${d.total}</span></div>
    <div class="status-item"><span class="label">Resolved:</span> <span class="value good">${d.resolved}</span></div>
    <div class="status-item"><span class="label">Pending:</span> <span class="value warn">${d.pending}</span></div>
    <div class="status-item"><span class="label">Date range:</span> <span class="value">${d.date_range ? d.date_range.from + ' to ' + d.date_range.to : 'N/A'}</span></div>
  `;

  // KPIs
  const kpi = document.getElementById('kpiRow');
  let overallAcc = d.resolved > 0 ? Object.values(d.horizons).reduce((s, h) => s + h.accuracy * h.n, 0) / Object.values(d.horizons).reduce((s, h) => s + h.n, 0) : null;
  let avgMCE = Object.keys(d.calibration_curve).length > 0 ? Object.values(d.calibration_curve).reduce((s, c) => s + c.mce, 0) / Object.keys(d.calibration_curve).length : null;
  const conv70 = Object.values(d.conviction);
  const acc70 = conv70.length > 0 ? conv70.reduce((s, c) => s + c.accuracy * c.n, 0) / conv70.reduce((s, c) => s + c.n, 0) : null;

  // Verdict
  let verdict = 'wait', verdictText = 'Collecting data...';
  if (d.resolved >= 50) {
    if (overallAcc >= 0.54 && (avgMCE == null || avgMCE <= 0.05)) {
      verdict = 'pass'; verdictText = 'Model is performing as expected';
    } else if (overallAcc < 0.50 || (avgMCE != null && avgMCE > 0.08)) {
      verdict = 'fail'; verdictText = 'Model performance below expectations — investigate';
    } else {
      verdict = 'wait'; verdictText = 'Borderline — need more data';
    }
  }

  kpi.innerHTML = `
    <div class="kpi"><div class="kpi-value ${accClass(overallAcc)}">${overallAcc != null ? fmt(overallAcc) : '--'}</div><div class="kpi-label">Overall Accuracy</div></div>
    <div class="kpi"><div class="kpi-value ${avgMCE != null ? mceClass(avgMCE) : 'neutral'}">${avgMCE != null ? (avgMCE * 100).toFixed(1) + '%' : '--'}</div><div class="kpi-label">Avg MCE (calibration error)</div></div>
    <div class="kpi"><div class="kpi-value" style="color:#58a6ff">${d.resolved}</div><div class="kpi-label">Resolved Predictions</div></div>
    <div class="kpi"><div class="kpi-value ${acc70 != null ? accClass(acc70) : 'neutral'}">${acc70 != null ? fmt(acc70) : '--'}</div><div class="kpi-label">Accuracy (p>=70%)</div></div>
  `;

  const grid = document.getElementById('grid');
  grid.innerHTML = '';

  // Verdict card
  grid.innerHTML += `<div class="card full-width"><div class="verdict ${verdict}">${verdictText}${d.resolved < 50 ? ' (' + d.resolved + '/50 resolved)' : ''}</div></div>`;

  // Horizon accuracy table
  let hTable = '<table><tr><th>Horizon</th><th>N</th><th>Accuracy</th><th>Avg Prob</th><th>MCE</th><th>Status</th></tr>';
  for (const h of HORIZONS) {
    const hd = d.horizons[h];
    const cal = d.calibration_curve[h];
    if (!hd) { hTable += `<tr><td>${h}</td><td colspan="5" class="neutral">No data</td></tr>`; continue; }
    const mce = cal ? cal.mce : null;
    const status = hd.n < 20 ? '<span class="neutral">collecting</span>' : (hd.accuracy >= 0.54 ? '<span class="good">OK</span>' : '<span class="bad">CHECK</span>');
    hTable += `<tr><td>${h}</td><td>${hd.n}</td><td class="${accClass(hd.accuracy)}">${fmt(hd.accuracy)}</td><td>${fmt(hd.avg_probability)}</td><td class="${mce != null ? mceClass(mce) : 'neutral'}">${mce != null ? fmt(mce) : '--'}</td><td>${status}</td></tr>`;
  }
  hTable += '</table>';
  grid.innerHTML += `<div class="card"><h2>Accuracy by Horizon</h2>${hTable}</div>`;

  // Pair breakdown table
  let pTable = '<table><tr><th>Pair</th><th>N</th><th>Accuracy</th>';
  for (const h of HORIZONS) pTable += `<th>${h}</th>`;
  pTable += '</tr>';
  for (const [pair, pd] of Object.entries(d.pairs).sort((a,b) => b[1].accuracy - a[1].accuracy)) {
    pTable += `<tr><td>${pair}</td><td>${pd.n}</td><td class="${accClass(pd.accuracy)}">${fmt(pd.accuracy)}</td>`;
    for (const h of HORIZONS) {
      const ph = pd.horizons[h];
      pTable += ph ? `<td class="${accClass(ph.accuracy)}">${fmt(ph.accuracy)}</td>` : '<td class="neutral">--</td>';
    }
    pTable += '</tr>';
  }
  pTable += '</table>';
  grid.innerHTML += `<div class="card"><h2>Accuracy by Pair</h2>${pTable}</div>`;

  // Calibration curve chart
  grid.innerHTML += '<div class="card"><h2>Reliability Diagram</h2><div class="chart-container"><canvas id="calChart"></canvas></div></div>';

  // Confidence distribution chart
  grid.innerHTML += '<div class="card"><h2>Confidence Distribution</h2><div class="chart-container"><canvas id="confChart"></canvas></div></div>';

  // Daily accuracy trend
  grid.innerHTML += '<div class="card full-width"><h2>Daily Accuracy Trend</h2><div class="chart-container"><canvas id="dailyChart"></canvas></div></div>';

  // Economic value table
  let eTable = '<table><tr><th>Horizon</th><th>Win Rate</th><th>Avg Move</th><th>EV%</th><th>N (p>=65%)</th></tr>';
  for (const h of HORIZONS) {
    const e = d.economic[h];
    if (!e) continue;
    const evClass = e.ev_pct > 0 ? 'good' : 'bad';
    eTable += `<tr><td>${h}</td><td class="${accClass(e.win_rate)}">${fmt(e.win_rate)}</td><td>${e.avg_move_pct.toFixed(4)}%</td><td class="${evClass}">${e.ev_pct.toFixed(4)}%</td><td>${e.n}</td></tr>`;
  }
  eTable += '</table>';

  // Accuracy by probability threshold
  let tTable = '<table><tr><th>Horizon</th><th>Threshold</th><th>Accuracy</th><th>N</th></tr>';
  for (const h of HORIZONS) {
    const hd = d.horizons[h];
    const c70 = d.conviction[h];
    if (!hd) continue;
    tTable += `<tr><td>${h}</td><td>All</td><td class="${accClass(hd.accuracy)}">${fmt(hd.accuracy)}</td><td>${hd.n}</td></tr>`;
    if (c70) tTable += `<tr><td></td><td>p&gt;=70%</td><td class="${accClass(c70.accuracy)}">${fmt(c70.accuracy)}</td><td>${c70.n}</td></tr>`;
  }
  tTable += '</table>';

  grid.innerHTML += `<div class="card"><h2>Economic Value (p >= 65%)</h2>${eTable}</div>`;
  grid.innerHTML += `<div class="card"><h2>Accuracy by Threshold</h2>${tTable}</div>`;

  // ── Prediction Explorer ──
  grid.innerHTML += `<div class="card full-width">
    <h2>Prediction Explorer</h2>
    <div style="display:flex; gap:12px; flex-wrap:wrap; margin-bottom:14px;">
      <select id="expPair" style="background:#0d1117;color:#c9d1d9;border:1px solid #30363d;border-radius:6px;padding:6px 10px;font-size:13px;">
        <option value="">All Pairs</option>
        <option>EURUSD</option><option>GBPUSD</option><option>USDJPY</option><option>USDCHF</option><option>AUDUSD</option><option>USDCAD</option><option>NZDUSD</option>
      </select>
      <select id="expHorizon" style="background:#0d1117;color:#c9d1d9;border:1px solid #30363d;border-radius:6px;padding:6px 10px;font-size:13px;">
        <option value="">All Horizons</option>
        <option>1H</option><option>4H</option><option>1D</option>
      </select>
      <select id="expResolved" style="background:#0d1117;color:#c9d1d9;border:1px solid #30363d;border-radius:6px;padding:6px 10px;font-size:13px;">
        <option value="">All</option>
        <option value="true">Resolved</option>
        <option value="false">Pending</option>
      </select>
      <button class="refresh-btn" onclick="loadPredictions()">Load</button>
    </div>
    <div id="expTable" style="max-height:400px;overflow-y:auto;"></div>
  </div>`;
  loadPredictions();

  // ── Render charts ──
  Object.values(charts).forEach(c => c.destroy());
  charts = {};

  // Calibration curve
  const calCtx = document.getElementById('calChart').getContext('2d');
  const calDatasets = [];
  for (const h of HORIZONS) {
    const cal = d.calibration_curve[h];
    if (!cal) continue;
    calDatasets.push({
      label: h,
      data: cal.points.map(p => ({ x: p.predicted * 100, y: p.actual * 100 })),
      borderColor: COLORS[h],
      backgroundColor: COLORS[h] + '40',
      pointRadius: 5,
      showLine: true,
      tension: 0.1,
    });
  }
  calDatasets.push({
    label: 'Perfect',
    data: [{x:50,y:50},{x:100,y:100}],
    borderColor: '#30363d',
    borderDash: [5,5],
    pointRadius: 0,
    showLine: true,
  });
  charts.cal = new Chart(calCtx, {
    type: 'scatter',
    data: { datasets: calDatasets },
    options: {
      responsive: true, maintainAspectRatio: false,
      scales: {
        x: { title: { display: true, text: 'Predicted Probability %', color: '#7d8590' }, min: 50, max: 100, grid: { color: '#21262d' }, ticks: { color: '#7d8590' } },
        y: { title: { display: true, text: 'Actual Accuracy %', color: '#7d8590' }, min: 30, max: 100, grid: { color: '#21262d' }, ticks: { color: '#7d8590' } },
      },
      plugins: { legend: { labels: { color: '#c9d1d9' } } },
    },
  });

  // Confidence distribution
  const confCtx = document.getElementById('confChart').getContext('2d');
  const confDatasets = [];
  for (const h of HORIZONS) {
    const cd = d.confidence_distribution[h];
    if (!cd) continue;
    confDatasets.push({
      label: h,
      data: cd.map(b => b.count),
      backgroundColor: COLORS[h] + '80',
      borderColor: COLORS[h],
      borderWidth: 1,
    });
  }
  const confLabels = d.confidence_distribution[HORIZONS.find(h => d.confidence_distribution[h])]?.map(b => b.range) || [];
  charts.conf = new Chart(confCtx, {
    type: 'bar',
    data: { labels: confLabels, datasets: confDatasets },
    options: {
      responsive: true, maintainAspectRatio: false,
      scales: {
        x: { grid: { color: '#21262d' }, ticks: { color: '#7d8590' } },
        y: { title: { display: true, text: 'Count', color: '#7d8590' }, grid: { color: '#21262d' }, ticks: { color: '#7d8590' } },
      },
      plugins: { legend: { labels: { color: '#c9d1d9' } } },
    },
  });

  // Daily accuracy trend
  const dailyCtx = document.getElementById('dailyChart').getContext('2d');
  charts.daily = new Chart(dailyCtx, {
    type: 'line',
    data: {
      labels: d.daily_accuracy.map(x => x.date),
      datasets: [
        {
          label: 'Accuracy',
          data: d.daily_accuracy.map(x => x.accuracy * 100),
          borderColor: '#58a6ff',
          backgroundColor: '#58a6ff20',
          fill: true,
          tension: 0.3,
        },
        {
          label: 'Avg Predicted Prob',
          data: d.daily_accuracy.map(x => x.avg_probability * 100),
          borderColor: '#f0883e',
          borderDash: [4,4],
          tension: 0.3,
        },
        {
          label: '50% baseline',
          data: d.daily_accuracy.map(() => 50),
          borderColor: '#f8514950',
          borderDash: [2,4],
          pointRadius: 0,
        },
      ],
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      scales: {
        x: { grid: { color: '#21262d' }, ticks: { color: '#7d8590', maxTicksLimit: 14 } },
        y: { title: { display: true, text: 'Accuracy %', color: '#7d8590' }, min: 30, max: 80, grid: { color: '#21262d' }, ticks: { color: '#7d8590' } },
      },
      plugins: { legend: { labels: { color: '#c9d1d9' } } },
    },
  });
}

async function loadPredictions() {
  const pair = document.getElementById('expPair')?.value || '';
  const horizon = document.getElementById('expHorizon')?.value || '';
  const resolved = document.getElementById('expResolved')?.value || '';
  const params = new URLSearchParams();
  if (pair) params.set('pair', pair);
  if (horizon) params.set('horizon', horizon);
  if (resolved) params.set('resolved', resolved);
  params.set('limit', '50');
  const container = document.getElementById('expTable');
  if (!container) return;
  try {
    const res = await fetch('/api/monitor/predictions?' + params.toString());
    if (res.status === 401) { showLogin(); return; }
    const data = await res.json();
    if (data.predictions.length === 0) {
      container.innerHTML = '<div class="neutral" style="padding:12px;">No predictions found.</div>';
      return;
    }
    let t = '<table><tr><th>Pair</th><th>Horizon</th><th>Dir</th><th>Prob</th><th>Logged</th><th>Maturity</th><th>Result</th><th>Return</th></tr>';
    for (const p of data.predictions) {
      const isResolved = p.resolved_at != null;
      const resultCell = !isResolved
        ? '<span class="neutral">Pending</span>'
        : p.correct === null
          ? '<span class="neutral">No signal</span>'
          : `<span class="${p.correct ? 'good' : 'bad'}">${p.correct ? 'Correct' : 'Wrong'}</span>`;
      const retCell = isResolved ? (p.actual_return != null ? p.actual_return.toFixed(4) + '%' : '--') : '--';
      const dir = p.direction === 'bullish' ? '<span class="good">UP</span>' : p.direction === 'bearish' ? '<span class="bad">DOWN</span>' : '<span class="neutral">FLAT</span>';
      t += `<tr><td>${p.pair}</td><td>${p.horizon}</td><td>${dir}</td><td>${(p.probability*100).toFixed(1)}%</td><td>${p.logged_at?.slice(0,16)||'--'}</td><td>${p.matures_at?.slice(0,16)||'--'}</td><td>${resultCell}</td><td>${retCell}</td></tr>`;
    }
    t += '</table>';
    container.innerHTML = `<div style="font-size:12px;color:#7d8590;margin-bottom:6px;">${data.count} predictions</div>` + t;
  } catch(e) {
    container.innerHTML = '<div class="bad" style="padding:12px;">Error: ' + e.message + '</div>';
  }
}

loadData();
setInterval(loadData, 5 * 60 * 1000);  // Auto-refresh every 5 min
</script>
</body>
</html>"""


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LumenY Paper Trading Tracker')
    parser.add_argument('command', choices=['setup', 'log', 'resolve', 'report', 'run'])
    args = parser.parse_args()

    if args.command == 'setup':
        setup_db()
        print(f'Done. Database at: {DB_PATH}')

    elif args.command == 'log':
        setup_db()
        async def _log():
            buf = await _init_buffer()
            await log_predictions(buf)
        asyncio.run(_log())

    elif args.command == 'resolve':
        setup_db()
        asyncio.run(resolve_outcomes())

    elif args.command == 'report':
        print_report()

    elif args.command == 'run':
        import uvicorn

        setup_db()

        async def _start():
            # Start the monitoring API server in the background
            config = uvicorn.Config(monitor_app, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)), log_level='info')
            server = uvicorn.Server(config)
            asyncio.create_task(server.serve())

            # Run the paper trading loop
            await run_loop()

        asyncio.run(_start())