"""
LumenY -- Paper Trading Tracker v5.1

Single horizon (4H forward), per-pair cooldown, meta-model filtering.

Two jobs:
  1. log_predictions()  -- runs every hour, logs model predictions to SQLite
  2. resolve_outcomes() -- runs every hour, checks past predictions and records actual outcomes

Usage:
  python paper_trading.py setup    -- create DB tables
  python paper_trading.py log      -- log predictions right now
  python paper_trading.py resolve  -- resolve any mature predictions
  python paper_trading.py report   -- print calibration report
  python paper_trading.py run      -- run both jobs in a loop (every hour)
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

# -- Paths --

BACKEND_DIR  = Path(__file__).parent
VOLUME_DIR   = Path(os.environ.get('VOLUME_PATH', str(BACKEND_DIR)))
DB_PATH      = VOLUME_DIR / 'paper_trading.db'
LOG_PATH     = BACKEND_DIR / 'paper_trading.log'

# -- Logging --

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[
        logging.FileHandler(LOG_PATH),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# -- Constants --

# Entry = 1H close at prediction time (bar T).
# Exit  = 1H close 4 hours later (bar T+4).
# matures_at = last_candle_time + 5H (so resolve fires after exit bar closes).
MATURITY_HOURS = 5

# Per-pair cooldown: once a trade is logged for a pair, skip that pair for 4 hours.
COOLDOWN_HOURS = 4

AVG_SPREAD = 0.00028


# -- Database --

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
            logged_at        TEXT NOT NULL,
            pair             TEXT NOT NULL,
            direction        TEXT NOT NULL,
            q25              REAL NOT NULL,
            q50              REAL NOT NULL,
            q75              REAL NOT NULL,
            meta_proba       REAL NOT NULL,
            is_tradeable     INTEGER NOT NULL,
            entry_price      REAL,
            matures_at       TEXT NOT NULL,
            resolved_at      TEXT,
            exit_price       REAL,
            actual_return    REAL,
            correct          INTEGER,
            notes            TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_matures_at ON predictions(matures_at);
        CREATE INDEX IF NOT EXISTS idx_pair       ON predictions(pair);
        CREATE INDEX IF NOT EXISTS idx_resolved   ON predictions(resolved_at);
        CREATE INDEX IF NOT EXISTS idx_logged_at  ON predictions(logged_at);

        CREATE TABLE IF NOT EXISTS hourly_log (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            logged_at   TEXT NOT NULL,
            pairs_logged INTEGER NOT NULL,
            pairs_skipped_cooldown INTEGER NOT NULL DEFAULT 0,
            errors      TEXT
        );
    """)
    conn.commit()
    conn.close()
    logger.info(f'Database ready at {DB_PATH}')


# -- Cooldown tracking --

def _get_cooldown_pairs(conn, now: datetime) -> set:
    """Return pairs that have a trade logged within the last COOLDOWN_HOURS."""
    cutoff = (now - timedelta(hours=COOLDOWN_HOURS)).isoformat()
    rows = conn.execute("""
        SELECT DISTINCT pair FROM predictions
        WHERE is_tradeable = 1 AND logged_at > ?
    """, (cutoff,)).fetchall()
    return {row['pair'] for row in rows}


# -- Inference --

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
    """
    import sys
    sys.path.insert(0, str(BACKEND_DIR))

    from data_service import PAIRS
    from inference import Predictor
    from features import compute_features_for_pair

    predictor = Predictor()

    predictions = {}
    prices = {}

    for pair in PAIRS:
        try:
            ohlcv = buf.get_ohlcv(pair)

            # Need 1m, 5m, 15m for microstructure features
            if '1m' not in ohlcv or '5m' not in ohlcv or '15m' not in ohlcv:
                logger.warning(f'Missing sub-hourly data for {pair}')
                continue

            df_1m = ohlcv['1m']
            df_5m = ohlcv['5m']
            df_15m = ohlcv['15m']

            if len(df_1m) < 120:  # need at least ~2 hours of 1m data
                logger.warning(f'Insufficient 1m data for {pair}: {len(df_1m)} bars')
                continue

            # Drop last candle from each TF -- may be incomplete (partially formed
            # from live minute aggregates). Matches backtest and main.py behavior.
            df_1m = df_1m.iloc[:-1] if len(df_1m) > 1 else df_1m
            df_5m = df_5m.iloc[:-1] if len(df_5m) > 1 else df_5m
            df_15m = df_15m.iloc[:-1] if len(df_15m) > 1 else df_15m

            # Compute microstructure features
            features_df = compute_features_for_pair(pair, df_1m, df_5m, df_15m)

            if features_df.empty:
                logger.warning(f'Empty features for {pair}')
                continue

            result = predictor.predict(features_df, pair)
            predictions[pair] = result

            # Entry price = close of last complete 1H candle
            if '1H' in ohlcv and not ohlcv['1H'].empty:
                latest_close = float(ohlcv['1H']['close'].iloc[-1])
            else:
                # Fallback: use last 1m close
                latest_close = float(df_1m['close'].iloc[-1])
            prices[pair] = latest_close

        except Exception as e:
            logger.error(f'Inference error for {pair}: {e}', exc_info=True)

    return predictions, prices


async def _run_inference(buf) -> tuple[dict, dict]:
    """Run inference in a thread so it doesn't block the event loop."""
    return await asyncio.to_thread(_run_inference_sync, buf)


# -- Log predictions --

async def log_predictions(buf):
    """Fetch predictions for all pairs and log them to the database."""
    now = datetime.now(timezone.utc)

    # Skip during weekend market closure
    # Also skip Friday evening when maturity would land in the weekend gap.
    # FX closes ~Fri 22:00 UTC, reopens ~Sun 21:00 UTC.
    # Backtest filter: no Saturday (dayofweek==5), no Sunday before 21:00.
    if now.weekday() == 5 or (now.weekday() == 6 and now.hour < 21):
        logger.info('Market closed (weekend) -- skipping inference.')
        return 0
    # On Friday, skip if the exit candle would not fully close before market
    # closes (~22:00 UTC). Maturity = last_candle + 5H, exit bar closes at
    # matures_at, so we need matures_at <= Fri 22:00.  Simpler: skip if
    # now + MATURITY_HOURS lands on Sat/Sun OR on Friday >= 22:00.
    if now.weekday() == 4:
        maturity_approx = now + timedelta(hours=MATURITY_HOURS)
        if maturity_approx.weekday() in (5, 6) or (maturity_approx.weekday() == 4 and maturity_approx.hour >= 22):
            logger.info('Friday late session -- exit candle would land after market close, skipping inference.')
            return 0

    logger.info('Running inference...')
    predictions, prices = await _run_inference(buf)

    now = datetime.now(timezone.utc)
    conn = get_db()
    logged = 0
    skipped_cooldown = 0
    errors = []

    # Get pairs on cooldown
    cooldown_pairs = _get_cooldown_pairs(conn, now)

    for pair, result in predictions.items():
        entry_price = prices.get(pair)

        # Get the timestamp of the last complete 1H candle
        ohlcv = buf.get_ohlcv(pair)
        if ohlcv and '1H' in ohlcv and not ohlcv['1H'].empty:
            last_candle_time = ohlcv['1H'].index[-1].to_pydatetime()
            if last_candle_time.tzinfo is None:
                last_candle_time = last_candle_time.replace(tzinfo=timezone.utc)
        else:
            last_candle_time = now

        matures_at = last_candle_time + timedelta(hours=MATURITY_HOURS)

        try:
            # Skip if a prediction for this exact maturity already exists
            existing = conn.execute("""
                SELECT id FROM predictions
                WHERE pair = ? AND matures_at = ?
            """, (pair, matures_at.isoformat())).fetchone()
            if existing:
                continue

            # Check cooldown for tradeable signals
            if result['is_tradeable'] and pair in cooldown_pairs:
                skipped_cooldown += 1
                # Still log it but mark as not tradeable due to cooldown
                result = dict(result)
                result['is_tradeable'] = False
                result['_cooldown_skipped'] = True

            conn.execute("""
                INSERT INTO predictions (
                    logged_at, pair, direction,
                    q25, q50, q75,
                    meta_proba, is_tradeable,
                    entry_price, matures_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                now.isoformat(),
                pair,
                result['direction'],
                result['q25'],
                result['q50'],
                result['q75'],
                result['meta_proba'],
                int(result['is_tradeable']),
                entry_price,
                matures_at.isoformat(),
            ))
            logged += 1

        except Exception as e:
            msg = f'{pair}: {e}'
            errors.append(msg)
            logger.error(msg)

    conn.execute(
        "INSERT INTO hourly_log (logged_at, pairs_logged, pairs_skipped_cooldown, errors) VALUES (?, ?, ?, ?)",
        (now.isoformat(), logged, skipped_cooldown,
         '; '.join(errors) if errors else None)
    )
    conn.commit()
    conn.close()

    logger.info(f'Logged {logged} predictions for {len(predictions)} pairs. '
                f'Cooldown skipped: {skipped_cooldown}. (candle T={last_candle_time.isoformat()})')
    return logged


# -- Resolve outcomes --

async def _fetch_close_price(pair: str, at_time: datetime) -> float | None:
    """
    Fetch the 1H candle close price at a specific time from Polygon REST.
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
        logger.info(f'Polygon {pair} at {at_time}: {len(results)} bars returned')
        if not results:
            return None

        # Find the bar whose CLOSE corresponds to at_time.
        # Bar opens at T, closes at T+1H. We want bar with open < at_time.
        target_ts = int(at_time.timestamp() * 1000)
        best = None
        for bar in results:
            if bar['t'] < target_ts:
                best = bar
            else:
                break

        if best is not None:
            # Validate: bar open should be exactly 1H before at_time
            # (tolerance of 5 min for slight timestamp drifts)
            expected_open_ms = target_ts - 3600 * 1000
            if abs(best['t'] - expected_open_ms) > 300_000:
                logger.warning(
                    f'{pair}: nearest bar open {best["t"]} too far from expected '
                    f'{expected_open_ms} (at_time={at_time}) -- likely weekend gap, skipping'
                )
                return None
            return float(best['c'])

        return None

    except Exception as e:
        logger.warning(f'Failed to fetch close price for {pair} at {at_time}: {e}')
        return None


async def resolve_outcomes():
    """
    Find predictions that have matured and not yet resolved.
    Fetches actual prices from Polygon REST.
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
        pair = row['pair']

        try:
            matures_at = datetime.fromisoformat(row['matures_at'])

            exit_price = await _fetch_close_price(pair, matures_at)
            if exit_price is None:
                logger.warning(f'No candle yet for {pair} matured at {matures_at}')
                continue

            entry_price = row['entry_price']

            if entry_price is None or entry_price == 0:
                logger.warning(f'No entry price for prediction {row["id"]}')
                continue

            # Log return (matches training labels)
            actual_return = np.log(exit_price / entry_price)

            # Was the directional prediction correct?
            # Only score tradeable predictions
            direction = row['direction']
            is_tradeable = row['is_tradeable']

            if not is_tradeable:
                correct = None  # non-signal, still record return
            elif direction == 'bullish':
                correct = int(actual_return > 0)
            elif direction == 'bearish':
                correct = int(actual_return < 0)
            else:
                correct = None

            conn.execute("""
                UPDATE predictions
                SET resolved_at   = ?,
                    exit_price    = ?,
                    actual_return = ?,
                    correct       = ?
                WHERE id = ?
            """, (
                now.isoformat(),
                exit_price,
                round(actual_return, 8),
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


# -- Report --

def print_report():
    """Print accuracy report on all resolved predictions."""
    conn = get_db()

    total = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
    resolved = conn.execute("SELECT COUNT(*) FROM predictions WHERE resolved_at IS NOT NULL").fetchone()[0]
    pending = total - resolved

    print('\n' + '=' * 65)
    print('LUMENY PAPER TRADING REPORT v5.1')
    print('=' * 65)
    print(f'Total predictions logged : {total:,}')
    print(f'Resolved                 : {resolved:,}')
    print(f'Pending                  : {pending:,}')

    if resolved == 0:
        print('\nNo resolved predictions yet.')
        conn.close()
        return

    df = pd.read_sql("""
        SELECT * FROM predictions WHERE resolved_at IS NOT NULL AND correct IS NOT NULL
    """, conn)
    df_all = pd.read_sql("SELECT * FROM predictions WHERE resolved_at IS NOT NULL", conn)

    if len(df) == 0:
        print('\nNo scored predictions yet.')
        conn.close()
        return

    print(f'\nDate range: {df["logged_at"].min()[:10]} -> {df["logged_at"].max()[:10]}')

    # -- Overall accuracy --
    print('\n' + '-' * 65)
    print('OVERALL (tradeable signals only)')
    print('-' * 65)
    overall_acc = df['correct'].mean()
    print(f'  Accuracy: {overall_acc:.1%}  (n={len(df):,})')

    # -- By meta_proba threshold --
    print('\n' + '-' * 65)
    print('ACCURACY BY META PROBABILITY THRESHOLD')
    print('-' * 65)
    thresholds = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
    print(f'  {"Threshold":<12} {"Accuracy":<12} {"Count":<8} {"Avg |Return|"}')
    print(f'  {"-"*48}')

    for t in thresholds:
        mask = df['meta_proba'] >= t
        sub = df[mask]
        if len(sub) < 5:
            continue
        acc = sub['correct'].mean()
        avg_ret = sub['actual_return'].abs().mean()
        print(f'  P>={t:.2f}      {acc:.1%}        {len(sub):<8} {avg_ret:.6f}')

    # -- By pair --
    print('\n' + '-' * 65)
    print('ACCURACY BY PAIR')
    print('-' * 65)
    print(f'  {"Pair":<10} {"Accuracy":<12} {"Count":<8} {"Win Rate"}')
    print(f'  {"-"*40}')

    for pair in sorted(df['pair'].unique()):
        p_df = df[df['pair'] == pair]
        if len(p_df) < 3:
            continue
        acc = p_df['correct'].mean()
        print(f'  {pair:<10} {acc:.1%}        {len(p_df):<8}')

    # -- Economic significance --
    print('\n' + '-' * 65)
    print('ECONOMIC SIGNIFICANCE')
    print('-' * 65)

    avg_move = df['actual_return'].abs().mean()
    win_rate = df['correct'].mean()
    ev = (win_rate * avg_move) - ((1 - win_rate) * avg_move) - AVG_SPREAD
    print(f'  Avg |move|: {avg_move:.6f}')
    print(f'  Win rate:   {win_rate:.1%}')
    print(f'  EV/trade:   {ev:.6f}  (after {AVG_SPREAD:.5f} spread)')

    # -- Weekly summary --
    print('\n' + '-' * 65)
    print('WEEKLY SUMMARY')
    print('-' * 65)
    df['logged_dt'] = pd.to_datetime(df['logged_at'])
    df['week'] = df['logged_dt'].dt.isocalendar().week.astype(int)
    df['year'] = df['logged_dt'].dt.isocalendar().year.astype(int)

    weekly = df.groupby(['year', 'week']).agg(
        n=('correct', 'count'),
        acc=('correct', 'mean'),
    ).reset_index()

    print(f'  {"Year":<6} {"Week":<6} {"n":<8} {"Accuracy"}')
    print(f'  {"-"*30}')
    for _, row in weekly.iterrows():
        print(f'  {int(row["year"]):<6} {int(row["week"]):<6} {int(row["n"]):<8} {row["acc"]:.1%}')

    print('\n' + '=' * 65)
    conn.close()


# -- Continuous loop --

async def run_loop():
    """Run log + resolve every hour indefinitely."""
    logger.info('Starting paper trading loop (every 60 minutes)...')
    while True:
        try:
            buf = await _init_buffer()
            await resolve_outcomes()
            await log_predictions(buf)
        except Exception as e:
            logger.error(f'Loop error: {e}', exc_info=True)

        # Sleep until XX:02 of the next hour — run as early as possible
        # after the 1H candle closes to minimize position entry delay.
        now = datetime.now(timezone.utc)
        next_run = (now + timedelta(hours=1)).replace(minute=2, second=0, microsecond=0)
        sleep_secs = (next_run - now).total_seconds()
        logger.info(f'Sleeping until {next_run.strftime("%H:%M")} UTC ({sleep_secs/60:.1f} min)...')
        await asyncio.sleep(sleep_secs)


# -- Monitoring API --

from fastapi import FastAPI, Query as Q, Depends, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import jwt

monitor_app = FastAPI(title='LumenY Paper Trading Monitor v5.1', version='2.0.0')
monitor_app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_methods=['*'], allow_headers=['*'])

# -- Auth --

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


# -- API endpoints --

@monitor_app.get('/api/monitor/report')
async def api_report(_: None = Depends(_require_auth)):
    """Full accuracy report."""
    return _get_report_data()


@monitor_app.get('/api/monitor/predictions')
async def api_predictions(
    pair: str = Q(default=None),
    resolved: bool = Q(default=None),
    tradeable: bool = Q(default=None),
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
    if resolved is not None:
        if resolved:
            query += " AND resolved_at IS NOT NULL"
        else:
            query += " AND resolved_at IS NULL"
    if tradeable is not None:
        query += " AND is_tradeable = ?"
        params.append(int(tradeable))
    query += " ORDER BY logged_at DESC LIMIT ?"
    params.append(limit)

    rows = conn.execute(query, params).fetchall()
    conn.close()
    return {'predictions': [dict(r) for r in rows], 'count': len(rows)}


@monitor_app.get('/api/monitor/summary')
async def api_summary(_: None = Depends(_require_auth)):
    """Quick summary."""
    conn = get_db()
    total = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
    resolved = conn.execute("SELECT COUNT(*) FROM predictions WHERE resolved_at IS NOT NULL").fetchone()[0]
    tradeable_total = conn.execute("SELECT COUNT(*) FROM predictions WHERE is_tradeable = 1").fetchone()[0]

    summary = {
        'total': total,
        'resolved': resolved,
        'pending': total - resolved,
        'tradeable_total': tradeable_total,
    }

    if resolved > 0:
        row = conn.execute(
            "SELECT COUNT(*) as n, AVG(correct) as acc FROM predictions WHERE resolved_at IS NOT NULL AND correct IS NOT NULL"
        ).fetchone()
        if row['n'] > 0:
            summary['accuracy'] = {'n': row['n'], 'accuracy': round(float(row['acc']), 4)}

    conn.close()
    return summary


@monitor_app.get('/api/monitor/debug')
async def api_debug():
    """Last 20 resolved predictions -- no auth required."""
    conn = get_db()
    rows = conn.execute("""
        SELECT id, pair, direction, q25, q50, q75, meta_proba, is_tradeable,
               entry_price, exit_price, actual_return, correct,
               logged_at, matures_at, resolved_at
        FROM predictions
        WHERE resolved_at IS NOT NULL
        ORDER BY id DESC LIMIT 20
    """).fetchall()
    conn.close()
    return {'predictions': [dict(r) for r in rows], 'count': len(rows)}


@monitor_app.get('/api/monitor/live-snapshot')
async def api_live_snapshot():
    """Fetch last 3 days of 1H OHLCV for each pair -- no auth required."""
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

                current_hour_ms = int(now.replace(minute=0, second=0, microsecond=0).timestamp() * 1000)
                closed_bars = [b for b in results if b['t'] < current_hour_ms]
                last_closed = closed_bars[-1] if closed_bars else None

                snapshot['pairs'][pair] = {
                    'total_bars_returned': len(results),
                    'closed_bars_count': len(closed_bars),
                    'last_closed_bar': {
                        'open_time_utc': datetime.fromtimestamp(last_closed['t'] / 1000, tz=timezone.utc).isoformat() if last_closed else None,
                        'close': last_closed['c'] if last_closed else None,
                    },
                    'expected_entry_price': last_closed['c'] if last_closed else None,
                }
            except Exception as e:
                snapshot['pairs'][pair] = {'error': str(e)}

    return snapshot


@monitor_app.get('/api/monitor/health')
async def api_health(_: None = Depends(_require_auth)):
    conn = get_db()
    last_log = conn.execute("SELECT * FROM hourly_log ORDER BY id DESC LIMIT 1").fetchone()
    conn.close()
    return {
        'status': 'ok',
        'db_path': str(DB_PATH),
        'last_log': dict(last_log) if last_log else None,
    }


def _get_report_data() -> dict:
    """Generate the full report as a JSON-serializable dict."""
    conn = get_db()
    total = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
    resolved_count = conn.execute("SELECT COUNT(*) FROM predictions WHERE resolved_at IS NOT NULL").fetchone()[0]
    tradeable_count = conn.execute("SELECT COUNT(*) FROM predictions WHERE is_tradeable = 1").fetchone()[0]
    pending_count = total - resolved_count

    report = {
        'total_predictions': total,
        'resolved': resolved_count,
        'pending': pending_count,
        'tradeable_total': tradeable_count,
        'accuracy': {},
        'by_threshold': [],
        'by_pair': {},
        'economic': {},
        'weekly': [],
    }

    if resolved_count == 0:
        conn.close()
        return report

    df = pd.read_sql("SELECT * FROM predictions WHERE resolved_at IS NOT NULL AND correct IS NOT NULL", conn)
    df_all = pd.read_sql("SELECT * FROM predictions WHERE resolved_at IS NOT NULL", conn)

    if len(df) == 0:
        conn.close()
        return report

    report['date_range'] = {
        'from': df['logged_at'].min()[:10],
        'to': df['logged_at'].max()[:10],
    }

    # Overall accuracy (tradeable only)
    report['accuracy'] = {
        'n': int(len(df)),
        'value': round(float(df['correct'].mean()), 4),
    }

    # By meta threshold (strict > to match backtest)
    for t in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
        mask = df['meta_proba'] > t
        sub = df[mask]
        if len(sub) < 3:
            continue
        report['by_threshold'].append({
            'threshold': t,
            'accuracy': round(float(sub['correct'].mean()), 4),
            'count': int(len(sub)),
            'avg_return': round(float(sub['actual_return'].abs().mean()), 8),
        })

    # By pair
    for pair in sorted(df['pair'].unique()):
        p_df = df[df['pair'] == pair]
        if len(p_df) < 3:
            continue
        report['by_pair'][pair] = {
            'n': int(len(p_df)),
            'accuracy': round(float(p_df['correct'].mean()), 4),
        }

    # Economic
    avg_move = float(df['actual_return'].abs().mean())
    win_rate = float(df['correct'].mean())
    ev = (win_rate * avg_move) - ((1 - win_rate) * avg_move) - AVG_SPREAD
    report['economic'] = {
        'avg_move': round(avg_move, 8),
        'win_rate': round(win_rate, 4),
        'ev_per_trade': round(ev, 8),
        'n': int(len(df)),
    }

    # Weekly
    df['logged_dt'] = pd.to_datetime(df['logged_at'])
    df['week'] = df['logged_dt'].dt.isocalendar().week.astype(int)
    df['year'] = df['logged_dt'].dt.isocalendar().year.astype(int)
    weekly = df.groupby(['year', 'week']).agg(n=('correct', 'count'), acc=('correct', 'mean')).reset_index()
    for _, row in weekly.iterrows():
        report['weekly'].append({
            'year': int(row['year']),
            'week': int(row['week']),
            'n': int(row['n']),
            'accuracy': round(float(row['acc']), 4),
        })

    conn.close()
    return report


@monitor_app.get('/api/monitor/dashboard-data')
async def api_dashboard_data(_: None = Depends(_require_auth)):
    """All data needed for the monitoring dashboard in one call."""
    conn = get_db()
    total = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
    resolved_count = conn.execute("SELECT COUNT(*) FROM predictions WHERE resolved_at IS NOT NULL").fetchone()[0]
    tradeable_count = conn.execute("SELECT COUNT(*) FROM predictions WHERE is_tradeable = 1").fetchone()[0]

    data = {
        'total': total,
        'resolved': resolved_count,
        'pending': total - resolved_count,
        'tradeable_total': tradeable_count,
        'accuracy': None,
        'by_threshold': [],
        'pairs': {},
        'daily_accuracy': [],
        'weekly': [],
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
    df_all = pd.read_sql("SELECT * FROM predictions WHERE resolved_at IS NOT NULL", conn)
    conn.close()

    if len(df) == 0:
        return data

    data['date_range'] = {
        'from': df['logged_at'].min()[:10],
        'to': df['logged_at'].max()[:10],
    }

    # Overall accuracy
    data['accuracy'] = {
        'n': int(len(df)),
        'value': round(float(df['correct'].mean()), 4),
    }

    # Compute PnL per trade: pred_dir * actual_return - spread
    # pred_dir = +1 for bullish, -1 for bearish
    df['pred_dir'] = df['direction'].map({'bullish': 1, 'bearish': -1}).fillna(0)
    df['pnl'] = df['pred_dir'] * df['actual_return'] - AVG_SPREAD

    # By meta threshold (strict > to match backtest)
    for t in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
        mask = df['meta_proba'] > t
        sub = df[mask]
        if len(sub) < 3:
            continue
        sub_pnl = sub['pnl']
        data['by_threshold'].append({
            'threshold': t,
            'accuracy': round(float(sub['correct'].mean()), 4),
            'count': int(len(sub)),
            'ev_bps': round(float(sub_pnl.mean() * 10000), 2),
            'total_pnl': round(float(sub_pnl.sum()), 6),
            'avg_move_bps': round(float(sub['actual_return'].abs().mean() * 10000), 2),
        })

    # Per-pair stats
    for pair in sorted(df['pair'].unique()):
        p_df = df[df['pair'] == pair]
        if len(p_df) < 1:
            continue
        p_pnl = p_df['pnl']
        data['pairs'][pair] = {
            'n': int(len(p_df)),
            'accuracy': round(float(p_df['correct'].mean()), 4),
            'ev_bps': round(float(p_pnl.mean() * 10000), 2),
            'total_pnl': round(float(p_pnl.sum()), 6),
        }

    # Daily accuracy + PnL trend
    df['logged_dt'] = pd.to_datetime(df['logged_at'])
    df['date'] = df['logged_dt'].dt.date.astype(str)
    daily = df.groupby('date').agg(
        n=('correct', 'count'),
        acc=('correct', 'mean'),
        daily_pnl=('pnl', 'sum'),
    ).reset_index()
    cum_pnl = 0.0
    for _, row in daily.iterrows():
        cum_pnl += row['daily_pnl']
        data['daily_accuracy'].append({
            'date': row['date'],
            'n': int(row['n']),
            'accuracy': round(float(row['acc']), 4),
            'daily_pnl_bps': round(float(row['daily_pnl'] * 10000), 2),
            'cum_pnl_bps': round(float(cum_pnl * 10000), 2),
        })

    # Weekly
    df['week'] = df['logged_dt'].dt.isocalendar().week.astype(int)
    df['year'] = df['logged_dt'].dt.isocalendar().year.astype(int)
    weekly = df.groupby(['year', 'week']).agg(
        n=('correct', 'count'),
        acc=('correct', 'mean'),
        weekly_pnl=('pnl', 'sum'),
    ).reset_index()
    for _, row in weekly.iterrows():
        data['weekly'].append({
            'year': int(row['year']),
            'week': int(row['week']),
            'n': int(row['n']),
            'accuracy': round(float(row['acc']), 4),
            'pnl_bps': round(float(row['weekly_pnl'] * 10000), 2),
        })

    # Economic
    avg_move = float(df['actual_return'].abs().mean())
    win_rate = float(df['correct'].mean())
    total_pnl = float(df['pnl'].sum())
    ev = float(df['pnl'].mean())
    data['economic'] = {
        'avg_move_bps': round(avg_move * 10000, 2),
        'win_rate': round(win_rate, 4),
        'ev_per_trade_bps': round(ev * 10000, 2),
        'total_pnl': round(total_pnl, 6),
        'total_pnl_bps': round(total_pnl * 10000, 2),
        'n': int(len(df)),
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
<title>LumenY Paper Trading v5.1</title>
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
      <h1>LumenY Paper Trading v5.1</h1>
      <div class="subtitle">Microstructure model &mdash; 4H forward, meta-filtered, per-pair cooldown</div>
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
function showLogin(error) {
  document.body.innerHTML = `
    <div style="display:flex;align-items:center;justify-content:center;min-height:100vh;background:#0a0e17;">
      <div style="background:#0d1420;border:1px solid #1e2d45;border-radius:12px;padding:40px;width:340px;text-align:center;">
        <div style="font-size:22px;font-weight:700;color:#58a6ff;margin-bottom:8px;">LumenY v5.1</div>
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

async function loadData() {
  try {
    const res = await fetch('/api/monitor/dashboard-data');
    if (res.status === 401) { showLogin(); return; }
    const d = await res.json();
    render(d);
  } catch(e) {
    document.getElementById('grid').innerHTML = '<div class="loading">Error: ' + e.message + '</div>';
  }
}

function render(d) {
  const sb = document.getElementById('statusBar');
  const lastLog = d.last_log ? d.last_log.logged_at : 'never';
  const cooldownSkipped = d.last_log ? (d.last_log.pairs_skipped_cooldown || 0) : 0;
  sb.innerHTML = `
    <div class="status-item"><span class="label">Last run:</span> <span class="value">${lastLog}</span></div>
    <div class="status-item"><span class="label">Total:</span> <span class="value">${d.total}</span></div>
    <div class="status-item"><span class="label">Resolved:</span> <span class="value good">${d.resolved}</span></div>
    <div class="status-item"><span class="label">Pending:</span> <span class="value warn">${d.pending}</span></div>
    <div class="status-item"><span class="label">Tradeable:</span> <span class="value">${d.tradeable_total}</span></div>
    <div class="status-item"><span class="label">Date range:</span> <span class="value">${d.date_range ? d.date_range.from + ' to ' + d.date_range.to : 'N/A'}</span></div>
  `;

  const kpi = document.getElementById('kpiRow');
  const acc = d.accuracy ? d.accuracy.value : null;
  const evBps = d.economic ? d.economic.ev_per_trade_bps : null;
  const totalPnlBps = d.economic ? d.economic.total_pnl_bps : null;
  const totalPnlRaw = d.economic ? d.economic.total_pnl : null;

  let verdict = 'wait', verdictText = 'Collecting data...';
  if (d.resolved >= 50) {
    if (acc >= 0.55) { verdict = 'pass'; verdictText = 'Model performing as expected'; }
    else if (acc < 0.50) { verdict = 'fail'; verdictText = 'Model underperforming -- investigate'; }
    else { verdict = 'wait'; verdictText = 'Borderline -- need more data'; }
  }

  kpi.innerHTML = `
    <div class="kpi"><div class="kpi-value ${acc != null ? accClass(acc) : 'neutral'}">${acc != null ? fmt(acc) : '--'}</div><div class="kpi-label">Win Rate (tradeable)</div></div>
    <div class="kpi"><div class="kpi-value ${evBps != null && evBps > 0 ? 'good' : evBps != null ? 'bad' : 'neutral'}">${evBps != null ? evBps.toFixed(1) + ' bps' : '--'}</div><div class="kpi-label">EV / Trade</div></div>
    <div class="kpi"><div class="kpi-value ${totalPnlBps != null && totalPnlBps > 0 ? 'good' : totalPnlBps != null ? 'bad' : 'neutral'}">${totalPnlBps != null ? totalPnlBps.toFixed(0) + ' bps' : '--'}</div><div class="kpi-label">Cumulative PnL</div></div>
    <div class="kpi"><div class="kpi-value ${totalPnlRaw != null && totalPnlRaw > 0 ? 'good' : totalPnlRaw != null ? 'bad' : 'neutral'}">${totalPnlRaw != null ? totalPnlRaw.toFixed(4) : '--'}</div><div class="kpi-label">Total PnL (log return)</div></div>
  `;

  const grid = document.getElementById('grid');
  grid.innerHTML = '';

  // Verdict
  grid.innerHTML += `<div class="card full-width"><div class="verdict ${verdict}">${verdictText}${d.resolved < 50 ? ' (' + d.resolved + '/50 resolved)' : ''}</div></div>`;

  // Accuracy by meta threshold
  if (d.by_threshold.length > 0) {
    let tTable = '<table><tr><th>Threshold</th><th>Win Rate</th><th>EV/Trade</th><th>Total PnL</th><th>Avg Move</th><th>Trades</th></tr>';
    for (const t of d.by_threshold) {
      const evC = t.ev_bps > 0 ? 'good' : 'bad';
      const pnlC = t.total_pnl > 0 ? 'good' : 'bad';
      const highlight = t.threshold === 0.55 ? ' style="background:#58a6ff10"' : '';
      tTable += `<tr${highlight}><td>P > ${t.threshold.toFixed(2)}</td><td class="${accClass(t.accuracy)}">${fmt(t.accuracy)}</td><td class="${evC}">${t.ev_bps.toFixed(1)} bps</td><td class="${pnlC}">${(t.total_pnl * 10000).toFixed(0)} bps</td><td>${t.avg_move_bps.toFixed(1)} bps</td><td>${t.count}</td></tr>`;
    }
    tTable += '</table>';
    grid.innerHTML += `<div class="card"><h2>Performance by Meta Threshold</h2>${tTable}</div>`;
  }

  // Pair breakdown
  if (Object.keys(d.pairs).length > 0) {
    let pTable = '<table><tr><th>Pair</th><th>N</th><th>Win Rate</th><th>EV/Trade</th><th>Total PnL</th></tr>';
    for (const [pair, pd] of Object.entries(d.pairs).sort((a,b) => b[1].total_pnl - a[1].total_pnl)) {
      const pnlC = pd.total_pnl > 0 ? 'good' : 'bad';
      pTable += `<tr><td>${pair}</td><td>${pd.n}</td><td class="${accClass(pd.accuracy)}">${fmt(pd.accuracy)}</td><td class="${pd.ev_bps > 0 ? 'good' : 'bad'}">${pd.ev_bps.toFixed(1)} bps</td><td class="${pnlC}">${(pd.total_pnl * 10000).toFixed(0)} bps</td></tr>`;
    }
    pTable += '</table>';
    grid.innerHTML += `<div class="card"><h2>Performance by Pair</h2>${pTable}</div>`;
  }

  // Daily charts
  if (d.daily_accuracy.length > 0) {
    grid.innerHTML += '<div class="card full-width"><h2>Daily Win Rate</h2><div class="chart-container"><canvas id="dailyChart"></canvas></div></div>';
    grid.innerHTML += '<div class="card full-width"><h2>Cumulative PnL (bps)</h2><div class="chart-container"><canvas id="pnlChart"></canvas></div></div>';
  }

  // Economic
  if (d.economic && d.economic.n > 0) {
    const e = d.economic;
    const evClass = e.ev_per_trade_bps > 0 ? 'good' : 'bad';
    const pnlClass = e.total_pnl_bps > 0 ? 'good' : 'bad';
    grid.innerHTML += `<div class="card"><h2>Economic Value</h2>
      <table>
        <tr><td>Win Rate</td><td class="${accClass(e.win_rate)}">${fmt(e.win_rate)}</td></tr>
        <tr><td>Avg |Move|</td><td>${e.avg_move_bps.toFixed(1)} bps</td></tr>
        <tr><td>EV / Trade</td><td class="${evClass}">${e.ev_per_trade_bps.toFixed(1)} bps</td></tr>
        <tr><td>Total PnL</td><td class="${pnlClass}">${e.total_pnl_bps.toFixed(0)} bps</td></tr>
        <tr><td>Trades</td><td>${e.n}</td></tr>
      </table>
    </div>`;
  }

  // Prediction Explorer
  grid.innerHTML += `<div class="card full-width">
    <h2>Prediction Explorer</h2>
    <div style="display:flex; gap:12px; flex-wrap:wrap; margin-bottom:14px;">
      <select id="expPair" style="background:#0d1117;color:#c9d1d9;border:1px solid #30363d;border-radius:6px;padding:6px 10px;font-size:13px;">
        <option value="">All Pairs</option>
        ${Object.keys(d.pairs || {}).map(p => '<option>' + p + '</option>').join('')}
      </select>
      <select id="expResolved" style="background:#0d1117;color:#c9d1d9;border:1px solid #30363d;border-radius:6px;padding:6px 10px;font-size:13px;">
        <option value="">All</option>
        <option value="true">Resolved</option>
        <option value="false">Pending</option>
      </select>
      <select id="expTradeable" style="background:#0d1117;color:#c9d1d9;border:1px solid #30363d;border-radius:6px;padding:6px 10px;font-size:13px;">
        <option value="">All</option>
        <option value="true">Tradeable</option>
        <option value="false">Non-tradeable</option>
      </select>
      <button class="refresh-btn" onclick="loadPredictions()">Load</button>
    </div>
    <div id="expTable" style="max-height:400px;overflow-y:auto;"></div>
  </div>`;
  loadPredictions();

  // Render charts
  if (d.daily_accuracy.length > 0) {
    // Win rate chart
    const dailyCtx = document.getElementById('dailyChart').getContext('2d');
    new Chart(dailyCtx, {
      type: 'line',
      data: {
        labels: d.daily_accuracy.map(x => x.date),
        datasets: [
          {
            label: 'Win Rate',
            data: d.daily_accuracy.map(x => x.accuracy * 100),
            borderColor: '#58a6ff',
            backgroundColor: '#58a6ff20',
            fill: true,
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
          y: { title: { display: true, text: 'Win Rate %', color: '#7d8590' }, min: 30, max: 80, grid: { color: '#21262d' }, ticks: { color: '#7d8590' } },
        },
        plugins: { legend: { labels: { color: '#c9d1d9' } } },
      },
    });

    // Cumulative PnL chart
    const pnlCtx = document.getElementById('pnlChart').getContext('2d');
    new Chart(pnlCtx, {
      type: 'line',
      data: {
        labels: d.daily_accuracy.map(x => x.date),
        datasets: [
          {
            label: 'Cumulative PnL (bps)',
            data: d.daily_accuracy.map(x => x.cum_pnl_bps),
            borderColor: '#3fb950',
            backgroundColor: '#3fb95020',
            fill: true,
            tension: 0.3,
          },
          {
            label: 'Daily PnL (bps)',
            data: d.daily_accuracy.map(x => x.daily_pnl_bps),
            borderColor: '#58a6ff80',
            type: 'bar',
            backgroundColor: d.daily_accuracy.map(x => x.daily_pnl_bps >= 0 ? '#3fb95060' : '#f8514960'),
          },
        ],
      },
      options: {
        responsive: true, maintainAspectRatio: false,
        scales: {
          x: { grid: { color: '#21262d' }, ticks: { color: '#7d8590', maxTicksLimit: 14 } },
          y: { title: { display: true, text: 'PnL (bps)', color: '#7d8590' }, grid: { color: '#21262d' }, ticks: { color: '#7d8590' } },
        },
        plugins: { legend: { labels: { color: '#c9d1d9' } } },
      },
    });
  }
}

async function loadPredictions() {
  const pair = document.getElementById('expPair')?.value || '';
  const resolved = document.getElementById('expResolved')?.value || '';
  const tradeable = document.getElementById('expTradeable')?.value || '';
  const params = new URLSearchParams();
  if (pair) params.set('pair', pair);
  if (resolved) params.set('resolved', resolved);
  if (tradeable) params.set('tradeable', tradeable);
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
    let t = '<table><tr><th>Pair</th><th>Dir</th><th>Meta P</th><th>Q50</th><th>Trade</th><th>Logged</th><th>Result</th><th>Return</th></tr>';
    for (const p of data.predictions) {
      const isResolved = p.resolved_at != null;
      const resultCell = !isResolved
        ? '<span class="neutral">Pending</span>'
        : p.correct === null
          ? '<span class="neutral">--</span>'
          : `<span class="${p.correct ? 'good' : 'bad'}">${p.correct ? 'Correct' : 'Wrong'}</span>`;
      const retCell = isResolved && p.actual_return != null ? (p.actual_return * 10000).toFixed(1) + ' bps' : '--';
      const dir = p.direction === 'bullish' ? '<span class="good">UP</span>' : p.direction === 'bearish' ? '<span class="bad">DOWN</span>' : '<span class="neutral">--</span>';
      const tradeCell = p.is_tradeable ? '<span class="good">YES</span>' : '<span class="neutral">no</span>';
      t += `<tr><td>${p.pair}</td><td>${dir}</td><td>${(p.meta_proba*100).toFixed(1)}%</td><td>${(p.q50*10000).toFixed(1)}</td><td>${tradeCell}</td><td>${p.logged_at?.slice(0,16)||'--'}</td><td>${resultCell}</td><td>${retCell}</td></tr>`;
    }
    t += '</table>';
    container.innerHTML = `<div style="font-size:12px;color:#7d8590;margin-bottom:6px;">${data.count} predictions</div>` + t;
  } catch(e) {
    container.innerHTML = '<div class="bad" style="padding:12px;">Error: ' + e.message + '</div>';
  }
}

loadData();
setInterval(loadData, 5 * 60 * 1000);
</script>
</body>
</html>"""


# -- CLI --

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LumenY Paper Trading v5.1')
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
            config = uvicorn.Config(monitor_app, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)), log_level='info')
            server = uvicorn.Server(config)
            asyncio.create_task(server.serve())
            await run_loop()

        asyncio.run(_start())
