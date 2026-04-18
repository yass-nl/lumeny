"""
LumenY -- Signal Tracker v8.0

MFE model (Q50 >= 30 pips, 8h horizon) + rule-based direction system.
Informational system: fires when a big move is likely, tells direction.
No automated order execution.

Two jobs:
  1. log_predictions()  -- runs every hour 7-20 UTC, logs model output to SQLite
  2. resolve_outcomes() -- runs every hour, checks past signals and records actual MFE

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
import smtplib
import sqlite3
import time
import warnings
from datetime import datetime, timezone, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

# -- Paths --

BACKEND_DIR = Path(__file__).parent
VOLUME_DIR  = Path(os.environ.get('VOLUME_PATH', str(BACKEND_DIR)))
DB_PATH     = VOLUME_DIR / 'paper_trading.db'
LOG_PATH    = BACKEND_DIR / 'paper_trading.log'

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

# MFE signal threshold (pips)
MFE_THRESH = 30.0

# Cooldown: once a signal fires for a pair, skip it for 8 trading hours
COOLDOWN_HOURS = 8

# Maturity: resolve 8h after signal bar closes + 1h buffer
MATURITY_HOURS = 9

# Email config (Gmail SMTP) — set via environment variables
EMAIL_FROM     = os.environ.get('EMAIL_FROM', '')
EMAIL_PASSWORD = os.environ.get('EMAIL_APP_PASSWORD', '')
EMAIL_TO       = os.environ.get('EMAIL_TO', EMAIL_FROM)


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
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            logged_at           TEXT NOT NULL,
            pair                TEXT NOT NULL,
            mfe_q50_pips        REAL NOT NULL,
            is_signal           INTEGER NOT NULL,
            direction           INTEGER,
            direction_label     TEXT,
            entry_price         REAL,
            matures_at          TEXT NOT NULL,
            resolved_at         TEXT,
            actual_mfe_pips     REAL,
            actual_mae_pips     REAL,
            fwd_72h_pips        REAL,
            correct             INTEGER,
            notes               TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_matures_at ON predictions(matures_at);
        CREATE INDEX IF NOT EXISTS idx_pair       ON predictions(pair);
        CREATE INDEX IF NOT EXISTS idx_resolved   ON predictions(resolved_at);
        CREATE INDEX IF NOT EXISTS idx_logged_at  ON predictions(logged_at);
        CREATE INDEX IF NOT EXISTS idx_is_signal  ON predictions(is_signal);

        CREATE TABLE IF NOT EXISTS hourly_log (
            id                     INTEGER PRIMARY KEY AUTOINCREMENT,
            logged_at              TEXT NOT NULL,
            pairs_logged           INTEGER NOT NULL,
            signals_fired          INTEGER NOT NULL DEFAULT 0,
            pairs_skipped_cooldown INTEGER NOT NULL DEFAULT 0,
            errors                 TEXT
        );
    """)
    conn.commit()
    conn.close()
    logger.info(f'Database ready at {DB_PATH}')


# -- Cooldown tracking --

def _get_cooldown_pairs(conn, now: datetime) -> set:
    """Return pairs that have a signal logged within the last COOLDOWN_HOURS."""
    cutoff = (now - timedelta(hours=COOLDOWN_HOURS)).isoformat()
    rows = conn.execute("""
        SELECT DISTINCT pair FROM predictions
        WHERE is_signal = 1 AND logged_at > ?
    """, (cutoff,)).fetchall()
    return {row['pair'] for row in rows}


# -- Email notifications --

def send_signal_email(signals: list[dict]):
    """Send email notification when new signals fire. Silently skips if not configured."""
    if not EMAIL_FROM or not EMAIL_PASSWORD or not EMAIL_TO:
        return
    if not signals:
        return

    try:
        subject = f"LumenY Signal{'s' if len(signals) > 1 else ''}: {', '.join(s['pair'] for s in signals)}"

        lines = ["=== LumenY Signal Alert ===\n"]
        for s in signals:
            dir_str  = s.get('direction_label', 'NO_RULE')
            mfe      = s.get('mfe_q50_pips', 0)
            price    = s.get('entry_price', 0)
            logged   = s.get('logged_at', '')[:16]
            matures  = s.get('matures_at', '')[:16]
            lines.append(f"Pair       : {s['pair']}")
            lines.append(f"MFE Q50    : {mfe:.1f} pips")
            lines.append(f"Direction  : {dir_str}")
            lines.append(f"Entry price: {price}")
            lines.append(f"Signal time: {logged} UTC")
            lines.append(f"Window ends: {matures} UTC")
            lines.append("")

        body = "\n".join(lines)

        msg = MIMEMultipart()
        msg['From']    = EMAIL_FROM
        msg['To']      = EMAIL_TO
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))

        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(EMAIL_FROM, EMAIL_PASSWORD)
            server.sendmail(EMAIL_FROM, EMAIL_TO, msg.as_string())

        logger.info(f'Signal email sent for {[s["pair"] for s in signals]}')

    except Exception as e:
        logger.warning(f'Email send failed: {e}')


# -- Cross-pair feature computation (needed by inference) --

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
PAIRS = list(CURRENCY_SIGN.keys())


def _build_cross_features(close_1h_all: dict) -> dict:
    """
    Compute cross-pair features (CSI, correlation, beta, relstr) per pair.
    Returns {pair: DataFrame indexed like close_1h_all[pair]}.
    Mirrors test_live_full.py::compute_all_cross_pair_features().
    """
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
        csi_rolling[f'{col}_72h'] = csi_df[col].rolling(72, min_periods=24).sum()
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
            cols[f'relstr_{sl}_1h']  = r - p_ret
            cols[f'relstr_{sl}_4h']  = np.log(c_pair / c_pair.shift(4))  - np.log(c_peer / c_peer.shift(4))
            cols[f'relstr_{sl}_24h'] = np.log(c_pair / c_pair.shift(24)) - np.log(c_peer / c_peer.shift(24))
            cols[f'peer_{sl}_ret_1h']  = p_ret
            cols[f'peer_{sl}_ret_4h']  = np.log(c_peer / c_peer.shift(4))
            cols[f'peer_{sl}_ret_24h'] = np.log(c_peer / c_peer.shift(24))
        for col in csi_df.columns:
            cols[col]          = csi_df[col]
            cols[f'{col}_24h'] = csi_rolling_df[f'{col}_24h']
            cols[f'{col}_72h'] = csi_rolling_df[f'{col}_72h']
        result[pair] = pd.DataFrame(cols, index=r.index).astype(np.float32)
    return result


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
    Run MFE + direction inference for all pairs.
    Returns (predictions_by_pair, prices_by_pair).
    """
    import sys
    sys.path.insert(0, str(BACKEND_DIR))

    from inference import Predictor
    from features import compute_features_for_pair
    from live_features_extra import compute_momentum_calendar_features

    predictor = Predictor()

    # 1. Gather 1H closes for cross-pair computation
    close_1h_all = {}
    for pair in PAIRS:
        ohlcv = buf.get_ohlcv(pair)
        if ohlcv and '1H' in ohlcv and not ohlcv['1H'].empty:
            close_1h_all[pair] = ohlcv['1H']['close']

    cross_features = _build_cross_features(close_1h_all) if close_1h_all else {}

    predictions = {}
    prices      = {}

    for pair in PAIRS:
        try:
            ohlcv = buf.get_ohlcv(pair)
            if not ohlcv:
                continue

            if '1m' not in ohlcv or '5m' not in ohlcv or '15m' not in ohlcv:
                logger.warning(f'Missing sub-hourly data for {pair}')
                continue

            df_1m  = ohlcv['1m']
            df_5m  = ohlcv['5m']
            df_15m = ohlcv['15m']
            df_1h  = ohlcv.get('1H')

            if len(df_1m) < 120:
                logger.warning(f'Insufficient 1m data for {pair}: {len(df_1m)} bars')
                continue

            # Base microstructure + contextual features
            features_df = compute_features_for_pair(pair, df_1m, df_5m, df_15m, df_1h)
            if features_df.empty:
                logger.warning(f'Empty features for {pair}')
                continue

            # Momentum / calendar features (vol_trend, dist_5d_high, etc.)
            if df_1h is not None and not df_1h.empty:
                from features import PIP_SIZE
                df_extra = compute_momentum_calendar_features(df_1h, PIP_SIZE[pair])
                features_df = features_df.join(df_extra.reindex(features_df.index), how='left')

            # Cross-pair features
            if pair in cross_features:
                features_df = features_df.join(
                    cross_features[pair].reindex(features_df.index), how='left'
                )

            features_df['pair'] = pair

            result = predictor.predict(features_df, pair)
            predictions[pair] = result

            # Entry price = close of last complete 1H candle
            if df_1h is not None and not df_1h.empty:
                prices[pair] = float(df_1h['close'].iloc[-1])
            else:
                prices[pair] = float(df_1m['close'].iloc[-1])

        except Exception as e:
            logger.error(f'Inference error for {pair}: {e}', exc_info=True)

    logger.info(f'Inference complete: {len(predictions)}/{len(PAIRS)} pairs succeeded')
    return predictions, prices


async def _run_inference(buf) -> tuple[dict, dict]:
    return await asyncio.to_thread(_run_inference_sync, buf)


# -- Log predictions --

async def log_predictions(buf):
    """Fetch predictions for all pairs and log them to the database."""
    now = datetime.now(timezone.utc)

    logger.info('Running inference...')
    predictions, prices = await _run_inference(buf)

    now  = datetime.now(timezone.utc)
    conn = get_db()
    logged           = 0
    signals_fired    = 0
    skipped_cooldown = 0
    errors           = []
    last_candle_time = now

    cooldown_pairs = _get_cooldown_pairs(conn, now)
    new_signals    = []

    for pair, result in predictions.items():
        entry_price = prices.get(pair)

        # Get timestamp of last complete 1H candle
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
                SELECT id FROM predictions WHERE pair = ? AND matures_at = ?
            """, (pair, matures_at.isoformat())).fetchone()
            if existing:
                continue

            is_signal = result['is_signal']

            # Cooldown: only applies to signals, not MFE-below-threshold logs
            if is_signal and pair in cooldown_pairs:
                skipped_cooldown += 1
                is_signal = False   # suppress signal flag — still log the bar

            conn.execute("""
                INSERT INTO predictions (
                    logged_at, pair, mfe_q50_pips, is_signal,
                    direction, direction_label,
                    entry_price, matures_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                now.isoformat(),
                pair,
                result['mfe_q50_pips'],
                int(is_signal),
                result['direction'],
                result['direction_label'],
                entry_price,
                matures_at.isoformat(),
            ))
            logged += 1

            if is_signal:
                signals_fired += 1
                new_signals.append({
                    'pair':            pair,
                    'mfe_q50_pips':    result['mfe_q50_pips'],
                    'direction_label': result['direction_label'],
                    'entry_price':     entry_price,
                    'logged_at':       now.isoformat(),
                    'matures_at':      matures_at.isoformat(),
                })

        except Exception as e:
            msg = f'{pair}: {e}'
            errors.append(msg)
            logger.error(msg)

    conn.execute(
        "INSERT INTO hourly_log (logged_at, pairs_logged, signals_fired, pairs_skipped_cooldown, errors) VALUES (?, ?, ?, ?, ?)",
        (now.isoformat(), logged, signals_fired, skipped_cooldown,
         '; '.join(errors) if errors else None)
    )
    conn.commit()
    conn.close()

    logger.info(
        f'Logged {logged} bars. Signals: {signals_fired}. '
        f'Cooldown skipped: {skipped_cooldown}. (candle T={last_candle_time.isoformat()})'
    )

    # Email on new signals
    if new_signals:
        send_signal_email(new_signals)

    return logged


# -- Resolve outcomes --

async def _fetch_1h_bars(pair: str, from_dt: datetime, to_dt: datetime) -> list:
    """Fetch 1H OHLCV bars from Polygon for a time window."""
    import httpx
    api_key = os.environ.get('POLYGON_API_KEY', os.environ.get('POLYGON_S3_SECRET_KEY', ''))
    ticker  = f'C:{pair}'
    url     = f'https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/hour/{from_dt.strftime("%Y-%m-%d")}/{to_dt.strftime("%Y-%m-%d")}'
    params  = {'apiKey': api_key, 'limit': 50000, 'sort': 'asc'}
    try:
        async with httpx.AsyncClient(timeout=20) as client:
            resp = await client.get(url, params=params)
            resp.raise_for_status()
            return resp.json().get('results', [])
    except Exception as e:
        logger.warning(f'Failed to fetch bars for {pair}: {e}')
        return []


async def resolve_outcomes():
    """
    Find signals that have matured and compute actual outcomes:
      - actual_mfe_pips : max(up_move, down_move) over the 8h window
      - actual_mae_pips : min(up_move, down_move) over the 8h window
      - fwd_72h_pips    : price at T+8h minus entry (signed, column name kept for DB compat)
      - correct         : 1 if direction matches fwd_8h sign, else 0 (NULL if no direction)
    """
    from features import PIP_SIZE

    now  = datetime.now(timezone.utc)
    conn = get_db()

    pending = conn.execute("""
        SELECT * FROM predictions
        WHERE resolved_at IS NULL
          AND matures_at <= ?
          AND is_signal = 1
    """, (now.isoformat(),)).fetchall()

    if not pending:
        logger.info('No pending signals to resolve.')
        conn.close()
        return 0

    logger.info(f'Resolving {len(pending)} matured signals...')
    resolved = 0

    for row in pending:
        pair = row['pair']
        try:
            matures_at = datetime.fromisoformat(row['matures_at'])
            if matures_at.tzinfo is None:
                matures_at = matures_at.replace(tzinfo=timezone.utc)

            # If matured > 48h ago and still unresolved, mark as expired
            if (now - matures_at).total_seconds() > 48 * 3600:
                conn.execute(
                    "UPDATE predictions SET resolved_at = ? WHERE id = ?",
                    (now.isoformat(), row['id'])
                )
                resolved += 1
                continue

            logged_at = datetime.fromisoformat(row['logged_at'])
            if logged_at.tzinfo is None:
                logged_at = logged_at.replace(tzinfo=timezone.utc)

            # Fetch ~12h of 1H bars starting from signal bar
            bars = await _fetch_1h_bars(
                pair,
                logged_at - timedelta(hours=2),
                matures_at + timedelta(hours=2),
            )
            if not bars:
                continue

            # Convert to arrays, filter to the 72h forward window
            bar_times  = [datetime.fromtimestamp(b['t'] / 1000, tz=timezone.utc) for b in bars]
            bar_highs  = [b['h'] for b in bars]
            bar_lows   = [b['l'] for b in bars]
            bar_closes = [b['c'] for b in bars]

            entry_price = row['entry_price']
            if entry_price is None or entry_price == 0:
                logger.warning(f'No entry price for prediction {row["id"]}')
                continue

            # Find entry bar index (closest bar at or after logged_at)
            entry_idx = None
            for i, t in enumerate(bar_times):
                if t >= logged_at:
                    entry_idx = i
                    break
            if entry_idx is None:
                continue

            # Window: next 8 bars (hours) after entry
            window_end_idx = min(entry_idx + 8, len(bars) - 1)
            if window_end_idx <= entry_idx:
                continue

            h_slice = bar_highs[entry_idx + 1 : window_end_idx + 1]
            l_slice = bar_lows[entry_idx + 1  : window_end_idx + 1]
            c_slice = bar_closes[entry_idx + 1: window_end_idx + 1]

            if not h_slice:
                continue

            pip = PIP_SIZE.get(pair, 0.0001)

            up_move   = (max(h_slice) - entry_price) / pip
            down_move = (entry_price - min(l_slice))  / pip

            actual_mfe_pips = round(max(up_move, down_move), 1)
            actual_mae_pips = round(min(up_move, down_move), 1)

            # Forward price at T+72h (last close in window)
            exit_price    = c_slice[-1]
            fwd_72h_pips  = round((exit_price - entry_price) / pip, 1)

            # Direction correctness
            direction = row['direction']
            if direction is None:
                correct = None
            else:
                correct = 1 if (direction == 1 and fwd_72h_pips > 0) or \
                                (direction == -1 and fwd_72h_pips < 0) else 0

            conn.execute("""
                UPDATE predictions
                SET resolved_at     = ?,
                    actual_mfe_pips = ?,
                    actual_mae_pips = ?,
                    fwd_72h_pips    = ?,
                    correct         = ?
                WHERE id = ?
            """, (
                now.isoformat(),
                actual_mfe_pips,
                actual_mae_pips,
                fwd_72h_pips,
                correct,
                row['id'],
            ))
            resolved += 1

        except Exception as e:
            logger.error(f'Resolution error for prediction {row["id"]}: {e}', exc_info=True)

    conn.commit()
    conn.close()
    logger.info(f'Resolved {resolved} signals.')
    return resolved


# -- Report (CLI) --

def print_report():
    conn = get_db()

    total    = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
    signals  = conn.execute("SELECT COUNT(*) FROM predictions WHERE is_signal = 1").fetchone()[0]
    resolved = conn.execute("SELECT COUNT(*) FROM predictions WHERE resolved_at IS NOT NULL AND is_signal = 1").fetchone()[0]
    pending  = signals - resolved

    print('\n' + '=' * 65)
    print('LUMENY SIGNAL TRACKER v7.0')
    print('=' * 65)
    print(f'Total bars logged        : {total:,}')
    print(f'Signals (MFE >= 70)      : {signals:,}')
    print(f'Resolved                 : {resolved:,}')
    print(f'Pending                  : {pending:,}')

    if resolved == 0:
        print('\nNo resolved signals yet.')
        conn.close()
        return

    df = pd.read_sql("""
        SELECT * FROM predictions WHERE resolved_at IS NOT NULL AND is_signal = 1
    """, conn)
    conn.close()

    if len(df) == 0:
        print('\nNo data.'); return

    print(f'\nDate range: {df["logged_at"].min()[:10]} -> {df["logged_at"].max()[:10]}')

    # MFE model accuracy
    df_mfe = df.dropna(subset=['actual_mfe_pips'])
    if len(df_mfe) > 0:
        mfe_acc = (df_mfe['actual_mfe_pips'] >= 70.0).mean()
        avg_actual_mfe = df_mfe['actual_mfe_pips'].mean()
        avg_pred_mfe   = df_mfe['mfe_q50_pips'].mean()
        print(f'\n--- MFE Model Accuracy ---')
        print(f'  Actual MFE >= 70 pips  : {mfe_acc:.1%}  (n={len(df_mfe)})')
        print(f'  Avg predicted MFE Q50  : {avg_pred_mfe:.1f} pips')
        print(f'  Avg actual MFE         : {avg_actual_mfe:.1f} pips')

    # Direction accuracy
    df_dir = df.dropna(subset=['correct'])
    if len(df_dir) > 0:
        dir_acc = df_dir['correct'].mean()
        print(f'\n--- Direction Accuracy (72h fwd) ---')
        print(f'  Accuracy : {dir_acc:.1%}  (n={len(df_dir)})')

    # By pair
    print(f'\n--- By Pair ---')
    for pair in sorted(df['pair'].unique()):
        p = df[df['pair'] == pair]
        p_dir = p.dropna(subset=['correct'])
        mfe_a = (p.dropna(subset=['actual_mfe_pips'])['actual_mfe_pips'] >= 70).mean() if len(p.dropna(subset=['actual_mfe_pips'])) > 0 else float('nan')
        dir_a = p_dir['correct'].mean() if len(p_dir) > 0 else float('nan')
        avg_mfe = p['mfe_q50_pips'].mean()
        print(f'  {pair:<10} n={len(p):<4}  MFE_acc={mfe_a:.0%}  Dir_acc={dir_a:.0%}  avg_Q50={avg_mfe:.1f}p')

    print('\n' + '=' * 65)


# -- Continuous loop --

async def run_loop():
    logger.info('Starting signal tracking loop (every hour)...')
    while True:
        try:
            buf = await _init_buffer()

            # 1. Resolve matured signals
            await resolve_outcomes()

            # 2. Run inference and log
            logged = await log_predictions(buf)

            # Retry if no bars logged (candle may not be ready yet)
            now_check = datetime.now(timezone.utc)
            if logged == 0 and now_check.minute < 3:
                for attempt in range(1, 4):
                    logger.info(f'No bars logged, candle may not be ready. Retry {attempt}/3 in 15s...')
                    await asyncio.sleep(15)
                    buf    = await _init_buffer()
                    logged = await log_predictions(buf)
                    if logged > 0:
                        break

        except Exception as e:
            logger.error(f'Loop error: {e}', exc_info=True)

        # Sleep until XX:00:10 of the next hour
        now      = datetime.now(timezone.utc)
        next_run = (now + timedelta(hours=1)).replace(minute=0, second=10, microsecond=0)
        if next_run <= now:
            next_run += timedelta(hours=1)
        sleep_secs = (next_run - now).total_seconds()
        logger.info(f'Sleeping until {next_run.strftime("%H:%M:%S")} UTC ({sleep_secs/60:.1f} min)...')
        await asyncio.sleep(sleep_secs)


# -- Monitoring API --

from fastapi import FastAPI, Query as Q, Depends, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
import jwt

monitor_app = FastAPI(title='LumenY Signal Tracker v7.0', version='7.0.0')
monitor_app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_methods=['*'], allow_headers=['*'])

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
        _JWT_SECRET, algorithm='HS256',
    )
    response = JSONResponse({'ok': True, 'expires_in_hours': _TOKEN_TTL_H})
    response.set_cookie(key='lumeny_session', value=token, httponly=True,
                        max_age=_TOKEN_TTL_H * 3600, samesite='lax')
    return response


@monitor_app.get('/api/signals')
async def api_signals(
    pair:     str  = Q(default=None),
    resolved: bool = Q(default=None),
    limit:    int  = Q(default=100),
    _: None = Depends(_require_auth),
):
    """Browse raw signals."""
    conn  = get_db()
    query = "SELECT * FROM predictions WHERE is_signal = 1"
    params = []
    if pair:
        query += " AND pair = ?"
        params.append(pair.upper())
    if resolved is not None:
        query += " AND resolved_at IS NOT NULL" if resolved else " AND resolved_at IS NULL"
    query += " ORDER BY logged_at DESC LIMIT ?"
    params.append(limit)
    rows = conn.execute(query, params).fetchall()
    conn.close()
    return {'signals': [dict(r) for r in rows], 'count': len(rows)}


@monitor_app.get('/api/monitor/health')
async def api_health():
    conn     = get_db()
    last_log = conn.execute("SELECT * FROM hourly_log ORDER BY id DESC LIMIT 1").fetchone()
    signals  = conn.execute("SELECT COUNT(*) FROM predictions WHERE is_signal = 1").fetchone()[0]
    pending  = conn.execute("SELECT COUNT(*) FROM predictions WHERE is_signal = 1 AND resolved_at IS NULL").fetchone()[0]
    conn.close()
    return {
        'status':        'ok',
        'db_path':       str(DB_PATH),
        'total_signals': signals,
        'pending':       pending,
        'last_log':      dict(last_log) if last_log else None,
    }


@monitor_app.get('/api/monitor/summary')
async def api_summary(_: None = Depends(_require_auth)):
    """Quick stats."""
    conn = get_db()
    total_bars = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
    total_sigs = conn.execute("SELECT COUNT(*) FROM predictions WHERE is_signal = 1").fetchone()[0]
    resolved   = conn.execute("SELECT COUNT(*) FROM predictions WHERE is_signal = 1 AND resolved_at IS NOT NULL").fetchone()[0]
    row = conn.execute("""
        SELECT COUNT(*) as n, AVG(correct) as acc
        FROM predictions WHERE is_signal = 1 AND resolved_at IS NOT NULL AND correct IS NOT NULL
    """).fetchone()
    mfe_row = conn.execute("""
        SELECT AVG(CASE WHEN actual_mfe_pips >= 70 THEN 1.0 ELSE 0.0 END) as mfe_acc,
               AVG(actual_mfe_pips) as avg_mfe
        FROM predictions WHERE is_signal = 1 AND actual_mfe_pips IS NOT NULL
    """).fetchone()
    conn.close()
    return {
        'total_bars':    total_bars,
        'total_signals': total_sigs,
        'resolved':      resolved,
        'pending':       total_sigs - resolved,
        'dir_accuracy':  round(float(row['acc']), 4) if row['n'] > 0 and row['acc'] is not None else None,
        'dir_n':         row['n'],
        'mfe_accuracy':  round(float(mfe_row['mfe_acc']), 4) if mfe_row['mfe_acc'] is not None else None,
        'avg_actual_mfe': round(float(mfe_row['avg_mfe']), 1) if mfe_row['avg_mfe'] is not None else None,
    }


@monitor_app.get('/api/monitor/predictions')
async def api_predictions(
    pair:      str  = Q(default=None),
    resolved:  bool = Q(default=None),
    signal_only: bool = Q(default=True),
    limit:     int  = Q(default=100),
    _: None = Depends(_require_auth),
):
    """Browse raw predictions with filters."""
    conn   = get_db()
    query  = "SELECT * FROM predictions WHERE 1=1"
    params = []
    if signal_only:
        query += " AND is_signal = 1"
    if pair:
        query += " AND pair = ?"
        params.append(pair.upper())
    if resolved is not None:
        query += " AND resolved_at IS NOT NULL" if resolved else " AND resolved_at IS NULL"
    query += " ORDER BY logged_at DESC LIMIT ?"
    params.append(limit)
    rows = conn.execute(query, params).fetchall()
    conn.close()
    return {'predictions': [dict(r) for r in rows], 'count': len(rows)}


@monitor_app.get('/api/monitor/report')
async def api_report(_: None = Depends(_require_auth)):
    """Full accuracy report."""
    conn = get_db()
    df = pd.read_sql(
        "SELECT * FROM predictions WHERE is_signal = 1 AND resolved_at IS NOT NULL", conn
    )
    conn.close()

    if len(df) == 0:
        return {'message': 'No resolved signals yet.'}

    df_mfe = df.dropna(subset=['actual_mfe_pips'])
    df_dir = df.dropna(subset=['correct'])

    by_pair = {}
    for pair in sorted(df['pair'].unique()):
        p     = df[df['pair'] == pair]
        p_mfe = p.dropna(subset=['actual_mfe_pips'])
        p_dir = p.dropna(subset=['correct'])
        by_pair[pair] = {
            'n':            int(len(p)),
            'avg_q50':      round(float(p['mfe_q50_pips'].mean()), 1),
            'mfe_acc':      round(float((p_mfe['actual_mfe_pips'] >= 70).mean()), 4) if len(p_mfe) > 0 else None,
            'avg_actual_mfe': round(float(p_mfe['actual_mfe_pips'].mean()), 1) if len(p_mfe) > 0 else None,
            'dir_acc':      round(float(p_dir['correct'].mean()), 4) if len(p_dir) > 0 else None,
            'dir_n':        int(len(p_dir)),
        }

    df['logged_dt'] = pd.to_datetime(df['logged_at'])
    df['month'] = df['logged_dt'].dt.to_period('M').astype(str)
    monthly = df.groupby('month').agg(
        n=('mfe_q50_pips', 'count'),
        avg_q50=('mfe_q50_pips', 'mean'),
        dir_acc=('correct', 'mean'),
        mfe_acc_raw=('actual_mfe_pips', lambda x: (x >= 70).mean() if x.notna().any() else None),
    ).reset_index()

    return {
        'date_range':   {'from': df['logged_at'].min()[:10], 'to': df['logged_at'].max()[:10]},
        'total_signals': int(len(df)),
        'mfe_accuracy': {
            'n':              int(len(df_mfe)),
            'pct_above_70':   round(float((df_mfe['actual_mfe_pips'] >= 70).mean()), 4) if len(df_mfe) > 0 else None,
            'avg_actual_mfe': round(float(df_mfe['actual_mfe_pips'].mean()), 1) if len(df_mfe) > 0 else None,
            'avg_pred_mfe':   round(float(df['mfe_q50_pips'].mean()), 1),
        },
        'dir_accuracy': {
            'n':     int(len(df_dir)),
            'value': round(float(df_dir['correct'].mean()), 4) if len(df_dir) > 0 else None,
        },
        'by_pair':   by_pair,
        'by_month':  monthly.to_dict(orient='records'),
    }


@monitor_app.get('/api/monitor/debug')
async def api_debug():
    """Last 20 resolved signals — no auth required."""
    conn = get_db()
    rows = conn.execute("""
        SELECT id, pair, mfe_q50_pips, is_signal, direction, direction_label,
               entry_price, actual_mfe_pips, actual_mae_pips, fwd_72h_pips, correct,
               logged_at, matures_at, resolved_at
        FROM predictions
        WHERE is_signal = 1 AND resolved_at IS NOT NULL
        ORDER BY id DESC LIMIT 20
    """).fetchall()
    conn.close()
    return {'predictions': [dict(r) for r in rows], 'count': len(rows)}


@monitor_app.get('/api/monitor/live-snapshot')
async def api_live_snapshot():
    """Last 3 days of 1H OHLCV per pair — no auth required."""
    import httpx
    api_key = os.environ.get('POLYGON_API_KEY', os.environ.get('POLYGON_S3_SECRET_KEY', ''))
    now      = datetime.now(timezone.utc)
    from_date = (now - timedelta(days=3)).strftime('%Y-%m-%d')
    to_date   = now.strftime('%Y-%m-%d')

    snapshot = {'generated_at': now.isoformat(), 'pairs': {}}

    async with httpx.AsyncClient(timeout=20) as client:
        for pair in PAIRS:
            ticker = f'C:{pair}'
            url    = f'https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/hour/{from_date}/{to_date}'
            try:
                resp = await client.get(url, params={'apiKey': api_key, 'limit': 200, 'sort': 'asc'})
                resp.raise_for_status()
                results = resp.json().get('results', [])
                if not results:
                    snapshot['pairs'][pair] = {'error': 'no data'}
                    continue
                current_hour_ms = int(now.replace(minute=0, second=0, microsecond=0).timestamp() * 1000)
                closed = [b for b in results if b['t'] < current_hour_ms]
                last   = closed[-1] if closed else None
                snapshot['pairs'][pair] = {
                    'bars_returned':  len(results),
                    'closed_bars':    len(closed),
                    'last_close':     float(last['c']) if last else None,
                    'last_bar_utc':   datetime.fromtimestamp(last['t'] / 1000, tz=timezone.utc).isoformat() if last else None,
                }
            except Exception as e:
                snapshot['pairs'][pair] = {'error': str(e)}

    return snapshot


def _get_dashboard_data() -> dict:
    """All data for the dashboard in one call."""
    conn = get_db()

    total_bars = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
    total_sigs = conn.execute("SELECT COUNT(*) FROM predictions WHERE is_signal = 1").fetchone()[0]
    resolved   = conn.execute("SELECT COUNT(*) FROM predictions WHERE is_signal = 1 AND resolved_at IS NOT NULL").fetchone()[0]
    pending    = total_sigs - resolved

    data = {
        'total_bars':    total_bars,
        'total_signals': total_sigs,
        'resolved':      resolved,
        'pending':       pending,
        'accuracy':      None,
        'mfe_accuracy':  None,
        'by_pair':       {},
        'recent_signals': [],
        'weekly':        [],
        'last_log':      None,
    }

    last_log = conn.execute("SELECT * FROM hourly_log ORDER BY id DESC LIMIT 1").fetchone()
    if last_log:
        data['last_log'] = dict(last_log)

    if resolved == 0:
        # Still return recent signals (pending ones)
        rows = conn.execute("""
            SELECT * FROM predictions WHERE is_signal = 1
            ORDER BY logged_at DESC LIMIT 30
        """).fetchall()
        data['recent_signals'] = [dict(r) for r in rows]
        conn.close()
        return data

    df = pd.read_sql(
        "SELECT * FROM predictions WHERE is_signal = 1 AND resolved_at IS NOT NULL", conn
    )

    if len(df) > 0:
        data['date_range'] = {
            'from': df['logged_at'].min()[:10],
            'to':   df['logged_at'].max()[:10],
        }

        # MFE model accuracy
        df_mfe = df.dropna(subset=['actual_mfe_pips'])
        if len(df_mfe) > 0:
            data['mfe_accuracy'] = {
                'n':              int(len(df_mfe)),
                'pct_above_70':   round(float((df_mfe['actual_mfe_pips'] >= 70).mean()), 4),
                'avg_actual_mfe': round(float(df_mfe['actual_mfe_pips'].mean()), 1),
                'avg_pred_mfe':   round(float(df_mfe['mfe_q50_pips'].mean()), 1),
            }

        # Direction accuracy
        df_dir = df.dropna(subset=['correct'])
        if len(df_dir) > 0:
            data['accuracy'] = {
                'n':     int(len(df_dir)),
                'value': round(float(df_dir['correct'].mean()), 4),
            }

        # By pair
        for pair in sorted(df['pair'].unique()):
            p     = df[df['pair'] == pair]
            p_dir = p.dropna(subset=['correct'])
            p_mfe = p.dropna(subset=['actual_mfe_pips'])
            data['by_pair'][pair] = {
                'n':            int(len(p)),
                'avg_q50':      round(float(p['mfe_q50_pips'].mean()), 1),
                'mfe_acc':      round(float((p_mfe['actual_mfe_pips'] >= 70).mean()), 4) if len(p_mfe) > 0 else None,
                'dir_acc':      round(float(p_dir['correct'].mean()), 4) if len(p_dir) > 0 else None,
                'dir_n':        int(len(p_dir)),
                'last_dir':     str(p['direction_label'].iloc[-1]) if len(p) > 0 else None,
            }

        # Weekly
        df['logged_dt'] = pd.to_datetime(df['logged_at'])
        df['week'] = df['logged_dt'].dt.isocalendar().week.astype(int)
        df['year'] = df['logged_dt'].dt.isocalendar().year.astype(int)
        weekly = df.groupby(['year', 'week']).agg(
            n=('mfe_q50_pips', 'count'),
            avg_q50=('mfe_q50_pips', 'mean'),
            dir_acc=('correct', 'mean'),
        ).reset_index()
        for _, row in weekly.iterrows():
            data['weekly'].append({
                'year':    int(row['year']),
                'week':    int(row['week']),
                'n':       int(row['n']),
                'avg_q50': round(float(row['avg_q50']), 1),
                'dir_acc': round(float(row['dir_acc']), 4) if not pd.isna(row['dir_acc']) else None,
            })

    # Recent signals (last 30, resolved + pending)
    rows = conn.execute("""
        SELECT * FROM predictions WHERE is_signal = 1
        ORDER BY logged_at DESC LIMIT 30
    """).fetchall()
    data['recent_signals'] = [dict(r) for r in rows]

    conn.close()
    return data


@monitor_app.get('/api/monitor/dashboard-data')
async def api_dashboard_data(_: None = Depends(_require_auth)):
    return _get_dashboard_data()


@monitor_app.get('/', response_class=HTMLResponse)
@monitor_app.get('/dashboard', response_class=HTMLResponse)
async def dashboard():
    return DASHBOARD_HTML


DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>LumenY v7.0</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #0a0e17; color: #e0e0e0; }
  .header { background: linear-gradient(135deg, #1a1f2e 0%, #0d1117 100%); padding: 20px 28px; border-bottom: 1px solid #21262d; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 12px; }
  .header h1 { font-size: 20px; font-weight: 700; color: #f0f0f0; }
  .header .sub { font-size: 12px; color: #7d8590; margin-top: 3px; }
  .refresh-btn { background: #21262d; color: #c9d1d9; border: 1px solid #30363d; border-radius: 6px; padding: 6px 14px; cursor: pointer; font-size: 13px; }
  .refresh-btn:hover { background: #30363d; }

  .kpi-row { display: flex; gap: 12px; flex-wrap: wrap; padding: 16px 28px 4px; }
  .kpi { background: #161b22; border: 1px solid #21262d; border-radius: 8px; padding: 14px 18px; flex: 1; min-width: 120px; }
  .kpi .kv { font-size: 26px; font-weight: 700; }
  .kpi .kl { font-size: 11px; color: #7d8590; margin-top: 3px; text-transform: uppercase; letter-spacing: .5px; }

  .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 14px; padding: 14px 28px 28px; }
  .card { background: #161b22; border: 1px solid #21262d; border-radius: 8px; padding: 18px; }
  .card h2 { font-size: 13px; font-weight: 600; color: #8b949e; margin-bottom: 12px; text-transform: uppercase; letter-spacing: .5px; }
  .card.full { grid-column: 1 / -1; }

  table { width: 100%; border-collapse: collapse; font-size: 13px; }
  th { text-align: left; color: #7d8590; font-weight: 500; padding: 5px 8px; border-bottom: 1px solid #21262d; font-size: 12px; }
  td { padding: 6px 8px; border-bottom: 1px solid #21262d20; }
  tr:last-child td { border-bottom: none; }

  .good    { color: #3fb950; }
  .warn    { color: #d29922; }
  .bad     { color: #f85149; }
  .neutral { color: #7d8590; }
  .long    { color: #3fb950; font-weight: 600; }
  .short   { color: #f85149; font-weight: 600; }
  .norule  { color: #7d8590; font-style: italic; }

  .signal-badge { display: inline-block; padding: 2px 7px; border-radius: 4px; font-size: 11px; font-weight: 600; }
  .signal-badge.long  { background: #0d3b1b; color: #3fb950; }
  .signal-badge.short { background: #3b0d0d; color: #f85149; }
  .signal-badge.none  { background: #1c1c2a; color: #7d8590; }

  .status-bar { display: flex; gap: 20px; flex-wrap: wrap; padding: 10px 28px; background: #0d1117; border-bottom: 1px solid #21262d; font-size: 12px; }
  .si .sl { color: #7d8590; } .si .sv { color: #58a6ff; font-weight: 600; margin-left: 4px; }

  .loading { text-align: center; padding: 60px; color: #7d8590; }
  .score-bar { height: 4px; background: #21262d; border-radius: 2px; margin-top: 5px; }
  .score-fill { height: 100%; border-radius: 2px; background: #58a6ff; }
</style>
</head>
<body>

<div class="header">
  <div>
    <h1>LumenY &mdash; Signal Tracker v7.0</h1>
    <div class="sub">MFE Q50 &ge; 70 pips &bull; Rule-based direction &bull; 72h window &bull; Hours 7-20 UTC</div>
  </div>
  <button class="refresh-btn" onclick="loadData()">Refresh</button>
</div>

<div class="status-bar" id="statusBar"></div>
<div class="kpi-row" id="kpiRow"></div>
<div class="grid" id="grid"><div class="loading">Loading...</div></div>

<script>
function showLogin(err) {
  document.body.innerHTML = `
    <div style="display:flex;align-items:center;justify-content:center;min-height:100vh;background:#0a0e17;">
      <div style="background:#0d1420;border:1px solid #1e2d45;border-radius:12px;padding:40px;width:320px;text-align:center;">
        <div style="font-size:20px;font-weight:700;color:#58a6ff;margin-bottom:6px;">LumenY v7.0</div>
        <div style="color:#7d8fa3;margin-bottom:24px;font-size:12px;">Signal Tracker</div>
        ${err ? '<div style="color:#f85149;margin-bottom:14px;font-size:12px;">' + err + '</div>' : ''}
        <input id="pw" type="password" placeholder="Password" autofocus
          style="width:100%;padding:9px 12px;border-radius:7px;border:1px solid #1e2d45;background:#111827;color:#e0e0e0;font-size:14px;margin-bottom:12px;outline:none;" />
        <button onclick="doLogin()"
          style="width:100%;padding:9px;border-radius:7px;border:none;background:#58a6ff;color:#0a0e17;font-weight:700;font-size:14px;cursor:pointer;">
          Sign in
        </button>
      </div>
    </div>`;
  document.getElementById('pw').addEventListener('keydown', e => { if (e.key === 'Enter') doLogin(); });
}

async function doLogin() {
  const password = document.getElementById('pw').value;
  try {
    const res = await fetch('/auth/login', {
      method: 'POST', headers: {'Content-Type':'application/json'},
      body: JSON.stringify({ password }),
    });
    if (!res.ok) { showLogin('Invalid password.'); return; }
    location.reload();
  } catch(e) { showLogin('Login failed: ' + e.message); }
}

function pct(v, d=1)   { return v != null ? (v * 100).toFixed(d) + '%' : '--'; }
function pip(v, d=1)   { return v != null ? v.toFixed(d) + 'p' : '--'; }
function accCls(v)     { return v == null ? 'neutral' : v >= 0.60 ? 'good' : v >= 0.52 ? 'warn' : 'bad'; }
function dirBadge(lbl) {
  if (lbl === 'LONG')    return '<span class="signal-badge long">LONG</span>';
  if (lbl === 'SHORT')   return '<span class="signal-badge short">SHORT</span>';
  return '<span class="signal-badge none">NO RULE</span>';
}
function resultCell(r) {
  if (r.resolved_at == null) return '<span class="neutral">Pending</span>';
  if (r.correct == null)     return '<span class="neutral">--</span>';
  return r.correct ? '<span class="good">Correct</span>' : '<span class="bad">Wrong</span>';
}

async function loadData() {
  try {
    const res = await fetch('/api/monitor/dashboard-data');
    if (res.status === 401) { showLogin(); return; }
    render(await res.json());
  } catch(e) {
    document.getElementById('grid').innerHTML = '<div class="loading">Error: ' + e.message + '</div>';
  }
}

function render(d) {
  // Status bar
  const ll = d.last_log;
  document.getElementById('statusBar').innerHTML = `
    <div class="si"><span class="sl">Last run:</span><span class="sv">${ll ? ll.logged_at.slice(0,16) : 'never'}</span></div>
    <div class="si"><span class="sl">Signals:</span><span class="sv">${d.total_signals}</span></div>
    <div class="si"><span class="sl">Resolved:</span><span class="sv">${d.resolved}</span></div>
    <div class="si"><span class="sl">Pending:</span><span class="sv">${d.pending}</span></div>
    ${d.date_range ? '<div class="si"><span class="sl">Range:</span><span class="sv">' + d.date_range.from + ' \u2192 ' + d.date_range.to + '</span></div>' : ''}
    ${ll && ll.signals_fired > 0 ? '<div class="si"><span class="sl">Last cycle signals:</span><span class="sv good">' + ll.signals_fired + '</span></div>' : ''}
  `;

  // KPIs
  const mfe  = d.mfe_accuracy;
  const dir  = d.accuracy;
  document.getElementById('kpiRow').innerHTML = `
    <div class="kpi">
      <div class="kv ${mfe ? (mfe.pct_above_70 >= 0.55 ? 'good' : mfe.pct_above_70 >= 0.45 ? 'warn' : 'bad') : 'neutral'}">${mfe ? pct(mfe.pct_above_70) : '--'}</div>
      <div class="kl">MFE Model Accuracy</div>
    </div>
    <div class="kpi">
      <div class="kv ${accCls(dir ? dir.value : null)}">${dir ? pct(dir.value) : '--'}</div>
      <div class="kl">Direction Accuracy</div>
    </div>
    <div class="kpi">
      <div class="kv neutral">${mfe ? pip(mfe.avg_actual_mfe) : '--'}</div>
      <div class="kl">Avg Actual MFE</div>
    </div>
    <div class="kpi">
      <div class="kv neutral">${d.total_signals}</div>
      <div class="kl">Total Signals</div>
    </div>
  `;

  const grid = document.getElementById('grid');
  grid.innerHTML = '';

  // Per-pair table
  const pairs = Object.entries(d.by_pair);
  if (pairs.length > 0) {
    let t = '<table><tr><th>Pair</th><th>N</th><th>Avg Q50</th><th>MFE Acc</th><th>Dir Acc</th><th>Last Dir</th></tr>';
    for (const [pair, p] of pairs.sort((a,b) => (b[1].avg_q50||0) - (a[1].avg_q50||0))) {
      t += `<tr>
        <td><b>${pair}</b></td>
        <td>${p.n}</td>
        <td>${pip(p.avg_q50)}</td>
        <td class="${p.mfe_acc != null ? (p.mfe_acc >= 0.55 ? 'good' : p.mfe_acc >= 0.45 ? 'warn' : 'bad') : 'neutral'}">${pct(p.mfe_acc)}</td>
        <td class="${accCls(p.dir_acc)}">${pct(p.dir_acc)}${p.dir_n > 0 ? ' <span class="neutral">('+p.dir_n+')</span>' : ''}</td>
        <td>${dirBadge(p.last_dir)}</td>
      </tr>`;
    }
    t += '</table>';
    grid.innerHTML += `<div class="card"><h2>Performance by Pair</h2>${t}</div>`;
  }

  // Weekly table
  if (d.weekly.length > 0) {
    let t = '<table><tr><th>Week</th><th>Signals</th><th>Avg Q50</th><th>Dir Acc</th></tr>';
    for (const w of d.weekly.slice().reverse()) {
      t += `<tr>
        <td>${w.year}-W${String(w.week).padStart(2,'0')}</td>
        <td>${w.n}</td>
        <td>${pip(w.avg_q50)}</td>
        <td class="${accCls(w.dir_acc)}">${pct(w.dir_acc)}</td>
      </tr>`;
    }
    t += '</table>';
    grid.innerHTML += `<div class="card"><h2>Weekly Summary</h2>${t}</div>`;
  }

  // Recent signals
  if (d.recent_signals.length > 0) {
    let t = '<table><tr><th>Time (UTC)</th><th>Pair</th><th>MFE Q50</th><th>Direction</th><th>Entry</th><th>Actual MFE</th><th>72h Move</th><th>Result</th></tr>';
    for (const s of d.recent_signals) {
      const actMfe  = s.actual_mfe_pips != null ? pip(s.actual_mfe_pips) : '--';
      const fwd72   = s.fwd_72h_pips != null ? (s.fwd_72h_pips >= 0 ? '+' : '') + s.fwd_72h_pips.toFixed(1) + 'p' : '--';
      const fwdCls  = s.fwd_72h_pips == null ? 'neutral' : s.fwd_72h_pips >= 0 ? 'good' : 'bad';
      t += `<tr>
        <td>${s.logged_at ? s.logged_at.slice(0,16) : '--'}</td>
        <td><b>${s.pair}</b></td>
        <td class="warn">${pip(s.mfe_q50_pips)}</td>
        <td>${dirBadge(s.direction_label)}</td>
        <td class="neutral">${s.entry_price ? s.entry_price.toFixed(5) : '--'}</td>
        <td class="${s.actual_mfe_pips != null && s.actual_mfe_pips >= 70 ? 'good' : 'neutral'}">${actMfe}</td>
        <td class="${fwdCls}">${fwd72}</td>
        <td>${resultCell(s)}</td>
      </tr>`;
    }
    t += '</table>';
    grid.innerHTML += `<div class="card full"><h2>Recent Signals (last 30)</h2><div style="overflow-x:auto;">${t}</div></div>`;
  }

  if (grid.innerHTML === '') {
    grid.innerHTML = '<div class="card full"><div class="loading neutral" style="padding:40px;">No signals yet. System is running &mdash; waiting for MFE Q50 &ge; 70 pips.</div></div>';
  }
}

loadData();
setInterval(loadData, 5 * 60 * 1000);
</script>
</body>
</html>"""


# -- CLI --

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LumenY Signal Tracker v7.0')
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
            config = uvicorn.Config(
                monitor_app,
                host='0.0.0.0',
                port=int(os.environ.get('PORT', 8080)),
                log_level='info',
            )
            server = uvicorn.Server(config)
            asyncio.create_task(server.serve())
            await run_loop()

        asyncio.run(_start())
