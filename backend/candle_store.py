"""
SQLite-backed 1-minute candle storage.

Persists candles to disk (Railway volume) so that on each hourly cycle
we only fetch the NEW candles from Polygon instead of re-downloading 7 days.

Schema: one row per (pair, timestamp) — 1-minute OHLCV.
On startup, backfills whatever is missing.  Each cycle, appends a few rows.
"""

import logging
import os
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

BACKEND_DIR = Path(__file__).parent
VOLUME_DIR = Path(os.environ.get('VOLUME_PATH', str(BACKEND_DIR)))
CANDLES_DB_PATH = VOLUME_DIR / 'candles.db'

# We need 25 hours of 1m TRADING data for feature computation (24h trailing + 1h current).
# Forex has a ~47h weekend gap, so we keep 96h (4 days) of calendar time to always
# have enough trading candles regardless of where we are in the week.
KEEP_HOURS = 96


def get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(str(CANDLES_DB_PATH))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def setup():
    """Create the candles_1m table if it doesn't exist."""
    conn = get_db()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS candles_1m (
            pair   TEXT    NOT NULL,
            ts     TEXT    NOT NULL,
            open   REAL    NOT NULL,
            high   REAL    NOT NULL,
            low    REAL    NOT NULL,
            close  REAL    NOT NULL,
            volume REAL    NOT NULL DEFAULT 0,
            PRIMARY KEY (pair, ts)
        );
        CREATE INDEX IF NOT EXISTS idx_candles_pair_ts ON candles_1m(pair, ts);
    """)
    conn.commit()
    conn.close()
    logger.info(f'Candle store ready at {CANDLES_DB_PATH}')


def get_last_timestamp(pair: str) -> datetime | None:
    """Return the most recent candle timestamp for a pair, or None."""
    conn = get_db()
    row = conn.execute(
        "SELECT MAX(ts) FROM candles_1m WHERE pair = ?", (pair,)
    ).fetchone()
    conn.close()
    if row and row[0]:
        return datetime.fromisoformat(row[0])
    return None


def get_candles(pair: str, hours: int = KEEP_HOURS) -> pd.DataFrame:
    """
    Load the last `hours` of 1m candles for a pair from the DB.
    Returns a DataFrame indexed by datetime, columns = [open, high, low, close, volume].
    """
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours)).replace(tzinfo=None)
    conn = get_db()
    df = pd.read_sql_query(
        "SELECT ts, open, high, low, close, volume FROM candles_1m WHERE pair = ? AND ts >= ? ORDER BY ts",
        conn,
        params=(pair, cutoff.isoformat()),
    )
    conn.close()

    if df.empty:
        return pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])

    df['datetime'] = pd.to_datetime(df['ts'])
    df = df.set_index('datetime')[['open', 'high', 'low', 'close', 'volume']]
    df = df.sort_index().drop_duplicates()
    return df


def append_candles(pair: str, df_1m: pd.DataFrame):
    """
    Insert 1m candles into the DB. Duplicates are silently ignored (INSERT OR IGNORE).
    df_1m must have a datetime index and columns [open, high, low, close, volume].
    """
    if df_1m.empty:
        return

    conn = get_db()
    rows = [
        (pair, ts.isoformat(), float(r['open']), float(r['high']),
         float(r['low']), float(r['close']), float(r['volume']))
        for ts, r in df_1m.iterrows()
    ]
    conn.executemany(
        "INSERT OR IGNORE INTO candles_1m (pair, ts, open, high, low, close, volume) VALUES (?, ?, ?, ?, ?, ?, ?)",
        rows,
    )
    conn.commit()
    conn.close()
    logger.debug(f'Appended {len(rows)} candles for {pair}')


def append_single_candle(pair: str, ts: datetime, o: float, h: float, l: float, c: float, v: float):
    """Append a single 1m candle (from WebSocket). Duplicate-safe."""
    conn = get_db()
    conn.execute(
        "INSERT OR IGNORE INTO candles_1m (pair, ts, open, high, low, close, volume) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (pair, ts.isoformat(), o, h, l, c, v),
    )
    conn.commit()
    conn.close()


def trim(keep_hours: int = KEEP_HOURS):
    """Delete candles older than keep_hours for all pairs."""
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=keep_hours)).replace(tzinfo=None)
    conn = get_db()
    result = conn.execute("DELETE FROM candles_1m WHERE ts < ?", (cutoff.isoformat(),))
    deleted = result.rowcount
    conn.commit()
    conn.close()
    if deleted > 0:
        logger.info(f'Trimmed {deleted} old candles (older than {keep_hours}h)')


def count(pair: str) -> int:
    """Return number of stored candles for a pair."""
    conn = get_db()
    row = conn.execute("SELECT COUNT(*) FROM candles_1m WHERE pair = ?", (pair,)).fetchone()
    conn.close()
    return row[0] if row else 0
