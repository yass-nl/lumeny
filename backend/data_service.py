"""
Data service v5.1 — fetches OHLCV from Polygon REST API,
maintains in-memory candle buffers for microstructure feature computation,
and provides the real-time WebSocket price relay.

Fetches 1m, 5m, 15m, 1H data.  1m/5m/15m are used for feature computation.
1H is used for entry/exit prices and resolution.
"""

import os
import asyncio
import json
import logging
from datetime import datetime, timedelta, timezone
import httpx
import numpy as np
import pandas as pd
import websockets

from features import PAIRS, resample_ohlcv
import candle_store

logger = logging.getLogger(__name__)

API_KEY = os.getenv('POLYGON_S3_SECRET_KEY', '')
REST_BASE = 'https://api.polygon.io'
WS_URI = 'wss://socket.massive.com/forex'

# Polygon ticker format
POLYGON_TICKERS = {pair: f'C:{pair[:3]}-{pair[3:]}' for pair in PAIRS}
# Also keep C:EURUSD format for REST
POLYGON_REST_TICKERS = {pair: f'C:{pair}' for pair in PAIRS}


async def fetch_historical_bars(
    pair: str,
    multiplier: int,
    timespan: str,
    from_date: str,
    to_date: str,
    limit: int = 50000,
    retries: int = 3,
) -> pd.DataFrame:
    """
    Fetch historical OHLCV bars from Polygon REST API.
    Follows pagination (next_url) to get all results.
    Returns a DataFrame indexed by datetime.
    """
    ticker = POLYGON_REST_TICKERS[pair]
    url = f'{REST_BASE}/v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{from_date}/{to_date}'
    params = {'apiKey': API_KEY, 'limit': limit, 'sort': 'asc'}

    all_results = []

    for attempt in range(retries):
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.get(url, params=params)
                resp.raise_for_status()
                data = resp.json()
                all_results.extend(data.get('results', []))

                # Follow pagination
                while 'next_url' in data:
                    next_url = data['next_url']
                    # Polygon next_url doesn't include apiKey
                    sep = '&' if '?' in next_url else '?'
                    resp = await client.get(f'{next_url}{sep}apiKey={API_KEY}')
                    resp.raise_for_status()
                    data = resp.json()
                    all_results.extend(data.get('results', []))
            break  # success
        except Exception as e:
            if attempt < retries - 1:
                wait = 2 ** attempt  # 1s, 2s
                logger.warning(f'fetch_historical_bars {pair} {timespan} attempt {attempt + 1} failed: {e!r} -- retrying in {wait}s')
                await asyncio.sleep(wait)
            else:
                raise

    if not all_results:
        return pd.DataFrame()

    df = pd.DataFrame(all_results)
    df['datetime'] = pd.to_datetime(df['t'], unit='ms', utc=True).dt.tz_localize(None)
    df = df.rename(columns={'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume'})
    df = df.set_index('datetime')[['open', 'high', 'low', 'close', 'volume']]
    df = df.sort_index().drop_duplicates()
    return df


class CandleBuffer:
    """
    Maintains in-memory OHLCV buffers for all pairs and timeframes.
    Backed by a SQLite candle store so that on restart / each cycle
    we only fetch NEW candles from Polygon instead of re-downloading 7 days.

    1m/5m/15m are needed for compute_features_for_pair().
    1H is needed for entry/exit price references and resolution.
    """

    def __init__(self):
        # {pair: {tf: pd.DataFrame}}
        self.buffers: dict[str, dict[str, pd.DataFrame]] = {}
        self._initialized = False

    async def initialize(self):
        """
        Load 1m candles from the SQLite store, fetch only what's missing
        from Polygon, persist new candles, build in-memory buffers.
        1H fetched separately (for entry/exit prices, not features).
        """
        candle_store.setup()
        logger.info('Initializing candle buffers (DB-backed)...')

        now = datetime.now(timezone.utc)
        to_date = now.strftime('%Y-%m-%d')
        now_naive = now.replace(tzinfo=None)

        # Full backfill window: 96h calendar time = enough to cover weekend gap + 25h trading data
        backfill_from = (now - timedelta(hours=candle_store.KEEP_HOURS))
        # 1H: 30 days for entry/exit price lookup; use tomorrow as to_date to ensure today's latest bars are included
        from_date_1h = (now - timedelta(days=30)).strftime('%Y-%m-%d')
        to_date_1h   = (now + timedelta(days=1)).strftime('%Y-%m-%d')

        for pair in PAIRS:
            self.buffers[pair] = {}

            # --- 1m: DB-backed incremental fetch ---
            try:
                last_ts = candle_store.get_last_timestamp(pair)
                n_stored = candle_store.count(pair)

                # Need backfill if: no data, last candle too old, or not enough candles
                # (< 1440 = less than 24h of 1m bars, meaning DB is incomplete)
                needs_backfill = (
                    last_ts is None
                    or last_ts < backfill_from.replace(tzinfo=None)
                    or n_stored < 1440
                )

                if needs_backfill:
                    # Cold start or insufficient data: backfill full window
                    from_date_1m = backfill_from.strftime('%Y-%m-%d')
                    logger.info(f'  {pair} 1m: backfilling from {from_date_1m} (stored={n_stored})...')
                    await asyncio.sleep(0.3)
                    df_new = await fetch_historical_bars(pair, 1, 'minute', from_date_1m, to_date)
                else:
                    # Incremental: fetch only since last stored candle
                    # Polygon REST requires YYYY-MM-DD (not datetime) for from/to params
                    fetch_from = (last_ts - timedelta(minutes=5))
                    from_str = fetch_from.strftime('%Y-%m-%d')
                    logger.info(f'  {pair} 1m: incremental from {from_str} ({int((now_naive - last_ts).total_seconds() / 60)} min gap)...')
                    await asyncio.sleep(0.15)
                    df_new = await fetch_historical_bars(pair, 1, 'minute', from_str, to_date)

                # Filter weekends and persist new candles
                if not df_new.empty:
                    df_new = df_new[~((df_new.index.dayofweek == 5) |
                                      ((df_new.index.dayofweek == 6) & (df_new.index.hour < 21)))]
                    df_new = df_new[df_new.index + timedelta(minutes=1) <= now_naive]
                    candle_store.append_candles(pair, df_new)

                # Load full working window from DB
                df_1m = candle_store.get_candles(pair)
                if not df_1m.empty:
                    self.buffers[pair]['1m'] = df_1m
                    logger.info(f'    {pair} 1m: {len(df_1m)} candles in buffer ({df_1m.index[0]} to {df_1m.index[-1]})')

                    # Resample 5m/15m from 1m — matches training pipeline
                    df_5m = resample_ohlcv(df_1m, '5min')
                    df_15m = resample_ohlcv(df_1m, '15min')
                    df_5m = df_5m[df_5m.index + timedelta(minutes=5) <= now_naive]
                    df_15m = df_15m[df_15m.index + timedelta(minutes=15) <= now_naive]
                    self.buffers[pair]['5m'] = df_5m
                    self.buffers[pair]['15m'] = df_15m
                else:
                    logger.warning(f'    {pair} 1m: no data in store')

            except Exception as e:
                logger.warning(f'    {pair} 1m error: {e}', exc_info=True)

            # --- 1H: still fetched from Polygon (for entry/exit prices, not features) ---
            await asyncio.sleep(0.3)
            try:
                df_1h = await fetch_historical_bars(pair, 1, 'hour', from_date_1h, to_date_1h)
                if not df_1h.empty:
                    df_1h = df_1h[~((df_1h.index.dayofweek == 5) |
                                    ((df_1h.index.dayofweek == 6) & (df_1h.index.hour < 21)))]
                    df_1h = df_1h[df_1h.index + timedelta(hours=1) <= now_naive]
                    self.buffers[pair]['1H'] = df_1h
                else:
                    logger.warning(f'    {pair} 1H: no data')
            except Exception as e:
                logger.warning(f'    {pair} 1H error: {e}')

        # Trim old candles from the DB
        candle_store.trim()

        self._initialized = True
        logger.info('Candle buffers initialized (DB-backed).')

    def get_ohlcv(self, pair: str) -> dict[str, pd.DataFrame]:
        """Get all timeframe buffers for a pair."""
        return self.buffers.get(pair, {})

    def get_latest_price(self, pair: str) -> dict | None:
        """Get the latest available price for a pair."""
        if pair not in self.buffers:
            return None
        # Use the finest available timeframe
        for tf in ['1m', '5m', '15m', '1H']:
            if tf in self.buffers[pair] and not self.buffers[pair][tf].empty:
                row = self.buffers[pair][tf].iloc[-1]
                return {
                    'pair': pair,
                    'price': float(row['close']),
                    'open': float(row['open']),
                    'high': float(row['high']),
                    'low': float(row['low']),
                    'close': float(row['close']),
                    'timestamp': str(self.buffers[pair][tf].index[-1]),
                }
        return None

    def append_candle(self, pair: str, candle: dict):
        """Append a new 1-min candle from WebSocket and update higher TF buffers."""
        if pair not in self.buffers:
            return

        ts = pd.Timestamp(candle['timestamp'])
        row_data = {
            'open': candle['open'],
            'high': candle['high'],
            'low': candle['low'],
            'close': candle['close'],
            'volume': candle.get('volume', 0),
        }
        new_row = pd.DataFrame([row_data], index=[ts])

        # Persist to SQLite store (duplicate-safe)
        try:
            candle_store.append_single_candle(
                pair, ts.to_pydatetime(),
                candle['open'], candle['high'], candle['low'], candle['close'],
                candle.get('volume', 0),
            )
        except Exception as e:
            logger.debug(f'candle_store append error for {pair}: {e}')

        # Append to 1m buffer directly
        if '1m' in self.buffers[pair]:
            self.buffers[pair]['1m'] = pd.concat([self.buffers[pair]['1m'], new_row])

        # Update higher TF buffers by aggregating the new 1m candle
        tf_rules = {'5m': '5min', '15m': '15min', '1H': '1h'}
        for tf_name, tf_rule in tf_rules.items():
            if tf_name not in self.buffers[pair]:
                continue
            buf = self.buffers[pair][tf_name]
            if not buf.empty:
                resampled_ts = ts.floor(tf_rule)
                last_ts = buf.index[-1]
                if resampled_ts == last_ts:
                    # Update the current bar
                    buf.loc[last_ts, 'high'] = max(buf.loc[last_ts, 'high'], candle['high'])
                    buf.loc[last_ts, 'low'] = min(buf.loc[last_ts, 'low'], candle['low'])
                    buf.loc[last_ts, 'close'] = candle['close']
                    buf.loc[last_ts, 'volume'] += candle.get('volume', 0)
                elif resampled_ts > last_ts:
                    # New bar
                    new_tf_row = pd.DataFrame([row_data], index=[resampled_ts])
                    self.buffers[pair][tf_name] = pd.concat([buf, new_tf_row])


class PriceFeed:
    """
    Connects to Massive/Polygon WebSocket for real-time forex data.
    Relays minute aggregates to connected frontend clients.
    """

    def __init__(self, candle_buffer: CandleBuffer):
        self.candle_buffer = candle_buffer
        self.clients: set = set()
        self._ws = None
        self._running = False

    async def connect(self):
        """Connect to Massive WebSocket and subscribe to all FX pairs."""
        self._running = True

        while self._running:
            try:
                async with websockets.connect(WS_URI) as ws:
                    self._ws = ws
                    logger.info('Connected to Massive WebSocket')

                    # Wait for connection message
                    msg = await ws.recv()
                    logger.info(f'WS connected: {msg}')

                    # Authenticate
                    await ws.send(json.dumps({'action': 'auth', 'params': API_KEY}))
                    msg = await ws.recv()
                    logger.info(f'WS auth: {msg}')

                    # Subscribe to minute aggregates for all pairs
                    subs = ','.join(f'CA.{POLYGON_TICKERS[p]}' for p in PAIRS)
                    await ws.send(json.dumps({'action': 'subscribe', 'params': subs}))
                    msg = await ws.recv()
                    logger.info(f'WS subscribed: {msg}')

                    # Listen for data
                    async for raw_msg in ws:
                        try:
                            events = json.loads(raw_msg)
                            for event in events:
                                if event.get('ev') == 'CA':
                                    await self._handle_aggregate(event)
                        except json.JSONDecodeError:
                            pass

            except websockets.ConnectionClosed:
                logger.warning('WebSocket disconnected, reconnecting in 5s...')
                await asyncio.sleep(5)
            except Exception as e:
                logger.error(f'WebSocket error: {e}, reconnecting in 10s...')
                await asyncio.sleep(10)

    async def _handle_aggregate(self, event: dict):
        """Process a minute aggregate event from Polygon."""
        sym = event.get('sym', '')  # e.g. "C:EUR-USD"
        pair_raw = sym.replace('C:', '').replace('-', '')  # -> "EURUSD"

        if pair_raw not in PAIRS:
            return

        candle = {
            'pair': pair_raw,
            'open': event.get('o', 0),
            'high': event.get('h', 0),
            'low': event.get('l', 0),
            'close': event.get('c', 0),
            'volume': event.get('v', 0),
            'timestamp': datetime.fromtimestamp(event['s'] / 1000).isoformat() if 's' in event else datetime.now(timezone.utc).isoformat(),
        }

        # Update candle buffer
        self.candle_buffer.append_candle(pair_raw, candle)

        # Relay to all connected frontend clients
        message = json.dumps({
            'type': 'candle',
            'data': candle,
        })
        disconnected = set()
        for client in self.clients:
            try:
                await client.send_text(message)
            except Exception:
                disconnected.add(client)
        self.clients -= disconnected

    def stop(self):
        self._running = False
