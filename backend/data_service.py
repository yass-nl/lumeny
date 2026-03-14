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
    Provides the data needed for microstructure feature computation.

    Fetches: 1m (3 days), 5m (3 days), 15m (7 days), 1H (30 days).
    1m/5m/15m are needed for compute_features_for_pair().
    1H is needed for entry/exit price references and resolution.
    """

    def __init__(self):
        # {pair: {tf: pd.DataFrame}}
        self.buffers: dict[str, dict[str, pd.DataFrame]] = {}
        self._initialized = False

    async def initialize(self):
        """
        Load enough historical data to compute microstructure features.
        Fetches 1m, 5m, 15m from Polygon REST directly.
        1H fetched for entry/exit price lookup (not needed for features).
        """
        logger.info('Initializing candle buffers from Polygon REST...')

        now = datetime.now(timezone.utc)
        to_date = now.strftime('%Y-%m-%d')

        # Bar durations in minutes for each fetched timeframe
        tf_durations_min = {'1m': 1, '5m': 5, '15m': 15, '1H': 60}

        # Fetch config: timeframe -> (multiplier, timespan, lookback_days)
        # 1m: 3 days gives ~60*24*3 = ~4320 bars (enough for warmup + current)
        # 5m: 3 days gives ~12*24*3 = ~864 bars
        # 15m: 7 days gives ~4*24*7 = ~672 bars
        # 1H: 30 days gives ~24*30 = ~720 bars (for price lookup / resolution)
        tf_fetch_config = {
            '1m':  (1,   'minute', 3),
            '5m':  (5,   'minute', 3),
            '15m': (15,  'minute', 7),
            '1H':  (1,   'hour',   30),
        }

        for pair in PAIRS:
            logger.info(f'  Fetching {pair}...')
            self.buffers[pair] = {}

            for tf_name, (multiplier, timespan, lookback_days) in tf_fetch_config.items():
                from_date = (now - timedelta(days=lookback_days)).strftime('%Y-%m-%d')
                await asyncio.sleep(0.3)  # avoid Polygon rate limit
                try:
                    df = await fetch_historical_bars(pair, multiplier, timespan, from_date, to_date)
                    if not df.empty:
                        # Filter out weekend market closure: Sat all day + Sun before 21:00 UTC
                        df = df[~((df.index.dayofweek == 5) |
                                  ((df.index.dayofweek == 6) & (df.index.hour < 21)))]
                        # Drop bars that haven't fully closed yet.
                        bar_dur = timedelta(minutes=tf_durations_min[tf_name])
                        now_naive = now.replace(tzinfo=None)
                        df = df[df.index + bar_dur <= now_naive]
                        self.buffers[pair][tf_name] = df
                        logger.info(f'    {pair} {tf_name}: {len(df)} candles ({df.index[0].date()} to {df.index[-1].date()})')
                    else:
                        logger.warning(f'    {pair} {tf_name}: no data')
                except Exception as e:
                    logger.warning(f'    {pair} {tf_name} error: {e}')

        self._initialized = True
        logger.info('Candle buffers initialized.')

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
