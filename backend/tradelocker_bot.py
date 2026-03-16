"""
TradeLocker Bot -- Automated trade execution for LumenY v5.1

Integrates with the paper trading loop:
  1. After log_predictions(), place market orders for tradeable signals
  2. Every loop iteration, close positions that have reached their 4H maturity

Uses TradeLocker REST API with JWT auth.
"""

import asyncio
import logging
import os
import time
from datetime import datetime, timezone, timedelta

import httpx

logger = logging.getLogger(__name__)

# -- Config --

TRADELOCKER_EMAIL = os.environ.get('TRADELOCKER_EMAIL', '')
TRADELOCKER_PASSWORD = os.environ.get('TRADELOCKER_PASSWORD', '')
TRADELOCKER_SERVER = os.environ.get('TRADELOCKER_SERVER', '')
TRADELOCKER_ACCOUNT_ID = os.environ.get('TRADELOCKER_ACCOUNT_ID', '')

# Base URL -- detect demo vs live from server name
BASE_URL = os.environ.get(
    'TRADELOCKER_BASE_URL',
    'https://demo.tradelocker.com/backend-api'
)

# Map our pair names to TradeLocker symbol format
# TradeLocker uses "EUR/USD" format; we'll resolve tradableInstrumentId at startup
PAIR_TO_SYMBOL = {
    'EURUSD': 'EUR/USD', 'GBPUSD': 'GBP/USD', 'USDJPY': 'USD/JPY',
    'USDCHF': 'USD/CHF', 'AUDUSD': 'AUD/USD', 'USDCAD': 'USD/CAD',
    'NZDUSD': 'NZD/USD', 'EURJPY': 'EUR/JPY', 'GBPJPY': 'GBP/JPY',
    'EURGBP': 'EUR/GBP', 'EURAUD': 'EUR/AUD', 'AUDJPY': 'AUD/JPY',
    'CADJPY': 'CAD/JPY', 'CHFJPY': 'CHF/JPY', 'AUDNZD': 'AUD/NZD',
}

# Position sizing (lots) -- conservative default
DEFAULT_LOT_SIZE = 0.3

# Max concurrent positions to avoid overexposure
MAX_CONCURRENT_POSITIONS = 5

# How long to hold a position (hours)
HOLD_HOURS = 4

# TradeLocker API returns positions/orders as arrays, not dicts.
# Column indices from /trade/config:
POS_ID = 0
POS_INSTRUMENT_ID = 1
POS_ROUTE_ID = 2
POS_SIDE = 3
POS_QTY = 4
POS_AVG_PRICE = 5

# Orders history column indices
ORD_ID = 0
ORD_INSTRUMENT_ID = 1
ORD_POSITION_ID = 16


class TradeLockerBot:
    def __init__(self):
        self.access_token = None
        self.refresh_token = None
        self.token_expires_at = 0
        self.acc_num = None
        self.account_id = TRADELOCKER_ACCOUNT_ID
        self.instrument_map = {}  # pair -> tradableInstrumentId
        self.route_id = None  # TRADE routeId
        self.info_route_id = None  # INFO routeId
        # Track open positions: {prediction_id: {positionId, pair, close_at}}
        self.open_positions = {}
        self._enabled = bool(TRADELOCKER_EMAIL and TRADELOCKER_PASSWORD and TRADELOCKER_SERVER)

    def is_enabled(self):
        return self._enabled

    async def initialize(self):
        """Auth, fetch accNum, and map instruments. Call once at startup."""
        if not self._enabled:
            logger.info('TradeLocker bot disabled (missing credentials)')
            return False

        try:
            await self._authenticate()
            await self._fetch_acc_num()
            await self._map_instruments()
            await self._recover_open_positions()
            logger.info(f'TradeLocker bot initialized. {len(self.instrument_map)} instruments mapped.')
            return True
        except Exception as e:
            logger.error(f'TradeLocker init failed: {e}', exc_info=True)
            self._enabled = False
            return False

    async def _authenticate(self):
        """Get JWT access token."""
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.post(f'{BASE_URL}/auth/jwt/token', json={
                'email': TRADELOCKER_EMAIL,
                'password': TRADELOCKER_PASSWORD,
                'server': TRADELOCKER_SERVER,
            })
            resp.raise_for_status()
            data = resp.json()

        self.access_token = data['accessToken']
        self.refresh_token = data['refreshToken']
        # Token typically expires in ~30 min; refresh proactively
        self.token_expires_at = time.time() + 1500  # 25 min
        logger.info('TradeLocker authenticated successfully')

    async def _refresh_auth(self):
        """Refresh the JWT token if expired or about to expire."""
        if time.time() < self.token_expires_at:
            return

        try:
            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.post(f'{BASE_URL}/auth/jwt/refresh', json={
                    'refreshToken': self.refresh_token,
                })
                resp.raise_for_status()
                data = resp.json()

            self.access_token = data['accessToken']
            self.refresh_token = data['refreshToken']
            self.token_expires_at = time.time() + 1500
            logger.info('TradeLocker token refreshed')
        except Exception as e:
            logger.warning(f'Token refresh failed, re-authenticating: {e}')
            await self._authenticate()

    def _headers(self):
        """Standard headers for API requests."""
        h = {'Authorization': f'Bearer {self.access_token}'}
        if self.acc_num is not None:
            h['accNum'] = str(self.acc_num)
        return h

    async def _fetch_acc_num(self):
        """Fetch accNum from the all-accounts endpoint."""
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(
                f'{BASE_URL}/auth/jwt/all-accounts',
                headers={'Authorization': f'Bearer {self.access_token}'},
            )
            resp.raise_for_status()
            data = resp.json()

        accounts = data.get('accounts', data) if isinstance(data, dict) else data
        if isinstance(accounts, list) and len(accounts) > 0:
            # Find matching account by ID or use first one
            for acc in accounts:
                acc_id = str(acc.get('id', ''))
                if acc_id == str(self.account_id) or not self.account_id:
                    self.acc_num = acc.get('accNum')
                    if not self.account_id:
                        self.account_id = acc_id
                    break

        if self.acc_num is None:
            raise ValueError(f'Could not find accNum for account {self.account_id}. Accounts: {accounts}')

        logger.info(f'TradeLocker accNum={self.acc_num}, accountId={self.account_id}')

    async def _map_instruments(self):
        """Fetch instruments and map our pair names to tradableInstrumentId."""
        await self._refresh_auth()

        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(
                f'{BASE_URL}/trade/accounts/{self.account_id}/instruments',
                headers=self._headers(),
            )
            resp.raise_for_status()
            data = resp.json()

        # TradeLocker wraps responses in {'s': 'ok', 'd': {...}}
        payload = data.get('d', data) if isinstance(data, dict) else data
        instruments = payload.get('instruments', payload) if isinstance(payload, dict) else payload

        # Debug: log the structure to understand the format
        if isinstance(instruments, list) and len(instruments) > 0:
            sample = instruments[0]
            logger.info(f'TradeLocker: {len(instruments)} instruments, sample keys: {list(sample.keys()) if isinstance(sample, dict) else type(sample)}')
            for inst in instruments[:3]:
                logger.info(f'TradeLocker: instrument sample: {inst}')
        else:
            logger.warning(f'TradeLocker: unexpected instruments format after unwrap: {type(instruments)}')
            if isinstance(payload, dict):
                logger.info(f'TradeLocker: payload keys: {list(payload.keys())}')
                # Try to log a small sample of the payload
                for k, v in payload.items():
                    sample_v = str(v)[:200] if not isinstance(v, list) else f'list[{len(v)}] first={v[0] if v else "empty"}'
                    logger.info(f'TradeLocker: payload[{k}] = {sample_v}')

        # Build a lookup: symbol name -> tradableInstrumentId + routeId
        symbol_lookup = {}
        if isinstance(instruments, list):
            for inst in instruments:
                if isinstance(inst, dict):
                    name = inst.get('name', '')
                    tid = inst.get('tradableInstrumentId')
                    # routeId is inside 'routes' array: [{id: X, type: 'TRADE'}, ...]
                    routes = inst.get('routes', [])
                    trade_rid = None
                    info_rid = None
                    for r in routes:
                        if r.get('type') == 'TRADE':
                            trade_rid = r.get('id')
                        elif r.get('type') == 'INFO':
                            info_rid = r.get('id')
                else:
                    continue
                if name and tid:
                    symbol_lookup[name] = {'id': tid, 'routeId': trade_rid, 'infoRouteId': info_rid}
                    # Also try without slash
                    symbol_lookup[name.replace('/', '')] = {'id': tid, 'routeId': trade_rid, 'infoRouteId': info_rid}

        for pair, symbol in PAIR_TO_SYMBOL.items():
            info = symbol_lookup.get(symbol) or symbol_lookup.get(pair)
            if info:
                self.instrument_map[pair] = info['id']
                if self.route_id is None and info.get('routeId'):
                    self.route_id = info['routeId']
            else:
                logger.warning(f'Instrument not found for {pair} ({symbol})')

        logger.info(f'Mapped instruments: {list(self.instrument_map.keys())}')

    async def _recover_open_positions(self):
        """
        On startup, check if there are open positions on TradeLocker
        that correspond to unresolved tradeable predictions in the DB.
        Schedule them for closing at their matures_at time.
        """
        try:
            positions = await self.get_open_positions_list()
            if not positions:
                logger.info('TradeLocker: no open positions to recover')
                return

            # Build reverse map: instrument_id -> pair (both int and str keys)
            inst_to_pair = {}
            for k, v in self.instrument_map.items():
                inst_to_pair[v] = k
                inst_to_pair[str(v)] = k

            # Get unresolved tradeable predictions from DB
            import sqlite3
            from pathlib import Path
            db_path = Path(os.environ.get('VOLUME_PATH', str(Path(__file__).parent))) / 'paper_trading.db'
            conn = sqlite3.connect(str(db_path))
            conn.row_factory = sqlite3.Row
            rows = conn.execute("""
                SELECT id, pair, matures_at FROM predictions
                WHERE is_tradeable = 1 AND resolved_at IS NULL
            """).fetchall()
            conn.close()

            unresolved = {r['pair']: dict(r) for r in rows}

            recovered = 0
            for pos in positions:
                # Positions come as arrays: [posId, instrumentId, routeId, side, qty, ...]
                if isinstance(pos, list):
                    pos_id = str(pos[POS_ID])
                    inst_id = str(pos[POS_INSTRUMENT_ID])
                else:
                    inst_id = str(pos.get('tradableInstrumentId', ''))
                    pos_id = str(pos.get('id') or pos.get('positionId'))

                pair = inst_to_pair.get(inst_id) or inst_to_pair.get(int(inst_id) if inst_id.isdigit() else inst_id)

                if pair and pair in unresolved:
                    pred = unresolved[pair]
                    matures_at = datetime.fromisoformat(pred['matures_at'])
                    self.open_positions[pred['id']] = {
                        'order_id': None,
                        'position_id': pos_id,
                        'pair': pair,
                        'side': None,
                        'close_at': matures_at,
                        'instrument_id': inst_id,
                    }
                    recovered += 1
                    logger.info(
                        f'TradeLocker: recovered {pair} positionId={pos_id}, '
                        f'close_at={matures_at.isoformat()}'
                    )

            if recovered:
                logger.info(f'TradeLocker: recovered {recovered} open position(s) from restart')

        except Exception as e:
            logger.warning(f'TradeLocker: position recovery failed: {e}', exc_info=True)

    async def place_trades(self, predictions_db):
        """
        Place market orders for tradeable signals from the latest inference.

        predictions_db: list of dicts from the DB (the newly logged predictions)
        Each has: id, pair, direction, is_tradeable, meta_proba, q50, matures_at
        """
        if not self._enabled:
            return

        await self._refresh_auth()

        tradeable = [p for p in predictions_db if p['is_tradeable']]
        if not tradeable:
            logger.info('TradeLocker: no tradeable signals this cycle')
            return

        # Sort by meta_proba descending -- prioritize highest confidence
        tradeable.sort(key=lambda p: p['meta_proba'], reverse=True)

        for pred in tradeable:
            # Check max concurrent positions
            if len(self.open_positions) >= MAX_CONCURRENT_POSITIONS:
                logger.info(
                    f'TradeLocker: max concurrent positions ({MAX_CONCURRENT_POSITIONS}) reached, '
                    f'skipping remaining signals'
                )
                break

            pair = pred['pair']
            direction = pred['direction']
            pred_id = pred['id']

            # Skip if already placed (e.g. from retry)
            if pred_id in self.open_positions:
                continue

            instrument_id = self.instrument_map.get(pair)
            if not instrument_id:
                logger.warning(f'TradeLocker: no instrument mapped for {pair}, skipping')
                continue

            side = 'buy' if direction == 'bullish' else 'sell'

            try:
                order_id = await self._place_market_order(
                    instrument_id=instrument_id,
                    side=side,
                    qty=DEFAULT_LOT_SIZE,
                )

                if order_id:
                    # Calculate when to close
                    matures_at = datetime.fromisoformat(pred['matures_at'])
                    close_at = matures_at  # matures_at already accounts for the 4H hold

                    self.open_positions[pred_id] = {
                        'order_id': order_id,
                        'position_id': None,  # filled async
                        'pair': pair,
                        'side': side,
                        'close_at': close_at,
                        'instrument_id': instrument_id,
                    }

                    logger.info(
                        f'TradeLocker: {side.upper()} {DEFAULT_LOT_SIZE} lots {pair} '
                        f'(meta={pred["meta_proba"]:.2f}, q50={pred["q50"]:.6f}) '
                        f'order_id={order_id}, close_at={close_at.isoformat()}'
                    )

            except Exception as e:
                logger.error(f'TradeLocker: failed to place order for {pair}: {e}', exc_info=True)

    async def _place_market_order(self, instrument_id, side, qty):
        """Place a market order and return the orderId."""
        body = {
            'qty': qty,
            'side': side,
            'tradableInstrumentId': instrument_id,
            'type': 'market',
            'routeId': self.route_id,
            'validity': 'IOC',  # Immediate or Cancel for market orders
        }

        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.post(
                f'{BASE_URL}/trade/accounts/{self.account_id}/orders',
                headers=self._headers(),
                json=body,
            )
            resp.raise_for_status()
            data = resp.json()

        # Response may be nested: {'s': 'ok', 'd': {'orderId': ...}}
        if isinstance(data, dict) and 'd' in data:
            inner = data['d']
        else:
            inner = data
        if isinstance(inner, dict):
            order_id = inner.get('orderId') or inner.get('id')
        else:
            order_id = inner  # might be just the ID directly
        logger.info(f'TradeLocker: order response: {data}')
        return order_id

    async def close_matured_positions(self):
        """Check open positions and close any that have reached maturity."""
        if not self._enabled or not self.open_positions:
            return

        await self._refresh_auth()
        now = datetime.now(timezone.utc)

        # First, resolve any order_ids to position_ids
        await self._resolve_position_ids()

        to_close = []
        for pred_id, pos in self.open_positions.items():
            if now >= pos['close_at']:
                to_close.append(pred_id)

        for pred_id in to_close:
            pos = self.open_positions[pred_id]
            position_id = pos.get('position_id')

            if not position_id:
                logger.warning(
                    f'TradeLocker: no positionId for pred {pred_id} ({pos["pair"]}), '
                    f'trying to find it...'
                )
                position_id = await self._find_position_by_instrument(pos['instrument_id'])

            if position_id:
                try:
                    await self._close_position(position_id)
                    logger.info(
                        f'TradeLocker: closed {pos["pair"]} position {position_id} '
                        f'(pred_id={pred_id})'
                    )
                except Exception as e:
                    logger.error(
                        f'TradeLocker: failed to close {pos["pair"]} '
                        f'position {position_id}: {e}'
                    )
                    continue
            else:
                logger.warning(
                    f'TradeLocker: could not find position for {pos["pair"]} '
                    f'(pred_id={pred_id}), may already be closed'
                )

            del self.open_positions[pred_id]

    async def _resolve_position_ids(self):
        """For orders without a positionId yet, check ordersHistory to find it."""
        unresolved = [
            (pid, pos) for pid, pos in self.open_positions.items()
            if pos.get('position_id') is None and pos.get('order_id')
        ]

        if not unresolved:
            return

        try:
            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.get(
                    f'{BASE_URL}/trade/accounts/{self.account_id}/ordersHistory',
                    headers=self._headers(),
                )
                resp.raise_for_status()
                data = resp.json()

            # Unwrap nested format
            if isinstance(data, dict) and 'd' in data:
                inner = data['d']
            else:
                inner = data
            orders = inner if isinstance(inner, list) else inner.get('orders', []) if isinstance(inner, dict) else []

            # Build order_id -> position info lookup
            order_lookup = {}
            for order in orders:
                if isinstance(order, list) and len(order) > ORD_POSITION_ID:
                    oid = str(order[ORD_ID])
                    posid = order[ORD_POSITION_ID]
                    if oid and posid:
                        order_lookup[oid] = str(posid)
                elif isinstance(order, dict):
                    oid = order.get('orderId') or order.get('id')
                    posid = order.get('positionId')
                    if oid and posid:
                        order_lookup[str(oid)] = str(posid)

            for pred_id, pos in unresolved:
                oid = str(pos['order_id'])
                if oid in order_lookup:
                    pos['position_id'] = order_lookup[oid]
                    logger.info(
                        f'TradeLocker: resolved order {oid} -> position {pos["position_id"]} '
                        f'for {pos["pair"]}'
                    )

        except Exception as e:
            logger.warning(f'TradeLocker: failed to resolve position IDs: {e}')

    async def _find_position_by_instrument(self, instrument_id):
        """Fallback: find an open position by instrument ID."""
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.get(
                    f'{BASE_URL}/trade/accounts/{self.account_id}/positions',
                    headers=self._headers(),
                )
                resp.raise_for_status()
                data = resp.json()

            # Unwrap nested format
            if isinstance(data, dict) and 'd' in data:
                inner = data['d']
            else:
                inner = data
            positions = inner if isinstance(inner, list) else inner.get('positions', []) if isinstance(inner, dict) else []
            for pos in positions:
                if isinstance(pos, list):
                    if str(pos[POS_INSTRUMENT_ID]) == str(instrument_id):
                        return str(pos[POS_ID])
                elif isinstance(pos, dict):
                    if str(pos.get('tradableInstrumentId')) == str(instrument_id):
                        return str(pos.get('id') or pos.get('positionId'))

        except Exception as e:
            logger.warning(f'TradeLocker: failed to find position by instrument: {e}')

        return None

    async def _close_position(self, position_id):
        """Close a position fully."""
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.request(
                'DELETE',
                f'{BASE_URL}/trade/positions/{position_id}',
                headers=self._headers(),
                json={'qty': 0},  # 0 = full close
            )
            resp.raise_for_status()

    async def get_account_state(self):
        """Fetch current account state (balance, equity, P&L)."""
        if not self._enabled:
            return None

        await self._refresh_auth()

        try:
            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.get(
                    f'{BASE_URL}/trade/accounts/{self.account_id}/state',
                    headers=self._headers(),
                )
                resp.raise_for_status()
                return resp.json()
        except Exception as e:
            logger.warning(f'TradeLocker: failed to get account state: {e}')
            return None

    async def get_open_positions_list(self):
        """Fetch all currently open positions from TradeLocker."""
        if not self._enabled:
            return []

        await self._refresh_auth()

        try:
            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.get(
                    f'{BASE_URL}/trade/accounts/{self.account_id}/positions',
                    headers=self._headers(),
                )
                resp.raise_for_status()
                data = resp.json()

            logger.info(f'TradeLocker: positions response keys: {list(data.keys()) if isinstance(data, dict) else type(data)}')
            # Unwrap nested format
            if isinstance(data, dict) and 'd' in data:
                inner = data['d']
            else:
                inner = data
            if isinstance(inner, dict):
                positions = inner.get('positions', [])
            elif isinstance(inner, list):
                positions = inner
            else:
                positions = []
            if positions:
                logger.info(f'TradeLocker: {len(positions)} open position(s), sample: {positions[0]}')
            return positions
        except Exception as e:
            logger.warning(f'TradeLocker: failed to get positions: {e}')
            return []


# Singleton
_bot = None

def get_bot() -> TradeLockerBot:
    global _bot
    if _bot is None:
        _bot = TradeLockerBot()
    return _bot
