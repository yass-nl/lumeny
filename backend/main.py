"""
LumenY Backend — FastAPI server.

Two channels:
  1. WebSocket /ws/prices  → real-time price feed from Polygon/Massive
  2. REST     /api/predict → model inference (features + 20 models + calibration)
"""

import asyncio
import logging
import os
import warnings
from contextlib import asynccontextmanager
from datetime import datetime, timezone

warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query
from fastapi.middleware.cors import CORSMiddleware

load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

from data_service import CandleBuffer, PriceFeed, PAIRS
from inference import Predictor
from features import build_feature_row

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

# ── Global state ────────────────────────────────────────────────
candle_buffer = CandleBuffer()
price_feed = PriceFeed(candle_buffer)
predictor: Predictor | None = None
# Cache predictions so we don't recompute on every request
prediction_cache: dict[str, dict] = {}
last_prediction_time: dict[str, datetime] = {}

PAIR_IDS = {pair: i for i, pair in enumerate(PAIRS)}

# ── Lifespan ────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global predictor

    logger.info('Loading models...')
    predictor = Predictor()
    logger.info(f'Loaded {len(predictor.models)} quantile models (v2, no calibrators)')

    logger.info('Initializing candle buffers...')
    await candle_buffer.initialize()

    # Run initial predictions for all pairs
    logger.info('Running initial predictions...')
    await _run_predictions_all()

    # Start WebSocket price feed in background
    ws_task = asyncio.create_task(price_feed.connect())
    logger.info('Price feed started.')

    yield

    price_feed.stop()
    ws_task.cancel()


app = FastAPI(title='LumenY API', version='1.0.0', lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)


# ── Prediction logic ────────────────────────────────────────────

def _derive_regime(features_df) -> dict:
    """Derive market regime from feature values."""
    last = features_df.iloc[-1]

    # ADX-based trend strength
    adx = last.get('adx_1H', 20)
    if adx > 40:
        regime = 'strong_trend'
    elif adx > 25:
        regime = 'trending'
    elif adx > 15:
        regime = 'ranging'
    else:
        regime = 'flat'

    # Volatility from ATR ratio and BB width
    atr_ratio = last.get('atr_ratio_1H', 1.0)
    bb_width = last.get('bb_width_1H', 0.01)

    if atr_ratio > 1.3:
        volatility = 'expanding'
    elif atr_ratio < 0.7:
        volatility = 'compressing'
    elif bb_width > 0.015:
        volatility = 'high'
    elif bb_width < 0.005:
        volatility = 'low'
    else:
        volatility = 'moderate'

    return {'regime': regime, 'volatility': volatility}


async def _run_predictions_all():
    """Run predictions for all pairs and cache results."""
    closes_1h = candle_buffer.get_all_1h_closes()

    for pair in PAIRS:
        try:
            ohlcv_raw = candle_buffer.get_ohlcv(pair)
            if not ohlcv_raw or '1H' not in ohlcv_raw:
                logger.warning(f'No 1H data for {pair}, skipping prediction.')
                continue

            # Drop the last candle from each timeframe — it may be incomplete.
            # After WebSocket starts, the buffer can contain partially-formed bars
            # from live minute aggregates.  Trimming ensures only fully-closed bars
            # are used for feature computation.
            ohlcv = {}
            for tf, df in ohlcv_raw.items():
                ohlcv[tf] = df.iloc[:-1] if len(df) > 1 else df

            features_df = build_feature_row(
                ohlcv, closes_1h, pair, PAIR_IDS[pair],
                expected_cols=predictor.feature_cols,
            )

            if features_df.empty:
                continue

            result = predictor.predict(features_df, pair)

            # Add price info
            price_info = candle_buffer.get_latest_price(pair)
            if price_info:
                result['price'] = price_info['price']
                result['price_timestamp'] = price_info['timestamp']

            # Add regime
            regime_info = _derive_regime(features_df)
            result['regime'] = regime_info['regime']
            result['volatility'] = regime_info['volatility']

            # Model metadata
            result['calibration'] = {
                'status': 'reliable',
                'method': 'raw_quantile_v2',
                'training_period': '2009-2024-06',
                'test_period': '2024-07 to 2025-12',
                'data_points': 685111,
                'data_years': 16,
            }

            prediction_cache[pair] = result
            last_prediction_time[pair] = datetime.now(timezone.utc)

        except Exception as e:
            logger.error(f'Prediction error for {pair}: {e}', exc_info=True)


# ── REST Endpoints ──────────────────────────────────────────────

@app.get('/api/predict/{pair}')
async def get_prediction(pair: str):
    """Get the latest prediction for a specific pair."""
    pair = pair.upper()
    if pair not in PAIRS:
        return {'error': f'Unknown pair: {pair}. Available: {PAIRS}'}

    if pair not in prediction_cache:
        return {'error': f'No prediction available yet for {pair}'}

    result = prediction_cache[pair]
    result['computed_at'] = last_prediction_time.get(pair, '').isoformat() if pair in last_prediction_time else None
    return result


@app.get('/api/predict')
async def get_all_predictions():
    """Get predictions for all pairs."""
    return {
        'pairs': prediction_cache,
        'computed_at': {
            pair: ts.isoformat() for pair, ts in last_prediction_time.items()
        },
    }


@app.post('/api/predict/refresh')
async def refresh_predictions():
    """Force recompute predictions for all pairs."""
    await _run_predictions_all()
    return {'status': 'ok', 'pairs_updated': list(prediction_cache.keys())}


@app.get('/api/prices/{pair}')
async def get_price(pair: str):
    """Get latest price for a pair."""
    pair = pair.upper()
    if pair not in PAIRS:
        return {'error': f'Unknown pair: {pair}'}
    price = candle_buffer.get_latest_price(pair)
    if not price:
        return {'error': f'No price data for {pair}'}
    return price


@app.get('/api/candles/{pair}')
async def get_candles(
    pair: str,
    timeframe: str = Query(default='1H', description='Timeframe: 5m, 15m, 1H, 4H, 1D'),
    limit: int = Query(default=200, description='Number of candles to return'),
):
    """Get historical candles for charting."""
    pair = pair.upper()
    if pair not in PAIRS:
        return {'error': f'Unknown pair: {pair}'}

    ohlcv = candle_buffer.get_ohlcv(pair)
    if timeframe not in ohlcv:
        return {'error': f'No {timeframe} data for {pair}'}

    df = ohlcv[timeframe].tail(limit)
    candles = []
    for ts, row in df.iterrows():
        candles.append({
            'time': int(ts.timestamp()),
            'open': round(float(row['open']), 5),
            'high': round(float(row['high']), 5),
            'low': round(float(row['low']), 5),
            'close': round(float(row['close']), 5),
            'volume': int(row['volume']),
        })
    return {'pair': pair, 'timeframe': timeframe, 'candles': candles}


@app.get('/api/health')
async def health():
    return {
        'status': 'ok',
        'models_loaded': predictor is not None,
        'pairs_with_data': [p for p in PAIRS if p in candle_buffer.buffers],
        'pairs_with_predictions': list(prediction_cache.keys()),
        'price_feed_clients': len(price_feed.clients),
    }


# ── WebSocket — Real-time price relay ───────────────────────────

@app.websocket('/ws/prices')
async def websocket_prices(websocket: WebSocket):
    """
    WebSocket endpoint for real-time price updates.
    Frontend connects here and receives candle updates as they arrive from Polygon.
    """
    await websocket.accept()
    price_feed.clients.add(websocket)
    logger.info(f'Client connected to price feed. Total: {len(price_feed.clients)}')

    try:
        # Send current prices immediately on connect
        for pair in PAIRS:
            price = candle_buffer.get_latest_price(pair)
            if price:
                await websocket.send_json({'type': 'price', 'data': price})

        # Keep connection alive and handle client messages
        while True:
            try:
                msg = await websocket.receive_text()
                # Client can request specific pair updates
                data = __import__('json').loads(msg)
                if data.get('action') == 'get_price':
                    pair = data.get('pair', '').upper()
                    price = candle_buffer.get_latest_price(pair)
                    if price:
                        await websocket.send_json({'type': 'price', 'data': price})
            except WebSocketDisconnect:
                break
    finally:
        price_feed.clients.discard(websocket)
        logger.info(f'Client disconnected. Total: {len(price_feed.clients)}')


if __name__ == '__main__':
    import uvicorn
    uvicorn.run('main:app', host='0.0.0.0', port=8000, reload=False)
