# LumenY — Machine learning research on FX markets

LumenY is a personal research project (March – May 2026) that forecasts the size and direction of moves on 15 FX pairs from minute-level market data, validates the models without lookahead bias, and runs them live through a FastAPI backend, a paper-trading loop and a trading bot.

It is mostly research and experimentation. The repository keeps the whole path: many model iterations, diagnostics, and ideas that did not work.

## What it does

- **Data:** pulls 1-minute OHLCV bars for 15 FX majors and crosses from the Polygon API and resamples them to 5m, 15m, 1H, 4H and daily.
- **Features:** 300+ features, including realized-volatility estimators (Parkinson, Garman-Klass, Rogers-Satchell, Yang-Zhang), microstructure statistics, cross-timeframe, cross-pair correlation and calendar features.
- **Models:** LightGBM quantile regression (Q10–Q90) predicting move size and direction over 1h to 72h horizons, tuned with pinball-loss cross-validation and calibrated.
- **Validation:** lookahead-bias tests (correlation and shift tests), a walk-forward backtest that rebuilds features only from data available at each timestamp, robustness checks across several out-of-sample periods, and SHAP analysis of feature contributions.
- **Live system:** a FastAPI backend with a WebSocket price feed and live inference, an automated paper-trading loop that logs and resolves every signal, and a bot that sends model signals to a broker API.
- **Dashboard:** a React + TypeScript front end with live charts and model signals.

## Repository structure

| Path | Contents |
| --- | --- |
| `notebooks*/` | Research notebooks and scripts, in chronological order: data pipeline, feature engineering, training, calibration and validation for each model generation |
| `backend/` | Production code: data service and candle buffer, feature computation, inference, paper trading, broker bot, FastAPI app |
| `src/` | React + TypeScript dashboard (Vite) |
| `test_*.py` | Research and validation scripts (lookahead tests, walk-forward, paper-trading replays, model evaluation), not unit tests |
| root `*.py` | Diagnostics and analyses: feature checks, capacity and liquidity analysis, model comparisons, robustness tests |

## Stack

Python, pandas, NumPy, SciPy, scikit-learn, LightGBM, SHAP, Jupyter, FastAPI, WebSockets, SQLite, React, TypeScript, Railway, Vercel.

## Running it

The backend needs a Polygon API key in a `.env` file at the repo root:

```
POLYGON_S3_SECRET_KEY=your_key
```

```bash
# backend
pip install -r backend/requirements.txt
cd backend && uvicorn main:app --reload

# dashboard
npm install
npm run dev
```

## Notes

- Results vary across periods and are not presented as a profitable strategy. The focus of the project is on building and validating the pipeline correctly.
- Nothing in this repository is financial advice.

## Author

Yassin Nouali — [github.com/yass-nl](https://github.com/yass-nl)
