# edx_currency_forecasting

Production-ready FastAPI service for forex forecasting with uploaded XGBoost and SARIMAX models.

## Model Files

Upload the trained models before starting the API:

- `app/models/xgboost_model.pkl`
- `app/models/sarimax_model.pkl`

The application loads both models during startup and fails fast with a clear error if either file is missing.

## Run Locally

```bash
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 10000
```

Swagger UI:

```text
http://localhost:10000/docs
```

## Prediction API

Only one user prediction endpoint is exposed:

```http
POST /predictions
```

Swagger shows three form fields:

- `base_currency`
- `target_currency`
- `target_date` in `YYYY-MM-DD` format, for example `2026-05-10`

Example response:

```json
{
  "pair": "USD_INR",
  "target_date": "2026-05-10",
  "predictions": {
    "xgboost": 93.45,
    "sarimax": 92.88
  }
}
```

## Data And Features

The service downloads latest daily OHLC forex data from Yahoo Finance with `yfinance`.

Yahoo pair conversion:

- `USD` + `INR` becomes `INR=X`
- Other pairs use `BASETARGET=X`, for example `EURINR=X`

XGBoost receives the full engineered feature set, including lag, rolling, return, RSI, candle, and required macro placeholder columns.

SARIMAX receives only:

- `hl_spread_pct`
- `close_position`
- `upper_shadow_pct`
- `lower_shadow_pct`
- `rsi_14`

## Cache And Scheduler

The app uses:

- In-memory cache
- Persistent file cache in `app/cache`
- 6-hour cache duration
- APScheduler refresh every 6 hours

At startup and during requests, valid cache is reused. Expired cache or stale day data triggers refresh.

## Render Deployment

Runtime:

```text
python-3.11.9
```

Start command:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 10000
```

Make sure the model files are present in `app/models/` in the deployed service.
