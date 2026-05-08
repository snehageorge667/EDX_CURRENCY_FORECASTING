from datetime import date
import logging

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from pydantic import BaseModel, Field

from app.services.predictor import ForexPredictor, PredictionError

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Predictions"])


class PredictionResponse(BaseModel):
    pair: str = Field(..., examples=["USD_INR"])
    target_date: date = Field(..., examples=["2026-05-10"])
    predictions: dict[str, float] = Field(
        ...,
        examples=[{"xgboost": 93.45, "sarimax": 92.88}],
    )


def get_predictor(request: Request) -> ForexPredictor:
    predictor = getattr(request.app.state, "predictor", None)
    if predictor is None:
        logger.error("Prediction service requested before startup completed")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Prediction service is not ready. Check model files and startup logs.",
        )
    return predictor


@router.post(
    "/predictions",
    response_model=PredictionResponse,
    summary="Forecast a forex rate with XGBoost and SARIMAX",
)
async def create_prediction(
    base_currency: str = Query(
        ...,
        description="Base currency code, for example USD.",
        examples=["USD"],
    ),
    target_currency: str = Query(
        ...,
        description="Target currency code, for example INR.",
        examples=["INR"],
    ),
    target_date: date = Query(
        ...,
        description="Forecast date in YYYY-MM-DD format. Example: 2026-05-10.",
        examples=["2026-05-10"],
    ),
    predictor: ForexPredictor = Depends(get_predictor),
) -> PredictionResponse:
    logger.info(
        "API request received: base_currency=%s target_currency=%s target_date=%s",
        base_currency,
        target_currency,
        target_date,
    )
    try:
        return predictor.predict(
            base_currency=base_currency,
            target_currency=target_currency,
            target_date=target_date,
        )
    except PredictionError as exc:
        logger.exception("Prediction failed: %s", exc)
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unexpected API failure")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unexpected prediction failure. Check logs for details.",
        ) from exc
