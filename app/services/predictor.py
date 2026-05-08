from datetime import date, datetime, timezone
from pathlib import Path
import logging
import pickle
import warnings
from typing import Any

import joblib
import numpy as np
import pandas as pd

from app.services.cache_manager import CacheManager
from app.services.feature_engineering import FeatureEngineer

logger = logging.getLogger(__name__)


class PredictionError(Exception):
    def __init__(self, message: str, status_code: int = 400) -> None:
        super().__init__(message)
        self.status_code = status_code


class ForexPredictor:
    def __init__(self, cache_manager: CacheManager) -> None:
        self.cache_manager = cache_manager
        self.feature_engineer = FeatureEngineer()
        self.models_dir = Path(__file__).resolve().parents[1] / "models"
        self.xgboost_model: Any | None = None
        self.sarimax_model: Any | None = None
        self.xgboost_features = self.feature_engineer.XGBOOST_FEATURES
        self.sarimax_features = self.feature_engineer.SARIMAX_FEATURES
        self.sarimax_scaler: Any | None = None

    def load_models(self) -> None:
        logger.info("Loading models from %s", self.models_dir)
        xgb_path = self.models_dir / "xgboost_model.pkl"
        sarimax_path = self.models_dir / "sarimax_model.pkl"
        missing = [str(path) for path in [xgb_path, sarimax_path] if not path.exists()]
        if missing:
            logger.error("Missing model files: %s", missing)
            raise RuntimeError(
                "Missing model files. Upload app/models/xgboost_model.pkl and "
                "app/models/sarimax_model.pkl before starting the API."
            )

        xgboost_payload = self._load_pickle(xgb_path)
        sarimax_payload = self._load_pickle(sarimax_path)

        self.xgboost_model, self.xgboost_features = self._unwrap_model_payload(
            payload=xgboost_payload,
            model_name="XGBoost",
            expected_features=self.feature_engineer.XGBOOST_FEATURES,
        )
        self.sarimax_model, self.sarimax_features = self._unwrap_model_payload(
            payload=sarimax_payload,
            model_name="SARIMAX",
            expected_features=self.feature_engineer.SARIMAX_FEATURES,
        )
        if isinstance(sarimax_payload, dict):
            self.sarimax_scaler = sarimax_payload.get("scaler")
        self._validate_xgboost_feature_contract()
        logger.info("Model loading completed successfully")

    def predict(
        self,
        base_currency: str,
        target_currency: str,
        target_date: date,
    ) -> dict[str, Any]:
        if self.xgboost_model is None or self.sarimax_model is None:
            raise PredictionError("Models are not loaded", status_code=503)

        today = datetime.now(timezone.utc).date()
        if target_date < today:
            raise PredictionError("target_date must be today or a future date in YYYY-MM-DD format")

        base = base_currency.strip().upper()
        target = target_currency.strip().upper()
        pair = f"{base}_{target}"

        try:
            frame = self.cache_manager.get_pair_data(base, target)
            if frame.empty:
                raise PredictionError("No cached forex data available for this pair", status_code=422)

            days_ahead = max((target_date - pd.Timestamp(frame.index.max()).date()).days, 1)
            logger.info("Predicting %s for %s days ahead", pair, days_ahead)

            xgb_value = self._forecast_xgboost(frame.copy(), days_ahead)
            sarimax_value = self._forecast_sarimax(frame.copy(), days_ahead)
        except PredictionError:
            raise
        except ValueError as exc:
            raise PredictionError(str(exc), status_code=422) from exc
        except Exception as exc:
            logger.exception("Model prediction failure for %s", pair)
            raise PredictionError(f"Prediction failed for {pair}: {exc}", status_code=500) from exc

        response = {
            "pair": pair,
            "target_date": target_date.isoformat(),
            "predictions": {
                "xgboost": round(float(xgb_value), 4),
                "sarimax": round(float(sarimax_value), 4),
            },
        }
        logger.info("Prediction completed: %s", response)
        return response

    def _forecast_xgboost(self, frame: pd.DataFrame, days_ahead: int) -> float:
        steps = days_ahead
        current = frame.copy()
        prediction = float(current["Close"].iloc[-1])
        for _ in range(steps):
            previous_close = float(current["Close"].iloc[-1])
            matrix = self.feature_engineer.xgboost_matrix(current)[self.xgboost_features].tail(1)
            predicted_log_return = self._predict_one(self.xgboost_model, matrix)
            prediction = self._close_from_log_return(previous_close, predicted_log_return)
            if not np.isfinite(prediction) or prediction <= 0:
                raise PredictionError("XGBoost returned an invalid forecast", status_code=500)
            current = self.feature_engineer.append_forecast_row(current, prediction)
        return prediction

    def _forecast_sarimax(self, frame: pd.DataFrame, days_ahead: int) -> float:
        steps = days_ahead
        current = frame.copy()
        sarimax_state = self.sarimax_model
        prediction = float(current["Close"].iloc[-1])
        for _ in range(steps):
            previous_close = float(current["Close"].iloc[-1])
            exog = self.feature_engineer.sarimax_matrix(current)[self.sarimax_features].tail(1)
            scaled_exog = self._scale_sarimax_exog(exog)
            predicted_log_return = self._sarimax_forecast_one(sarimax_state, scaled_exog)
            prediction = self._close_from_log_return(previous_close, predicted_log_return)
            if not np.isfinite(prediction) or prediction <= 0:
                raise PredictionError("SARIMAX returned an invalid forecast", status_code=500)
            sarimax_state = self._append_sarimax_state(sarimax_state, predicted_log_return, scaled_exog)
            current = self.feature_engineer.append_forecast_row(current, prediction)
        return prediction

    def _sarimax_forecast_one(self, model: Any, exog: pd.DataFrame) -> float:
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=FutureWarning, module="statsmodels")
                warnings.filterwarnings("ignore", message="No supported index is available.*")
                if hasattr(model, "forecast"):
                    result = model.forecast(steps=1, exog=exog)
                elif hasattr(model, "get_forecast"):
                    result = model.get_forecast(steps=1, exog=exog).predicted_mean
                else:
                    raise AttributeError("SARIMAX model has no forecast method")
            return self._first_numeric(result)
        except Exception as exc:
            logger.exception("SARIMAX forecast failed")
            raise PredictionError(f"SARIMAX forecast failed: {exc}", status_code=500) from exc

    def _append_sarimax_state(self, model: Any, predicted_log_return: float, exog: pd.DataFrame) -> Any:
        if not hasattr(model, "append"):
            return model
        if not self._can_append_sarimax_state(model):
            logger.debug("Skipping SARIMAX state append because saved model index is not append-compatible")
            return model
        try:
            next_index = self._next_sarimax_index(model, exog)
            endog_name = getattr(getattr(model, "model", None).data.orig_endog, "name", None)
            endog = pd.Series([predicted_log_return], index=next_index, name=endog_name)
            exog_step = exog.copy()
            exog_step.index = next_index
            return model.append(endog=endog, exog=exog_step, refit=False)
        except Exception:
            logger.warning("Failed to append SARIMAX forecast state; continuing with original model state")
            return model

    @staticmethod
    def _can_append_sarimax_state(model: Any) -> bool:
        model_index = getattr(getattr(model, "model", None), "_index", None)
        orig_exog = getattr(getattr(model, "model", None).data, "orig_exog", None)
        orig_exog_index = getattr(orig_exog, "index", None)
        if isinstance(model_index, pd.RangeIndex) and isinstance(orig_exog_index, pd.DatetimeIndex):
            return False
        return True

    @staticmethod
    def _next_sarimax_index(model: Any, exog: pd.DataFrame) -> pd.Index:
        model_index = getattr(getattr(model, "model", None), "_index", None)
        if isinstance(model_index, pd.RangeIndex):
            return pd.RangeIndex(start=model.nobs, stop=model.nobs + 1, step=1)
        if isinstance(model_index, pd.DatetimeIndex):
            freq = model_index.freq or pd.infer_freq(model_index)
            offset = pd.tseries.frequencies.to_offset(freq or "D")
            return pd.DatetimeIndex([model_index[-1] + offset])
        return pd.Index([pd.Timestamp(exog.index[-1]) + pd.Timedelta(days=1)])

    def _scale_sarimax_exog(self, exog: pd.DataFrame) -> pd.DataFrame:
        if self.sarimax_scaler is None:
            return exog
        try:
            scaled = self.sarimax_scaler.transform(exog)
            return pd.DataFrame(scaled, columns=exog.columns, index=exog.index)
        except Exception as exc:
            logger.exception("SARIMAX exogenous scaling failed")
            raise PredictionError(f"SARIMAX exogenous scaling failed: {exc}", status_code=500) from exc

    @staticmethod
    def _close_from_log_return(previous_close: float, predicted_log_return: float) -> float:
        return float(previous_close * np.exp(predicted_log_return))

    def _predict_one(self, model: Any, matrix: pd.DataFrame) -> float:
        try:
            if hasattr(model, "predict"):
                result = model.predict(matrix)
                return self._first_numeric(result)
            raise AttributeError("XGBoost model has no predict method")
        except Exception as exc:
            logger.exception("XGBoost prediction failed")
            raise PredictionError(f"XGBoost prediction failed: {exc}", status_code=500) from exc

    def _validate_xgboost_feature_contract(self) -> None:
        expected = self.xgboost_features
        model_features = getattr(self.xgboost_model, "feature_names_in_", None)
        if model_features is None and hasattr(self.xgboost_model, "get_booster"):
            try:
                model_features = self.xgboost_model.get_booster().feature_names
            except Exception:
                model_features = None
        if model_features is not None:
            model_features = list(model_features)
        if model_features:
            missing = sorted(set(model_features) - set(expected))
            if missing:
                raise RuntimeError(f"XGBoost model expects unknown features: {missing}")
        logger.info("XGBoost feature contract validated")

    def _unwrap_model_payload(
        self,
        payload: Any,
        model_name: str,
        expected_features: list[str],
    ) -> tuple[Any, list[str]]:
        if not isinstance(payload, dict):
            logger.info("%s pickle contains a direct model object", model_name)
            return payload, expected_features

        if "model" not in payload:
            raise RuntimeError(f"{model_name} model payload is missing the 'model' key")

        model = payload["model"]
        payload_features = payload.get("features") or expected_features
        if not isinstance(payload_features, list):
            payload_features = list(payload_features)

        unknown = sorted(set(payload_features) - set(expected_features))
        missing = sorted(set(expected_features) - set(payload_features))
        if unknown:
            raise RuntimeError(f"{model_name} payload contains unknown features: {unknown}")
        if model_name == "SARIMAX" and payload_features != expected_features:
            raise RuntimeError(
                "SARIMAX payload must use only "
                f"{expected_features}; found {payload_features}"
            )
        if model_name == "XGBoost" and missing:
            raise RuntimeError(f"XGBoost payload is missing required features: {missing}")

        logger.info(
            "%s model unwrapped from payload with %s features",
            model_name,
            len(payload_features),
        )
        return model, payload_features

    @staticmethod
    def _first_numeric(result: Any) -> float:
        if isinstance(result, pd.Series):
            return float(result.iloc[0])
        if isinstance(result, pd.DataFrame):
            return float(result.iloc[0, 0])
        array = np.asarray(result).reshape(-1)
        if array.size == 0:
            raise ValueError("Model returned an empty prediction")
        return float(array[0])

    @staticmethod
    def _load_pickle(path: Path) -> Any:
        try:
            return joblib.load(path)
        except Exception:
            with path.open("rb") as file:
                return pickle.load(file)
