import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class FeatureEngineer:
    MACRO_FEATURES = [
        "REAL_GDP_PER_CAPITA",
        "TREASURY_YIELD",
        "FEDERAL_FUNDS_RATE",
        "CPI",
        "INFLATION",
        "UNEMPLOYMENT",
    ]
    BASE_TECHNICAL_FEATURES = [
        "cc_log_return",
        "oc_log_return",
        "overnight_gap",
        "hl_spread_pct",
        "close_position",
        "upper_shadow_pct",
        "lower_shadow_pct",
    ]
    LAGS = [1, 2, 3, 5, 7, 10, 14]
    ROLLING_WINDOWS = [3, 5, 7, 14, 21]
    XGBOOST_FEATURES = [
        "REAL_GDP_PER_CAPITA",
        "TREASURY_YIELD",
        "FEDERAL_FUNDS_RATE",
        "CPI",
        "INFLATION",
        "UNEMPLOYMENT",
        "cc_log_return",
        "oc_log_return",
        "overnight_gap",
        "hl_spread_pct",
        "close_position",
        "upper_shadow_pct",
        "lower_shadow_pct",
        "cc_log_return_lag_1",
        "cc_log_return_lag_2",
        "cc_log_return_lag_3",
        "cc_log_return_lag_5",
        "cc_log_return_lag_7",
        "cc_log_return_lag_10",
        "cc_log_return_lag_14",
        "oc_log_return_lag_1",
        "oc_log_return_lag_2",
        "oc_log_return_lag_3",
        "oc_log_return_lag_5",
        "oc_log_return_lag_7",
        "oc_log_return_lag_10",
        "oc_log_return_lag_14",
        "overnight_gap_lag_1",
        "overnight_gap_lag_2",
        "overnight_gap_lag_3",
        "overnight_gap_lag_5",
        "overnight_gap_lag_7",
        "overnight_gap_lag_10",
        "overnight_gap_lag_14",
        "hl_spread_pct_lag_1",
        "hl_spread_pct_lag_2",
        "hl_spread_pct_lag_3",
        "hl_spread_pct_lag_5",
        "hl_spread_pct_lag_7",
        "hl_spread_pct_lag_10",
        "hl_spread_pct_lag_14",
        "close_position_lag_1",
        "close_position_lag_2",
        "close_position_lag_3",
        "close_position_lag_5",
        "close_position_lag_7",
        "close_position_lag_10",
        "close_position_lag_14",
        "upper_shadow_pct_lag_1",
        "upper_shadow_pct_lag_2",
        "upper_shadow_pct_lag_3",
        "upper_shadow_pct_lag_5",
        "upper_shadow_pct_lag_7",
        "upper_shadow_pct_lag_10",
        "upper_shadow_pct_lag_14",
        "lower_shadow_pct_lag_1",
        "lower_shadow_pct_lag_2",
        "lower_shadow_pct_lag_3",
        "lower_shadow_pct_lag_5",
        "lower_shadow_pct_lag_7",
        "lower_shadow_pct_lag_10",
        "lower_shadow_pct_lag_14",
        "return_mean_3",
        "return_std_3",
        "return_mean_5",
        "return_std_5",
        "return_mean_7",
        "return_std_7",
        "return_mean_14",
        "return_std_14",
        "return_mean_21",
        "return_std_21",
        "return_accel_1",
        "return_accel_3",
        "return_accel_5",
        "rsi_14",
    ]
    SARIMAX_FEATURES = [
        "hl_spread_pct",
        "close_position",
        "upper_shadow_pct",
        "lower_shadow_pct",
        "rsi_14",
    ]

    def create_all_features(self, data: pd.DataFrame) -> pd.DataFrame:
        logger.info("Starting full XGBoost feature engineering")
        frame = data.copy().sort_index()
        frame = self._add_macro_placeholders(frame)
        frame = self._add_technical_features(frame)
        frame = self._add_lag_features(frame)
        frame = self._add_rolling_features(frame)
        frame = self._add_return_acceleration(frame)
        frame = self._finalize(frame)
        logger.info("Feature engineering completed with %s rows", len(frame))
        return frame

    def xgboost_matrix(self, frame: pd.DataFrame) -> pd.DataFrame:
        return self._safe_feature_matrix(frame, self.XGBOOST_FEATURES)

    def sarimax_matrix(self, frame: pd.DataFrame) -> pd.DataFrame:
        return self._safe_feature_matrix(frame, self.SARIMAX_FEATURES)

    def append_forecast_row(self, frame: pd.DataFrame, predicted_close: float) -> pd.DataFrame:
        last = frame.iloc[-1]
        next_date = pd.Timestamp(frame.index.max()) + pd.Timedelta(days=1)
        synthetic = pd.DataFrame(
            {
                "Open": [float(last["Close"])],
                "High": [max(float(last["Close"]), float(predicted_close))],
                "Low": [min(float(last["Close"]), float(predicted_close))],
                "Close": [float(predicted_close)],
            },
            index=[next_date],
        )
        base_ohlc = pd.concat([frame[["Open", "High", "Low", "Close"]], synthetic])
        return self.create_all_features(base_ohlc)

    def _add_macro_placeholders(self, frame: pd.DataFrame) -> pd.DataFrame:
        # Uploaded XGBoost models expect macro columns. Yahoo Finance does not supply
        # those values, so stable neutral placeholders keep the feature contract intact.
        for column in self.MACRO_FEATURES:
            frame[column] = 0.0
        return frame

    def _add_technical_features(self, frame: pd.DataFrame) -> pd.DataFrame:
        previous_close = frame["Close"].shift(1)
        price_range = (frame["High"] - frame["Low"]).replace(0, np.nan)
        frame["cc_log_return"] = np.log(frame["Close"] / previous_close)
        frame["oc_log_return"] = np.log(frame["Close"] / frame["Open"])
        frame["overnight_gap"] = np.log(frame["Open"] / previous_close)
        frame["hl_spread_pct"] = (frame["High"] - frame["Low"]) / frame["Close"]
        frame["close_position"] = (frame["Close"] - frame["Low"]) / price_range
        frame["upper_shadow_pct"] = (frame["High"] - frame[["Open", "Close"]].max(axis=1)) / frame["Close"]
        frame["lower_shadow_pct"] = (frame[["Open", "Close"]].min(axis=1) - frame["Low"]) / frame["Close"]
        frame["rsi_14"] = self._rsi(frame["Close"], window=14)
        return frame

    def _add_lag_features(self, frame: pd.DataFrame) -> pd.DataFrame:
        for feature in self.BASE_TECHNICAL_FEATURES:
            for lag in self.LAGS:
                frame[f"{feature}_lag_{lag}"] = frame[feature].shift(lag)
        return frame

    def _add_rolling_features(self, frame: pd.DataFrame) -> pd.DataFrame:
        for window in self.ROLLING_WINDOWS:
            frame[f"return_mean_{window}"] = frame["cc_log_return"].rolling(window).mean()
            frame[f"return_std_{window}"] = frame["cc_log_return"].rolling(window).std()
        return frame

    def _add_return_acceleration(self, frame: pd.DataFrame) -> pd.DataFrame:
        for lag in [1, 3, 5]:
            frame[f"return_accel_{lag}"] = frame["cc_log_return"] - frame["cc_log_return"].shift(lag)
        return frame

    def _finalize(self, frame: pd.DataFrame) -> pd.DataFrame:
        frame = frame.replace([np.inf, -np.inf], np.nan)
        frame = frame.ffill().bfill()
        frame = frame.dropna(subset=["Close"])
        for feature in self.XGBOOST_FEATURES:
            if feature not in frame.columns:
                frame[feature] = 0.0
        frame[self.XGBOOST_FEATURES] = frame[self.XGBOOST_FEATURES].fillna(0.0)
        return frame

    def _safe_feature_matrix(self, frame: pd.DataFrame, features: list[str]) -> pd.DataFrame:
        missing = [feature for feature in features if feature not in frame.columns]
        if missing:
            raise ValueError(f"Feature matrix is missing required features: {missing}")
        matrix = frame[features].replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0.0)
        if matrix.empty:
            raise ValueError("Feature matrix is empty")
        return matrix.astype(float)

    @staticmethod
    def _rsi(close: pd.Series, window: int) -> pd.Series:
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(window).mean()
        loss = (-delta.clip(upper=0)).rolling(window).mean()
        rs = gain / loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))
