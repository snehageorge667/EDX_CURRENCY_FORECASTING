import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ForexPreprocessor:
    REQUIRED_COLUMNS = ["Open", "High", "Low", "Close"]

    def clean(self, data: pd.DataFrame) -> pd.DataFrame:
        logger.info("Preprocessing forex dataset with %s rows", len(data))
        frame = data.copy()

        missing = [column for column in self.REQUIRED_COLUMNS if column not in frame.columns]
        if missing:
            raise ValueError(f"Dataset is missing required OHLC columns: {missing}")

        frame = frame[self.REQUIRED_COLUMNS].replace([np.inf, -np.inf], np.nan)
        frame = frame.apply(pd.to_numeric, errors="coerce")
        frame = frame.dropna(subset=self.REQUIRED_COLUMNS)
        frame = frame[frame["Close"] > 0]
        frame = frame[frame["High"] >= frame["Low"]]

        if frame.empty:
            raise ValueError("Dataset has no valid OHLC rows after preprocessing")

        logger.info("Preprocessing completed with %s valid rows", len(frame))
        return frame
