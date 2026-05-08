from datetime import datetime, timezone
import logging
import re

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


class ForexDataFetcher:
    VALID_CURRENCY = re.compile(r"^[A-Z]{3}$")

    def download(
        self,
        base_currency: str,
        target_currency: str,
        period: str = "5y",
    ) -> pd.DataFrame:
        base = self._validate_currency(base_currency, "base_currency")
        target = self._validate_currency(target_currency, "target_currency")
        ticker = self.to_yahoo_ticker(base, target)

        logger.info("Downloading forex data from Yahoo Finance: ticker=%s", ticker)
        try:
            data = yf.download(
                ticker,
                period=period,
                interval="1d",
                auto_adjust=False,
                progress=False,
                threads=False,
            )
        except Exception as exc:
            logger.exception("Yahoo Finance download failed for %s", ticker)
            raise ValueError(f"Yahoo Finance download failed for {ticker}: {exc}") from exc

        if data is None or data.empty:
            logger.error("Yahoo Finance returned empty dataset for %s", ticker)
            raise ValueError(f"No Yahoo Finance data found for currency pair {base}/{target}")

        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        data = data.rename(columns=str.title)
        data.index = pd.to_datetime(data.index)
        data = data.sort_index()

        logger.info(
            "Downloaded %s rows for %s through %s",
            len(data),
            ticker,
            datetime.now(timezone.utc).date(),
        )
        return data

    @staticmethod
    def to_yahoo_ticker(base_currency: str, target_currency: str) -> str:
        base = base_currency.upper()
        target = target_currency.upper()
        if base == target:
            raise ValueError("base_currency and target_currency must be different")
        if base == "USD":
            return f"{target}=X"
        return f"{base}{target}=X"

    def _validate_currency(self, currency: str, field_name: str) -> str:
        normalized = currency.strip().upper()
        if not self.VALID_CURRENCY.match(normalized):
            raise ValueError(f"{field_name} must be a 3-letter ISO currency code")
        return normalized
