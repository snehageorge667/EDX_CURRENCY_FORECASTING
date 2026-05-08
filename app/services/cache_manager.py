from datetime import datetime, timedelta, timezone
from pathlib import Path
import logging
import pickle
from typing import Any

import pandas as pd

from app.services.data_fetcher import ForexDataFetcher
from app.services.feature_engineering import FeatureEngineer
from app.services.preprocessor import ForexPreprocessor

logger = logging.getLogger(__name__)


class CacheManager:
    CACHE_TTL = timedelta(hours=6)

    def __init__(self) -> None:
        self.cache_dir = Path(__file__).resolve().parents[1] / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.memory_cache: dict[str, dict[str, Any]] = {}
        self.fetcher = ForexDataFetcher()
        self.preprocessor = ForexPreprocessor()
        self.feature_engineer = FeatureEngineer()
        logger.info("Cache manager initialized at %s", self.cache_dir)

    def get_pair_data(self, base_currency: str, target_currency: str) -> pd.DataFrame:
        pair_key = self._pair_key(base_currency, target_currency)
        cached = self._get_valid_memory_cache(pair_key)
        if cached is not None:
            logger.info("Using in-memory cache for %s", pair_key)
            return cached.copy()

        cached = self._load_valid_file_cache(pair_key)
        if cached is not None:
            logger.info("Using persistent file cache for %s", pair_key)
            self.memory_cache[pair_key] = {
                "created_at": datetime.now(timezone.utc),
                "data": cached.copy(),
            }
            return cached.copy()

        return self.refresh_pair(base_currency, target_currency, force=True)

    def refresh_pair(
        self,
        base_currency: str,
        target_currency: str,
        force: bool = False,
    ) -> pd.DataFrame:
        pair_key = self._pair_key(base_currency, target_currency)
        if not force:
            cached = self._get_valid_memory_cache(pair_key)
            if cached is not None and not self._has_new_day_available(cached):
                logger.info("Skipping refresh for %s because cache is current", pair_key)
                return cached.copy()

        logger.info("Refreshing forex cache for %s", pair_key)
        raw = self.fetcher.download(base_currency, target_currency)
        clean = self.preprocessor.clean(raw)
        engineered = self.feature_engineer.create_all_features(clean)
        if engineered.empty:
            raise ValueError(f"No usable engineered rows for {pair_key}")

        payload = {
            "created_at": datetime.now(timezone.utc),
            "data": engineered,
        }
        self.memory_cache[pair_key] = payload
        self._save_file_cache(pair_key, payload)
        logger.info("Cache refreshed for %s with %s rows", pair_key, len(engineered))
        return engineered.copy()

    def known_pairs(self) -> list[tuple[str, str]]:
        pairs = {("USD", "INR")}
        for pair_key in self.memory_cache:
            pairs.add(self._split_pair_key(pair_key))
        for cache_file in self.cache_dir.glob("*.pkl"):
            pairs.add(self._split_pair_key(cache_file.stem))
        return sorted(pairs)

    def _get_valid_memory_cache(self, pair_key: str) -> pd.DataFrame | None:
        payload = self.memory_cache.get(pair_key)
        if not payload:
            return None
        if self._is_expired(payload["created_at"]):
            logger.info("In-memory cache expired for %s", pair_key)
            return None
        data = payload["data"]
        if self._has_new_day_available(data):
            logger.info("New day data may be available for %s", pair_key)
            return None
        return data

    def _load_valid_file_cache(self, pair_key: str) -> pd.DataFrame | None:
        path = self._cache_path(pair_key)
        if not path.exists():
            logger.info("No persistent cache found for %s", pair_key)
            return None
        try:
            with path.open("rb") as file:
                payload = pickle.load(file)
            created_at = payload["created_at"]
            data = payload["data"]
            if self._is_expired(created_at) or self._has_new_day_available(data):
                logger.info("Persistent cache expired for %s", pair_key)
                return None
            return data
        except Exception:
            logger.exception("Failed to load cache file for %s", pair_key)
            return None

    def _save_file_cache(self, pair_key: str, payload: dict[str, Any]) -> None:
        try:
            with self._cache_path(pair_key).open("wb") as file:
                pickle.dump(payload, file)
            logger.info("Persistent cache saved for %s", pair_key)
        except Exception:
            logger.exception("Failed to save persistent cache for %s", pair_key)

    def _is_expired(self, created_at: datetime) -> bool:
        if created_at.tzinfo is None:
            created_at = created_at.replace(tzinfo=timezone.utc)
        return datetime.now(timezone.utc) - created_at >= self.CACHE_TTL

    def _has_new_day_available(self, data: pd.DataFrame) -> bool:
        if data.empty:
            return True
        latest_date = pd.Timestamp(data.index.max()).date()
        return latest_date < datetime.now(timezone.utc).date()

    def _cache_path(self, pair_key: str) -> Path:
        return self.cache_dir / f"{pair_key}.pkl"

    @staticmethod
    def _pair_key(base_currency: str, target_currency: str) -> str:
        return f"{base_currency.upper()}_{target_currency.upper()}"

    @staticmethod
    def _split_pair_key(pair_key: str) -> tuple[str, str]:
        parts = pair_key.upper().split("_", 1)
        if len(parts) != 2 or not parts[0] or not parts[1]:
            return ("USD", "INR")
        return (parts[0], parts[1])
