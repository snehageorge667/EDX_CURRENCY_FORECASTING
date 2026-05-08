import logging

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger

from app.services.cache_manager import CacheManager

logger = logging.getLogger(__name__)


class ForexRefreshScheduler:
    def __init__(self, cache_manager: CacheManager) -> None:
        self.cache_manager = cache_manager
        self.scheduler = BackgroundScheduler(timezone="UTC")

    def start(self) -> None:
        logger.info("Starting scheduler with 6 hour refresh interval")
        self.scheduler.add_job(
            self.refresh_default_pairs,
            trigger=IntervalTrigger(hours=6),
            id="forex_cache_refresh",
            replace_existing=True,
            max_instances=1,
        )
        self.scheduler.start()

    def shutdown(self) -> None:
        if self.scheduler.running:
            logger.info("Stopping scheduler")
            self.scheduler.shutdown(wait=False)

    def refresh_default_pairs(self) -> None:
        logger.info("Scheduler refresh started")
        try:
            for base_currency, target_currency in self.cache_manager.known_pairs():
                self.cache_manager.refresh_pair(base_currency, target_currency, force=False)
            logger.info("Scheduler refresh completed")
        except Exception:
            logger.exception("Scheduler refresh failed")
