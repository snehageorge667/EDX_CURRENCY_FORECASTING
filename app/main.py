from contextlib import asynccontextmanager
from pathlib import Path
import logging
from logging.handlers import RotatingFileHandler

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.scheduler import ForexRefreshScheduler
from app.services.cache_manager import CacheManager
from app.services.predictor import ForexPredictor

BASE_DIR = Path(__file__).resolve().parent
LOG_DIR = BASE_DIR / "logs"
LOG_FILE = LOG_DIR / "app.log"


def configure_logging() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers.clear()

    file_handler = RotatingFileHandler(
        LOG_FILE,
        maxBytes=5_000_000,
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)


configure_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Application startup initiated")
    cache_manager = CacheManager()
    predictor = ForexPredictor(cache_manager=cache_manager)
    predictor.load_models()

    scheduler = ForexRefreshScheduler(cache_manager=cache_manager)
    scheduler.start()

    app.state.cache_manager = cache_manager
    app.state.predictor = predictor
    app.state.scheduler = scheduler

    logger.info("Application startup completed")
    try:
        yield
    finally:
        logger.info("Application shutdown initiated")
        scheduler.shutdown()
        logger.info("Application shutdown completed")


app = FastAPI(
    title="EDX Currency Forecasting API",
    description="Forecast forex rates using uploaded XGBoost and SARIMAX models.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)
