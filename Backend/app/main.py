import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.config import settings
from app.database import init_db
from app.routes.auth import router as auth_router
from app.routes.prediction import router as prediction_router
from app.services.inference_service import MLInferenceService
from app.workers.inference_worker import InferenceWorker

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("app.main")

# Singleton background worker instance
worker_instance = InferenceWorker()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan context managing database startup, ML pre-warming, and background worker threads."""
    logger.info("Starting up Backend Asynchronous ML Inference Pipeline...")

    # 1. Initialize Database Tables
    init_db()
    logger.info("Database tables initialized successfully.")

    # 2. Ensure Storage Directory exists
    settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    # 3. Pre-warm ML Model Singleton instance
    try:
        MLInferenceService.get_instance().load_model()
        logger.info("ML Model pre-warmed and cached in memory.")
    except Exception as e:
        logger.warning(f"Could not pre-warm ML model at startup: {e}. Model will load on first job.")

    # 4. Start Background Inference Worker Thread
    worker_instance.start()

    yield

    # Shutdown logic
    logger.info("Shutting down Backend application...")
    worker_instance.stop(timeout=5.0)
    logger.info("Shutdown complete.")


app = FastAPI(
    title="Plant Leaf Disease Asynchronous ML Inference API",
    description="Production-grade asynchronous job-based ML inference system with multi-user security and state consistency.",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Configure CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API Routers
app.include_router(auth_router)
app.include_router(prediction_router)


@app.get("/health", tags=["Health Check"])
def health_check():
    """Health check endpoint verifying application status."""
    return {"status": "ok", "app": "Plant Leaf Disease Asynchronous Inference Pipeline"}
