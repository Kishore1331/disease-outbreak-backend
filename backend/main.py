import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

from backend.config import settings
from backend.services.model_service import model_service
from backend.services.weather_service import weather_service
from backend.api import (
    routes_health,
    routes_dashboard,
    routes_prediction,
    routes_weather,
    routes_wards,
    routes_metrics,
    routes_auth
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger("outbreakwatch.main")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    logger.info(f"Environment: {settings.ENVIRONMENT}, Debug: {settings.DEBUG}")
    
    # Pre-cache initial weather and ensure data service initialized
    try:
        await weather_service.get_current_weather()
    except Exception as e:
        logger.warning(f"Could not pre-fetch weather: {e}")

    yield
    # Shutdown
    logger.info("OutbreakWatch API shutting down cleanly.")

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Backend API for AI-Powered Disease Outbreak Prediction Platform (HT-13 / SDG 3).",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Sanitized Error Handling (prevents leaking stack traces or internal OS paths)
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    errors = []
    for err in exc.errors():
        loc = " -> ".join([str(l) for l in err.get("loc", [])])
        errors.append(f"{loc}: {err.get('msg')}")
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": "Validation Error",
            "message": "Invalid request parameters provided.",
            "details": errors
        }
    )

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception on {request.method} {request.url.path}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal Server Error",
            "message": "An error occurred while processing the surveillance request.",
            "path": request.url.path
        }
    )

# Include Routers
app.include_router(routes_health.router)
app.include_router(routes_dashboard.router)
app.include_router(routes_prediction.router)
app.include_router(routes_weather.router)
app.include_router(routes_wards.router)
app.include_router(routes_metrics.router)
app.include_router(routes_auth.router)

@app.get("/", tags=["Root"])
async def root():
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "problem_statement": "HT-13 — Disease Outbreak Prediction",
        "sdg": "SDG 3 — Good Health and Well-Being",
        "documentation": "/docs",
        "redoc": "/redoc",
        "health_check": "/api/health",
        "dashboard": "/api/dashboard"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host=settings.HOST, port=settings.PORT, reload=True)
