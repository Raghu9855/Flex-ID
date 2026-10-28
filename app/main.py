from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, RedirectResponse
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.logging import logger
from app.core.exceptions import (
    FlexIDException,
    flex_id_exception_handler,
    global_exception_handler,
)
from app.db.session import init_db, SessionLocal, get_db
from app.db.seeder import seed_database
from app.services.model_manager import model_manager
from app.api.v1.router import api_v1_router
from app.api.v1.health import check_health


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifecycle manager: handles database setup, artifact verification,
    and ML model warmup before accepting HTTP traffic.
    """
    logger.info("Initializing %s v%s ...", settings.PROJECT_NAME, settings.VERSION)
    # 1. Initialize DB schema
    init_db()

    # 2. Seed default models and verified metrics
    db = SessionLocal()
    try:
        seed_database(db)
    finally:
        db.close()

    # 3. Warm up inference engine
    model_manager.warmup()
    logger.info("Application startup sequence complete. Ready for requests.")

    yield

    logger.info("Shutting down %s ...", settings.PROJECT_NAME)


def create_application() -> FastAPI:
    app = FastAPI(
        title=settings.PROJECT_NAME,
        version=settings.VERSION,
        description=(
            "Production-oriented backend and inference API for **FLEX-ID** "
            "(Federated Learning Explainable Intrusion Detection System). "
            "Provides validated multi-class traffic classification, batch inference, "
            "federated experiment tracking, and explainable AI analytics."
        ),
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    # CORS Middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Global Exception Handlers
    app.add_exception_handler(FlexIDException, flex_id_exception_handler)
    app.add_exception_handler(Exception, global_exception_handler)

    # Frontend UI and Static Assets
    frontend_dir = settings.PROJECT_ROOT_PATH / "frontend"
    if frontend_dir.exists():
        css_dir = frontend_dir / "css"
        if css_dir.exists():
            app.mount("/css", StaticFiles(directory=str(css_dir)), name="css")
        js_dir = frontend_dir / "js"
        if js_dir.exists():
            app.mount("/js", StaticFiles(directory=str(js_dir)), name="js")

        @app.get("/", tags=["UI"], include_in_schema=False)
        @app.get("/ui", tags=["UI"], include_in_schema=False)
        async def serve_ui():
            index_path = frontend_dir / "index.html"
            if index_path.exists():
                return FileResponse(str(index_path))
            return RedirectResponse(url="/docs")
    else:
        @app.get("/", tags=["Root"], include_in_schema=False)
        async def root_redirect():
            return RedirectResponse(url="/docs")

    # Root Health Check (Convenient for Docker / K8s probes)
    app.add_api_route("/health", check_health, methods=["GET"], tags=["Health & Status"], include_in_schema=True)

    # Mount API V1
    app.include_router(api_v1_router, prefix=settings.API_V1_PREFIX)

    return app


app = create_application()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
    )
