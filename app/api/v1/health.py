from fastapi import APIRouter, Depends, status
from sqlalchemy.orm import Session
from sqlalchemy import text
from app.core.config import settings
from app.db.session import get_db
from app.schemas.health import HealthResponse
from app.services.model_manager import model_manager
from app.repositories.model_repository import ModelRepository

router = APIRouter(tags=["Health & Status"])

@router.get("/health", response_model=HealthResponse, summary="Application Health Probe")
def check_health(db: Session = Depends(get_db)):
    """
    Evaluates backend health, database connectivity, and ML model serving readiness.
    """
    db_connected = False
    try:
        db.execute(text("SELECT 1"))
        db_connected = True
    except Exception:
        db_connected = False

    active_model = settings.DEFAULT_MODEL_VERSION
    try:
        model_repo = ModelRepository(db)
        active_rec = model_repo.get_active()
        if active_rec:
            active_model = active_rec.model_version
    except Exception:
        pass

    components = {
        "database": "reachable" if db_connected else "unreachable",
        "model_manager": "ready",
        "classes_loaded": len(model_manager.class_names),
        "scaler_loaded": model_manager.scaler is not None,
        "models_cached": list(model_manager._models_cache.keys()),
    }

    overall_status = "healthy" if db_connected else "degraded"

    return HealthResponse(
        status=overall_status,
        environment=settings.ENVIRONMENT,
        version=settings.VERSION,
        database_connected=db_connected,
        active_model=active_model,
        components=components,
    )
