from fastapi import APIRouter
from app.api.v1.health import router as health_router
from app.api.v1.models import router as models_router
from app.api.v1.predictions import router as predictions_router
from app.api.v1.experiments import router as experiments_router
from app.api.v1.explanations import router as explanations_router

api_v1_router = APIRouter()

api_v1_router.include_router(health_router)
api_v1_router.include_router(models_router)
api_v1_router.include_router(predictions_router)
api_v1_router.include_router(experiments_router)
api_v1_router.include_router(explanations_router)
