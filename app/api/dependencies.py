from fastapi import Depends
from sqlalchemy.orm import Session
from app.db.session import get_db
from app.repositories.model_repository import ModelRepository
from app.repositories.prediction_repository import PredictionRepository
from app.repositories.experiment_repository import ExperimentRepository
from app.services.inference_service import InferenceService
from app.services.fl_service import FederatedLearningService

def get_model_repository(db: Session = Depends(get_db)) -> ModelRepository:
    return ModelRepository(db)

def get_prediction_repository(db: Session = Depends(get_db)) -> PredictionRepository:
    return PredictionRepository(db)

def get_experiment_repository(db: Session = Depends(get_db)) -> ExperimentRepository:
    return ExperimentRepository(db)

def get_inference_service(db: Session = Depends(get_db)) -> InferenceService:
    return InferenceService(db=db)

def get_fl_service(db: Session = Depends(get_db)) -> FederatedLearningService:
    return FederatedLearningService(db=db)
