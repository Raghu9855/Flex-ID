from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from app.api.dependencies import get_inference_service, get_prediction_repository
from app.services.inference_service import InferenceService
from app.repositories.prediction_repository import PredictionRepository
from app.schemas.prediction import (
    TrafficFeaturesInput,
    PredictionResponse,
    BatchTrafficInput,
    BatchPredictionResponse,
)

router = APIRouter(prefix="/predictions", tags=["Intrusion Detection"])

@router.post(
    "",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Predict Network Traffic Intrusion",
)
def predict_traffic(
    payload: TrafficFeaturesInput,
    service: InferenceService = Depends(get_inference_service),
):
    """
    Accepts 29 network traffic flow metrics and evaluates whether the packet stream
    represents Benign traffic or an active Intrusion (DoS, BruteForce, Botnet, Web Attack).
    """
    return service.predict_single(payload)

@router.post(
    "/batch",
    response_model=BatchPredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Batch Predict Network Traffic",
)
def predict_traffic_batch(
    payload: BatchTrafficInput,
    service: InferenceService = Depends(get_inference_service),
):
    """
    Processes an array of network flow records concurrently with bounded latency and resource usage.
    """
    return service.predict_batch(payload)

@router.get(
    "/{prediction_id}",
    summary="Retrieve Audited Prediction Log",
)
def get_prediction_log(
    prediction_id: str,
    repo: PredictionRepository = Depends(get_prediction_repository),
):
    """
    Retrieves stored audit log and prediction outcome by prediction ID.
    """
    log = repo.get_by_id(prediction_id)
    if not log:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Prediction record '{prediction_id}' was not found in audit store.",
        )

    return {
        "id": log.id,
        "model_id": log.model_id,
        "predicted_class": log.predicted_class,
        "predicted_class_id": log.predicted_class_id,
        "confidence": log.confidence,
        "latency_ms": log.latency_ms,
        "created_at": log.created_at,
        "input_summary": log.input_summary,
    }
