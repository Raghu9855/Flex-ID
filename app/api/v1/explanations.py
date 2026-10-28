from fastapi import APIRouter, status
from app.schemas.explanation import ExplanationRequest, ExplanationResponse
from app.services.explanation_service import explanation_service

router = APIRouter(prefix="/explanations", tags=["Explainable AI (XAI)"])

@router.post(
    "",
    response_model=ExplanationResponse,
    status_code=status.HTTP_200_OK,
    summary="Explain Network Traffic Prediction",
)
def explain_prediction(payload: ExplanationRequest):
    """
    Computes local feature attribution and retrieves federated SHAP global importance
    to explain why a particular network packet flow was classified as benign or malicious.
    """
    return explanation_service.explain(payload)
