from typing import List, Dict, Optional, Union
from pydantic import BaseModel, Field
from app.schemas.prediction import TrafficFeaturesInput

class FeatureContribution(BaseModel):
    feature: str = Field(..., description="Feature name")
    importance: float = Field(..., description="Absolute or directional attribution magnitude")
    direction: str = Field(..., description="'positive' (increases attack probability) or 'negative' (favors benign)")

class ExplanationRequest(BaseModel):
    features: Union[Dict[str, float], List[float]] = Field(..., description="Traffic flow feature values")
    model_version: Optional[str] = Field(default=None, description="Model version to explain")
    top_k: int = Field(default=5, ge=1, le=29, description="Number of top features to return")

class ExplanationResponse(BaseModel):
    model_version: str
    predicted_class: str
    is_intrusion: bool
    confidence: float
    method: str = Field(default="Federated-SHAP / Fast Feature Attribution")
    top_contributing_features: List[FeatureContribution]
    global_importance_baseline: Optional[Dict[str, float]] = None
    summary: str = Field(..., description="Human-readable explanation for security analysts")
