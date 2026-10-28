from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, Field

class ModelMetadataResponse(BaseModel):
    id: str
    model_version: str
    strategy: str
    weights_path: str
    input_features_count: int = 29
    num_classes: int = 13
    accuracy: Optional[float] = None
    macro_f1: Optional[float] = None
    weighted_f1: Optional[float] = None
    is_active: bool = False
    created_at: datetime
    class_names: List[str] = Field(default_factory=list)

class ModelListResponse(BaseModel):
    total_models: int
    active_model: str
    models: List[ModelMetadataResponse]
