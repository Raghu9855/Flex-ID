from typing import List
from fastapi import APIRouter, Depends, HTTPException, status
from app.api.dependencies import get_model_repository
from app.repositories.model_repository import ModelRepository
from app.schemas.model import ModelMetadataResponse, ModelListResponse
from app.services.model_manager import model_manager
from app.core.exceptions import ModelNotFoundError

router = APIRouter(prefix="/models", tags=["Models & Serving"])

@router.get("", response_model=ModelListResponse, summary="List Registered Models")
def list_models(repo: ModelRepository = Depends(get_model_repository)):
    """
    Returns all registered federated model checkpoints, including metrics and active status.
    """
    records = repo.list()
    active_record = repo.get_active()
    active_version = active_record.model_version if active_record else "none"

    model_list = [
        ModelMetadataResponse(
            id=m.id,
            model_version=m.model_version,
            strategy=m.strategy,
            weights_path=m.weights_path,
            input_features_count=m.input_features_count,
            num_classes=m.num_classes,
            accuracy=m.accuracy,
            macro_f1=m.macro_f1,
            weighted_f1=m.weighted_f1,
            is_active=m.is_active,
            created_at=m.created_at,
            class_names=model_manager.class_names,
        )
        for m in records
    ]

    return ModelListResponse(
        total_models=len(model_list),
        active_model=active_version,
        models=model_list,
    )

@router.get("/{model_id_or_version}", response_model=ModelMetadataResponse, summary="Get Model Details")
def get_model_details(model_id_or_version: str, repo: ModelRepository = Depends(get_model_repository)):
    """
    Retrieves metadata, metrics, and weights path for a specific model ID or version name.
    """
    model = repo.get_by_version(model_id_or_version)
    if not model:
        model = repo.get_by_id(model_id_or_version)

    if not model:
        raise ModelNotFoundError(f"Model '{model_id_or_version}' not found.")

    return ModelMetadataResponse(
        id=model.id,
        model_version=model.model_version,
        strategy=model.strategy,
        weights_path=model.weights_path,
        input_features_count=model.input_features_count,
        num_classes=model.num_classes,
        accuracy=model.accuracy,
        macro_f1=model.macro_f1,
        weighted_f1=model.weighted_f1,
        is_active=model.is_active,
        created_at=model.created_at,
        class_names=model_manager.class_names,
    )

@router.post("/{model_version}/activate", response_model=ModelMetadataResponse, summary="Activate Model for Inference")
def activate_model(model_version: str, repo: ModelRepository = Depends(get_model_repository)):
    """
    Sets the designated model version as the default active model for live traffic inference.
    """
    model = repo.set_active(model_version)
    if not model:
        raise ModelNotFoundError(f"Model version '{model_version}' not found.")

    # Pre-cache in model_manager
    model_manager.get_model(model.model_version, model.weights_path)

    return ModelMetadataResponse(
        id=model.id,
        model_version=model.model_version,
        strategy=model.strategy,
        weights_path=model.weights_path,
        input_features_count=model.input_features_count,
        num_classes=model.num_classes,
        accuracy=model.accuracy,
        macro_f1=model.macro_f1,
        weighted_f1=model.weighted_f1,
        is_active=model.is_active,
        created_at=model.created_at,
        class_names=model_manager.class_names,
    )
