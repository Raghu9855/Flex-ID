from typing import List
from fastapi import APIRouter, Depends, status
from app.api.dependencies import get_fl_service
from app.services.fl_service import FederatedLearningService
from app.schemas.experiment import (
    ExperimentCreate,
    ExperimentResponse,
    ExperimentListResponse,
    MetricRecord,
)

router = APIRouter(prefix="/experiments", tags=["Federated Learning"])

@router.get("", response_model=ExperimentListResponse, summary="List Federated Experiments")
def list_experiments(
    skip: int = 0,
    limit: int = 50,
    service: FederatedLearningService = Depends(get_fl_service),
):
    """
    Retrieves all past and currently executing federated training experiments.
    """
    experiments = service.list_experiments(skip=skip, limit=limit)
    return ExperimentListResponse(
        total_experiments=len(experiments),
        experiments=experiments,
    )

@router.post("", response_model=ExperimentResponse, status_code=status.HTTP_202_ACCEPTED, summary="Trigger Federated Experiment")
def create_experiment(
    payload: ExperimentCreate,
    service: FederatedLearningService = Depends(get_fl_service),
):
    """
    Initiates a new federated learning training run with specified aggregation strategy,
    number of rounds, client partitioning, and adversarial attack configuration.
    """
    return service.create_experiment(payload)

@router.get("/{experiment_id}", response_model=ExperimentResponse, summary="Get Experiment Details")
def get_experiment_details(
    experiment_id: str,
    service: FederatedLearningService = Depends(get_fl_service),
):
    """
    Retrieves execution parameters, current status, and summary metrics for an experiment.
    """
    return service.get_experiment(experiment_id)

@router.get("/{experiment_id}/metrics", response_model=List[MetricRecord], summary="Get Round-Level Metrics")
def get_experiment_metrics(
    experiment_id: str,
    service: FederatedLearningService = Depends(get_fl_service),
):
    """
    Returns the progression of training loss, evaluation accuracy, and F1-score across all rounds.
    """
    return service.get_metrics(experiment_id)
