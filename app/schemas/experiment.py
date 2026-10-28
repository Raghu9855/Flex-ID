from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, Field

class ExperimentCreate(BaseModel):
    name: str = Field(..., min_length=3, max_length=128, description="Experiment name")
    strategy: str = Field(default="fedprox", description="Aggregation strategy: fedavg, fedprox, krum, multikrum, trimmed_mean, median")
    rounds: int = Field(default=10, ge=1, le=100, description="Total training rounds")
    num_clients: int = Field(default=4, ge=2, le=32, description="Number of logical federated clients")
    proximal_mu: float = Field(default=0.1, ge=0.0, description="FedProx proximal regularizer mu")
    is_attack: bool = Field(default=False, description="Whether to include adversarial clients")
    attack_type: Optional[str] = Field(default=None, description="Attack type if is_attack=True: flip, noise, backdoor, byzantine, adaptive")

class MetricRecord(BaseModel):
    round_number: int
    train_loss: Optional[float] = None
    accuracy: Optional[float] = None
    f1_score: Optional[float] = None

class ExperimentResponse(BaseModel):
    id: str
    name: str
    strategy: str
    rounds: int
    num_clients: int
    proximal_mu: float
    is_attack: bool
    attack_type: Optional[str] = None
    status: str
    error_message: Optional[str] = None
    created_at: datetime
    completed_at: Optional[datetime] = None
    metrics: List[MetricRecord] = Field(default_factory=list)

class ExperimentListResponse(BaseModel):
    total_experiments: int
    experiments: List[ExperimentResponse]
