from typing import List, Optional
from datetime import datetime, timezone
from sqlalchemy.orm import Session
from sqlalchemy import select, desc
from app.models.experiment import Experiment
from app.models.experiment_metric import ExperimentMetric

class ExperimentRepository:
    def __init__(self, db: Session):
        self.db = db

    def get_by_id(self, experiment_id: str) -> Optional[Experiment]:
        return self.db.scalars(
            select(Experiment).where(Experiment.id == experiment_id)
        ).first()

    def list(self, skip: int = 0, limit: int = 50) -> List[Experiment]:
        return list(
            self.db.scalars(
                select(Experiment).order_by(desc(Experiment.created_at)).offset(skip).limit(limit)
            ).all()
        )

    def count(self) -> int:
        return len(self.db.scalars(select(Experiment.id)).all())

    def create(
        self,
        name: str,
        strategy: str,
        rounds: int,
        num_clients: int,
        proximal_mu: float = 0.1,
        is_attack: bool = False,
        attack_type: Optional[str] = None,
        status: str = "completed",
    ) -> Experiment:
        exp = Experiment(
            name=name,
            strategy=strategy,
            rounds=rounds,
            num_clients=num_clients,
            proximal_mu=proximal_mu,
            is_attack=is_attack,
            attack_type=attack_type,
            status=status,
            completed_at=datetime.now(timezone.utc) if status == "completed" else None,
        )
        self.db.add(exp)
        self.db.commit()
        self.db.refresh(exp)
        return exp

    def add_metric(
        self,
        experiment_id: str,
        round_number: int,
        train_loss: Optional[float] = None,
        accuracy: Optional[float] = None,
        f1_score: Optional[float] = None,
    ) -> ExperimentMetric:
        metric = ExperimentMetric(
            experiment_id=experiment_id,
            round_number=round_number,
            train_loss=train_loss,
            accuracy=accuracy,
            f1_score=f1_score,
        )
        self.db.add(metric)
        self.db.commit()
        self.db.refresh(metric)
        return metric

    def update_status(
        self,
        experiment_id: str,
        status: str,
        error_message: Optional[str] = None,
    ) -> Optional[Experiment]:
        exp = self.get_by_id(experiment_id)
        if not exp:
            return None
        exp.status = status
        exp.error_message = error_message
        if status in ("completed", "failed"):
            exp.completed_at = datetime.now(timezone.utc)
        self.db.commit()
        self.db.refresh(exp)
        return exp
