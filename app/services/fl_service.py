import asyncio
import os
import threading
import time
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.logging import logger
from app.core.exceptions import ExperimentNotFoundError
from app.db.session import SessionLocal
from app.models.experiment import Experiment
from app.models.experiment_metric import ExperimentMetric
from app.schemas.experiment import ExperimentCreate, ExperimentResponse, MetricRecord
from app.repositories.experiment_repository import ExperimentRepository


class FederatedLearningService:
    def __init__(self, db: Session):
        self.db = db
        self.repo = ExperimentRepository(db)

    def list_experiments(self, skip: int = 0, limit: int = 50) -> List[ExperimentResponse]:
        records = self.repo.list(skip=skip, limit=limit)
        return [self._format_experiment(exp) for exp in records]

    def get_experiment(self, experiment_id: str) -> ExperimentResponse:
        exp = self.repo.get_by_id(experiment_id)
        if not exp:
            raise ExperimentNotFoundError(experiment_id)
        return self._format_experiment(exp)

    def get_metrics(self, experiment_id: str) -> List[MetricRecord]:
        exp = self.repo.get_by_id(experiment_id)
        if not exp:
            raise ExperimentNotFoundError(experiment_id)
        return [
            MetricRecord(
                round_number=m.round_number,
                train_loss=m.train_loss,
                accuracy=m.accuracy,
                f1_score=m.f1_score,
            )
            for m in exp.metrics
        ]

    def create_experiment(self, payload: ExperimentCreate) -> ExperimentResponse:
        """
        Registers a new experiment and starts background simulated orchestration.
        """
        exp = self.repo.create(
            name=payload.name,
            strategy=payload.strategy.lower(),
            rounds=payload.rounds,
            num_clients=payload.num_clients,
            proximal_mu=payload.proximal_mu,
            is_attack=payload.is_attack,
            attack_type=payload.attack_type,
            status="running",
        )

        # Launch non-blocking background runner thread
        thread = threading.Thread(
            target=self._run_simulation_background,
            args=(exp.id, payload.strategy, payload.rounds, payload.proximal_mu, payload.is_attack),
            daemon=True,
        )
        thread.start()

        return self._format_experiment(exp)

    @staticmethod
    def _run_simulation_background(
        experiment_id: str,
        strategy: str,
        rounds: int,
        proximal_mu: float,
        is_attack: bool,
    ) -> None:
        """
        Simulates progressive federated rounds with realistic loss degradation/convergence
        and records metrics safely into database without blocking the API server.
        """
        db = SessionLocal()
        try:
            repo = ExperimentRepository(db)
            logger.info("Started background FL simulation for experiment %s (strategy=%s, rounds=%d)", experiment_id, strategy, rounds)

            # Baseline metrics based on strategy
            base_loss = 0.85
            base_acc = 0.70
            base_f1 = 0.40

            for r in range(1, rounds + 1):
                time.sleep(0.5)  # Simulate compute step

                decay = (r / rounds)
                if strategy == "fedprox":
                    loss = max(0.30, base_loss * (1.0 - 0.55 * decay))
                    acc = min(0.915, base_acc + (0.21 * decay))
                    f1 = min(0.60, base_f1 + (0.19 * decay))
                else:  # FedAvg
                    loss = max(0.35, base_loss * (1.0 - 0.50 * decay))
                    acc = min(0.83, base_acc + (0.12 * decay))
                    f1 = min(0.45, base_f1 + (0.05 * decay))

                if is_attack and strategy == "fedavg":
                    # Severe degradation under attack
                    acc = max(0.72, acc - 0.08)
                    f1 = max(0.35, f1 - 0.07)
                elif is_attack and strategy == "fedprox":
                    # More robust under attack
                    acc = max(0.88, acc - 0.02)
                    f1 = max(0.51, f1 - 0.02)

                repo.add_metric(
                    experiment_id=experiment_id,
                    round_number=r,
                    train_loss=round(float(loss), 4),
                    accuracy=round(float(acc), 4),
                    f1_score=round(float(f1), 4),
                )

            repo.update_status(experiment_id, "completed")
            logger.info("Experiment %s successfully completed in background.", experiment_id)
        except Exception as e:
            logger.exception("Error in background FL simulation: %s", e)
            try:
                repo = ExperimentRepository(db)
                repo.update_status(experiment_id, "failed", error_message=str(e))
            except Exception:
                pass
        finally:
            db.close()

    def _format_experiment(self, exp: Experiment) -> ExperimentResponse:
        metrics_list = [
            MetricRecord(
                round_number=m.round_number,
                train_loss=m.train_loss,
                accuracy=m.accuracy,
                f1_score=m.f1_score,
            )
            for m in exp.metrics
        ]
        return ExperimentResponse(
            id=exp.id,
            name=exp.name,
            strategy=exp.strategy,
            rounds=exp.rounds,
            num_clients=exp.num_clients,
            proximal_mu=exp.proximal_mu,
            is_attack=exp.is_attack,
            attack_type=exp.attack_type,
            status=exp.status,
            error_message=exp.error_message,
            created_at=exp.created_at,
            completed_at=exp.completed_at,
            metrics=metrics_list,
        )
