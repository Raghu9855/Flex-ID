import json
import os
from sqlalchemy.orm import Session
from app.core.config import settings
from app.core.logging import logger
from app.models.experiment import Experiment
from app.models.experiment_metric import ExperimentMetric
from app.models.model_record import ModelRecord

def seed_database(db: Session) -> None:
    """Populates baseline models and historical experiment records if database is empty."""
    existing_models = db.query(ModelRecord).count()
    if existing_models > 0:
        logger.info("Database already seeded with %d models. Skipping seed.", existing_models)
        return

    logger.info("Seeding database with verified baseline models and experiment records...")

    # 1. Seed Clean FedProx Experiment
    exp_fedprox = Experiment(
        name="FedProx Baseline (10 Rounds, Non-IID alpha=0.4)",
        strategy="fedprox",
        rounds=10,
        num_clients=4,
        proximal_mu=0.1,
        is_attack=False,
        status="completed",
    )
    db.add(exp_fedprox)
    db.flush()

    # 2. Seed Clean FedAvg Experiment
    exp_fedavg = Experiment(
        name="FedAvg Baseline (10 Rounds, Non-IID alpha=0.4)",
        strategy="fedavg",
        rounds=10,
        num_clients=4,
        proximal_mu=0.0,
        is_attack=False,
        status="completed",
    )
    db.add(exp_fedavg)
    db.flush()

    # 3. Seed Attack FedProx Experiment
    exp_fedprox_atk = Experiment(
        name="FedProx Adversarial (Label-Flipping, 10 Rounds)",
        strategy="fedprox",
        rounds=10,
        num_clients=4,
        proximal_mu=0.1,
        is_attack=True,
        attack_type="flip",
        status="completed",
    )
    db.add(exp_fedprox_atk)
    db.flush()

    # 4. Seed Attack FedAvg Experiment
    exp_fedavg_atk = Experiment(
        name="FedAvg Adversarial (Label-Flipping, 10 Rounds)",
        strategy="fedavg",
        rounds=10,
        num_clients=4,
        proximal_mu=0.0,
        is_attack=True,
        attack_type="flip",
        status="completed",
    )
    db.add(exp_fedavg_atk)
    db.flush()

    # Seed round metrics if round_metrics_*.csv exist
    fedprox_csv = settings.RESULTS_DIR / "round_metrics_fedprox.csv"
    if fedprox_csv.exists():
        try:
            import pandas as pd
            df_fp = pd.read_csv(fedprox_csv)
            for _, row in df_fp.iterrows():
                db.add(ExperimentMetric(
                    experiment_id=exp_fedprox.id,
                    round_number=int(row.get("round", row.get("Round", 1))),
                    train_loss=float(row.get("train_loss", row.get("Loss", 0.0))) if pd.notnull(row.get("train_loss", row.get("Loss", None))) else None,
                    accuracy=float(row.get("accuracy", row.get("Accuracy", 0.0))) if pd.notnull(row.get("accuracy", row.get("Accuracy", None))) else None,
                    f1_score=float(row.get("f1", row.get("F1", 0.0))) if pd.notnull(row.get("f1", row.get("F1", None))) else None,
                ))
        except Exception as e:
            logger.warning("Could not seed FedProx round metrics from CSV: %s", e)

    fedavg_csv = settings.RESULTS_DIR / "round_metrics_fedavg.csv"
    if fedavg_csv.exists():
        try:
            import pandas as pd
            df_fa = pd.read_csv(fedavg_csv)
            for _, row in df_fa.iterrows():
                db.add(ExperimentMetric(
                    experiment_id=exp_fedavg.id,
                    round_number=int(row.get("round", row.get("Round", 1))),
                    train_loss=float(row.get("train_loss", row.get("Loss", 0.0))) if pd.notnull(row.get("train_loss", row.get("Loss", None))) else None,
                    accuracy=float(row.get("accuracy", row.get("Accuracy", 0.0))) if pd.notnull(row.get("accuracy", row.get("Accuracy", None))) else None,
                    f1_score=float(row.get("f1", row.get("F1", 0.0))) if pd.notnull(row.get("f1", row.get("F1", None))) else None,
                ))
        except Exception as e:
            logger.warning("Could not seed FedAvg round metrics from CSV: %s", e)

    # 5. Register Model Records pointing to actual weights
    models = [
        ModelRecord(
            model_version="fedprox-v1",
            strategy="fedprox",
            weights_path="results/fedproxeachround/round-10-weights.pkl",
            experiment_id=exp_fedprox.id,
            input_features_count=29,
            num_classes=13,
            accuracy=0.9093,
            macro_f1=0.5899,
            weighted_f1=0.9109,
            is_active=True,  # Default active serving model
        ),
        ModelRecord(
            model_version="fedavg-v1",
            strategy="fedavg",
            weights_path="results/fedavgeachround/round-10-weights.pkl",
            experiment_id=exp_fedavg.id,
            input_features_count=29,
            num_classes=13,
            accuracy=0.8147,
            macro_f1=0.4350,
            weighted_f1=0.8382,
            is_active=False,
        ),
        ModelRecord(
            model_version="fedprox-attack-v1",
            strategy="fedprox",
            weights_path="results/fedproxunderattack/round-10-weights.pkl",
            experiment_id=exp_fedprox_atk.id,
            input_features_count=29,
            num_classes=13,
            accuracy=0.8882,
            macro_f1=0.5186,
            weighted_f1=0.8495,
            is_active=False,
        ),
        ModelRecord(
            model_version="fedavg-attack-v1",
            strategy="fedavg",
            weights_path="results/fedunderattack/round-10-weights.pkl",
            experiment_id=exp_fedavg_atk.id,
            input_features_count=29,
            num_classes=13,
            accuracy=0.8549,
            macro_f1=0.4422,
            weighted_f1=0.8577,
            is_active=False,
        ),
    ]

    for model in models:
        db.add(model)

    db.commit()
    logger.info("Database seeding completed successfully.")
