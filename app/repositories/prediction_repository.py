import uuid
from typing import List, Optional
from sqlalchemy.orm import Session
from sqlalchemy import select, desc
from app.models.prediction_log import PredictionLog

class PredictionRepository:
    def __init__(self, db: Session):
        self.db = db

    def create(
        self,
        model_id: str,
        predicted_class: str,
        predicted_class_id: int,
        confidence: float,
        latency_ms: float,
        input_summary: Optional[str] = None,
        prediction_id: Optional[str] = None,
    ) -> PredictionLog:
        log = PredictionLog(
            id=prediction_id or str(uuid.uuid4()),
            model_id=model_id,
            predicted_class=predicted_class,
            predicted_class_id=predicted_class_id,
            confidence=confidence,
            latency_ms=latency_ms,
            input_summary=input_summary,
        )
        self.db.add(log)
        self.db.commit()
        self.db.refresh(log)
        return log

    def get_by_id(self, prediction_id: str) -> Optional[PredictionLog]:
        return self.db.scalars(
            select(PredictionLog).where(PredictionLog.id == prediction_id)
        ).first()

    def list(self, skip: int = 0, limit: int = 50) -> List[PredictionLog]:
        return list(
            self.db.scalars(
                select(PredictionLog).order_by(desc(PredictionLog.created_at)).offset(skip).limit(limit)
            ).all()
        )

    def count(self) -> int:
        return len(self.db.scalars(select(PredictionLog.id)).all())
