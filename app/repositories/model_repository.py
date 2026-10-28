from typing import List, Optional
from sqlalchemy.orm import Session
from sqlalchemy import select, desc, update
from app.models.model_record import ModelRecord

class ModelRepository:
    def __init__(self, db: Session):
        self.db = db

    def get_by_id(self, model_id: str) -> Optional[ModelRecord]:
        return self.db.scalars(
            select(ModelRecord).where(ModelRecord.id == model_id)
        ).first()

    def get_by_version(self, model_version: str) -> Optional[ModelRecord]:
        return self.db.scalars(
            select(ModelRecord).where(ModelRecord.model_version == model_version)
        ).first()

    def get_active(self) -> Optional[ModelRecord]:
        return self.db.scalars(
            select(ModelRecord).where(ModelRecord.is_active == True)
        ).first()

    def list(self) -> List[ModelRecord]:
        return list(
            self.db.scalars(
                select(ModelRecord).order_by(desc(ModelRecord.created_at))
            ).all()
        )

    def count(self) -> int:
        return len(self.db.scalars(select(ModelRecord.id)).all())

    def create(
        self,
        model_version: str,
        strategy: str,
        weights_path: str,
        experiment_id: Optional[str] = None,
        input_features_count: int = 29,
        num_classes: int = 13,
        accuracy: Optional[float] = None,
        macro_f1: Optional[float] = None,
        weighted_f1: Optional[float] = None,
        is_active: bool = False,
    ) -> ModelRecord:
        if is_active:
            # Set all others to inactive
            self.db.execute(
                update(ModelRecord).values(is_active=False)
            )

        model = ModelRecord(
            model_version=model_version,
            strategy=strategy,
            weights_path=weights_path,
            experiment_id=experiment_id,
            input_features_count=input_features_count,
            num_classes=num_classes,
            accuracy=accuracy,
            macro_f1=macro_f1,
            weighted_f1=weighted_f1,
            is_active=is_active,
        )
        self.db.add(model)
        self.db.commit()
        self.db.refresh(model)
        return model

    def set_active(self, model_version: str) -> Optional[ModelRecord]:
        model = self.get_by_version(model_version)
        if not model:
            return None
        self.db.execute(update(ModelRecord).values(is_active=False))
        model.is_active = True
        self.db.commit()
        self.db.refresh(model)
        return model
