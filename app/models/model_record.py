import uuid
from datetime import datetime, timezone
from typing import Optional, List
from sqlalchemy import String, Integer, Float, Boolean, DateTime, ForeignKey, Index
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.db.base import Base

def utc_now() -> datetime:
    return datetime.now(timezone.utc)

class ModelRecord(Base):
    __tablename__ = "model_records"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    model_version: Mapped[str] = mapped_column(String(64), unique=True, nullable=False, index=True)
    strategy: Mapped[str] = mapped_column(String(64), nullable=False)
    weights_path: Mapped[str] = mapped_column(String(255), nullable=False)
    experiment_id: Mapped[Optional[str]] = mapped_column(String(36), ForeignKey("experiments.id", ondelete="SET NULL"), nullable=True)
    input_features_count: Mapped[int] = mapped_column(Integer, nullable=False, default=29)
    num_classes: Mapped[int] = mapped_column(Integer, nullable=False, default=13)
    accuracy: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    macro_f1: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    weighted_f1: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)

    # Relationships
    experiment: Mapped[Optional["Experiment"]] = relationship("Experiment", back_populates="models")
    predictions: Mapped[List["PredictionLog"]] = relationship("PredictionLog", back_populates="model")
