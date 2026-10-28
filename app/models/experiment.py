import uuid
from datetime import datetime, timezone
from typing import List, Optional
from sqlalchemy import String, Integer, Float, Boolean, DateTime, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.db.base import Base

def utc_now() -> datetime:
    return datetime.now(timezone.utc)

class Experiment(Base):
    __tablename__ = "experiments"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    strategy: Mapped[str] = mapped_column(String(64), nullable=False, default="fedprox")
    rounds: Mapped[int] = mapped_column(Integer, nullable=False, default=10)
    num_clients: Mapped[int] = mapped_column(Integer, nullable=False, default=4)
    proximal_mu: Mapped[float] = mapped_column(Float, nullable=False, default=0.1)
    is_attack: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    attack_type: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="completed")  # pending, running, completed, failed
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    # Relationships
    metrics: Mapped[List["ExperimentMetric"]] = relationship(
        "ExperimentMetric", back_populates="experiment", cascade="all, delete-orphan", order_by="ExperimentMetric.round_number"
    )
    models: Mapped[List["ModelRecord"]] = relationship(
        "ModelRecord", back_populates="experiment"
    )
