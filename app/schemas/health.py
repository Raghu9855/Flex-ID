from datetime import datetime, timezone
from typing import Dict, Any
from pydantic import BaseModel, Field

class HealthResponse(BaseModel):
    status: str = Field(default="healthy", description="Overall system health status")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    environment: str = Field(..., description="Running environment (development/production)")
    version: str = Field(..., description="API version")
    database_connected: bool = Field(..., description="Database connectivity status")
    active_model: str = Field(..., description="Active default model version")
    components: Dict[str, Any] = Field(default_factory=dict, description="Detailed component statuses")
