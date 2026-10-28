from typing import List, Dict, Optional, Union, Any
from pydantic import BaseModel, Field, model_validator

CANONICAL_FEATURES: List[str] = [
    "Dst Port", "Protocol", "Flow Duration", "Tot Fwd Pkts", "Tot Bwd Pkts",
    "TotLen Fwd Pkts", "TotLen Bwd Pkts", "Fwd Pkt Len Max", "Fwd Pkt Len Mean",
    "Bwd Pkt Len Max", "Bwd Pkt Len Mean", "Flow Byts/s", "Flow Pkts/s",
    "Flow IAT Mean", "Flow IAT Max", "Fwd IAT Mean", "Bwd IAT Mean",
    "Fwd Header Len", "Bwd Header Len", "Fwd Pkts/s", "Bwd Pkts/s",
    "Pkt Len Mean", "Pkt Len Max", "Pkt Len Var", "SYN Flag Cnt",
    "RST Flag Cnt", "ACK Flag Cnt", "Init Fwd Win Byts", "Init Bwd Win Byts"
]

class TrafficFeaturesInput(BaseModel):
    """
    Accepts either:
    1. A dictionary of named features (matching or case-insensitive to CANONICAL_FEATURES)
    2. An ordered list of 29 numeric values.
    """
    features: Union[Dict[str, float], List[float]] = Field(
        ...,
        description="29 network flow features either as named key-value dictionary or ordered list.",
        examples=[
            {
                "Dst Port": 80.0,
                "Protocol": 6.0,
                "Flow Duration": 125000.0,
                "Tot Fwd Pkts": 10.0,
                "Tot Bwd Pkts": 8.0,
                "TotLen Fwd Pkts": 500.0,
                "TotLen Bwd Pkts": 1200.0,
                "Fwd Pkt Len Max": 200.0,
                "Fwd Pkt Len Mean": 50.0,
                "Bwd Pkt Len Max": 400.0,
                "Bwd Pkt Len Mean": 150.0,
                "Flow Byts/s": 13600.0,
                "Flow Pkts/s": 144.0,
                "Flow IAT Mean": 8500.0,
                "Flow IAT Max": 25000.0,
                "Fwd IAT Mean": 12000.0,
                "Bwd IAT Mean": 15000.0,
                "Fwd Header Len": 200.0,
                "Bwd Header Len": 160.0,
                "Fwd Pkts/s": 80.0,
                "Bwd Pkts/s": 64.0,
                "Pkt Len Mean": 94.4,
                "Pkt Len Max": 400.0,
                "Pkt Len Var": 15200.0,
                "SYN Flag Cnt": 0.0,
                "RST Flag Cnt": 0.0,
                "ACK Flag Cnt": 1.0,
                "Init Fwd Win Byts": 26883.0,
                "Init Bwd Win Byts": 26883.0
            }
        ]
    )
    model_version: Optional[str] = Field(default=None, description="Optional target model version (e.g. 'fedprox-v1')")

    @model_validator(mode="after")
    def validate_features_shape(self) -> "TrafficFeaturesInput":
        if isinstance(self.features, list):
            if len(self.features) != len(CANONICAL_FEATURES):
                raise ValueError(
                    f"Feature array must have exactly {len(CANONICAL_FEATURES)} elements; received {len(self.features)}."
                )
        elif isinstance(self.features, dict):
            # Normalize dictionary keys
            clean_keys = {k.lower().replace("_", " ").strip(): k for k in self.features.keys()}
            missing = []
            for col in CANONICAL_FEATURES:
                c_norm = col.lower().replace("_", " ").strip()
                if c_norm not in clean_keys:
                    missing.append(col)
            if missing:
                raise ValueError(f"Missing required feature keys: {missing[:5]} (total {len(missing)} missing).")
        return self

class PredictionResponse(BaseModel):
    prediction_id: str
    model_version: str
    predicted_class: str = Field(..., description="Predicted network traffic classification (e.g., Benign, DoS attacks-Hulk)")
    predicted_class_id: int
    is_intrusion: bool = Field(..., description="True if classified as an attack traffic type, False if Benign")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Softmax confidence score")
    class_probabilities: Dict[str, float] = Field(default_factory=dict, description="Probabilities for all recognized classes")
    latency_ms: float = Field(..., description="Inference latency in milliseconds")

class BatchTrafficInput(BaseModel):
    records: List[TrafficFeaturesInput] = Field(..., min_length=1, max_length=1000)
    model_version: Optional[str] = None

class BatchPredictionResponse(BaseModel):
    model_version: str
    total_processed: int
    intrusions_detected: int
    predictions: List[PredictionResponse]
    total_latency_ms: float
