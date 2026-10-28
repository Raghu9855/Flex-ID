import time
import uuid
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.logging import logger
from app.core.exceptions import (
    FeatureValidationError,
    ModelInferenceError,
    BatchSizeExceededError,
)
from app.schemas.prediction import (
    CANONICAL_FEATURES,
    TrafficFeaturesInput,
    PredictionResponse,
    BatchTrafficInput,
    BatchPredictionResponse,
)
from app.services.model_manager import model_manager
from app.repositories.model_repository import ModelRepository
from app.repositories.prediction_repository import PredictionRepository


class InferenceService:
    def __init__(self, db: Optional[Session] = None):
        self.db = db
        self.model_repo = ModelRepository(db) if db else None
        self.prediction_repo = PredictionRepository(db) if db else None

    def _extract_feature_vector(self, item: TrafficFeaturesInput) -> np.ndarray:
        """
        Converts either a dictionary or an ordered list into a 1D float array of shape (29,).
        """
        if isinstance(item.features, list):
            if len(item.features) != len(CANONICAL_FEATURES):
                raise FeatureValidationError(
                    f"Expected {len(CANONICAL_FEATURES)} feature values, received {len(item.features)}."
                )
            arr = np.array(item.features, dtype=np.float32)
        elif isinstance(item.features, dict):
            # Normalize user keys to lower-case without underscores
            normalized_dict = {
                k.lower().replace("_", " ").strip(): float(v)
                for k, v in item.features.items()
            }
            ordered = []
            for col in CANONICAL_FEATURES:
                c_norm = col.lower().replace("_", " ").strip()
                if c_norm in normalized_dict:
                    ordered.append(normalized_dict[c_norm])
                else:
                    raise FeatureValidationError(f"Missing required feature: '{col}'")
            arr = np.array(ordered, dtype=np.float32)
        else:
            raise FeatureValidationError("Features must be a dictionary or list.")

        # Check for non-finite values
        if not np.all(np.isfinite(arr)):
            arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)

        return arr

    def _preprocess_batch(self, feature_matrix: np.ndarray) -> np.ndarray:
        """
        Scales feature matrix into [0.0, 1.0].
        If features are already in [0, 1], keeps them intact; if raw values (>1.0),
        uses MinMaxScaler if available or clips within sensible bounds.
        """
        scaler = model_manager.scaler
        # Check if values exceed 1.0 (indicating raw network packet stats)
        needs_scaling = np.any(feature_matrix > 1.0) or np.any(feature_matrix < 0.0)

        if needs_scaling and scaler is not None:
            try:
                import pandas as pd
                cols = getattr(scaler, "feature_names_in_", None)
                if cols is not None and len(cols) == feature_matrix.shape[1]:
                    feature_matrix = scaler.transform(pd.DataFrame(feature_matrix, columns=cols))
                else:
                    feature_matrix = scaler.transform(feature_matrix)
            except Exception as e:
                logger.warning("Scaler transform failed: %s. Clipping to [0, 1].", e)

        # Always strictly bound features within [0.0, 1.0] for the neural network
        feature_matrix = np.clip(feature_matrix, 0.0, 1.0)
        return feature_matrix.astype(np.float32)

    def predict_single(self, input_data: TrafficFeaturesInput) -> PredictionResponse:
        """
        Executes inference for a single network flow record.
        """
        start_time = time.perf_counter()

        model_version = input_data.model_version or settings.DEFAULT_MODEL_VERSION
        model = model_manager.get_model(model_version)

        raw_vector = self._extract_feature_vector(input_data)
        scaled_matrix = self._preprocess_batch(raw_vector.reshape(1, -1))

        try:
            probabilities = model(scaled_matrix, training=False).numpy()[0]
        except Exception:
            try:
                probabilities = model.predict(scaled_matrix, verbose=0)[0]
            except Exception as e:
                logger.exception("Inference execution failed: %s", e)
                raise ModelInferenceError(f"Model prediction failed: {str(e)}")

        latency_ms = (time.perf_counter() - start_time) * 1000.0

        predicted_class_id = int(np.argmax(probabilities))
        class_names = model_manager.class_names
        predicted_class = (
            class_names[predicted_class_id]
            if predicted_class_id < len(class_names)
            else f"Class_{predicted_class_id}"
        )
        is_intrusion = predicted_class.lower() != "benign"
        confidence = float(probabilities[predicted_class_id])

        # Form class probability dictionary
        prob_dict = {
            class_names[i]: round(float(p), 4)
            for i, p in enumerate(probabilities)
            if i < len(class_names)
        }

        prediction_id = str(uuid.uuid4())

        # Audit to DB if session provided
        if self.db and self.model_repo and self.prediction_repo:
            try:
                db_model = self.model_repo.get_by_version(model_version)
                model_db_id = db_model.id if db_model else prediction_id
                self.prediction_repo.create(
                    prediction_id=prediction_id,
                    model_id=model_db_id,
                    predicted_class=predicted_class,
                    predicted_class_id=predicted_class_id,
                    confidence=confidence,
                    latency_ms=latency_ms,
                    input_summary=f"Features max={scaled_matrix.max():.2f}, min={scaled_matrix.min():.2f}",
                )
            except Exception as e:
                logger.warning("Failed to persist prediction audit log: %s", e)

        return PredictionResponse(
            prediction_id=prediction_id,
            model_version=model_version,
            predicted_class=predicted_class,
            predicted_class_id=predicted_class_id,
            is_intrusion=is_intrusion,
            confidence=confidence,
            class_probabilities=prob_dict,
            latency_ms=round(latency_ms, 2),
        )

    def predict_batch(self, batch_data: BatchTrafficInput) -> BatchPredictionResponse:
        """
        Executes bounded batch inference for multiple traffic flows.
        """
        start_time = time.perf_counter()
        records_count = len(batch_data.records)

        if records_count > settings.MAX_BATCH_SIZE:
            raise BatchSizeExceededError(settings.MAX_BATCH_SIZE)

        model_version = batch_data.model_version or settings.DEFAULT_MODEL_VERSION
        model = model_manager.get_model(model_version)

        # Vectorize extraction
        vectors = [self._extract_feature_vector(rec) for rec in batch_data.records]
        raw_matrix = np.vstack(vectors)
        scaled_matrix = self._preprocess_batch(raw_matrix)

        try:
            prob_matrix = model(scaled_matrix, training=False).numpy()
        except Exception:
            try:
                prob_matrix = model.predict(scaled_matrix, batch_size=256, verbose=0)
            except Exception as e:
                logger.exception("Batch inference execution failed: %s", e)
                raise ModelInferenceError(f"Batch prediction failed: {str(e)}")

        total_latency_ms = (time.perf_counter() - start_time) * 1000.0
        per_record_latency = total_latency_ms / max(1, records_count)

        class_names = model_manager.class_names
        predictions: List[PredictionResponse] = []
        intrusions_detected = 0

        for i in range(records_count):
            probs = prob_matrix[i]
            cid = int(np.argmax(probs))
            cname = class_names[cid] if cid < len(class_names) else f"Class_{cid}"
            is_attk = cname.lower() != "benign"
            if is_attk:
                intrusions_detected += 1

            prob_dict = {
                class_names[idx]: round(float(p), 4)
                for idx, p in enumerate(probs)
                if idx < len(class_names)
            }

            predictions.append(
                PredictionResponse(
                    prediction_id=str(uuid.uuid4()),
                    model_version=model_version,
                    predicted_class=cname,
                    predicted_class_id=cid,
                    is_intrusion=is_attk,
                    confidence=float(probs[cid]),
                    class_probabilities=prob_dict,
                    latency_ms=round(per_record_latency, 2),
                )
            )

        return BatchPredictionResponse(
            model_version=model_version,
            total_processed=records_count,
            intrusions_detected=intrusions_detected,
            predictions=predictions,
            total_latency_ms=round(total_latency_ms, 2),
        )
