import os
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from app.core.config import settings
from app.core.logging import logger
from app.core.exceptions import ModelNotFoundError, ScalerNotFoundError

# Ensure ml directory is on sys.path for importing model definition
ML_DIR = str(settings.PROJECT_ROOT_PATH / "ml")
if ML_DIR not in sys.path:
    sys.path.insert(0, ML_DIR)

from model import create_dnn_model


class ModelManager:
    """
    Singleton service responsible for loading, compiling, caching, and serving
    trained Keras intrusion detection models and preprocessing artifacts.
    """
    _instance: Optional["ModelManager"] = None

    def __new__(cls) -> "ModelManager":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if getattr(self, "_initialized", False):
            return

        self._models_cache: Dict[str, Any] = {}
        self._scaler: Optional[Any] = None
        self._label_encoder: Optional[Any] = None
        self._class_names: List[str] = []

        self._load_label_encoder()
        self._load_scaler()
        self._initialized = True

    def _load_label_encoder(self) -> None:
        le_path = settings.DATA_DIR / "label_encoder.pkl"
        if not le_path.exists():
            logger.warning("Label encoder not found at %s. Using default binary classes.", le_path)
            self._class_names = ["Benign", "Malicious"]
            return

        try:
            with open(le_path, "rb") as f:
                self._label_encoder = pickle.load(f)
            self._class_names = [str(c) for c in self._label_encoder.classes_]
            logger.info("Loaded label encoder with %d classes: %s", len(self._class_names), self._class_names[:5])
        except Exception as e:
            logger.error("Failed to load label encoder: %s", e)
            self._class_names = ["Benign", "Malicious"]

    def _load_scaler(self) -> None:
        scaler_path = settings.DATA_DIR / "scaler.pkl"
        if not scaler_path.exists():
            logger.warning("Scaler not found at %s.", scaler_path)
            return

        try:
            with open(scaler_path, "rb") as f:
                self._scaler = pickle.load(f)
            logger.info("Loaded MinMaxScaler artifact successfully.")
        except Exception as e:
            logger.error("Failed to load scaler from %s: %s", scaler_path, e)

    @property
    def class_names(self) -> List[str]:
        return self._class_names

    @property
    def scaler(self) -> Optional[Any]:
        return self._scaler

    def get_model(self, model_version: str, weights_path: Optional[str] = None) -> Any:
        """
        Retrieves a compiled Keras model from the cache or loads its weights from disk.
        """
        if model_version in self._models_cache:
            return self._models_cache[model_version]

        if not weights_path:
            # Map standard versions to files
            version_map = {
                "fedprox-v1": "results/fedproxeachround/round-10-weights.pkl",
                "fedavg-v1": "results/fedavgeachround/round-10-weights.pkl",
                "fedprox-attack-v1": "results/fedproxunderattack/round-10-weights.pkl",
                "fedavg-attack-v1": "results/fedunderattack/round-10-weights.pkl",
            }
            weights_path = version_map.get(model_version)

        if not weights_path:
            raise ModelNotFoundError(f"Model version '{model_version}' not registered and no weights path provided.")

        full_weights_path = settings.PROJECT_ROOT_PATH / weights_path
        if not full_weights_path.exists():
            raise ModelNotFoundError(f"Weights file not found at: {full_weights_path}")

        logger.info("Loading model weights for version '%s' from: %s", model_version, full_weights_path)
        try:
            with open(full_weights_path, "rb") as f:
                weights = pickle.load(f)

            input_dim = 29
            num_classes = len(self._class_names)
            model = create_dnn_model(input_shape=input_dim, num_classes=num_classes)
            model.set_weights(weights)

            # Warm-up compile
            self._models_cache[model_version] = model
            logger.info("Model '%s' successfully cached in memory.", model_version)
            return model
        except Exception as e:
            logger.exception("Failed to build model '%s': %s", model_version, e)
            raise ModelNotFoundError(f"Failed to instantiate model '{model_version}': {str(e)}")

    def warmup(self) -> None:
        """Pre-loads default active model and runs a dry inference pass to eliminate cold start."""
        try:
            default_version = settings.DEFAULT_MODEL_VERSION
            model = self.get_model(default_version)
            dummy_input = np.zeros((1, 29), dtype=np.float32)
            _ = model(dummy_input, training=False)
            logger.info("Warmup complete for default model: %s", default_version)
        except Exception as e:
            logger.warning("Warmup failed (non-fatal): %s", e)


model_manager = ModelManager()
