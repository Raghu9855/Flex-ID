import numpy as np
import pytest
from app.services.inference_service import InferenceService
from app.schemas.prediction import TrafficFeaturesInput, CANONICAL_FEATURES
from app.core.exceptions import FeatureValidationError

def test_inference_service_feature_extraction_list():
    service = InferenceService()
    input_data = TrafficFeaturesInput(features=[0.5] * 29)
    vec = service._extract_feature_vector(input_data)
    assert vec.shape == (29,)
    assert np.allclose(vec, 0.5)

def test_inference_service_feature_extraction_dict(sample_benign_features):
    service = InferenceService()
    input_data = TrafficFeaturesInput(features=sample_benign_features)
    vec = service._extract_feature_vector(input_data)
    assert vec.shape == (29,)
    assert vec[0] == 80.0  # Dst Port

def test_inference_service_nan_handling():
    service = InferenceService()
    features = [0.1] * 29
    features[5] = float("nan")
    features[6] = float("inf")
    input_data = TrafficFeaturesInput(features=features)
    vec = service._extract_feature_vector(input_data)
    assert np.all(np.isfinite(vec))
    assert vec[5] == 0.0
    assert vec[6] == 1.0

def test_inference_service_preprocess_batch_clipping():
    service = InferenceService()
    raw = np.array([[0.2, 0.8, -0.5, 1.5] + [0.5] * 25], dtype=np.float32)
    processed = service._preprocess_batch(raw)
    assert processed.min() >= 0.0
    assert processed.max() <= 1.0
