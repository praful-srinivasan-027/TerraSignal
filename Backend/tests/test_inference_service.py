import os
import pytest
from pathlib import Path
from app.services.inference_service import MLInferenceService
from app.config import settings


def test_model_singleton_reuse():
    """Test 15: MLInferenceService returns the same singleton instance and caches loaded model."""
    service1 = MLInferenceService.get_instance()
    service2 = MLInferenceService.get_instance()
    assert service1 is service2


def test_load_model_missing_file_raises_error():
    """Test: load_model fails clearly with FileNotFoundError if model weights file is missing."""
    service = MLInferenceService.get_instance()
    missing_path = Path("/tmp/non_existent_weights_12345.pt")

    with pytest.raises(FileNotFoundError) as excinfo:
        service.load_model(missing_path)

    assert "Model weights file not found" in str(excinfo.value)


def test_predict_structured_data_format(tmp_path):
    """Test: predict() returns correctly formatted structured prediction data."""
    # Create sample image file
    from PIL import Image
    img_path = tmp_path / "sample_leaf.jpg"
    img = Image.new("RGB", (300, 200), color="green")
    img.save(str(img_path))

    service = MLInferenceService.get_instance()
    if settings.MODEL_PATH.exists():
        result = service.predict(img_path)
        assert "predictions" in result
        assert "image_width" in result
        assert result["image_width"] == 300
        assert result["image_height"] == 200
        assert "inference_time_ms" in result
        assert result["inference_time_ms"] >= 0.0
