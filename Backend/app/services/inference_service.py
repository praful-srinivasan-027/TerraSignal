import time
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
import threading
import torch
import cv2
from ultralytics import YOLO
from app.config import settings

logger = logging.getLogger(__name__)


class MLInferenceService:
    """
    ML Inference Service abstraction.
    Manages YOLO model loading, caching, hardware acceleration (CUDA/CPU),
    and structured prediction output formatting.
    """

    _instance: Optional["MLInferenceService"] = None
    _lock = threading.Lock()

    def __init__(self):
        self._model: Optional[YOLO] = None
        self._device: str = "cpu"
        self._weights_path: Optional[Path] = None

    @classmethod
    def get_instance(cls) -> "MLInferenceService":
        """Singleton accessor for thread-safe model service instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def load_model(self, weights_path: Optional[Path | str] = None) -> YOLO:
        """
        Load or reuse cached YOLO model instance. Detects CUDA hardware acceleration.
        Fails clearly if model file does not exist.
        """
        target_path = Path(weights_path or settings.MODEL_PATH)

        if self._model is not None and self._weights_path == target_path:
            return self._model

        with self._lock:
            if self._model is not None and self._weights_path == target_path:
                return self._model

            if not target_path.exists():
                logger.error(f"ML model weights file not found at: {target_path}")
                raise FileNotFoundError(
                    f"Model weights file not found at '{target_path}'. Please verify configuration."
                )

            # Detect hardware device
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Loading YOLO model from: {target_path} (Device: {self._device})")

            model = YOLO(str(target_path))
            self._model = model
            self._weights_path = target_path
            logger.info("YOLO Model successfully loaded and cached in memory.")
            return self._model

    def predict(
        self,
        image_path: Path | str,
        conf_threshold: Optional[float] = None,
        iou_threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Run inference on stored image file and return structured prediction data.

        Returns:
            {
                "predictions": [
                    {
                        "class_id": int,
                        "class_name": str,
                        "confidence": float,
                        "x1": float,
                        "y1": float,
                        "x2": float,
                        "y2": float
                    }
                ],
                "image_width": int,
                "image_height": int,
                "inference_time_ms": float,
                "model_version": str
            }
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Input image file not found at: {image_path}")

        model = self.load_model()
        conf = conf_threshold if conf_threshold is not None else settings.CONFIDENCE_THRESHOLD
        iou = iou_threshold if iou_threshold is not None else settings.IOU_THRESHOLD

        # Read image to obtain exact width and height dimensions
        img_bgr = cv2.imread(str(image_path))
        if img_bgr is None:
            raise ValueError(f"Unable to read image file at: {image_path}")
        height, width = img_bgr.shape[:2]

        start_time = time.perf_counter()
        results = model.predict(
            source=str(image_path),
            conf=conf,
            iou=iou,
            device=self._device,
            verbose=False,
        )
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        predictions: List[Dict[str, Any]] = []
        if len(results) > 0 and results[0].boxes is not None:
            for box in results[0].boxes:
                cls_id = int(box.cls[0].item())
                cls_name = results[0].names[cls_id] if (results[0].names and cls_id in results[0].names) else str(cls_id)
                confidence = float(box.conf[0].item())
                xyxy = box.xyxy[0].tolist()

                predictions.append({
                    "class_id": cls_id,
                    "class_name": cls_name,
                    "confidence": float(confidence),
                    "x1": float(xyxy[0]),
                    "y1": float(xyxy[1]),
                    "x2": float(xyxy[2]),
                    "y2": float(xyxy[3]),
                })

        return {
            "predictions": predictions,
            "image_width": width,
            "image_height": height,
            "inference_time_ms": round(elapsed_ms, 2),
            "model_version": settings.MODEL_VERSION,
        }
