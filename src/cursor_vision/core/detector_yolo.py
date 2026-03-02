from pathlib import Path
from typing import Any, Union

from ultralytics import YOLO
import torch


class YOLODetector:
    """YOLO detector with automatic TensorRT fallback.
    
    Loads TensorRT engine (.engine) if available (same name as .pt but with .engine extension),
    otherwise falls back to standard YOLO model.
    """
    
    def __init__(self, model_path: str = "models/yolov8n.pt", conf: float = 0.1):
        self.model_path = Path(model_path)
        self.conf = float(conf)
        self._setup_model()

    def _setup_model(self) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Check for TensorRT engine file first
        # If model is .pt, look for .engine with same base name
        trt_path = self.model_path.with_suffix('.engine')
        if not trt_path.exists() and self.model_path.suffix == '.pt':
            # Also check for YOLO naming pattern: model.engine vs model.pt
            trt_path = self.model_path.parent / (self.model_path.stem + '.engine')
        
        if trt_path.exists() and self.device == "cuda":
            self._load_tensorrt(trt_path)
        else:
            self._load_standard()
    
    def _load_tensorrt(self, trt_path: Path) -> None:
        """Load TensorRT engine model."""
        self.model = YOLO(str(trt_path))
        self.model.to(self.device)
        self.model.conf = self.conf
        print(f"✓ TensorRT YOLO model loaded: {trt_path}")
        
    def _load_standard(self) -> None:
        """Load standard YOLO model."""
        self.model = YOLO(str(self.model_path))
        self.model.to(self.device)
        self.model.conf = self.conf
        print(f"✓ Standard YOLO model loaded: {self.model_path}")

    def _perform_detection(self, image) -> Any:
        return self.model.predict(
            image,
            conf=self.model.conf,
            half=self.device == "cuda",
            verbose=False,
        )

    def detect(self, image) -> Any:
        results = self._perform_detection(image)
        return results, image
