from typing import List
from pathlib import Path

import cv2
import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModel


def _is_tensorrt_available():
    """Check if torch_tensorrt is installed."""
    try:
        import torch_tensorrt
        return True
    except ImportError:
        return False



class HistogramEmbedder:
    def __init__(self, bins=(8, 8, 8)):
        self.bins = bins

    def embed(self, crops_bgr: List[np.ndarray]) -> np.ndarray:
        if len(crops_bgr) == 0:
            return np.zeros((0, int(np.prod(self.bins))), dtype=np.float32)

        feats = []
        for crop in crops_bgr:
            if crop is None or crop.size == 0:
                feats.append(np.zeros(int(np.prod(self.bins)), dtype=np.float32))
                continue

            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0, 1, 2], None, self.bins, [0, 180, 0, 256, 0, 256])
            hist = hist.flatten().astype(np.float32)
            norm = np.linalg.norm(hist) + 1e-8
            feats.append(hist / norm)

        return np.stack(feats, axis=0)


class DinoHFEmbedder:
    """DINOv3 embedder with automatic TensorRT fallback.
    
    Loads TensorRT compiled model (.ep) if available,
    otherwise falls back to HuggingFace transformers.
    """
    
    def __init__(self, model_dir: str, device: str = "auto", batch_size: int = 32):
        self.batch_size = batch_size
        self.image_size = 224  # DINOv3 expected input size
        self.model_dir = Path(model_dir) if model_dir else Path("/ros2_ws/src/cencosud-counter-vision/models/dino")
        
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        # Check for TensorRT model first
        trt_path = self.model_dir / "compiled_model.ep"
        if trt_path.exists() and _is_tensorrt_available() and self.device.type == "cuda":
            self._load_tensorrt(trt_path)
        else:
            self._load_huggingface()

    def _load_tensorrt(self, trt_path: Path):
        """Load TensorRT compiled model (like andina does)."""
        import torch_tensorrt
        
        self.processor = AutoImageProcessor.from_pretrained(self.model_dir, trust_remote_code=True)
        
        exported_program = torch_tensorrt.load(str(trt_path))
        self.model = exported_program.module().to(self.device)
        self.model.eval()
        self._use_tensorrt = True
        print(f"✓ TensorRT DINO model loaded: {trt_path}")
        
    def _load_huggingface(self):
        """Load standard HuggingFace model."""
        self.processor = AutoImageProcessor.from_pretrained(self.model_dir, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(self.model_dir, trust_remote_code=True).to(self.device)
        self.model.eval()
        self._use_tensorrt = False
        print(f"✓ HuggingFace DINO model loaded: {self.model_dir}")

    def embed(self, crops_bgr: List[np.ndarray]) -> np.ndarray:
        if len(crops_bgr) == 0:
            return np.zeros((0, 384), dtype=np.float32)

        all_embeddings = []
        for i in range(0, len(crops_bgr), self.batch_size):
            batch = crops_bgr[i : i + self.batch_size]
            
            # Preprocess images for DINOv3
            batch_tensors = []
            for crop in batch:
                if crop is None or crop.size == 0:
                    crop = np.zeros((8, 8, 3), dtype=np.uint8)
                # Convert BGR to RGB and resize to 224x224
                rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                rgb = cv2.resize(rgb, (self.image_size, self.image_size))
                # Normalize to [0, 1] then to standard ImageNet normalization
                tensor = torch.from_numpy(rgb).float() / 255.0
                # Use ImageNet normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3)
                std = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3)
                tensor = (tensor - mean) / std
                tensor = tensor.permute(2, 0, 1)  # HWC -> CHW
                batch_tensors.append(tensor)
            
            batch_tensor = torch.stack(batch_tensors).to(self.device)
            
            with torch.no_grad():
                if self._use_tensorrt:
                    # TensorRT returns BaseModelOutputWithPooling
                    outputs = self.model(pixel_values=batch_tensor)
                    cls_tokens = outputs.pooler_output  # [B, D]
                else:
                    # HuggingFace returns last_hidden_state
                    outputs = self.model(pixel_values=batch_tensor)
                    cls_tokens = outputs.last_hidden_state[:, 0, :]  # [B, D]
                
            emb_np = cls_tokens.detach().cpu().numpy().astype(np.float32)
            # L2 normalize
            norms = np.linalg.norm(emb_np, axis=1, keepdims=True) + 1e-8
            all_embeddings.append(emb_np / norms)

        return np.concatenate(all_embeddings, axis=0)


def create_embedder(kind: str, model_dir: str, device: str, batch_size: int):
    kind = (kind or "dino").strip().lower()
    if kind == "dino":
        return DinoHFEmbedder(model_dir=model_dir, device=device, batch_size=batch_size)
    if kind == "hist":
        return HistogramEmbedder()
    raise ValueError(f"Embedder no soportado: {kind}")

