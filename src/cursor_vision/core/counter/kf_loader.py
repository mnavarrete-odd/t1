import json
import re
from pathlib import Path
from typing import List

import cv2
import numpy as np

from .types import DetectionRaw, FrameData


class KFLoader:
    def __init__(self, kfs_dir: str):
        self.kfs_dir = Path(kfs_dir)
        self.data_dir = self.kfs_dir / "data"
        self.images_dir = self.kfs_dir / "images"
        self.depth_dir = self.kfs_dir / "depth"

    def _json_sort_key(self, json_path: Path, payload: dict) -> int:
        frame_index = payload.get("frame_index")
        if frame_index is not None:
            return int(frame_index)
        m = re.search(r"(\d+)", json_path.stem)
        return int(m.group(1)) if m else 0

    def _parse_area_bbox(self, payload: dict) -> np.ndarray | None:
        metrics = payload.get("metrics") or {}
        area_bbox = metrics.get("area_bbox")
        if area_bbox is None:
            area_bbox = metrics.get("area_bbox_raw")
        if area_bbox is None:
            return None
        if not isinstance(area_bbox, (list, tuple)) or len(area_bbox) != 4:
            return None
        try:
            bbox = np.array(area_bbox, dtype=np.float32)
        except Exception:
            return None
        if not np.all(np.isfinite(bbox)):
            return None
        if float(bbox[2]) <= 0.0 or float(bbox[3]) <= 0.0:
            return None
        return bbox

    def load(self, max_frames: int = 0) -> List[FrameData]:
        if not self.data_dir.exists():
            raise FileNotFoundError(f"No existe carpeta data: {self.data_dir}")
        if not self.images_dir.exists():
            raise FileNotFoundError(f"No existe carpeta images: {self.images_dir}")

        entries = []
        for json_path in sorted(self.data_dir.glob("*.json")):
            with open(json_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            entries.append((self._json_sort_key(json_path, payload), json_path, payload))

        entries.sort(key=lambda x: x[0])
        if max_frames > 0:
            entries = entries[:max_frames]

        frames: List[FrameData] = []
        for frame_idx_sorted, json_path, payload in entries:
            image_name = payload.get("image_name") or f"{json_path.stem}.jpg"
            image_path = self.images_dir / image_name
            if not image_path.exists():
                png_candidate = self.images_dir / f"{Path(image_name).stem}.png"
                if png_candidate.exists():
                    image_path = png_candidate
                else:
                    continue

            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                continue

            depth_path = self.depth_dir / f"{Path(image_name).stem}.npy"
            depth_map = None
            if depth_path.exists():
                try:
                    depth_map = np.load(str(depth_path))
                except Exception:
                    depth_map = None

            detections_raw = []
            for det in payload.get("detections", []):
                bbox = det.get("bbox")
                if not bbox or len(bbox) != 4:
                    continue
                class_id = det.get("class_id")
                class_name = det.get("class_name") or str(class_id)
                if class_id is None:
                    continue
                detections_raw.append(
                    DetectionRaw(
                        class_id=int(class_id),
                        class_name=str(class_name),
                        bbox_cxcywh=np.array(bbox, dtype=np.float32),
                        confidence=float(det.get("confidence", 0.0)),
                    )
                )

            frame_index = payload.get("frame_index")
            if frame_index is None:
                frame_index = frame_idx_sorted

            area_bbox_cxcywh = self._parse_area_bbox(payload)

            frames.append(
                FrameData(
                    frame_index=int(frame_index),
                    image_name=image_name,
                    image=image,
                    depth_map=depth_map,
                    area_bbox_cxcywh=area_bbox_cxcywh,
                    detections=detections_raw,
                )
            )

        return frames
