"""
DebugWriter: optional persistence of keyframes and counter tracking frames
for debugging purposes, controlled by runtime flags.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence

import cv2
import numpy as np

from core.counter.types import FrameResult, PreparedDetection
from core.counter.visualize import draw_tracking_frame, build_video
from core.photographer.types import KeyframeSaveRequest, bbox_xyxy_to_cxcywh

logger = logging.getLogger(__name__)


class DebugWriter:
    """
    Saves keyframe images, depth maps, metadata JSONs, and optionally
    accumulates tracking visualisation frames to produce a debug video
    at the end of each task.
    """

    def __init__(
        self,
        output_dir: str,
        save_keyframes: bool = False,
        save_counter_frames: bool = False,
        save_counter_video: bool = False,
        video_fps: int = 4,
    ):
        self.output_dir = Path(output_dir)
        self.save_keyframes = save_keyframes
        self.save_counter_frames = save_counter_frames
        self.save_counter_video = save_counter_video
        self.video_fps = video_fps

        self._task_id: Optional[str] = None
        self._task_dir: Optional[Path] = None
        self._tracking_frames_dir: Optional[Path] = None
        self._kf_count = 0

    @property
    def enabled(self) -> bool:
        return self.save_keyframes or self.save_counter_frames or self.save_counter_video

    def begin_task(self, task_id: str) -> None:
        """Called when a new task (ITEM_START) begins."""
        self._task_id = task_id
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._task_dir = self.output_dir / f"{task_id}_{ts}"
        self._kf_count = 0

        if self.save_keyframes:
            (self._task_dir / "images").mkdir(parents=True, exist_ok=True)
            (self._task_dir / "data").mkdir(parents=True, exist_ok=True)
            (self._task_dir / "depth").mkdir(parents=True, exist_ok=True)

        if self.save_counter_frames or self.save_counter_video:
            self._tracking_frames_dir = self._task_dir / "tracking_frames"
            self._tracking_frames_dir.mkdir(parents=True, exist_ok=True)

    def save_keyframe(self, kf_request: KeyframeSaveRequest) -> None:
        """Save a keyframe (image + JSON + depth) to disk."""
        if not self.save_keyframes or self._task_dir is None:
            return

        stem = f"KF_{self._kf_count:04d}"
        self._kf_count += 1

        # Image
        image = kf_request.original_image
        if image is None:
            image = kf_request.image
        if isinstance(image, np.ndarray):
            img_path = self._task_dir / "images" / f"{stem}.jpg"
            cv2.imwrite(str(img_path), image)

        # Depth
        depth = kf_request.original_depth
        if isinstance(depth, np.ndarray):
            depth_path = self._task_dir / "depth" / f"{stem}.npy"
            np.save(str(depth_path), depth)

        # Metadata JSON
        meta = {
            "event_type": kf_request.event_type,
            "frame_index": kf_request.frame_index,
            "metrics": kf_request.metrics.to_dict() if kf_request.metrics else None,
            "detections": [
                {
                    "class_id": d.class_id,
                    "class_name": d.class_name,
                    "bbox_cxcywh": list(bbox_xyxy_to_cxcywh(d.bbox)),
                    "confidence": d.confidence,
                }
                for d in (kf_request.detections or [])
            ],
        }
        json_path = self._task_dir / "data" / f"{stem}.json"
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=True)

    def save_tracking_frame(
        self,
        frame_data,
        prepared: list[PreparedDetection],
        result: FrameResult,
    ) -> None:
        """Save a counter tracking visualisation frame for the debug video."""
        if (
            not self.save_counter_frames
            and not self.save_counter_video
        ) or self._tracking_frames_dir is None:
            return

        out_path = (
            self._tracking_frames_dir
            / f"frame_{result.frame_index:06d}.jpg"
        )
        try:
            draw_tracking_frame(frame_data, prepared, result, str(out_path))
        except Exception as exc:
            logger.warning("Failed to draw tracking frame: %s", exc)

    def end_task(self) -> None:
        """Called when a task ends (ITEM_END). Builds debug video if enabled."""
        if self.save_counter_video and self._tracking_frames_dir is not None:
            video_path = self._task_dir / "tracking.mp4"
            try:
                build_video(
                    frames_dir=str(self._tracking_frames_dir),
                    out_video_path=str(video_path),
                    fps=self.video_fps,
                )
            except Exception as exc:
                logger.warning("Failed to build tracking video: %s", exc)

        self._task_id = None
        self._task_dir = None
        self._tracking_frames_dir = None
        self._kf_count = 0
