"""
CameraPipeline: per-camera orchestration of Detector -> Photographer -> Counter.

Each instance subscribes to one camera's RGB-D topics, runs YOLO detection,
feeds frames to its own Photographer instance, and sends keyframes to its
own CounterBridge for tracking and counting.
"""
from __future__ import annotations

import logging
import threading
import time
from collections import deque
from typing import Any, Callable, Optional, Sequence

import cv2
import numpy as np
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from message_filters import Subscriber, ApproximateTimeSynchronizer
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from core.detector_yolo import YOLODetector
from core.photographer.photographer import Photographer
from core.photographer.types import (
    DetectionData,
    FrameMetrics,
    KeyframeSaveRequest,
)
from core.counter.types import FrameResult

from lib.counter_bridge import CounterBridge
from lib.debug_writer import DebugWriter

logger = logging.getLogger(__name__)


def _yolo_results_to_detections(result) -> list[DetectionData]:
    """Convert a single YOLO result object to a list of DetectionData (xyxy)."""
    dets: list[DetectionData] = []
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return dets
    track_ids = None
    if getattr(boxes, "id", None) is not None:
        track_ids = boxes.id.cpu().numpy().astype(int).tolist()
    names = getattr(result, "names", {})
    for idx, box in enumerate(boxes):
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        class_name = names.get(cls, str(cls)) if isinstance(names, dict) else str(cls)
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()
        tracking_id = track_ids[idx] if track_ids is not None and idx < len(track_ids) else None
        dets.append(
            DetectionData(
                class_id=cls,
                class_name=class_name,
                bbox=(float(x1), float(y1), float(x2), float(y2)),
                confidence=conf,
                tracking_id=tracking_id,
            )
        )
    return dets


class CameraPipeline:
    """
    Full per-camera pipeline: subscribe -> detect -> photograph -> count.
    """

    def __init__(
        self,
        node: Node,
        camera_name: str,
        detector: YOLODetector,
        detector_lock: threading.Lock,
        photographer: Photographer,
        counter_bridge: CounterBridge,
        *,
        debug_writer: Optional[DebugWriter] = None,
        detection_fps: float = 10.0,
        detection_mode: str = "no_drop",
        sync_queue_size: int = 100,
        sync_max_delay: float = 0.3,
        max_queue_size: int = 200,
        on_count_result: Optional[Callable[[str, FrameResult], None]] = None,
    ):
        self.node = node
        self.camera_name = camera_name
        self.detector = detector
        self.detector_lock = detector_lock
        self.photographer = photographer
        self.counter_bridge = counter_bridge
        self.debug_writer = debug_writer
        self.on_count_result = on_count_result

        self.detection_fps = detection_fps
        self.detection_mode = detection_mode
        self.max_queue_size = max_queue_size

        self._active = False
        self._frame_idx = 0
        self._detection_period = 1.0 / detection_fps if detection_fps > 0 else 0.0
        self._last_detection_time = 0.0

        # Latest frame cache for force_keyframe
        self._latest_rgb: Optional[np.ndarray] = None
        self._latest_depth: Optional[np.ndarray] = None
        self._latest_detections: list[DetectionData] = []
        self._latest_metrics: Optional[FrameMetrics] = None
        self._latest_lock = threading.Lock()

        # Frame queue and processing thread
        self._frame_queue: deque = deque()
        self._queue_lock = threading.Lock()
        self._processing_thread: Optional[threading.Thread] = None
        self._running = False

        # CV bridge
        self._bridge = CvBridge()

        # ROS subscribers with time sync
        qos = qos_profile_sensor_data
        self._rgb_sub = Subscriber(
            node, Image, f"/{camera_name}/color/image_raw", qos_profile=qos
        )
        self._depth_sub = Subscriber(
            node, Image, f"/{camera_name}/depth/image_raw", qos_profile=qos
        )
        self._time_sync = ApproximateTimeSynchronizer(
            [self._rgb_sub, self._depth_sub], sync_queue_size, sync_max_delay
        )
        self._time_sync.registerCallback(self._on_synchronized_images)

        # Start processing thread
        self._start_processing_thread()

    # ------------------------------------------------------------------
    # Activation control
    # ------------------------------------------------------------------

    def activate(self) -> None:
        self._active = True

    def deactivate(self) -> None:
        self._active = False

    @property
    def active(self) -> bool:
        return self._active

    # ------------------------------------------------------------------
    # Force keyframe (bypass normal Photographer logic)
    # ------------------------------------------------------------------

    def force_keyframe(self, event_type: str) -> Optional[FrameResult]:
        """
        Capture an immediate keyframe using the latest available frame,
        run it through the Photographer's emit_event directly, and feed
        the result to the CounterBridge.

        Returns the FrameResult if successful, None otherwise.
        """
        with self._latest_lock:
            rgb = self._latest_rgb
            depth = self._latest_depth
            dets = list(self._latest_detections)
            metrics = self._latest_metrics

        if rgb is None:
            logger.warning(
                "[%s] force_keyframe(%s): no frame available yet",
                self.camera_name, event_type,
            )
            return None

        if metrics is None:
            h, w = rgb.shape[:2]
            metrics = FrameMetrics(
                frame_index=self._frame_idx,
                image_w=w,
                image_h=h,
                area_bbox=None,
                area_bbox_raw=None,
                area_class_name=None,
                area_confidence=None,
                area_stable_frames=0,
                count_in_area=0,
                coverage_ratio=0.0,
                movement_score=0.0,
                occlusion_ratio=0.0,
                has_person_near=False,
                class_counts={},
            )

        kf_event = self.photographer.emit_event(
            event_type=event_type,
            frame_index=self._frame_idx,
            image=rgb.copy(),
            metrics=metrics,
            detections=dets,
            original_image=rgb.copy(),
            original_depth=depth.copy() if depth is not None else None,
        )

        save_requests = list(self.photographer._pending_save_requests)
        self.photographer._pending_save_requests.clear()

        result = None
        for kf_req in save_requests:
            result = self.counter_bridge.process_keyframe(kf_req, depth)
            if self.debug_writer and self.debug_writer.enabled:
                self.debug_writer.save_keyframe(kf_req)
            if self.on_count_result:
                self.on_count_result(self.camera_name, result)

        return result

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset photographer state, counter, and frame index."""
        self.counter_bridge.reset()
        self._frame_idx = 0
        with self._latest_lock:
            self._latest_rgb = None
            self._latest_depth = None
            self._latest_detections = []
            self._latest_metrics = None

    def get_count_result(self) -> dict:
        """Return current counting summary for this camera."""
        return {
            "camera_name": self.camera_name,
            "running_units": self.counter_bridge.running_units,
            "frames_processed": self.counter_bridge._frame_counter,
        }

    # ------------------------------------------------------------------
    # ROS callback
    # ------------------------------------------------------------------

    def _on_synchronized_images(self, rgb_msg: Image, depth_msg: Image) -> None:
        current_time = time.time()

        # FPS throttling
        if self.detection_fps > 0:
            if current_time - self._last_detection_time < self._detection_period:
                return
            self._last_detection_time = current_time

        # Only queue frames when pipeline is active
        if not self._active:
            # Still convert and cache the latest frame for force_keyframe
            try:
                rgb_cv = self._bridge.imgmsg_to_cv2(rgb_msg, desired_encoding="bgr8")
                depth_np = self._bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
                with self._latest_lock:
                    self._latest_rgb = rgb_cv
                    self._latest_depth = depth_np
            except Exception:
                pass
            return

        try:
            rgb_cv = self._bridge.imgmsg_to_cv2(rgb_msg, desired_encoding="bgr8")
            depth_np = self._bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
        except Exception as exc:
            logger.error("[%s] Image conversion failed: %s", self.camera_name, exc)
            return

        frame_data = {
            "rgb": rgb_cv,
            "depth": depth_np,
            "timestamp": current_time,
        }

        with self._queue_lock:
            if self.detection_mode == "no_drop":
                if len(self._frame_queue) >= self.max_queue_size:
                    self._frame_queue.popleft()
                self._frame_queue.append(frame_data)
            else:
                self._frame_queue.clear()
                self._frame_queue.append(frame_data)

    # ------------------------------------------------------------------
    # Processing thread
    # ------------------------------------------------------------------

    def _start_processing_thread(self) -> None:
        if self._processing_thread is not None and self._processing_thread.is_alive():
            return
        self._running = True
        self._processing_thread = threading.Thread(
            target=self._processing_loop, daemon=True,
            name=f"CameraPipeline_{self.camera_name}",
        )
        self._processing_thread.start()

    def _processing_loop(self) -> None:
        while self._running:
            frame_data = None
            with self._queue_lock:
                if self._frame_queue:
                    frame_data = self._frame_queue.popleft()

            if frame_data is None:
                time.sleep(0.001)
                continue

            try:
                self._process_frame(frame_data["rgb"], frame_data["depth"])
            except Exception as exc:
                logger.error("[%s] Frame processing error: %s", self.camera_name, exc)

    def _process_frame(self, rgb: np.ndarray, depth: np.ndarray) -> None:
        """Full pipeline for one frame: detect -> photograph -> count."""
        # 1. Detect (serialised via lock for shared GPU model)
        with self.detector_lock:
            results, _ = self.detector.detect(rgb)

        detections = _yolo_results_to_detections(results[0]) if results else []

        # 2. Photographer update
        metrics, events, save_requests = self.photographer.update(
            frame_index=self._frame_idx,
            image=rgb,
            detections=detections,
            depth_image=depth,
        )

        # Cache latest state for force_keyframe
        with self._latest_lock:
            self._latest_rgb = rgb
            self._latest_depth = depth
            self._latest_detections = detections
            self._latest_metrics = metrics

        # 3. For each keyframe produced, run Counter
        for kf_req in save_requests:
            result = self.counter_bridge.process_keyframe(kf_req, depth)

            if self.debug_writer and self.debug_writer.enabled:
                self.debug_writer.save_keyframe(kf_req)

            if self.on_count_result:
                self.on_count_result(self.camera_name, result)

        self._frame_idx += 1

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def stop(self) -> None:
        self._running = False
        self._active = False
        if self._processing_thread is not None:
            self._processing_thread.join(timeout=2.0)
        with self._queue_lock:
            self._frame_queue.clear()
