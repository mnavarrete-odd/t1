"""
Unit tests for CameraPipeline: verifies force_keyframe logic,
activation/deactivation, and frame processing with mocked dependencies.
"""
import sys
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Mock heavy deps that require GPU/packages not available in test env
_mock_modules = [
    "ultralytics", "torch", "torchvision", "transformers",
    "cv_bridge", "rclpy", "rclpy.node", "rclpy.qos",
    "message_filters", "sensor_msgs", "sensor_msgs.msg",
]
for _m in _mock_modules:
    if _m not in sys.modules:
        sys.modules[_m] = MagicMock()

from core.photographer.types import (
    DetectionData,
    FrameMetrics,
    KeyframeSaveRequest,
)
from core.counter.types import FrameResult


class TestCameraPipelineUnit:
    """
    Tests CameraPipeline methods in isolation using mocks
    (no ROS or GPU required).
    """

    def _make_pipeline(self):
        """Build a CameraPipeline with all ROS/GPU deps mocked out."""
        from lib.camera_pipeline.camera_pipeline import CameraPipeline

        with patch.object(CameraPipeline, "__init__", lambda self, *a, **kw: None):
            pipeline = CameraPipeline.__new__(CameraPipeline)

        pipeline.camera_name = "test_cam"
        pipeline.detector = MagicMock()
        pipeline.detector_lock = threading.Lock()
        pipeline.photographer = MagicMock()
        pipeline.counter_bridge = MagicMock()
        pipeline.debug_writer = None
        pipeline.on_count_result = MagicMock()

        pipeline._active = False
        pipeline._frame_idx = 0
        pipeline._latest_rgb = None
        pipeline._latest_depth = None
        pipeline._latest_detections = []
        pipeline._latest_metrics = None
        pipeline._latest_lock = threading.Lock()

        pipeline._frame_queue = []
        pipeline._queue_lock = threading.Lock()
        pipeline._running = False
        pipeline._processing_thread = None
        pipeline.node = MagicMock()
        pipeline.detection_fps = 10.0
        pipeline.detection_mode = "no_drop"
        pipeline.max_queue_size = 200
        pipeline._detection_period = 0.1
        pipeline._last_detection_time = 0.0

        return pipeline

    def test_activate_deactivate(self):
        pipeline = self._make_pipeline()

        assert pipeline.active is False
        pipeline.activate()
        assert pipeline.active is True
        pipeline.deactivate()
        assert pipeline.active is False

    def test_force_keyframe_no_frame(self):
        """force_keyframe returns None when no frame is cached."""
        pipeline = self._make_pipeline()
        result = pipeline.force_keyframe("TASK-START")
        assert result is None

    def test_force_keyframe_with_cached_frame(self):
        """force_keyframe emits event and processes through counter_bridge."""
        pipeline = self._make_pipeline()

        rgb = np.zeros((480, 640, 3), dtype=np.uint8)
        depth = np.ones((480, 640), dtype=np.float32) * 2.0

        pipeline._latest_rgb = rgb
        pipeline._latest_depth = depth
        pipeline._latest_detections = [
            DetectionData(
                class_id=0, class_name="cajas",
                bbox=(10.0, 20.0, 100.0, 150.0), confidence=0.9,
            )
        ]
        pipeline._latest_metrics = FrameMetrics(
            frame_index=0, image_w=640, image_h=480,
            area_bbox=(50, 50, 300, 400), area_bbox_raw=(50, 50, 300, 400),
            area_class_name="area_de_trabajo_carro",
            area_confidence=0.9, area_stable_frames=5,
            count_in_area=1, coverage_ratio=0.3,
            movement_score=0.0, occlusion_ratio=0.0,
            has_person_near=False, class_counts={"cajas": 1},
        )

        mock_kf_req = MagicMock(spec=KeyframeSaveRequest)
        pipeline.photographer._pending_save_requests = [mock_kf_req]

        mock_result = FrameResult(frame_index=0)
        mock_result.running_units = 3.0
        pipeline.counter_bridge.process_keyframe.return_value = mock_result

        result = pipeline.force_keyframe("TASK-START")

        pipeline.photographer.emit_event.assert_called_once()
        call_kwargs = pipeline.photographer.emit_event.call_args
        assert call_kwargs[1].get("event_type", call_kwargs[0][0] if call_kwargs[0] else None) == "TASK-START" or call_kwargs.kwargs.get("event_type") == "TASK-START"

        pipeline.counter_bridge.process_keyframe.assert_called_once_with(mock_kf_req, depth)
        pipeline.on_count_result.assert_called_once_with("test_cam", mock_result)

        assert result is mock_result

    def test_reset_clears_state(self):
        pipeline = self._make_pipeline()
        pipeline._frame_idx = 42
        pipeline._latest_rgb = np.zeros((1, 1, 3), dtype=np.uint8)

        pipeline.reset()

        assert pipeline._frame_idx == 0
        assert pipeline._latest_rgb is None
        pipeline.counter_bridge.reset.assert_called_once()

    def test_get_count_result(self):
        pipeline = self._make_pipeline()
        pipeline.counter_bridge.running_units = 7.5
        pipeline.counter_bridge._frame_counter = 4

        result = pipeline.get_count_result()

        assert result["camera_name"] == "test_cam"
        assert result["running_units"] == 7.5
        assert result["frames_processed"] == 4


class TestYoloResultsToDetections:
    """Tests for the YOLO-to-DetectionData conversion function."""

    def test_empty_result(self):
        from lib.camera_pipeline.camera_pipeline import _yolo_results_to_detections

        mock_result = MagicMock()
        mock_result.boxes = None
        assert _yolo_results_to_detections(mock_result) == []

    def test_single_detection(self):
        from lib.camera_pipeline.camera_pipeline import _yolo_results_to_detections

        mock_box = MagicMock()
        mock_box.cls = [MagicMock(__getitem__=lambda self, i: 0)]
        mock_box.cls[0].__int__ = lambda self: 0
        mock_box.conf = [MagicMock(__getitem__=lambda self, i: 0.95)]
        mock_box.conf[0].__float__ = lambda self: 0.95

        xyxy_row = MagicMock()
        xyxy_row.cpu.return_value = xyxy_row
        xyxy_row.numpy.return_value = MagicMock(
            tolist=MagicMock(return_value=[10.0, 20.0, 100.0, 200.0])
        )
        mock_box.xyxy = [xyxy_row]

        mock_boxes = MagicMock()
        mock_boxes.__iter__ = MagicMock(return_value=iter([mock_box]))
        mock_boxes.__len__ = MagicMock(return_value=1)
        mock_boxes.id = None

        mock_result = MagicMock()
        mock_result.boxes = mock_boxes
        mock_result.names = {0: "cajas"}

        dets = _yolo_results_to_detections(mock_result)
        assert len(dets) == 1
        assert dets[0].class_name == "cajas"
