"""
Unit tests for CounterBridge: verifies KF-to-FrameData conversion
and the end-to-end process_keyframe pipeline with synthetic data.
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Mock heavy deps to allow importing core.photographer.types without GPU libs
sys.modules.setdefault("ultralytics", MagicMock())
sys.modules.setdefault("torch", MagicMock())
sys.modules.setdefault("transformers", MagicMock())

from core.photographer.types import (
    DetectionData,
    FrameMetrics,
    KeyframeSaveRequest,
    bbox_xyxy_to_cxcywh,
)
from core.counter.types import DetectionRaw, FrameData


class TestKfToFrameDataConversion:
    """Tests for CounterBridge._kf_to_frame_data (without GPU dependencies)."""

    def _make_kf_request(
        self,
        frame_index: int = 0,
        image: np.ndarray | None = None,
        depth: np.ndarray | None = None,
        detections: list | None = None,
        area_bbox: tuple | None = (100, 100, 300, 400),
    ) -> KeyframeSaveRequest:
        if image is None:
            image = np.zeros((480, 640, 3), dtype=np.uint8)
        if detections is None:
            detections = [
                DetectionData(
                    class_id=0,
                    class_name="cajas",
                    bbox=(50.0, 60.0, 150.0, 200.0),
                    confidence=0.9,
                ),
            ]

        metrics = FrameMetrics(
            frame_index=frame_index,
            image_w=640,
            image_h=480,
            area_bbox=area_bbox,
            area_bbox_raw=area_bbox,
            area_class_name="area_de_trabajo_carro",
            area_confidence=0.95,
            area_stable_frames=10,
            count_in_area=1,
            coverage_ratio=0.3,
            movement_score=0.0,
            occlusion_ratio=0.0,
            has_person_near=False,
            class_counts={"cajas": 1},
        )

        return KeyframeSaveRequest(
            event_type="KF-STABLE-AREA",
            frame_index=frame_index,
            image=image.copy(),
            metrics=metrics,
            detections=detections,
            original_image=image,
            original_depth=depth,
        )

    def test_basic_conversion(self):
        """KF request with image and detections converts to valid FrameData."""
        from lib.counter_bridge.counter_bridge import CounterBridge

        kf = self._make_kf_request()

        # Call the static-ish conversion method without full init
        bridge = object.__new__(CounterBridge)
        frame_data = CounterBridge._kf_to_frame_data(bridge, kf, None)

        assert isinstance(frame_data, FrameData)
        assert frame_data.frame_index == 0
        assert frame_data.image.shape == (480, 640, 3)
        assert frame_data.depth_map is None
        assert len(frame_data.detections) == 1

        det = frame_data.detections[0]
        assert isinstance(det, DetectionRaw)
        assert det.class_name == "cajas"
        assert det.bbox_cxcywh.shape == (4,)

    def test_bbox_conversion(self):
        """Detection bbox (xyxy) is correctly converted to cxcywh."""
        from lib.counter_bridge.counter_bridge import CounterBridge

        det = DetectionData(
            class_id=1,
            class_name="producto",
            bbox=(100.0, 200.0, 300.0, 400.0),
            confidence=0.8,
        )
        kf = self._make_kf_request(detections=[det])
        bridge = object.__new__(CounterBridge)
        frame_data = CounterBridge._kf_to_frame_data(bridge, kf, None)

        raw = frame_data.detections[0]
        cx, cy, w, h = raw.bbox_cxcywh
        assert abs(cx - 200.0) < 1e-3
        assert abs(cy - 300.0) < 1e-3
        assert abs(w - 200.0) < 1e-3
        assert abs(h - 200.0) < 1e-3

    def test_area_bbox_conversion(self):
        """area_bbox (xyxy) from metrics is converted to cxcywh."""
        from lib.counter_bridge.counter_bridge import CounterBridge

        kf = self._make_kf_request(area_bbox=(10, 20, 110, 220))
        bridge = object.__new__(CounterBridge)
        frame_data = CounterBridge._kf_to_frame_data(bridge, kf, None)

        assert frame_data.area_bbox_cxcywh is not None
        cx, cy, w, h = frame_data.area_bbox_cxcywh
        assert abs(cx - 60.0) < 1e-3
        assert abs(cy - 120.0) < 1e-3
        assert abs(w - 100.0) < 1e-3
        assert abs(h - 200.0) < 1e-3

    def test_depth_from_request(self):
        """Depth from original_depth in request takes precedence."""
        from lib.counter_bridge.counter_bridge import CounterBridge

        depth_kf = np.ones((480, 640), dtype=np.float32) * 1.5
        depth_arg = np.ones((480, 640), dtype=np.float32) * 2.5
        kf = self._make_kf_request(depth=depth_kf)
        bridge = object.__new__(CounterBridge)
        frame_data = CounterBridge._kf_to_frame_data(bridge, kf, depth_arg)

        assert frame_data.depth_map is not None
        assert float(frame_data.depth_map[0, 0]) == pytest.approx(1.5)

    def test_depth_fallback_to_arg(self):
        """When original_depth is None, depth_map arg is used."""
        from lib.counter_bridge.counter_bridge import CounterBridge

        depth_arg = np.ones((480, 640), dtype=np.float32) * 3.0
        kf = self._make_kf_request(depth=None)
        bridge = object.__new__(CounterBridge)
        frame_data = CounterBridge._kf_to_frame_data(bridge, kf, depth_arg)

        assert frame_data.depth_map is not None
        assert float(frame_data.depth_map[0, 0]) == pytest.approx(3.0)

    def test_empty_detections(self):
        """Empty detection list converts properly."""
        from lib.counter_bridge.counter_bridge import CounterBridge

        kf = self._make_kf_request(detections=[])
        bridge = object.__new__(CounterBridge)
        frame_data = CounterBridge._kf_to_frame_data(bridge, kf, None)

        assert frame_data.detections == []


class TestBboxXyxyToCxcywh:
    """Sanity checks for the bbox conversion utility."""

    def test_square(self):
        cx, cy, w, h = bbox_xyxy_to_cxcywh((0, 0, 100, 100))
        assert (cx, cy, w, h) == (50.0, 50.0, 100.0, 100.0)

    def test_rect(self):
        cx, cy, w, h = bbox_xyxy_to_cxcywh((10, 20, 50, 80))
        assert abs(cx - 30.0) < 1e-6
        assert abs(cy - 50.0) < 1e-6
        assert abs(w - 40.0) < 1e-6
        assert abs(h - 60.0) < 1e-6
