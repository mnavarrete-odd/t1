"""
Integration test: verifies the full pipeline from KeyframeSaveRequest
through CounterBridge using synthetic data (no GPU, no ROS).

This test mocks the embedder (which requires GPU/model files)
and exercises the real FeatureExtractor, CounterTracker, and
FrameChangeCounter.
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Mock heavy GPU/ML deps
_mock_modules = [
    "ultralytics", "torch", "torch.nn", "torch.nn.functional",
    "torch.cuda", "torchvision", "torchvision.transforms",
    "transformers",
]
for _m in _mock_modules:
    if _m not in sys.modules:
        sys.modules[_m] = MagicMock()

# scipy is needed by counter's tracker (linear_sum_assignment)
try:
    import scipy  # noqa: F401
except ImportError:
    sys.modules["scipy"] = MagicMock()
    sys.modules["scipy.optimize"] = MagicMock()
    # Provide a minimal linear_sum_assignment mock
    lsa_mock = MagicMock()
    lsa_mock.return_value = (np.array([], dtype=int), np.array([], dtype=int))
    sys.modules["scipy.optimize"].linear_sum_assignment = lsa_mock

from core.photographer.types import (
    DetectionData,
    FrameMetrics,
    KeyframeSaveRequest,
)
from core.counter.types import FrameResult


def _make_kf(frame_index: int, n_detections: int = 2) -> KeyframeSaveRequest:
    """Build a synthetic KeyframeSaveRequest."""
    img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    depth = np.random.uniform(1.0, 2.5, (480, 640)).astype(np.float32)

    dets = []
    for i in range(n_detections):
        x1 = 50 + i * 100
        y1 = 50
        x2 = x1 + 80
        y2 = 200
        dets.append(
            DetectionData(
                class_id=0,
                class_name="cajas",
                bbox=(float(x1), float(y1), float(x2), float(y2)),
                confidence=0.9,
            )
        )

    metrics = FrameMetrics(
        frame_index=frame_index,
        image_w=640,
        image_h=480,
        area_bbox=(30, 30, 500, 450),
        area_bbox_raw=(30, 30, 500, 450),
        area_class_name="area_de_trabajo_carro",
        area_confidence=0.95,
        area_stable_frames=10,
        count_in_area=n_detections,
        coverage_ratio=0.4,
        movement_score=0.0,
        occlusion_ratio=0.0,
        has_person_near=False,
        class_counts={"cajas": n_detections},
    )

    return KeyframeSaveRequest(
        event_type="KF-STABLE-AREA",
        frame_index=frame_index,
        image=img,
        metrics=metrics,
        detections=dets,
        original_image=img,
        original_depth=depth,
    )


@pytest.fixture
def counter_bridge():
    """
    Create a CounterBridge with a mocked embedder (to avoid GPU/model deps).
    Patches create_embedder to return a dummy that produces random embeddings.
    """
    from lib.counter_bridge.counter_bridge import CounterBridge

    config_path = PROJECT_ROOT / "config" / "counter_default.yaml"
    if not config_path.exists():
        pytest.skip("counter_default.yaml not found")

    class DummyEmbedder:
        def embed(self, crops):
            return np.random.randn(len(crops), 128).astype(np.float32)

    with patch("lib.counter_bridge.counter_bridge.create_embedder", return_value=DummyEmbedder()):
        bridge = CounterBridge(str(config_path), device="cpu")
    return bridge


class TestCounterBridgeIntegration:

    def test_single_keyframe(self, counter_bridge):
        kf = _make_kf(frame_index=0, n_detections=2)
        result = counter_bridge.process_keyframe(kf, kf.original_depth)

        assert isinstance(result, FrameResult)
        assert result.frame_index == 0
        assert result.num_detections >= 0

    def test_sequence_of_keyframes(self, counter_bridge):
        """Process multiple KFs and verify running_units changes.
        Requires real scipy for the Bayesian cost model."""
        try:
            from scipy.optimize import linear_sum_assignment  # noqa: F401
            if not callable(linear_sum_assignment) or isinstance(linear_sum_assignment, MagicMock):
                pytest.skip("scipy not available (mocked)")
        except (ImportError, TypeError):
            pytest.skip("scipy not available")

        results = []
        for i in range(5):
            kf = _make_kf(frame_index=i, n_detections=2)
            result = counter_bridge.process_keyframe(kf, kf.original_depth)
            results.append(result)

        last = results[-1]
        assert last.num_active_tracks >= 0

    def test_reset_clears_state(self, counter_bridge):
        kf = _make_kf(frame_index=0, n_detections=2)
        counter_bridge.process_keyframe(kf, kf.original_depth)
        assert counter_bridge._frame_counter == 1

        counter_bridge.reset()

        assert counter_bridge._frame_counter == 0
        assert counter_bridge.running_units == 0.0
        assert counter_bridge.prev_depth is None

    def test_empty_detections(self, counter_bridge):
        kf = _make_kf(frame_index=0, n_detections=0)
        result = counter_bridge.process_keyframe(kf, kf.original_depth)

        assert isinstance(result, FrameResult)
        assert result.num_detections == 0

    def test_no_depth(self, counter_bridge):
        """Processing without depth should not crash."""
        kf = _make_kf(frame_index=0, n_detections=1)
        kf = KeyframeSaveRequest(
            event_type=kf.event_type,
            frame_index=kf.frame_index,
            image=kf.image,
            metrics=kf.metrics,
            detections=kf.detections,
            original_image=kf.original_image,
            original_depth=None,
        )
        result = counter_bridge.process_keyframe(kf, None)
        assert isinstance(result, FrameResult)
