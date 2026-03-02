"""
Unit tests for TaskController: verifies the task lifecycle
(item_start / item_end / order_end) with mocked CameraPipelines.
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Mock heavy deps that TaskController doesn't actually use directly
sys.modules.setdefault("cv2", MagicMock())
sys.modules.setdefault("cv_bridge", MagicMock())
sys.modules.setdefault("torch", MagicMock())
sys.modules.setdefault("ultralytics", MagicMock())
sys.modules.setdefault("rclpy", MagicMock())
sys.modules.setdefault("rclpy.node", MagicMock())
sys.modules.setdefault("rclpy.qos", MagicMock())
sys.modules.setdefault("message_filters", MagicMock())
sys.modules.setdefault("sensor_msgs", MagicMock())
sys.modules.setdefault("sensor_msgs.msg", MagicMock())

from lib.task_controller.task_controller import TaskController


def _make_mock_pipeline(camera_name: str = "cam_test") -> MagicMock:
    pipeline = MagicMock()
    pipeline.camera_name = camera_name
    pipeline.counter_bridge = MagicMock()
    pipeline.counter_bridge.running_units = 0.0
    pipeline.counter_bridge._frame_counter = 0
    pipeline.get_count_result.return_value = {
        "camera_name": camera_name,
        "running_units": 5.0,
        "frames_processed": 3,
    }
    pipeline.force_keyframe.return_value = None
    return pipeline


class TestTaskController:

    def test_item_start_activates_pipelines(self):
        p1 = _make_mock_pipeline("primary")
        p2 = _make_mock_pipeline("secondary")
        tc = TaskController({"primary": p1, "secondary": p2})

        ctx = {"order_id": "ORD-1", "hu_id": "HU-A", "sku": "SKU-X", "quantity": 5}
        tc.on_item_start(ctx)

        assert tc.task_active is True
        p1.activate.assert_called_once()
        p2.activate.assert_called_once()
        p1.force_keyframe.assert_called_once_with("TASK-START")
        p2.force_keyframe.assert_called_once_with("TASK-START")

    def test_item_end_deactivates_pipelines(self):
        p1 = _make_mock_pipeline("primary")
        tc = TaskController({"primary": p1})

        tc.on_item_start({"order_id": "O", "hu_id": "H", "sku": "S"})
        tc.on_item_end({"order_id": "O", "hu_id": "H", "sku": "S"})

        assert tc.task_active is False
        p1.deactivate.assert_called_once()
        p1.force_keyframe.assert_any_call("TASK-END")

    def test_item_end_collects_results(self):
        p1 = _make_mock_pipeline("primary")
        p1.get_count_result.return_value = {
            "camera_name": "primary",
            "running_units": 10.0,
            "frames_processed": 5,
        }
        tc = TaskController({"primary": p1})

        tc.on_item_start({"order_id": "O", "hu_id": "H", "sku": "S"})
        tc.on_item_end({"order_id": "O", "hu_id": "H", "sku": "S"})

        assert len(tc._order_item_results) == 1
        item_result = tc._order_item_results[0]
        assert item_result["results"]["primary"]["running_units"] == 10.0

    def test_order_end_resets_and_returns_summary(self):
        p1 = _make_mock_pipeline("primary")
        p2 = _make_mock_pipeline("secondary")
        tc = TaskController({"primary": p1, "secondary": p2})

        tc.on_item_start({"order_id": "O", "hu_id": "H", "sku": "S1"})
        tc.on_item_end({"order_id": "O", "hu_id": "H", "sku": "S1"})

        summary = tc.on_order_end()

        assert "final_counts" in summary
        assert "items" in summary
        assert len(summary["items"]) == 1
        p1.reset.assert_called_once()
        p2.reset.assert_called_once()
        assert len(tc._order_item_results) == 0

    def test_multiple_items_in_order(self):
        p1 = _make_mock_pipeline("primary")
        tc = TaskController({"primary": p1})

        for sku in ["SKU-A", "SKU-B", "SKU-C"]:
            tc.on_item_start({"order_id": "O", "hu_id": "H", "sku": sku})
            tc.on_item_end({"order_id": "O", "hu_id": "H", "sku": sku})

        assert len(tc._order_item_results) == 3

        summary = tc.on_order_end()
        assert len(summary["items"]) == 3
        assert len(tc._order_item_results) == 0

    def test_item_start_without_prior_order(self):
        """TaskController doesn't require ORDER_START -- it just activates."""
        p1 = _make_mock_pipeline("cam")
        tc = TaskController({"cam": p1})

        tc.on_item_start({"order_id": "O", "hu_id": "H", "sku": "S"})
        assert tc.task_active is True

    def test_debug_writers_called(self):
        p1 = _make_mock_pipeline("primary")
        dw = MagicMock()
        dw.enabled = True
        tc = TaskController({"primary": p1}, debug_writers={"primary": dw})

        tc.on_item_start({"order_id": "O", "hu_id": "H", "sku": "S"})
        dw.begin_task.assert_called_once()

        tc.on_item_end({"order_id": "O", "hu_id": "H", "sku": "S"})
        dw.end_task.assert_called_once()
