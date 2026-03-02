"""
TaskController: manages the lifecycle of tasks (ITEM_START / ITEM_END /
ORDER_END), coordinating the CameraPipelines accordingly.

Cycle:
  ITEM_START  -> activate pipelines, force initial KF, start debug writers
  (during)    -> Photographer produces KFs -> Counter updates
  ITEM_END    -> force final KF, deactivate pipelines, end debug writers
  ORDER_END   -> collect final results, full reset
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Dict, Optional

if TYPE_CHECKING:
    from lib.camera_pipeline import CameraPipeline
    from lib.debug_writer import DebugWriter

logger = logging.getLogger(__name__)


class TaskController:

    def __init__(
        self,
        camera_pipelines: Dict[str, CameraPipeline],
        debug_writers: Optional[Dict[str, DebugWriter]] = None,
    ):
        self.pipelines = camera_pipelines
        self.debug_writers = debug_writers or {}
        self.task_active = False
        self._current_context: Optional[dict] = None

        # Accumulated results per item within an order
        self._order_item_results: list[dict] = []

    # ------------------------------------------------------------------
    # Task lifecycle
    # ------------------------------------------------------------------

    def on_item_start(self, context: dict) -> None:
        """
        Called when ITEM_START is received.
        Activates all pipelines, forces an initial keyframe, and starts
        debug writers.
        """
        self.task_active = True
        self._current_context = context

        task_id = (
            f"{context.get('order_id', 'ORD')}_"
            f"{context.get('hu_id', 'HU')}_"
            f"{context.get('sku', 'SKU')}"
        )

        for cam_name, pipeline in self.pipelines.items():
            # Start debug writer for this task if enabled
            dw = self.debug_writers.get(cam_name)
            if dw and dw.enabled:
                dw.begin_task(f"{task_id}_{cam_name}")

            pipeline.activate()
            pipeline.force_keyframe("TASK-START")

        logger.info("Task started: %s", task_id)

    def on_item_end(self, context: dict) -> None:
        """
        Called when ITEM_END is received.
        Forces a final keyframe, deactivates pipelines, and collects
        per-item results.
        """
        item_results: Dict[str, dict] = {}

        for cam_name, pipeline in self.pipelines.items():
            pipeline.force_keyframe("TASK-END")
            pipeline.deactivate()

            item_results[cam_name] = pipeline.get_count_result()

            dw = self.debug_writers.get(cam_name)
            if dw and dw.enabled:
                dw.end_task()

        self.task_active = False

        self._order_item_results.append({
            "context": dict(context),
            "results": item_results,
        })

        logger.info(
            "Task ended. Results: %s",
            {k: v.get("running_units", 0) for k, v in item_results.items()},
        )

    def on_order_end(self) -> dict:
        """
        Called when ORDER_END is received.
        Collects all accumulated results, resets everything, and returns
        the final order summary.
        """
        final_results: Dict[str, dict] = {}
        for cam_name, pipeline in self.pipelines.items():
            final_results[cam_name] = pipeline.get_count_result()
            pipeline.reset()

        order_summary = {
            "final_counts": final_results,
            "items": list(self._order_item_results),
        }
        self._order_item_results.clear()
        self._current_context = None

        logger.info(
            "Order ended. Final counts: %s",
            {k: v.get("running_units", 0) for k, v in final_results.items()},
        )

        return order_summary
