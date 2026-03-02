from __future__ import annotations

from .base import KeyframeHandler


class AreaSetHandler(KeyframeHandler):
    def __init__(self) -> None:
        self._prev_area_ready = False
        self._prev_area_class: str | None = None

    def update(self, ctx):
        events = []
        if ctx.area_ready and ctx.area_state:
            if (not self._prev_area_ready) or (self._prev_area_class != ctx.area_state.class_name):
                events.append(
                    ctx.emit_event(
                        "KF-AREA-SET",
                        ctx.frame_index,
                        ctx.event_image,
                        ctx.metrics,
                        ctx.detections,
                    )
                )
        self._prev_area_ready = ctx.area_ready
        self._prev_area_class = ctx.area_state.class_name if ctx.area_state else None
        return events
