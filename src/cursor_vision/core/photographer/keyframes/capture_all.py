from __future__ import annotations

from .base import KeyframeHandler


class CaptureAllHandler(KeyframeHandler):
    def __init__(self, cfg) -> None:
        self.cfg = cfg

    def update(self, ctx):
        if not self.cfg.capture_all:
            return []
        return [
            ctx.emit_event(
                "frame",
                ctx.frame_index,
                ctx.event_image,
                ctx.metrics,
                ctx.detections,
            )
        ]
