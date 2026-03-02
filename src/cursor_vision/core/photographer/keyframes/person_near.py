from __future__ import annotations

from .base import KeyframeHandler


class PersonNearHandler(KeyframeHandler):
    def update(self, ctx):
        if not ctx.metrics.has_person_near:
            return []
        person_image = ctx.render_labels(ctx.person_dets)
        return [
            ctx.emit_event(
                "KF-PERSON-NEAR",
                ctx.frame_index,
                person_image,
                ctx.metrics,
                ctx.detections,
            )
        ]
