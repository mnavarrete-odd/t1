from __future__ import annotations

from pathlib import Path
from typing import Iterable, Literal, Mapping, Sequence, overload
import math

import cv2

from .area import AreaSelector
from .config import PhotographerConfig
from .context import FrameContext
from .render import LabelRenderer
from .routing import build_keyframe_signal
from .types import (
    DetectionData,
    FrameMetrics,
    KeyframeEvent,
    KeyframeSaveRequest,
    KeyframeSignal,
)
from .utils import (
    bbox_center,
    boxes_in_area,
    coerce_detection,
    coverage_ratio,
    expand_bbox,
    movement_score,
    occlusion_ratio_by_body,
    person_near,
)
from .keyframes import (
    AreaEmptyHandler,
    AreaSetHandler,
    CaptureAllHandler,
    OcclusionHandler,
    PersonNearHandler,
    ProductInHandHandler,
    StableReconfirmHandler,
    StableAreaHandler,
)


class Photographer:
    def __init__(self, config: PhotographerConfig) -> None:
        self.cfg = config
        self.outdir = Path(config.outdir)
        self._pending_save_requests: list[KeyframeSaveRequest] = []
        self._prev_boxes: list[DetectionData] = []
        self._area_selector = AreaSelector(config)
        self._label_renderer = LabelRenderer()
        self._handlers = [
            CaptureAllHandler(config),
            AreaSetHandler(),
            StableAreaHandler(config),
            AreaEmptyHandler(config),
            ProductInHandHandler(config),
            OcclusionHandler(config),
            StableReconfirmHandler(config),
        ]
        if self.cfg.person_near_enabled:
            self._handlers.insert(4, PersonNearHandler())

    def emit_event(
        self,
        event_type: str,
        frame_index: int,
        image: "cv2.typing.MatLike",
        metrics: FrameMetrics,
        detections: Sequence[DetectionData],
        filename_tag: str | None = None,
        folder_tag: str | None = None,
        filename_override: str | None = None,
        original_image_path: str | Path | None = None,
        original_image: "cv2.typing.MatLike" | None = None,
        original_depth: object | None = None,
        kf_test_skip: bool = False,
    ) -> KeyframeEvent:
        request = KeyframeSaveRequest(
            event_type=event_type,
            frame_index=frame_index,
            image=image,
            metrics=metrics,
            detections=list(detections),
            filename_tag=filename_tag,
            folder_tag=folder_tag,
            filename_override=filename_override,
            original_image_path=original_image_path,
            original_image=original_image,
            original_depth=original_depth,
            kf_test_skip=kf_test_skip,
        )
        self._pending_save_requests.append(request)
        image_path = self.outdir / request.resolved_folder_name() / request.resolved_filename()
        event = KeyframeEvent(
            event_type=event_type,
            frame_index=frame_index,
            image_path=str(image_path),
            metrics=metrics,
        )
        return event

    @overload
    def update(
        self,
        frame_index: int,
        image: "cv2.typing.MatLike",
        detections: Iterable[DetectionData | Mapping[str, object]],
        *,
        original_image_path: str | Path | None = None,
        depth_image: object | None = None,
        include_signals: Literal[False] = False,
    ) -> tuple[FrameMetrics, list[KeyframeEvent], list[KeyframeSaveRequest]]: ...

    @overload
    def update(
        self,
        frame_index: int,
        image: "cv2.typing.MatLike",
        detections: Iterable[DetectionData | Mapping[str, object]],
        *,
        original_image_path: str | Path | None = None,
        depth_image: object | None = None,
        include_signals: Literal[True],
    ) -> tuple[
        FrameMetrics,
        list[KeyframeEvent],
        list[KeyframeSaveRequest],
        list[KeyframeSignal],
    ]: ...

    def update(
        self,
        frame_index: int,
        image: "cv2.typing.MatLike",
        detections: Iterable[DetectionData | Mapping[str, object]],
        *,
        original_image_path: str | Path | None = None,
        depth_image: object | None = None,
        include_signals: bool = False,
    ) -> (
        tuple[FrameMetrics, list[KeyframeEvent], list[KeyframeSaveRequest]]
        | tuple[
            FrameMetrics,
            list[KeyframeEvent],
            list[KeyframeSaveRequest],
            list[KeyframeSignal],
        ]
    ):
        self._pending_save_requests.clear()
        dets = [coerce_detection(d) for d in detections]
        h, w = image.shape[:2]

        class_counts: dict[str, int] = {}
        for d in dets:
            class_counts[d.class_name] = class_counts.get(d.class_name, 0) + 1

        area_state, area_ready = self._area_selector.update(dets, frame_index, w, h)
        area_bbox_raw = area_state.bbox if area_ready and area_state else None
        area_bbox = (
            expand_bbox(area_bbox_raw, self.cfg.area_expand_ratio, w, h)
            if area_bbox_raw
            else None
        )

        boxes_in_area_list = boxes_in_area(dets, area_bbox, self.cfg.box_classes)
        if self.cfg.movement_by_area and area_bbox:
            box_dets = list(boxes_in_area_list)
        else:
            box_dets = [d for d in dets if d.class_name in self.cfg.box_classes]
        person_dets = [d for d in dets if d.class_name in self.cfg.person_classes]
        hand_dets = [d for d in dets if d.class_name in self.cfg.hand_classes]
        occluder_dets = person_dets

        count_in_area = len(boxes_in_area_list)
        coverage = coverage_ratio(area_bbox, boxes_in_area_list)

        occlusion_ratio = occlusion_ratio_by_body(area_bbox, occluder_dets)
        has_person_near = person_near(area_bbox, person_dets, self.cfg.person_dist_px)
        movement = movement_score(box_dets, self._prev_boxes, w, h)

        metrics = FrameMetrics(
            frame_index=frame_index,
            image_w=w,
            image_h=h,
            area_bbox=area_bbox,
            area_bbox_raw=area_bbox_raw,
            area_class_name=area_state.class_name if area_state else None,
            area_confidence=area_state.confidence if area_state else None,
            area_stable_frames=area_state.stable_frames if area_state else 0,
            count_in_area=count_in_area,
            coverage_ratio=coverage,
            movement_score=movement,
            occlusion_ratio=occlusion_ratio,
            has_person_near=has_person_near,
            class_counts=class_counts,
        )

        def render_labels(extra_detections: Sequence[DetectionData] | None = None):
            return self._label_renderer.render(
                image,
                area_bbox,
                area_state.class_name if area_state else None,
                extra_detections=extra_detections,
            )

        event_image = render_labels()

        if area_bbox:
            ax, ay = bbox_center(area_bbox)
            person_with_dist: list[DetectionData] = []
            for d in person_dets:
                px, py = bbox_center(d.bbox)
                dist = math.hypot(px - ax, py - ay)
                person_with_dist.append(
                    type(d)(
                        class_id=d.class_id,
                        class_name=d.class_name,
                        bbox=d.bbox,
                        confidence=d.confidence,
                        extra_data={**(d.extra_data or {}), "dist_px": dist},
                        tracking_id=d.tracking_id,
                    )
                )
            person_dets = person_with_dist

        _raw_image = image
        hand_near_dets: list[DetectionData] = []
        if area_bbox:
            for d in hand_dets:
                hx, hy = bbox_center(d.bbox)
                dist = math.hypot(hx - ax, hy - ay)
                if dist <= self.cfg.hand_dist_px:
                    hand_near_dets.append(
                        type(d)(
                            class_id=d.class_id,
                            class_name=d.class_name,
                            bbox=d.bbox,
                            confidence=d.confidence,
                            extra_data={**(d.extra_data or {}), "dist_px": dist},
                            tracking_id=d.tracking_id,
                        )
                    )

        def emit_event(
            event_type: str,
            frame_index: int,
            image: "cv2.typing.MatLike",
            metrics: FrameMetrics,
            detections: Sequence[DetectionData],
            filename_tag: str | None = None,
            folder_tag: str | None = None,
            filename_override: str | None = None,
            kf_test_skip: bool = False,
        ) -> KeyframeEvent:
            return self.emit_event(
                event_type,
                frame_index,
                image,
                metrics,
                detections,
                filename_tag=filename_tag,
                folder_tag=folder_tag,
                filename_override=filename_override,
                original_image_path=original_image_path,
                original_image=_raw_image,  # Pass the clean frame, not the handler's annotated image
                original_depth=depth_image,
                kf_test_skip=kf_test_skip,
            )

        ctx = FrameContext(
            frame_index=frame_index,
            image=image,
            detections=dets,
            metrics=metrics,
            area_state=area_state,
            area_bbox=area_bbox,
            area_ready=area_ready,
            event_image=event_image,
            person_dets=person_dets,
            hand_dets=hand_dets,
            hand_near_dets=hand_near_dets,
            box_dets=box_dets,
            render_labels=render_labels,
            emit_event=emit_event,
        )

        events: list[KeyframeEvent] = []
        for handler in self._handlers:
            events.extend(handler.update(ctx))

        self._prev_boxes = box_dets
        save_requests = list(self._pending_save_requests)
        self._pending_save_requests.clear()
        if include_signals:
            signals = [build_keyframe_signal(request) for request in save_requests]
            return metrics, events, save_requests, signals
        return metrics, events, save_requests
