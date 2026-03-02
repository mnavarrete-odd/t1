from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence
import math

from .config import PhotographerConfig
from .types import DetectionData
from .utils import bbox_area, bbox_center, bbox_intersection


@dataclass
class WorkAreaState:
    bbox: tuple[float, float, float, float]
    class_name: str
    confidence: float
    stable_frames: int
    last_seen_frame: int


@dataclass
class AreaCandidate:
    bbox: tuple[float, float, float, float]
    class_name: str
    confidence: float
    area: float
    center_dist: float


class AreaSelector:
    def __init__(self, cfg: PhotographerConfig) -> None:
        self.cfg = cfg
        self._frames_seen = 0
        self._area_candidates: list[AreaCandidate] = []
        self._area_fixed: WorkAreaState | None = None
        self._area_state: WorkAreaState | None = None
        self._refit_candidate: WorkAreaState | None = None
        self._refit_frames = 0

    def update(
        self,
        dets: Sequence[DetectionData],
        frame_index: int,
        image_w: int,
        image_h: int,
    ) -> tuple[WorkAreaState | None, bool]:
        self._frames_seen += 1

        if self._area_fixed is None and self.cfg.area_warmup_frames > 0:
            self._collect_candidates(dets, image_w, image_h)
            if self._frames_seen >= self.cfg.area_warmup_frames:
                self._area_fixed = self._select_fixed_area(frame_index)
                if self._area_fixed is not None:
                    self._area_candidates.clear()

        if self._area_fixed is not None:
            if self.cfg.area_refit_enabled:
                self._maybe_refit(dets, frame_index, image_w, image_h)
            area_state = self._area_fixed
        else:
            area_state = self._update_dynamic(dets, frame_index)

        area_ready = bool(
            area_state
            and (
                self._area_fixed is not None
                or area_state.stable_frames >= self.cfg.area_stable_frames
            )
        )
        return area_state, area_ready

    def _maybe_refit(
        self,
        dets: Sequence[DetectionData],
        frame_index: int,
        image_w: int,
        image_h: int,
    ) -> None:
        if self._area_fixed is None:
            self._refit_candidate = None
            self._refit_frames = 0
            return

        area_dets = [
            d
            for d in dets
            if d.class_name in self.cfg.area_classes
            and d.confidence >= self.cfg.area_min_conf
        ]
        if not area_dets:
            self._refit_candidate = None
            self._refit_frames = 0
            return

        candidate = max(area_dets, key=lambda d: (d.confidence, bbox_area(d.bbox)))
        if not self._is_candidate_different(
            candidate.bbox, self._area_fixed.bbox, image_w, image_h
        ):
            self._refit_candidate = None
            self._refit_frames = 0
            return

        if (
            self._refit_candidate is not None
            and candidate.class_name == self._refit_candidate.class_name
            and self._candidate_matches_previous(candidate.bbox, self._refit_candidate.bbox)
        ):
            self._refit_frames += 1
        else:
            self._refit_candidate = WorkAreaState(
                bbox=candidate.bbox,
                class_name=candidate.class_name,
                confidence=candidate.confidence,
                stable_frames=1,
                last_seen_frame=frame_index,
            )
            self._refit_frames = 1

        if self._refit_frames >= self.cfg.area_refit_frames:
            self._area_fixed = WorkAreaState(
                bbox=candidate.bbox,
                class_name=candidate.class_name,
                confidence=candidate.confidence,
                stable_frames=max(
                    self.cfg.area_stable_frames, self.cfg.area_refit_frames
                ),
                last_seen_frame=frame_index,
            )
            self._refit_candidate = None
            self._refit_frames = 0

    def _is_candidate_different(
        self,
        candidate_bbox: tuple[float, float, float, float],
        fixed_bbox: tuple[float, float, float, float],
        image_w: int,
        image_h: int,
    ) -> bool:
        diag = math.hypot(image_w, image_h) or 1.0
        cx1, cy1 = bbox_center(candidate_bbox)
        cx2, cy2 = bbox_center(fixed_bbox)
        dist = math.hypot(cx1 - cx2, cy1 - cy2)
        min_px = self.cfg.area_refit_center_dist_min_px
        max_px = self.cfg.area_refit_center_dist_max_px
        if min_px < 0:
            min_px = 0.0
        if max_px < min_px:
            return False
        return min_px <= dist <= max_px

    def _candidate_matches_previous(
        self,
        candidate_bbox: tuple[float, float, float, float],
        prev_bbox: tuple[float, float, float, float],
    ) -> bool:
        return self._bbox_iou(candidate_bbox, prev_bbox) >= 0.7

    def _bbox_iou(
        self,
        a: tuple[float, float, float, float],
        b: tuple[float, float, float, float],
    ) -> float:
        inter = bbox_intersection(a, b)
        if inter <= 0:
            return 0.0
        union = bbox_area(a) + bbox_area(b) - inter
        if union <= 0:
            return 0.0
        return inter / union

    def _collect_candidates(
        self,
        dets: Sequence[DetectionData],
        image_w: int,
        image_h: int,
    ) -> None:
        cx_img = image_w * 0.5
        cy_img = image_h * 0.5
        for d in dets:
            if d.class_name not in self.cfg.area_classes:
                continue
            if d.confidence < self.cfg.area_min_conf:
                continue
            area = bbox_area(d.bbox)
            cx, cy = bbox_center(d.bbox)
            dist = math.hypot(cx - cx_img, cy - cy_img)
            self._area_candidates.append(
                AreaCandidate(
                    bbox=d.bbox,
                    class_name=d.class_name,
                    confidence=d.confidence,
                    area=area,
                    center_dist=dist,
                )
            )

    def _select_fixed_area(self, frame_index: int) -> WorkAreaState | None:
        if not self._area_candidates:
            return None
        best = max(
            self._area_candidates,
            key=lambda c: (c.area, -c.center_dist, c.confidence),
        )
        return WorkAreaState(
            bbox=best.bbox,
            class_name=best.class_name,
            confidence=best.confidence,
            stable_frames=max(self.cfg.area_stable_frames, self.cfg.area_warmup_frames),
            last_seen_frame=frame_index,
        )

    def _update_dynamic(
        self,
        dets: Sequence[DetectionData],
        frame_index: int,
    ) -> WorkAreaState | None:
        area_dets = [
            d
            for d in dets
            if d.class_name in self.cfg.area_classes
            and d.confidence >= self.cfg.area_min_conf
        ]
        if area_dets:
            candidate = max(area_dets, key=lambda d: (d.confidence, bbox_area(d.bbox)))
            if self._area_state and candidate.class_name == self._area_state.class_name:
                stable_frames = self._area_state.stable_frames + 1
            else:
                stable_frames = 1
            self._area_state = WorkAreaState(
                bbox=candidate.bbox,
                class_name=candidate.class_name,
                confidence=candidate.confidence,
                stable_frames=stable_frames,
                last_seen_frame=frame_index,
            )
        else:
            if self._area_state:
                missing = frame_index - self._area_state.last_seen_frame
                if missing > self.cfg.area_hold_frames:
                    self._area_state = None
        return self._area_state
