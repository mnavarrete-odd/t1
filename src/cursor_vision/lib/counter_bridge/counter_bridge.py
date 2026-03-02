"""
CounterBridge: adaptador in-memory que conecta los KeyframeSaveRequests del
Photographer con el pipeline de tracking y conteo de CounterVision,
sin necesidad de guardar keyframes a disco.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from core.counter.containment import ContainmentConfig
from core.counter.costs import CostConfig
from core.counter.counting import CountingConfig, FrameChangeCounter
from core.counter.depth_topdown import DepthTopdownConfig
from core.counter.embedder import create_embedder
from core.counter.feature_extractor import FeatureExtractor
from core.counter.filters import (
    filter_detections_by_class,
    filter_detections_by_depth_work_area,
    normalize_class_name,
)
from core.counter.tracker import CounterTracker, TrackerConfig
from core.counter.types import DetectionRaw, FrameData, FrameResult, PreparedDetection
from core.counter.workarea_mask import WorkAreaMaskConfig
from core.photographer.types import KeyframeSaveRequest, bbox_xyxy_to_cxcywh

logger = logging.getLogger(__name__)


def _load_yaml(path: str) -> dict:
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _parse_contained_by_rules(raw_rules: dict) -> Dict[int, Tuple[int, float]]:
    parsed: Dict[int, Tuple[int, float]] = {}
    for key, value in (raw_rules or {}).items():
        try:
            contained = int(key)
        except Exception:
            continue
        container = None
        factor = 1.0
        if isinstance(value, dict):
            container = value.get("container")
            factor = value.get("factor", 1.0)
        elif isinstance(value, (list, tuple)) and len(value) >= 1:
            container = value[0]
            if len(value) > 1:
                factor = value[1]
        if container is None:
            continue
        try:
            parsed[contained] = (int(container), float(factor))
        except Exception:
            continue
    return parsed


def _invert_contained_by_rules(
    rules: Dict[int, Tuple[int, float]],
) -> Dict[int, Tuple[int, float]]:
    inverted: Dict[int, Tuple[int, float]] = {}
    for contained, (container, factor) in (rules or {}).items():
        if container not in inverted:
            inverted[int(container)] = (int(contained), float(factor))
    return inverted


def _parse_int_float_map(raw_map: dict) -> Dict[int, float]:
    out: Dict[int, float] = {}
    for key, value in (raw_map or {}).items():
        try:
            out[int(key)] = float(value)
        except Exception:
            continue
    return out


def _parse_int_tuple(raw_values) -> Tuple[int, ...]:
    out = []
    for value in raw_values or []:
        try:
            out.append(int(value))
        except Exception:
            continue
    return tuple(sorted(set(out)))


def _resolve_path_from_base(path_value: str, base_dir: Path) -> str:
    p = Path(path_value)
    if p.is_absolute():
        return str(p)
    return str((base_dir / p).resolve())


def _resolve_optional_path_from_base(path_value: str, base_dir: Path) -> str:
    val = (path_value or "").strip()
    if not val:
        return ""
    return _resolve_path_from_base(val, base_dir)


class CounterBridge:
    """
    Receives KeyframeSaveRequests in-memory and runs the full Counter pipeline
    (feature extraction, tracking, counting) without touching disk.
    """

    def __init__(self, config_path: str, device: str = "cuda"):
        cfg_path = Path(config_path)
        cfg = _load_yaml(str(cfg_path))
        cfg_dir = cfg_path.parent

        self.allowed_class_names = [
            normalize_class_name(x)
            for x in cfg.get("classes", {}).get(
                "allowed_names", ["cajas", "folio", "manga", "saco", "producto"]
            )
        ]

        # Embedder
        embed_cfg = cfg.get("embedder", {})
        embedder_kind = embed_cfg.get("type", "dino")
        model_dir_cfg = embed_cfg.get("model_dir", "../models/dino")
        model_dir = _resolve_path_from_base(model_dir_cfg, cfg_dir)
        batch_size = int(embed_cfg.get("batch_size", 32))

        self.embedder = create_embedder(
            kind=embedder_kind,
            model_dir=model_dir,
            device=device,
            batch_size=batch_size,
        )

        # Feature extractor
        feat_cfg = cfg.get("features", {})
        depth_scale = float(feat_cfg.get("depth_scale", 1.0))
        depth_topdown_cfg = self._build_depth_topdown_cfg(cfg, depth_scale)
        self.workarea_cfg = self._build_workarea_cfg(cfg)
        self.containment_cfg = self._build_containment_cfg(cfg)

        intrinsics = {
            "fx": float(self.workarea_cfg.fx),
            "fy": float(self.workarea_cfg.fy),
            "ppx": float(self.workarea_cfg.cx),
            "ppy": float(self.workarea_cfg.cy),
        }
        self.extractor = FeatureExtractor(
            embedder=self.embedder,
            depth_patch_radius=int(feat_cfg.get("depth_patch_radius", 2)),
            depth_scale=depth_scale,
            intrinsics=intrinsics,
            rotation_matrix=None,
            depth_topdown_cfg=depth_topdown_cfg,
            use_depth_topdown=True,
        )

        # Tracker
        cost_cfg_payload = dict(cfg.get("cost", {}) or {})
        if "correct_distributions_path" in cost_cfg_payload:
            cost_cfg_payload["correct_distributions_path"] = (
                _resolve_optional_path_from_base(
                    str(cost_cfg_payload.get("correct_distributions_path", "")),
                    cfg_dir,
                )
            )
        if "incorrect_distributions_path" in cost_cfg_payload:
            cost_cfg_payload["incorrect_distributions_path"] = (
                _resolve_optional_path_from_base(
                    str(cost_cfg_payload.get("incorrect_distributions_path", "")),
                    cfg_dir,
                )
            )
        if "depth_discount_class_ids" in cost_cfg_payload:
            try:
                cost_cfg_payload["depth_discount_class_ids"] = tuple(
                    int(x)
                    for x in (cost_cfg_payload.get("depth_discount_class_ids") or [])
                )
            except Exception:
                cost_cfg_payload["depth_discount_class_ids"] = ()

        cost_cfg = CostConfig(**cost_cfg_payload)
        tracker_cfg = TrackerConfig(**cfg.get("tracking", {}))
        if tracker_cfg.contained_by_rules:
            tracker_cfg.contained_by_rules = _parse_contained_by_rules(
                tracker_cfg.contained_by_rules
            )
        if not tracker_cfg.contained_by_rules and self.containment_cfg.rules:
            tracker_cfg.contained_by_rules = dict(self.containment_cfg.rules)

        self.tracker = CounterTracker(tracker_cfg=tracker_cfg, cost_cfg=cost_cfg)

        # Counting
        counting_cfg = self._build_counting_cfg(
            cfg, fallback_contained_by_rules=tracker_cfg.contained_by_rules
        )
        self.change_counter = FrameChangeCounter(counting_cfg)

        self.prev_depth: Optional[np.ndarray] = None
        self._frame_counter = 0
        self.last_frame_data: Optional[FrameData] = None
        self.last_prepared: Optional[List[PreparedDetection]] = None
        self.last_result: Optional[FrameResult] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_keyframe(
        self,
        kf_request: KeyframeSaveRequest,
        depth_map: Optional[np.ndarray] = None,
    ) -> FrameResult:
        """
        Convert a KeyframeSaveRequest into a FrameData, run the full counter
        pipeline, and return the FrameResult with unit counts.
        """
        frame_data = self._kf_to_frame_data(kf_request, depth_map)

        filtered = filter_detections_by_class(
            frame_data.detections, self.allowed_class_names
        )
        filtered, _ = filter_detections_by_depth_work_area(
            filtered,
            frame_data.area_bbox_cxcywh,
            frame_data.depth_map,
            workarea_cfg=self.workarea_cfg,
            containment_cfg=self.containment_cfg,
            rotation_matrix=None,
        )

        prepared = self.extractor.prepare(
            frame_data, filtered, prev_depth_map=self.prev_depth
        )

        result = self.tracker.step(
            frame_index=frame_data.frame_index,
            detections=prepared,
            image_shape=frame_data.image.shape[:2],
        )
        result = self.change_counter.apply(result, prepared)

        self.prev_depth = frame_data.depth_map
        self._frame_counter += 1
        self.last_frame_data = frame_data
        self.last_prepared = prepared
        self.last_result = result

        return result

    def reset(self) -> None:
        """Reset tracker and counter for a new task cycle."""
        self.tracker.reset()
        self.change_counter.reset()
        self.prev_depth = None
        self._frame_counter = 0
        self.last_frame_data = None
        self.last_prepared = None
        self.last_result = None

    @property
    def running_units(self) -> float:
        return self.change_counter.running_units

    # ------------------------------------------------------------------
    # Conversion helpers
    # ------------------------------------------------------------------

    def _kf_to_frame_data(
        self,
        kf_request: KeyframeSaveRequest,
        depth_map: Optional[np.ndarray],
    ) -> FrameData:
        """
        Convert a KeyframeSaveRequest (Photographer output) into a FrameData
        (Counter input).  Key transformations:
          - image: use original_image if available, else the annotated image
          - depth_map: use original_depth from KF request, or the explicit depth_map arg
          - detections: DetectionData (xyxy) -> DetectionRaw (cxcywh)
          - area_bbox: xyxy -> cxcywh
        """
        image = kf_request.original_image
        if image is None:
            image = kf_request.image
        if not isinstance(image, np.ndarray):
            image = np.zeros((480, 640, 3), dtype=np.uint8)

        kf_depth = kf_request.original_depth
        if kf_depth is not None and isinstance(kf_depth, np.ndarray):
            final_depth = kf_depth
        else:
            final_depth = depth_map

        area_bbox_cxcywh = None
        if kf_request.metrics and kf_request.metrics.area_bbox:
            cx, cy, w, h = bbox_xyxy_to_cxcywh(kf_request.metrics.area_bbox)
            area_bbox_cxcywh = np.array([cx, cy, w, h], dtype=np.float32)

        detections_raw: List[DetectionRaw] = []
        for det in kf_request.detections or []:
            cx, cy, w, h = bbox_xyxy_to_cxcywh(det.bbox)
            detections_raw.append(
                DetectionRaw(
                    class_id=det.class_id,
                    class_name=det.class_name,
                    bbox_cxcywh=np.array([cx, cy, w, h], dtype=np.float32),
                    confidence=det.confidence,
                )
            )

        return FrameData(
            frame_index=kf_request.frame_index,
            image_name=kf_request.resolved_filename(),
            image=image,
            depth_map=final_depth,
            area_bbox_cxcywh=area_bbox_cxcywh,
            detections=detections_raw,
        )

    # ------------------------------------------------------------------
    # Config builders (mirror counterMain.py logic)
    # ------------------------------------------------------------------

    @staticmethod
    def _build_workarea_cfg(cfg: dict) -> WorkAreaMaskConfig:
        workarea = cfg.get("workarea", {}) or {}
        intr = workarea.get("intrinsics", {}) or {}
        return WorkAreaMaskConfig(
            enabled=bool(workarea.get("enabled", True)),
            max_depth_mm=float(workarea.get("max_depth_mm", 2800.0)),
            bbox_mask_threshold=float(workarea.get("bbox_mask_threshold", 0.40)),
            area_bbox_mask_threshold=float(
                workarea.get("area_bbox_mask_threshold", 0.20)
            ),
            depth_scale_to_mm=float(workarea.get("depth_scale_to_mm", 1.0)),
            downsample_scale=int(workarea.get("downsample_scale", 4)),
            grid_resolution_mm=float(workarea.get("grid_resolution_mm", 50.0)),
            center_tolerance_px=int(workarea.get("center_tolerance_px", 2)),
            fx=float(intr.get("fx", 1125.636)),
            fy=float(intr.get("fy", 1125.133)),
            cx=float(intr.get("cx", 952.45)),
            cy=float(intr.get("cy", 550.62)),
        )

    @staticmethod
    def _build_containment_cfg(cfg: dict) -> ContainmentConfig:
        containment = cfg.get("containment", {}) or {}
        parsed_rules = _parse_contained_by_rules(
            containment.get("rules", {}) or {}
        )
        return ContainmentConfig(
            enabled=bool(containment.get("enabled", True)),
            ioa_threshold=float(containment.get("ioa_threshold", 0.70)),
            rules=parsed_rules,
        )

    @staticmethod
    def _build_depth_topdown_cfg(cfg: dict, depth_scale: float) -> DepthTopdownConfig:
        top = cfg.get("depth_topdown", {}) or {}
        return DepthTopdownConfig(
            depth_scale=float(top.get("depth_scale", depth_scale)),
            max_depth_m=float(top.get("max_depth_m", 3.0)),
            resolution_cm=float(top.get("resolution_cm", 1.0)),
            histogram_bins=int(top.get("histogram_bins", 300)),
            histogram_top_n=int(top.get("histogram_top_n", 10)),
            inset_ratio=float(top.get("inset_ratio", 0.5)),
            center_crop_ratio=float(top.get("center_crop_ratio", 0.5)),
        )

    @staticmethod
    def _build_counting_cfg(
        cfg: dict,
        *,
        fallback_contained_by_rules: Optional[Dict[int, Tuple[int, float]]] = None,
    ) -> CountingConfig:
        counting = cfg.get("counting", {}) or {}
        contained_by_rules = _parse_contained_by_rules(
            counting.get("contained_by_rules", {}) or {}
        )
        if not contained_by_rules and fallback_contained_by_rules:
            contained_by_rules = dict(fallback_contained_by_rules)
        contains_rules = _parse_contained_by_rules(
            counting.get("contains_rules", {}) or {}
        )
        if not contains_rules and contained_by_rules:
            contains_rules = _invert_contained_by_rules(contained_by_rules)

        return CountingConfig(
            enabled=bool(counting.get("enabled", True)),
            ignore_bootstrap_new_on_first_frame=bool(
                counting.get("ignore_bootstrap_new_on_first_frame", True)
            ),
            use_depth_for_units=bool(counting.get("use_depth_for_units", True)),
            stackable_class_ids=_parse_int_tuple(
                counting.get("stackable_class_ids", [])
            ),
            default_height_m=float(counting.get("default_height_m", 0.20)),
            class_heights_m=_parse_int_float_map(
                counting.get("class_heights_m", {})
            ),
            tall_height_threshold_m=float(
                counting.get("tall_height_threshold_m", 0.14)
            ),
            round_up_threshold_tall=float(
                counting.get("round_up_threshold_tall", 0.50)
            ),
            round_up_threshold_short=float(
                counting.get("round_up_threshold_short", 0.60)
            ),
            min_depth_delta_m=float(counting.get("min_depth_delta_m", 0.01)),
            require_positive_depth_delta=bool(
                counting.get("require_positive_depth_delta", True)
            ),
            min_units_per_detection=int(counting.get("min_units_per_detection", 1)),
            max_units_per_detection=int(counting.get("max_units_per_detection", 6)),
            class_factors=_parse_int_float_map(counting.get("class_factors", {})),
            use_containment_factors=bool(
                counting.get("use_containment_factors", False)
            ),
            contained_by_rules=contained_by_rules,
            contains_rules=contains_rules,
            state_eps=float(counting.get("state_eps", 1e-6)),
        )
