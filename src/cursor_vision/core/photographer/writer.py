from __future__ import annotations

from pathlib import Path
from typing import Sequence
import json
import shutil

import cv2
import numpy as np

from core.detector_yolo import YOLODetector

from .config import PhotographerConfig
from .routing import build_keyframe_signal
from .render import LabelRenderer
from .types import DetectionData, KeyframeEvent, KeyframeSaveRequest, bbox_xyxy_to_cxcywh


class KeyframeWriter:
    _product_detector_cache: dict[tuple[str, float], YOLODetector] = {}

    def __init__(self, config: PhotographerConfig) -> None:
        self.cfg = config
        self.outdir = Path(config.outdir)
        self.outdir.mkdir(parents=True, exist_ok=True)
        self._events_path = self.outdir / "keyframes.jsonl"
        if self.cfg.clear_events:
            self._events_path.write_text("", encoding="utf-8")
        self._label_renderer = LabelRenderer()
        self._kf_test_last_count: int | None = None
        self._product_detector: YOLODetector | None = None
        self._setup_product_detector()

    def save_many(self, requests: Sequence[KeyframeSaveRequest]) -> list[KeyframeEvent]:
        events: list[KeyframeEvent] = []
        for request in requests:
            events.append(self.save_event(request))
        return events

    def save_event(self, request: KeyframeSaveRequest) -> KeyframeEvent:
        filename = request.resolved_filename()
        folder = self.outdir / request.resolved_folder_name()
        folder.mkdir(parents=True, exist_ok=True)
        image_path = folder / filename
        cv2.imwrite(str(image_path), request.image)

        signal = build_keyframe_signal(request)
        event_group, event_stage = signal.event_group, signal.event_stage
        event_payload = {
            "event_type": request.event_type,
            "frame_index": request.frame_index,
            "image_path": str(image_path),
            "image_name": image_path.name,
            "original_image_path": (
                str(request.original_image_path)
                if request.original_image_path
                else None
            ),
            "original_image_name": (
                Path(request.original_image_path).name
                if request.original_image_path
                else None
            ),
            "reason": request.event_type,
            "event_group": event_group,
            "event_stage": event_stage,
            "metrics": request.metrics.to_dict(),
            "config": self._config_snapshot(),
            "detections": [
                {
                    "class_id": d.class_id,
                    "class_name": d.class_name,
                    "bbox": list(bbox_xyxy_to_cxcywh(d.bbox)),
                    "confidence": d.confidence,
                    "extra_data": dict(d.extra_data) if d.extra_data else {},
                    "tracking_id": d.tracking_id,
                }
                for d in request.detections
            ],
        }

        json_path = image_path.with_suffix(".json")
        with json_path.open("w", encoding="utf-8") as jf:
            json.dump(event_payload, jf, ensure_ascii=True)

        if signal.is_kfs_final:
            self._copy_to_kf_dir(
                "KFs",
                request,
                event_payload,
                overwrite=signal.kfs_overwrite,
            )

        if signal.is_kf_test_candidate:
            should_copy = True
            if signal.kf_test_requires_count_change and self._kf_test_last_count is not None:
                should_copy = request.metrics.count_in_area != self._kf_test_last_count
            if should_copy:
                self._copy_to_kf_dir(
                    "KF-TEST",
                    request,
                    event_payload,
                    overwrite=signal.kf_test_overwrite,
                )
                self._kf_test_last_count = request.metrics.count_in_area

        event = KeyframeEvent(
            event_type=request.event_type,
            frame_index=request.frame_index,
            image_path=str(image_path),
            metrics=request.metrics,
        )
        with self._events_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event.to_dict(), ensure_ascii=True) + "\n")
        print(f"[EVENT] {request.event_type} frame={request.frame_index} path={image_path}")
        return event

    def _copy_to_kf_dir(
        self,
        root_name: str,
        request: KeyframeSaveRequest,
        event_payload: dict[str, object],
        *,
        overwrite: bool,
    ) -> None:
        kfs_root = self.outdir / root_name
        images_dir = kfs_root / "images"
        annotated_dir = kfs_root / "annotated"
        data_dir = kfs_root / "data"
        annotated_producto_dir = kfs_root / "annotated_producto"
        kfs_stem = f"KF-PHOTO-{request.frame_index:06d}"
        dest_image = images_dir / f"{kfs_stem}.jpg"
        dest_annotated = annotated_dir / f"{kfs_stem}.jpg"
        dest_annotated_producto = annotated_producto_dir / f"{kfs_stem}.jpg"
        dest_json = data_dir / f"{kfs_stem}.json"
        if dest_image.exists() and not overwrite:
            return
        try:
            images_dir.mkdir(parents=True, exist_ok=True)
            annotated_dir.mkdir(parents=True, exist_ok=True)
            if self._product_detector is not None:
                annotated_producto_dir.mkdir(parents=True, exist_ok=True)
            data_dir.mkdir(parents=True, exist_ok=True)

            clean_image = None
            if request.original_image is not None:
                clean_image = request.original_image
            elif request.original_image_path:
                clean_image = cv2.imread(str(request.original_image_path))
            
            if clean_image is None:
                clean_image = request.image

            allowed = set(self.cfg.box_classes)
            allowed.update(self.cfg.hand_classes)
            allowed.update(self.cfg.person_classes)
            allowed.add("manga")
            annotated_dets = [d for d in request.detections if d.class_name in allowed]

            annotated_image = self._label_renderer.render(
                clean_image,
                request.metrics.area_bbox,
                request.metrics.area_class_name,
                extra_detections=annotated_dets,
            )

            if not cv2.imwrite(str(dest_image), clean_image):
                print(f"[WARN] No se pudo escribir imagen limpia: {dest_image}")
            if not cv2.imwrite(str(dest_annotated), annotated_image):
                print(f"[WARN] No se pudo escribir imagen anotada: {dest_annotated}")
            if self._product_detector is not None:
                product_results, _ = self._product_detector.detect(clean_image)
                product_dets = self._detections_from_results(product_results)
                product_annotated = self._label_renderer.render(
                    clean_image,
                    request.metrics.area_bbox,
                    request.metrics.area_class_name,
                    extra_detections=product_dets,
                )
                if not cv2.imwrite(str(dest_annotated_producto), product_annotated):
                    print(
                        f"[WARN] No se pudo escribir imagen anotada de productos: "
                        f"{dest_annotated_producto}"
                    )

            if request.original_depth is not None:
                depth_dir = kfs_root / "depth"
                depth_dir.mkdir(parents=True, exist_ok=True)
                depth_name = f"{kfs_stem}.npy"
                try:
                    np.save(str(depth_dir / depth_name), request.original_depth)
                except Exception as exc:
                    print(f"[WARN] No se pudo guardar depth en memoria: {exc}")
            elif self.cfg.depth_dir and request.original_image_path:
                depth_dir = kfs_root / "depth"
                depth_dir.mkdir(parents=True, exist_ok=True)
                orig_name = Path(request.original_image_path).name
                depth_src = self.cfg.depth_dir / orig_name
                if not depth_src.exists():
                    stem = Path(orig_name).stem
                    matches = sorted(self.cfg.depth_dir.glob(f"{stem}.*"))
                    depth_src = matches[0] if matches else None
                if depth_src and depth_src.exists():
                    depth_name = f"{kfs_stem}{depth_src.suffix}"
                    shutil.copy2(depth_src, depth_dir / depth_name)

            kfs_payload = dict(event_payload)
            kfs_payload.pop("image_path", None)
            kfs_payload.pop("original_image_path", None)
            kfs_payload["image_name"] = dest_image.name
            with dest_json.open("w", encoding="utf-8") as jf:
                json.dump(kfs_payload, jf, ensure_ascii=True)
        except OSError as exc:
            print(f"[WARN] No se pudo copiar KF a {kfs_root}: {exc}")

    def _config_snapshot(self) -> dict[str, object]:
        return {
            "person_dist_px": self.cfg.person_dist_px,
            "person_near_enabled": self.cfg.person_near_enabled,
            "hand_dist_px": self.cfg.hand_dist_px,
            "area_expand_ratio": self.cfg.area_expand_ratio,
            "area_refit_enabled": self.cfg.area_refit_enabled,
            "area_refit_frames": self.cfg.area_refit_frames,
            "area_refit_center_dist_min_px": self.cfg.area_refit_center_dist_min_px,
            "area_refit_center_dist_max_px": self.cfg.area_refit_center_dist_max_px,
            "stable_area_frames": self.cfg.stable_area_frames,
            "stable_area_movement_max": self.cfg.stable_area_movement_max,
            "stable_area_require_count_stability": self.cfg.stable_area_require_count_stability,
            "movement_by_area": self.cfg.movement_by_area,
            "stable_reconfirm_frames": self.cfg.stable_reconfirm_frames,
            "stable_empty_frames": self.cfg.stable_empty_frames,
            "occlusion_start_ratio": self.cfg.occlusion_start_ratio,
            "occlusion_end_ratio": self.cfg.occlusion_end_ratio,
            "occlusion_start_frames": self.cfg.occlusion_start_frames,
            "occlusion_end_frames": self.cfg.occlusion_end_frames,
            "occlusion_pre_offset_frames": self.cfg.occlusion_pre_offset_frames,
            "occlusion_post_offset_frames": self.cfg.occlusion_post_offset_frames,
            "occlusion_item_classes": self.cfg.occlusion_item_classes,
            "occlusion_change_count_min": self.cfg.occlusion_change_count_min,
            "occlusion_change_coverage_min": self.cfg.occlusion_change_coverage_min,
            "occlusion_change_confirm_frames": self.cfg.occlusion_change_confirm_frames,
            "occlusion_change_cooldown_frames": self.cfg.occlusion_change_cooldown_frames,
            "product_start_frames": self.cfg.product_start_frames,
            "product_end_frames": self.cfg.product_end_frames,
            "product_pre_offset_frames": self.cfg.product_pre_offset_frames,
            "product_post_offset_frames": self.cfg.product_post_offset_frames,
            "product_save_prepost": self.cfg.product_save_prepost,
        }

    def _setup_product_detector(self) -> None:
        if self.cfg.product_kf_model is None:
            return
        model_path = Path(self.cfg.product_kf_model)
        if not model_path.exists():
            print(
                f"[WARN] product_kf_model no existe: {model_path}. "
                "Se desactiva annotated_producto."
            )
            return
        key = (str(model_path.resolve()), float(self.cfg.product_kf_conf))
        detector = self._product_detector_cache.get(key)
        if detector is None:
            detector = YOLODetector(str(model_path), conf=float(self.cfg.product_kf_conf))
            self._product_detector_cache[key] = detector
        self._product_detector = detector

    def _detections_from_results(self, results) -> list[DetectionData]:
        dets: list[DetectionData] = []
        if not results:
            return dets
        result = results[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return dets
        track_ids = None
        if getattr(boxes, "id", None) is not None:
            track_ids = boxes.id.cpu().numpy().astype(int).tolist()
        names = getattr(result, "names", {})
        for idx, box in enumerate(boxes):
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            if isinstance(names, dict):
                class_name = names.get(cls, str(cls))
            else:
                class_name = names[cls] if 0 <= cls < len(names) else str(cls)
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()
            tracking_id = None
            if track_ids is not None and idx < len(track_ids):
                tracking_id = track_ids[idx]
            dets.append(
                DetectionData(
                    class_id=cls,
                    class_name=class_name,
                    bbox=(float(x1), float(y1), float(x2), float(y2)),
                    confidence=conf,
                    tracking_id=tracking_id,
                )
            )
        return dets
