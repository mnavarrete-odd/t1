import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import List

from .types import FrameData, FrameResult, PreparedDetection


class Reporter:
    def __init__(self):
        self.rows = []
        self.frame_summaries = []

    def add(self, frame: FrameData, detections: List[PreparedDetection], result: FrameResult):
        assignment_map = {a.det_idx: a for a in result.assignments}

        for i, det in enumerate(detections):
            a = assignment_map.get(i)
            self.rows.append(
                {
                    "frame_index": frame.frame_index,
                    "image_name": frame.image_name,
                    "det_idx": i,
                    "class_id": det.class_id,
                    "class_name": det.class_name,
                    "track_id": a.track_id if a else -1,
                    "status": a.status if a else "none",
                    "cost": a.cost if a else -1.0,
                    "reason": a.reason if a else "",
                    "center_norm": a.center_norm if a else float("nan"),
                    "aspect_rel_diff": a.aspect_rel_diff if a else float("nan"),
                    "height_depth_rel_diff": a.height_depth_rel_diff if a else float("nan"),
                    "cosine_distance": a.cosine_distance if a else float("nan"),
                    "depth_delta_match": a.depth_delta if a else float("nan"),
                    "match_probability": a.match_probability if a else float("nan"),
                    "count_action": a.count_action if a else "none",
                    "count_reason": a.count_reason if a else "",
                    "count_units": a.count_units if a else 0.0,
                    "depth_center": det.depth_center,
                    "product_depth": det.product_depth,
                    "floor_depth": det.floor_depth,
                    "depth_delta_det": det.depth_delta,
                    "height_depth": det.height_depth,
                    "cx": det.centroid[0],
                    "cy": det.centroid[1],
                    "aspect_ratio": det.aspect_ratio,
                }
            )

        self.frame_summaries.append(
            {
                "frame_index": frame.frame_index,
                "image_name": frame.image_name,
                "num_detections": result.num_detections,
                "num_matched": result.num_matched,
                "num_new": result.num_new,
                "num_suppressed": result.num_suppressed,
                "num_lost_tracks": result.num_lost_tracks,
                "num_removed": result.num_removed,
                "num_active_tracks": result.num_active_tracks,
                "added_units": result.added_units,
                "removed_units": result.removed_units,
                "net_units": result.net_units,
                "running_units": result.running_units,
                "change_state": result.change_state,
                "change_detail": result.change_detail,
                "added_by_class": dict(result.added_by_class),
                "removed_by_class": dict(result.removed_by_class),
                "lost_tracks": [asdict(t) for t in result.lost_tracks],
            }
        )

    def write(self, out_dir: str):
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)

        csv_path = out / "matches.csv"
        json_path = out / "summary.json"

        if self.rows:
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(self.rows[0].keys()))
                writer.writeheader()
                writer.writerows(self.rows)

        total_added_units = float(sum(f.get("added_units", 0.0) for f in self.frame_summaries))
        total_removed_units = float(sum(f.get("removed_units", 0.0) for f in self.frame_summaries))
        final_running_units = (
            float(self.frame_summaries[-1].get("running_units", 0.0))
            if self.frame_summaries
            else 0.0
        )
        summary_payload = {
            "num_frames": len(self.frame_summaries),
            "total_added_units": total_added_units,
            "total_removed_units": total_removed_units,
            "final_running_units": final_running_units,
            "frames": self.frame_summaries,
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(summary_payload, f, indent=2)
