from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class DetectionRaw:
    class_id: int
    class_name: str
    bbox_cxcywh: np.ndarray  # [cx, cy, w, h]
    confidence: float


@dataclass
class FrameData:
    frame_index: int
    image_name: str
    image: np.ndarray
    depth_map: Optional[np.ndarray]
    area_bbox_cxcywh: Optional[np.ndarray]
    detections: List[DetectionRaw]


@dataclass
class PreparedDetection:
    class_id: int
    class_name: str
    confidence: float
    bbox_cxcywh: np.ndarray
    bbox_xyxy: Tuple[int, int, int, int]
    centroid: Tuple[float, float]
    aspect_ratio: float
    height_depth: float
    depth_center: float
    embedding: np.ndarray
    product_depth: float = float("nan")
    floor_depth: float = float("nan")
    depth_delta: float = float("nan")


@dataclass
class TrackState:
    track_id: int
    class_id: int
    class_name: str
    bbox_cxcywh: np.ndarray
    centroid: Tuple[float, float]
    aspect_ratio: float
    height_depth: float
    depth_center: float
    embedding: np.ndarray
    product_depth: float = float("nan")
    floor_depth: float = float("nan")
    depth_delta: float = float("nan")
    last_frame_index: int = -1
    hits: int = 1
    misses: int = 0
    age: int = 1


@dataclass
class DetectionAssignment:
    det_idx: int
    track_id: int
    status: str  # "matched" | "new" | "suppressed"
    cost: float
    reason: str = ""
    center_norm: float = float("nan")
    aspect_rel_diff: float = float("nan")
    height_depth_rel_diff: float = float("nan")
    cosine_distance: float = float("nan")
    depth_delta: float = float("nan")
    match_probability: float = float("nan")
    count_action: str = "none"  # "added" | "ignored" | "none"
    count_reason: str = ""
    count_units: float = 0.0


@dataclass
class LostTrackRecord:
    track_id: int
    class_id: int
    class_name: str
    product_depth: float
    floor_depth: float
    depth_delta: float
    count_action: str = "none"  # "removed" | "ignored" | "none"
    count_reason: str = ""
    count_units: float = 0.0
    count_factor: float = 1.0


@dataclass
class FrameResult:
    frame_index: int
    assignments: List[DetectionAssignment] = field(default_factory=list)
    num_detections: int = 0
    num_matched: int = 0
    num_new: int = 0
    num_active_tracks: int = 0
    num_suppressed: int = 0
    num_lost_tracks: int = 0
    num_removed: int = 0
    added_units: float = 0.0
    removed_units: float = 0.0
    net_units: float = 0.0
    running_units: float = 0.0
    change_state: str = "no_change"  # "added" | "removed" | "no_change"
    change_detail: str = "none"  # "added_only" | "removed_only" | "mixed" | "none"
    added_by_class: Dict[str, float] = field(default_factory=dict)
    removed_by_class: Dict[str, float] = field(default_factory=dict)
    lost_tracks: List[LostTrackRecord] = field(default_factory=list)
