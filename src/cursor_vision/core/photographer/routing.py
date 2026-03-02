from __future__ import annotations

from .types import KeyframeSaveRequest, KeyframeSignal


def build_keyframe_signal(request: KeyframeSaveRequest) -> KeyframeSignal:
    event_group, event_stage = request.event_group_stage()
    is_post_occlusion = request.event_type == "KF-OCCLUSION" and event_stage == "02"
    is_post_occlusion_item = (
        request.event_type == "KF-OCCLUSION-ITEM" and event_stage == "02"
    )

    is_kfs_final = False
    kfs_overwrite = False
    if is_post_occlusion or is_post_occlusion_item:
        is_kfs_final = True
        kfs_overwrite = True
    elif request.event_type in {"KF-AREA-SET", "KF-AREA-EMPTY"}:
        is_kfs_final = True

    is_kf_test_candidate = False
    kf_test_overwrite = False
    kf_test_requires_count_change = False
    if is_post_occlusion:
        is_kf_test_candidate = True
        kf_test_overwrite = True
    elif request.event_type == "KF-AREA-EMPTY" and not request.kf_test_skip:
        is_kf_test_candidate = True
    elif request.event_type == "KF-STABLE-RECONFIRM" and not request.kf_test_skip:
        is_kf_test_candidate = True
        kf_test_requires_count_change = True

    return KeyframeSignal(
        request=request,
        event_group=event_group,
        event_stage=event_stage,
        is_debug_event=True,
        is_kfs_final=is_kfs_final,
        kfs_overwrite=kfs_overwrite,
        is_kf_test_candidate=is_kf_test_candidate,
        kf_test_overwrite=kf_test_overwrite,
        kf_test_requires_count_change=kf_test_requires_count_change,
    )

