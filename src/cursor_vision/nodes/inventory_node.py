#!/usr/bin/env python3
"""
inventory_node.py

Main ROS2 node for CursorVision.  Orchestrates:
  - State machine (order / HU lifecycle)
  - Per-camera pipelines (detection -> photographer -> counter)
  - Task controller (item start/end/order end)
  - Count result publication
"""
import sys
import json
import time
import threading
from pathlib import Path
from typing import Dict, Optional

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from ament_index_python.packages import get_package_share_directory

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from cursor_vision.msg import PickerEvent, PickerPlan, CountResult

from lib.state_machine import OrderStateMachine, HUState, StateInterpreter
from lib.camera_pipeline import CameraPipeline
from lib.counter_bridge import CounterBridge
from lib.task_controller import TaskController
from lib.debug_writer import DebugWriter

from core.detector_yolo import YOLODetector
from core.photographer.photographer import Photographer
from core.photographer.config import PhotographerConfig


class InventoryNode(Node):

    def __init__(self):
        super().__init__("inventory_node")

        self._declare_parameters()
        self._load_parameters()

        # State machine
        self.state_machine = OrderStateMachine(
            on_state_change=self._on_state_machine_change,
            on_hu_state_change=self._on_hu_state_change,
        )
        self.interpreter = StateInterpreter(line_width=55)

        self.event_count = 0
        self.current_order_data = None
        self.active_item_ctx = None
        self._last_event_type = None

        # Shared detector
        self._init_detector()

        # Per-camera pipelines
        self._debug_writers: Dict[str, DebugWriter] = {}
        self._init_pipelines()

        # Task controller
        self.task_controller = TaskController(
            camera_pipelines=self.camera_pipelines,
            debug_writers=self._debug_writers if any(dw.enabled for dw in self._debug_writers.values()) else None,
        )

        # ROS subscribers & publishers
        self._setup_ros_io()

        self._print_startup_summary()

    # ------------------------------------------------------------------
    # Parameters
    # ------------------------------------------------------------------

    def _declare_parameters(self):
        # Cameras
        self.declare_parameter("camera_names", "primary_camera,secondary_camera")

        # Detection
        self.declare_parameter("enable_detection", True)
        self.declare_parameter("detection_fps", 10.0)
        self.declare_parameter("detection_mode", "no_drop")

        # YOLO
        self.declare_parameter("detector_model_path", "")
        self.declare_parameter("detector_confidence", 0.5)
        self.declare_parameter("detector_prefer_tensorrt", True)

        # Sync
        self.declare_parameter("sync_queue_size", 100)
        self.declare_parameter("sync_max_delay", 0.3)
        self.declare_parameter("max_frame_queue_size", 200)

        # Counter config
        self.declare_parameter("counter_config_path", "")
        self.declare_parameter("counter_device", "cuda")

        # Photographer config
        self.declare_parameter("photographer_config_path", "")

        # Topics
        self.declare_parameter("topic_picker_events", "/picker/events")
        self.declare_parameter("topic_picker_plan", "/picker/plan")

        # Debug
        self.declare_parameter("debug_save_keyframes", False)
        self.declare_parameter("debug_save_counter_video", False)
        self.declare_parameter("debug_output_dir", "/tmp/cursor_debug")
        self.declare_parameter("debug_video_fps", 4)

        # Stats
        self.declare_parameter("stats_publish_interval", 5.0)

    def _load_parameters(self):
        self.camera_names = [
            n.strip()
            for n in self.get_parameter("camera_names").value.split(",")
            if n.strip()
        ]
        self.enable_detection = self.get_parameter("enable_detection").value
        self.detection_fps = self.get_parameter("detection_fps").value
        self.detection_mode = self.get_parameter("detection_mode").value

        self.detector_model_path = self.get_parameter("detector_model_path").value
        self.detector_confidence = self.get_parameter("detector_confidence").value

        self.sync_queue_size = min(self.get_parameter("sync_queue_size").value, 500)
        self.sync_max_delay = self.get_parameter("sync_max_delay").value
        self.max_frame_queue_size = min(self.get_parameter("max_frame_queue_size").value, 500)

        self.counter_config_path = self.get_parameter("counter_config_path").value
        self.counter_device = self.get_parameter("counter_device").value

        self.photographer_config_path = self.get_parameter("photographer_config_path").value

        self.topic_picker_events = self.get_parameter("topic_picker_events").value
        self.topic_picker_plan = self.get_parameter("topic_picker_plan").value

        self.debug_save_keyframes = self.get_parameter("debug_save_keyframes").value
        self.debug_save_counter_video = self.get_parameter("debug_save_counter_video").value
        self.debug_output_dir = self.get_parameter("debug_output_dir").value
        self.debug_video_fps = self.get_parameter("debug_video_fps").value

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _resolve_config_path(self, param_value: str, default_name: str) -> str:
        """Resolve a config path: if empty, look in the package share dir."""
        if param_value and param_value.strip():
            p = Path(param_value)
            if p.exists():
                return str(p)
        try:
            pkg_share = get_package_share_directory("cursor_vision")
            candidate = Path(pkg_share) / "config" / default_name
            if candidate.exists():
                return str(candidate)
        except Exception:
            pass
        local = PROJECT_ROOT / "config" / default_name
        if local.exists():
            return str(local)
        raise FileNotFoundError(
            f"Config not found: param='{param_value}', default='{default_name}'"
        )

    def _resolve_model_path(self, model_path_str: str) -> str:
        """Resolve model path from param, package share, or project root."""
        if model_path_str and model_path_str.strip():
            p = Path(model_path_str)
            if p.is_absolute() and p.exists():
                return str(p)
            # Try from package share
            try:
                pkg_share = Path(get_package_share_directory("cursor_vision"))
                resolved = pkg_share / model_path_str
                if resolved.exists():
                    return str(resolved)
                alt = resolved.with_suffix(".engine" if resolved.suffix == ".pt" else ".pt")
                if alt.exists():
                    return str(alt)
            except Exception:
                pass
            # Try from project root
            resolved = PROJECT_ROOT / model_path_str
            if resolved.exists():
                return str(resolved)

        # Defaults
        for candidate in ["models/11-NEW.engine", "models/11-NEW.pt"]:
            for base in [PROJECT_ROOT]:
                p = base / candidate
                if p.exists():
                    return str(p)
            try:
                pkg_share = Path(get_package_share_directory("cursor_vision"))
                p = pkg_share / candidate
                if p.exists():
                    return str(p)
            except Exception:
                pass

        raise FileNotFoundError(f"No YOLO model found (param: '{model_path_str}')")

    def _init_detector(self):
        if not self.enable_detection:
            self.detector = None
            self.detector_lock = threading.Lock()
            return

        model_path = self._resolve_model_path(self.detector_model_path)
        self.detector = YOLODetector(model_path, conf=self.detector_confidence)
        self.detector_lock = threading.Lock()

    def _load_photographer_config(self) -> PhotographerConfig:
        """Build a PhotographerConfig from the photographer YAML config."""
        config_path = self._resolve_config_path(
            self.photographer_config_path, "photographer_config.yaml"
        )
        import yaml
        with open(config_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

        outdir = Path(self.debug_output_dir) / "photographer_out"
        outdir.mkdir(parents=True, exist_ok=True)

        tuple_fields = [
            "area_classes", "box_classes", "person_classes",
            "hand_classes", "empty_classes", "occlusion_item_classes",
        ]
        kwargs = {}
        for key, value in raw.items():
            if key in tuple_fields:
                kwargs[key] = tuple(value) if isinstance(value, list) else (value,)
            elif key == "outdir":
                continue
            else:
                kwargs[key] = value

        # Ensure required tuple fields have defaults
        kwargs.setdefault("area_classes", ("area_de_trabajo_carro", "area_de_trabajo_pallet"))
        kwargs.setdefault("box_classes", ("cajas", "folio", "manga", "saco", "producto"))
        kwargs.setdefault("person_classes", ("persona",))
        kwargs.setdefault("hand_classes", ("mano",))
        kwargs.setdefault("empty_classes", ())

        return PhotographerConfig(outdir=outdir, **kwargs)

    def _init_pipelines(self):
        self.camera_pipelines: Dict[str, CameraPipeline] = {}

        if not self.enable_detection or self.detector is None:
            return

        counter_cfg_path = self._resolve_config_path(
            self.counter_config_path, "counter_default.yaml"
        )
        photographer_cfg = self._load_photographer_config()

        for cam_name in self.camera_names:
            counter_bridge = CounterBridge(
                config_path=counter_cfg_path,
                device=self.counter_device,
            )

            cam_photographer_cfg = photographer_cfg.for_outdir(
                photographer_cfg.outdir / cam_name
            )
            photographer = Photographer(cam_photographer_cfg)

            debug_writer = DebugWriter(
                output_dir=str(Path(self.debug_output_dir) / cam_name),
                save_keyframes=self.debug_save_keyframes,
                save_counter_video=self.debug_save_counter_video,
                video_fps=self.debug_video_fps,
            )
            self._debug_writers[cam_name] = debug_writer

            pipeline = CameraPipeline(
                node=self,
                camera_name=cam_name,
                detector=self.detector,
                detector_lock=self.detector_lock,
                photographer=photographer,
                counter_bridge=counter_bridge,
                debug_writer=debug_writer,
                detection_fps=self.detection_fps,
                detection_mode=self.detection_mode,
                sync_queue_size=self.sync_queue_size,
                sync_max_delay=self.sync_max_delay,
                max_queue_size=self.max_frame_queue_size,
                on_count_result=self._on_count_result,
            )

            self.camera_pipelines[cam_name] = pipeline

    # ------------------------------------------------------------------
    # ROS I/O
    # ------------------------------------------------------------------

    def _setup_ros_io(self):
        # Subscribers
        self.event_sub = self.create_subscription(
            PickerEvent, self.topic_picker_events, self.on_picker_event, 10
        )
        self.plan_sub = self.create_subscription(
            PickerPlan, self.topic_picker_plan, self.on_picker_plan, 10
        )

        # Publishers
        self.count_pubs: Dict[str, object] = {}
        for cam_name in self.camera_names:
            self.count_pubs[cam_name] = self.create_publisher(
                CountResult, f"/camera/{cam_name}/count", 10
            )

        self.summary_pub = self.create_publisher(
            String, "/inventory/summary", 10
        )
        self.state_pub = self.create_publisher(
            String, "/state_machine/summary", 10
        )

        self._last_state_summary = None

    # ------------------------------------------------------------------
    # Count result callback (called from CameraPipeline threads)
    # ------------------------------------------------------------------

    def _on_count_result(self, camera_name: str, result) -> None:
        """Publish a CountResult message for the given camera."""
        pub = self.count_pubs.get(camera_name)
        if pub is None:
            return

        msg = CountResult()
        msg.stamp = self.get_clock().now().to_msg()
        msg.camera_name = camera_name
        msg.frame_index = result.frame_index
        msg.running_units = float(result.running_units)
        msg.added_units = float(result.added_units)
        msg.removed_units = float(result.removed_units)
        msg.net_units = float(result.net_units)
        msg.change_state = str(result.change_state)
        msg.num_active_tracks = int(result.num_active_tracks)
        msg.num_detections = int(result.num_detections)
        msg.num_matched = int(result.num_matched)
        msg.num_new = int(result.num_new)
        msg.num_lost_tracks = int(result.num_lost_tracks)

        pub.publish(msg)

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def on_picker_event(self, msg: PickerEvent):
        self.event_count += 1
        self._last_event_type = msg.event_type
        context = self._msg_to_context(msg)

        if msg.event_type == "ITEM_START":
            self.active_item_ctx = {
                "order_id": context.get("order_id", ""),
                "hu_id": context.get("hu_id", ""),
                "sku": context.get("sku", ""),
                "quantity": int(context.get("quantity", 0)),
            }
            self.state_machine.process_event(msg.event_type, context)
            self.task_controller.on_item_start(self.active_item_ctx)

        elif msg.event_type == "ITEM_END":
            self.state_machine.process_event(msg.event_type, context)
            self.task_controller.on_item_end(context)
            self.active_item_ctx = None

        elif msg.event_type == "ORDER_END":
            self.state_machine.process_event(msg.event_type, context)
            order_summary = self.task_controller.on_order_end()
            self.active_item_ctx = None
            self.current_order_data = None
            self._publish_order_summary(order_summary)

        else:
            self.state_machine.process_event(msg.event_type, context)

        self._publish_state_machine_summary()
        self._print_runtime_summary()

    def on_picker_plan(self, msg: PickerPlan):
        self._last_event_type = "ORDER_START"

        hus_list = []
        for hu in msg.hus:
            hus_list.append({
                "hu_id": hu.hu_id,
                "items": [
                    {"sku": item.sku, "description": item.description, "quantity": item.quantity}
                    for item in hu.items
                ],
            })

        self.current_order_data = {
            "order_id": msg.order_id,
            "description": msg.description,
            "hus": hus_list,
        }

        context = self._plan_msg_to_context(msg)
        self.state_machine.process_event("ORDER_START", context)
        self._publish_state_machine_summary()
        self._print_runtime_summary()

    # ------------------------------------------------------------------
    # Message conversion
    # ------------------------------------------------------------------

    def _msg_to_context(self, msg: PickerEvent) -> dict:
        ctx = {
            "order_id": msg.order_id,
            "hu_id": msg.hu_id,
            "sku": msg.sku,
            "description": msg.description,
            "quantity": msg.quantity,
            "item_index": msg.item_index,
            "hu_index": msg.hu_index,
            "total_hus": msg.total_hus,
        }
        if msg.event_type == PickerEvent.ORDER_START:
            ctx["hus"] = []
        return ctx

    def _plan_msg_to_context(self, msg: PickerPlan) -> dict:
        hus_data = []
        for hu in msg.hus:
            items_data = [
                {"sku": item.sku, "description": item.description, "quantity": item.quantity}
                for item in hu.items
            ]
            hus_data.append({"hu_id": hu.hu_id, "items": items_data})
        return {
            "order_id": msg.order_id,
            "description": msg.description,
            "total_hus": msg.total_hus,
            "hus": hus_data,
        }

    # ------------------------------------------------------------------
    # Publishers
    # ------------------------------------------------------------------

    def _publish_state_machine_summary(self):
        sm = self.state_machine
        summary = {
            "version": "1.0",
            "timestamp": time.time(),
            "order_state": sm.order_state.value if sm.order_state else "IDLE",
            "order_id": sm.order_id or None,
            "is_active": sm.is_in_progress or sm.is_paused,
            "active_hu_id": sm.active_hu_id,
            "hu_count": sm.hu_count,
            "active_item": dict(self.active_item_ctx) if self.active_item_ctx else None,
        }
        summary_json = json.dumps(summary, sort_keys=True)
        if summary_json != self._last_state_summary:
            self._last_state_summary = summary_json
            msg = String()
            msg.data = summary_json
            self.state_pub.publish(msg)

    def _publish_order_summary(self, order_summary: dict):
        msg = String()
        msg.data = json.dumps(order_summary, default=str)
        self.summary_pub.publish(msg)
        self.get_logger().info("Order summary published to /inventory/summary")

    # ------------------------------------------------------------------
    # Terminal logging
    # ------------------------------------------------------------------

    def _on_state_machine_change(self, sm: OrderStateMachine):
        pass

    def _on_hu_state_change(self, hu, old_state: HUState, new_state: HUState):
        pass

    def _print_startup_summary(self):
        print("\n" + "=" * 70)
        print("  CURSOR VISION - INVENTORY NODE STARTUP")
        print("=" * 70)

        checks = [
            ("State Machine", self.state_machine is not None),
            ("Detector", self.detector is not None or not self.enable_detection),
        ]
        for cam_name in self.camera_names:
            ok = cam_name in self.camera_pipelines
            checks.append((f"Pipeline [{cam_name}]", ok))

        print("\n[Startup Check]")
        for name, ok in checks:
            print(f"  - {name:.<40} {'OK' if ok else 'ERROR'}")

        print(f"\n[Configuration]")
        print(f"  Cameras .................. {', '.join(self.camera_names)}")
        print(f"  Detection Enabled ........ {'YES' if self.enable_detection else 'NO'}")
        print(f"  Detection FPS ............ {self.detection_fps}")
        print(f"  Detection Mode ........... {self.detection_mode}")
        print(f"  Debug KF Save ............ {'YES' if self.debug_save_keyframes else 'NO'}")
        print(f"  Debug Counter Video ...... {'YES' if self.debug_save_counter_video else 'NO'}")

        print("\n" + "=" * 70)
        print("  System ready. Waiting for events...")
        print("=" * 70 + "\n")

    def _print_runtime_summary(self):
        sm = self.state_machine
        print("\n" + "-" * 70)
        print("  STATE UPDATE")
        print("-" * 70)
        print(f"  Event .................... {self._last_event_type or 'N/A'}")
        print(f"  Order State .............. {sm.order_state.value if sm.order_state else 'IDLE'}")
        print(f"  Order ID ................. {sm.order_id or 'N/A'}")
        print(f"  Active HU ................ {sm.active_hu_id or 'None'}")
        print(f"  Task Active .............. {'YES' if self.task_controller.task_active else 'NO'}")

        if self.active_item_ctx:
            print(f"  Active SKU ............... {self.active_item_ctx.get('sku', 'N/A')}")
            print(f"  Expected Quantity ........ {self.active_item_ctx.get('quantity', 0)}")

        for cam_name, pipeline in self.camera_pipelines.items():
            units = pipeline.counter_bridge.running_units
            print(f"  [{cam_name}] Units ....... {units:.3f}")

        print("-" * 70 + "\n")

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def cleanup(self):
        for pipeline in self.camera_pipelines.values():
            pipeline.stop()
        self.current_order_data = None
        self.active_item_ctx = None


def main(args=None):
    rclpy.init(args=args)
    node = InventoryNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.cleanup()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
