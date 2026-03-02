#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
import sys

from ament_index_python.packages import get_package_share_directory
import rclpy
from rclpy.node import Node
import yaml

from cursor_vision.msg import PickerEvent

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class SapEventNode(Node):
    """
    Minimal SAP event node.

    Supported events:
      - ITEM_START
      - ITEM_END
      - ORDER_END
    """

    SUPPORTED_EVENTS = {"ITEM_START", "ITEM_END", "ORDER_END"}

    def __init__(self):
        super().__init__("sap_event_node")
        self.event_pub = self.create_publisher(PickerEvent, "/picker/events", 10)
        self.config = self._load_config()

        self.declare_parameter("host", self.config["server"]["host"])
        self.declare_parameter("port", self.config["server"]["port"])
        self.declare_parameter("api_endpoint", self.config["api"]["endpoint"])

        self.host = self.get_parameter("host").get_parameter_value().string_value
        self.port = self.get_parameter("port").get_parameter_value().integer_value
        self.endpoint = (
            self.get_parameter("api_endpoint").get_parameter_value().string_value
        )

        self.auth_enabled = bool(self.config.get("auth", {}).get("enabled", False))
        self.api_key = str(self.config.get("auth", {}).get("api_key", "")).strip()
        self.max_request_size = int(self.config.get("api", {}).get("max_request_size", 10_485_760))

        self._server = HTTPServer((self.host, self.port), self._make_handler())
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        self.get_logger().info(
            f"SAP signal API listening on http://{self.host}:{self.port}{self.endpoint}"
        )

    def _load_config(self) -> dict:
        try:
            pkg_share = get_package_share_directory("cursor_vision")
            config_path = Path(pkg_share) / "config" / "sap_event_config.yaml"
        except Exception:
            config_path = PROJECT_ROOT / "config" / "sap_event_config.yaml"

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        except Exception:
            return {
                "server": {"host": "0.0.0.0", "port": 8080},
                "auth": {"enabled": False, "api_key": ""},
                "api": {
                    "endpoint": "/api/signal",
                    "timeout": 30,
                    "max_request_size": 10_485_760,
                },
            }

    def _make_handler(self):
        node = self

        class Handler(BaseHTTPRequestHandler):
            def _send(self, code: int, data: dict):
                raw = json.dumps(data).encode("utf-8")
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def _auth_ok(self) -> bool:
                if not node.auth_enabled:
                    return True
                if not node.api_key:
                    return False
                header_key = self.headers.get("X-API-Key", "")
                bearer = self.headers.get("Authorization", "").replace("Bearer ", "")
                return header_key == node.api_key or bearer == node.api_key

            def do_POST(self):
                if self.path != node.endpoint:
                    return self._send(404, {"ok": False, "error": "not_found"})

                if not self._auth_ok():
                    return self._send(401, {"ok": False, "error": "unauthorized"})

                length = int(self.headers.get("Content-Length", "0"))
                if length > node.max_request_size:
                    return self._send(400, {"ok": False, "error": "payload_too_large"})

                raw = self.rfile.read(length) if length > 0 else b"{}"
                try:
                    payload = json.loads(raw.decode("utf-8"))
                except Exception:
                    return self._send(400, {"ok": False, "error": "invalid_json"})

                try:
                    ok, error = node.handle_payload(payload)
                except Exception as exc:
                    node.get_logger().error(f"handle_payload error: {exc}")
                    return self._send(500, {"ok": False, "error": "internal_error"})
                if not ok:
                    return self._send(400, {"ok": False, "error": error})
                return self._send(200, {"ok": True})

            def log_message(self, fmt, *args):
                return

        return Handler

    def handle_payload(self, payload: dict) -> tuple[bool, str]:
        event_type = str(payload.get("event_type", "")).strip().upper()
        if event_type not in self.SUPPORTED_EVENTS:
            return False, "unsupported_event_type"

        order_id = str(payload.get("order_id", "")).strip()
        if not order_id:
            return False, "missing_order_id"

        if event_type == "ITEM_START":
            hu_id = str(payload.get("hu_id", "")).strip()
            sku = str(payload.get("sku", "")).strip()
            quantity = payload.get("quantity", None)
            if not hu_id or not sku or quantity is None:
                return False, "item_start_missing_fields"
        elif event_type == "ITEM_END":
            hu_id = str(payload.get("hu_id", "")).strip()
            if not hu_id:
                return False, "item_end_missing_hu_id"

        msg = PickerEvent()
        msg.event_type = event_type
        msg.order_id = order_id
        msg.hu_id = str(payload.get("hu_id", ""))
        msg.sku = str(payload.get("sku", ""))
        msg.description = str(payload.get("description", ""))
        msg.quantity = int(payload.get("quantity", 0) or 0)
        msg.item_index = int(payload.get("item_index", 0) or 0)
        msg.hu_index = int(payload.get("hu_index", 0) or 0)
        msg.total_hus = int(payload.get("total_hus", 0) or 0)
        msg.timestamp = self.get_clock().now().to_msg()
        self.event_pub.publish(msg)

        # Keep full payload for debugging/extensibility without changing ROS msg schema.
        self.get_logger().info(
            json.dumps(
                {
                    "component": "sap_event_node",
                    "event_type": event_type,
                    "order_id": order_id,
                    "payload": payload,
                },
                ensure_ascii=True,
            )
        )
        return True, ""

    def destroy_node(self):
        try:
            self._server.shutdown()
            self._server.server_close()
        except Exception:
            pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = SapEventNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
