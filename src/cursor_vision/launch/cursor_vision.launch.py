#!/usr/bin/env python3
"""
Launch file for CursorVision: inventory_node + sap_event_node.

Usage:
    ros2 launch cursor_vision cursor_vision.launch.py
"""
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():

    inventory_cfg = PathJoinSubstitution([
        FindPackageShare("cursor_vision"),
        "config",
        "inventory_node_config.yaml",
    ])
    sap_cfg = PathJoinSubstitution([
        FindPackageShare("cursor_vision"),
        "config",
        "sap_event_config.yaml",
    ])

    inventory_node = Node(
        package="cursor_vision",
        executable="inventory_node",
        name="inventory_node",
        output="screen",
        emulate_tty=True,
        parameters=[inventory_cfg],
    )

    sap_event_node = Node(
        package="cursor_vision",
        executable="sap_event_node",
        name="sap_event_node",
        output="screen",
        emulate_tty=True,
        parameters=[sap_cfg],
    )

    return LaunchDescription([
        inventory_node,
        sap_event_node,
    ])
