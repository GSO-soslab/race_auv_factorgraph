"""Launch the dock_fuser node.

Usage::

    ros2 launch race_auv_factorgraph dock_fuser.launch.py                 # simulation config
    ros2 launch race_auv_factorgraph dock_fuser.launch.py config:=/path/to/dock_fuser.yaml

On the vehicle, use race_auv_bringup's bringup_camera_tag_factorgraph.launch.py
instead: it starts the cameras, the tag detectors and dock_fuser with the
vehicle config (race_auv_bringup/config/dock_fuser.yaml).
    ros2 launch race_auv_factorgraph dock_fuser.launch.py use_sim_time:=true   # bag replay with --clock
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    share = get_package_share_directory("race_auv_factorgraph")
    return LaunchDescription([
        DeclareLaunchArgument(
            "config", default_value=os.path.join(share, "config", "dock_fuser_sim.yaml"),
            description="dock_fuser parameter file (default: simulation)"),
        DeclareLaunchArgument("use_sim_time", default_value="false"),
        Node(
            package="race_auv_factorgraph",
            executable="dock_fuser_node",
            name="dock_fuser",
            output="screen",
            parameters=[LaunchConfiguration("config"),
                        {"use_sim_time": LaunchConfiguration("use_sim_time")}],
        ),
    ])
