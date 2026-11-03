"""Launch the race_auv_factorgraph node (simulation config by default).

Usage::

    ros2 launch race_auv_factorgraph factorgraph.launch.py
    ros2 launch race_auv_factorgraph factorgraph.launch.py config:=/path/to/params.yaml
    ros2 launch race_auv_factorgraph factorgraph.launch.py use_sim_time:=true   # bag replay with --clock

The node runs as /race_auv/race_auv_factorgraph, so its topics are
/race_auv/race_auv_factorgraph/<name> (docs/ROS_INTERFACE.md).

On the vehicle, use race_auv_bringup's bringup_camera_tag_factorgraph.launch.py
instead: it starts the cameras, the tag detectors and this node with the
vehicle config (race_auv_bringup/config/race_auv_factorgraph.yaml).
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
            "config", default_value=os.path.join(share, "config", "factorgraph_sim.yaml"),
            description="parameter file (default: simulation)"),
        DeclareLaunchArgument("use_sim_time", default_value="false"),
        Node(
            package="race_auv_factorgraph",
            executable="factorgraph_node",
            namespace="race_auv",
            name="race_auv_factorgraph",
            output="screen",
            parameters=[LaunchConfiguration("config"),
                        {"use_sim_time": LaunchConfiguration("use_sim_time")}],
        ),
    ])
