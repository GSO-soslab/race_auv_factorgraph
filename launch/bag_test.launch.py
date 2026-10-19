"""Re-run tag detection + dock_fuser on a recorded bag window, and record the results.

Usage::

    ros2 launch race_auv_factorgraph bag_test.launch.py \\
        bag:=/home/dark3090/ros2bag/race/tank_test_09_16_26/rosbag2_2026_09_16-18_24_05 \\
        start_offset:=95 duration:=155 [rate:=1.0] [out:=/path/to/output_bag]

Then plot::

    ros2 run race_auv_factorgraph plot_bev <output_bag>

What runs
---------
1. ``ros2 bag record`` (sim time) of the inputs and every output of interest
   into ``out``. It starts first, so nothing published is missed.
2. ``tag_detector_node`` (``race_auv_camera_tag_factorgraph``, namespace
   ``cam_front``) on the recorded front-camera images: rectify, tag36h11,
   image checks, then the 4-corner ``apriltag_detection/tags``.
3. ``dock_fuser`` with the tank config (debug topics on).
4. ``ros2 bag play`` of the window ``[start_offset, start_offset + duration]``
   seconds from the start of the bag, with ``--clock``. Only the camera
   images, the recorded EKF odometry and the DVL velocity are replayed: the old detector
   outputs in the bag are deliberately left out.

Everything uses simulated time from the bag's ``/clock``, so stamps in the
output bag match the original recording. When playback ends, the launch
waits ``shutdown_delay`` seconds (letting the detector and the fuser finish
their last frames) and then shuts everything down. Stopping the recorder
with SIGINT closes the output bag cleanly.

Playback runs at ``rate`` (default 1.0 = real time). The detector processes
the newest image on each tick of its 5 Hz timer (``config/tank_tag_detector.yaml``),
i.e. every 3rd image of the 15 Hz camera, as on the vehicle.

Detection-only pass (for comparing backends offline with ``compare_backends``)::

    ros2 launch race_auv_factorgraph bag_test.launch.py bag:=<bag> start_offset:=0 \
        duration:=100000 run_fuser:=false record_images:=false

records just the corner detections and the EKF. Every backend can then be
replayed on exactly the same detections.
"""

import os
import time

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, EmitEvent, ExecuteProcess,
                            OpaqueFunction, RegisterEventHandler, TimerAction)
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

# Topics replayed from the input bag.
PLAY_TOPICS = [
    "/race/stellar2/image/compressed",   # front camera (cam_front)
    "/race_auv/odometry/filtered",        # recorded robot_localization EKF
    "/race_auv/nucleus_node/bottom_lock_velocity_common",   # DVL: silent while bottom lock is lost
]

# Topics recorded into the output bag.
RECORD_TOPICS = [
    "/race_auv/odometry/filtered",
    "/race_auv/nucleus_node/bottom_lock_velocity_common",
    "/cam_front/apriltag_detection/tags",
    "/cam_front/apriltag_detection/detections3d",
    "/cam_front/apriltag_detection/camera_info",
    "/cam_front/apriltag_detection/image",
    "/race_auv/dock_point/pose",
    "/dock_fuser/auv_pose",
    "/dock_fuser/station_pose",
    "/dock_fuser/keyframe",
    "/diagnostics",
    "/tf",
    "/tf_static",
]


def _write_source(out: str, bag: str, start_offset: str, duration: str) -> list:
    """Write ``<out>/source.yaml``: the original recording and the replayed window."""
    with open(os.path.join(out, "source.yaml"), "w", encoding="utf-8") as f:
        f.write(f"source_bag: {os.path.basename(os.path.normpath(bag))}\n"
                f"source_path: {os.path.abspath(bag)}\n"
                f"start_offset: {float(start_offset)}\n"
                f"duration: {float(duration)}\n")
    return []


def _setup(context, *args, **kwargs):
    lc = lambda name: LaunchConfiguration(name).perform(context)  # noqa: E731
    bag = os.path.expanduser(lc("bag"))
    if not bag:
        raise RuntimeError("bag:=<path to the input bag directory> is required")
    out = lc("out") or os.path.join(
        os.path.expanduser("~/ros2bag/dock_fuser_tests"),
        f"{os.path.basename(os.path.normpath(bag))}_t{lc('start_offset')}_"
        f"{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(os.path.dirname(out), exist_ok=True)

    run_fuser = lc("run_fuser").lower() in ("1", "true", "yes")
    topics = [t for t in RECORD_TOPICS
              if (lc("record_images").lower() in ("1", "true", "yes") or not t.endswith("/image"))
              and (run_fuser or not (t.startswith("/dock_fuser") or "dock_point" in t))]
    record = ExecuteProcess(
        cmd=["ros2", "bag", "record", "--use-sim-time", "-s", "mcap", "-o", out, *topics],
        output="screen", sigterm_timeout="10")

    detector = Node(
        package="race_auv_camera_tag_factorgraph", executable="tag_detector_node",
        namespace="cam_front", name="tag_detector", output="screen",
        parameters=[lc("detector_config"), {"use_sim_time": True}])

    fuser = Node(
        package="race_auv_factorgraph", executable="dock_fuser_node",
        name="dock_fuser", output="screen",
        parameters=[lc("fuser_config"), {"use_sim_time": True}])

    play = ExecuteProcess(
        cmd=["ros2", "bag", "play", bag, "--clock", "100",
             "--start-offset", lc("start_offset"),
             "--playback-duration", lc("duration"),
             "--rate", lc("rate"),
             "--topics", *PLAY_TOPICS],
        output="screen")

    return [
        record, detector, *([fuser] if run_fuser else []),
        # Give the recorder and the nodes time to start and discover each
        # other before the first message is played.
        TimerAction(period=float(lc("startup_delay")), actions=[play]),
        RegisterEventHandler(OnProcessExit(target_action=play, on_exit=[
            # Note which recording and window this bag came from, for the
            # plot titles (tools/replay_bag.py:bag_source).
            OpaqueFunction(function=lambda ctx: _write_source(out, bag, lc("start_offset"),
                                                              lc("duration"))),
            TimerAction(period=float(lc("shutdown_delay")),
                        actions=[EmitEvent(event=Shutdown(reason="bag playback finished"))]),
        ])),
        ExecuteProcess(cmd=["echo", f"[bag_test] recording to {out}"], output="screen"),
    ]


def generate_launch_description():
    share = get_package_share_directory("race_auv_factorgraph")
    return LaunchDescription([
        DeclareLaunchArgument("bag", default_value="", description="input bag directory"),
        DeclareLaunchArgument("start_offset", default_value="95",
                              description="seconds from the start of the bag"),
        DeclareLaunchArgument("duration", default_value="155", description="seconds to play"),
        DeclareLaunchArgument("rate", default_value="1.0", description="playback rate"),
        DeclareLaunchArgument("out", default_value="",
                              description="output bag (default ~/ros2bag/dock_fuser_tests/...)"),
        DeclareLaunchArgument("detector_config",
                              default_value=os.path.join(share, "config", "tank_tag_detector.yaml")),
        DeclareLaunchArgument("fuser_config",
                              default_value=os.path.join(share, "config", "tank_dock_fuser.yaml")),
        DeclareLaunchArgument("run_fuser", default_value="true",
                              description="false: detection-only pass (no dock_fuser)"),
        DeclareLaunchArgument("record_images", default_value="true",
                              description="record the annotated detector image (large; "
                                          "needs publish_image: true in the detector config)"),
        DeclareLaunchArgument("startup_delay", default_value="4.0"),
        DeclareLaunchArgument("shutdown_delay", default_value="3.0"),
        OpaqueFunction(function=_setup),
    ])
