"""Reading recorded bags: shared by every offline tool.

* :func:`open_reader`, :func:`bag_start_time`: open a bag (mcap or sqlite3).
* :func:`bag_source`, :func:`source_start_time`: which recording a bag came
  from (``source.yaml``, written by ``launch/bag_test.launch.py`` and
  ``redetect_bag``), for plot titles and a time axis in seconds into the
  original recording.
* :func:`load_camera_info`: the detector's rectified intrinsics from the bag.
* :data:`LIVE_TOPICS`, :func:`live_topic`, :func:`is_init_jump`: the live
  node's recorded analysis topics, current and former names.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

# The live node's per-keyframe analysis topics, as a bag may hold them: the
# current name first, then the former dock_fuser node's (bags recorded before
# 2026-10-29, when the node became /race_auv/race_auv_factorgraph).
LIVE_TOPICS = {
    "auv_pose": ("/race_auv/race_auv_factorgraph/auv_pose", "/dock_fuser/auv_pose"),
    "station_pose": ("/race_auv/race_auv_factorgraph/dock_pose", "/dock_fuser/station_pose"),
    "keyframe": ("/race_auv/race_auv_factorgraph/keyframe", "/dock_fuser/keyframe"),
}

# Plots and summaries mark keyframes whose X_k initial value moved at least
# this far from the odometry prediction (KeyframeResult.init_shift): the tag
# came back after a gap and the drift of the gap was corrected at once. In
# steady tracking the shift is millimetres. Presentation only: the estimator
# itself has no such threshold.
INIT_JUMP_M = 0.05


def live_topic(role: str, bag_topics) -> Optional[str]:
    """The topic of ``LIVE_TOPICS[role]`` this bag has (None if neither)."""
    return next((t for t in LIVE_TOPICS[role] if t in bag_topics), None)


def is_init_jump(kf: Dict[str, Any]) -> bool:
    """Whether a recorded ``~/keyframe`` record marks an initial-value jump (``INIT_JUMP_M``).

    Records made before ``init_shift_m`` existed carry ``seeded_from_tag``
    (``relocalized`` in older ones) instead.
    """
    if "init_shift_m" in kf:
        return float(kf["init_shift_m"]) >= INIT_JUMP_M
    return bool(kf.get("seeded_from_tag", kf.get("relocalized")))


def open_reader(bag: str):
    """A ``rosbag2_py.SequentialReader`` on ``bag`` (mcap or sqlite3, by its files)."""
    import rosbag2_py
    reader = rosbag2_py.SequentialReader()
    storage_id = "mcap" if any(f.endswith(".mcap") for f in os.listdir(bag)) else "sqlite3"
    reader.open(rosbag2_py.StorageOptions(uri=bag, storage_id=storage_id),
                rosbag2_py.ConverterOptions("cdr", "cdr"))
    return reader


def bag_start_time(bag: str) -> float:
    """Recording start of the bag (seconds), from its metadata."""
    return open_reader(bag).get_metadata().starting_time.nanoseconds * 1e-9


def _source_yaml(bag: str) -> Optional[Dict[str, Any]]:
    """The bag's ``source.yaml`` (original recording and replayed window), if it has one."""
    path = os.path.join(bag, "source.yaml")
    if not os.path.exists(path):
        return None
    import yaml
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def bag_source(bag: str) -> str:
    """Display name of the recording a bag came from, for plot titles.

    The original recording (and the replayed window) from ``source.yaml``;
    without it, the bag's own directory name.
    """
    name = os.path.basename(os.path.normpath(bag))
    src = _source_yaml(bag)
    if src is None:
        return name
    out = str(src.get("source_bag", name))
    if float(src.get("start_offset", 0.0)):
        out += f" (from {float(src['start_offset']):g} s)"
    return out


def source_start_time(bag: str) -> float:
    """Start (s) of the original recording a bag came from, the zero of the plot time axes.

    From ``source.yaml``: the start of the recording at ``source_path`` if it
    is still there, else this bag's start minus the replayed
    ``start_offset``. Without ``source.yaml``: this bag's own start. Times on
    that axis then read as seconds into the original recording, whichever
    window of it the bag holds.
    """
    src = _source_yaml(bag)
    if src is None:
        return bag_start_time(bag)
    original = src.get("source_path", "")
    if original and os.path.isdir(original):
        return bag_start_time(original)
    return bag_start_time(bag) - float(src.get("start_offset", 0.0))


def load_camera_info(bag: str, setup) -> None:
    """Complete the cameras without config intrinsics from the bag's recorded camera_info.

    Same source as the live node: the tag detector's latched, rectified
    camera_info (``bag_test.launch.py`` records it). Older detection bags
    have none; their config must then set ``cameras.<name>.fx ...``.
    """
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from sensor_msgs.msg import CameraInfo

    missing = {setup.camera_info_topics[n]: n for n in setup.missing_intrinsics()}
    if not missing:
        return
    reader = open_reader(bag)
    reader.set_filter(rosbag2_py.StorageFilter(topics=list(missing)))
    while missing and reader.has_next():
        topic, data, _ = reader.read_next()
        if topic in missing:
            msg = deserialize_message(data, CameraInfo)
            setup.add_camera_info(missing.pop(topic), msg.k, msg.width, msg.height)
    if missing:
        raise RuntimeError(
            f"no intrinsics for camera(s) {sorted(missing.values())}: the config sets no "
            f"cameras.<name>.fx/fy/cx/cy/width/height and the bag has no {sorted(missing)}")
