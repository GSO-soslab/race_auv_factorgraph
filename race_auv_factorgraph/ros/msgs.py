"""Conversions between ROS messages and the core types (gtsam poses, tag detections).

The node and the offline tools that read recorded bags both use these, so a
message means the same thing live and offline. Message types only: nothing
here needs a running ROS node.
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

import gtsam
import numpy as np
from builtin_interfaces.msg import Time
from geometry_msgs.msg import Pose, TransformStamped

from ..core.common.geometry import pose_from_quat_xyzw, quat_xyzw
from ..core.common.model import Detection, TagKey, TagModel, normalize_family


def stamp_to_sec(stamp: Time) -> float:
    """``builtin_interfaces/Time`` -> seconds."""
    return float(stamp.sec) + 1e-9 * float(stamp.nanosec)


def sec_to_stamp(t: float) -> Time:
    """Seconds -> ``builtin_interfaces/Time``."""
    sec = int(math.floor(t))
    return Time(sec=sec, nanosec=int(round((t - sec) * 1e9)) % 1_000_000_000)


def pose_from_msg(msg) -> gtsam.Pose3:
    """The pose in a ``geometry_msgs/Pose``, or in any message that carries one.

    PoseStamped, PoseWithCovariance(Stamped) and Odometry nest the Pose under
    one or two ``.pose`` fields; they are unwrapped until the position is reached.
    """
    while not hasattr(msg, "position"):
        msg = msg.pose
    p, q = msg.position, msg.orientation
    return pose_from_quat_xyzw((p.x, p.y, p.z), (q.x, q.y, q.z, q.w))


def pose_to_msg(T: gtsam.Pose3) -> Pose:
    """gtsam pose -> ``geometry_msgs/Pose``."""
    msg = Pose()
    msg.position.x, msg.position.y, msg.position.z = map(float, T.translation())
    o = msg.orientation
    o.x, o.y, o.z, o.w = quat_xyzw(T)
    return msg


def transform_msg(T: gtsam.Pose3, parent: str, child: str, stamp: Time) -> TransformStamped:
    """TF message for the pose ``T`` of frame ``child`` in frame ``parent``."""
    msg = TransformStamped()
    msg.header.stamp = stamp
    msg.header.frame_id = parent
    msg.child_frame_id = child
    t = msg.transform.translation
    t.x, t.y, t.z = map(float, T.translation())
    r = msg.transform.rotation
    r.x, r.y, r.z, r.w = quat_xyzw(T)
    return msg


def tag_detections(camera: str, msg, tags: Dict[TagKey, TagModel]
                   ) -> Tuple[List[Detection], List[TagKey]]:
    """The station tags in an ``apriltag_msgs/AprilTagDetectionArray``, as core detections.

    Returns ``(detections, unknown)``: one :class:`Detection` (4 corners, in
    the message's order) per detection of a tag in ``tags``, and the keys of
    detected tags that are not on the station (ignored).
    """
    detections, unknown = [], []
    for d in msg.detections:
        key = (normalize_family(d.family), int(d.id))
        if key not in tags:
            unknown.append(key)
            continue
        corners = np.array([[c.x, c.y] for c in d.corners], dtype=float)
        detections.append(Detection(camera, key, corners))
    return detections, unknown
