"""ROS 2 node ``dock_fuser``: subscriptions, parameters, TF and pose output (§3, §10).

All estimation logic lives in ``core/`` (no ROS imports there). This node only

1. loads the parameters, the camera intrinsics (YAML) and the AUV / station
   URDFs, and builds the core :class:`~race_auv_factorgraph.core.backend.estimator.DockEstimator`;
2. feeds EKF odometry and per-camera AprilTag corner detections into the
   :class:`~race_auv_factorgraph.core.frontend.scheduler.KeyframeScheduler`, and each
   keyframe it releases into the estimator;
3. publishes the results.

Inputs
------
``odom_topic`` (``nav_msgs/Odometry``)
    The ``robot_localization`` EKF output: ``odom -> base_link`` pose.
``cameras.<name>.detections_topic`` (``apriltag_msgs/AprilTagDetectionArray``)
    Corner detections of one camera, published by ``apriltag_detector_node``
    on its ``apriltag_detection/tags`` topic. The corners must be in the
    **rectified** image, in the canonical order (see
    ``core/common/model.py:tag_corner_points``). ``header.stamp`` is the image stamp.
    One camera only. Every message, also one without detections, becomes
    one keyframe at that stamp: the detector's ``publish_rate`` (5 Hz on the
    vehicle) is the graph's keyframe rate.

Timing: only header stamps are used, never arrival times, so zenoh latency
between the front-seat computer (EKF) and the Jetson (camera, detector,
this node) does not matter. The two clocks must be synchronized (chrony);
``ekf_age_s`` in ``/diagnostics`` exposes an offset.

Outputs (§10)
-------------
TF ``graph_frame -> station_base_frame`` (e.g. ``race_auv/odom -> race_station/base_link``)
    ``E_k * X_k^-1 * S * T_dock_base``, published at the EKF rate with each
    EKF message's stamp. Composed with the EKF's ``odom -> base_link`` this
    gives ``base_link -> station = X_k^-1 * S * T_dock_base``, the graph's
    drift-corrected relative pose. Corrections are rate-limited.
TF ``station_base_frame -> dock_frame`` (static, optional)
    From the station URDF, so ``dock_point`` exists even when the station's
    own ``robot_state_publisher`` is not running.
``dock_pose_topic`` (``geometry_msgs/PoseWithCovarianceStamped``)
    ``X_k^-1 * S``: dock_point in ``base_frame``, once per keyframe while
    tracking. The covariance is that of the docking pose itself
    (``estimator.covariance_mode``, default "relative"), expressed along the
    ``base_frame`` axes in ROS order.
``/diagnostics`` (``diagnostic_msgs/DiagnosticArray``)
    Station state, accepted/rejected tags, update time, init progress.

Debug outputs (``publish_debug: true``), one message per keyframe
-----------------------------------------------------------------
``~/auv_pose`` (``geometry_msgs/PoseStamped``, ``graph_frame``)
    ``X_k``: the graph's drift-corrected AUV pose as estimated *at that
    keyframe* (the real-time estimate, not re-smoothed later).
``~/station_pose`` (``geometry_msgs/PoseStamped``, ``graph_frame``)
    ``S``: dock_point in odom, while tracking.
``~/keyframe`` (``std_msgs/String``, JSON)
    ``{"stamp", "state", "camera", "n_tags", "accepted": [tags],
    "rejected": [[tag, reason]], "initialized_now", "recovered",
    "seeded_from_tag", "dvl_sigma", "init_note", "update_ms"}``. ``rejected``
    are detections that could not be used (tag not on the station, PnP failed).
"""

from __future__ import annotations

import json
import logging
import math
from typing import Dict, List, Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy

import gtsam
import numpy as np
from apriltag_msgs.msg import AprilTagDetectionArray
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from geometry_msgs.msg import PoseWithCovarianceStamped, TransformStamped, TwistWithCovarianceStamped
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import CameraInfo
from std_msgs.msg import String
from tf2_ros import StaticTransformBroadcaster, TransformBroadcaster

from ..core.backend.estimator import DockEstimator, KeyframeResult
from ..core.common.geometry import gtsam_cov_to_ros, pose_from_quat_xyzw, quat_xyzw, step_towards
from ..core.common.model import Detection, TagFrame, normalize_family, tag_key_str
from ..core.frontend.scheduler import KeyframeScheduler
from ..core.init.state_machine import StationState
from .params import NODE_DEFAULTS, build_setup, camera_defaults, estimator_defaults

# Reliable, like the publishers (the EKF and the tag detector). A small
# queue absorbs short processing stalls during an iSAM2 update.
_QOS = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.VOLATILE,
                  history=HistoryPolicy.KEEP_LAST, depth=50)
# DVL velocity: only its arrival matters; best effort matches any publisher.
_DVL_QOS = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.VOLATILE,
                      history=HistoryPolicy.KEEP_LAST, depth=10)
# The tag detector's rectified camera_info: published once, latched.
_LATCHED_QOS = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                          durability=DurabilityPolicy.TRANSIENT_LOCAL,
                          history=HistoryPolicy.KEEP_LAST, depth=1)


# A detector or EKF stream silent for longer than this is reported as stalled
# in /diagnostics (the detector runs at 5 Hz, the EKF at >= 20 Hz).
_STALL_S = 1.0


def _stamp_to_sec(stamp) -> float:
    return float(stamp.sec) + 1e-9 * float(stamp.nanosec)


def _sec_to_stamp(t: float):
    from builtin_interfaces.msg import Time
    sec = int(math.floor(t))
    return Time(sec=sec, nanosec=int(round((t - sec) * 1e9)) % 1_000_000_000)


class _RosLogHandler(logging.Handler):
    """Forward Python ``logging`` records from ``core/`` to the node's ROS logger."""

    def __init__(self, node: Node) -> None:
        super().__init__()
        self._log = node.get_logger()

    def emit(self, record: logging.LogRecord) -> None:
        msg = self.format(record)
        if record.levelno >= logging.ERROR:
            self._log.error(msg)
        elif record.levelno >= logging.WARNING:
            self._log.warn(msg)
        else:
            self._log.info(msg)


class DockFuserNode(Node):

    def __init__(self) -> None:
        super().__init__("dock_fuser")
        log = logging.getLogger("race_auv_factorgraph")
        log.addHandler(_RosLogHandler(self))
        log.setLevel(logging.INFO)
        log.propagate = False

        # ================================================================ parameters
        # Defaults live in ros/params.py, shared with the offline bag replay.
        # Camera parameters are declared after camera_names is known.
        for name, default in {**NODE_DEFAULTS, **estimator_defaults()}.items():
            self.declare_parameter(name, default)
        for cam in self.get_parameter("camera_names").value:
            for name, default in camera_defaults(cam).items():
                self.declare_parameter(name, default)

        g = lambda name: self.get_parameter(name).value  # noqa: E731
        self.graph_frame = g("graph_frame")
        self.base_frame = g("base_frame")
        self.station_base_frame = g("station_base_frame")
        self.dock_frame = g("dock_frame")
        self.max_trans_rate = float(g("output_max_trans_rate"))
        self.max_rot_rate = float(g("output_max_rot_rate"))

        # ================================================================ models
        setup = build_setup(g, self.get_logger().warn)
        self.setup = setup
        # Shared with the estimator: a camera whose intrinsics come from the
        # detector's camera_info is added to this dict when that arrives.
        self.cameras = setup.cameras
        self.tags = setup.tags
        self.T_dock_base = setup.T_dock_base
        self._camera_frames = setup.camera_frames
        cfg = setup.cfg

        self.estimator = DockEstimator(cfg, self.cameras, self.tags, self.T_dock_base)
        # One keyframe per detector message, released once the EKF covers its
        # stamp (see core/frontend/scheduler.py for the timing rules).
        self.scheduler = KeyframeScheduler(ekf_buffer_seconds=float(g("ekf_buffer_seconds")))

        # ================================================================ output state
        self._target: Optional[gtsam.Pose3] = None     # latest odom -> station base from the graph
        self._published: Optional[gtsam.Pose3] = None  # rate-limited value actually published
        self._last_tf_stamp: Optional[float] = None
        self._last_state = self.estimator.state
        self._unknown_tags: Dict[str, int] = {}
        self._last_result: Optional[KeyframeResult] = None
        # Local (Jetson) clock time at which the newest EKF / detector message
        # arrived, with that message's header stamp. Used only for the
        # ekf_age_s / frame_age_s diagnostics, never for estimation.
        self._ekf_seen: Optional[tuple] = None    # (receive time, header stamp)
        self._frame_seen: Optional[tuple] = None

        # ================================================================ ROS I/O
        self.tf_pub = TransformBroadcaster(self)
        self.dock_pub = self.create_publisher(PoseWithCovarianceStamped, g("dock_pose_topic"), 10)
        self.diag_pub = self.create_publisher(DiagnosticArray, "/diagnostics", 10)
        self.debug = bool(g("publish_debug"))
        if self.debug:
            self.dbg_auv_pub = self.create_publisher(PoseStamped, "~/auv_pose", 50)
            self.dbg_station_pub = self.create_publisher(PoseStamped, "~/station_pose", 50)
            self.dbg_kf_pub = self.create_publisher(String, "~/keyframe", 50)
        if g("publish_station_static_tf"):
            self._static_pub = StaticTransformBroadcaster(self)
            self._static_pub.sendTransform(self._tf_msg(
                self.T_dock_base.inverse(), self.station_base_frame, self.dock_frame,
                self.get_clock().now().to_msg()))

        self.create_subscription(Odometry, g("odom_topic"), self._odom_cb, _QOS)
        # DVL bottom lock: only the message stamps matter (silence = no lock).
        self._dvl_seen: Optional[tuple] = None
        if g("dvl_topic"):
            self.create_subscription(TwistWithCovarianceStamped, g("dvl_topic"), self._dvl_cb,
                                     _DVL_QOS)
            self.get_logger().info(f"DVL bottom lock from {g('dvl_topic')}")
        for name, topic in setup.camera_topics.items():
            self.create_subscription(AprilTagDetectionArray, topic,
                                     lambda msg, n=name: self._detections_cb(n, msg), _QOS)
            self.get_logger().info(f"camera {name}: detections on {topic}")
        # Rectified intrinsics, latched by the tag detector. Subscribed for
        # every camera: it completes a camera without YAML intrinsics, and
        # cross-checks one that has them.
        for name, topic in setup.camera_info_topics.items():
            self.create_subscription(CameraInfo, topic,
                                     lambda msg, n=name: self._camera_info_cb(n, msg), _LATCHED_QOS)
            if name in self.cameras:
                self.get_logger().info(f"camera {name}: intrinsics from the YAML, "
                                       f"cross-checked against {topic}")
            else:
                self.get_logger().info(f"camera {name}: waiting for intrinsics on {topic}")
        # No processing timer: keyframes are run from the two callbacks above,
        # the moment the scheduler can release them (see _process).
        self.create_timer(1.0, self._publish_diagnostics)

        self.get_logger().info(
            f"dock_fuser ready: fixed-lag {cfg.smoother_lag:g} s, {len(setup.camera_topics)} cameras, "
            f"tags {[tag_key_str(k) for k in self.tags]}, odom={g('odom_topic')}")

    # ==================================================================== callbacks
    def _dvl_cb(self, msg: TwistWithCovarianceStamped) -> None:
        """Record that the DVL had bottom lock at this stamp (the velocity itself is not used)."""
        t = _stamp_to_sec(msg.header.stamp)
        self._dvl_seen = (self._now(), t)
        self.estimator.dvl.add(t)

    def _odom_cb(self, msg: Odometry) -> None:
        """Buffer the EKF pose, republish the station TF, run any frame it unblocks.

        A detector frame waits until an EKF pose stamped at or after it has
        arrived, so a new EKF message is what usually releases the frame.
        """
        t = _stamp_to_sec(msg.header.stamp)
        self._ekf_seen = (self._now(), t)
        p, q = msg.pose.pose.position, msg.pose.pose.orientation
        self.scheduler.add_odometry(t, pose_from_quat_xyzw((p.x, p.y, p.z), (q.x, q.y, q.z, q.w)))
        self._process()
        self._publish_station_tf(t, msg.header.stamp)

    def _detections_cb(self, camera: str, msg: AprilTagDetectionArray) -> None:
        """Convert one detector message into a TagFrame and queue it as a keyframe.

        Every message is queued, also one with no detections: it still
        becomes a keyframe (odometry only), see core/frontend/scheduler.py.
        """
        if msg.header.frame_id and msg.header.frame_id != self._camera_frames[camera]:
            self.get_logger().warn(
                f"{camera}: detections stamped in frame '{msg.header.frame_id}', expected "
                f"'{self._camera_frames[camera]}'. Check the topic mapping.",
                throttle_duration_sec=10.0)
        dets: List[Detection] = []
        for d in msg.detections:
            key = (normalize_family(d.family), int(d.id))
            if key not in self.tags:
                name = tag_key_str(key)
                self._unknown_tags[name] = self._unknown_tags.get(name, 0) + 1
                continue
            corners = np.array([[c.x, c.y] for c in d.corners], dtype=float)
            dets.append(Detection(camera, key, corners))
        t = _stamp_to_sec(msg.header.stamp)
        self._frame_seen = (self._now(), t)
        self.scheduler.add_frame(TagFrame(t, camera, dets))
        self._process()  # runs now if the EKF already covers t, else on a later EKF message

    def _camera_info_cb(self, camera: str, msg: CameraInfo) -> None:
        """Take the camera's rectified intrinsics from the detector (or check the YAML ones).

        The detector's camera_info describes its rectified image (``k`` = the
        pinhole matrix the corners follow, zero distortion), so it is used
        as is. Until it arrives, frames wait in the scheduler (see _process).
        """
        cam = self.cameras.get(camera)
        if cam is None:
            cam = self.setup.add_camera_info(camera, msg.k, msg.width, msg.height)
            self.get_logger().info(
                f"camera {camera}: rectified intrinsics from camera_info: {cam.width}x{cam.height} "
                f"fx {cam.fx:.4f} fy {cam.fy:.4f} cx {cam.cx:.4f} cy {cam.cy:.4f}")
            return
        diff = max(abs(cam.fx - msg.k[0]), abs(cam.fy - msg.k[4]),
                   abs(cam.cx - msg.k[2]), abs(cam.cy - msg.k[5]))
        if diff > 0.5 or (cam.width, cam.height) != (msg.width, msg.height):
            self.get_logger().warn(
                f"camera {camera}: YAML intrinsics differ from the detector's camera_info by up "
                f"to {diff:.1f} px; the YAML values are used. Remove them from the config to use "
                f"camera_info (fx {msg.k[0]:.4f} fy {msg.k[4]:.4f} cx {msg.k[2]:.4f} "
                f"cy {msg.k[5]:.4f}, {msg.width}x{msg.height}).")

    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _process(self) -> None:
        """Run every keyframe the scheduler releases. Never raises.

        Nothing runs before every camera has its intrinsics: the frames wait
        in the scheduler meanwhile (the camera_info is latched, so this is
        normally only the first moment after start-up).
        """
        if self.setup.missing_intrinsics():
            return
        for kf in self.scheduler.poll():
            try:
                res = self.estimator.process_keyframe(kf.stamp, kf.E, kf.frame)
            except Exception as e:  # the node must never die on an estimator error
                self.get_logger().error(f"keyframe at {kf.stamp:.3f} failed: {e!r}")
                continue
            self._handle_result(res)
            if self.debug:
                self._publish_debug(res, kf.frame.camera, len(kf.frame.detections))

    # ==================================================================== outputs
    def _handle_result(self, res: KeyframeResult) -> None:
        self._last_result = res
        if res.state != self._last_state:
            self.get_logger().info(f"station state {self._last_state.value} -> {res.state.value}")
            self._last_state = res.state
        if res.initialized_now:
            self._published = None  # first value: jump, do not rate-limit from nothing
        if res.state != StationState.TRACKING or res.S is None:
            if res.state != StationState.TRACKING:
                self._target = self._published = None
            return

        # odom -> station base = E_k * X_k^-1 * S * T_dock_base = D^-1 * S * T_dock_base (§10)
        self._target = res.drift_correction.inverse().compose(res.S).compose(self.T_dock_base)

        # /dock_point/pose: X_k^-1 * S in base_link, with the docking-pose covariance.
        dock = res.dock_in_base
        msg = PoseWithCovarianceStamped()
        msg.header.stamp = _sec_to_stamp(res.stamp)
        msg.header.frame_id = self.base_frame
        t = dock.translation()
        x, y, z, w = quat_xyzw(dock)
        msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z = map(float, t)
        o = msg.pose.pose.orientation
        o.x, o.y, o.z, o.w = x, y, z, w
        if res.dock_cov is not None:
            msg.pose.covariance = gtsam_cov_to_ros(res.dock_cov, dock.rotation().matrix()).ravel().tolist()
        else:
            msg.pose.covariance = [-1.0] + [0.0] * 35  # ROS convention: covariance unknown
        self.dock_pub.publish(msg)

    def _publish_debug(self, res: KeyframeResult, camera: str, n_tags: int) -> None:
        """Per-keyframe analysis topics (see the module docstring)."""
        def pose_msg(T: gtsam.Pose3) -> PoseStamped:
            m = PoseStamped()
            m.header.stamp = _sec_to_stamp(res.stamp)
            m.header.frame_id = self.graph_frame
            t = T.translation()
            m.pose.position.x, m.pose.position.y, m.pose.position.z = map(float, t)
            o = m.pose.orientation
            o.x, o.y, o.z, o.w = quat_xyzw(T)
            return m

        self.dbg_auv_pub.publish(pose_msg(res.X))
        if res.S is not None:
            self.dbg_station_pub.publish(pose_msg(res.S))
        self.dbg_kf_pub.publish(String(data=json.dumps({
            "stamp": res.stamp,
            "state": res.state.value,
            "camera": camera,
            "n_tags": n_tags,   # known tags in the detector message (0: odometry-only keyframe)
            "accepted": [tag_key_str(t) for t in res.accepted],
            "rejected": [[tag_key_str(t), r] for t, r in res.rejected],
            "initialized_now": res.initialized_now,
            "recovered": res.recovered,
            "seeded_from_tag": res.seeded_from_tag,
            "dvl_sigma": round(res.dvl_extra_sigma, 4),
            "init_note": res.init_note,
            "update_ms": round(res.update_time * 1000.0, 3),
        })))

    def _publish_station_tf(self, t: float, stamp) -> None:
        """Publish ``graph_frame -> station_base_frame``, rate-limited toward the graph's value."""
        if self._target is None:
            return
        if self._published is None:
            self._published = self._target
        elif self._last_tf_stamp is not None:
            dt = max(0.0, t - self._last_tf_stamp)
            self._published = step_towards(self._published, self._target,
                                           self.max_trans_rate * dt, self.max_rot_rate * dt)
        self._last_tf_stamp = t
        self.tf_pub.sendTransform(self._tf_msg(self._published, self.graph_frame,
                                               self.station_base_frame, stamp))

    @staticmethod
    def _tf_msg(T: gtsam.Pose3, parent: str, child: str, stamp) -> TransformStamped:
        msg = TransformStamped()
        msg.header.stamp = stamp
        msg.header.frame_id = parent
        msg.child_frame_id = child
        tr = T.translation()
        msg.transform.translation.x, msg.transform.translation.y, msg.transform.translation.z = \
            map(float, tr)
        r = msg.transform.rotation
        r.x, r.y, r.z, r.w = quat_xyzw(T)
        return msg

    def _publish_diagnostics(self) -> None:
        est, sched, res = self.estimator, self.scheduler, self._last_result
        st = DiagnosticStatus()
        st.name = "dock_fuser"
        st.hardware_id = "race_auv"
        st.level = DiagnosticStatus.OK if est.state == StationState.TRACKING else DiagnosticStatus.WARN
        st.message = est.state.value
        kv = {
            "state": est.state.value,
            "keyframes": str(est.k + 1),
            "tag_factors_total": str(est.n_tag_factors),
            "tag_rejections_total": str(est.n_rejected),
            "backend_recoveries": str(est.n_recoveries),
            # Detector messages dropped: re-published / out-of-order stamps,
            # older than the EKF buffer, or pushed out of a full queue.
            "frames_dropped": str(sched.n_dropped),
            # Frames waiting for an EKF pose at or after their stamp. Grows
            # when the EKF stream from the front-seat computer stalls.
            "frames_waiting": str(sched.n_pending),
            "unknown_tags": ", ".join(f"{k} x{n}" for k, n in self._unknown_tags.items()),
        }
        if res is not None:
            kv["last_update_ms"] = f"{res.update_time * 1000:.1f}"
            kv["last_accepted"] = ", ".join(tag_key_str(t) for t in res.accepted)
            kv["last_rejected"] = "; ".join(f"{tag_key_str(t)}: {r}" for t, r in res.rejected)
            kv["init_note"] = res.init_note
            D = res.drift_correction
            kv["drift_correction_m"] = f"{float(np.linalg.norm(D.translation())):.3f}"
            kv["drift_correction_yaw_deg"] = f"{math.degrees(D.rotation().yaw()):.2f}"
        # Timing across machines. age = (local clock when the message arrived)
        # - (its header stamp), i.e. transport + processing delay *plus* any
        # clock offset between the computers. ekf_age_s should be a few ms
        # (zenoh latency); a negative or large value means the Pi and the
        # Jetson clocks disagree (check chrony). frame_age_s is the detector
        # latency. "*_silent_s" is how long ago the last message arrived.
        now = self._now()
        for name, seen in (("ekf", self._ekf_seen), ("frame", self._frame_seen),
                           *((("dvl", self._dvl_seen),) if self.estimator.dvl.active else ())):
            if seen is None:
                kv[f"{name}_age_s"] = kv[f"{name}_silent_s"] = "never received"
                continue
            kv[f"{name}_age_s"] = f"{seen[0] - seen[1]:.3f}"
            kv[f"{name}_silent_s"] = f"{now - seen[0]:.2f}"
        # No detector messages means no new keyframes at all (the detector is
        # the keyframe clock), so flag a stall even while TRACKING.
        stalled = [n for n, seen in (("detector", self._frame_seen), ("EKF", self._ekf_seen))
                   if seen is None or now - seen[0] > _STALL_S]
        if stalled:
            st.level = DiagnosticStatus.WARN
            st.message = f"{est.state.value}; no {' / '.join(stalled)} messages for > {_STALL_S:g} s"
        missing = self.setup.missing_intrinsics()
        if missing:
            st.level = DiagnosticStatus.WARN
            st.message = f"waiting for camera_info of {', '.join(missing)}"
        st.values = [KeyValue(key=k, value=v) for k, v in kv.items()]
        msg = DiagnosticArray()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.status = [st]
        self.diag_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = DockFuserNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
