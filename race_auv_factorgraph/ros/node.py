"""ROS 2 node ``race_auv_factorgraph``: the ROS interface of the dock factor graph.

Runs as ``/race_auv/race_auv_factorgraph`` (namespace ``race_auv``), so every
topic below is ``/race_auv/race_auv_factorgraph/<name>``. The estimation
itself is in ``core/`` (no ROS there). This node only

1. loads the parameters, the AUV / station URDFs and the camera intrinsics,
   and builds the :class:`~race_auv_factorgraph.core.backend.estimator.DockEstimator`;
2. feeds it the EKF odometry and the AprilTag corner detections: every
   message of the keyframe camera's detector becomes one keyframe (5 Hz);
3. publishes the results below.

The full interface, the architecture and the mapping from the old dock_fuser
topics are in ``docs/ROS_INTERFACE.md``.

Frames
------
``race_auv/odom`` (``odom_frame``)
    The EKF's frame (robot_localization, ``odom -> base_link`` at 20 Hz).
    Continuous, but drifts relative to the station.
``race_auv/odom_graph`` (``odom_graph_frame``)
    The factor graph's drift-corrected odom frame. It starts equal to
    ``odom`` (the first keyframe is pinned to the EKF) and the two separate
    once tags pull the graph's AUV pose away from the drifting EKF. The
    station is fixed in it. This is the frame for docking guidance and for
    display (Foxglove 3D panel: set it as the display frame).

TF tree (``*`` = published here)::

    race_auv/odom --EKF--> race_auv/base_link --static--> cameras, cg_link, ...
         |
         +--*--> race_auv/odom_graph
                      +--*--> race_station/base_link
                                   +--*static--> race_station/dock_point
                                                      +--*--> race_station/apriltag36h11_<id>

Inputs
------
``odom_topic`` (``nav_msgs/Odometry``)
    The EKF: ``odom -> base_link`` pose and body-frame twist.
``dvl_topic`` (``geometry_msgs/TwistWithCovarianceStamped``; "" = off)
    DVL bottom-lock velocity. Only the stamps are used: a silence means the
    EKF dead-reckons, and the EKF snaps when the lock returns
    (``core/frontend/dvl_monitor.py``).
``cameras.<name>.detections_topic`` (``apriltag_msgs/AprilTagDetectionArray``)
    Corners in the rectified image, header stamp = image stamp. Every
    message of ``keyframe_camera`` (tags or not) is one keyframe; tags of the
    other (attached) cameras become factors on those keyframes.
``.../camera_info`` (``sensor_msgs/CameraInfo``, latched)
    The detector's rectified intrinsics (unless given in the YAML).

Only header stamps are used, never arrival times, so the latency between the
front-seat computer (EKF) and the Jetson does not matter; their clocks must
be synchronized (chrony). ``ekf_age_s`` in ``/diagnostics`` shows an offset.

Outputs
-------
``~/odometry`` (``nav_msgs/Odometry``, ``odom_graph -> base_link``), every EKF message
    The AUV pose for docking guidance: the graph's latest keyframe pose moved
    by the EKF's motion since then (``KeyframeResult.auv_pose_at``).
    Exactly the EKF until the first tag. Twist (and its covariance) is the
    EKF's, in base_link axes. Pose covariance is not estimated (all zero).
``~/dock_state`` (``race_auv_msgs/DockState``), every keyframe
    Station state (NO_STATION / CANDIDATE / TRACKING), the dock pose in
    ``odom_graph`` with the docking-pose covariance, the time a tag was last
    seen and the tag factors of this keyframe.
``~/dock_pose`` (``geometry_msgs/PoseWithCovarianceStamped``, ``odom_graph``), every keyframe while TRACKING
    The same dock pose and covariance as a standard message: the Foxglove 3D
    panel draws it with its covariance ellipsoid; the offline tools read it.
TF ``odom -> odom_graph``, every EKF message
    Consistent with ``~/odometry``: ``odom_graph -> base_link`` through TF is
    the published pose.
TF ``odom_graph -> station_base_frame``, every EKF message while TRACKING
    ``S * T_dock_base``: the station, fixed in ``odom_graph``.
TF ``station_base_frame -> dock_frame`` (static) and ``dock_frame -> <station>/apriltag<family>_<id>``
    Only with ``publish_station_frames`` (when no station
    ``robot_state_publisher`` runs, which would publish the same frames):
    dock_point from the station URDF, and each tag at the graph's estimate
    ``T_j`` (its URDF pose until it has been seen).
``/diagnostics`` (``diagnostic_msgs/DiagnosticArray``), 1 Hz
    The shared ROS diagnostics topic (hence not under the node's name).

Analysis and display outputs (``publish_debug: true``)
------------------------------------------------------
``~/auv_pose`` (``geometry_msgs/PoseStamped``, ``odom_graph``), every keyframe
    ``X_k``, the graph's AUV pose as estimated at that keyframe (the
    real-time estimate, before later smoothing). Read by the offline tools.
``~/keyframe`` (``std_msgs/String``, JSON), every keyframe
    ``{"stamp", "state", "camera", "n_tags", "n_tag_factors", "accepted":
    [tags], "rejected": [[tag, reason]], "initialized_now", "recovered",
    "init_shift_m", "dvl_sigma", "init_note", "update_ms"}``, plus
    ``"attached": [{"camera", "stamp", "anchor_k", "accepted", "rejected"}]``
    with attached cameras. Read by the offline tools.
``~/path`` and ``~/ekf_path`` (``nav_msgs/Path``, ``odom_graph``), at ``path_rate`` Hz
    The last ``path_length`` keyframes of the graph's ``X_k`` and of the
    EKF's ``E_k``, both drawn in ``odom_graph``. Since the two frames start
    out equal, ``~/ekf_path`` is where the EKF thinks the AUV went in that
    frame, and the gap between the two tracks is the EKF's drift.
"""

from __future__ import annotations

import json
import logging
import math
from collections import deque
from typing import Dict, Optional

import gtsam
import rclpy
from apriltag_msgs.msg import AprilTagDetectionArray
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from geometry_msgs.msg import (PoseStamped, PoseWithCovariance, PoseWithCovarianceStamped,
                               TwistWithCovarianceStamped)
from nav_msgs.msg import Odometry, Path
from race_auv_msgs.msg import DockState
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo
from std_msgs.msg import String
from tf2_ros import StaticTransformBroadcaster, TransformBroadcaster

from ..core.backend.estimator import DockEstimator, KeyframeResult
from ..core.common.geometry import gtsam_cov_to_ros
from ..core.common.model import TagFrame, tag_key_str
from ..core.frontend.scheduler import KeyframeScheduler
from ..core.init.state_machine import StationState
from .msgs import (pose_from_msg, pose_to_msg, sec_to_stamp, stamp_to_sec, tag_detections,
                   transform_msg)
from .params import NODE_DEFAULTS, NODE_NAME, build_setup, camera_defaults, estimator_defaults

# ---------------------------------------------------------------------- QoS
# Inputs: reliable, like their publishers (the EKF and the tag detector). The
# queue absorbs short stalls while an iSAM2 update runs.
_QOS_IN = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.VOLATILE,
                     history=HistoryPolicy.KEEP_LAST, depth=50)
# DVL: only that messages arrive matters; best effort matches any publisher.
_QOS_DVL = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.VOLATILE,
                      history=HistoryPolicy.KEEP_LAST, depth=10)
# The detector's rectified camera_info: published once, latched.
_QOS_LATCHED = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                          durability=DurabilityPolicy.TRANSIENT_LOCAL,
                          history=HistoryPolicy.KEEP_LAST, depth=1)

# A detector or EKF stream silent for longer than this is reported as stalled
# in /diagnostics (the detector runs at 5 Hz, the EKF at >= 20 Hz).
_STALL_S = 1.0

# StationState -> DockState.state
_DOCK_STATE = {
    StationState.NO_STATION: DockState.NO_STATION,
    StationState.CANDIDATE: DockState.CANDIDATE,
    StationState.TRACKING: DockState.TRACKING,
}


# ---------------------------------------------------------------------- outputs from a keyframe
def dock_pose_with_covariance(res: KeyframeResult) -> PoseWithCovariance:
    """The dock in odom_graph (``S``) with the docking-pose covariance along the odom_graph axes.

    The estimator's ``dock_cov`` is the covariance of the docking pose
    ``X_k^-1 S`` in its own tangent space (gtsam: right perturbation,
    ``[rot, trans]`` order). Holding the AUV fixed, a perturbation of the
    docking pose is the same perturbation of ``S`` in S's local frame
    (``X (X^-1 S) Exp(xi) = S Exp(xi)``), so rotating it by ``S``'s rotation
    expresses it along the odom_graph axes, in ROS ``[trans, rot]`` order.
    All zero when not computed (``covariance_mode: none``).
    """
    msg = PoseWithCovariance()
    msg.pose = pose_to_msg(res.S)
    if res.dock_cov is not None:
        msg.covariance = gtsam_cov_to_ros(res.dock_cov, res.S.rotation().matrix()).ravel().tolist()
    return msg


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


class FactorGraphNode(Node):
    """Feeds the EKF and the tag detections to the dock estimator and publishes its results."""

    def __init__(self) -> None:
        super().__init__(NODE_NAME)
        log = logging.getLogger("race_auv_factorgraph")
        log.addHandler(_RosLogHandler(self))
        log.setLevel(logging.INFO)
        log.propagate = False

        # ================================================================ parameters
        # Defaults live in ros/params.py, shared with the offline bag replay.
        # Camera parameters are declared once camera_names is known.
        for name, default in {**NODE_DEFAULTS, **estimator_defaults()}.items():
            self.declare_parameter(name, default)
        for cam in self.get_parameter("camera_names").value:
            for name, default in camera_defaults(cam).items():
                self.declare_parameter(name, default)
        g = lambda name: self.get_parameter(name).value  # noqa: E731

        self.odom_frame = g("odom_frame")
        self.odom_graph_frame = g("odom_graph_frame")
        self.base_frame = g("base_frame")
        self.station_base_frame = g("station_base_frame")
        self.dock_frame = g("dock_frame")

        # ================================================================ estimator
        self.setup = build_setup(g, self.get_logger().warn)
        # Shared with the estimator: a camera whose intrinsics come from the
        # detector's camera_info is added to this dict when that arrives.
        self.cameras = self.setup.cameras
        cfg = self.setup.cfg
        self.estimator = DockEstimator(cfg, self.cameras, self.setup.tags)
        # Holds each detector frame until the EKF covers its stamp, then
        # releases it as a keyframe (core/frontend/scheduler.py).
        self.scheduler = KeyframeScheduler(ekf_buffer_seconds=float(g("ekf_buffer_seconds")),
                                           attached_cameras=self.setup.attached_cameras)

        # ================================================================ output state
        self._last_result: Optional[KeyframeResult] = None   # newest solved keyframe
        self._last_state = self.estimator.state
        self._last_tag_stamp: Optional[float] = None   # newest keyframe with a tag factor
        self._unknown_tags: Dict[str, int] = {}        # detected IDs not on the station
        # Local clock time at which the newest EKF / detector / DVL message
        # arrived, with its header stamp. For /diagnostics only, never for
        # estimation.
        self._ekf_seen: Optional[tuple] = None
        self._frame_seen: Optional[tuple] = None
        self._dvl_seen: Optional[tuple] = None

        # ================================================================ publishers
        self.tf_pub = TransformBroadcaster(self)
        self.odom_pub = self.create_publisher(Odometry, "~/odometry", 10)
        self.dock_state_pub = self.create_publisher(DockState, "~/dock_state", 10)
        self.dock_pose_pub = self.create_publisher(PoseWithCovarianceStamped, "~/dock_pose", 10)
        self.diag_pub = self.create_publisher(DiagnosticArray, "/diagnostics", 10)

        self.debug = bool(g("publish_debug"))
        if self.debug:
            self.auv_pose_pub = self.create_publisher(PoseStamped, "~/auv_pose", 50)
            self.keyframe_pub = self.create_publisher(String, "~/keyframe", 50)
            self.path_pub = self.create_publisher(Path, "~/path", 1)
            self.ekf_path_pub = self.create_publisher(Path, "~/ekf_path", 1)
            # The last path_length keyframes (deque drops the oldest); the
            # whole path is re-published at path_rate, not per keyframe,
            # because the message grows with the track.
            n = int(g("path_length"))
            self._path_X: deque = deque(maxlen=n)
            self._path_E: deque = deque(maxlen=n)
            self._path_dirty = False
            self.create_timer(1.0 / float(g("path_rate")), self._publish_paths)

        # Station frames: only when no station robot_state_publisher runs (it
        # would publish the same names). Tag frame = dock_point's prefix + the
        # tag's URDF link name, e.g. race_station/apriltag36h11_146.
        self._tag_frames: Dict = {}
        if g("publish_station_frames"):
            prefix = self.dock_frame.rsplit("/", 1)[0] + "/" if "/" in self.dock_frame else ""
            self._tag_frames = {tag: prefix + link for tag, link in self.setup.tag_links.items()}
            self._static_pub = StaticTransformBroadcaster(self)
            self._static_pub.sendTransform(transform_msg(
                self.setup.T_dock_base.inverse(), self.station_base_frame, self.dock_frame,
                self.get_clock().now().to_msg()))

        # ================================================================ subscriptions
        self.create_subscription(Odometry, g("odom_topic"), self._odom_cb, _QOS_IN)
        if g("dvl_topic"):
            self.create_subscription(TwistWithCovarianceStamped, g("dvl_topic"), self._dvl_cb,
                                     _QOS_DVL)
            self.get_logger().info(f"DVL bottom lock from {g('dvl_topic')}")
        for name, topic in self.setup.camera_topics.items():
            self.create_subscription(AprilTagDetectionArray, topic,
                                     lambda msg, n=name: self._detections_cb(n, msg), _QOS_IN)
            self.get_logger().info(f"camera {name}: detections on {topic}")
        # Rectified intrinsics, latched by the tag detector: completes a camera
        # without YAML intrinsics, cross-checks one that has them.
        for name, topic in self.setup.camera_info_topics.items():
            self.create_subscription(CameraInfo, topic,
                                     lambda msg, n=name: self._camera_info_cb(n, msg), _QOS_LATCHED)
            if name in self.cameras:
                self.get_logger().info(f"camera {name}: intrinsics from the YAML, "
                                       f"cross-checked against {topic}")
            else:
                self.get_logger().info(f"camera {name}: waiting for intrinsics on {topic}")
        # No processing timer: keyframes run from the callbacks, the moment the
        # scheduler can release them (see _process).
        self.create_timer(1.0, self._publish_diagnostics)

        roles = ", ".join([f"{self.setup.keyframe_camera} (keyframes)"]
                          + [f"{c} (attached)" for c in self.setup.attached_cameras])
        self.get_logger().info(
            f"{NODE_NAME} ready: fixed-lag {cfg.smoother_lag:g} s, cameras {roles}, "
            f"tags {[tag_key_str(k) for k in self.setup.tags]}, odom={g('odom_topic')}, "
            f"TF {self.odom_frame} -> {self.odom_graph_frame}")

    # ==================================================================== input callbacks
    def _odom_cb(self, msg: Odometry) -> None:
        """EKF message: buffer it, run any keyframe it releases, publish the 20 Hz outputs.

        A detector frame waits until an EKF pose stamped at or after it has
        arrived (its EKF pose is interpolated at the image stamp), so a new
        EKF message is what usually releases a keyframe.
        """
        t = stamp_to_sec(msg.header.stamp)
        self._ekf_seen = (self._now(), t)
        E = pose_from_msg(msg)
        self.scheduler.add_odometry(t, E)
        self._process()
        self._publish_ekf_rate_outputs(msg, t, E)

    def _detections_cb(self, camera: str, msg: AprilTagDetectionArray) -> None:
        """Detector message: convert it to a TagFrame and queue it (it becomes a keyframe later).

        Every message is queued, also one without detections: on the keyframe
        camera it is still a keyframe (odometry only).
        """
        if msg.header.frame_id and msg.header.frame_id != self.setup.camera_frames[camera]:
            self.get_logger().warn(
                f"{camera}: detections stamped in frame '{msg.header.frame_id}', expected "
                f"'{self.setup.camera_frames[camera]}'. Check the topic mapping.",
                throttle_duration_sec=10.0)
        dets, unknown = tag_detections(camera, msg, self.setup.tags)
        for key in unknown:
            name = tag_key_str(key)
            self._unknown_tags[name] = self._unknown_tags.get(name, 0) + 1
        t = stamp_to_sec(msg.header.stamp)
        self._frame_seen = (self._now(), t)
        self.scheduler.add_frame(TagFrame(t, camera, dets))
        self._process()   # runs now if the EKF already covers t, else on a later EKF message

    def _dvl_cb(self, msg: TwistWithCovarianceStamped) -> None:
        """DVL message = bottom lock at this stamp (the velocity itself is not used)."""
        t = stamp_to_sec(msg.header.stamp)
        self._dvl_seen = (self._now(), t)
        self.estimator.dvl.add(t)

    def _camera_info_cb(self, camera: str, msg: CameraInfo) -> None:
        """Take a camera's rectified intrinsics from its detector, or check the YAML ones.

        The detector's camera_info describes its rectified image (``k`` = the
        pinhole matrix the corners follow, zero distortion). Until it arrives,
        frames wait in the scheduler (see _process).
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

    # ==================================================================== keyframes
    def _process(self) -> None:
        """Run every keyframe the scheduler releases. Never raises.

        Nothing runs before every camera has its intrinsics: the frames wait
        in the scheduler meanwhile (camera_info is latched, so this is only
        the first moment after start-up).
        """
        if self.setup.missing_intrinsics():
            return
        for kf in self.scheduler.poll():
            try:
                res = self.estimator.process_keyframe(kf.stamp, kf.E, kf.frame)
            except Exception as e:  # the node must never die on an estimator error
                self.get_logger().error(f"keyframe at {kf.stamp:.3f} failed: {e!r}")
                continue
            self._on_keyframe(res, kf.frame)
        # Attached cameras after the keyframes, so each anchor keyframe exists.
        for af in self.scheduler.poll_attached():
            try:
                self.estimator.add_attached_frame(af.stamp, af.E, af.frame)
            except Exception as e:
                self.get_logger().error(f"attached frame ({af.frame.camera}) at {af.stamp:.3f} "
                                        f"failed: {e!r}")

    def _on_keyframe(self, res: KeyframeResult, frame: Optional[TagFrame]) -> None:
        """One keyframe solved: update the output state and publish the per-keyframe topics."""
        self._last_result = res
        if res.state != self._last_state:
            self.get_logger().info(f"station state {self._last_state.value} -> {res.state.value}")
            if res.state != StationState.TRACKING:
                # Station forgotten (optimizer reset): the dock was never seen
                # by the new estimate.
                self._last_tag_stamp = None
            self._last_state = res.state

        n_tag_factors = len(res.accepted) + res.n_attached_accepted
        if n_tag_factors > 0:
            self._last_tag_stamp = res.stamp

        stamp = sec_to_stamp(res.stamp)
        self._publish_dock(res, stamp, n_tag_factors)
        if self.debug:
            self._publish_debug(res, stamp, frame, n_tag_factors)

    def _publish_dock(self, res: KeyframeResult, stamp, n_tag_factors: int) -> None:
        """``~/dock_state`` (always) and ``~/dock_pose`` (while TRACKING)."""
        msg = DockState()
        msg.header.stamp = stamp
        msg.header.frame_id = self.odom_graph_frame
        msg.state = _DOCK_STATE[res.state]
        msg.n_tag_factors = min(n_tag_factors, 255)
        if self._last_tag_stamp is not None:
            msg.last_tag_stamp = sec_to_stamp(self._last_tag_stamp)
        if res.S is not None:
            msg.dock = dock_pose_with_covariance(res)
            pose = PoseWithCovarianceStamped()
            pose.header = msg.header
            pose.pose = msg.dock
            self.dock_pose_pub.publish(pose)
        self.dock_state_pub.publish(msg)

    # ==================================================================== EKF-rate outputs
    def _publish_ekf_rate_outputs(self, ekf: Odometry, t: float, E: gtsam.Pose3) -> None:
        """``~/odometry`` and the TF frames, with the EKF message's stamp."""
        # The newest keyframe's pose moved by the EKF's motion since it,
        # unless the DVL lost bottom lock (or re-locked) in between. No
        # keyframe yet: odom_graph = odom, the EKF itself.
        kf = self._last_result
        if kf is None:
            pose = E
        else:
            pose = kf.auv_pose_at(E, ekf_trusted=self.estimator.dvl.extra_sigma(kf.stamp, t) == 0.0)

        odom = Odometry()
        odom.header.stamp = ekf.header.stamp
        odom.header.frame_id = self.odom_graph_frame
        odom.child_frame_id = self.base_frame
        odom.pose.pose = pose_to_msg(pose)
        # Body-frame velocity: the same in odom and odom_graph.
        odom.twist = ekf.twist
        self.odom_pub.publish(odom)

        # odom -> odom_graph: base_link is at E in odom and at `pose` in
        # odom_graph, so odom_graph sits at E * pose^-1 in odom. Through TF,
        # odom_graph -> base_link is then exactly the published pose.
        tfs = [transform_msg(E.compose(pose.inverse()), self.odom_frame, self.odom_graph_frame,
                       ekf.header.stamp)]
        S = kf.S if kf is not None else None      # dock_point in odom_graph, while TRACKING
        if S is not None:
            tfs.append(transform_msg(S.compose(self.setup.T_dock_base), self.odom_graph_frame,
                               self.station_base_frame, ekf.header.stamp))
            if self._tag_frames:
                est = self.estimator.tag_estimates()
                tfs += [transform_msg(est.get(tag, self.setup.tags[tag].T_dock_tag), self.dock_frame,
                                frame, ekf.header.stamp)
                        for tag, frame in self._tag_frames.items()]
        self.tf_pub.sendTransform(tfs)

    # ==================================================================== analysis / display
    def _publish_debug(self, res: KeyframeResult, stamp, frame: Optional[TagFrame],
                       n_tag_factors: int) -> None:
        """``~/auv_pose``, ``~/keyframe``, and the keyframe's points of ``~/path`` / ``~/ekf_path``."""
        def stamped(T: gtsam.Pose3) -> PoseStamped:
            m = PoseStamped()
            m.header.stamp = stamp
            m.header.frame_id = self.odom_graph_frame
            m.pose = pose_to_msg(T)
            return m

        X = stamped(res.X)
        self.auv_pose_pub.publish(X)
        self._path_X.append(X)
        # The EKF's own numbers drawn in odom_graph (see the module docstring).
        self._path_E.append(stamped(res.E))
        self._path_dirty = True

        self.keyframe_pub.publish(String(data=json.dumps({
            "stamp": res.stamp,
            "state": res.state.value,
            "camera": frame.camera if frame is not None else "",
            # Station tags in the detector message (0 = odometry-only keyframe).
            "n_tags": len(frame.detections) if frame is not None else 0,
            # Tag factors this keyframe added (also from attached cameras).
            "n_tag_factors": n_tag_factors,
            "accepted": [tag_key_str(t) for t in res.accepted],
            "rejected": [[tag_key_str(t), r] for t, r in res.rejected],
            "initialized_now": res.initialized_now,
            "recovered": res.recovered,
            "init_shift_m": round(res.init_shift, 4),
            "dvl_sigma": round(res.dvl_extra_sigma, 4),
            "init_note": res.init_note,
            "update_ms": round(res.update_time * 1000.0, 3),
            **({"attached": [{"camera": a.camera, "stamp": a.stamp, "anchor_k": a.anchor_k,
                              "accepted": [tag_key_str(t) for t in a.accepted],
                              "rejected": [[tag_key_str(t), r] for t, r in a.rejected]}
                             for a in res.attached]}
               if self.setup.attached_cameras else {}),
        })))

    def _publish_paths(self) -> None:
        """Timer (``path_rate``): re-publish both capped paths if a keyframe came in."""
        if not self._path_dirty:
            return
        self._path_dirty = False
        for pub, poses in ((self.path_pub, self._path_X), (self.ekf_path_pub, self._path_E)):
            msg = Path()
            msg.header.stamp = poses[-1].header.stamp     # newest keyframe
            msg.header.frame_id = self.odom_graph_frame
            msg.poses = list(poses)
            pub.publish(msg)

    # ==================================================================== diagnostics
    def _publish_diagnostics(self) -> None:
        """``/diagnostics`` (1 Hz): station state, inputs, timing, how far the graph corrects the EKF."""
        est, sched, res = self.estimator, self.scheduler, self._last_result
        st = DiagnosticStatus()
        st.name = NODE_NAME
        st.hardware_id = "race_auv"
        st.level = DiagnosticStatus.OK if est.state == StationState.TRACKING else DiagnosticStatus.WARN
        st.message = est.state.value
        now = self._now()
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
            # Seconds since a keyframe last added a tag factor (in stamps).
            "last_tag_age_s": ("never" if self._last_tag_stamp is None or self._ekf_seen is None
                               else f"{self._ekf_seen[1] - self._last_tag_stamp:.2f}"),
        }
        if res is not None:
            kv["last_update_ms"] = f"{res.update_time * 1000:.1f}"
            kv["last_accepted"] = ", ".join(tag_key_str(t) for t in res.accepted)
            kv["last_rejected"] = "; ".join(f"{tag_key_str(t)}: {r}" for t, r in res.rejected)
            kv["init_note"] = res.init_note
            # How far the graph has moved the AUV from where the EKF puts it.
            kv["drift_correction_m"] = f"{res.drift_distance:.3f}"
            kv["drift_correction_yaw_deg"] = f"{math.degrees(res.drift_correction.rotation().yaw()):.2f}"
        # Timing across machines. age = (local clock at arrival) - (header
        # stamp): transport + processing delay *plus* any clock offset.
        # ekf_age_s should be a few ms; a negative or large value means the
        # Pi and Jetson clocks disagree (check chrony). frame_age_s is the
        # detector latency. "*_silent_s" is how long ago the last one arrived.
        for name, seen in (("ekf", self._ekf_seen), ("frame", self._frame_seen),
                           *((("dvl", self._dvl_seen),) if est.dvl.active else ())):
            if seen is None:
                kv[f"{name}_age_s"] = kv[f"{name}_silent_s"] = "never received"
                continue
            kv[f"{name}_age_s"] = f"{seen[0] - seen[1]:.3f}"
            kv[f"{name}_silent_s"] = f"{now - seen[0]:.2f}"
        # No detector messages = no keyframes at all (the detector is the
        # keyframe clock), so flag a stall even while TRACKING.
        stalled = [n for n, seen in (("detector", self._frame_seen), ("EKF", self._ekf_seen))
                   if seen is None or now - seen[0] > _STALL_S]
        if stalled:
            st.level = DiagnosticStatus.WARN
            st.message = f"{est.state.value}; no {' / '.join(stalled)} messages for > {_STALL_S:g} s"
        if self.setup.attached_cameras and res is not None:
            kv["attached_cameras"] = ", ".join(self.setup.attached_cameras)
            kv["last_attached_accepted"] = ", ".join(
                f"{a.camera}:{tag_key_str(t)}" for a in res.attached for t in a.accepted)
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
    node = FactorGraphNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
