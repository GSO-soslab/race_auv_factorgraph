"""Unit tests for the small building blocks: geometry, URDF, pose buffer, scheduler, PnP."""


import gtsam
import numpy as np

from race_auv_factorgraph.core.common.config import config_from_dict
from race_auv_factorgraph.core.factors.projection import project_tag
from race_auv_factorgraph.core.common.geometry import (gtsam_cov_to_ros, pose_distance,
                                                pose_from_xyz_rpy, quat_xyzw)
from race_auv_factorgraph.core.common.model import Detection, TagFrame, normalize_family
from race_auv_factorgraph.core.frontend.pnp import ippe_tag
from race_auv_factorgraph.core.frontend.pose_buffer import PoseBuffer
from race_auv_factorgraph.core.frontend.scheduler import KeyframeScheduler
from race_auv_factorgraph.core.common.urdf import Urdf
from race_auv_factorgraph.tools.synthetic import default_cameras, default_tags

STATION_URDF = """<?xml version="1.0"?>
<robot name="race_station">
  <link name="base_link"/>
  <link name="dock_point"/>
  <joint name="dock_point_joint" type="fixed">
    <origin xyz="-0.410 0.325 -0.04" rpy="3.1415926 0.0 0.0"/>
    <parent link="base_link"/><child link="dock_point"/>
  </joint>
  <link name="apriltag36h11_146"/>
  <joint name="t146" type="fixed">
    <origin xyz="0.430 0 0.405" rpy="-1.5707963 0 -1.5707963"/>
    <parent link="dock_point"/><child link="apriltag36h11_146"/>
  </joint>
  <link name="apriltag25h9_02"/>
  <joint name="t2" type="fixed">
    <origin xyz="-0.81 0 -0.21" rpy="-1.5707963 0 -1.5707963"/>
    <parent link="dock_point"/><child link="apriltag25h9_02"/>
  </joint>
</robot>"""


# ----------------------------------------------------------------------- geometry
def test_quaternion_roundtrip():
    p = pose_from_xyz_rpy([1, 2, 3], [0.1, -0.2, 0.3])
    x, y, z, w = quat_xyzw(p)
    q = gtsam.Rot3.Quaternion(w, x, y, z)
    assert p.rotation().equals(q, 1e-9)


def test_cov_to_ros_reorders_blocks():
    cov = np.diag([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])  # [rot, trans]
    ros = gtsam_cov_to_ros(cov, np.eye(3))
    np.testing.assert_allclose(np.diag(ros), [4, 5, 6, 1, 2, 3])


def test_config_from_dict_casts_and_ignores_unknown():
    cfg = config_from_dict({"sigma_px": 2, "smoother_lag": 12, "backend": "isam2", "bogus": 1})
    assert cfg.sigma_px == 2.0 and isinstance(cfg.sigma_px, float)
    assert cfg.smoother_lag == 12.0      # cast to the field's type
    assert not hasattr(cfg, "backend")   # removed key (old YAMLs) is ignored


# ----------------------------------------------------------------------- URDF
def test_urdf_tag_poses_and_inverse_chain():
    urdf = Urdf(STATION_URDF)
    tags = urdf.tag_poses("dock_point")
    assert set(tags) == {("36h11", 146), ("25h9", 2)}
    np.testing.assert_allclose(tags[("36h11", 146)].translation(), [0.43, 0, 0.405], atol=1e-9)
    # base_link expressed in dock_point is the inverse of the joint.
    T_db = urdf.transform("dock_point", "base_link")
    joint = pose_from_xyz_rpy([-0.410, 0.325, -0.04], [3.1415926, 0, 0])
    assert T_db.equals(joint.inverse(), 1e-9)


def test_normalize_family():
    assert normalize_family("tag36h11") == "36h11"
    assert normalize_family("36h11") == "36h11"


# ----------------------------------------------------------------------- pose buffer
def test_pose_buffer_interpolates_and_refuses_extrapolation():
    buf = PoseBuffer(max_age=10.0)
    buf.add(0.0, pose_from_xyz_rpy([0, 0, 0], [0, 0, 0]))
    buf.add(1.0, pose_from_xyz_rpy([1, 0, 0], [0, 0, 0.2]))
    mid = buf.interpolate(0.5)
    np.testing.assert_allclose(mid.translation(), [0.5, 0, 0], atol=1e-12)
    assert abs(mid.rotation().yaw() - 0.1) < 1e-9
    assert buf.interpolate(1.01) is None
    assert buf.interpolate(-0.01) is None


# ----------------------------------------------------------------------- scheduler
def _frame(t, tags=True, cam="cam_front"):
    dets = [Detection(cam, ("36h11", 146), np.zeros((4, 2)))] if tags else []
    return TagFrame(t, cam, dets)


def test_scheduler_one_keyframe_per_detector_message():
    """Every detector message becomes a keyframe at its stamp, empty ones too."""
    s = KeyframeScheduler()
    for i in range(11):                       # EKF at 20 Hz, 0 .. 0.5 s
        s.add_odometry(i * 0.05, gtsam.Pose3(gtsam.Rot3(), gtsam.Point3(i * 0.05, 0, 0)))
    for t, tags in [(0.1, True), (0.3, False), (0.5, True)]:   # 5 Hz detector
        assert s.add_frame(_frame(t, tags))
    kfs = s.poll()
    assert [k.stamp for k in kfs] == [0.1, 0.3, 0.5]
    assert [bool(k.frame.detections) for k in kfs] == [True, False, True]
    # E is interpolated at the image stamp, not taken from the nearest EKF message.
    assert abs(kfs[1].E.translation()[0] - 0.3) < 1e-9


def test_scheduler_waits_for_ekf_and_drops_repeated_stamps():
    """A frame waits until an EKF pose at/after its stamp arrives (late EKF over the network)."""
    s = KeyframeScheduler()
    s.add_odometry(0.0, gtsam.Pose3())
    assert s.add_frame(_frame(0.12))
    assert not s.add_frame(_frame(0.12))       # re-published image -> dropped
    assert not s.add_frame(_frame(0.05))       # out of order -> dropped
    assert s.poll() == [] and s.n_pending == 1  # EKF has not reached 0.12 yet
    s.add_odometry(0.1, gtsam.Pose3())
    assert s.poll() == []                      # still not covered
    s.add_odometry(0.15, gtsam.Pose3())
    kfs = s.poll()
    assert len(kfs) == 1 and kfs[0].stamp == 0.12 and s.n_pending == 0
    assert s.n_dropped == 2


def test_scheduler_drops_oldest_when_ekf_stalls():
    s = KeyframeScheduler(max_pending=3)
    s.add_odometry(0.0, gtsam.Pose3())
    for i in range(1, 6):
        s.add_frame(_frame(i * 0.2))
    assert s.n_pending == 3 and s.n_dropped == 2
    s.add_odometry(2.0, gtsam.Pose3())
    assert [round(k.stamp, 3) for k in s.poll()] == [0.6, 0.8, 1.0]   # the newest three


# ----------------------------------------------------------------------- PnP
def _observe(X, S, cam, tag, noise=0.0, seed=0):
    uv, _ = project_tag(X, S, tag.T_dock_tag, cam, tag)
    return uv + np.random.default_rng(seed).normal(0, noise, uv.shape)


def test_ippe_recovers_true_pose():
    cams, tags = default_cameras(), default_tags()
    cam, tag = cams["cam_front"], tags[("36h11", 146)]
    S = pose_from_xyz_rpy([0, 0, 0], [0, 0, 0])
    X = pose_from_xyz_rpy([-2.0, 0.2, 0.3], [0, 0, 0.1])
    sols = ippe_tag(Detection("cam_front", tag.key, _observe(X, S, cam, tag)), cam, tag)
    T_true = X.compose(cam.T_bc).between(S.compose(tag.T_dock_tag))
    assert min(pose_distance(T, T_true)[0] for T, _ in sols) < 1e-6


# The per-image checks (border, minimum side, PnP fit) moved to the tag
# detector: race_auv_camera_tag_factorgraph/test/test_tags.py.


def _getter(**over):
    """Parameter accessor with the node defaults (plus per-camera ones), like getter_from_yaml."""
    from race_auv_factorgraph.ros.params import NODE_DEFAULTS, camera_defaults, estimator_defaults
    d = {**NODE_DEFAULTS, **estimator_defaults()}
    for name in over.get("camera_names", NODE_DEFAULTS["camera_names"]):
        d.update(camera_defaults(name))
    d.update(over)
    return lambda k: d[k]


def test_setup_camera_roles():
    """keyframe_camera drives keyframes; every other enabled camera is attached; disabled is unused."""
    import pytest
    from race_auv_factorgraph.ros.params import build_setup
    pytest.importorskip("ament_index_python")
    two = dict(camera_names=["cam_front", "cam_down"], keyframe_camera="cam_front")
    s = build_setup(_getter(**two), lambda m: None)
    assert s.keyframe_camera == "cam_front" and s.attached_cameras == ["cam_down"]
    assert set(s.camera_topics) == {"cam_front", "cam_down"}
    # cam_down switched off: not subscribed, not attached (today's single-camera setup).
    off = build_setup(_getter(**two, **{"cameras.cam_down.enabled": False}), lambda m: None)
    assert off.attached_cameras == [] and set(off.camera_topics) == {"cam_front"}
    # Default keyframe camera: the first enabled one.
    first = build_setup(_getter(camera_names=["cam_front", "cam_down"]), lambda m: None)
    assert first.keyframe_camera == "cam_front"


def test_setup_refuses_bad_camera_config():
    import pytest
    from race_auv_factorgraph.ros.params import build_setup
    with pytest.raises(ValueError, match="not an enabled camera"):
        build_setup(_getter(camera_names=["cam_front", "cam_down"], keyframe_camera="cam_down",
                            **{"cameras.cam_down.enabled": False}), lambda m: None)
    with pytest.raises(ValueError, match="no enabled camera"):
        build_setup(_getter(**{"cameras.cam_front.enabled": False}), lambda m: None)


# ----------------------------------------------------------------------- attached cameras
def test_scheduler_attached_frames_never_become_keyframes():
    """An attached camera's frames with tags wait only for the EKF; empty ones are discarded."""
    s = KeyframeScheduler(attached_cameras=["cam_down"])
    s.add_odometry(0.0, gtsam.Pose3())
    assert s.add_frame(_frame(0.10))                         # keyframe camera
    assert s.add_frame(_frame(0.13, cam="cam_down"))          # attached, with a tag
    assert not s.add_frame(_frame(0.18, cam="cam_down", tags=False))   # empty: not queued
    assert not s.add_frame(_frame(0.13, cam="cam_down"))      # re-published image: dropped
    assert s.add_frame(_frame(0.12))   # attached stamps do not affect the keyframe order
    assert s.poll() == [] and s.poll_attached() == []         # EKF not there yet
    s.add_odometry(0.2, gtsam.Pose3(gtsam.Rot3(), gtsam.Point3(0.2, 0, 0)))
    assert [k.stamp for k in s.poll()] == [0.10, 0.12]
    att = s.poll_attached()
    assert [(a.stamp, a.frame.camera) for a in att] == [(0.13, "cam_down")]
    assert abs(att[0].E.translation()[0] - 0.13) < 1e-9       # EKF at the attached stamp
    assert s.n_dropped == 1 and s.n_pending == 0


def test_attached_extrinsic_equals_camera_at_image_time():
    """Projection through X_j with T_bc' = dE T_bc equals projection from X_j dE with T_bc."""
    import dataclasses
    cams, tags = default_cameras(), default_tags()
    cam, tag = cams["cam_down"], tags[("36h11", 176)]
    S = pose_from_xyz_rpy([0, 0, 0], [0, 0, 0])
    X_j = pose_from_xyz_rpy([-0.4, 0.05, 0.6], [0.02, -0.01, 0.05])
    dE = pose_from_xyz_rpy([0.03, -0.01, 0.004], [0.002, 0.001, 0.01])   # 0.2 s of motion
    cam_at_anchor = dataclasses.replace(cam, T_bc=dE.compose(cam.T_bc))
    uv_attached, _ = project_tag(X_j, S, tag.T_dock_tag, cam_at_anchor, tag)
    uv_direct, _ = project_tag(X_j.compose(dE), S, tag.T_dock_tag, cam, tag)
    assert np.abs(uv_attached - uv_direct).max() < 1e-9
    assert cam.T_bc.equals(default_cameras()["cam_down"].T_bc, 1e-12)   # original untouched


def test_attached_frame_rejected_before_tracking():
    from race_auv_factorgraph.core.backend.estimator import DockEstimator
    from race_auv_factorgraph.core.common.config import EstimatorConfig
    est = DockEstimator(EstimatorConfig(), default_cameras(), default_tags())
    est.process_keyframe(0.0, gtsam.Pose3())
    est.add_attached_frame(0.05, gtsam.Pose3(), _frame(0.05, cam="cam_down"))
    est.add_attached_frame(0.25, gtsam.Pose3(), _frame(0.25, cam="cam_down"))
    res = est.process_keyframe(0.2, gtsam.Pose3())
    # Anchored in the next keyframe's update; the later frame waits for the one after.
    assert [a.stamp for a in res.attached] == [0.05] and res.n_attached_accepted == 0
    assert "before TRACKING" in res.attached[0].rejected[0][1]
    assert [a.stamp for a in est.process_keyframe(0.4, gtsam.Pose3()).attached] == [0.25]


# ----------------------------------------------------------------------- initial value of X_k
def test_drift_covariance_accumulates_odometry_steps():
    """The prior of _init_from_tags: step variances add up while no tag is seen, and
    moving forward turns yaw uncertainty into lateral uncertainty (the adjoint)."""
    from race_auv_factorgraph.core.backend.estimator import DockEstimator
    from race_auv_factorgraph.core.common.config import EstimatorConfig
    from race_auv_factorgraph.core.factors.odometry import odometry_step_sigmas
    cfg = EstimatorConfig()
    # Standing still: P is the sum of the 5 steps' variances.
    est = DockEstimator(cfg, default_cameras(), default_tags())
    for i in range(6):
        est.process_keyframe(0.2 * i, gtsam.Pose3())
    sig = odometry_step_sigmas(gtsam.Pose3(), gtsam.Pose3(), 0.2, cfg)
    np.testing.assert_allclose(est._drift_cov, 5 * np.diag(sig ** 2), rtol=1e-9)
    # Driving 0.5 m per step along body x: the yaw (pitch) noise of each step
    # swings the position sideways (up) over the distance still to go, 2.0, 1.5,
    # 1.0, 0.5 and 0 m; along the track (x) only the steps' own noise adds up.
    est = DockEstimator(cfg, default_cameras(), default_tags())
    for i in range(6):
        est.process_keyframe(0.2 * i, gtsam.Pose3(gtsam.Rot3(), gtsam.Point3(0.5 * i, 0, 0)))
    P = est._drift_cov
    sig = odometry_step_sigmas(gtsam.Pose3(), gtsam.Pose3(gtsam.Rot3(), gtsam.Point3(0.5, 0, 0)),
                               0.2, cfg)
    lever = sig[2] ** 2 * (2.0 ** 2 + 1.5 ** 2 + 1.0 ** 2 + 0.5 ** 2)
    np.testing.assert_allclose(P[3, 3], 5 * sig[3] ** 2, rtol=1e-9)
    np.testing.assert_allclose([P[4, 4], P[5, 5]], 5 * sig[3] ** 2 + lever, rtol=1e-6)


# ----------------------------------------------------------------------- node outputs
def _node_module():
    """ros/node.py, or skip when ROS (rclpy, race_auv_msgs) is not available."""
    import pytest
    pytest.importorskip("rclpy")
    pytest.importorskip("race_auv_msgs")
    from race_auv_factorgraph.ros import node
    return node


def test_auv_pose_at_is_keyframe_pose_moved_by_ekf_motion():
    """pose(t) = X_k (E_k^-1 E(t)); X_k itself when the EKF's motion is not trusted (DVL)."""
    from race_auv_factorgraph.core.backend.estimator import KeyframeResult
    from race_auv_factorgraph.core.init.state_machine import StationState
    X = pose_from_xyz_rpy([-1.5, 0.3, 0.4], [0.01, 0.02, 0.2])        # graph frame
    E = pose_from_xyz_rpy([-1.2, 0.1, 0.38], [0.012, 0.018, 0.15])    # drifted EKF, odom
    kf = KeyframeResult(k=7, stamp=1.0, state=StationState.TRACKING, X=X, E=E)
    step = pose_from_xyz_rpy([0.03, 0.0, 0.0], [0.0, 0.0, 0.01])      # AUV motion after k

    # At the keyframe stamp: the graph's pose.
    assert kf.auv_pose_at(E).equals(X, 1e-9)
    # Later: the same motion, from the graph's pose; i.e. the correction X E^-1 applied to E(t).
    pose = kf.auv_pose_at(E.compose(step))
    assert pose.equals(X.compose(step), 1e-9)
    assert pose.equals(kf.drift_correction.compose(E.compose(step)), 1e-9)
    # EKF snap (DVL re-lock, 1.65 m): its motion is not trusted, the pose stays X_k.
    snap = pose_from_xyz_rpy([1.65, -0.3, 0.0], [0.0, 0.0, 0.02])
    assert kf.auv_pose_at(snap.compose(E), ekf_trusted=False).equals(X, 1e-12)


def test_dock_pose_covariance_along_odom_graph_axes():
    """The docking-pose covariance (local, gtsam [rot, trans]) rotated by S, in ROS [trans, rot]."""
    node = _node_module()
    from race_auv_factorgraph.core.backend.estimator import KeyframeResult
    from race_auv_factorgraph.core.init.state_machine import StationState
    S = pose_from_xyz_rpy([2.0, -1.0, 0.5], [0.0, 0.0, np.pi / 2])    # dock yawed 90 deg
    res = KeyframeResult(k=3, stamp=1.0, state=StationState.TRACKING, X=gtsam.Pose3(),
                         E=gtsam.Pose3(), S=S, dock_cov=np.diag([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))
    msg = node.dock_pose_with_covariance(res)
    assert (msg.pose.position.x, msg.pose.position.y, msg.pose.position.z) == (2.0, -1.0, 0.5)
    cov = np.array(msg.covariance).reshape(6, 6)
    # Dock-frame x (variance 4) lies along odom_graph y after the 90 deg yaw, and vice versa.
    np.testing.assert_allclose(np.diag(cov), [5.0, 4.0, 6.0, 2.0, 1.0, 3.0], atol=1e-9)


def test_params_file_sections(tmp_path):
    """The node's section under any ROS 2 name form, '/**' first, and the former dock_fuser name."""
    from race_auv_factorgraph.ros.params import flatten_params_yaml
    new = tmp_path / "new.yaml"
    new.write_text("/**:\n  ros__parameters: {tag_ids: [1], odom_topic: shared}\n"
                   "/**/race_auv_factorgraph:\n  ros__parameters: {odom_topic: mine, "
                   "estimator: {sigma_px: 2.0}}\n"
                   "/**/other_node:\n  ros__parameters: {odom_topic: theirs}\n")
    flat = flatten_params_yaml(str(new))
    assert flat == {"tag_ids": [1], "odom_topic": "mine", "estimator.sigma_px": 2.0}
    old = tmp_path / "old.yaml"
    old.write_text("dock_fuser:\n  ros__parameters: {odom_topic: legacy}\n")
    assert flatten_params_yaml(str(old)) == {"odom_topic": "legacy"}


# ----------------------------------------------------------------------- DVL monitor
def test_dvl_monitor_drift_and_snap():
    """No extra sigma with bottom lock; drift sigma while silent; silence-long sigma on re-lock."""
    from race_auv_factorgraph.core.common.config import EstimatorConfig
    from race_auv_factorgraph.core.frontend.dvl_monitor import DvlMonitor
    cfg = EstimatorConfig()                    # timeout 0.5 s, 0.1 m/s, margin 0.3 s
    m = DvlMonitor(cfg)
    assert m.extra_sigma(0.0, 0.2) == 0.0      # inactive before any message
    for t in np.arange(0.0, 10.01, 0.2):       # lock until 10 s
        m.add(float(t))
    for t in np.arange(16.0, 20.01, 0.2):      # silent 10 -> 16 s, then lock again
        m.add(float(t))
    assert m.extra_sigma(5.0, 5.2) == 0.0                              # bottom lock
    assert abs(m.extra_sigma(12.0, 12.2) - 0.1 * 0.2) < 1e-9          # blind: drift
    assert abs(m.extra_sigma(10.4, 10.6) - 0.1 * 0.1) < 1e-9          # blind from 10.5 s
    assert abs(m.extra_sigma(15.9, 16.1) - (0.1 * 0.1 + 0.1 * 6.0)) < 1e-9  # re-lock at 16 s: snap
    assert abs(m.extra_sigma(16.1, 16.3) - 0.1 * 6.0) < 1e-9          # still inside the margin
    assert m.extra_sigma(16.4, 16.6) == 0.0                           # snap over


def test_dvl_monitor_ongoing_silence():
    from race_auv_factorgraph.core.common.config import EstimatorConfig
    from race_auv_factorgraph.core.frontend.dvl_monitor import DvlMonitor
    m = DvlMonitor(EstimatorConfig())
    m.add(0.0)
    assert m.extra_sigma(0.2, 0.4) == 0.0                              # within the timeout
    assert abs(m.extra_sigma(2.0, 2.2) - 0.1 * 0.2) < 1e-9            # no message since 0 s
