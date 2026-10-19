"""Unit tests for the small building blocks: geometry, URDF, pose buffer, scheduler, PnP."""


import gtsam
import numpy as np

from race_auv_factorgraph.core.common.config import config_from_dict
from race_auv_factorgraph.core.factors.projection import project_tag
from race_auv_factorgraph.core.common.geometry import (gtsam_cov_to_ros, pose_distance,
                                                pose_from_xyz_rpy, quat_xyzw, step_towards)
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


def test_step_towards_limits():
    a = gtsam.Pose3()
    b = pose_from_xyz_rpy([1.0, 0, 0], [0, 0, 0.5])
    c = step_towards(a, b, max_trans=0.1, max_rot=1.0)
    assert abs(pose_distance(a, c)[0] - 0.1) < 1e-9
    assert step_towards(a, b, 10.0, 10.0).equals(b, 1e-12)


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


# ----------------------------------------------------------------------- PnP + gates
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


def test_detection_min_side():
    """Shortest side of the corner quad (sets the seed's PnP range tolerance)."""
    d = Detection("cam_front", ("36h11", 146), np.array([[0, 0], [40, 0], [40, 30], [0, 30.]]))
    assert abs(d.min_side_px() - 30.0) < 1e-12


def test_setup_refuses_more_than_one_camera():
    """Keyframes follow one detector's messages; two cameras must fail loudly at startup."""
    import pytest
    from race_auv_factorgraph.ros.params import build_setup
    names = {"camera_names": ["cam_front", "cam_down"]}
    with pytest.raises(ValueError, match="exactly one camera"):
        build_setup(lambda k: names[k], lambda m: None)


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
