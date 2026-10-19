"""Synthetic docking approach: true trajectory, drifting EKF, noisy tag corners.

Used by ``test/test_scenarios.py`` and runnable as a demo::

    python3 -m race_auv_factorgraph.tools.synthetic

The world reproduces the real geometry without needing any other package:

* cameras: the AUV URDF extrinsics of ``cam_front`` / ``cam_down`` (optical
  frames) and the simulated rectified intrinsics (82 deg HFOV, 1600x1200);
* tags: the race_station URDF poses of the five tag36h11 tags relative to
  ``dock_point``, with black-square sizes 0.12 m / 0.04 m.

Ground truth: the station sits at ``S_true`` in the true odom frame, and the
AUV flies a gently weaving approach toward the front tags. The EKF is
simulated as ``E(t) = Drift(t) * X_true(t)``, where ``Drift`` is a growing
x/y translation plus a yaw rotation about world z. Roll, pitch and z are
unaffected, as they are in reality (gravity and pressure do not drift).
Caveat: the yaw part rotates about the odom *origin*, so it also creates a
position error proportional to the distance from the origin. That is harsher
than a real DVL/IMU EKF, whose position error grows with the distance
travelled. Keep ``drift_yaw`` small when the AUV is far from the origin.

The quantity to compare against truth is the docking pose ``X_k^-1 * S``
(dock_point in base_link). It is drift-free, unlike ``S`` itself.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import gtsam
import numpy as np

from ..core.backend.estimator import DockEstimator, KeyframeResult
from ..core.factors.projection import project_tag
from ..core.common.geometry import pose_from_xyz_rpy
from ..core.common.model import CameraModel, Detection, TagFrame, TagKey, TagModel
from ..core.frontend.scheduler import KeyframeScheduler


# ===========================================================================
# Geometry copied from the URDFs (see core/common/urdf.py for the real loader)
# ===========================================================================
def default_cameras() -> Dict[str, CameraModel]:
    """``cam_front`` and ``cam_down`` with the AUV URDF extrinsics.

    In the URDF the cameras hang off ``nose_tip_link``, which sits at
    x = 0.7 m in ``base_link``. The composed ``base_link -> camera`` poses are
    used here directly. Intrinsics: Stonefish 82 deg HFOV at 1600x1200, i.e.
    fx = fy = 800 / tan(41 deg) = 920.29, centred.
    """
    f = 800.0 / math.tan(math.radians(41.0))
    return {
        "cam_front": CameraModel("cam_front", f, f, 800.0, 600.0, 1600, 1200,
                                 pose_from_xyz_rpy([0.696, 0.0, 0.035], [-1.571, 0.0, -1.571])),
        "cam_down": CameraModel("cam_down", f, f, 800.0, 600.0, 1600, 1200,
                                pose_from_xyz_rpy([0.395, 0.0, -0.128], [0.0, 3.1415, 1.571])),
    }


def default_tags() -> Dict[TagKey, TagModel]:
    """The five tag36h11 tags of the race_station URDF (poses in ``dock_point``)."""
    h = math.pi / 2
    spec = {
        146: ([0.430, 0.0, 0.405], [-h, 0.0, -h], 0.12),
        176: ([0.0, 0.25, -0.012], [math.pi, 0.0, -h], 0.12),
        185: ([0.0, -0.25, -0.012], [math.pi, 0.0, -h], 0.12),
        541: ([0.470, 0.0, 0.160], [-h, 0.0, -h], 0.04),
        558: ([0.470, 0.0, 0.095], [-h, 0.0, -h], 0.04),
    }
    return {("36h11", i): TagModel(("36h11", i), size, pose_from_xyz_rpy(xyz, rpy))
            for i, (xyz, rpy, size) in spec.items()}


def default_T_dock_base() -> gtsam.Pose3:
    """Station base_link in dock_point: inverse of the URDF dock_point_joint."""
    return pose_from_xyz_rpy([-0.410, 0.325, -0.04], [3.1415926, 0.0, 0.0]).inverse()


# ===========================================================================
# Scenario
# ===========================================================================
@dataclass
class ScenarioConfig:
    duration: float = 25.0
    odom_rate: float = 20.0
    # Detector rate = keyframe rate (one keyframe per detector message,
    # core/frontend/scheduler.py). 5 Hz is what the vehicle's Jetson runs.
    camera_rate: float = 5.0
    # Station dock_point in the true odom frame (upright, yawed).
    station_xyz: Tuple[float, float, float] = (4.0, 1.0, -3.0)
    station_yaw: float = 0.3
    # Approach in the dock_point frame: from start_x to end_x along +x,
    # weaving laterally (amplitude, period) for viewpoint diversity.
    start_x: float = -7.0
    end_x: float = -1.2
    height: float = 0.33            # base_link height above dock_point
    weave_amp: float = 0.4
    weave_period: float = 8.0
    # EKF drift rates (true odom -> EKF odom).
    drift_xy: Tuple[float, float] = (0.01, -0.006)   # m/s
    drift_yaw: float = math.radians(0.2)             # rad/s
    # Detection model.
    pixel_noise: float = 0.5
    max_range: float = 9.0
    min_area: float = 300.0
    border: float = 12.0
    min_view_angle_deg: float = 15.0
    # Real mounting error of each tag vs. its URDF pose (applied to the truth).
    mount_error_sigma: Tuple[float, float] = (0.0, 0.0)   # (rad, m)
    # Detections are suppressed in these [t0, t1) windows (tags lost). The
    # detector still publishes, with no tags, so keyframes continue.
    blackouts: List[Tuple[float, float]] = field(default_factory=list)
    # DVL bottom-lock dropouts [t0, t1): no DVL message, and the EKF dead-
    # reckons with a velocity error (dvl_dropout_vel, m/s in odom x, y), so
    # its position drifts by vel * (t - t0); at t1 the DVL returns and the
    # EKF snaps back to its pre-dropout error, like robot_localization does
    # (core/frontend/dvl_monitor.py). DVL messages otherwise at dvl_rate.
    dvl_dropouts: List[Tuple[float, float]] = field(default_factory=list)
    dvl_dropout_vel: Tuple[float, float] = (0.08, -0.05)
    dvl_rate: float = 5.0
    # Only these tags are ever detected (None = all).
    visible_tags: Optional[List[int]] = None
    cameras: Tuple[str, ...] = ("cam_front",)   # one camera: the estimator supports one
    seed: int = 0


@dataclass
class Scenario:
    cfg: ScenarioConfig
    cameras: Dict[str, CameraModel]
    tags: Dict[TagKey, TagModel]
    T_dock_base: gtsam.Pose3
    S_true: gtsam.Pose3
    odometry: List[Tuple[float, gtsam.Pose3]]           # (stamp, E)
    truth: Dict[float, gtsam.Pose3]                      # stamp -> X_true
    frames: List[TagFrame]
    dvl: List[float] = field(default_factory=list)       # DVL bottom-lock message stamps

    def true_pose(self, t: float) -> gtsam.Pose3:
        return _true_pose(self.cfg, self.S_true, t)

    def true_dock_in_base(self, t: float) -> gtsam.Pose3:
        return self.true_pose(t).between(self.S_true)


def _true_pose(c: ScenarioConfig, S_true: gtsam.Pose3, t: float) -> gtsam.Pose3:
    """True AUV base_link pose in odom at time t (expressed via the dock frame)."""
    a = min(max(t / c.duration, 0.0), 1.0)
    x = c.start_x + (c.end_x - c.start_x) * a
    y = c.weave_amp * math.sin(2 * math.pi * t / c.weave_period)
    # Heading along +x of dock_point, with the yaw following the weave a little.
    yaw = 0.15 * math.cos(2 * math.pi * t / c.weave_period)
    roll = math.radians(2.0) * math.sin(0.7 * t)
    pitch = math.radians(1.5) * math.sin(0.5 * t)
    return S_true.compose(pose_from_xyz_rpy([x, y, c.height], [roll, pitch, yaw]))


def _drift(c: ScenarioConfig, t: float) -> gtsam.Pose3:
    return gtsam.Pose3(gtsam.Rot3.Rz(c.drift_yaw * t),
                       gtsam.Point3(c.drift_xy[0] * t, c.drift_xy[1] * t, 0.0))


def view_angle_to_plane(T_camera_tag: gtsam.Pose3) -> float:
    """Angle (rad) between the camera ray to the tag centre and the tag plane.

    90 deg: the camera looks straight at the tag; 0 deg: edge-on. Real
    detectors stop finding tags below roughly 15 deg, which the synthetic
    detection model reproduces with ``min_view_angle_deg``.
    """
    n = T_camera_tag.rotation().matrix()[:, 2]
    t = T_camera_tag.translation()
    r = t / max(float(np.linalg.norm(t)), 1e-9)
    return float(math.asin(min(1.0, abs(float(n @ r)))))


def make_scenario(c: ScenarioConfig = ScenarioConfig()) -> Scenario:
    rng = np.random.default_rng(c.seed)
    cams = default_cameras()
    tags = default_tags()
    S_true = pose_from_xyz_rpy(c.station_xyz, [0.0, 0.0, c.station_yaw])

    # True tag mountings may differ from the URDF.
    T_true = {k: t.T_dock_tag.retract(np.concatenate([
        rng.normal(0, c.mount_error_sigma[0], 3), rng.normal(0, c.mount_error_sigma[1], 3)]))
        for k, t in tags.items()}

    odom, truth = [], {}
    for i in range(int(c.duration * c.odom_rate) + 1):
        t = i / c.odom_rate
        X = _true_pose(c, S_true, t)
        truth[t] = X
        E = _drift(c, t).compose(X)
        for d0, d1 in c.dvl_dropouts:   # dead-reckoning drift, gone at re-lock (the snap)
            if d0 <= t < d1:
                vx, vy = c.dvl_dropout_vel
                E = gtsam.Pose3(gtsam.Rot3(), gtsam.Point3(vx * (t - d0), vy * (t - d0), 0.0)).compose(E)
        odom.append((t, E))
    dvl = [t for t in np.arange(0.0, c.duration, 1.0 / c.dvl_rate)
           if not any(d0 <= t < d1 for d0, d1 in c.dvl_dropouts)]
    dvl = sorted(dvl + [d1 for _, d1 in c.dvl_dropouts if d1 < c.duration])   # first message back

    # One detector message per camera tick, like apriltag_detector_node: with
    # the visible tags, or empty (blackout / nothing in view).
    frames: List[TagFrame] = []
    for ci, cam_name in enumerate(c.cameras):
        cam = cams[cam_name]
        offset = 0.31 * (ci + 1) / c.camera_rate   # not aligned with the EKF ticks
        t = offset
        while t < c.duration:
            dets = []
            if not any(b0 <= t < b1 for b0, b1 in c.blackouts):
                X = _true_pose(c, S_true, t)
                for key, tag in tags.items():
                    if c.visible_tags is not None and key[1] not in c.visible_tags:
                        continue
                    uv, depth = project_tag(X, S_true, T_true[key], cam, tag)
                    if np.any(depth < 0.1) or np.mean(depth) > c.max_range:
                        continue
                    if (uv[:, 0].min() < c.border or uv[:, 1].min() < c.border
                            or uv[:, 0].max() > cam.width - 1 - c.border
                            or uv[:, 1].max() > cam.height - 1 - c.border):
                        continue
                    T_ct = X.compose(cam.T_bc).between(S_true.compose(T_true[key]))
                    if view_angle_to_plane(T_ct) < math.radians(c.min_view_angle_deg):
                        continue
                    noisy = uv + rng.normal(0, c.pixel_noise, uv.shape)
                    d = Detection(cam_name, key, noisy)
                    if d.area() < c.min_area:
                        continue
                    dets.append(d)
            frames.append(TagFrame(t, cam_name, dets))
            t += 1.0 / c.camera_rate
    frames.sort(key=lambda f: f.stamp)
    return Scenario(c, cams, tags, default_T_dock_base(), S_true, odom, truth, frames,
                    [float(t) for t in dvl])


# ===========================================================================
# Runner (same event flow as the ROS node: odometry + frames -> scheduler -> estimator)
# ===========================================================================
def run(scn: Scenario, est: DockEstimator) -> List[KeyframeResult]:
    """Feed the scenario through a :class:`KeyframeScheduler` into ``est``.

    Frames are delivered 0.1 s after their stamp, emulating detector latency;
    the scheduler releases each one once the EKF covers its stamp.
    """
    sched = KeyframeScheduler()
    latency = 0.1
    events = [(t, 0, E) for t, E in scn.odometry] + \
             [(f.stamp + latency, 1, f) for f in scn.frames] + \
             [(t + 0.01, 2, t) for t in scn.dvl]          # DVL: stamps only, 10 ms transport
    events.sort(key=lambda e: (e[0], e[1]))
    results = []
    for _, kind, payload in events:
        if kind == 0:
            sched.add_odometry(_, payload)
        elif kind == 1:
            sched.add_frame(payload)
        else:
            est.dvl.add(payload)
        for kf in sched.poll():
            results.append(est.process_keyframe(kf.stamp, kf.E, kf.frame))
    return results


def dock_error(scn: Scenario, r: KeyframeResult) -> Tuple[float, float]:
    """(translation [m], rotation [rad]) error of the docking pose at a keyframe."""
    from ..core.common.geometry import pose_distance
    return pose_distance(r.dock_in_base, scn.true_dock_in_base(r.stamp))


def main() -> None:  # pragma: no cover - demo
    from ..core.common.config import EstimatorConfig
    scn = make_scenario(ScenarioConfig())
    est = DockEstimator(EstimatorConfig(), scn.cameras, scn.tags, scn.T_dock_base)
    results = run(scn, est)
    first = next((r for r in results if r.initialized_now), None)
    n_tag = sum(1 for f in scn.frames if f.detections)
    print(f"{len(scn.frames)} detector frames ({n_tag} with tags), {len(results)} keyframes")
    if first is None:
        print("station never initialized; last note:", results[-1].init_note)
        return
    print(f"initialized at t = {first.stamp:.2f} s")
    for r in results[::max(1, len(results) // 15)]:
        if r.S is not None:
            dt, dr = dock_error(scn, r)
            print(f"t={r.stamp:6.2f}  {r.state.value:9s}  dock error {dt * 100:5.1f} cm "
                  f"{math.degrees(dr):4.1f} deg  update {r.update_time * 1000:5.1f} ms")


if __name__ == "__main__":  # pragma: no cover
    main()
