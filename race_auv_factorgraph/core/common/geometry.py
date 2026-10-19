"""Small Pose3 / rotation helpers shared by every core module.

Conventions used throughout the package
----------------------------------------
* ``Pose3`` values follow the ``T_a_b`` reading: "pose of frame b expressed in
  frame a". Composition is ``T_a_c = T_a_b * T_b_c`` (gtsam ``compose``).
* The graph frame (``odom``) is z-up (REP-103), so "world up" is ``(0, 0, 1)``.
* gtsam's ``Pose3`` tangent vector is ordered ``[rx, ry, rz, tx, ty, tz]``
  (rotation first). ROS covariance matrices are ordered translation first.
  :func:`gtsam_cov_to_ros` does the reordering.
* Euler angles follow the URDF / ROS convention: fixed-axis roll-pitch-yaw,
  i.e. ``R = Rz(yaw) @ Ry(pitch) @ Rx(roll)``.

Everything here is pure numpy + gtsam (no ROS) so it can run in unit tests and
offline tools.
"""

from __future__ import annotations

from typing import Tuple

import gtsam
import numpy as np


def jac_buffer(rows: int, cols: int) -> np.ndarray:
    """Zeroed Fortran-ordered buffer for a gtsam ``OptionalJacobian`` output.

    Methods such as ``Pose3.compose(other, H1, H2)`` write their Jacobians into
    pre-allocated numpy arrays, and the binding requires column-major
    (Fortran) storage, hence ``order="F"``. A Jacobian with respect to a Pose3
    has 6 columns in tangent order ``[rx, ry, rz, tx, ty, tz]``.
    """
    return np.zeros((rows, cols), order="F")


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------
def pose_from_xyz_rpy(xyz, rpy) -> gtsam.Pose3:
    """Build a Pose3 from a translation and URDF-style roll/pitch/yaw.

    ``gtsam.Rot3.RzRyRx(r, p, y)`` returns ``Rz(y) @ Ry(p) @ Rx(r)``, which
    is exactly the URDF ``<origin rpy="r p y">`` convention.
    """
    r, p, y = (float(v) for v in rpy)
    return gtsam.Pose3(gtsam.Rot3.RzRyRx(r, p, y), gtsam.Point3(*map(float, xyz)))


def pose_from_matrix(T: np.ndarray) -> gtsam.Pose3:
    """Build a Pose3 from a 4x4 homogeneous matrix.

    The rotation block is re-orthonormalized with an SVD first. Matrices that
    come from PnP or message conversions are only orthonormal to ~1e-7, and
    gtsam asserts on badly conditioned rotations in debug builds.
    """
    T = np.asarray(T, dtype=float)
    U, _, Vt = np.linalg.svd(T[:3, :3])
    R = U @ Vt
    if np.linalg.det(R) < 0:  # reflection -> flip the last singular vector
        U[:, -1] *= -1.0
        R = U @ Vt
    return gtsam.Pose3(gtsam.Rot3(R), gtsam.Point3(*T[:3, 3]))


def pose_from_rvec_tvec(rvec, tvec) -> gtsam.Pose3:
    """Convert an OpenCV ``(rvec, tvec)`` pair into a Pose3.

    OpenCV's PnP returns the transform that maps *object* points into the
    *camera* frame, i.e. ``T_camera_object``. The axis-angle ``rvec`` is turned
    into a rotation with gtsam's exponential map (identical to
    ``cv2.Rodrigues``).
    """
    rvec = np.asarray(rvec, dtype=float).reshape(3)
    tvec = np.asarray(tvec, dtype=float).reshape(3)
    return gtsam.Pose3(gtsam.Rot3.Expmap(rvec), gtsam.Point3(*tvec))


def pose_from_quat_xyzw(position, quat_xyzw) -> gtsam.Pose3:
    """Build a Pose3 from a ROS-style position + quaternion ``(x, y, z, w)``.

    Note the argument order of ``gtsam.Rot3.Quaternion`` is ``(w, x, y, z)``.
    """
    x, y, z, w = (float(v) for v in quat_xyzw)
    return gtsam.Pose3(gtsam.Rot3.Quaternion(w, x, y, z),
                       gtsam.Point3(*map(float, position)))


def quat_xyzw(pose: gtsam.Pose3) -> Tuple[float, float, float, float]:
    """Return the rotation of ``pose`` as a ROS quaternion ``(x, y, z, w)``."""
    q = pose.rotation().toQuaternion()  # gtsam returns a Quaternion object
    return float(q.x()), float(q.y()), float(q.z()), float(q.w())


# ---------------------------------------------------------------------------
# Measures
# ---------------------------------------------------------------------------
def rotation_angle(R: gtsam.Rot3) -> float:
    """Angle (rad) of the rotation, i.e. the norm of its axis-angle vector."""
    return float(np.linalg.norm(gtsam.Rot3.Logmap(R)))


def pose_distance(a: gtsam.Pose3, b: gtsam.Pose3) -> Tuple[float, float]:
    """Return ``(translation distance [m], rotation angle [rad])`` between two poses.

    The translation distance is measured in the parent frame (so it does not
    depend on the orientation of either pose). The angle is that of the
    relative rotation ``R_a^T R_b``.
    """
    dt = float(np.linalg.norm(a.translation() - b.translation()))
    dr = rotation_angle(a.rotation().between(b.rotation()))
    return dt, dr


def interpolate(a: gtsam.Pose3, b: gtsam.Pose3, alpha: float) -> gtsam.Pose3:
    """Interpolate between two poses.

    The translation is interpolated linearly and the rotation with SLERP,
    so the result is independent of the parent frame's orientation.
    ``alpha = 0`` returns ``a`` and ``alpha = 1`` returns ``b``.
    """
    alpha = float(alpha)
    R = a.rotation().slerp(alpha, b.rotation())
    t = (1.0 - alpha) * a.translation() + alpha * b.translation()
    return gtsam.Pose3(R, gtsam.Point3(*t))


def step_towards(current: gtsam.Pose3, target: gtsam.Pose3,
                 max_trans: float, max_rot: float) -> gtsam.Pose3:
    """Move ``current`` toward ``target`` by at most ``max_trans`` m / ``max_rot`` rad.

    Used to rate-limit the published ``odom -> station`` correction (§10) so a
    controller never sees a jump. The step fraction is the largest one that
    keeps both the translation and the rotation within their limits.
    """
    dt, dr = pose_distance(current, target)
    frac = 1.0
    if dt > 1e-12 and max_trans >= 0.0:
        frac = min(frac, max_trans / dt)
    if dr > 1e-12 and max_rot >= 0.0:
        frac = min(frac, max_rot / dr)
    if frac >= 1.0:
        return target
    return interpolate(current, target, frac)


# ---------------------------------------------------------------------------
# Covariance
# ---------------------------------------------------------------------------
def gtsam_cov_to_ros(cov6: np.ndarray, R_parent_child: np.ndarray) -> np.ndarray:
    """Convert a gtsam Pose3 marginal covariance to ROS ``PoseWithCovariance`` layout.

    gtsam reports the covariance of a pose in the pose's own *local* tangent
    space, ordered ``[rot, trans]``: the uncertainty is a small perturbation
    ``xi`` applied on the right, ``T * Exp(xi)``. ROS expects the covariance
    expressed along the axes of the *parent* (header) frame and ordered
    ``[trans, rot]``.

    To first order, a local translation perturbation ``dt`` shows up in the
    parent frame as ``R dt``, and a local rotation perturbation ``dth`` as
    ``R dth``. Rotating both 3x3 blocks by ``R`` and swapping the block order
    gives the ROS matrix. (The small coupling term between rotation
    uncertainty and the lever arm is neglected. It is negligible for the
    short lever arms here.)
    """
    R = np.asarray(R_parent_child, dtype=float)
    J = np.zeros((6, 6))
    J[:3, :3] = R
    J[3:, 3:] = R
    cov_parent = J @ np.asarray(cov6, dtype=float) @ J.T  # still [rot, trans]
    out = np.zeros((6, 6))
    out[:3, :3] = cov_parent[3:, 3:]  # trans-trans
    out[3:, 3:] = cov_parent[:3, :3]  # rot-rot
    out[:3, 3:] = cov_parent[3:, :3]  # trans-rot
    out[3:, :3] = cov_parent[:3, 3:]  # rot-trans
    return out
