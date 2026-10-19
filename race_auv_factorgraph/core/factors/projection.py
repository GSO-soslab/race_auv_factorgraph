"""Tag projection factor: AUV pose, station and tag pose vs corner pixels (§5).

================== ============== ===================================================
Factor             Keys           Residual
================== ============== ===================================================
tag projection     X_k, S, T_j    pi(K_C, (X_k T_bc)^-1 S T_j c_i) - uv_i, i = 1..4 (8 rows)
================== ============== ===================================================

The ``CustomFactor`` computes analytic Jacobians by chaining the Jacobians
that gtsam's ``Pose3`` methods return. ``test/test_jacobians.py`` checks them
against finite differences.

Jacobian conventions
--------------------
gtsam differentiates with respect to a *right* perturbation of each pose,
``X -> X * Exp(xi)`` with ``xi = [rx, ry, rz, tx, ty, tz]``. So a Jacobian
with respect to a Pose3 key has 6 columns in that order. Methods such as
``Pose3.compose(other, H1, H2)`` fill pre-allocated, Fortran-ordered numpy
arrays with their Jacobians (:func:`~..common.geometry.jac_buffer`).
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import gtsam
import numpy as np

from ..common.geometry import jac_buffer as _F
from ..common.model import CameraModel, TagModel
from .noise import huber


def project_tag(X: gtsam.Pose3, S: gtsam.Pose3, T: gtsam.Pose3,
                cam: CameraModel, tag: TagModel) -> Tuple[np.ndarray, np.ndarray]:
    """Predict the 4 corner pixels of ``tag`` seen from AUV pose ``X``.

    The chain is the measurement model of the projection factor:

    1. camera pose in odom:     ``P = X * T_bc``
    2. tag pose in odom:        ``W = S * T_j``
    3. corner i in odom:        ``q_i = W * c_i``
    4. corner i in camera:      ``p_i = P^-1 * q_i``
    5. pinhole projection:      ``u = fx * x/z + cx``,  ``v = fy * y/z + cy``

    Returns ``(uv, depth)``: ``(4, 2)`` predicted pixels and ``(4,)`` corner
    depths ``z`` in the camera frame. Depths <= 0 mean the corner is behind
    the camera, and those pixels are meaningless. Used by the gates and by
    evaluation; the factor below repeats the same chain with Jacobians.
    """
    P = X.compose(cam.T_bc)
    W = S.compose(T)
    uv = np.zeros((4, 2))
    depth = np.zeros(4)
    for i, c in enumerate(tag.corners):
        p = P.transformTo(W.transformFrom(c))
        depth[i] = p[2]
        z = p[2] if abs(p[2]) > 1e-9 else 1e-9
        uv[i] = (cam.fx * p[0] / z + cam.cx, cam.fy * p[1] / z + cam.cy)
    return uv, depth


def reprojection_rms(X: gtsam.Pose3, S: gtsam.Pose3, T: gtsam.Pose3,
                     cam: CameraModel, tag: TagModel, corners: np.ndarray) -> float:
    """RMS pixel distance between observed and predicted corners.

    Returns ``inf`` if any corner is predicted behind the camera.
    """
    uv, depth = project_tag(X, S, T, cam, tag)
    if np.any(depth <= 0.0):
        return float("inf")
    return float(np.sqrt(np.mean(np.sum((uv - corners) ** 2, axis=1))))


class TagProjectionFactor:
    """Builds the 8-row tag projection ``CustomFactor`` on ``(X_k, S, T_j)``.

    One factor per tag observation groups the four corners (§5 notes). It
    keeps them rigid, and it means 4x fewer Python callbacks than one factor
    per corner, which matters because every linearization calls back into
    Python.

    The residual is ``predicted - observed`` in pixels, stacked as
    ``[u0, v0, u1, v1, u2, v2, u3, v3]``.
    """

    def __init__(self, x_key: int, s_key: int, t_key: int,
                 cam: CameraModel, tag: TagModel, corners: np.ndarray,
                 sigma_px: float, huber_k: float, min_depth: float) -> None:
        self.keys = [x_key, s_key, t_key]
        self.cam = cam
        self.tag = tag
        self.observed = np.asarray(corners, dtype=float).reshape(4, 2)
        self.sigma_px = float(sigma_px)
        self.min_depth = float(min_depth)
        noise = huber(gtsam.noiseModel.Isotropic.Sigma(8, sigma_px), huber_k)
        # The CustomFactor keeps a reference to the bound method, so the
        # camera, tag and corner data travel with the factor.
        self.factor = gtsam.CustomFactor(noise, self.keys, self._error)

    def _error(self, this: gtsam.CustomFactor, values: gtsam.Values,
               H: Optional[List[np.ndarray]]) -> np.ndarray:
        X = values.atPose3(self.keys[0])
        S = values.atPose3(self.keys[1])
        T = values.atPose3(self.keys[2])
        cam = self.cam
        want_H = H is not None

        # 1. Camera pose in odom, P = X * T_bc. T_bc is a constant, so only
        #    dP/dX is needed (the second Jacobian buffer is discarded).
        dP_dX = _F(6, 6)
        P = X.compose(cam.T_bc, dP_dX, _F(6, 6))

        # 2. Tag pose in odom, W = S * T_j, with dW/dS and dW/dT.
        dW_dS, dW_dT = _F(6, 6), _F(6, 6)
        W = S.compose(T, dW_dS, dW_dT)

        err = np.zeros(8)
        J_X = np.zeros((8, 6))
        J_S = np.zeros((8, 6))
        J_T = np.zeros((8, 6))
        for i, c in enumerate(self.tag.corners):
            # 3. Corner in odom, q = W * c   (dq/dW: 3x6; c is constant)
            dq_dW = _F(3, 6)
            q = W.transformFrom(c, dq_dW, _F(3, 3))
            # 4. Corner in the camera frame, p = P^-1 * q   (dp/dP: 3x6, dp/dq: 3x3)
            dp_dP, dp_dq = _F(3, 6), _F(3, 3)
            p = P.transformTo(q, dp_dP, dp_dq)

            # 5. Pinhole projection. If the optimizer pushes a corner behind
            #    (or almost onto) the camera plane mid-iteration, the depth is
            #    clamped. The residual stays finite, and the clamped depth has
            #    zero derivative, so the step is driven by x and y only.
            x, y, z = p
            clamped = z < self.min_depth
            if clamped:
                z = self.min_depth
            u = cam.fx * x / z + cam.cx
            v = cam.fy * y / z + cam.cy
            err[2 * i] = u - self.observed[i, 0]
            err[2 * i + 1] = v - self.observed[i, 1]

            if want_H:
                # d(u, v)/d(x, y, z) for the pinhole model.
                dz = 0.0 if clamped else 1.0
                d_uv_dp = np.array([
                    [cam.fx / z, 0.0, -dz * cam.fx * x / (z * z)],
                    [0.0, cam.fy / z, -dz * cam.fy * y / (z * z)],
                ])
                # Chain rule through steps 1-4:
                #   d uv / dX = d uv/dp * dp/dP * dP/dX
                #   d uv / dS = d uv/dp * dp/dq * dq/dW * dW/dS   (same for T)
                d_uv_dW = d_uv_dp @ dp_dq @ dq_dW
                J_X[2 * i:2 * i + 2] = d_uv_dp @ dp_dP @ dP_dX
                J_S[2 * i:2 * i + 2] = d_uv_dW @ dW_dS
                J_T[2 * i:2 * i + 2] = d_uv_dW @ dW_dT

        if want_H:
            H[0] = J_X
            H[1] = J_S
            H[2] = J_T
        return err

    def rms_px(self, values: gtsam.Values) -> float:
        """RMS corner error (px) at ``values``, without the robust kernel."""
        e = self._error(self.factor, values, None)
        return float(np.sqrt(np.mean(e.reshape(4, 2) ** 2) * 2.0))
