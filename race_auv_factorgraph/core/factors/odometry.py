"""Relative-motion odometry factor and its random-walk noise (§5).

The EKF's x, y and yaw drift, so they enter the graph only as *relative*
motion between consecutive keyframes:

    measured:   Z = E_{k-1}^-1 * E_k        (EKF motion, in the X_{k-1} body frame)
    residual:   (X_{k-1}^-1 * X_k) (-) Z    (gtsam BetweenFactorPose3)

Noise model: random walk. For each axis

    sigma^2 = q_t * dt + q_d * dd

where ``dt`` is the time between the keyframes and ``dd`` the distance the EKF
says the AUV moved. Variances add over consecutive intervals, so the total
uncertainty between two times does not depend on how many keyframes lie
between them. The keyframe rate then changes how often the graph is updated,
not how much it trusts the odometry.

A Huber kernel lets tag observations win a disagreement with the odometry.
While the DVL has no bottom lock (and on the EKF's snap when it returns) the
caller adds extra translation sigma (``core/frontend/dvl_monitor.py``).
"""

from __future__ import annotations

import math

import gtsam
import numpy as np

from ..common.config import EstimatorConfig
from .noise import huber


def odometry_sigmas(dt: float, dd: float, cfg: EstimatorConfig) -> np.ndarray:
    """Per-axis sigmas ``[rx, ry, rz, tx, ty, tz]`` for an interval of ``dt`` s and ``dd`` m."""
    dt = max(0.0, float(dt))
    dd = max(0.0, float(dd))
    s_rot = max(math.sqrt(cfg.odom_q_rot_t * dt + cfg.odom_q_rot_d * dd), cfg.odom_min_sigma_rot)
    s_tr = max(math.sqrt(cfg.odom_q_trans_t * dt + cfg.odom_q_trans_d * dd),
               cfg.odom_min_sigma_trans)
    return np.array([s_rot] * 3 + [s_tr] * 3)


def odometry_factor(key_prev: int, key_cur: int,
                    E_prev: gtsam.Pose3, E_cur: gtsam.Pose3,
                    dt: float, cfg: EstimatorConfig,
                    extra_sigma_trans: float = 0.0) -> gtsam.BetweenFactorPose3:
    """Between factor carrying the EKF motion from ``E_prev`` to ``E_cur``.

    ``extra_sigma_trans`` (m) is added in quadrature to the translation
    sigma: the DVL monitor's distrust of this step (bottom lock lost, or the
    EKF snap when it returns; see ``core/frontend/dvl_monitor.py``). The
    step is then barely trusted, and tag observations (when there are any)
    decide where the robot is. Rotation keeps its normal sigma: the EKF's
    dead-reckoning error and its snap are position-only (yaw comes from the
    IMU).
    """
    Z = E_prev.between(E_cur)  # E_prev^-1 * E_cur
    dd = float(np.linalg.norm(Z.translation()))
    sig = odometry_sigmas(dt, dd, cfg)
    if extra_sigma_trans > 0.0:
        sig[3:] = np.sqrt(sig[3:] ** 2 + extra_sigma_trans ** 2)
    noise = gtsam.noiseModel.Diagonal.Sigmas(sig)
    return gtsam.BetweenFactorPose3(key_prev, key_cur, Z, huber(noise, cfg.odom_huber_k))
