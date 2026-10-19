"""Pose priors: the EKF anchor, the tag URDF prior and recovery priors (§5, §9).

================== ============== ===================================================
Factor             Keys           Residual
================== ============== ===================================================
EKF anchor prior   X_0            X_0 (-) E_0                  (see estimator.add_anchor)
tag URDF prior     T_j            T_j (-) T_j^URDF
================== ============== ===================================================
"""

from __future__ import annotations

import gtsam
import numpy as np

from .noise import pose_sigmas


def pose_prior(key: int, value: gtsam.Pose3, sigma_rot: float, sigma_trans: float
               ) -> gtsam.PriorFactorPose3:
    """Gaussian prior ``key (-) value`` with diagonal rotation/translation sigmas."""
    return gtsam.PriorFactorPose3(key, value, pose_sigmas(sigma_rot, sigma_trans))


def pose_prior_cov(key: int, value: gtsam.Pose3, cov: np.ndarray) -> gtsam.PriorFactorPose3:
    """Gaussian prior with a full 6x6 covariance (gtsam ``[rot, trans]`` order).

    Used when rebuilding the backend after an optimizer failure (§9): the
    marginal covariances of the last good estimate become the new priors.
    """
    cov = 0.5 * (cov + cov.T) + 1e-12 * np.eye(6)  # symmetric, strictly PD
    return gtsam.PriorFactorPose3(key, value, gtsam.noiseModel.Gaussian.Covariance(cov))
