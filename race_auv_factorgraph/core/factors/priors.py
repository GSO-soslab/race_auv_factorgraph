"""Pose priors: the EKF anchor, the tag URDF prior and recovery priors (§5, §9).

================== ============== ===================================================
Factor             Keys           Residual
================== ============== ===================================================
EKF anchor prior   X_0            X_0 (-) E_0                  (see estimator.add_anchor)
tag URDF prior     T_j            T_j (-) T_j^URDF
================== ============== ===================================================
"""

from __future__ import annotations

import math

import gtsam
import numpy as np

from ..common.config import EstimatorConfig
from ..common.model import TagModel
from .noise import huber, pose_sigmas


def pose_prior(key: int, value: gtsam.Pose3, sigma_rot: float, sigma_trans: float
               ) -> gtsam.PriorFactorPose3:
    """Gaussian prior ``key (-) value`` with diagonal rotation/translation sigmas."""
    return gtsam.PriorFactorPose3(key, value, pose_sigmas(sigma_rot, sigma_trans))


def pose_prior_cov(key: int, value: gtsam.Pose3, cov: np.ndarray,
                   huber_k: float = 0.0) -> gtsam.PriorFactorPose3:
    """Prior with a full 6x6 covariance (gtsam ``[rot, trans]`` order), Huber if ``huber_k > 0``.

    Used where a covariance is known instead of fixed sigmas: the smoother's
    marginals when the backend is rebuilt after an optimizer failure (§9), and
    the priors of the initial-value solve (``DockEstimator._init_from_tags``).
    """
    cov = 0.5 * (cov + cov.T) + 1e-12 * np.eye(6)  # symmetric, strictly PD
    return gtsam.PriorFactorPose3(key, value,
                                  huber(gtsam.noiseModel.Gaussian.Covariance(cov), huber_k))


def tag_mount_prior(key: int, tag: TagModel, cfg: EstimatorConfig) -> gtsam.PriorFactorPose3:
    """The tag's mounting prior: ``T_j`` at its URDF pose (``sigma_mount_rot_deg``, ``sigma_mount_trans``)."""
    return pose_prior(key, tag.T_dock_tag, math.radians(cfg.sigma_mount_rot_deg),
                      cfg.sigma_mount_trans)
