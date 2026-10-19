"""Noise-model helpers shared by the factor builders (§5)."""

from __future__ import annotations

import gtsam
import numpy as np


def huber(base: gtsam.noiseModel.Base, k: float) -> gtsam.noiseModel.Base:
    """Wrap ``base`` in a Huber robust kernel with threshold ``k`` (in sigmas).

    Below ``k`` the factor behaves like a normal Gaussian (quadratic cost).
    Above it the cost grows only linearly, so a single bad measurement cannot
    dominate the solution. ``k <= 0`` disables the kernel.
    """
    if k <= 0.0:
        return base
    return gtsam.noiseModel.Robust.Create(gtsam.noiseModel.mEstimator.Huber.Create(k), base)


def pose_sigmas(sigma_rot: float, sigma_trans: float) -> gtsam.noiseModel.Diagonal:
    """Diagonal Pose3 noise: 3 rotation sigmas (rad), then 3 translation sigmas (m)."""
    return gtsam.noiseModel.Diagonal.Sigmas(
        np.array([sigma_rot] * 3 + [sigma_trans] * 3, dtype=float))
