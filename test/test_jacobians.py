"""Finite-difference Jacobian check for every CustomFactor (§5 notes, §12 step 1).

For each factor, the analytic Jacobians returned through ``CustomFactor`` are
compared against central differences of the residual. The perturbation is
applied the same way gtsam does: on the right of each pose,
``X -> X * Exp(eps * e_j)`` (``Pose3.retract``).
"""

import gtsam
import numpy as np
import pytest


from race_auv_factorgraph.core.factors.projection import TagProjectionFactor
from race_auv_factorgraph.core.common.geometry import pose_from_xyz_rpy
from race_auv_factorgraph.core.common.model import CameraModel, TagModel


def numerical_jacobians(factor: gtsam.CustomFactor, values: gtsam.Values,
                        eps: float = 1e-6):
    """Central-difference Jacobian of ``unwhitenedError`` w.r.t. each Pose3 key."""
    keys = list(factor.keys())
    out = []
    for key in keys:
        base = values.atPose3(key)
        J = np.zeros((factor.dim(), 6))
        for j in range(6):
            d = np.zeros(6)
            d[j] = eps
            vp, vm = gtsam.Values(values), gtsam.Values(values)
            vp.update(key, base.retract(d))
            vm.update(key, base.retract(-d))
            J[:, j] = (factor.unwhitenedError(vp) - factor.unwhitenedError(vm)) / (2 * eps)
        out.append(J)
    return out


def analytic_jacobians(factor: gtsam.CustomFactor, values: gtsam.Values):
    """Jacobians as gtsam sees them, i.e. from the linearized factor.

    ``linearize`` returns a whitened JacobianFactor ``A = W * H``. For the
    isotropic models used here, ``W = 1/sigma * I``, so the raw Jacobian is
    recovered by multiplying by sigma. With a robust kernel, linearize also
    reweights, so the tests disable Huber.
    """
    lin = factor.linearize(values)
    A, _ = lin.jacobian()
    sigma = factor.noiseModel().sigmas()[0]
    A = A * sigma
    return [A[:, 6 * i:6 * (i + 1)] for i in range(len(factor.keys()))]


@pytest.fixture
def front_camera():
    # Same extrinsic as the AUV URDF cam_front: optical frame looking along +x body.
    return CameraModel("cam_front", 920.0, 921.0, 800.0, 600.0, 1600, 1200,
                       pose_from_xyz_rpy([0.696, 0.0, 0.035], [-1.571, 0.0, -1.571]))


@pytest.mark.parametrize("seed", range(5))
def test_tag_projection_jacobians(front_camera, seed):
    rng = np.random.default_rng(seed)
    tag = TagModel(("36h11", 146), 0.12,
                   pose_from_xyz_rpy([0.43, 0.0, 0.405], [-1.5708, 0.0, -1.5708]))
    # AUV 3 m in front of the station facing it (+x), with random perturbations.
    X = pose_from_xyz_rpy([-3.0, 0.1, -1.0] + rng.normal(0, 0.2, 3),
                          rng.normal(0, 0.05, 3))
    # Station dock_point rpy (pi, 0, 0) as in the station URDF.
    S = pose_from_xyz_rpy([0.0, 0.0, -1.0] + rng.normal(0, 0.1, 3),
                          [np.pi, 0.0, 0.0] + rng.normal(0, 0.05, 3))
    T = tag.T_dock_tag.retract(rng.normal(0, 0.01, 6))

    values = gtsam.Values()
    values.insert(1, X)
    values.insert(2, S)
    values.insert(3, T)
    corners = rng.normal(0, 3, (4, 2)) + np.array([[800, 600]] * 4)
    f = TagProjectionFactor(1, 2, 3, front_camera, tag, corners,
                            sigma_px=1.0, huber_k=0.0, min_depth=0.05).factor

    for Ja, Jn in zip(analytic_jacobians(f, values), numerical_jacobians(f, values)):
        np.testing.assert_allclose(Ja, Jn, rtol=1e-4, atol=1e-3)


def test_tag_projection_residual_zero_at_truth(front_camera):
    """Corners generated from the model give a zero residual."""
    from race_auv_factorgraph.core.factors.projection import project_tag

    tag = TagModel(("36h11", 146), 0.12,
                   pose_from_xyz_rpy([0.43, 0.0, 0.405], [-1.5708, 0.0, -1.5708]))
    X = pose_from_xyz_rpy([-3.0, 0.0, -1.0], [0.0, 0.0, 0.0])
    S = pose_from_xyz_rpy([0.0, 0.0, -1.0], [np.pi, 0.0, 0.0])
    uv, depth = project_tag(X, S, tag.T_dock_tag, front_camera, tag)
    assert np.all(depth > 0)
    values = gtsam.Values()
    values.insert(1, X)
    values.insert(2, S)
    values.insert(3, tag.T_dock_tag)
    f = TagProjectionFactor(1, 2, 3, front_camera, tag, uv, 1.0, 0.0, 0.05).factor
    np.testing.assert_allclose(f.unwhitenedError(values), 0.0, atol=1e-9)
