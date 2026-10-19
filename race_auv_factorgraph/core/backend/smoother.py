"""Fixed-lag smoother backend (§9).

The estimator always runs ``gtsam.IncrementalFixedLagSmoother``: iSAM2 plus
marginalization of every variable whose timestamp is older than
``newest timestamp - smoother_lag``. This bounds the graph, so the cost per
keyframe stays flat on a long run.

Why not plain iSAM2? Every tag factor is attached to the single station
variable ``S``. Each time iSAM2 relinearizes ``S`` it re-evaluates all of
those factors, and with a graph that grows forever this slowed to several
seconds per keyframe on a 25-minute tank run. For *post-processing* the whole
graph is re-optimized at once instead (batch Levenberg-Marquardt,
``tools/batch_lm.py``), which needs no incremental solver at all.

Two lifetimes of a variable matter here:

* ``X_k`` (AUV pose at keyframe k) is stamped with its keyframe time and is
  marginalized once it is older than the lag. This is the intended behaviour.
* ``S`` and the ``T_j`` must live forever, so the estimator re-stamps them with
  the current time on every update (:func:`keys.restamp`). They are never
  marginalized.

All exceptions from gtsam (for example ``IndeterminantLinearSystemException``,
which the Python bindings surface as ``RuntimeError``) propagate to the
caller. ``estimator.DockEstimator`` catches them and rebuilds the backend.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import gtsam
import numpy as np

from ..common.config import EstimatorConfig


class Backend:
    """Fixed-lag smoother with an update / estimate / covariance API."""

    def __init__(self, cfg: EstimatorConfig) -> None:
        self.cfg = cfg
        params = gtsam.ISAM2Params()
        # iSAM2 relinearization: a variable is relinearized when its update
        # exceeds the threshold, checked every `relinearizeSkip` updates.
        params.setRelinearizeThreshold(cfg.isam_relinearize_threshold)
        params.relinearizeSkip = int(cfg.isam_relinearize_skip)
        self._smoother = gtsam.IncrementalFixedLagSmoother(float(cfg.smoother_lag), params)
        # Cached result of the last update. Reading the estimate is not free,
        # and the estimator queries it several times per keyframe.
        self._estimate = gtsam.Values()
        # Every factor ever added, kept for batch post-processing
        # (full_graph). Holding the factor objects costs a little memory and
        # no compute: the smoother itself never sees this copy.
        self._all_factors = gtsam.NonlinearFactorGraph()

    # ------------------------------------------------------------------ update
    def update(self, graph: gtsam.NonlinearFactorGraph, values: gtsam.Values,
               timestamps: Dict[int, float]) -> None:
        """Add new factors and new variables, then re-solve incrementally.

        ``timestamps`` (key -> seconds) must hold every *new* key; the
        smoother refuses a variable without one. Existing keys may also be
        listed to re-stamp them (that is how ``S`` and the ``T_j`` are kept
        alive).
        """
        self._all_factors.push_back(graph)
        self._smoother.update(graph, values, {int(k): float(t) for k, t in timestamps.items()})
        # Extra iterations with no new factors: each is one more
        # Gauss-Newton step on the variables affected by the new factors.
        for _ in range(self.cfg.isam_extra_iterations):
            self._smoother.update(gtsam.NonlinearFactorGraph(), gtsam.Values(), {})
        self._estimate = self._smoother.calculateEstimate()

    # ------------------------------------------------------------------ queries
    @property
    def estimate(self) -> gtsam.Values:
        """Current estimate of every variable still in the smoother (recent keys only)."""
        return self._estimate

    def exists(self, key: int) -> bool:
        """True if ``key`` is a variable in the smoother (not yet marginalized)."""
        return self._estimate.exists(key)

    def pose(self, key: int) -> Optional[gtsam.Pose3]:
        return self._estimate.atPose3(key) if self._estimate.exists(key) else None

    def marginal_covariance(self, key: int) -> Optional[np.ndarray]:
        """6x6 marginal covariance of a Pose3 key (gtsam ``[rot, trans]`` order).

        Returns ``None`` if the key does not exist or the query fails. This
        is called on the output path, which must never crash.
        """
        if not self.exists(key):
            return None
        try:
            return np.asarray(self._smoother.marginalCovariance(key))
        except RuntimeError:
            return None

    def full_graph(self) -> gtsam.NonlinearFactorGraph:
        """Every factor added to this backend since it was created.

        Used for batch post-processing (``tools/batch_lm.py``): the same
        factors the real-time run accepted, re-optimized all at once. The
        smoother itself has replaced old factors with marginals internally,
        so this is a separate copy taken as factors are added.
        """
        return self._all_factors

    def joint_marginal_covariance(self, keys: List[int]) -> Optional[np.ndarray]:
        """Joint covariance of several Pose3 keys, ``6n x 6n`` (key order as given).

        The smoother does not expose a joint marginal in the Python
        bindings, so this relinearizes its current factor graph at the
        current estimate and runs ``gtsam.Marginals`` on it. The graph
        includes the marginal factors that summarize the variables already
        marginalized out, so the result stays correct. The cost grows with
        the graph size (bounded by the lag); callers rate-limit it.
        """
        if not all(self.exists(k) for k in keys):
            return None
        try:
            marginals = gtsam.Marginals(self._smoother.getFactors(), self._estimate)
            kv = gtsam.KeyVector()
            for k in keys:
                kv.append(k)
            return np.asarray(marginals.jointMarginalCovariance(kv).fullMatrix())
        except RuntimeError:
            return None
