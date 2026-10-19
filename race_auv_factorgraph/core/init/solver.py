"""Station initialization from one or more tags over several frames (§7).

Projection factors are non-convex and diverge from a poor initial guess, so
``S``, the ``T_j`` and every tag factor enter the graph only after a reliable
first estimate of the station. That estimate must be obtainable from a
**single tag** (usually the only one visible on approach), which is
ambiguous in any one frame. So several frames are combined, linked by the
odometry chain:

1. **Per-frame hypotheses** (§7.1, :func:`frame_hypotheses`). PnP gives
   ``T_C_dock`` (two solutions for a single tag). Each becomes a station pose
   in odom, ``S_h = X_hat_k * T_bc * T_C_dock``. Then prune:

   * *ratio*: if one solution reprojects much better, keep only it.
     Otherwise both stay; voting and the batch solve resolve the flip.

2. **Multi-frame voting** (§7.2, :func:`vote`). The station is static and the
   short-term odometry is accurate, so the correct hypothesis stays put in
   odom from frame to frame while the flipped one moves with the viewpoint.
   The pose supported by the most frames wins, provided the supporting views
   are diverse enough and no competitor comes close.

3. **Batch solve** (§7.3, :func:`batch_solve`). A small graph over the
   window (buffered ``X_k`` + odometry + ``S`` + ``T_j`` +
   every buffered projection factor) is optimized from the winner, and from
   the competitor if there is one. Accept if the winner explains all corners
   well and clearly beats the competitor.

Why the drift before the first sighting does not matter (§7.0): ``S_h`` is
computed from the EKF-chained ``X_hat_k``, so the station is placed where it
is *in the EKF's current, drifted odom*, which is the frame everything else
uses. The docking quantity ``X_k^-1 * S`` contains no odometry at all.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import gtsam
import numpy as np

from ..common import keys as K
from ..common.config import EstimatorConfig
from ..common.geometry import pose_distance
from ..common.model import CameraModel, Detection, TagKey, TagModel
from ..factors.odometry import odometry_factor
from ..factors.priors import pose_prior
from ..factors.projection import TagProjectionFactor
from ..frontend.pnp import station_solutions


# ===========================================================================
# Data
# ===========================================================================
@dataclass
class Hypothesis:
    """One candidate station pose from one frame.

    Attributes
    ----------
    S
        Candidate ``dock_point`` pose in odom.
    pnp_rms
        RMS reprojection error (px) of the PnP solution it came from.
    cam_pos
        Camera position in odom at that frame. Used to measure viewpoint
        diversity between frames.
    """

    S: gtsam.Pose3
    pnp_rms: float
    cam_pos: np.ndarray

    @property
    def range(self) -> float:
        """Distance from the camera to the station origin (m)."""
        return float(np.linalg.norm(self.S.translation() - self.cam_pos))


@dataclass
class BufferedFrame:
    """A tag frame waiting for initialization (§7: "buffered, not dropped").

    ``k`` is the frame's keyframe index; its ``X_k`` already exists in the
    backend. ``E`` is the EKF pose at the frame (for the batch-solve
    odometry factors). ``detections`` already passed
    gates 1 and 2.
    """

    k: int
    stamp: float
    E: gtsam.Pose3
    camera: str
    detections: List[Detection]
    hypotheses: List[Hypothesis] = field(default_factory=list)


@dataclass
class VoteResult:
    winner: Optional[Hypothesis] = None
    support: List[int] = field(default_factory=list)    # indices of supporting frames
    competitor: Optional[Hypothesis] = None
    competitor_votes: int = 0
    ready: bool = False                                  # all §7.2 conditions met
    reason: str = ""


@dataclass
class BatchResult:
    ok: bool
    S: Optional[gtsam.Pose3] = None
    T: Dict[TagKey, gtsam.Pose3] = field(default_factory=dict)
    error: float = float("inf")       # total (robust) graph error after LM
    max_rms_px: float = float("inf")  # worst per-detection RMS corner error


# ===========================================================================
# §7.1 Per-frame hypotheses
# ===========================================================================
def frame_hypotheses(X_hat: gtsam.Pose3, cam: CameraModel, dets: List[Detection],
                     tags: Dict[TagKey, TagModel], cfg: EstimatorConfig
                     ) -> Tuple[List[Hypothesis], str]:
    """Station hypotheses from one frame, after the ratio test (§7.1).

    Returns ``(hypotheses, note)``; ``note`` explains an empty result.
    """
    sols = station_solutions(dets, cam, tags)
    if not sols:
        return [], "PnP failed"

    # Ratio test. With two IPPE solutions, a clearly better one means the
    # view is unambiguous (e.g. strong perspective), so drop the other.
    if len(sols) == 2 and sols[0][1] < cfg.init_ambiguity_ratio * sols[1][1]:
        sols = sols[:1]

    P = X_hat.compose(cam.T_bc)               # camera pose in odom
    # S = X_hat * T_bc * T_C_dock for each surviving solution
    return [Hypothesis(S=P.compose(T_cd), pnp_rms=err, cam_pos=P.translation())
            for T_cd, err in sols], ""


# ===========================================================================
# §7.2 Multi-frame voting
# ===========================================================================
def _close(a: Hypothesis, b: Hypothesis, cfg: EstimatorConfig) -> bool:
    """Do two hypotheses describe the same station pose (within tolerance)?

    Both tolerances grow with range, like the PnP error.
    """
    dt, dr = pose_distance(a.S, b.S)
    rng = max(a.range, b.range)
    trans_tol = max(cfg.init_vote_trans_tol, cfg.init_vote_trans_tol_rel * rng)
    rot_tol = math.radians(max(cfg.init_vote_rot_tol_deg, cfg.init_vote_rot_tol_deg_per_m * rng))
    return dt <= trans_tol and dr <= rot_tol


def _support(c: Hypothesis, frames: List[BufferedFrame], cfg: EstimatorConfig) -> List[int]:
    """Indices of the frames that have *some* hypothesis close to ``c``.

    A frame votes at most once per candidate, even if both of its
    hypotheses are close.
    """
    return [i for i, f in enumerate(frames) if any(_close(c, h, cfg) for h in f.hypotheses)]


def viewpoint_diversity(cam_positions: List[np.ndarray], station_pos: np.ndarray
                        ) -> Tuple[float, float]:
    """``(max viewing-direction change [rad], max lateral baseline [m])``.

    Viewing direction: unit vector from the station to the camera. Lateral
    baseline: camera displacement perpendicular to the mean viewing
    direction. Moving straight toward the tag adds neither, and such frames
    only average noise; they cannot separate the flip (§7.2).
    """
    if len(cam_positions) < 2:
        return 0.0, 0.0
    C = np.asarray(cam_positions)
    D = C - station_pos
    D = D / np.maximum(np.linalg.norm(D, axis=1, keepdims=True), 1e-9)
    m = D.mean(axis=0)
    m = m / max(np.linalg.norm(m), 1e-9)
    max_angle, max_lat = 0.0, 0.0
    for i in range(len(C)):
        for j in range(i + 1, len(C)):
            max_angle = max(max_angle, math.acos(float(np.clip(D[i] @ D[j], -1.0, 1.0))))
            delta = C[i] - C[j]
            lateral = delta - (delta @ m) * m
            max_lat = max(max_lat, float(np.linalg.norm(lateral)))
    return max_angle, max_lat


def vote(frames: List[BufferedFrame], cfg: EstimatorConfig) -> VoteResult:
    """Find the station pose supported by the most frames (§7.2).

    Every hypothesis of every frame is a candidate. Its support is the set of
    frames with a hypothesis close to it. The candidate with the largest
    support wins (ties go to the lower PnP error). The best candidate *not*
    close to the winner is the competitor (typically the flipped solution).

    ``ready`` is set when the winner has at least ``init_min_votes``
    supporting frames, those frames show viewpoint diversity, and the
    competitor has at most ``init_max_competitor_ratio`` of the winner's
    votes.
    """
    res = VoteResult()
    candidates = [h for f in frames for h in f.hypotheses]
    if not candidates:
        res.reason = "no hypotheses"
        return res

    scored = [(len(_support(c, frames, cfg)), -c.pnp_rms, idx) for idx, c in enumerate(candidates)]
    scored.sort(reverse=True)
    res.winner = candidates[scored[0][2]]
    res.support = _support(res.winner, frames, cfg)

    for votes, _, idx in scored[1:]:
        c = candidates[idx]
        if not _close(c, res.winner, cfg):
            res.competitor, res.competitor_votes = c, votes
            break

    n = len(res.support)
    if n < cfg.init_min_votes:
        res.reason = f"{n}/{cfg.init_min_votes} votes"
        return res
    if res.competitor_votes > cfg.init_max_competitor_ratio * n:
        res.reason = f"competitor too strong ({res.competitor_votes} vs {n} votes)"
        return res

    # Viewpoint diversity, measured on one hypothesis per supporting frame.
    cams = []
    for i in res.support:
        best = min((h for h in frames[i].hypotheses if _close(res.winner, h, cfg)),
                   key=lambda h: pose_distance(h.S, res.winner.S)[0])
        cams.append(best.cam_pos)
    angle, lateral = viewpoint_diversity(cams, res.winner.S.translation())
    if (angle < math.radians(cfg.init_min_view_change_deg)
            and lateral < cfg.init_min_lateral_baseline):
        res.reason = (f"not enough viewpoint change ({math.degrees(angle):.1f} deg, "
                      f"{lateral * 100:.1f} cm lateral)")
        return res

    res.ready = True
    return res


# ===========================================================================
# §7.3 Batch solve
# ===========================================================================
def batch_solve(frames: List[BufferedFrame], S0: gtsam.Pose3,
                X_est: Dict[int, gtsam.Pose3],
                cams: Dict[str, CameraModel], tags: Dict[TagKey, TagModel],
                cfg: EstimatorConfig) -> BatchResult:
    """Optimize the window's small graph from station guess ``S0`` (§7.3).

    Graph contents:

    * one ``X_k`` per buffered frame, initialized at the backend's current
      estimate ``X_est[k]`` and held there by a loose prior;
    * odometry between consecutive buffered frames (same builder as the main
      graph);
    * ``S`` initialized at ``S0``;
    * one ``T_j`` per tag in the window, at its URDF pose with the mounting prior;
    * a projection factor for every buffered detection.

    Levenberg-Marquardt (not iSAM2) is used here because the window is small
    and solved once from scratch.
    """
    graph = gtsam.NonlinearFactorGraph()
    values = gtsam.Values()
    frames = sorted(frames, key=lambda f: f.k)
    prior_rot = math.radians(cfg.init_batch_prior_sigma_rot_deg)

    prev: Optional[BufferedFrame] = None
    for f in frames:
        xk = K.X(f.k)
        if values.exists(xk):  # defensive: one camera frame per keyframe
            continue
        values.insert(xk, X_est[f.k])
        graph.add(pose_prior(xk, X_est[f.k], prior_rot, cfg.init_batch_prior_sigma_trans))
        if prev is not None:
            graph.add(odometry_factor(K.X(prev.k), xk, prev.E, f.E, f.stamp - prev.stamp, cfg))
        prev = f

    values.insert(K.S, S0)
    tag_index = K.TagIndex()
    proj: List[TagProjectionFactor] = []
    mount_rot = math.radians(cfg.sigma_mount_rot_deg)
    for f in frames:
        for d in f.detections:
            tkey = tag_index.key(d.tag)
            if not values.exists(tkey):
                values.insert(tkey, tags[d.tag].T_dock_tag)
                graph.add(pose_prior(tkey, tags[d.tag].T_dock_tag, mount_rot, cfg.sigma_mount_trans))
            pf = TagProjectionFactor(K.X(f.k), K.S, tkey, cams[f.camera], tags[d.tag], d.corners,
                                     cfg.sigma_px, cfg.tag_huber_k, cfg.min_projection_depth)
            proj.append(pf)
            graph.add(pf.factor)

    params = gtsam.LevenbergMarquardtParams()
    params.setMaxIterations(int(cfg.init_batch_max_iterations))
    try:
        result = gtsam.LevenbergMarquardtOptimizer(graph, values, params).optimize()
    except RuntimeError:
        return BatchResult(ok=False)

    return BatchResult(
        ok=True,
        S=result.atPose3(K.S),
        T={tag: result.atPose3(tag_index.key(tag)) for tag in tag_index.tags()},
        error=float(graph.error(result)),
        max_rms_px=max((pf.rms_px(result) for pf in proj), default=0.0),
    )


@dataclass
class InitDecision:
    accepted: bool
    S: Optional[gtsam.Pose3] = None
    T: Dict[TagKey, gtsam.Pose3] = field(default_factory=dict)
    reason: str = ""


def try_initialize(frames: List[BufferedFrame], X_est: Dict[int, gtsam.Pose3],
                   cams: Dict[str, CameraModel], tags: Dict[TagKey, TagModel],
                   cfg: EstimatorConfig) -> InitDecision:
    """Run voting and, if it is conclusive, the batch solve.

    Acceptance (§7.3):

    * the winner's worst per-detection RMS error is below
      ``init_batch_max_rms_px``, and
    * if a competitor exists, its batch solution ends with at least
      ``init_batch_min_error_ratio`` times the winner's error.
    """
    v = vote(frames, cfg)
    if not v.ready:
        return InitDecision(False, reason=v.reason)

    win = batch_solve(frames, v.winner.S, X_est, cams, tags, cfg)
    if not win.ok:
        return InitDecision(False, reason="batch solve failed")
    if win.max_rms_px > cfg.init_batch_max_rms_px:
        return InitDecision(False, reason=f"batch residual {win.max_rms_px:.1f} px too high")

    if v.competitor is not None:
        comp = batch_solve(frames, v.competitor.S, X_est, cams, tags, cfg)
        # If the competitor converges to the same pose, it is not a real
        # alternative; otherwise it must be clearly worse.
        if comp.ok:
            same = pose_distance(comp.S, win.S)[0] < cfg.init_vote_trans_tol
            if not same and comp.error < cfg.init_batch_min_error_ratio * win.error:
                return InitDecision(
                    False, reason=f"ambiguous: competitor error {comp.error:.1f} "
                                  f"vs winner {win.error:.1f}")

    return InitDecision(True, S=win.S, T=win.T, reason="accepted")
