"""The dock estimator: owns the backend and runs the per-keyframe cycle (§6, §9, §10).

Per keyframe (:meth:`DockEstimator.process_keyframe`)
-----------------------------------------------------
1. **Stamp.** The caller supplies the keyframe time and ``E_k``, the EKF
   pose interpolated at that time (see ``core/frontend/scheduler.py``).
2. **Create X_k.** Initial value ``X_hat_k = X_{k-1} * (E_{k-1}^-1 * E_k)``:
   the previous *corrected* pose moved by the EKF's motion. The first
   keyframe starts at ``E_0``.
3. **Motion factor.** Odometry (or the anchor prior for the first keyframe),
   de-weighted while the DVL had no bottom lock and on the EKF snap when it
   returns (``core/frontend/dvl_monitor.py``).
4. **Tag factors.** Every detection of a station tag is used: the per-image
   checks (border, size, PnP fit) already ran in the tag detector, and the
   projection factor's Huber kernel bounds whatever bad detection is left.
   Its IPPE solutions are computed once here and reused. Then:

   * Before ``S`` exists: PnP hypotheses, then buffer the frame for
     initialization (§7).
   * In TRACKING: one projection factor per accepted tag. A tag seen for the
     first time gets a new ``T_j`` with its URDF prior. If the tag-implied
     AUV pose explains the corners better than the odometry prediction
     ``X_hat_k``, it becomes ``X_k``'s initial value (:meth:`_seed_from_tags`).

5. **Update the smoother.** If initialization succeeds on this keyframe, a second
   update inserts ``S``, the ``T_j`` and **all** buffered projection factors.
   Frames of *attached* cameras received since the previous keyframe
   (:meth:`DockEstimator.add_attached_frame`) whose stamp is not after this
   keyframe are anchored now and their factors go into the same update.
6. **Read back** ``X_k``, ``S`` and the ``T_j``.
7. **Output** quantities for the node (§10): the drift correction
   ``D = X_k * E_k^-1``, the ``odom -> station`` transform ``D^-1 * S`` and the
   docking pose ``X_k^-1 * S``.

GPS-readiness
-------------
Only two things assume the graph frame is ``odom``: :meth:`add_anchor` (the
gauge prior ``X_0 = E_0``) and the output transforms built from ``D``. A
future GPS mode replaces :meth:`add_anchor` and publishes ``D`` as
``map -> odom``. The rest of the cycle is frame-agnostic.
"""

from __future__ import annotations

import dataclasses
import logging
import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

import gtsam
import numpy as np

from ..common import keys as K
from ..common.config import EstimatorConfig
from ..common.model import CameraModel, Detection, TagFrame, TagKey, TagModel, tag_key_str
from ..factors.odometry import odometry_factor
from ..factors.priors import pose_prior, pose_prior_cov
from ..factors.projection import TagProjectionFactor, reprojection_rms
from ..frontend.dvl_monitor import DvlMonitor
from ..frontend.pnp import Solution, ippe_tag
from ..init.solver import BufferedFrame, frame_hypotheses
from ..init.state_machine import StationInitializer, StationState
from .smoother import Backend

log = logging.getLogger(__name__)


@dataclass
class AttachedResult:
    """What one attached camera's frame contributed (see :meth:`DockEstimator.add_attached_frame`)."""

    stamp: float                         # image stamp of the attached frame
    camera: str
    anchor_k: Optional[int] = None       # keyframe its factors act on
    accepted: List[TagKey] = field(default_factory=list)
    rejected: List[Tuple[TagKey, str]] = field(default_factory=list)
    # (tag, T_bt): tag pose in the ANCHOR keyframe's body frame,
    # ``(E_j^-1 E(t)) * T_bc * T_ct`` with the single-image PnP pose
    # ``T_ct``. For plots: ``X_j * T_bt`` is where this observation puts the tag.
    tag_obs: List[Tuple[TagKey, gtsam.Pose3]] = field(default_factory=list)


@dataclass
class KeyframeResult:
    """What one keyframe produced. The node turns it into outputs and diagnostics."""

    k: int
    stamp: float
    state: StationState
    X: gtsam.Pose3                       # corrected AUV pose in odom
    E: gtsam.Pose3                       # EKF pose in odom
    S: Optional[gtsam.Pose3] = None      # station dock_point in odom (TRACKING only)
    # 6x6 covariance (gtsam [rot, trans] order, tangent space of the docking
    # pose X_k^-1 * S); see EstimatorConfig.covariance_mode.
    dock_cov: Optional[np.ndarray] = None
    accepted: List[TagKey] = field(default_factory=list)
    rejected: List[Tuple[TagKey, str]] = field(default_factory=list)
    init_note: str = ""                  # why initialization did not (yet) succeed
    initialized_now: bool = False
    recovered: bool = False              # backend was rebuilt on this keyframe
    seeded_from_tag: bool = False        # X_k started at the tag-implied pose (_seed_from_tags)
    dvl_extra_sigma: float = 0.0         # extra odometry sigma (m) from a DVL dropout / re-lock
    update_time: float = 0.0             # seconds spent in this keyframe
    # The frame's usable detections as single-image poses: (tag, camera,
    # T_camera_tag of the best IPPE solution). Not used by the estimator
    # itself; for plots (where a tag lands in odom through E_k vs X_k).
    tag_obs: List[Tuple[TagKey, str, gtsam.Pose3]] = field(default_factory=list)
    # Attached frames (other cameras) anchored in this keyframe's update, to
    # this keyframe or a recent one; the accepted ones' factors entered the
    # graph now. ``accepted`` above stays this keyframe's own camera only.
    attached: List[AttachedResult] = field(default_factory=list)

    @property
    def n_attached_accepted(self) -> int:
        """Tag factors this update added from attached cameras."""
        return sum(len(a.accepted) for a in self.attached)

    @property
    def drift_correction(self) -> gtsam.Pose3:
        """``D = X_k * E_k^-1``: the transform that maps the EKF's odom onto the graph's.

        ``D`` is identity while the graph agrees with the EKF. It grows as
        tag observations pull ``X_k`` away from the drifting ``E_k``.
        """
        return self.X.compose(self.E.inverse())

    @property
    def drift_distance(self) -> float:
        """3D distance (m) between the graph's and the EKF's AUV position, ``||t(X_k) - t(E_k)||``.

        The size of the drift correction as plotted and reported. Unlike the
        translation of ``D``, it does not depend on where the odom origin is
        (``D``'s translation also carries yaw correction x distance from it).
        """
        return float(np.linalg.norm(self.X.translation() - self.E.translation()))

    @property
    def dock_in_base(self) -> Optional[gtsam.Pose3]:
        """``X_k^-1 * S``: dock_point in the AUV base_link. This is what docking uses."""
        return None if self.S is None else self.X.between(self.S)


class DockEstimator:
    """Factor-graph estimator of the docking station pose.

    Parameters
    ----------
    cfg
        Tuning parameters.
    cameras
        ``name -> CameraModel`` for every camera whose detections may arrive.
    tags
        ``(family, id) -> TagModel`` for every tag on the station that may be
        used. Detections of other tags are ignored.
    T_dock_base
        Station ``base_link`` expressed in ``dock_point`` (inverse of the
        station URDF ``dock_point_joint``). Only used for the
        ``odom -> station base_link`` output.
    """

    def __init__(self, cfg: EstimatorConfig, cameras: Dict[str, CameraModel],
                 tags: Dict[TagKey, TagModel],
                 T_dock_base: Optional[gtsam.Pose3] = None) -> None:
        self.cfg = cfg
        self.cameras = cameras
        self.tags = tags
        self.T_dock_base = T_dock_base if T_dock_base is not None else gtsam.Pose3()
        self.backend = Backend(cfg)
        self.tag_index = K.TagIndex()
        self.initializer = StationInitializer(cfg)
        # Fed with DVL bottom-lock message stamps by the node / replay
        # (add_dvl); inactive (no effect) until the first one arrives.
        self.dvl = DvlMonitor(cfg)

        self.k = -1                                   # index of the last keyframe
        self._prev_stamp: Optional[float] = None
        self._prev_E: Optional[gtsam.Pose3] = None
        self._prev_X: Optional[gtsam.Pose3] = None    # corrected pose of the last keyframe
        self.tracking_since: Optional[float] = None
        # Cached docking-pose covariance and when it was computed (it is
        # refreshed at most every cfg.covariance_period seconds).
        self._cov_cache: Optional[np.ndarray] = None
        self._cov_stamp: Optional[float] = None
        self.last_result: Optional[KeyframeResult] = None
        # Recent keyframes (k, stamp, E_k), for anchoring attached frames.
        self._history: Deque[Tuple[int, float, gtsam.Pose3]] = deque(maxlen=200)
        # Attached frames waiting to be anchored: (stamp, E(t), frame).
        self._attached_pending: Deque[Tuple[float, gtsam.Pose3, TagFrame]] = deque(maxlen=100)
        # Cumulative counters for diagnostics.
        self.n_tag_factors = 0
        self.n_rejected = 0
        self.n_recoveries = 0

    # ======================================================================
    # Public API
    # ======================================================================
    @property
    def state(self) -> StationState:
        return self.initializer.state

    def add_anchor(self, k: int, E: gtsam.Pose3, graph: gtsam.NonlinearFactorGraph) -> None:
        """Tie the graph to the EKF's odom frame (the gauge prior, §5 notes).

        Odometry and projection factors only constrain poses *relative* to
        each other. Without this prior the whole graph (every ``X_k``
        together with ``S``) could move rigidly in all 6 DoF without changing
        any residual. A very tight prior ``X_k = E_k`` on the first keyframe
        fixes that freedom and makes the graph frame equal to ``odom``. Once
        ``X_0`` leaves the fixed-lag window, the smoother keeps this
        information as a marginal prior on the remaining variables.

        This is the GPS seam: a GPS mode would anchor to ENU instead.
        """
        graph.add(pose_prior(K.X(k), E, self.cfg.anchor_sigma_rot, self.cfg.anchor_sigma_trans))

    def process_keyframe(self, stamp: float, E: gtsam.Pose3,
                         frame: Optional[TagFrame] = None) -> KeyframeResult:
        """Run one keyframe of the cycle (§6).

        ``frame`` is the detector message at this keyframe. With no
        detections (or ``frame=None``) the keyframe gets odometry only, which
        keeps the ``X`` chain going while no tag is in view.

        Keyframes must arrive in increasing ``stamp`` order (the scheduler
        guarantees it).
        """
        t0 = time.perf_counter()
        cfg = self.cfg
        k = self.k + 1
        xk = K.X(k)
        graph = gtsam.NonlinearFactorGraph()
        values = gtsam.Values()
        timestamps: Dict[int, float] = {xk: stamp}
        result = KeyframeResult(k=k, stamp=stamp, state=self.state, X=E, E=E)

        # ---- 2. Initial value of X_k, 3. motion factors ------------------
        if self._prev_X is None:
            # First keyframe: X_0 = E_0, pinned by the anchor prior.
            X_hat = E
            self.add_anchor(k, E, graph)
        else:
            # Previous corrected pose, moved by the EKF's relative motion.
            X_hat = self._prev_X.compose(self._prev_E.between(E))
            # Extra distrust while the DVL had no bottom lock, and on the
            # EKF snap when it returned (0 with lock, or with no DVL topic).
            dvl_sigma = self.dvl.extra_sigma(self._prev_stamp, stamp)
            result.dvl_extra_sigma = dvl_sigma
            graph.add(odometry_factor(K.X(k - 1), xk, self._prev_E, E,
                                      stamp - self._prev_stamp, cfg, dvl_sigma))
        values.insert(xk, X_hat)

        # ---- 4. Tag factors ----------------------------------------------
        pending: Optional[BufferedFrame] = None
        if frame is not None and frame.detections:
            if frame.camera not in self.cameras:
                result.rejected += [(d.tag, f"unknown camera '{frame.camera}'")
                                    for d in frame.detections]
            else:
                dets, sols = self._usable_detections(frame, result)
                result.tag_obs = [(d.tag, frame.camera, sols[d.tag][0][0]) for d in dets]
                if dets and self.state != StationState.TRACKING:
                    pending = self._prepare_init_frame(k, stamp, E, X_hat, frame.camera,
                                                       dets, result)
                elif dets:
                    self._add_tracking_factors(k, stamp, X_hat, frame.camera, dets, sols, graph,
                                               values, timestamps, result)

        result.attached = self._attach_pending(k, stamp, E, graph, values, timestamps)

        if self.state == StationState.TRACKING:
            # Keep S and the T_j inside the fixed-lag window forever (§9).
            K.restamp(timestamps, [K.S] + self.tag_index.keys(), stamp)

        # ---- 5. Update ---------------------------------------------------
        try:
            self.backend.update(graph, values, timestamps)
        except RuntimeError as e:
            log.warning("backend update failed at keyframe %d (%s); rebuilding", k, e)
            self._recover(k, stamp, E, X_hat)
            result.recovered = True
            pending = None  # the rebuilt graph lost the buffered X_k; restart init

        if pending is not None:
            self._run_initialization(pending, stamp, result)

        # ---- 6. Read back ------------------------------------------------
        self.k = k
        X_k = self.backend.pose(xk)
        if X_k is None:
            X_k = X_hat
        self._prev_X, self._prev_E, self._prev_stamp = X_k, E, stamp
        self._history.append((k, stamp, E))
        result.X = X_k
        result.state = self.state
        if self.state == StationState.TRACKING:
            result.S = self.backend.pose(K.S)
            result.dock_cov = self._dock_covariance(xk, stamp)
        self.n_tag_factors += len(result.accepted) + result.n_attached_accepted
        self.n_rejected += len(result.rejected) + sum(len(a.rejected) for a in result.attached)
        result.update_time = time.perf_counter() - t0
        self.last_result = result
        return result

    def add_attached_frame(self, stamp: float, E: gtsam.Pose3, frame: TagFrame) -> None:
        """Queue another camera's frame for attaching to the nearest keyframe (§6, attached cameras).

        The frame never becomes a keyframe. Its tags become projection
        factors on the keyframe ``X_j`` nearest ``stamp`` (before or after):
        the camera was at ``X(t) T_bc`` when the image was taken, and between
        ``t_j`` and ``t`` the AUV moved by what the EKF measured,
        ``dE = E_j^-1 E(t)`` (``t`` may be before ``t_j``), so

            X(t) T_bc  =  X_j * dE * T_bc.

        The factor is the ordinary projection factor on ``X_j`` with the
        camera extrinsic ``dE * T_bc`` in place of ``T_bc``: no new variable,
        no new factor type. Only the EKF's error over ``|t - t_j|`` enters:
        at most half a camera frame (33 ms) with a detector synced to the
        keyframe camera (``sync_to``), up to one detector period without.

        The frame is anchored in :meth:`process_keyframe`, inside its update
        (no extra optimizer run): at the first keyframe not older than it, so
        that both neighbours of ``t`` are known and a synced frame lands on
        its own keyframe. See :meth:`_attach_pending` for when it is rejected.
        """
        self._attached_pending.append((float(stamp), E, frame))

    def _attach_pending(self, k: int, stamp: float, E: gtsam.Pose3,
                        graph: gtsam.NonlinearFactorGraph, values: gtsam.Values,
                        timestamps: Dict[int, float]) -> List[AttachedResult]:
        """Anchor the queued attached frames not after keyframe ``k`` and add their factors.

        Candidates are this keyframe (being created in this update) and the
        recent ones still in the smoother. A frame is rejected (every
        detection, with the reason) before TRACKING (initialization uses the
        keyframe camera), when its nearest keyframe is more than
        ``attach_max_dt`` away (the keyframe camera was late or silent), or
        when the DVL had no bottom lock between the two (the EKF steps there).
        A tag first seen by an attached camera gets its ``T_j`` at the URDF
        pose with the mounting prior, as for the keyframe camera.
        """
        due = [p for p in self._attached_pending if p[0] <= stamp]
        if not due:
            return []
        self._attached_pending = deque((p for p in self._attached_pending if p[0] > stamp),
                                       maxlen=self._attached_pending.maxlen)
        cfg = self.cfg
        mount_rot = math.radians(cfg.sigma_mount_rot_deg)
        candidates = [*self._history, (k, stamp, E)]
        out: List[AttachedResult] = []
        for t, E_t, frame in due:
            r = AttachedResult(stamp=t, camera=frame.camera)
            out.append(r)

            def reject_all(reason: str) -> None:
                r.rejected += [(d.tag, reason) for d in frame.detections]

            if frame.camera not in self.cameras:
                reject_all(f"unknown camera '{frame.camera}'")
                continue
            if self.state != StationState.TRACKING:
                reject_all("attached camera before TRACKING (initialization uses the keyframe "
                           "camera)")
                continue
            j, t_j, E_j = min(candidates, key=lambda c: abs(c[1] - t))
            if abs(t - t_j) > cfg.attach_max_dt:
                reject_all(f"no keyframe within attach_max_dt ({cfg.attach_max_dt:g} s)")
                continue
            if j != k and not self.backend.exists(K.X(j)):
                reject_all("anchor keyframe already marginalized")
                continue
            if self.dvl.extra_sigma(min(t, t_j), max(t, t_j)) > 0.0:
                reject_all("DVL dropout or re-lock between the keyframe and the image")
                continue
            r.anchor_k = j
            dets, sols = self._usable_detections(frame, r)
            cam = self.cameras[frame.camera]
            cam_at_anchor = dataclasses.replace(cam, T_bc=E_j.between(E_t).compose(cam.T_bc))
            for d in dets:
                tkey = self.tag_index.key(d.tag)
                if not self.backend.exists(tkey) and not values.exists(tkey):
                    values.insert(tkey, self.tags[d.tag].T_dock_tag)
                    graph.add(pose_prior(tkey, self.tags[d.tag].T_dock_tag, mount_rot,
                                         cfg.sigma_mount_trans))
                    timestamps[tkey] = stamp
                    log.info("tag %s entered the graph (attached camera)", tag_key_str(d.tag))
                graph.add(TagProjectionFactor(
                    K.X(j), K.S, tkey, cam_at_anchor, self.tags[d.tag], d.corners,
                    cfg.sigma_px, cfg.tag_huber_k, cfg.min_projection_depth).factor)
                r.accepted.append(d.tag)
                r.tag_obs.append((d.tag, cam_at_anchor.T_bc.compose(sols[d.tag][0][0])))
        return out

    # ======================================================================
    # Helpers for evaluation / outputs
    # ======================================================================
    def tag_estimates(self) -> Dict[TagKey, gtsam.Pose3]:
        """Current ``T_j`` estimates (pose of each tag in dock_point)."""
        out = {}
        for tag in self.tag_index.tags():
            p = self.backend.pose(self.tag_index.key(tag))
            if p is not None:
                out[tag] = p
        return out

    def predict_rms(self, det: Detection, X: Optional[gtsam.Pose3] = None) -> Optional[float]:
        """Reprojection RMS (px) of ``det`` at the current estimate.

        Used for held-out-frame and leave-one-tag-out validation (§13). The
        detection is *not* added to the graph. ``X`` defaults to the newest
        keyframe pose.
        """
        if self.state != StationState.TRACKING or det.camera not in self.cameras:
            return None
        X = X if X is not None else self._prev_X
        S = self.backend.pose(K.S)
        tag = self.tags.get(det.tag)
        if S is None or tag is None or X is None:
            return None
        T = self.tag_estimates().get(det.tag, tag.T_dock_tag)
        return reprojection_rms(X, S, T, self.cameras[det.camera], tag, det.corners)

    # ======================================================================
    # Step 4: which detections are used
    # ======================================================================
    def _usable_detections(self, frame: TagFrame, result: "KeyframeResult | AttachedResult"
                           ) -> Tuple[List[Detection], Dict[TagKey, List[Solution]]]:
        """The frame's detections of station tags, with their IPPE solutions (best first).

        There is no gate here (§8). The tag detector already dropped poor
        quads (border, minimum size, PnP fit), initialization votes over
        several frames, and the projection factor's Huber kernel bounds a
        bad detection that remains. A detection is only dropped if its tag
        is not on the station, or if PnP fails on it (a degenerate quad the
        detector's fit check normally prevents). The solutions are reused by
        initialization (station hypotheses) and the keyframe seed.
        """
        cam = self.cameras[frame.camera]
        dets: List[Detection] = []
        sols: Dict[TagKey, List[Solution]] = {}
        for d in frame.detections:
            tag = self.tags.get(d.tag)
            if tag is None:
                result.rejected.append((d.tag, "tag not in the station model"))
                continue
            s = ippe_tag(d, cam, tag)
            if not s:
                result.rejected.append((d.tag, "PnP failed"))
                continue
            dets.append(d)
            sols[d.tag] = s
        return dets, sols

    # ======================================================================
    # Step 4, before initialization
    # ======================================================================
    def _prepare_init_frame(self, k: int, stamp: float, E: gtsam.Pose3, X_hat: gtsam.Pose3,
                            camera: str, dets: List[Detection],
                            result: KeyframeResult) -> Optional[BufferedFrame]:
        """PnP hypotheses for gated detections seen before ``S`` exists (§7.1).

        ``dets`` are the frame's usable detections. The frame is returned for
        buffering in the initialization window; the caller hands it to
        :meth:`_run_initialization` after the backend update has created ``X_k``.
        """
        hyps, note = frame_hypotheses(X_hat, self.cameras[camera], dets, self.tags, self.cfg)
        if note:
            result.init_note = note
        return BufferedFrame(k=k, stamp=stamp, E=E, camera=camera,
                             detections=dets, hypotheses=hyps)

    def _run_initialization(self, pending: BufferedFrame, stamp: float,
                            result: KeyframeResult) -> None:
        """Buffer ``pending``, try to initialize, and insert the station on success (§7.3)."""
        init = self.initializer
        # Frames whose X_k left the fixed-lag window cannot be used any more.
        init.frames = [f for f in init.frames if self.backend.exists(K.X(f.k))]
        X_est = {f.k: self.backend.pose(K.X(f.k)) for f in init.frames + [pending]}
        decision = init.add_frame(pending, X_est, self.cameras, self.tags)
        result.init_note = decision.reason
        if not decision.accepted:
            return

        # ---- On accept: insert S, the T_j, their URDF priors and all buffered factors.
        cfg = self.cfg
        graph = gtsam.NonlinearFactorGraph()
        values = gtsam.Values()
        timestamps: Dict[int, float] = {K.S: stamp}
        values.insert(K.S, decision.S)
        mount_rot = math.radians(cfg.sigma_mount_rot_deg)
        for tag, T in decision.T.items():
            tkey = self.tag_index.key(tag)
            # Start from the batch solution; the prior stays at the URDF value.
            values.insert(tkey, T)
            graph.add(pose_prior(tkey, self.tags[tag].T_dock_tag, mount_rot, cfg.sigma_mount_trans))
            timestamps[tkey] = stamp
        for f in init.frames:
            for d in f.detections:
                pf = TagProjectionFactor(K.X(f.k), K.S, self.tag_index.key(d.tag),
                                         self.cameras[f.camera], self.tags[d.tag], d.corners,
                                         cfg.sigma_px, cfg.tag_huber_k, cfg.min_projection_depth)
                graph.add(pf.factor)
                if f.k == pending.k:
                    result.accepted.append(d.tag)
        try:
            self.backend.update(graph, values, timestamps)
        except RuntimeError as e:
            log.warning("inserting the initialized station failed (%s); staying in CANDIDATE", e)
            init.reset()
            self.tag_index.clear()
            result.accepted.clear()
            return
        n = sum(len(f.detections) for f in init.frames)
        log.info("station initialized from %d frames (%d tag observations)", len(init.frames), n)
        init.frames.clear()
        self.tracking_since = stamp
        result.initialized_now = True

    # ======================================================================
    # Step 4, in TRACKING
    # ======================================================================
    def _add_tracking_factors(self, k: int, stamp: float, X_hat: gtsam.Pose3, camera: str,
                              dets: List[Detection], sols: Dict[TagKey, List[Solution]],
                              graph: gtsam.NonlinearFactorGraph,
                              values: gtsam.Values, timestamps: Dict[int, float],
                              result: KeyframeResult) -> None:
        """Add one projection factor per detection, and pick ``X_k``'s initial value.

        1. **Initial value of X_k** (:meth:`_seed_from_tags`). It only sets
           where the optimizer starts from (the linearization point), never
           which measurements are used.
        2. **Factors.** A tag seen for the first time gets a new variable
           ``T_j`` at its URDF pose, with the mounting prior.
        """
        cfg = self.cfg
        cam = self.cameras[camera]
        S = self.backend.pose(K.S)
        T_est = self.tag_estimates()

        X_seed = self._seed_from_tags(X_hat, dets, sols, cam, S, T_est)
        if X_seed is not None:
            values.update(K.X(k), X_seed)
            result.seeded_from_tag = True
            log.info("keyframe %d starts at the tag-implied pose (%.3f m from the odometry "
                     "prediction)", k,
                     float(np.linalg.norm(X_seed.translation() - X_hat.translation())))

        mount_rot = math.radians(cfg.sigma_mount_rot_deg)
        for d in dets:
            tkey = self.tag_index.key(d.tag)
            if not self.backend.exists(tkey) and not values.exists(tkey):
                # First sighting of this tag: new T_j at its URDF pose, with its prior.
                values.insert(tkey, self.tags[d.tag].T_dock_tag)
                graph.add(pose_prior(tkey, self.tags[d.tag].T_dock_tag, mount_rot,
                                     cfg.sigma_mount_trans))
                timestamps[tkey] = stamp
                log.info("tag %s entered the graph", tag_key_str(d.tag))
            graph.add(TagProjectionFactor(K.X(k), K.S, tkey, cam, self.tags[d.tag], d.corners,
                                          cfg.sigma_px, cfg.tag_huber_k,
                                          cfg.min_projection_depth).factor)
            result.accepted.append(d.tag)

    def _seed_from_tags(self, X_hat: gtsam.Pose3, dets: List[Detection],
                        sols: Dict[TagKey, List[Solution]],
                        cam: CameraModel, S: gtsam.Pose3,
                        T_est: Dict[TagKey, gtsam.Pose3]) -> Optional[gtsam.Pose3]:
        """Tag-implied AUV pose, if it explains the corners better than ``X_hat``.

        ``X_hat`` chains the EKF motion from the previous keyframe, so after
        a while without tags (or a DVL dropout) it carries all the drift since
        then. The smoother (iSAM2 inside) linearizes a new variable at its
        initial value; starting far from the answer costs iterations and, at
        close range, can put
        corners behind the camera. Each accepted tag gives a better
        starting point.

        **Translation only.** Inverting the full PnP pose,
        ``X = S T_j T_ct^-1 T_bc^-1``, multiplies the PnP *rotation* error by
        the range (a few degrees at 2.5 m is decimetres; the flipped IPPE
        solution is worse). The tag *position* in the camera, ``p_ct``, is
        precise (exact bearing, 1-3 % range error, the same for both IPPE
        solutions). So the predicted orientation ``R = R_X_hat`` is kept
        (roll and pitch follow the EKF's gravity-referenced tilt through the
        odometry chain; yaw drift is small and the factors
        correct it), and only the translation is solved so that the tag
        centre lands where the camera measured it:

            R (R_bc p_ct + t_bc) + t = p_tag    =>    t = p_tag - R (R_bc p_ct + t_bc)

        with ``p_tag`` the tag centre in odom from the current ``S * T_j``.

        **Choice.** Poses are scored by the mean reprojection RMS of all
        accepted tags. The best-scoring candidate (one per tag) is returned
        if it beats ``X_hat`` *and* lies farther from it than its own
        uncertainty, ``seed_min_shift + range * seed_size_tol_px / side_px``
        (5 cm plus the PnP range noise of that tag: the range change for an
        apparent-size error of ``seed_size_tol_px``); otherwise ``None`` (keep
        ``X_hat``). Without the distance
        threshold the candidate "wins" on most frames by a fraction of a
        pixel (it fits this frame's tag exactly, while ``X_hat`` carries a
        few mm of odometry noise), which is harmless but makes the flag
        meaningless. The threshold is a distance, not pixels, because the
        pixel size of a given error grows as 1 / range: in the tank test a
        5 px threshold fired on 740 keyframes, 666 of them for sub-cm
        differences at under 0.8 m range.
        """
        def score(X: gtsam.Pose3) -> float:
            return float(np.mean([
                reprojection_rms(X, S, T_est.get(d.tag, self.tags[d.tag].T_dock_tag),
                                 cam, self.tags[d.tag], d.corners) for d in dets]))

        best_score = score(X_hat)
        R = X_hat.rotation()
        best, best_tol = None, 0.0
        for d in dets:
            p_ct = sols[d.tag][0][0].translation()                    # tag centre in the camera
            p_tag = S.compose(T_est.get(d.tag, self.tags[d.tag].T_dock_tag)).translation()
            t = p_tag - R.rotate(cam.T_bc.transformFrom(p_ct))       # camera offset in body, rotated
            X_cand = gtsam.Pose3(R, gtsam.Point3(*t))
            sc = score(X_cand)
            if sc < best_score:
                # The proposal is itself uncertain by the PnP range noise
                # (range ~ f * size / side, so d range = range * d side / side).
                rng = float(np.linalg.norm(p_ct))
                tol = self.cfg.seed_min_shift + rng * self.cfg.seed_size_tol_px / max(
                    d.min_side_px(), 1.0)
                best, best_score, best_tol = X_cand, sc, tol
        if best is None:
            return None
        shift = float(np.linalg.norm(best.translation() - X_hat.translation()))
        return best if shift > best_tol else None

    def _dock_covariance(self, xk: int, stamp: float) -> Optional[np.ndarray]:
        """Covariance of the docking pose ``X_k^-1 * S`` (see ``covariance_mode``).

        "relative": with the joint marginal ``C`` of ``(X_k, S)`` and the
        Jacobians ``H1, H2`` of ``between(X_k, S)``, the first-order
        covariance of the relative pose is ``[H1 H2] C [H1 H2]^T``. It
        correctly cancels the AUV's absolute (odom) uncertainty, which is
        shared by ``X_k`` and ``S`` and does not affect docking.
        """
        cfg = self.cfg
        if cfg.covariance_mode == "none":
            return None
        fresh = self._cov_stamp is not None and stamp - self._cov_stamp < cfg.covariance_period
        if fresh and self._cov_cache is not None:
            return self._cov_cache
        cov = None
        if cfg.covariance_mode == "station":
            cov = self.backend.marginal_covariance(K.S)
        else:
            joint = self.backend.joint_marginal_covariance([xk, K.S])
            X, S = self.backend.pose(xk), self.backend.pose(K.S)
            if joint is not None and X is not None and S is not None:
                H1 = np.zeros((6, 6), order="F")
                H2 = np.zeros((6, 6), order="F")
                X.between(S, H1, H2)
                A = np.hstack([H1, H2])
                cov = A @ joint @ A.T
        self._cov_cache, self._cov_stamp = cov, stamp
        return cov

    # ======================================================================
    # §9 Error recovery
    # ======================================================================
    def _recover(self, k: int, stamp: float, E: gtsam.Pose3, X_hat: gtsam.Pose3) -> None:
        """Rebuild the backend from the last good estimate after an optimizer failure.

        The new graph contains the current keyframe ``X_k`` (at its predicted
        value), plus ``S`` and the ``T_j`` when tracking. Each gets a prior
        from its marginal covariance in the old backend, or default sigmas if
        that query also fails. This keyframe's tag observations are dropped.
        If even that minimal graph fails, everything is reset: the station is
        forgotten and initialization starts again. The node never exits on
        an optimizer error.
        """
        self.n_recoveries += 1
        cfg = self.cfg
        old = self.backend
        tracking = self.state == StationState.TRACKING
        S = old.pose(K.S) if tracking else None
        T_vals = {tag: old.pose(self.tag_index.key(tag)) for tag in self.tag_index.tags()}

        def prior(key: int, value: gtsam.Pose3, s_rot: float, s_tr: float):
            cov = old.marginal_covariance(key)
            if cov is not None and np.all(np.isfinite(cov)):
                return pose_prior_cov(key, value, cov)
            return pose_prior(key, value, s_rot, s_tr)

        xk = K.X(k)
        graph = gtsam.NonlinearFactorGraph()
        values = gtsam.Values()
        timestamps = {xk: stamp}
        values.insert(xk, X_hat)
        # X_k does not exist in the old backend (its update is what failed),
        # so it is held at its predicted value with the marginal covariance of
        # X_{k-1}: the best available measure of how well the corrected AUV
        # pose is known.
        cov = old.marginal_covariance(K.X(k - 1))
        if cov is not None and np.all(np.isfinite(cov)):
            graph.add(pose_prior_cov(xk, X_hat, cov))
        else:
            graph.add(pose_prior(xk, X_hat, math.radians(1.0), 0.05))
        if tracking and S is not None:
            values.insert(K.S, S)
            graph.add(prior(K.S, S, math.radians(2.0), 0.05))
            timestamps[K.S] = stamp
            mount_rot = math.radians(cfg.sigma_mount_rot_deg)
            for tag, T in T_vals.items():
                if T is None:
                    continue
                tkey = self.tag_index.key(tag)
                values.insert(tkey, T)
                graph.add(prior(tkey, T, mount_rot, cfg.sigma_mount_trans))
                timestamps[tkey] = stamp

        self.backend = Backend(cfg)
        self.initializer.frames.clear()
        try:
            self.backend.update(graph, values, timestamps)
            return
        except RuntimeError as e:
            log.error("backend rebuild failed (%s); resetting the station estimate", e)

        # Last resort: forget the station, keep the corrected AUV pose.
        self.backend = Backend(cfg)
        self.initializer.reset()
        self.tag_index.clear()
        self.tracking_since = None
        graph = gtsam.NonlinearFactorGraph()
        values = gtsam.Values()
        values.insert(xk, X_hat)
        graph.add(pose_prior(xk, X_hat, cfg.anchor_sigma_rot, cfg.anchor_sigma_trans))
        self.backend.update(graph, values, {xk: stamp})
