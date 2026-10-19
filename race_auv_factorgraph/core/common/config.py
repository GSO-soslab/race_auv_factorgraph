"""Estimator tuning parameters.

Every field of :class:`EstimatorConfig` is exposed by the ROS node as a
parameter ``estimator.<field_name>`` with the default given here, so this file
is the single reference for what each knob does. Section numbers refer to
``docs/DOCK_FUSER_PIPELINE.md``.

Units: metres, radians, seconds and pixels, unless the name says otherwise
(``*_deg`` fields are degrees, converted where they are used).
"""

from __future__ import annotations

from dataclasses import dataclass, fields


@dataclass
class EstimatorConfig:
    # ------------------------------------------------------------------ §9 backend
    # The backend is always gtsam's IncrementalFixedLagSmoother (core/backend/smoother.py):
    # AUV poses X_k older than `smoother_lag` seconds are marginalized, so
    # the cost per keyframe stays flat on long runs. Plain iSAM2 is not
    # offered: every tag factor touches the station variable S, and with a
    # graph that grows forever each relinearization of S slowed to seconds
    # per keyframe. Post-processing uses batch LM on the full graph instead
    # (tools/batch_lm.py).
    #
    # Keep the lag well above the initialization window and the first few
    # seconds of tracking. Marginalization freezes the linearization point,
    # so a short lag (e.g. 5 s) turns the station's early, still-loose
    # estimate into a biased prior. In the synthetic approach, 5 s gave
    # 0.6 cm final error versus 0.15 cm at 10 s and 30 s.
    smoother_lag: float = 30.0
    # iSAM2 relinearization inside the smoother: relinearize a variable when
    # its update exceeds this threshold; check every `isam_relinearize_skip`
    # updates.
    isam_relinearize_threshold: float = 0.01
    isam_relinearize_skip: int = 1
    # Extra iSAM2 update() calls (with no new factors) after each keyframe.
    # Each one is an additional Gauss-Newton step on the affected variables.
    isam_extra_iterations: int = 1

    # ------------------------------------------------------------------ §5 priors
    # EKF init prior on X_0 (and on the first keyframe after a backend
    # rebuild): "very tight". It only fixes the gauge (graph frame = odom).
    anchor_sigma_rot: float = 1e-4
    anchor_sigma_trans: float = 1e-4
    # Tag URDF prior on each T_j: mounting tolerance of a tag on the station.
    sigma_mount_rot_deg: float = 2.0
    sigma_mount_trans: float = 0.01

    # ------------------------------------------------------------------ §5 odometry
    # Random-walk odometry noise. Per axis, sigma^2 = q_t * dt + q_d * dd,
    # where dt is the keyframe interval [s] and dd the distance travelled [m].
    # The total uncertainty over an interval then does not depend on how many
    # keyframes it is split into.
    odom_q_rot_t: float = 1e-5      # rad^2 / s
    odom_q_rot_d: float = 1e-4      # rad^2 / m
    odom_q_trans_t: float = 1e-4    # m^2 / s
    odom_q_trans_d: float = 1e-3    # m^2 / m
    # Lower bounds so that a zero-length interval never yields a singular
    # (infinitely confident) factor.
    odom_min_sigma_rot: float = 1e-4
    odom_min_sigma_trans: float = 1e-4
    # Huber threshold (in sigmas) on the odometry factor: lets tags win a
    # disagreement with the odometry (§5 notes).
    odom_huber_k: float = 3.0
    # DVL bottom-lock monitor (core/frontend/dvl_monitor.py; node parameter
    # dvl_topic). While the DVL is silent the EKF dead-reckons and drifts;
    # when it returns the EKF snaps back by about that drift. Odometry steps
    # get extra translation sigma: dvl_blind_speed * (blind time of the step),
    # and on the re-lock step dvl_blind_speed * (length of the silence).
    # Tank run: snaps <= ~0.1 m/s * silence (11.2 s -> 1.05 m).
    dvl_timeout: float = 0.5        # s without a DVL message = bottom lock lost (DVL at 5 Hz)
    dvl_blind_speed: float = 0.1    # m/s, how fast the dead-reckoned position may drift
    dvl_relock_margin: float = 0.3  # s after re-lock that still count as the snap step

    # ------------------------------------------------------------------ §5 tag factor
    # Corner measurement noise, per pixel coordinate.
    sigma_px: float = 1.0
    # Huber threshold (in sigmas) on the tag projection factor.
    tag_huber_k: float = 2.0
    # Minimum depth used in the projection (avoids division by ~0 if the
    # optimizer moves a corner behind the camera mid-iteration).
    min_projection_depth: float = 0.05

    # ------------------------------------------------------------------ §8 detections
    # No gate in the fuser: every detection of a station tag becomes a factor.
    # The tag detector (race_auv_camera_tag_factorgraph) drops poor quads
    # (border, minimum side, PnP fit) and the Huber kernel above bounds the rest.

    # Tag-implied initial value of X_k (DockEstimator._seed_from_tags): used
    # when the tag-implied pose fits the accepted tags better than the
    # odometry prediction X_hat_k AND lies farther from it than
    # seed_min_shift + range * seed_size_tol_px / side_px: this distance plus
    # the proposal's own PnP range noise (the range change for an apparent-size
    # error of seed_size_tol_px pixels). Only the optimizer's starting point
    # changes, never which measurements are used. A distance, not a pixel threshold: at 0.3 m
    # range 3 mm is already 10 px. Tank test 30-420 s: steady-tracking
    # differences are 0.3-1 cm (2 cm max), real drift events 10-12 cm.
    seed_min_shift: float = 0.05
    seed_size_tol_px: float = 2.0

    # ------------------------------------------------------------------ §7 initialization
    # Hypotheses come from PnP. Keep only the better IPPE solution when its
    # reprojection error is below `ratio` times the other's (unambiguous view).
    init_ambiguity_ratio: float = 0.3
    # Sliding window of buffered tag frames.
    init_window_duration: float = 3.0
    init_window_max_frames: int = 15
    # Voting: hypotheses from different frames agree within these tolerances.
    # Both grow with range, because single-tag PnP noise does: at 6 m a
    # 12 cm tag is ~18 px wide, and its PnP rotation is good to only several
    # degrees.
    init_vote_trans_tol: float = 0.10
    init_vote_trans_tol_rel: float = 0.05   # fraction of range
    init_vote_rot_tol_deg: float = 5.0
    init_vote_rot_tol_deg_per_m: float = 1.5
    # Winner needs at least this many supporting frames...
    init_min_votes: int = 5
    # ...with viewpoint diversity: viewing direction change, OR lateral
    # camera displacement (perpendicular to the viewing direction).
    init_min_view_change_deg: float = 3.0
    init_min_lateral_baseline: float = 0.05
    # A competing cluster must have at most this fraction of the winner's
    # votes. The default 1.0 only requires the winner not to be outvoted:
    # far away, the true and the flipped cluster often tie on votes, and the
    # batch solve below (which scores both against every corner in the
    # window) is the deciding test. Lower it to demand a vote margin too.
    init_max_competitor_ratio: float = 1.0
    # Batch solve acceptance (§7.3).
    init_batch_max_rms_px: float = 4.0
    init_batch_min_error_ratio: float = 2.0   # competitor error / winner error
    init_batch_max_iterations: int = 50
    # Loose priors holding the buffered X_k near their current estimates.
    init_batch_prior_sigma_rot_deg: float = 5.0
    init_batch_prior_sigma_trans: float = 0.5

    # ------------------------------------------------------------------ §10 output
    # Covariance published with /dock_point/pose:
    #   "relative": covariance of the docking pose X_k^-1 * S itself, from the
    #               joint marginal of (X_k, S). This is the honest number for
    #               docking. It needs a full gtsam.Marginals pass over the
    #               smoother's graph (~15 ms at ~800 factors, growing with
    #               plain iSAM2), so it is refreshed at most every
    #               `covariance_period` seconds and reused in between.
    #   "station":  marginal covariance of S alone (cheap). It is expressed in
    #               odom, so it also contains the AUV's accumulated odometry
    #               uncertainty and overstates the docking uncertainty.
    #   "none":     publish "unknown" (covariance[0] = -1).
    covariance_mode: str = "relative"
    covariance_period: float = 1.0


def config_from_dict(values: dict) -> EstimatorConfig:
    """Build a config from a ``{field_name: value}`` dict, ignoring unknown keys.

    Values are cast to the type of the field's default, so a ROS parameter
    given as ``5`` for a float field still works.
    """
    cfg = EstimatorConfig()
    for f in fields(EstimatorConfig):
        if f.name in values and values[f.name] is not None:
            default = getattr(cfg, f.name)
            setattr(cfg, f.name, type(default)(values[f.name]))
    return cfg
