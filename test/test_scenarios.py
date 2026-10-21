"""Synthetic end-to-end scenarios (§12 step 3, §13 "Simulation").

Each test flies the synthetic approach from ``tools/synthetic.py`` through
the scheduler and the estimator, and checks the docking pose ``X_k^-1 * S``
against ground truth. Tolerances are loose enough to be robust to the noise
seed but tight enough to catch a broken factor, a wrong frame or a flipped
initialization (a flip gives decimetre / tens-of-degree errors).
"""

import math

import numpy as np
import pytest

from race_auv_factorgraph.core.common import keys as K
from race_auv_factorgraph.core.common.config import EstimatorConfig
from race_auv_factorgraph.core.backend.estimator import DockEstimator
from race_auv_factorgraph.core.init.state_machine import StationState
from race_auv_factorgraph.tools.synthetic import (ScenarioConfig, dock_error,
                                                  make_scenario, run)


def fly(scenario_cfg: ScenarioConfig, **est_kwargs):
    scn = make_scenario(scenario_cfg)
    est = DockEstimator(EstimatorConfig(**est_kwargs), scn.cameras, scn.tags, scn.T_dock_base)
    return scn, est, run(scn, est)


def final_error(scn, results, window: float = 2.0):
    """Worst docking error over the last ``window`` seconds."""
    end = results[-1].stamp
    errs = [dock_error(scn, r) for r in results if r.S is not None and r.stamp > end - window]
    assert errs, "never reached TRACKING"
    return max(e[0] for e in errs), max(e[1] for e in errs)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_clean_approach(seed):
    """All tags, EKF drifting: initializes, then converges to ~mm close in."""
    scn, est, res = fly(ScenarioConfig(seed=seed, station_yaw=0.3 - 0.4 * seed))
    assert est.state == StationState.TRACKING
    assert sum(r.initialized_now for r in res) == 1
    dt, dr = final_error(scn, res)
    assert dt < 0.02 and dr < math.radians(1.0)


def test_single_tag_only():
    """Only tag 146 is ever visible: multi-frame single-tag initialization (§7)."""
    scn, est, res = fly(ScenarioConfig(visible_tags=[146]))
    assert est.state == StationState.TRACKING
    # Right after initialization a single far tag is loose, but it must not
    # be the flipped solution (that would be tens of degrees off).
    first = next(r for r in res if r.initialized_now)
    assert dock_error(scn, first)[1] < math.radians(20.0)
    dt, dr = final_error(scn, res)
    assert dt < 0.05 and dr < math.radians(3.0)


def test_drift_is_corrected():
    """Large EKF drift: the relative pose stays accurate and D absorbs the drift."""
    scn, est, res = fly(ScenarioConfig(drift_xy=(0.05, -0.03), drift_yaw=math.radians(1.0)))
    dt, dr = final_error(scn, res)
    assert dt < 0.03 and dr < math.radians(1.5)
    # The EKF drifted ~1.5 m over the run; the graph's correction must be
    # substantial (not identity).
    D = res[-1].drift_correction
    assert float((D.translation() ** 2).sum()) ** 0.5 > 0.2


def test_tags_lost_and_reacquired():
    """6 s without any tag: drift builds up, then tracking resumes."""
    scn, est, res = fly(ScenarioConfig(blackouts=[(10.0, 16.0)]))
    after = [r for r in res if r.stamp > 16.5]
    assert any(r.accepted for r in after), "tags were never re-admitted after the blackout"
    dt, dr = final_error(scn, res)
    assert dt < 0.02 and dr < math.radians(1.0)


def test_short_lag_marginalizes_but_keeps_tracking():
    """A 10 s lag on a 25 s run: old X_k are marginalized, S and the T_j
    (re-stamped every update) are not."""
    scn, est, res = fly(ScenarioConfig(), smoother_lag=10.0)
    assert not est.backend.exists(K.X(0))
    assert est.backend.exists(K.S)
    assert est.state == StationState.TRACKING
    dt, dr = final_error(scn, res)
    assert dt < 0.02 and dr < math.radians(1.0)


def test_mounting_error_tolerated():
    """Real tag mountings off by ~1 deg / 5 mm from the URDF."""
    scn, est, res = fly(ScenarioConfig(mount_error_sigma=(math.radians(1.0), 0.005), seed=3))
    dt, dr = final_error(scn, res)
    assert dt < 0.03 and dr < math.radians(2.0)


def test_recovers_from_backend_failure(monkeypatch):
    """An optimizer exception mid-run rebuilds the backend and keeps tracking (§9)."""
    scn = make_scenario(ScenarioConfig())
    est = DockEstimator(EstimatorConfig(), scn.cameras, scn.tags, scn.T_dock_base)

    from race_auv_factorgraph.core.backend import smoother as backend_mod
    original = backend_mod.Backend.update
    calls = {"n": 0}

    def flaky(self, graph, values, timestamps):
        calls["n"] += 1
        if calls["n"] == 80:  # well into TRACKING
            raise RuntimeError("Indeterminant linear system (injected)")
        return original(self, graph, values, timestamps)

    monkeypatch.setattr(backend_mod.Backend, "update", flaky)
    res = run(scn, est)
    assert est.n_recoveries == 1
    assert any(r.recovered for r in res)
    assert est.state == StationState.TRACKING
    dt, dr = final_error(scn, res)
    assert dt < 0.03 and dr < math.radians(1.5)


def test_docking_covariance():
    """The docking-pose covariance exists, is a valid covariance, and is
    tighter than the odom-referenced marginal of S (which also carries the
    AUV's accumulated odometry uncertainty)."""
    scn, est, res = fly(ScenarioConfig(), smoother_lag=10.0)
    last = res[-1]
    C = last.dock_cov
    assert C is not None and C.shape == (6, 6)
    np.testing.assert_allclose(C, C.T, atol=1e-12)
    assert np.all(np.linalg.eigvalsh(C) > 0)
    S_marg = est.backend.marginal_covariance(K.S)
    assert np.trace(C[3:, 3:]) < np.trace(S_marg[3:, 3:])


def test_reacquired_after_long_blackout_with_large_drift():
    """Tags lost for 8 s while the EKF drifts 0.24 m (~75 px at 2.5 m), as
    in the close-range part of the tank test. Nothing compares the tag with
    the drifted estimate, so it is used again on the first frame it is seen,
    and the keyframe starts at the tag-implied pose instead of the drifted
    odometry prediction."""
    # Yaw drift is left out: the synthetic drift rotates about the odom
    # origin, which turns yaw drift into a position error growing with the
    # distance from the origin, much faster than a real EKF.
    sc = ScenarioConfig(blackouts=[(10.0, 18.0)], drift_xy=(0.025, -0.015), drift_yaw=0.0)
    scn, est, res = fly(sc)
    after = [r for r in res if r.stamp > 18.0]
    first = next(r for r in after if r.accepted)
    # Used again on the first frame after the blackout (5 Hz).
    assert first.stamp < 18.3
    assert first.seeded_from_tag
    # Seeding happens only after the drift, not in steady tracking.
    assert sum(r.seeded_from_tag for r in res) <= 3
    assert sum(len(r.rejected) for r in res) <= 5
    dt, dr = final_error(scn, res)
    assert dt < 0.02 and dr < math.radians(1.0)


def test_batch_lm_on_full_graph():
    """Batch post-processing (tools/batch_lm.py): LM on the full graph the
    real-time run built, started from the real-time estimates. The fixed-lag
    smoother (10 s lag) has marginalized the early part of this run, but the
    backend's copy of every factor still holds all of it. LM must not end
    above its starting error, and the smoothed poses stay accurate against
    ground truth."""
    from race_auv_factorgraph.core.common.geometry import pose_distance
    from race_auv_factorgraph.tools.batch_lm import optimize_full_graph

    scn, est, res = fly(ScenarioConfig(), smoother_lag=10.0)
    assert not est.backend.exists(K.X(0))            # really marginalized
    lm, stats = optimize_full_graph(est, {r.k: r.X for r in res})
    assert stats["error_lm"] <= stats["error_init"] + 1e-6
    S_lm = lm.atPose3(K.S)
    track = [r for r in res if r.S is not None]
    errs = [pose_distance(lm.atPose3(K.X(r.k)).between(S_lm), scn.true_dock_in_base(r.stamp))[0]
            for r in track[len(track) // 2:]]
    assert max(errs) < 0.03


def test_dvl_dropout_deweights_odometry():
    """A 6 s DVL dropout with tags in view: the EKF dead-reckons 0.57 m off and
    snaps back when the DVL returns, as robot_localization did in the tank
    test (1.05 m after an 11 s dropout). With the DVL monitor those odometry
    steps are de-weighted and the tags decide; without it the drift and the
    snap pull the docking pose (synthetic: p95 14.5 vs 10.9 cm over 5 seeds)."""
    sc = ScenarioConfig(dvl_dropouts=[(8.0, 14.0)])

    def p95_from(res, t0):
        return float(np.percentile([dock_error(scn, r)[0] for r in res
                                    if r.S is not None and r.stamp > t0], 95))

    scn, est, res = fly(sc)
    assert sum(r.dvl_extra_sigma > 0 for r in res) > 20          # the dropout's steps
    scn_off = make_scenario(sc)
    scn_off.dvl = []                                              # control: no DVL topic
    est_off = DockEstimator(EstimatorConfig(), scn_off.cameras, scn_off.tags, scn_off.T_dock_base)
    res_off = run(scn_off, est_off)
    assert p95_from(res, 8.0) < p95_from(res_off, 8.0)
    dt, dr = final_error(scn, res)
    assert dt < 0.02 and dr < math.radians(1.0)



def test_attached_down_camera_takes_over_at_the_dock():
    """cam_front loses the tags at the end of a pass over the station; cam_down (attached) sees
    the bottom tags. Its factors enter the graph on the cam_front keyframes (no keyframes of
    its own) and keep the docking pose at least as good as cam_front alone."""
    base = dict(height=0.6, end_x=0.0)
    alone_scn, _, alone = fly(ScenarioConfig(**base))
    both_scn, _, both = fly(ScenarioConfig(**base, cameras=("cam_front", "cam_down")))
    # Same keyframes: only cam_front's messages create them.
    assert [r.stamp for r in both] == [r.stamp for r in alone]
    attached = [a for r in both for a in r.attached]
    accepted = [t for a in attached for t in a.accepted]
    assert accepted and {t[1] for t in accepted} <= {176, 185, 541, 558}
    assert all(a.anchor_k is not None for a in attached if a.accepted)
    # Final dock error over the last 2 s, where cam_front is blind and the
    # graph dead-reckons on the EKF alone (2.75 cm here) unless cam_down sees
    # the bottom tags (0.35 cm).
    t_alone, _ = final_error(alone_scn, alone)
    t_both, _ = final_error(both_scn, both)
    assert t_both < 0.5 * t_alone
    assert t_both < 0.01


def test_attached_camera_tags_enter_after_initialization():
    """cam_front reports only the front tags, cam_down only the bottom tags (per-camera
    camera_tag_ids). The station initializes from cam_front; the bottom tags' T_j are created
    on cam_down's first sighting (URDF pose + mounting prior), with no initialization of their
    own, and cam_down then holds the docking pose while cam_front is blind."""
    tags = {"cam_front": [146, 541, 558], "cam_down": [176, 185]}
    base = dict(height=0.6, end_x=0.0, camera_tags=tags)
    alone_scn, _, alone = fly(ScenarioConfig(**base))
    scn, est, res = fly(ScenarioConfig(**base, cameras=("cam_front", "cam_down")))
    front = {d.tag[1] for f in scn.frames if f.camera == "cam_front" for d in f.detections}
    assert front <= {146, 541, 558}
    accepted = {t[1] for r in res for a in r.attached for t in a.accepted}
    assert accepted == {176, 185}
    assert {("36h11", 176), ("36h11", 185)} <= set(est.tag_estimates())
    t_alone, _ = final_error(alone_scn, alone)
    t_both, _ = final_error(scn, res)
    assert t_both < 0.5 * t_alone and t_both < 0.01


@pytest.mark.parametrize("offset", [-0.03, 0.02])
def test_synced_down_camera_anchors_to_its_own_keyframe(offset):
    """With cam_down's detector synced to cam_front (sync_to), each attached frame is anchored
    to the cam_front keyframe nearest it, also when its image is slightly *earlier* (the
    previous keyframe is 0.2 s back): the EKF only has to bridge |offset|."""
    tags = {"cam_front": [146, 541, 558], "cam_down": [176, 185]}
    scn, _, res = fly(ScenarioConfig(height=0.6, end_x=0.0, camera_tags=tags, sync_cameras=True,
                                     sync_offset=offset, cameras=("cam_front", "cam_down")))
    stamp = {r.k: r.stamp for r in res}
    gaps = [a.stamp - stamp[a.anchor_k] for r in res for a in r.attached if a.accepted]
    assert gaps and max(abs(g - offset) for g in gaps) < 1e-6
    t, _ = final_error(scn, res)
    assert t < 0.01
