"""Batch Levenberg-Marquardt post-processing: full smoothing of a recorded run.

Usage::

    ros2 run race_auv_factorgraph batch_lm BAG --config tank_dock_fuser.yaml [--csv out.csv]

The real-time estimator (iSAM2) publishes each ``X_k`` as known *at keyframe
k*: a filtering estimate that only uses measurements up to that moment. A
post-processing solve re-optimizes the whole trajectory at once, so every
``X_k`` also benefits from tag observations made *after* it (smoothing),
and from a full nonlinear re-optimization instead of iSAM2's incremental,
thresholded relinearization.

How the batch graph is obtained
-------------------------------
The bag is replayed through the same scheduler and estimator as the node
(``replay_bag.replay``). The backend keeps a copy of every factor it
receives (``Backend.full_graph``), including those the fixed-lag smoother
has long since marginalized. The batch problem is then exactly the graph the
real-time run built: the same keyframes, the same EKF odometry factors, and the same tag factors. Levenberg-Marquardt optimizes
it from scratch, initialized at the real-time per-keyframe estimates.

Two estimates of every ``X_k`` are compared:

``X_rt``    real-time (filter) estimate, published at keyframe k;
``X_lm``    batch LM optimum (full smoothing).
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import gtsam
import numpy as np

from ..core.common import keys as K
from ..core.common.geometry import pose_distance
from ..core.init.state_machine import StationState
from ..ros.params import build_setup, getter_from_yaml
from .replay_bag import replay


@dataclass
class BatchSolution:
    stamps: Dict[int, float] = field(default_factory=dict)        # k -> keyframe stamp
    E: Dict[int, gtsam.Pose3] = field(default_factory=dict)       # EKF pose at keyframe k
    X_rt: Dict[int, gtsam.Pose3] = field(default_factory=dict)    # real-time estimate
    X_lm: Dict[int, gtsam.Pose3] = field(default_factory=dict)    # batch LM optimum
    S_rt: Dict[int, gtsam.Pose3] = field(default_factory=dict)    # real-time S at keyframe k
    S_lm: Optional[gtsam.Pose3] = None
    tracking_since: Optional[float] = None
    error_init: float = float("nan")     # graph error at the real-time estimates
    error_lm: float = float("nan")       # graph error at the LM optimum
    lm_iterations: int = 0
    lm_seconds: float = 0.0
    n_factors: int = 0


def optimize_full_graph(est, X_init: Dict[int, gtsam.Pose3], max_iterations: int = 100):
    """Levenberg-Marquardt on every factor the estimator's backend received.

    ``X_init`` (keyframe index -> pose) initializes the ``X_k``; normally the
    real-time estimates, i.e. what a post-processor finds in the log. ``S``
    and the ``T_j`` start at the smoother's final values (they are never
    marginalized). Returns ``(result, stats)``.
    """
    graph = est.backend.full_graph()
    final = est.backend.estimate              # smoother's final estimate (recent keys only)
    init = gtsam.Values()
    for key in graph.keyVector():
        if init.exists(key):
            continue
        sym = gtsam.Symbol(key)
        if chr(sym.chr()) == "x" and sym.index() in X_init:
            init.insert(key, X_init[sym.index()])
        elif final.exists(key):
            init.insert(key, final.atPose3(key))

    params = gtsam.LevenbergMarquardtParams()
    params.setMaxIterations(int(max_iterations))
    params.setRelativeErrorTol(1e-7)
    params.setAbsoluteErrorTol(1e-9)
    t0 = time.perf_counter()
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, init, params)
    result = optimizer.optimize()
    stats = {
        "n_factors": int(graph.size()),
        "seconds": time.perf_counter() - t0,
        "iterations": int(optimizer.iterations()),
        "error_init": float(graph.error(init)),
        "error_lm": float(graph.error(result)),
    }
    return result, stats


def batch_solve(bag: str, config, max_iterations: int = 100) -> BatchSolution:
    """Replay ``bag`` with ``config``, then solve the whole graph with LM."""
    get = getter_from_yaml(config)
    setup = build_setup(get, lambda m: print("warning:", m))
    # Covariance is not needed here.
    cfg = dataclasses.replace(setup.cfg, covariance_mode="none")
    sol = BatchSolution()

    def on_keyframe(est, res, held) -> None:
        sol.stamps[res.k] = res.stamp
        sol.E[res.k] = res.E
        sol.X_rt[res.k] = res.X
        if res.S is not None:
            sol.S_rt[res.k] = res.S

    est, _ = replay(bag, get, setup, cfg=cfg, on_keyframe=on_keyframe)
    if est.n_recoveries:
        print(f"warning: {est.n_recoveries} backend rebuilds during the replay; the batch graph "
              "only covers the part after the last one")
    sol.tracking_since = est.tracking_since

    result, stats = optimize_full_graph(est, sol.X_rt, max_iterations)
    sol.n_factors = stats["n_factors"]
    sol.lm_seconds = stats["seconds"]
    sol.lm_iterations = stats["iterations"]
    sol.error_init = stats["error_init"]
    sol.error_lm = stats["error_lm"]

    for k in sol.stamps:
        if result.exists(K.X(k)):
            sol.X_lm[k] = result.atPose3(K.X(k))
    if est.state == StationState.TRACKING:
        sol.S_lm = result.atPose3(K.S)
    return sol


def summarize(sol: BatchSolution) -> List[str]:
    """Human-readable comparison lines (also printed by plot_bev --batch)."""
    lines = [f"batch LM: {sol.n_factors} factors, {len(sol.X_lm)} poses, "
             f"{sol.lm_iterations} iterations in {sol.lm_seconds:.1f} s; graph error "
             f"real-time {sol.error_init:.1f} -> LM {sol.error_lm:.1f}"]
    track = [k for k in sol.X_lm if sol.tracking_since is not None
             and sol.stamps[k] >= sol.tracking_since]

    def stats(a: Dict[int, gtsam.Pose3], b: Dict[int, gtsam.Pose3], ks: List[int]) -> str:
        d = [pose_distance(a[k], b[k]) for k in ks if k in a and k in b]
        if not d:
            return "n/a"
        t = np.array([x[0] for x in d]) * 100
        r = np.degrees([x[1] for x in d])
        return (f"position median {np.median(t):.2f} / max {t.max():.2f} cm, "
                f"rotation median {np.median(r):.2f} / max {r.max():.2f} deg")

    lines.append(f"  X real-time vs LM (tracking): {stats(sol.X_rt, sol.X_lm, track)}")
    if sol.S_lm is not None and sol.S_rt:
        last = max(sol.S_rt)
        dt, dr = pose_distance(sol.S_rt[last], sol.S_lm)
        lines.append(f"  S final real-time vs LM: {dt * 100:.2f} cm, {math.degrees(dr):.2f} deg")
        # Docking pose X_k^-1 S: real-time (X_rt, S_rt at k) vs smoothed (X_lm, S_lm).
        dock = {k: sol.X_rt[k].between(sol.S_rt[k]) for k in track if k in sol.S_rt}
        dock_lm = {k: sol.X_lm[k].between(sol.S_lm) for k in track}
        lines.append(f"  docking pose X^-1 S, real-time vs LM: {stats(dock, dock_lm, track)}")
    return lines


def main(argv: List[str] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("bag", help="bag with EKF odometry and the corner-detection topics")
    ap.add_argument("--config", required=True, nargs="+",
                    help="dock_fuser parameter file(s) used for the run, merged in order")
    ap.add_argument("--csv", default="", help="optional per-keyframe output")
    ap.add_argument("--max-iterations", type=int, default=100)
    args = ap.parse_args(argv)

    sol = batch_solve(args.bag, args.config, args.max_iterations)
    print("\n".join(summarize(sol)))
    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["k", "stamp"] + [f"{n}_{a}" for n in ("E", "X_rt", "X_lm")
                                         for a in ("x", "y", "z", "yaw_deg")])
            for k in sorted(sol.X_lm):
                row = [k, sol.stamps[k]]
                for P in (sol.E[k], sol.X_rt[k], sol.X_lm[k]):
                    t = P.translation()
                    row += [t[0], t[1], t[2], math.degrees(P.rotation().yaw())]
                w.writerow(row)
        print(f"wrote {args.csv}")


if __name__ == "__main__":
    main()
