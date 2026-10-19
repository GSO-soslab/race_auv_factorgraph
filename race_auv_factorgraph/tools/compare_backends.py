"""Compare fixed-lag smoother settings against a batch LM reference on one recorded run.

Usage::

    ros2 run race_auv_factorgraph compare_backends DETECTION_BAG \\
        --config tank_dock_fuser.yaml --out /path/to/report \\
        [--variants fixed_lag:30 fixed_lag:60] [--reference fixed_lag:30]
        [--max-wall fixed_lag:60=900] [--start 30 --end 420]

``DETECTION_BAG`` needs the EKF odometry and the corner-detection topics, e.g.
the output of ``bag_test.launch.py ... run_fuser:=false record_images:=false``.

What is compared
----------------
Each *variant* replays the bag through the same scheduler and estimator as
the live node (``replay_bag.replay``), differing only in the smoother lag:

``fixed_lag:<s>``  fixed-lag smoother that marginalizes variables older than <s> s.

For each variant the **real-time** estimate is recorded: ``X_k`` and ``S`` as
known at keyframe k, plus the update time of that keyframe. The variants run
in parallel processes on identical input, so they get identical keyframes;
only their estimates differ.

The reference is **batch Levenberg-Marquardt** on the full graph of the
``--reference`` variant's replay (``batch_lm.optimize_full_graph``): every
factor that run accepted, re-optimized at once (full smoothing,
post-processing). The backend keeps a copy of every factor it receives, so
this works for a fixed-lag run too.

A longer lag costs more per keyframe, because every tag factor is attached
to the single station variable ``S`` and each relinearization of ``S``
re-evaluates all of them within the window (tank run, 30-420 s: median update
1 ms and p95 49 ms at 30 s lag, p95 150 ms at 60 s). ``--max-wall fixed_lag:60=900`` stops a variant after that many seconds of wall time;
its partial result is compared over the keyframes it completed.

Differences to the reference are reported for the robot pose ``X_k`` and for
the docking pose ``X_k^-1 * S`` that docking uses. Covariance computation is
switched off in every variant (``covariance_mode: none``) so that the update
times measure the estimator alone.

Outputs (in ``--out``): ``comparison.png`` (full and zoomed BEV, drift
correction, docking-pose difference to LM, update time, tag/seed/DVL-dropout
raster, blind periods shaded), ``summary.txt``, ``summary.csv``, and
``meta.json`` plus one ``variant_<name>.npz`` per variant. ``--replot``
redraws from those without replaying.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import multiprocessing as mp
import os
import time
from typing import Dict, List, Tuple

import numpy as np

# Plot colours per variant (reference LM is black).
_COLORS = ["tab:blue", "tab:green", "tab:purple", "tab:brown", "tab:cyan", "tab:olive"]


def parse_variant(spec: str) -> Tuple[str, dict]:
    """``"fixed_lag:30"`` -> (name, estimator overrides)."""
    if spec.startswith("fixed_lag:"):
        lag = float(spec.split(":", 1)[1])
        return f"fixed_lag_{lag:g}s", {"smoother_lag": lag}
    raise ValueError(f"unknown variant '{spec}' (use fixed_lag:<seconds>)")


def _mat(P) -> np.ndarray:
    return np.asarray(P.matrix(), dtype=float)


class _Budget(Exception):
    """Raised from the keyframe callback to stop a variant at its wall-time budget."""


def run_variant(bag: str, config, name: str, overrides: dict, batch: bool,
                max_wall: float = float("inf"), window: Tuple = (None, None)) -> dict:
    """Replay the bag with one backend; return per-keyframe arrays (picklable).

    Runs in a worker process: imports are local so the parent stays light.
    ``max_wall`` stops the replay after that many seconds; the keyframes done
    so far are returned (``completed = False``).
    """
    from ..core.init.state_machine import StationState
    from ..ros.params import build_setup, getter_from_yaml
    from .replay_bag import replay

    get = getter_from_yaml(config)
    setup = build_setup(get, lambda m: None)
    cfg = dataclasses.replace(setup.cfg, covariance_mode="none", **overrides)
    rec = {k: [] for k in ("k", "stamp", "X", "E", "S", "has_S", "update_ms", "n_acc",
                           "n_rej", "seeded", "tagframe", "init", "dvl_sigma")}
    t_start = time.perf_counter()
    last_print = [t_start]

    def on_keyframe(est, res, held) -> None:
        rec["k"].append(res.k)
        rec["stamp"].append(res.stamp)
        rec["X"].append(_mat(res.X))
        rec["E"].append(_mat(res.E))
        rec["has_S"].append(res.S is not None)
        rec["S"].append(_mat(res.S) if res.S is not None else np.full((4, 4), np.nan))
        rec["update_ms"].append(res.update_time * 1000.0)
        rec["n_acc"].append(len(res.accepted))
        rec["n_rej"].append(len(res.rejected))
        rec["seeded"].append(res.seeded_from_tag)
        rec["tagframe"].append(bool(res.accepted or res.rejected)
                               or res.initialized_now)
        rec["init"].append(res.initialized_now)
        rec["dvl_sigma"].append(res.dvl_extra_sigma)
        now = time.perf_counter()
        if now - last_print[0] > 60.0:
            last_print[0] = now
            print(f"[{name}] keyframe {res.k}, {now - t_start:.0f} s elapsed", flush=True)
        if now - t_start > max_wall:
            raise _Budget()

    est, completed = None, True
    try:
        est, _ = replay(bag, get, setup, cfg=cfg, on_keyframe=on_keyframe,
                        start=window[0], end=window[1])
    except _Budget:
        completed = False
    out = {key: np.asarray(v) for key, v in rec.items()}
    out["name"] = name
    out["completed"] = completed
    out["wall_s"] = time.perf_counter() - t_start
    out["n_recoveries"] = est.n_recoveries if est is not None else 0
    print(f"[{name}] {'done' if completed else 'stopped at the wall-time budget'}: "
          f"{len(rec['k'])} keyframes in {out['wall_s']:.0f} s", flush=True)

    if batch and est is not None:
        from ..core.common import keys as K
        from ..core.common.geometry import pose_from_matrix
        from .batch_lm import optimize_full_graph
        X_init = {int(k): pose_from_matrix(X) for k, X in zip(out["k"], out["X"])}
        result, stats = optimize_full_graph(est, X_init)
        out["X_lm"] = np.array([_mat(result.atPose3(K.X(int(k)))) if result.exists(K.X(int(k)))
                                else np.full((4, 4), np.nan) for k in out["k"]])
        out["S_lm"] = (_mat(result.atPose3(K.S)) if est.state == StationState.TRACKING
                       else np.full((4, 4), np.nan))
        out["lm_stats"] = stats
        print(f"[{name}] batch LM: {stats['iterations']} iterations, {stats['seconds']:.1f} s, "
              f"error {stats['error_init']:.1f} -> {stats['error_lm']:.1f}", flush=True)
    return out


def _worker(args):
    return run_variant(*args)


# ===========================================================================
# Analysis
# ===========================================================================
def _dock(X: np.ndarray, S: np.ndarray) -> np.ndarray:
    """Docking pose translation X^-1 S for stacked 4x4s (n,4,4) and S (n,4,4) or (4,4)."""
    Xi = np.linalg.inv(X)
    return (Xi @ S)[..., :3, 3]


def analyse(results: List[dict], ref: dict) -> Tuple[List[dict], Dict[str, dict]]:
    """Per-variant summary rows and per-variant time series against the LM reference."""
    X_lm_all, S_lm = ref["X_lm"], ref["S_lm"]
    dock_lm_all = _dock(X_lm_all, np.broadcast_to(S_lm, X_lm_all.shape))
    rows, series = [], {}
    for r in results:
        n = len(r["stamp"])                     # a budget-stopped variant covers a prefix
        assert np.allclose(r["stamp"], ref["stamp"][:n]), "variants produced different keyframes"
        X_lm, dock_lm = X_lm_all[:n], dock_lm_all[:n]
        trk = r["has_S"]
        vis = trk & (r["n_acc"] > 0)            # keyframes with accepted tag factors
        dpos = np.linalg.norm(r["X"][:, :2, 3] - X_lm[:, :2, 3], axis=1) * 100
        ddock = np.full(len(dpos), np.nan)
        ddock[trk] = np.linalg.norm(_dock(r["X"][trk], r["S"][trk]) - dock_lm[trk], axis=1) * 100
        u = r["update_ms"]
        S_last = r["S"][np.where(trk)[0][-1]] if trk.any() else None

        def q(a, p):
            a = a[np.isfinite(a)]
            return float(np.percentile(a, p)) if len(a) else float("nan")

        init_t = r["stamp"][r["init"]][0] if r["init"].any() else float("nan")
        rows.append({
            "variant": r["name"],
            "completed": "yes" if r["completed"] else "no (budget)",
            "covers_s": float(r["stamp"][-1] - r["stamp"][0]),
            "wall_s": float(r["wall_s"]),
            "init_s": init_t - r["stamp"][0],
            "keyframes": len(u),
            "tag_kf_accepted": int((r["n_acc"] > 0).sum()),
            "tag_kf_rejected": int(((r["n_rej"] > 0) & (r["n_acc"] == 0)).sum()),
            "tags_rejected": int(r["n_rej"].sum()),
            "seeded_from_tag": int(_seeded(r).sum()),
            "recoveries": int(r["n_recoveries"]),
            "update_ms_median": q(u, 50), "update_ms_p95": q(u, 95),
            "update_ms_p99": q(u, 99), "update_ms_max": float(u.max()),
            "total_compute_s": float(u.sum() / 1000.0),
            "dock_vs_lm_cm_median_visible": q(ddock[vis], 50),
            "dock_vs_lm_cm_p95_visible": q(ddock[vis], 95),
            "dock_vs_lm_cm_median_all": q(ddock, 50),
            "dock_vs_lm_cm_p95_all": q(ddock, 95),
            "dock_vs_lm_cm_max_all": q(ddock, 100),
            "pos_vs_lm_cm_median": q(dpos[trk], 50),
            "pos_vs_lm_cm_max": q(dpos[trk], 100),
            "S_final_vs_lm_cm": (float(np.linalg.norm(S_last[:3, 3] - S_lm[:3, 3]) * 100)
                                 if S_last is not None and np.all(np.isfinite(S_lm)) else float("nan")),
        })
        series[r["name"]] = {"dpos": dpos, "ddock": ddock}
    return rows, series


def tag_observations(bag: str, setup, start, end) -> Dict[float, Tuple[str, np.ndarray]]:
    """Tag centre in the camera frame for every tag frame in the window.

    Returns ``{round(stamp, 6): (camera, p_ct)}``, with ``p_ct`` from the best
    IPPE solution of the first known tag in the frame. Keyframe stamps equal
    frame stamps, so the dict can be indexed with keyframe stamps.
    """
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message

    from ..core.common.model import Detection, normalize_family
    from ..core.frontend.pnp import ippe_tag
    from .replay_bag import _open_reader

    reader = _open_reader(bag)
    t_bag0 = reader.get_metadata().starting_time.nanoseconds * 1e-9
    cam_by_topic = {topic: cam for cam, topic in setup.camera_topics.items()}
    types = {t.name: t.type for t in reader.get_all_topics_and_types()}
    cls = {t: get_message(types[t]) for t in cam_by_topic if t in types}
    out: Dict[float, Tuple[str, np.ndarray]] = {}
    while reader.has_next():
        topic, data, tns = reader.read_next()
        if topic not in cls:
            continue
        t_rel = tns * 1e-9 - t_bag0
        if (start is not None and t_rel < start) or (end is not None and t_rel > end):
            continue
        msg = deserialize_message(data, cls[topic])
        cam = setup.cameras[cam_by_topic[topic]]
        for d in msg.detections:
            key = (normalize_family(d.family), int(d.id))
            if key not in setup.tags:
                continue
            det = Detection(cam.name, key, np.array([[c.x, c.y] for c in d.corners], dtype=float))
            sols = ippe_tag(det, cam, setup.tags[key])
            if sols:
                stamp = msg.header.stamp.sec + 1e-9 * msg.header.stamp.nanosec
                out[round(stamp, 6)] = (cam.name, sols[0][0].translation())
            break
    return out


# Line style per variant: (colour, label suffix). The reference LM is black dashed.
_VARIANT_COLORS = {"fixed_lag_30s": "tab:blue", "fixed_lag_60s": "tab:orange"}


def _seeded(r) -> np.ndarray:
    """Per-keyframe "X_k seeded from tag" flags (``reloc`` in runs saved before the lean gate)."""
    return np.asarray(r["seeded"] if "seeded" in r else r["reloc"]).astype(bool)


def plot(results: List[dict], ref: dict, series: Dict[str, dict], setup, out: str,
         t_bag0: float, obs: Dict[float, Tuple[str, np.ndarray]], title: str) -> str:
    """One figure in the style of plot_bev: full + zoomed BEV, then time series.

    Markers (tag frames, tag-seeded keyframes, init, DVL dropouts) come from the
    reference run, whose full graph defines the batch LM solution.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    from ..core.common.geometry import pose_from_matrix

    t = ref["stamp"] - t_bag0                         # seconds into the bag
    colors = {r["name"]: _VARIANT_COLORS.get(r["name"], _COLORS[i % len(_COLORS)])
              for i, r in enumerate(results)}
    E, X_ref, X_lm = ref["E"], ref["X"], ref["X_lm"]
    trk = ref["has_S"]
    acc = ref["n_acc"] > 0
    rej = (ref["n_rej"] > 0) & ~acc
    reloc = _seeded(ref)
    dvl = ref["dvl_sigma"] > 0 if "dvl_sigma" in ref else np.zeros(len(t), bool)
    i_init = np.where(ref["init"])[0]
    S_lm = pose_from_matrix(ref["S_lm"]) if np.all(np.isfinite(ref["S_lm"])) else None

    # Blind periods: while tracking, gaps of > 2 s between accepted tag frames.
    blind = []
    tv = t[acc & trk]
    for a, b in zip(tv[:-1], tv[1:]):
        if b - a > 2.0:
            blind.append((a, b))

    # Per-detection tag positions in odom, through the EKF and through the poses.
    idx = {round(s, 6): i for i, s in enumerate(ref["stamp"])}
    via = {"ekf": [], "lm": [], "rt": []}
    for stamp, (cam_name, p_ct) in obs.items():
        i = idx.get(stamp)
        if i is None:
            continue
        p_b = setup.cameras[cam_name].T_bc.transformFrom(p_ct)          # tag centre in base_link
        for key, M in (("ekf", E[i]), ("lm", X_lm[i]), ("rt", X_ref[i])):
            if np.all(np.isfinite(M)):
                via[key].append(M[:3, :3] @ p_b + M[:3, 3])
    via = {k: np.array(v).reshape(-1, 3) for k, v in via.items()}

    fig = plt.figure(figsize=(17, 19))
    # Rows: BEV pair, legend strip, three time series, raster.
    gs = fig.add_gridspec(6, 2, height_ratios=[3.0, 0.32, 1.0, 1.0, 1.0, 0.35],
                          hspace=0.3, wspace=0.12)
    ax_full, ax_zoom = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])

    def draw_bev(ax, detail: bool) -> None:
        # Tag-frame markers go *under* the tracks (light, small), so the
        # tracks stay readable where thousands of frames overlap.
        ms = 10 if detail else 4
        ax.scatter(X_ref[acc, 0, 3], X_ref[acc, 1, 3], s=ms, color="limegreen", zorder=2, alpha=0.5,
                   lw=0, label=f"tag frame accepted ({acc.sum()})")
        if rej.any():
            ax.scatter(X_ref[rej, 0, 3], X_ref[rej, 1, 3], s=ms * 2, color="tab:red", zorder=2,
                       label=f"tag frame rejected ({rej.sum()})")
        ax.plot(E[:, 0, 3], E[:, 1, 3], color="0.6", lw=1.0, zorder=3, label="EKF (raw)")
        for r in results:
            ax.plot(r["X"][:, 0, 3], r["X"][:, 1, 3], color=colors[r["name"]], lw=1.3, zorder=4,
                    label=f"{r['name']} (real time)")
        ax.plot(X_lm[:, 0, 3], X_lm[:, 1, 3], color="k", lw=1.3, ls="--", zorder=5,
                label="batch LM (post-processing)")
        ax.plot(E[0, 0, 3], E[0, 1, 3], "o", color="0.3", ms=6, zorder=6)
        ax.annotate("start", E[0, :2, 3], textcoords="offset points", xytext=(5, 5), fontsize=8)
        if reloc.any():
            ax.scatter(X_ref[reloc, 0, 3], X_ref[reloc, 1, 3], s=45 if detail else 22, marker="D",
                       facecolors="none", edgecolors="magenta", lw=1.2, zorder=6,
                       label=f"X_k seeded from tag ({reloc.sum()})")
        if dvl.any():
            ax.scatter(E[dvl, 0, 3], E[dvl, 1, 3], s=6, color="red", zorder=7,
                       label=f"odometry de-weighted, DVL dropout ({dvl.sum()} keyframes)")
        if len(i_init):
            p = X_ref[i_init[0], :2, 3]
            ax.plot(p[0], p[1], "*", color="k", ms=18 if detail else 14, zorder=8,
                    label=f"station init (t = {t[i_init[0]]:.1f} s)")
        if detail and len(via["ekf"]):
            ax.scatter(via["ekf"][:, 0], via["ekf"][:, 1], marker="x", s=10, lw=0.7, color="0.45",
                       zorder=3, label="tag via EKF pose")
            ax.scatter(via["rt"][:, 0], via["rt"][:, 1], s=5, color=colors[ref["name"]], alpha=0.5,
                       zorder=3, label=f"tag via {ref['name']} pose")
            ax.scatter(via["lm"][:, 0], via["lm"][:, 1], s=5, color="k", alpha=0.5, zorder=3,
                       label="tag via batch LM pose")
        # Final station of every variant (hollow), LM station drawn in full.
        for r in results:
            if r["has_S"].any():
                Sv = r["S"][np.where(r["has_S"])[0][-1]]
                ax.plot(Sv[0, 3], Sv[1, 3], "s", mfc="none", mec=colors[r["name"]], mew=1.8,
                        ms=12 if detail else 9, zorder=7)
        if S_lm is not None:
            d = S_lm.translation()
            ax.plot(d[0], d[1], "s", color="k", ms=7, zorder=8, label="dock_point (batch LM; hollow = real time)")
            xa = S_lm.rotation().matrix()[:, 0]
            ax.arrow(d[0], d[1], 0.25 * xa[0], 0.25 * xa[1], width=0.008, color="k", zorder=8)
            base = S_lm.compose(setup.T_dock_base).translation()
            ax.plot(base[0], base[1], "D", color="purple", ms=6, zorder=8, label="station base_link")
            for key, tag in setup.tags.items():
                W = S_lm.compose(tag.T_dock_tag)
                c = W.translation()
                ex = W.rotation().matrix()[:, 0] * tag.size / 2
                outn = -W.rotation().matrix()[:, 2]           # tag z points INTO the tag
                ax.plot([c[0] - ex[0], c[0] + ex[0]], [c[1] - ex[1], c[1] + ex[1]],
                        color="crimson", lw=3, zorder=9)
                ax.arrow(c[0], c[1], 0.18 * outn[0], 0.18 * outn[1], width=0.005, color="crimson", zorder=9)
                ax.annotate(f"tag {key[1]}", c[:2], textcoords="offset points", xytext=(6, -12),
                            fontsize=8, color="crimson")
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("odom x [m]")
        ax.set_ylabel("odom y [m]")

    draw_bev(ax_full, detail=False)
    draw_bev(ax_zoom, detail=True)
    ax_full.set_title("full run", fontsize=10)
    ax_zoom.set_title("docking area (zoom)", fontsize=10)
    # Zoom window: the docking area, i.e. tag frames within 2.5 m of the tag,
    # the per-detection tag clouds and the station, padded.
    if S_lm is not None:
        tag_xy = np.array([S_lm.compose(tg.T_dock_tag).translation()[:2] for tg in setup.tags.values()])
        near = acc & (np.linalg.norm(X_ref[:, :2, 3] - tag_xy.mean(0), axis=1) < 2.5)
        pts = [X_ref[near, :2, 3], tag_xy, S_lm.translation()[None, :2],
               S_lm.compose(setup.T_dock_base).translation()[None, :2]]
        pts += [via[k][:, :2] for k in via if len(via[k])]
    else:
        pts = [X_ref[acc, :2, 3]]
    P = np.vstack([q for q in pts if len(q)])
    lo, hi = P.min(0) - 0.25, P.max(0) + 0.25
    ax_zoom.set_xlim(lo[0], hi[0])
    ax_zoom.set_ylim(lo[1], hi[1])
    ax_zoom.set_aspect("equal", adjustable="box")   # keep the zoom limits
    ax_full.add_patch(Rectangle(lo, *(hi - lo), fill=False, ls=":", lw=1.2, color="0.2", zorder=10))
    ax_full.annotate("zoom", (lo[0], hi[1]), textcoords="offset points", xytext=(2, 3), fontsize=8)
    # Legend in its own strip under the BEV pair.
    ax_leg = fig.add_subplot(gs[1, :])
    ax_leg.axis("off")
    handles, labels = ax_zoom.get_legend_handles_labels()
    ax_leg.legend(handles, labels, loc="center", ncol=5, fontsize=8, frameon=True, markerscale=1.5)

    # ---------------------------------------------------------------- time series
    axs = [fig.add_subplot(gs[i, :]) for i in range(2, 6)]
    for a in axs[1:]:
        a.sharex(axs[0])

    def drift(X: np.ndarray) -> np.ndarray:
        """Drift correction: 3D distance between the graph's and the EKF's AUV position."""
        n = len(X)
        return np.linalg.norm(X[:, :3, 3] - E[:n, :3, 3], axis=1)

    for r in results:
        n = len(r["stamp"])
        axs[0].plot(t[:n], drift(r["X"]), color=colors[r["name"]], lw=1.0,
                    label=f"{r['name']} (real time)")
        axs[1].plot(t[:n], series[r["name"]]["ddock"], color=colors[r["name"]], lw=1.0, label=r["name"])
        axs[2].scatter(t[:n], r["update_ms"], s=1.5, color=colors[r["name"]], label=r["name"])
    axs[0].plot(t, drift(X_lm), color="k", ls="--", lw=1.0, label="batch LM")
    axs[0].set_ylabel("drift correction\n‖t(X) − t(E)‖ 3D [m]")
    axs[1].set_ylabel("docking pose X^-1 S\nvs batch LM [cm]")
    axs[2].set_ylabel("update time [ms]")
    axs[2].set_yscale("log")
    axs[3].vlines(t[acc], 0.55, 1.0, color="tab:green", lw=0.5)
    if rej.any():
        axs[3].vlines(t[rej], 0.55, 1.0, color="tab:red", lw=0.5)
    axs[3].vlines(t[reloc], 0.05, 0.45, color="magenta", lw=1.0)
    if dvl.any():
        axs[3].vlines(t[dvl], 0.15, 0.35, color="red", lw=0.5)
    axs[3].set_yticks([0.25, 0.78])
    axs[3].set_yticklabels(["tag seed / DVL dropout", "tag frames"], fontsize=8)
    axs[3].set_ylim(0, 1.05)
    axs[3].set_xlabel("time [s into the bag]")
    for k, a in enumerate(axs):
        for j, (b0, b1) in enumerate(blind):
            a.axvspan(b0, b1, color="0.88", lw=0, zorder=0,
                      label="no tag in view" if (k < 2 and j == 0) else None)
        if len(i_init):
            a.axvline(t[i_init[0]], color="k", ls="--", lw=0.8)
        a.grid(True, alpha=0.3)
        if k < 3:
            a.legend(fontsize=7, loc="upper left", markerscale=4, ncol=2)
    fig.suptitle(title, fontsize=12, y=0.99)
    fig.subplots_adjust(top=0.95)
    path = os.path.join(out, "comparison.png")
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def main(argv: List[str] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("bag", help="bag with EKF odometry and corner-detection topics")
    ap.add_argument("--config", required=True, nargs="+", help="dock_fuser parameter file(s), merged in order")
    ap.add_argument("--out", required=True, help="report directory")
    ap.add_argument("--variants", nargs="+", default=["fixed_lag:30", "fixed_lag:60"],
                    help="one or more fixed_lag:<seconds>")
    ap.add_argument("--reference", default="fixed_lag:30",
                    help="variant whose full graph is solved with batch LM (the reference)")
    ap.add_argument("--max-wall", nargs="*", default=[],
                    help="per-variant wall-time budget, e.g. fixed_lag:60=900 (seconds); not for the reference")
    ap.add_argument("--start", type=float, default=None, help="window start, s from bag start")
    ap.add_argument("--end", type=float, default=None, help="window end, s from bag start")
    ap.add_argument("--replot", action="store_true",
                    help="redraw the figure and summary from the saved results in --out (no replay)")
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    meta_path = os.path.join(args.out, "meta.json")

    if args.replot:
        with open(meta_path) as f:
            meta = json.load(f)
        results = []
        for v in meta["variants"]:
            d = dict(np.load(os.path.join(args.out, f"variant_{v['name']}.npz")))
            d.update(v)
            results.append(d)
        ref_name = meta["reference"]
        ref = next(r for r in results if r["name"] == ref_name)
        ref["lm_stats"] = meta["lm_stats"]
        args.start, args.end = meta["start"], meta["end"]
    else:
        variants = [parse_variant(v) for v in args.variants]
        ref_name, ref_over = parse_variant(args.reference)
        if ref_name not in [n for n, _ in variants]:
            variants.insert(0, (ref_name, ref_over))
        budgets = {}
        for b in args.max_wall:
            spec, sec = b.rsplit("=", 1)
            budgets[parse_variant(spec)[0]] = float(sec)
        budgets.pop(ref_name, None)   # the reference must finish
        window = (args.start, args.end)
        jobs = [(args.bag, args.config, n, o, n == ref_name, budgets.get(n, float("inf")), window)
                for n, o in variants]
        # 'spawn': fresh interpreters, no state shared with the parent.
        with mp.get_context("spawn").Pool(len(jobs)) as pool:
            results = pool.map(_worker, jobs)
        for r in results:
            np.savez_compressed(os.path.join(args.out, f"variant_{r['name']}.npz"),
                                **{k: v for k, v in r.items() if isinstance(v, np.ndarray)})
        ref = next(r for r in results if r["name"] == ref_name)
        with open(meta_path, "w") as f:
            json.dump({"bag": args.bag, "config": args.config, "start": args.start, "end": args.end,
                       "reference": ref_name, "lm_stats": ref["lm_stats"],
                       "variants": [{"name": r["name"], "completed": bool(r["completed"]),
                                     "wall_s": float(r["wall_s"]),
                                     "n_recoveries": int(r["n_recoveries"])} for r in results]},
                      f, indent=1)

    rows, series = analyse(results, ref)
    from ..ros.params import build_setup, getter_from_yaml
    setup = build_setup(getter_from_yaml(args.config), lambda m: None)
    from .replay_bag import bag_source, bag_start_time, load_camera_info
    load_camera_info(args.bag, setup)
    obs = tag_observations(args.bag, setup, args.start, args.end)
    win = f"{args.start or 0:g}-{args.end if args.end is not None else 'end'} s"
    fig_path = plot(results, ref, series, setup, args.out, bag_start_time(args.bag), obs,
                    f"{bag_source(args.bag)}, window {win}: "
                    f"real-time backends vs batch LM")

    st = ref["lm_stats"]
    lines = [f"bag: {args.bag}, window {win}",
             f"keyframes: {len(ref['stamp'])} over {ref['stamp'][-1] - ref['stamp'][0]:.0f} s",
             f"batch LM reference (full graph of the {ref_name} run): {st['n_factors']} factors, "
             f"{st['iterations']} iterations, {st['seconds']:.1f} s; graph error at the real-time "
             f"estimates {st['error_init']:.1f} -> LM {st['error_lm']:.1f}",
             ""]
    keys = list(rows[0])
    w = max(len(k) for k in keys)
    lines.append(" " * w + "".join(f"{r['variant']:>18s}" for r in rows))
    for k in keys[1:]:
        cells = []
        for r in rows:
            v = r[k]
            cells.append(f"{v:18.2f}" if isinstance(v, float) else f"{v:>18}")
        lines.append(f"{k:<{w}}" + "".join(cells))
    lines.append("")
    lines.append("visible = keyframes with accepted tag factors; all = every keyframe while tracking. "
                 "Covariance off in all variants.")
    text = "\n".join(lines)
    print(text)
    with open(os.path.join(args.out, "summary.txt"), "w") as f:
        f.write(text + "\n")
    with open(os.path.join(args.out, "summary.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=keys)
        wr.writeheader()
        wr.writerows(rows)
    print(f"wrote {fig_path}, {args.out}/summary.txt, summary.csv")


if __name__ == "__main__":
    main()
