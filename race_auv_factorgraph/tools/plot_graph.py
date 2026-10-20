"""Draw the actual factor graph a replay built, over a short time window.

Usage::

    # the graph over one short window
    ros2 run race_auv_factorgraph plot_graph DETECTION_BAG --config tank_dock_fuser.yaml \\
        --start 30 --window 90.5 95.5 --label A --frame-stride 3 --out graph_init.png

    # overview: the whole run, with a labelled box around each window
    ros2 run race_auv_factorgraph plot_graph DETECTION_BAG --config tank_dock_fuser.yaml \\
        --start 30 --overview 420 --window 90.5 95.5 --window 228.5 232.5 --out overview.png

All times are seconds into the **original recording**: ``bag_test.launch.py``
writes ``source.yaml`` into its output bag (which recording, from where), so a
bag holding only part of a recording keeps that recording's time axis; the
title names the recording. The bag is replayed from ``--start`` (it must match
the run you want to look at, because the estimator starts fresh there) up to
the end of ``--window``. The factors are
taken from ``Backend.full_graph()``, i.e. exactly what the estimator inserted,
and every factor touching a keyframe inside the window is drawn. A whole run
is far too dense to read; a few seconds (~25 poses at 5 keyframes/s) shows
the structure. ``--frame-stride N`` keeps every Nth detector image, as for
``replay_bag``: 3 replays the 15 Hz tank bags at the vehicle's 5 Hz.

Two panels
----------
**Top: graph structure.** Circles are variables, squares are factors.

* ``X_k`` along the bottom row, placed at the keyframe time. Filled blue: the
  keyframe has tag factors. White: odometry only (a detector message with no
  tag in view, or none usable). A grey outline ring marks tags that were seen
  but rejected (not on the station, PnP failed).
* ``S`` and the ``T_j`` above, centred.
* Factors: odometry (blue squares between consecutive ``X_k``), tag
  projection (red, one per tag observation, with an edge to ``X_k``, ``S``
  and ``T_j``), tag URDF prior (green, on each ``T_j``), anchor prior (black,
  only if ``X_0`` is in the window).
* Projection factors inserted *after* their keyframe are drawn in orange:
  these are the buffered observations that entered all at once when the
  station was initialized (§7.3).
* The odometry factor linking the window to the keyframe before it is drawn
  as a stub on the left (the rest of the graph, not drawn). ``X_k`` older than
  the smoother lag have been marginalized into a prior inside the smoother;
  that prior is internal to gtsam and not in this copy of the graph.

**Bottom: the same graph in space** (odom x-y, metres, equal aspect). The
real-time ``X_k`` estimates (blue, as known at each keyframe), the EKF ``E_k``
(green dashed), ``S`` with its x axis, and each tag at ``S * T_j``. Every
projection factor is a thin line from the AUV pose to the tag it observed,
coloured as above.

Overview (``--overview END``)
-----------------------------
Replays ``--start .. END`` and draws the whole run so the windows can be
placed in context (``--view-start FROM`` draws only ``FROM .. END`` of it,
the replay still starting at ``--start``). Top: bird's-eye view of the EKF ``E_k`` and the real-time
``X_k`` track, keyframes with tag factors, the final ``S`` and tags, and a
labelled box (A, B, ...) around the poses of each ``--window``. Every tag
observation is also drawn where it puts the tag in odom, once through the EKF
(``E_k T_bc T_ct``, green like the EKF track) and once through the graph
(``X_k T_bc T_ct``, blue like the graph track), with ``T_ct`` the single-image PnP pose: the tag does not move, so the
EKF cloud smears out with the drift while the graph's stays tight. Bottom: the
drift correction ``||t(X_k) - t(E_k)||`` (3D distance between the graph's and
the EKF's AUV position) over time, a raster of
keyframes with tag factors, the time before TRACKING, and the same windows
shaded with their labels.
"""

from __future__ import annotations

import argparse
import dataclasses
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import gtsam
import numpy as np

from ..core.common import keys as K
from ..core.init.state_machine import StationState
from ..ros.params import build_setup, getter_from_yaml
from .replay_bag import bag_source, bag_start_time, replay, source_start_time

# Colours shared by both panels.
C_X_TAG = "#2b6cb0"      # keyframe with tag factors
C_ODOM = "#2b6cb0"
C_PROJ = "#c53030"       # projection factor added at its own keyframe
C_PROJ_INIT = "#dd6b20"  # projection factor added later (buffered, at initialization)
C_S = "#c05621"
C_T = "#2f855a"
C_ANCHOR = "#1a202c"
C_EKF = "#38a169"         # EKF E_k: green, distinct from the graph track (blue)
C_HL = "#d69e2e"         # overview: highlighted window segment
C_KF_TAG = "#e53e3e"     # overview: keyframe with tag factors, dots on the graph track (red)


@dataclass
class KeyframeInfo:
    """What the replay recorded for one keyframe (only the fields the plot needs)."""

    k: int
    t: float                       # seconds into the original recording
    X: gtsam.Pose3                 # real-time estimate at this keyframe
    E: gtsam.Pose3                 # EKF pose
    state: StationState
    gated_out: bool                # tags seen but rejected
    initialized_now: bool
    first_factor: int              # index in full_graph of this keyframe's first factor
    n_factors: int = 0             # number of factors inserted at this keyframe


@dataclass
class GraphFactor:
    """One factor of the window, classified by its keys."""

    kind: str                      # "odom", "proj", "urdf", "anchor", "prior_x"
    keys: List[int]
    inserted_at: int               # keyframe index k at which the factor entered the backend


@dataclass
class Window:
    kfs: List[KeyframeInfo]
    factors: List[GraphFactor]
    S: Optional[gtsam.Pose3]
    T: Dict[int, gtsam.Pose3]      # T_j key -> estimate (pose of tag j in dock_point)
    t_names: Dict[int, str]        # T_j key -> "36h11:146"
    stub_from: Optional[int] = None  # X key before the window linked by odometry (left stub)
    meta: dict = field(default_factory=dict)


def _chr(key: int) -> str:
    """Symbol letter of a key ('x', 's', 't'); the binding returns it as int or str."""
    c = gtsam.Symbol(key).chr()
    return chr(c) if isinstance(c, int) else c


def collect(bag: str, config, start: float, w0: float, w1: float,
            frame_stride: int = 1) -> Window:
    """Replay ``start .. w1`` and return the variables and factors of keyframes in ``[w0, w1]``.

    Which keyframe inserted which factor is reconstructed from the size of
    ``Backend.full_graph()`` after every keyframe: the backend appends each
    update's factors in order, so factors ``[size_before, size_after)``
    belong to that keyframe. This is how the projection factors added at
    initialization (for frames buffered seconds earlier) are told apart.
    """
    get = getter_from_yaml(config)
    setup = build_setup(get, lambda m: None)
    # Covariance does not affect the graph; skipping it makes the replay faster.
    cfg = dataclasses.replace(setup.cfg, covariance_mode="none")
    # Times are seconds into the original recording (source.yaml), so the
    # windows mean the same whichever slice of it the bag holds.
    t_bag0 = source_start_time(bag)
    shift = bag_start_time(bag) - t_bag0          # bag start, in recording time
    kfs: List[KeyframeInfo] = []
    size = [0]

    def on_kf(est, r, held) -> None:
        n = est.backend.full_graph().size()
        kfs.append(KeyframeInfo(k=r.k, t=r.stamp - t_bag0, X=r.X, E=r.E, state=r.state,
                                gated_out=bool(r.rejected) and not r.accepted,
                                initialized_now=r.initialized_now,
                                first_factor=size[0], n_factors=n - size[0]))
        size[0] = n

    est, _ = replay(bag, get, setup, cfg=cfg, on_keyframe=on_kf,
                    start=None if start is None else max(0.0, start - shift), end=w1 - shift,
                    frame_stride=frame_stride)
    if est.n_recoveries:
        print(f"warning: {est.n_recoveries} backend recoveries; only the last backend's "
              f"factors are available")

    # Factor index -> keyframe that inserted it.
    owner = np.empty(size[0], dtype=int)
    for kf in kfs:
        owner[kf.first_factor:kf.first_factor + kf.n_factors] = kf.k

    in_win = [kf for kf in kfs if w0 <= kf.t <= w1]
    if not in_win:
        raise SystemExit(f"no keyframes in {w0}-{w1} s")
    win_x = {K.X(kf.k) for kf in in_win}

    graph = est.backend.full_graph()
    factors: List[GraphFactor] = []
    stub_from = None
    t_keys = set()
    for i in range(graph.size()):
        f = graph.at(i)
        if f is None:
            continue
        keys = list(f.keys())
        # Keep a factor if it touches a pose in the window, or it is the URDF
        # prior of a tag (drawn once per T_j).
        touches = [k for k in keys if k in win_x]
        chrs = [_chr(k) for k in keys]
        if len(keys) == 3:                                  # X_k, S, T_j
            kind = "proj"
        elif len(keys) == 2:                                # X_{k-1}, X_k
            kind = "odom"
        elif chrs == ["t"]:
            kind = "urdf"
        elif chrs == ["x"]:
            kind = "anchor" if gtsam.Symbol(keys[0]).index() == 0 else "prior_x"
        else:
            kind = "other"
        if kind == "urdf":
            factors.append(GraphFactor(kind, keys, int(owner[i])))
            continue
        if not touches:
            continue
        if kind == "odom" and len(touches) == 1:
            # The odometry factor from the last keyframe before the window.
            stub_from = next(k for k in keys if k not in win_x)
            continue
        if kind == "proj":
            t_keys.add(keys[2])
        factors.append(GraphFactor(kind, keys, int(owner[i])))

    # Only the URDF priors of tags actually observed in the window.
    factors = [f for f in factors if f.kind != "urdf" or f.keys[0] in t_keys]
    names = {est.tag_index.key(tag): f"{tag[0]}:{tag[1]}" for tag in est.tag_index.tags()}
    T = {k: est.backend.pose(k) for k in t_keys if est.backend.pose(k) is not None}
    return Window(kfs=in_win, factors=factors, S=est.backend.pose(K.S), T=T,
                  t_names={k: names.get(k, K.describe(k)) for k in t_keys},
                  stub_from=stub_from,
                  meta={"lag": cfg.smoother_lag, "bag": os.path.basename(os.path.normpath(bag))})


def plot(win: Window, png: str, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    kf_by_key = {K.X(kf.k): kf for kf in win.kfs}
    t_of = {key: kf.t for key, kf in kf_by_key.items()}
    t_lo, t_hi = min(t_of.values()), max(t_of.values())
    span = max(t_hi - t_lo, 1e-3)
    t_mid = 0.5 * (t_lo + t_hi)
    proj = [f for f in win.factors if f.kind == "proj"]
    has_tag = {f.keys[0] for f in proj}

    fig, (ax, bx) = plt.subplots(2, 1, figsize=(15, 12), gridspec_kw={"height_ratios": [1.0, 1.15]})

    # ------------------------------------------------------------ structure
    # Rows: X at y=0, projection factors at y=1, S at y=2, T_j at y=3 (+ one row per extra tag).
    Y_X, Y_P, Y_S, Y_T = 0.0, 1.0, 2.0, 3.0
    t_rows = {key: Y_T + 0.6 * i for i, key in enumerate(sorted(win.T))}
    # S and T_j are drawn centred; with several tags spread them a little.
    S_pos = (t_mid, Y_S)
    T_pos = {key: (t_mid + (i - (len(t_rows) - 1) / 2) * 0.12 * span, y)
             for i, (key, y) in enumerate(sorted(t_rows.items()))}

    def square(x, y, c, s=26, z=4):
        ax.scatter([x], [y], marker="s", s=s, c=c, zorder=z, linewidths=0)

    # Edges first (drawn under the nodes).
    for f in proj:
        x_key, _, t_key = f.keys
        tx = t_of[x_key]
        late = f.inserted_at > kf_by_key[x_key].k
        c = C_PROJ_INIT if late else C_PROJ
        ax.plot([tx, tx], [Y_X, Y_P], c=c, lw=0.7, alpha=0.8, zorder=1)
        ax.plot([tx, S_pos[0]], [Y_P, S_pos[1]], c=C_S, lw=0.5, alpha=0.35, zorder=1)
        if t_key in T_pos:
            ax.plot([tx, T_pos[t_key][0]], [Y_P, T_pos[t_key][1]], c=C_T, lw=0.5, alpha=0.25, zorder=1)
        square(tx, Y_P, c)
    xs = sorted(t_of.values())
    ax.plot(xs, [Y_X] * len(xs), c=C_ODOM, lw=0.8, alpha=0.6, zorder=1)
    for f in win.factors:
        if f.kind == "odom":
            a, b = (t_of[k] for k in f.keys)
            square(0.5 * (a + b), Y_X, C_ODOM, s=10)
        elif f.kind in ("anchor", "prior_x"):
            x = t_of[f.keys[0]]
            ax.plot([x, x], [Y_X, Y_X - 0.35], c=C_ANCHOR, lw=0.8)
            square(x, Y_X - 0.35, C_ANCHOR)
        elif f.kind == "urdf" and f.keys[0] in T_pos:
            x, y = T_pos[f.keys[0]]
            ax.plot([x, x + 0.04 * span], [y, y], c=C_T, lw=0.8)
            square(x + 0.04 * span, y, C_T)
    if win.stub_from is not None:
        x0 = t_lo - 0.035 * span
        ax.plot([x0, t_lo], [Y_X, Y_X], c=C_ODOM, lw=0.8, ls=":")
        ax.text(x0, Y_X + 0.12, "← earlier graph\n(marginalized after lag)", fontsize=8,
                ha="left", va="bottom", color="#4a5568")

    # Variables.
    for key, kf in kf_by_key.items():
        tag = key in has_tag
        ax.scatter([kf.t], [Y_X], s=70, zorder=5, c=C_X_TAG if tag else "white",
                   edgecolors=C_X_TAG, linewidths=1.0)
        if kf.gated_out:
            ax.scatter([kf.t], [Y_X], s=150, zorder=4, facecolors="none", edgecolors="#718096",
                       linewidths=0.8)
        if kf.initialized_now:
            ax.annotate("station initialized", (kf.t, Y_X), (kf.t, Y_X - 0.55), fontsize=9,
                        ha="center", arrowprops=dict(arrowstyle="->", lw=0.8))
    if win.S is not None and proj:
        ax.scatter(*S_pos, s=600, c="#fde2c8", edgecolors=C_S, linewidths=1.6, zorder=5)
        ax.text(*S_pos, "S", ha="center", va="center", fontsize=12, zorder=6)
    for key, (x, y) in T_pos.items():
        ax.scatter([x], [y], s=600, c="#d5f0dc", edgecolors=C_T, linewidths=1.6, zorder=5)
        ax.text(x, y, "T", ha="center", va="center", fontsize=12, zorder=6)
        # Label to the right of the URDF prior square, clear of the legend.
        ax.text(x + 0.055 * span, y, win.t_names[key], ha="left", va="center", fontsize=9,
                color=C_T)

    # States as a shaded band under the X row.
    for kf_a, kf_b in zip(win.kfs, win.kfs[1:]):
        if kf_a.state != StationState.TRACKING:
            ax.axvspan(kf_a.t, kf_b.t, ymin=0, ymax=0.035, color="#fbd38d", lw=0)
    n_late = sum(1 for f in proj if f.inserted_at > kf_by_key[f.keys[0]].k)
    ax.set_yticks([Y_X, Y_P, Y_S] + list(t_rows.values()))
    ax.set_yticklabels(["poses X_k", "tag factors", "station S"] + ["tag T_j"] * len(t_rows))
    ax.set_ylim(Y_X - 1.0, max(t_rows.values(), default=Y_T) + 1.0)
    ax.set_xlim(t_lo - 0.05 * span, t_hi + 0.03 * span)
    ax.set_xlabel("time since recording start (s)")
    ax.set_title(f"{len(kf_by_key)} poses, {len(proj)} tag projection factors "
                 f"({n_late} inserted at initialization), "
                 f"{sum(f.kind == 'odom' for f in win.factors)} odometry factors", fontsize=10)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    legend = [
        Line2D([], [], marker="o", ls="", mfc=C_X_TAG, mec=C_X_TAG, label="X_k with tag factors"),
        Line2D([], [], marker="o", ls="", mfc="white", mec=C_X_TAG, label="X_k, odometry only"),
        Line2D([], [], marker="o", ls="", mfc="none", mec="#718096", ms=10,
               label="tag seen, rejected"),
        Line2D([], [], marker="s", ls="", c=C_ODOM, ms=4, label="odometry factor"),
        Line2D([], [], marker="s", ls="", c=C_PROJ, label="tag projection factor"),
        Line2D([], [], marker="s", ls="", c=C_PROJ_INIT, label="… inserted at initialization"),
        Line2D([], [], marker="s", ls="", c=C_T, label="tag URDF prior"),
        Line2D([], [], lw=6, c="#fbd38d", label="not yet TRACKING"),
    ]
    ax.legend(handles=legend, loc="upper left", fontsize=8, ncol=4, frameon=False,
              bbox_to_anchor=(0.0, 1.0))

    # ------------------------------------------------------------ space
    X = np.array([kf.X.translation() for kf in win.kfs])
    E = np.array([kf.E.translation() for kf in win.kfs])
    bx.plot(E[:, 0], E[:, 1], ls="--", c=C_EKF, lw=1.2, label="EKF E_k")
    bx.plot(X[:, 0], X[:, 1], c=C_X_TAG, lw=1.0, alpha=0.6)
    tagged = np.array([key in has_tag for key in kf_by_key])
    bx.scatter(X[tagged, 0], X[tagged, 1], s=18, c=C_X_TAG, zorder=4, label="X_k with tag factors")
    bx.scatter(X[~tagged, 0], X[~tagged, 1], s=18, c="white", edgecolors=C_X_TAG, zorder=4,
               label="X_k, odometry only")
    if win.S is not None:
        for f in proj:
            x_key, _, t_key = f.keys
            T = win.T.get(t_key)
            if T is None:
                continue
            p = kf_by_key[x_key].X.translation()
            q = win.S.compose(T).translation()
            late = f.inserted_at > kf_by_key[x_key].k
            bx.plot([p[0], q[0]], [p[1], q[1]], c=C_PROJ_INIT if late else C_PROJ, lw=0.5,
                    alpha=0.35, zorder=2)
        s = win.S.translation()
        ax_dir = win.S.rotation().matrix()[:, 0]
        bx.scatter([s[0]], [s[1]], s=120, c="#fde2c8", edgecolors=C_S, linewidths=1.5, zorder=5,
                   label="S (dock_point)")
        bx.annotate("", (s[0] + 0.3 * ax_dir[0], s[1] + 0.3 * ax_dir[1]), (s[0], s[1]),
                    arrowprops=dict(arrowstyle="->", color=C_S, lw=1.5))
        for key, T in win.T.items():
            q = win.S.compose(T).translation()
            bx.scatter([q[0]], [q[1]], marker="s", s=60, c="#d5f0dc", edgecolors=C_T, zorder=5)
            bx.text(q[0], q[1], "  " + win.t_names[key], fontsize=8, color=C_T, va="center")
    bx.plot([], [], c=C_PROJ, lw=1, label="tag projection factor (AUV → tag)")
    if any(f.inserted_at > kf_by_key[f.keys[0]].k for f in proj):
        bx.plot([], [], c=C_PROJ_INIT, lw=1, label="… inserted at initialization")
    bx.set_aspect("equal", adjustable="datalim")
    bx.set_xlabel("odom x (m)")
    bx.set_ylabel("odom y (m)")
    bx.grid(alpha=0.3)
    bx.legend(fontsize=8, loc="best")
    bx.set_title("Same window in space (real-time estimates; S and T_j at the window end)",
                 fontsize=10)

    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(png, dpi=150, bbox_inches="tight")
    print(f"wrote {png}")


def overview(bag: str, config, start: Optional[float], end: float,
             windows: List[Tuple[float, float]], png: str, title: str,
             frame_stride: int = 1, view_start: Optional[float] = None,
             timing: Optional[str] = None) -> None:
    """Whole-run BEV and time strip with a labelled box per window (see module docstring).

    ``view_start`` crops what is drawn to ``view_start .. end`` while the
    estimator still runs from ``start``, so the drawn estimates are the same
    as in a full overview (and in the window plots), just a shorter stretch.

    ``timing`` (a label for the machine that ran the node, e.g. "Jetson Orin
    Nano") adds a third panel: the per-keyframe update time of this replay,
    on this computer, against the ``update_ms`` the live node recorded in the
    bag (``/dock_fuser/keyframe``, ``publish_debug: true``). Both process the
    same detections keyframe by keyframe. The replay then keeps the config's
    covariance mode, because the node computes the covariance inside the timed
    update too.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Rectangle

    get = getter_from_yaml(config)
    setup = build_setup(get, lambda m: None)
    # Covariance does not change the estimates; skipping it is faster, unless
    # the update time itself is being compared.
    cfg = setup.cfg if timing else dataclasses.replace(setup.cfg, covariance_mode="none")
    t_bag0 = source_start_time(bag)               # seconds into the original recording
    shift = bag_start_time(bag) - t_bag0
    rec = []   # (t, X position, E position, has tag factors, tracking, seeded, update ms)
    # Where each tag observation puts the tag in odom: the single-image PnP
    # pose T_ct chained onto the EKF pose (E_k T_bc T_ct) and onto the graph's
    # real-time pose (X_k T_bc T_ct). The tag is fixed, so the spread of each
    # cloud shows how consistent that AUV track is with the camera.
    tag_E, tag_X, tag_t = [], [], []

    def on_kf(est, r, held) -> None:
        rec.append((r.stamp - t_bag0, r.X.translation(), r.E.translation(), bool(r.accepted),
                    r.state == StationState.TRACKING, r.seeded_from_tag, r.update_time * 1000))
        for _, cam, T_ct in r.tag_obs:
            T_bt = est.cameras[cam].T_bc.compose(T_ct)
            tag_E.append(r.E.compose(T_bt).translation())
            tag_X.append(r.X.compose(T_bt).translation())
            tag_t.append(r.stamp - t_bag0)

    est, _ = replay(bag, get, setup, cfg=cfg, on_keyframe=on_kf,
                    start=None if start is None else max(0.0, start - shift), end=end - shift,
                    frame_stride=frame_stride)
    if view_start is not None:
        rec = [r for r in rec if r[0] >= view_start]
        keep = [i for i, ti in enumerate(tag_t) if ti >= view_start]
        tag_E, tag_X = [tag_E[i] for i in keep], [tag_X[i] for i in keep]
    t = np.array([r[0] for r in rec])
    X = np.array([r[1] for r in rec])
    E = np.array([r[2] for r in rec])
    tag = np.array([r[3] for r in rec])
    trk = np.array([r[4] for r in rec])
    seeded = np.array([r[5] for r in rec])
    upd_pc = np.array([r[6] for r in rec])
    labels = [chr(ord("A") + i) for i in range(len(windows))]

    if timing:
        fig, (ax, bx, cx) = plt.subplots(3, 1, figsize=(13, 17), sharex=False,
                                         gridspec_kw={"height_ratios": [3, 1, 1]})
    else:
        fig, (ax, bx) = plt.subplots(2, 1, figsize=(13, 13), gridspec_kw={"height_ratios": [3, 1]})

    # ------------------------------------------------------------ BEV
    ax.plot(E[:, 0], E[:, 1], ls="--", c=C_EKF, lw=1.1, label="EKF E_k")
    # The graph track is one solid line through every keyframe; the dots on
    # it mark keyframes with tag factors (red, small, so the line stays readable).
    ax.plot(X[:, 0], X[:, 1], c=C_X_TAG, lw=1.3, label="graph X_k (real time)")
    ax.scatter(X[tag, 0], X[tag, 1], s=3, c=C_KF_TAG, linewidths=0, zorder=3)
    ax.scatter([], [], s=20, c=C_KF_TAG, linewidths=0, label="keyframe with tag factors")
    # Keyframes whose optimizer starting guess for X_k came from the tag
    # instead of the odometry chain (DockEstimator._seed_from_tags): the tag
    # returned after a gap and the drifted odometry guess was far off. Nothing
    # is reset; these are the visible steps in the real-time track, where the
    # drift of the gap is corrected at once.
    if seeded.any():
        ax.scatter(X[seeded, 0], X[seeded, 1], s=45, marker="D", facecolors="none",
                   edgecolors="magenta", linewidths=1.2, zorder=6,
                   label=f"tag back after a gap: X_k initial guess from the tag, "
                         f"not the odometry ({seeded.sum()})")
    # Where the two tracks start and end (they coincide at the start: X_0 = E_0).
    ax.scatter([X[0, 0]], [X[0, 1]], s=60, marker="o", c="white", edgecolors="black", zorder=7,
               label=f"start, {t[0]:.0f} s")
    for P, c, name in ((X, C_X_TAG, "graph"), (E, C_EKF, "EKF")):
        ax.scatter([P[-1, 0]], [P[-1, 1]], s=80, marker="X", c=c, edgecolors="black",
                   linewidths=0.8, zorder=7)
        # Graph label to the left (S and the tags sit to the right of the
        # end of a docking run), EKF label below right.
        left = name == "graph"
        ax.annotate(f"{name} end, {t[-1]:.0f} s", (P[-1, 0], P[-1, 1]),
                    xytext=(-8, 6) if left else (8, -12), ha="right" if left else "left",
                    textcoords="offset points", fontsize=9, color=c, fontweight="bold",
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=1.0))
    if tag_E:
        qE, qX = np.array(tag_E), np.array(tag_X)

        def spread(q: np.ndarray) -> Tuple[np.ndarray, float, float]:
            """Centre, RMS and max horizontal distance (m) of a cloud around its own mean.

            The tag does not move, so ideally every observation lands on one
            point. The centre is the cloud's own mean (not the graph's tag
            estimate), so both clouds are measured the same way: consistency
            only. Horizontal (x-y), to match the bird's-eye view the circles
            are drawn in (the 3D numbers differ by < 1 mm here).
            """
            c = q[:, :2].mean(axis=0)
            r = np.linalg.norm(q[:, :2] - c, axis=1)
            return c, float(np.sqrt(np.mean(r ** 2))), float(r.max())

        # Graph cloud on top of the (hollow) final tag marker: it is tight
        # enough to hide under it. Empty scatters give readable legend markers.
        ax.scatter(qE[:, 0], qE[:, 1], s=5, c=C_EKF, alpha=0.35, linewidths=0, zorder=2)
        ax.scatter(qX[:, 0], qX[:, 1], s=5, c=C_X_TAG, alpha=0.6, linewidths=0, zorder=6)
        for q, col, name, expr in ((qE, C_EKF, "EKF", "E_k"), (qX, C_X_TAG, "graph", "X_k")):
            c, rms, rmax = spread(q)
            # Filled circle: RMS radius (~63 % of a Gaussian cloud inside),
            # i.e. exactly the number in the legend. Dashed ring: the max,
            # which a single outlier sets, so it is only an outline.
            ax.add_patch(Circle(c, rms, facecolor=col, edgecolor=col, alpha=0.18, lw=1.2, zorder=4))
            ax.add_patch(Circle(c, rmax, fill=False, edgecolor=col, alpha=0.7, lw=1.0, ls="--",
                                zorder=4))
            ax.scatter([], [], s=25, c=col, alpha=0.7, linewidths=0,
                       label=f"tag via {name}, {expr} T_bc T_ct: spread RMS {rms * 100:.1f} cm "
                             f"(circle), max {rmax * 100:.1f} cm (dashed)")
            print(f"tag via {name}: {len(q)} obs, horizontal spread RMS {rms * 100:.1f} cm, "
                  f"max {rmax * 100:.1f} cm")
    S = est.backend.pose(K.S)
    if S is not None:
        s = S.translation()
        d = S.rotation().matrix()[:, 0]
        ax.scatter([s[0]], [s[1]], s=140, c="#fde2c8", edgecolors=C_S, linewidths=1.5, zorder=5,
                   label="S (dock_point, final)")
        ax.annotate("", (s[0] + 0.4 * d[0], s[1] + 0.4 * d[1]), (s[0], s[1]),
                    arrowprops=dict(arrowstyle="->", color=C_S, lw=1.5))
        for i, tg in enumerate(est.tag_index.tags()):
            T = est.backend.pose(est.tag_index.key(tg))
            if T is not None:
                q = S.compose(T).translation()
                # Black, not the tag green: the EKF cloud around it is green.
                ax.scatter([q[0]], [q[1]], marker="s", s=70, facecolors="none", edgecolors="black",
                           linewidths=1.4, zorder=5,
                           label="tag S T_j (graph, final)" if i == 0 else None)
                ax.text(q[0], q[1], f"  {tg[0]}:{tg[1]}", fontsize=9, color="black", va="center")
    for lab, (w0, w1) in zip(labels, windows):
        m = (t >= w0) & (t <= w1)
        if not m.any():
            print(f"warning: no keyframes in window {lab} ({w0}-{w1} s)")
            continue
        pts = np.vstack([X[m, :2], E[m, :2]])   # box covers both tracks in the window
        lo, hi = pts.min(axis=0), pts.max(axis=0)
        pad = 0.12
        size = np.maximum(hi - lo + 2 * pad, 0.35)   # never smaller than 35 cm, stays visible
        c = 0.5 * (lo + hi)
        ax.add_patch(Rectangle(c - size / 2, size[0], size[1], fill=False, ec="#1a202c", lw=1.4,
                               zorder=6))
        # The track can pass the same place several times: highlight this
        # window's own segment (graph solid, EKF dashed) so the box is unambiguous.
        ax.plot(X[m, 0], X[m, 1], c=C_HL, lw=3.0, zorder=7, solid_capstyle="round")
        ax.plot(E[m, 0], E[m, 1], c=C_HL, lw=2.0, ls="--", zorder=7)
        # Label above the box, or below it when the station marker is close
        # above (the label's white background would hide S).
        above = S is None or not (abs(S.translation()[0] - c[0]) < 1.0
                                  and 0 < S.translation()[1] - c[1] < 0.6)
        ly = c[1] + size[1] / 2 + 0.04 if above else c[1] - size[1] / 2 - 0.04
        ax.text(c[0] - size[0] / 2, ly, f"{lab}: {w0:g}–{w1:g} s",
                fontsize=10, fontweight="bold", va="bottom" if above else "top", zorder=8,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.5))
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("odom x (m)")
    ax.set_ylabel("odom y (m)")
    ax.grid(alpha=0.3)
    ax.plot([], [], c=C_HL, lw=3.0, label="window segment (graph solid, EKF dashed)")
    ax.legend(fontsize=9, loc="best")

    # ------------------------------------------------------------ time strip
    # Drift correction: 3D distance between the graph's and the EKF's AUV
    # position at each keyframe (independent of where the odom origin is).
    D = np.linalg.norm(X - E, axis=1)
    bx.plot(t, D, c=C_X_TAG, lw=1.0,
            label="drift correction ‖t(X_k) − t(E_k)‖, 3D (m)")
    ymax = max(float(D.max()) * 1.15, 0.05)
    bx.vlines(t[tag], ymax * 0.93, ymax, color=C_KF_TAG, lw=0.4, label="keyframe with tag factors")
    pre = ~trk
    if pre.any():
        bx.axvspan(t[0], t[pre][-1], color="#fbd38d", alpha=0.5, lw=0, label="not yet TRACKING")
    for lab, (w0, w1) in zip(labels, windows):
        bx.axvspan(w0, w1, color="#1a202c", alpha=0.18, lw=0)
        bx.text(0.5 * (w0 + w1), ymax * 0.82, lab, ha="center", fontsize=10, fontweight="bold")
    bx.set_ylim(0, ymax)
    bx.set_xlim(t[0], t[-1])
    bx.set_xlabel("time since recording start (s)")
    bx.set_ylabel("drift correction, 3D (m)")
    bx.legend(fontsize=8, loc="lower left", ncol=3, frameon=False, bbox_to_anchor=(0.0, 1.0))
    bx.grid(alpha=0.3)

    if timing:
        _timing_panel(cx, bag, t_bag0, t, upd_pc, timing, frame_stride)

    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(png, dpi=150, bbox_inches="tight")
    print(f"wrote {png}")


def _live_update_times(bag: str) -> Dict[float, float]:
    """``update_ms`` per keyframe stamp from the live node's ``/dock_fuser/keyframe`` (JSON)."""
    import json

    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from std_msgs.msg import String
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=bag), rosbag2_py.ConverterOptions("", ""))
    reader.set_filter(rosbag2_py.StorageFilter(topics=["/dock_fuser/keyframe"]))
    out: Dict[float, float] = {}
    while reader.has_next():
        _, data, _ = reader.read_next()
        j = json.loads(deserialize_message(data, String).data)
        out[round(j["stamp"], 3)] = float(j["update_ms"])
    return out


def _timing_panel(cx, bag: str, t_bag0: float, t: np.ndarray, upd_pc: np.ndarray,
                  live_label: str, frame_stride: int) -> None:
    """Per-keyframe update time: this replay (this computer) vs the live node (from the bag).

    Log scale: odometry-only keyframes take ~1 ms on a desktop, tag
    keyframes and marginalization tens of ms on the Jetson. The keyframe
    period (1 / detector rate) is the budget: a longer update makes frames
    queue up (``frames_waiting``) and eventually drop.
    """
    import platform
    live = _live_update_times(bag)
    if not live:
        raise SystemExit(f"--timing: no /dock_fuser/keyframe in {bag} (record the live node "
                         f"with publish_debug: true)")
    t_live = np.array(sorted(live)) - t_bag0
    u_live = np.array([live[k] for k in sorted(live)])
    m = (t_live >= t[0]) & (t_live <= t[-1])
    t_live, u_live = t_live[m], u_live[m]
    cpu = platform.processor() or platform.machine()
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as f:
            cpu = next(ln.split(":", 1)[1].strip() for ln in f if ln.startswith("model name"))
    except (OSError, StopIteration):
        pass

    def stats(u: np.ndarray) -> str:
        return (f"median {np.median(u):.1f}, p95 {np.percentile(u, 95):.1f}, "
                f"max {u.max():.1f} ms")
    cx.plot(t_live, u_live, c="#dd6b20", lw=0.6, alpha=0.9,
            label=f"{live_label}, live node ({stats(u_live)})")
    cx.plot(t, upd_pc, c="#2b6cb0", lw=0.6, alpha=0.9,
            label=f"PC replay, {cpu} ({stats(upd_pc)})")
    dt = np.median(np.diff(t_live)) if len(t_live) > 1 else 0.2
    budget = 1000.0 * dt
    cx.axhline(budget, c="#c53030", lw=1.0, ls="--",
               label=f"keyframe period {budget:.0f} ms ({1000 / budget:.0f} Hz detector)")
    cx.set_yscale("log")
    lo = max(1e-3, 0.7 * float(min(u_live.min(), upd_pc.min())))   # nothing below the axis
    cx.set_ylim(lo, budget * 3)
    cx.set_xlim(t[0], t[-1])
    cx.set_xlabel("time since recording start (s)")
    cx.set_ylabel("dock_fuser update per keyframe (ms)")
    cx.legend(fontsize=8, loc="lower left", ncol=1, frameon=False, bbox_to_anchor=(0.0, 1.0))
    cx.grid(alpha=0.3, which="both")
    print(f"update time: live {stats(u_live)} ({len(u_live)} keyframes); "
          f"PC replay {stats(upd_pc)} ({len(upd_pc)} keyframes)")


def main(argv: List[str] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("bag", help="rosbag2 directory with EKF odometry and tag detections")
    ap.add_argument("--config", required=True, nargs="+", help="dock_fuser parameter file(s), merged in order")
    ap.add_argument("--start", type=float, default=None,
                    help="replay start, s into the original recording (the estimator starts fresh here)")
    ap.add_argument("--window", type=float, nargs=2, action="append", required=True,
                    metavar=("FROM", "TO"),
                    help="time window, s into the original recording (a few seconds). Graph mode: exactly "
                         "one. Overview: repeat to box several")
    ap.add_argument("--label", default="", help="graph mode: label in the title, e.g. A")
    ap.add_argument("--overview", type=float, default=None, metavar="END",
                    help="draw the whole run up to END (s into the original recording) with a box per window")
    ap.add_argument("--view-start", type=float, default=None, metavar="FROM",
                    help="overview: draw only FROM..END (s into the recording); the replay still "
                         "starts at --start, so the estimates are those of the full run")
    ap.add_argument("--timing", default=None, metavar="LABEL",
                    help="overview: add a panel with the per-keyframe update time of this replay "
                         "(this computer) against the live node's, recorded in the bag "
                         "(/dock_fuser/keyframe); LABEL names the machine that ran the node")
    ap.add_argument("--frame-stride", type=int, default=1,
                    help="keep every Nth detector image, e.g. 3 to replay a 15 Hz bag at 5 Hz")
    ap.add_argument("--out", default="graph.png", help="output image")
    args = ap.parse_args(argv)
    name = bag_source(args.bag)        # the original recording, if bag_test noted it
    # "replay from" only when the replay starts later than the bag itself
    # (a bag that holds part of a recording already says where it starts).
    bag_begins = bag_start_time(args.bag) - source_start_time(args.bag)
    since = (f" (replay from {args.start:g} s)"
             if args.start is not None and args.start > bag_begins + 1.0 else "")
    rate = f", frame stride {args.frame_stride}" if args.frame_stride > 1 else ""
    # Which estimate is drawn: the real-time X_k of the fixed-lag smoother
    # with this config's lag (as the node computes it), not a re-smoothing.
    lag = float(getter_from_yaml(args.config)("estimator.smoother_lag"))
    backend = f"\nfixed-lag smoother {lag:g} s, real-time estimates"
    if args.overview is not None:
        t0 = (args.view_start if args.view_start is not None
              else args.start if args.start is not None else 0.0)
        overview(args.bag, args.config, args.start, args.overview,
                 [tuple(w) for w in args.window], args.out,
                 f"{name}, {t0:g}–{args.overview:g} s{rate}: "
                 f"{'part of the run' if args.view_start is not None else 'whole run'}, "
                 f"with the factor-graph windows{backend}",
                 frame_stride=args.frame_stride, view_start=args.view_start,
                 timing=args.timing)
        return
    if len(args.window) != 1:
        ap.error("graph mode takes exactly one --window (use --overview to box several)")
    w0, w1 = args.window[0]
    win = collect(args.bag, args.config, args.start, w0, w1, args.frame_stride)
    prefix = f"{args.label}: " if args.label else ""
    plot(win, args.out, f"{prefix}Factor graph, {name}, {w0:g}–{w1:g} s{since}{rate}{backend}")


if __name__ == "__main__":
    main()
