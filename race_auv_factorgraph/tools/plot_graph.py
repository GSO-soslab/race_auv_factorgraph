"""Draw the actual factor graph a replay built, over a short time window.

Usage::

    # the graph over one short window
    ros2 run race_auv_factorgraph plot_graph DETECTION_BAG --config tank_factorgraph.yaml \\
        --start 30 --window 90.5 95.5 --label A --frame-stride 3 --out graph_init.png

    # overview: the whole run, with a labelled box around each window
    ros2 run race_auv_factorgraph plot_graph DETECTION_BAG --config tank_factorgraph.yaml \\
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
EKF cloud smears out with the drift while the graph's stays tight. At the end
of each track a triangle marks where its last keyframe leaves each camera
(``end pose * T_bc``), offset from the base_link cross by the lever arm. Bottom: the
drift correction ``||t(X_k) - t(E_k)||`` (3D distance between the graph's and
the EKF's AUV position) over time, a raster of
keyframes with tag factors, the time before TRACKING, and the same windows
shaded with their labels. The graph track carries a time mark: a tick
perpendicular to the track with the recording time written beside it, every
30 s (``--time-every``), so a track that crosses the same place twice can be read chronologically.

``--batch`` adds the batch post-processing solution (``tools/batch_lm.py``):
after the replay, the *same* graph is re-solved once with Levenberg-Marquardt
from the real-time estimates, so every ``X_k`` also profits from the tag
observations made after it (full smoothing, no fixed-lag marginalization).
Drawn as a purple dash-dot track and a second drift-correction curve, plus a
tag cloud chained through the smoothed poses; the summary (graph error, real-
time vs LM shift, S shift) is printed.
"""

from __future__ import annotations

import argparse
import dataclasses
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import gtsam
import numpy as np

from ..core.common import keys as K
from ..core.common.geometry import pose_distance
from ..core.common.model import tag_key_str
from ..core.init.state_machine import StationState
from ..ros.params import build_setup, getter_from_yaml
from .bag import INIT_JUMP_M, bag_source, bag_start_time, live_topic, open_reader, source_start_time
from .plotting import cloud_spread, cpu_name, direction_marks, time_labels, timing_stats, window_box
from .replay_bag import replay

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
C_LM = "#6b46c1"         # overview --batch: batch LM track (purple, dash-dot)
FAINT = 0.15             # overview --focus-from: alpha of everything before the focus


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
                    start=start, end=w1,
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
    names = {est.tag_index.key(tag): tag_key_str(tag) for tag in est.tag_index.tags()}
    T = {k: est.backend.pose(k) for k in t_keys if est.backend.pose(k) is not None}
    return Window(kfs=in_win, factors=factors, S=est.backend.pose(K.S), T=T,
                  t_names={k: names.get(k, K.describe(k)) for k in t_keys},
                  stub_from=stub_from)


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
    # The graph frame (odom_graph), called map here: see overview().
    bx.set_xlabel("map x (m)")
    bx.set_ylabel("map y (m)")
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
             timing: Optional[str] = None, batch: bool = False,
             focus_from: Optional[float] = None, direction: bool = False,
             time_every: float = 30.0, zoom_graph: Optional[float] = None) -> None:
    """Whole-run BEV and time strip with a labelled box per window (see module docstring).

    ``view_start`` crops what is drawn to ``view_start .. end`` while the
    estimator still runs from ``start``, so the drawn estimates are the same
    as in a full overview (and in the window plots), just a shorter stretch.

    ``focus_from`` keeps everything drawn but fades what comes before it
    (alpha ``FAINT``): the earlier route stays as context, the eye goes to
    ``focus_from .. end``. The tag clouds and their RMS/max circles, the time
    marks and the start marker then cover the focus stretch only, so the
    numbers in the legend describe what is in focus. Unlike ``view_start``,
    nothing is cropped from the axes.

    ``zoom_graph`` (m) fits the BEV to the graph's own content (``X_k``, the
    batch track, ``S`` and the tags) plus that margin, instead of to
    everything drawn. For runs where the EKF runs away (a long DVL dropout):
    its track and tag clouds then leave the frame instead of shrinking the
    graph track to a dot.

    ``time_every`` (s) spaces the time marks on the graph track. ``direction``
    puts a triangle at each time mark on the graph and the EKF track
    (same keyframe on both), pointing the way the AUV travels there
    (:func:`plotting.direction_marks`); before ``focus_from`` they are faint
    and unlabelled.

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
    from matplotlib.patches import Circle

    get = getter_from_yaml(config)
    setup = build_setup(get, lambda m: None)
    # Covariance does not change the estimates; skipping it is faster, unless
    # the update time itself is being compared.
    cfg = setup.cfg if timing else dataclasses.replace(setup.cfg, covariance_mode="none")
    t_bag0 = source_start_time(bag)               # seconds into the original recording
    rec = []   # (t, X position, E position, has tag factors, tracking, init jump, update ms, k)
    # Where each tag observation puts the tag in odom: the single-image PnP
    # pose T_ct chained onto the EKF pose (E_k T_bc T_ct) and onto the graph's
    # real-time pose (X_k T_bc T_ct). The tag is fixed, so the spread of each
    # cloud shows how consistent that AUV track is with the camera. One cloud
    # per tag: several tags sit at different places.
    tag_E, tag_X, tag_t, tag_id = [], [], [], []
    tag_k, tag_bt = [], []     # per observation: its keyframe, T_bc T_ct (--batch)
    kf_X = {}                  # k -> real-time X_k pose (--batch LM initial guess)
    kf_E = {}                  # k -> EKF pose E_k (end camera marker)

    def on_kf(est, r, held) -> None:
        rec.append((r.stamp - t_bag0, r.X.translation(), r.E.translation(), bool(r.accepted),
                    r.state == StationState.TRACKING, r.init_shift >= INIT_JUMP_M,
                    r.update_time * 1000, r.k))
        kf_X[r.k] = r.X
        kf_E[r.k] = r.E
        for tag, cam, T_ct in r.tag_obs:
            T_bt = est.cameras[cam].T_bc.compose(T_ct)
            tag_E.append(r.E.compose(T_bt).translation())
            tag_X.append(r.X.compose(T_bt).translation())
            tag_t.append(r.stamp - t_bag0)
            tag_id.append(tag)
            tag_k.append(r.k)
            tag_bt.append(T_bt)

    est, _ = replay(bag, get, setup, cfg=cfg, on_keyframe=on_kf,
                    start=start, end=end,
                    frame_stride=frame_stride)
    # The anchor (X_0 = E_0) is the first replayed keyframe: there the EKF's
    # odom and the graph frame coincide (axis labels below).
    t_anchor = rec[0][0]
    if view_start is not None:
        rec = [r for r in rec if r[0] >= view_start]
        keep = [i for i, ti in enumerate(tag_t) if ti >= view_start]
        tag_E, tag_X = [tag_E[i] for i in keep], [tag_X[i] for i in keep]
        tag_id = [tag_id[i] for i in keep]
        tag_k, tag_bt = [tag_k[i] for i in keep], [tag_bt[i] for i in keep]
    t = np.array([r[0] for r in rec])
    X = np.array([r[1] for r in rec])
    E = np.array([r[2] for r in rec])
    tag = np.array([r[3] for r in rec])
    trk = np.array([r[4] for r in rec])
    jump = np.array([r[5] for r in rec], dtype=bool)
    upd_pc = np.array([r[6] for r in rec])
    ks = np.array([r[7] for r in rec], dtype=int)
    labels = [chr(ord("A") + i) for i in range(len(windows))]
    # Focus mask: keyframes drawn at full strength (all of them without --focus-from).
    fm = t >= focus_from if focus_from is not None else np.ones(len(t), dtype=bool)
    if not fm.any():
        raise SystemExit(f"--focus-from {focus_from:g}: no keyframes after it (run ends at {t[-1]:.0f} s)")
    i_f = int(np.argmax(fm))                      # first keyframe in focus

    def track(P: np.ndarray, m: np.ndarray, **kw) -> None:
        """Plot a track: faint up to the focus start, full strength from there.

        ``m`` is the focus mask of P's points; the faint part ends on the first
        focus point so the two pieces join.
        """
        if m.all():
            ax.plot(P[:, 0], P[:, 1], **kw)
            return
        j = int(np.argmax(m))
        faint = {k: v for k, v in kw.items() if k != "label"}
        ax.plot(P[:j + 1, 0], P[:j + 1, 1], alpha=FAINT, **faint)
        ax.plot(P[j:, 0], P[j:, 1], **kw)

    # --batch: re-solve the very graph this replay built, once, with LM, from
    # the real-time estimates (full smoothing; tools/batch_lm.py). The whole
    # replay initializes it (kf_X, pre-crop), only the drawing is cropped.
    lm, S_lm = None, None
    if batch:
        from .batch_lm import optimize_full_graph
        result, st = optimize_full_graph(est, kf_X)
        lm = {int(k): result.atPose3(K.X(int(k))) for k in kf_X if result.exists(K.X(int(k)))}
        if est.state == StationState.TRACKING and result.exists(K.S):
            S_lm = result.atPose3(K.S)
        d = [pose_distance(kf_X[k], lm[k]) for k in sorted(lm)]
        tp = np.array([x[0] for x in d]) * 100
        rr = np.degrees([x[1] for x in d])
        print(f"batch LM: {st['n_factors']} factors, {len(lm)} poses, "
              f"{st['iterations']} iterations in {st['seconds']:.1f} s; graph error "
              f"real-time {st['error_init']:.1f} -> LM {st['error_lm']:.1f}")
        print(f"  X real-time vs LM: position median {np.median(tp):.2f} / max {tp.max():.2f} cm, "
              f"rotation median {np.median(rr):.2f} / max {rr.max():.2f} deg")
        S_rt = est.backend.pose(K.S)
        if S_lm is not None and S_rt is not None:
            dt_, dr_ = pose_distance(S_rt, S_lm)
            print(f"  S final real-time vs LM: {dt_ * 100:.2f} cm, {math.degrees(dr_):.2f} deg")

    if timing:
        fig, (ax, bx, cx) = plt.subplots(3, 1, figsize=(13, 17), sharex=False,
                                         gridspec_kw={"height_ratios": [3, 1, 1]})
    else:
        fig, (ax, bx) = plt.subplots(2, 1, figsize=(13, 13), gridspec_kw={"height_ratios": [3, 1]})

    # ------------------------------------------------------------ BEV
    track(E, fm, ls="--", c=C_EKF, lw=1.1,
          label=f"EKF E_k (odom coordinates; odom = map at {t_anchor:.0f} s)")
    # The graph track is one solid line through every keyframe; the dots on
    # it mark keyframes with tag factors (red, small, so the line stays readable).
    track(X, fm, c=C_X_TAG, lw=1.3, label="graph X_k (real time)")
    ax.scatter(X[tag & fm, 0], X[tag & fm, 1], s=3, c=C_KF_TAG, linewidths=0, zorder=3)
    ax.scatter(X[tag & ~fm, 0], X[tag & ~fm, 1], s=3, c=C_KF_TAG, alpha=FAINT, linewidths=0,
               zorder=3)
    ax.scatter([], [], s=20, c=C_KF_TAG, linewidths=0, label="keyframe with tags")
    lm_xy = lmm = None
    if lm:
        lmm = np.array([int(k) in lm for k in ks])
        lm_xy = np.array([lm[int(k)].translation() for k in ks if int(k) in lm])
        track(lm_xy, fm[lmm], ls="-.", c=C_LM, lw=1.2, label="batch LM X_k (smoothed)")
    # Keyframes whose optimizer starting guess for X_k moved at least
    # INIT_JUMP_M from the odometry prediction (DockEstimator._init_from_tags):
    # the tag returned after a gap and the drifted odometry guess was far off.
    # Nothing is reset; these are the visible steps in the real-time track,
    # where the drift of the gap is corrected at once.
    if (jump & ~fm).any():
        ax.scatter(X[jump & ~fm, 0], X[jump & ~fm, 1], s=45, marker="D", facecolors="none",
                   edgecolors="magenta", linewidths=1.2, alpha=FAINT, zorder=6)
    if (jump & fm).any():
        n_gap = int((jump & fm).sum())
        ax.scatter(X[jump & fm, 0], X[jump & fm, 1], s=45, marker="D", facecolors="none",
                   edgecolors="magenta", linewidths=1.2, zorder=6,
                   label=f"tag back after a gap, {n_gap} keyframe{'s' if n_gap != 1 else ''}")
    # Where the two tracks start and end (they coincide at the start: X_0 = E_0).
    ax.scatter([X[0, 0]], [X[0, 1]], s=60, marker="o", c="white", edgecolors="black", zorder=7,
               alpha=1.0 if fm.all() else FAINT, label=None if not fm.all() else f"start, {t[0]:.0f} s")
    if not fm.all():
        # Both tracks enter the focus stretch here, each at its own pose.
        for P, c in ((X, C_X_TAG), (E, C_EKF)):
            ax.scatter([P[i_f, 0]], [P[i_f, 1]], s=60, marker="o", c="white", edgecolors=c,
                       linewidths=1.5, zorder=7)
        ax.scatter([], [], s=60, marker="o", c="white", edgecolors="black",
                   label=f"focus start, {t[i_f]:.0f} s (earlier: faint)")
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
    # Where the last keyframe leaves each camera, not just base_link: the end
    # pose chained through T_bc. The offset from the x marker is the lever arm.
    k_end = int(ks[-1])
    for kf_map, c, name in ((kf_X, C_X_TAG, "graph"), (kf_E, C_EKF, "EKF")):
        P3 = kf_map.get(k_end)
        if P3 is None:
            continue
        for cam, cam_setup in est.cameras.items():
            q = P3.compose(cam_setup.T_bc).translation()
            ax.scatter([q[0]], [q[1]], marker="^", s=55, c=c, edgecolors="black",
                       linewidths=0.6, zorder=7)
            ax.annotate(f"{name} end {cam}", (q[0], q[1]), xytext=(0, 9), ha="center",
                        textcoords="offset points", fontsize=8, color=c,
                        bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=1.0))
    tfm = np.array(tag_t) >= focus_from if focus_from is not None else np.ones(len(tag_t), dtype=bool)
    if tag_E and not tfm.any():
        print(f"no tag observations after --focus-from {focus_from:g}: tag clouds left out")
    if tag_E and tfm.any():
        # Observations before the focus: faint dots only, no circles.
        qE_all, qX_all = np.array(tag_E), np.array(tag_X)
        ax.scatter(qE_all[~tfm, 0], qE_all[~tfm, 1], s=5, c=C_EKF, alpha=0.35 * FAINT,
                   linewidths=0, zorder=2)
        ax.scatter(qX_all[~tfm, 0], qX_all[~tfm, 1], s=5, c=C_X_TAG, alpha=0.6 * FAINT,
                   linewidths=0, zorder=6)
        qE, qX = qE_all[tfm], qX_all[tfm]

        # Graph cloud on top of the (hollow) final tag marker: it is tight
        # enough to hide under it. Empty scatters give readable legend markers.
        ax.scatter(qE[:, 0], qE[:, 1], s=5, c=C_EKF, alpha=0.35, linewidths=0, zorder=2)
        ax.scatter(qX[:, 0], qX[:, 1], s=5, c=C_X_TAG, alpha=0.6, linewidths=0, zorder=6)
        ids = np.array([tag_key_str(t) for t in tag_id])[tfm]
        tag_k_f = [k for k, f in zip(tag_k, tfm) if f]
        tag_bt_f = [bt for bt, f in zip(tag_bt, tfm) if f]
        methods = [(qE, ids, C_EKF, "EKF"), (qX, ids, C_X_TAG, "graph")]
        if lm:
            ok = np.array([int(k) in lm for k in tag_k_f])
            if ok.any():
                qLM = np.array([lm[int(k)].compose(bt).translation()
                                for k, bt in zip(tag_k_f, tag_bt_f) if int(k) in lm])
                ax.scatter(qLM[:, 0], qLM[:, 1], s=5, c=C_LM, alpha=0.6, linewidths=0, zorder=6)
                methods.append((qLM, ids[ok], C_LM, "batch LM"))
        for q, mid, col, name in methods:
            per_tag = []
            for tag_name in sorted(set(ids)):
                c, rms, rmax = cloud_spread(q[mid == tag_name])
                # Filled circle: RMS radius (~63 % of a Gaussian cloud inside),
                # i.e. exactly the number in the legend. Dashed ring: the max,
                # which a single outlier sets, so it is only an outline.
                # Zoomed on the graph, a runaway EKF's circles (metres) would
                # cover the whole view: the legend keeps their numbers.
                if zoom_graph is None or name != "EKF":
                    ax.add_patch(Circle(c, rms, facecolor=col, edgecolor=col, alpha=0.18, lw=1.2,
                                        zorder=4))
                    ax.add_patch(Circle(c, rmax, fill=False, edgecolor=col, alpha=0.7, lw=1.0,
                                        ls="--", zorder=4))
                per_tag.append(f"{tag_name} {rms * 100:.1f}/{rmax * 100:.1f}")
                print(f"tag {tag_name} via {name}: {int((mid == tag_name).sum())} obs, horizontal spread "
                      f"RMS {rms * 100:.1f} cm, max {rmax * 100:.1f} cm")
            since = f" (from {focus_from:g} s)" if focus_from is not None else ""
            ax.scatter([], [], s=25, c=col, alpha=0.7, linewidths=0,
                       label=f"tag cloud via {name}{since}, RMS/max cm:\n" + ", ".join(per_tag))
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
                ax.text(q[0], q[1], f"  {tag_key_str(tg)}", fontsize=9, color="black", va="center")
        # Each final tag seen from each track's end camera, in the camera's
        # optical axes (x right, y down, z forward): (P_end T_bc)^-1 S T_j.
        # Both use the graph's S T_j, so the EKF line answers "where would
        # the tag be in the image if the EKF pose were right".
        for kf_map, name in ((kf_X, "graph"), (kf_E, "EKF")):
            P3 = kf_map.get(k_end)
            if P3 is None:
                continue
            for cam, cam_setup in est.cameras.items():
                for tg in est.tag_index.tags():
                    T = est.backend.pose(est.tag_index.key(tg))
                    if T is None:
                        continue
                    q = P3.compose(cam_setup.T_bc).inverse().compose(S.compose(T)).translation()
                    print(f"end {t[-1]:.0f} s, {name} pose: tag {tag_key_str(tg)} in {cam}: "
                          f"right {q[0]:+.2f}, down {q[1]:+.2f}, forward {q[2]:+.2f} m")
    if S_lm is not None:
        d = S_lm.translation()
        ax.scatter([d[0]], [d[1]], marker="D", s=90, c="#e9d8fd", edgecolors=C_LM,
                   linewidths=1.2, zorder=5, label="S (dock_point, batch LM)")
    # Windows that end before the focus are not drawn; they keep their letter,
    # so the labels match the full overview.
    shown = [(lab, w) for lab, w in zip(labels, windows)
             if focus_from is None or w[1] >= focus_from]
    for lab, (w0, w1) in shown:
        m = (t >= w0) & (t <= w1)
        if not m.any():
            print(f"warning: no keyframes in window {lab} ({w0}-{w1} s)")
            continue
        # Label above the box, or below it when the station marker is close
        # above (the label's white background would hide S).
        c = 0.5 * (np.vstack([X[m, :2], E[m, :2]]).min(axis=0)
                   + np.vstack([X[m, :2], E[m, :2]]).max(axis=0))
        above = S is None or not (abs(S.translation()[0] - c[0]) < 1.0
                                  and 0 < S.translation()[1] - c[1] < 0.6)
        window_box(ax, X[m], E[m], f"{lab}: {w0:g}–{w1:g} s", above=above)
        # The track can pass the same place several times: highlight this
        # window's own segment (graph solid, EKF dashed) so the box is unambiguous.
        ax.plot(X[m, 0], X[m, 1], c=C_HL, lw=3.0, zorder=7, solid_capstyle="round")
        ax.plot(E[m, 0], E[m, 1], c=C_HL, lw=2.0, ls="--", zorder=7)
    # Time marks every 30 s on the graph track: the route can cross the same
    # place several times; the labels make the passes readable in order.
    time_labels(ax, X[fm], t[fm], every=time_every, color=C_X_TAG)
    if direction:
        for P, c in ((X, C_X_TAG), (E, C_EKF)):
            if not fm.all():
                direction_marks(ax, P[:i_f + 1], t[:i_f + 1], time_every, c, alpha=FAINT)
            direction_marks(ax, P[i_f:], t[i_f:], time_every, c)
        ax.scatter([], [], marker=">", s=40, c="grey", edgecolors="white", linewidths=0.5,
                   label="direction of travel at the time marks (graph blue, EKF green)")
    ax.set_aspect("equal", adjustable="datalim")
    if zoom_graph is not None:
        pts = [X[:, :2]]
        if lm_xy is not None:
            pts.append(lm_xy[:, :2])
        if S is not None:
            pts.append(S.translation()[None, :2])
            pts += [S.compose(est.backend.pose(est.tag_index.key(tg))).translation()[None, :2]
                    for tg in est.tag_index.tags() if est.backend.pose(est.tag_index.key(tg)) is not None]
        P = np.vstack(pts)
        lo, hi = P.min(axis=0) - zoom_graph, P.max(axis=0) + zoom_graph
        ax.set_xlim(lo[0], hi[0])
        ax.set_ylim(lo[1], hi[1])
        print(f"zoom on the graph: x {lo[0]:.1f}..{hi[0]:.1f}, y {lo[1]:.1f}..{hi[1]:.1f} m "
              f"(EKF track spans x {E[:, 0].min():.1f}..{E[:, 0].max():.1f}, "
              f"y {E[:, 1].min():.1f}..{E[:, 1].max():.1f} m)")
    # Axes: the graph frame (the node's odom_graph), called map here: fixed to
    # the tank, the station is constant in it. X_k, S and the tags are native
    # to it. E_k is drawn with its odom coordinates as they are: at the anchor
    # odom = odom_graph, so this is the EKF's dead reckoning from there, and
    # the gap between the tracks is the drift since. (Converting X_k into
    # odom through odom -> odom_graph = E_k X_k^-1 would just give E_k back.)
    ax.set_xlabel(f"map x (m)   [map = odom_graph; = odom at {t_anchor:.0f} s]")
    ax.set_ylabel("map y (m)")
    ax.grid(alpha=0.3)
    if shown:
        ax.plot([], [], c=C_HL, lw=3.0, label="window segment (graph solid, EKF dashed)")
    ax.scatter([], [], marker="^", s=45, facecolors="none", edgecolors="black",
               linewidths=0.6, label="end pose, camera (T_bc)")
    ax.plot([], [], ls="", marker="|", ms=9, c=C_X_TAG, label=f"time marks, {time_every:g} s")
    ax.legend(fontsize=8, loc="lower right")

    # ------------------------------------------------------------ time strip
    # Drift correction: 3D distance between the graph's and the EKF's AUV
    # position at each keyframe (independent of where the odom origin is).
    D = np.linalg.norm(X - E, axis=1)
    Dlm = np.linalg.norm(lm_xy - E[lmm], axis=1) if lm else None
    ymax = max(float(D.max()) * 1.15, 0.05)
    bx.plot(t, D, c=C_X_TAG, lw=1.0,
            label="drift correction ‖t(X_k) − t(E_k)‖, 3D (m)")
    if Dlm is not None:
        bx.plot(t[lmm], Dlm, ls="-.", c=C_LM, lw=1.0,
                label="drift correction, batch LM (m)")
    bx.vlines(t[tag], ymax * 0.93, ymax, color=C_KF_TAG, lw=0.4, label="keyframe with tag factors")
    pre = ~trk
    if pre.any():
        bx.axvspan(t[0], t[pre][-1], color="#fbd38d", alpha=0.5, lw=0, label="not yet TRACKING")
    for lab, (w0, w1) in shown:
        bx.axvspan(w0, w1, color="#1a202c", alpha=0.18, lw=0)
        bx.text(0.5 * (w0 + w1), ymax * 0.82, lab, ha="center", fontsize=10, fontweight="bold")
    if not fm.all():
        # Fade the strip before the focus too: a white veil over that stretch.
        bx.axvspan(t[0], t[i_f], color="white", alpha=1.0 - FAINT, lw=0, zorder=5)
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
    """``update_ms`` per keyframe stamp from the live node's recorded ``~/keyframe`` (JSON)."""
    import json

    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from std_msgs.msg import String
    reader = open_reader(bag)
    topic = live_topic("keyframe", {t.name for t in reader.get_all_topics_and_types()})
    if topic is None:
        return {}
    reader.set_filter(rosbag2_py.StorageFilter(topics=[topic]))
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
    live = _live_update_times(bag)
    if not live:
        raise SystemExit(f"--timing: no keyframe records of the live node in {bag} (record "
                         f"~/keyframe with publish_debug: true)")
    t_live = np.array(sorted(live)) - t_bag0
    u_live = np.array([live[k] for k in sorted(live)])
    m = (t_live >= t[0]) & (t_live <= t[-1])
    t_live, u_live = t_live[m], u_live[m]
    cpu, stats = cpu_name(), timing_stats
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
    cx.set_ylabel("factor-graph update per keyframe (ms)")
    cx.legend(fontsize=8, loc="lower left", ncol=1, frameon=False, bbox_to_anchor=(0.0, 1.0))
    cx.grid(alpha=0.3, which="both")
    print(f"update time: live {stats(u_live)} ({len(u_live)} keyframes); "
          f"PC replay {stats(upd_pc)} ({len(upd_pc)} keyframes)")


def main(argv: List[str] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("bag", help="rosbag2 directory with EKF odometry and tag detections")
    ap.add_argument("--config", required=True, nargs="+", help="node parameter file(s), merged in order")
    ap.add_argument("--start", type=float, default=None,
                    help="replay start, s into the original recording (the estimator starts fresh here)")
    ap.add_argument("--window", type=float, nargs=2, action="append", default=[],
                    metavar=("FROM", "TO"),
                    help="time window, s into the original recording (a few seconds). Graph mode: exactly "
                         "one. Overview: repeat to box several, or leave out for no boxes")
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
    ap.add_argument("--batch", action="store_true",
                    help="overview: after the replay, re-solve the same graph once with batch LM "
                         "(tools/batch_lm.py) and draw the smoothed X_k track, its drift "
                         "correction and its tag cloud alongside the real-time ones")
    ap.add_argument("--focus-from", type=float, default=None, metavar="FROM",
                    help="overview: draw everything before FROM (s into the recording) faint, "
                         "so FROM..END stands out; tag clouds and time marks cover FROM..END only")
    ap.add_argument("--direction", action="store_true",
                    help="overview: a triangle at each time mark on the graph and EKF tracks, "
                         "pointing the way the AUV travels")
    ap.add_argument("--zoom-graph", type=float, default=None, metavar="MARGIN",
                    help="overview: fit the BEV to the graph track, S and the tags plus MARGIN m; "
                         "an EKF that ran away leaves the frame")
    ap.add_argument("--time-every", type=float, default=30.0, metavar="SECONDS",
                    help="overview: spacing of the time marks (and --direction triangles)")
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
    if args.batch:
        backend += " + batch LM re-solve of the whole graph"
    if args.overview is not None:
        t0 = (args.view_start if args.view_start is not None
              else args.start if args.start is not None else 0.0)
        overview(args.bag, args.config, args.start, args.overview,
                 [tuple(w) for w in args.window], args.out,
                 f"{name}, {t0:g}–{args.overview:g} s{rate}: "
                 f"{'part of the run' if args.view_start is not None else 'whole run'}"
                 f"{f', focus on {args.focus_from:g}–{args.overview:g} s' if args.focus_from is not None else ''}"
                 f"{', with the factor-graph windows' if args.window else ''}{backend}",
                 frame_stride=args.frame_stride, view_start=args.view_start,
                 timing=args.timing, batch=args.batch, focus_from=args.focus_from,
                 direction=args.direction, time_every=args.time_every,
                 zoom_graph=args.zoom_graph)
        return
    if len(args.window) != 1:
        ap.error("graph mode takes exactly one --window (use --overview to box several)")
    w0, w1 = args.window[0]
    win = collect(args.bag, args.config, args.start, w0, w1, args.frame_stride)
    prefix = f"{args.label}: " if args.label else ""
    plot(win, args.out, f"{prefix}Factor graph, {name}, {w0:g}–{w1:g} s{since}{rate}{backend}")


if __name__ == "__main__":
    main()
