"""Analyze a recorded live run: EKF vs the node's fixed-lag smoother vs batch LM, from one YAML.

Usage (a plain script; source ROS first for rosbag2_py and the URDF packages)::

    python3 race_auv_factorgraph/tools/analyze_run.py config/analyze_run_example.yaml

``RUN.yaml`` (see ``config/analyze_run_example.yaml``) names the bag, the
dock_fuser parameter files the live node ran with, the time range, the
windows to box and draw as factor graphs, and which outputs to make. One
YAML per recording keeps every analysis reproducible.

What is compared
----------------
Three estimates of the AUV pose at every keyframe ``k`` (one keyframe per
detector message):

``E_k``     the EKF, **recorded** (``/race_auv/odometry/filtered``,
            interpolated at the keyframe stamp, exactly as the node does);
``X_rt``    the node's fixed-lag smoother, **recorded** live
            (``/dock_fuser/auv_pose``, ``publish_debug: true``): the
            real-time estimate, as known at keyframe k;
``X_lm``    batch Levenberg-Marquardt over the **whole** run (full
            smoothing): every ``X_k`` also uses tag observations made after it.

Nothing is re-detected. The batch graph is rebuilt by replaying the bag's
*recorded* detector output (corner detections) and EKF through the same
scheduler and estimator as the node (``replay_bag.replay``, with the YAML's
dock_fuser config). The backend keeps a copy of every factor it receives,
including those the smoother marginalized (``Backend.full_graph``), and LM
re-optimizes that graph at once, initialized at the replay's real-time
estimates (``batch_lm.optimize_full_graph``). The replay also yields the
single-image PnP pose ``T_ct`` of every tag observation, used for the tag
clouds below, and a check that the replay reproduces the live run
(``X_rt`` recorded vs replayed, printed in the summary).

Outputs (in ``out_dir``)
------------------------
``overview.png``
    The docs' overview figure (``plot_graph --overview``) for a live run.
    Top left: bird's-eye view of ``E_k``, ``X_rt`` and ``X_lm``; a yellow band
    where the keyframes had tag factors, and the keyframes seeded from the
    tag (from the live ``/dock_fuser/keyframe``); the final station ``S``
    (live and LM) and tags; a labelled box per window; and every tag
    observation chained onto each track (``E_k T_bc T_ct``, ``X_k T_bc
    T_ct``): the tag does not move, so the spread of each cloud measures how
    consistent that track is with the camera. Top right: the same tracks in
    the vertical planes of ``side_views`` (default y-z). Below, on one time
    axis: drift correction ``||t(X_k) - t(E_k)||`` of the live smoother and
    of LM; odom z (depth) of the three; and (``timing``) the live node's
    update time per keyframe (``update_ms``), optionally against this
    computer's replay.
``rt_vs_batch.png``
    How far the live estimate was from the batch optimum over time: AUV
    position ``|t(X_rt) - t(X_lm)|``, the docking pose ``X^-1 S`` that
    docking actually uses (live ``X_rt^-1 S_rt`` vs ``X_lm^-1 S_lm``), its
    yaw, and the live station estimate ``S_rt`` against the LM one.
``graph_<label>.png``
    For each window with ``graph: true``: the factor graph the estimator
    built over that window (``plot_graph.collect`` / ``plot``).
``keyframes.csv``, ``summary.txt``
    Per-keyframe x, y, z, yaw of ``E``, ``X_rt``, ``X_lm`` and the docking
    pose; the summary statistics printed at the end.

Every time panel also shades the DVL bottom-lock dropouts (no message on
the config's ``dvl_topic`` for longer than ``estimator.dvl_timeout``): the
EKF dead-reckons there and snaps back on re-lock.

``start`` / ``end`` bound the replay, and the estimator starts fresh at
``start``. To look at part of a live run, keep the replay on the whole run
(the live node started with the bag) and crop the drawing with
``view_start`` / ``view_end``: the estimates are then the live ones and the LM
optimum of the whole run, just a shorter stretch of them.

Times are seconds into the recording (``replay_bag.source_start_time``), the
same axis as the other tools, so window times can be read off any of them.
"""

from __future__ import annotations

import argparse
import copy
import csv
import dataclasses
import json
import math
import os
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import gtsam
import numpy as np

if __package__ in (None, ""):
    # Run as a file (python3 .../tools/analyze_run.py): make the package
    # importable from this source tree, so the relative imports below work
    # without installing it.
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    __package__ = "race_auv_factorgraph.tools"

from ..core.common import keys as K
from ..core.common.geometry import pose_distance, pose_from_quat_xyzw
from ..core.frontend.pose_buffer import PoseBuffer
from ..core.init.state_machine import StationState
from ..ros.params import build_setup, getter_from_yaml
from .batch_lm import optimize_full_graph
from .replay_bag import _open_reader, bag_source, bag_start_time, replay, source_start_time

# The three tracks, Okabe-Ito colours (colour-blind safe, far apart in hue).
C_E = "#009E73"           # EKF: bluish green, dashed
C_RT = "#0072B2"          # live fixed-lag smoother: blue
C_LM = "#D55E00"          # batch LM: vermillion
# Keyframes with tag factors: a wide yellow band under the tracks (Okabe-Ito
# yellow, the one hue left free by the tracks), ochre ticks on the time panels.
C_TAGKF = "#F0E442"
C_TICK = "#C9A800"
C_RP = "#CC79A7"          # replay with the run YAML's settings (reddish purple), when not the live run
C_ATT = "#56B4E9"         # tag factors from an attached camera (cam_down): sky-blue triangles
C_HL = "#7f7f7f"          # window segments: grey, clear of the yellow band
LW_RP = 2.0
# Line widths: the live track is drawn wide and LM narrower on top of it, so
# where they coincide LM shows with a blue edge instead of hiding the live one.
LW_E, LW_RT, LW_LM, LW_T = 1.8, 2.8, 1.6, 1.6


def _style() -> None:
    """Larger fonts for every figure of this tool (explicit sizes below follow suit)."""
    import matplotlib
    matplotlib.rcParams.update({"font.size": 13, "axes.titlesize": 13, "axes.labelsize": 13,
                                "legend.fontsize": 11, "xtick.labelsize": 12,
                                "ytick.labelsize": 12})

# Every key the run YAML may set, with its default. Nested dicts are merged
# key by key, so a YAML only lists what differs.
DEFAULTS: Dict[str, Any] = {
    "bag": "",                       # rosbag2 directory (required)
    "out_dir": "",                   # default: <bag>/analysis
    "title": "",                     # default: the recording's name
    # dock_fuser parameter files the live node ran with, merged in order
    # (like --params-file). "package://<pkg>/<path>" resolves in the
    # package's share directory; a relative path is relative to this YAML.
    "dock_fuser_config": [],
    # dock_fuser parameters merged on top of those files (last wins), in the
    # node's own nesting, e.g. {tag_ids: [54], tag_sizes: [0.22]} to pin the
    # tags the live run used. Written to <out_dir>/dock_fuser_overrides.yaml
    # and appended to dock_fuser_config, so every replay here sees them.
    "dock_fuser_overrides": {},
    "start": None,                   # s into the recording; replay starts fresh here
    "end": None,                     # s into the recording; default: end of the bag
    "view_start": None,              # draw only from here (the replay still starts at start)
    "view_end": None,                # ... up to here (the replay and LM still cover start..end)
    "frame_stride": 1,               # keep every Nth detector image (1 = as recorded)
    # Also draw the replay's own real-time track (fixed-lag, as replayed with
    # this YAML's settings): "auto" when it differs from the recorded live
    # track (median > 1 mm: other detections or settings than the live run,
    # e.g. a redetect_bag.py bag), or true / false.
    "replay_track": "auto",
    "topics": {                      # recorded topics of the live run
        "ekf": "/race_auv/odometry/filtered",
        "auv_pose": "/dock_fuser/auv_pose",
        "station_pose": "/dock_fuser/station_pose",
        "keyframe": "/dock_fuser/keyframe",
        "dvl": None,                 # bottom-lock velocity; None: the config's dvl_topic, "": off
    },
    "batch": {
        "enabled": True,
        "max_iterations": 100,
    },
    "timing": {
        "enabled": True,
        "live_label": "live node",   # the machine that ran the node, e.g. "Jetson Orin Nano"
        # Also time the replay on this computer. The replay then keeps the
        # config's covariance mode (the node computes it inside the timed
        # update), which makes it slower; otherwise covariance is skipped.
        "replay": False,
    },
    # Boxes on the overview and shaded spans on the time panels; graph: true
    # also draws that window's factor graph (a separate replay up to its end).
    "windows": [],                   # [{label: A, from: 148.0, to: 153.0, graph: true}, ...]
    "outputs": {
        "overview": True,
        "rt_vs_batch": True,
        "graphs": True,
        "csv": True,
    },
    # Side views beside the bird's-eye view in overview.png, each two of x, y, z.
    "side_views": ["yz"],
    "dpi": 150,
}


# ---------------------------------------------------------------------------- config


def _merge(base: Dict[str, Any], over: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if k not in base:
            raise SystemExit(f"unknown key '{k}' in run YAML (known: {', '.join(base)})")
        # An empty default dict is free-form (dock_fuser_overrides): taken as given.
        nested = isinstance(base[k], dict) and base[k] and isinstance(v, dict)
        out[k] = _merge(base[k], v) if nested else v
    return out


def _resolve(path: str, base_dir: str) -> str:
    """``package://pkg/rel`` -> the package's share dir; relative -> relative to the YAML."""
    path = os.path.expanduser(os.path.expandvars(str(path)))
    if path.startswith("package://"):
        from ament_index_python.packages import get_package_share_directory
        pkg, _, rel = path[len("package://"):].partition("/")
        return os.path.join(get_package_share_directory(pkg), rel)
    return path if os.path.isabs(path) else os.path.normpath(os.path.join(base_dir, path))


def write_overrides(cfg: Dict[str, Any]) -> None:
    """Append ``dock_fuser_overrides`` to the config list as a generated parameter file."""
    if not cfg["dock_fuser_overrides"]:
        return
    import yaml
    path = os.path.join(cfg["out_dir"], "dock_fuser_overrides.yaml")
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump({"dock_fuser": {"ros__parameters": cfg["dock_fuser_overrides"]}}, f,
                       sort_keys=False)
    cfg["dock_fuser_config"].append(path)


def load_run_config(path: str) -> Dict[str, Any]:
    import yaml
    with open(path, encoding="utf-8") as f:
        cfg = _merge(DEFAULTS, yaml.safe_load(f) or {})
    base = os.path.dirname(os.path.abspath(path))
    if not cfg["bag"]:
        raise SystemExit(f"{path}: 'bag' is required")
    if not cfg["dock_fuser_config"]:
        raise SystemExit(f"{path}: 'dock_fuser_config' is required (the node's parameter files)")
    cfg["bag"] = _resolve(cfg["bag"], base)
    cfg["dock_fuser_config"] = [_resolve(p, base) for p in cfg["dock_fuser_config"]]
    cfg["out_dir"] = _resolve(cfg["out_dir"], base) if cfg["out_dir"] else os.path.join(
        cfg["bag"], "analysis")
    for i, w in enumerate(cfg["windows"]):
        w.setdefault("label", chr(ord("A") + i))
        w.setdefault("graph", True)
        w["from"], w["to"] = float(w["from"]), float(w["to"])
    return cfg


# ---------------------------------------------------------------------------- data


def _stamp(header) -> float:
    return float(header.stamp.sec) + 1e-9 * float(header.stamp.nanosec)


def dvl_gaps(stamps: List[float], timeout: float, t0: float) -> List[Tuple[float, float]]:
    """Bottom-lock dropouts: spans (s into the recording) with no DVL message for > ``timeout``.

    Same rule as the estimator's DVL monitor (core/frontend/dvl_monitor.py):
    lock is lost ``timeout`` after the last message, and back at the next one.
    """
    st = np.sort(np.asarray(stamps, dtype=float)) - t0
    if len(st) < 2:
        return []
    gap = np.flatnonzero(np.diff(st) > timeout)
    return [(float(st[i] + timeout), float(st[i + 1])) for i in gap]


def _key(t: float) -> float:
    """Dictionary key for a keyframe stamp: the node and the replay use the same detector stamp."""
    return round(float(t), 4)


@dataclass
class Recorded:
    """What the live node and the EKF published, read from the bag."""

    E: PoseBuffer = field(default_factory=lambda: PoseBuffer(max_age=1e12))
    X: Dict[float, gtsam.Pose3] = field(default_factory=dict)      # stamp -> live X_rt
    S: Dict[float, gtsam.Pose3] = field(default_factory=dict)      # stamp -> live S_rt
    kf: Dict[float, dict] = field(default_factory=dict)            # stamp -> /dock_fuser/keyframe
    dvl: List[float] = field(default_factory=list)                  # DVL bottom-lock stamps


def read_recorded(bag: str, topics: Dict[str, str]) -> Recorded:
    """One pass over the bag's EKF and dock_fuser debug topics (all the bag holds of them)."""
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message

    reader = _open_reader(bag)
    types = {t.name: t.type for t in reader.get_all_topics_and_types()}
    wanted = {role: t for role, t in topics.items() if t}
    missing = [t for t in wanted.values() if t not in types]
    if missing:
        print("warning: not in the bag:", ", ".join(missing))
    by_topic = {t: role for role, t in wanted.items() if t in types}
    cls = {t: get_message(types[t]) for t in by_topic}
    reader.set_filter(rosbag2_py.StorageFilter(topics=list(by_topic)))
    rec = Recorded()
    while reader.has_next():
        topic, data, _ = reader.read_next()
        msg = deserialize_message(data, cls[topic])
        role = by_topic[topic]
        if role == "dvl":
            rec.dvl.append(_stamp(msg.header))
            continue
        if role == "keyframe":
            j = json.loads(msg.data)
            rec.kf[_key(j["stamp"])] = j
            continue
        pose = msg.pose.pose if role == "ekf" else msg.pose
        P = pose_from_quat_xyzw((pose.position.x, pose.position.y, pose.position.z),
                                (pose.orientation.x, pose.orientation.y, pose.orientation.z,
                                 pose.orientation.w))
        t = _stamp(msg.header)
        if role == "ekf":
            rec.E.add(t, P)
        elif role == "auv_pose":
            rec.X[_key(t)] = P
        elif role == "station_pose":
            rec.S[_key(t)] = P
    print(f"recorded: {len(rec.E)} EKF poses, {len(rec.X)} live X_k, {len(rec.S)} live S, "
          f"{len(rec.kf)} keyframe records, {len(rec.dvl)} DVL messages")
    return rec


@dataclass
class Batch:
    """The replay (rebuilt graph) and its LM optimum, per keyframe index k."""

    stamps: Dict[int, float] = field(default_factory=dict)
    X_replay: Dict[int, gtsam.Pose3] = field(default_factory=dict)  # replay's real-time X_k
    X_lm: Dict[int, gtsam.Pose3] = field(default_factory=dict)
    accepted: Dict[int, bool] = field(default_factory=dict)
    tracking: Dict[int, bool] = field(default_factory=dict)
    seeded: Dict[int, bool] = field(default_factory=dict)
    update_ms: Dict[int, float] = field(default_factory=dict)
    S_replay: Dict[int, gtsam.Pose3] = field(default_factory=dict)  # replay's S at keyframe k
    # (k, tag, camera, T_bt): body -> tag of every usable observation (single-
    # image PnP), on keyframe k; for an attached camera k is its anchor keyframe
    # and T_bt includes the EKF motion since then.
    tag_obs: List[Tuple[int, Any, str, gtsam.Pose3]] = field(default_factory=list)
    seen_tags: set = field(default_factory=set)                     # tags with observations
    # anchor keyframe k -> [(camera, tag)] tag factors added by attached cameras
    attached: Dict[int, List[Tuple[str, Any]]] = field(default_factory=dict)
    S_lm: Optional[gtsam.Pose3] = None
    T_lm: Dict[str, gtsam.Pose3] = field(default_factory=dict)      # "36h11:54" -> S_lm T_j
    stats: Dict[str, float] = field(default_factory=dict)
    lag: float = float("nan")


def run_batch(cfg: Dict[str, Any], setup, get, shift: float) -> Batch:
    """Replay the recorded detections + EKF through the estimator, then LM on the full graph."""
    est_cfg = setup.cfg if cfg["timing"]["replay"] else dataclasses.replace(
        setup.cfg, covariance_mode="none")
    b = Batch(lag=float(est_cfg.smoother_lag))

    def on_kf(est, r, held) -> None:
        b.stamps[r.k] = r.stamp
        b.X_replay[r.k] = r.X
        b.accepted[r.k] = bool(r.accepted)
        b.tracking[r.k] = r.state == StationState.TRACKING
        b.seeded[r.k] = bool(r.seeded_from_tag)
        b.update_ms[r.k] = r.update_time * 1000.0
        if r.S is not None:
            b.S_replay[r.k] = r.S
        for tag, cam, T_ct in r.tag_obs:
            b.seen_tags.add(tag)
            b.tag_obs.append((r.k, tag, cam, est.cameras[cam].T_bc.compose(T_ct)))
        for a in r.attached:              # attached frames whose factors entered now
            for tag in a.accepted:
                b.attached.setdefault(a.anchor_k, []).append((a.camera, tag))
            for tag, T_bt in a.tag_obs:
                b.seen_tags.add(tag)
                b.tag_obs.append((a.anchor_k, tag, a.camera, T_bt))

    # replay() windows by seconds from the bag's own start; cfg times are
    # seconds into the recording, hence the shift.
    start = None if cfg["start"] is None else max(0.0, cfg["start"] - shift)
    end = None if cfg["end"] is None else cfg["end"] - shift
    print("replaying the recorded detections through the estimator...")
    est, sched = replay(cfg["bag"], get, setup, cfg=est_cfg, on_keyframe=on_kf, start=start,
                        end=end, frame_stride=int(cfg["frame_stride"]))
    print(f"replay: {len(b.stamps)} keyframes, final state {est.state.value}, "
          f"{est.n_tag_factors} tag factors, {est.n_recoveries} backend recoveries")
    if not cfg["batch"]["enabled"]:
        return b
    if est.n_recoveries:
        print("warning: backend rebuilt during the replay; the batch graph only covers the "
              "part after the last rebuild")
    print("batch LM over the whole graph...")
    result, b.stats = optimize_full_graph(est, b.X_replay, cfg["batch"]["max_iterations"])
    for k in b.stamps:
        if result.exists(K.X(k)):
            b.X_lm[k] = result.atPose3(K.X(k))
    if result.exists(K.S):
        b.S_lm = result.atPose3(K.S)
        for tg in est.tag_index.tags():
            key = est.tag_index.key(tg)
            if result.exists(key):
                b.T_lm[f"{tg[0]}:{tg[1]}"] = b.S_lm.compose(result.atPose3(key))
    print(f"batch LM: {b.stats['n_factors']} factors, {b.stats['iterations']} iterations in "
          f"{b.stats['seconds']:.1f} s, error {b.stats['error_init']:.1f} -> "
          f"{b.stats['error_lm']:.1f}")
    return b


@dataclass
class Series:
    """Everything aligned on the replay's keyframes (arrays of length n, NaN where missing)."""

    k: np.ndarray
    t: np.ndarray                    # s into the recording
    E: List[Optional[gtsam.Pose3]]
    X_rt: List[Optional[gtsam.Pose3]]
    X_lm: List[Optional[gtsam.Pose3]]
    S_rt: List[Optional[gtsam.Pose3]]
    tag: np.ndarray                  # keyframe with tag factors (live record, else replay)
    tracking: np.ndarray
    seeded: np.ndarray
    upd_live: np.ndarray             # ms
    upd_replay: np.ndarray           # ms
    X_rp: List[Optional[gtsam.Pose3]] = field(default_factory=list)  # replay's real-time X_k
    S_rp: List[Optional[gtsam.Pose3]] = field(default_factory=list)
    tag_rp: np.ndarray = None        # replay keyframe with tag factors (own or attached)
    attached: np.ndarray = None      # anchor keyframe of attached-camera tag factors
    show_rp: bool = False            # draw the replay track (it is not the live run)
    dvl_gaps: List[Tuple[float, float]] = field(default_factory=list)   # s into the recording


def align(rec: Recorded, b: Batch, t0: float, view_start: Optional[float],
          view_end: Optional[float], gaps: List[Tuple[float, float]],
          replay_track: Any = "auto") -> Series:
    """The replay's keyframes in ``view_start .. view_end``, with the recorded and LM poses."""
    ks = [k for k in sorted(b.stamps)
          if (view_start is None or b.stamps[k] - t0 >= view_start)
          and (view_end is None or b.stamps[k] - t0 <= view_end)]
    if not ks:
        raise SystemExit("no keyframes in the requested range")
    st = [b.stamps[k] for k in ks]
    kfs = [rec.kf.get(_key(s)) for s in st]
    n_live = sum(1 for s in st if _key(s) in rec.X)
    if n_live < 0.9 * len(ks):
        print(f"warning: only {n_live} of {len(ks)} replay keyframes have a live X_k with the "
              f"same stamp (frame_stride or the topics may not match the live run)")
    return Series(
        k=np.array(ks), t=np.array(st) - t0,
        E=[rec.E.interpolate(s) for s in st],
        X_rt=[rec.X.get(_key(s)) for s in st],
        X_lm=[b.X_lm.get(k) for k in ks],
        S_rt=[rec.S.get(_key(s)) for s in st],
        tag=np.array([bool(j["accepted"]) if j else b.accepted[k] for j, k in zip(kfs, ks)]),
        tracking=np.array([j["state"] == "TRACKING" if j else b.tracking[k]
                           for j, k in zip(kfs, ks)]),
        seeded=np.array([bool(j.get("seeded_from_tag")) if j else b.seeded[k]
                         for j, k in zip(kfs, ks)]),
        upd_live=np.array([float(j["update_ms"]) if j else np.nan for j in kfs]),
        upd_replay=np.array([b.update_ms[k] for k in ks]),
        X_rp=[b.X_replay[k] for k in ks],
        S_rp=[b.S_replay.get(k) for k in ks],
        tag_rp=np.array([b.accepted[k] or k in b.attached for k in ks]),
        attached=np.array([k in b.attached for k in ks]),
        show_rp=_show_replay(replay_track, [rec.X.get(_key(s)) for s in st],
                             [b.X_replay[k] for k in ks]),
        dvl_gaps=[(max(a, st[0] - t0), min(c, st[-1] - t0)) for a, c in gaps
                  if c >= st[0] - t0 and a <= st[-1] - t0],
    )


def _show_replay(mode: Any, live: List[Optional[gtsam.Pose3]],
                 replay: List[gtsam.Pose3]) -> bool:
    """``replay_track``: true / false, or "auto" = the replay is not the live run (> 1 mm)."""
    if str(mode).lower() in ("true", "1", "yes"):
        return True
    if str(mode).lower() in ("false", "0", "no"):
        return False
    d = [np.linalg.norm(a.translation() - c.translation()) for a, c in zip(live, replay)
         if a is not None]
    return not d or float(np.median(d)) > 1e-3


def _xyz(poses: List[Optional[gtsam.Pose3]]) -> np.ndarray:
    """n x 3 positions, NaN rows for missing poses (matplotlib leaves a gap there)."""
    return np.array([P.translation() if P is not None else [np.nan] * 3 for P in poses])


def _last(poses: List[Optional[gtsam.Pose3]]) -> Optional[gtsam.Pose3]:
    return next((P for P in reversed(poses) if P is not None), None)


def _stats_cm_deg(pairs: List[Tuple[gtsam.Pose3, gtsam.Pose3]]) -> str:
    d = [pose_distance(a, c) for a, c in pairs]
    if not d:
        return "n/a"
    t = np.array([x[0] for x in d]) * 100
    r = np.degrees([x[1] for x in d])
    return (f"position median {np.median(t):.2f} / p95 {np.percentile(t, 95):.2f} / max "
            f"{t.max():.2f} cm, rotation median {np.median(r):.2f} / max {r.max():.2f} deg")


# ---------------------------------------------------------------------------- figures


def _tag_clouds(s: Series, b: Batch, which: str) -> Dict[Any, np.ndarray]:
    """Per tag, where each observation puts it in odom through one track (E, X_rt, X_rp, X_lm).

    One cloud per tag: the tags sit at different places, so only a single
    tag's observations should land on one point.
    """
    idx = {int(k): i for i, k in enumerate(s.k)}
    track = {"E": s.E, "X_rt": s.X_rt, "X_rp": s.X_rp, "X_lm": s.X_lm}[which]
    pts: Dict[Any, list] = {}
    for k, tag, _, T_bt in b.tag_obs:
        if k in idx and track[idx[k]] is not None:
            pts.setdefault(tag, []).append(track[idx[k]].compose(T_bt).translation())
    return {tag: np.array(v).reshape(-1, 3) for tag, v in sorted(pts.items())}


def _spread(q: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """Centre, RMS and max horizontal distance (m) of a tag cloud around its own mean.

    The tag is fixed, so ideally every observation lands on one point; the
    cloud's own mean is the centre, so all clouds are measured the same way.
    """
    c = q[:, :2].mean(axis=0)
    r = np.linalg.norm(q[:, :2] - c, axis=1)
    return c, float(np.sqrt(np.mean(r ** 2))), float(r.max())


def _planes(cfg: Dict[str, Any]) -> List[Tuple[str, int, int]]:
    """``side_views`` as (name, axis index, axis index), e.g. ("yz", 1, 2)."""
    axis = {"x": 0, "y": 1, "z": 2}
    out = []
    for p in cfg["side_views"]:
        p = str(p).lower()
        if len(p) != 2 or any(c not in axis for c in p) or p[0] == p[1]:
            raise SystemExit(f"side_views: '{p}' is not a plane like yz or xz")
        out.append((p, axis[p[0]], axis[p[1]]))
    return out


def _mark_step(span: float, target: int = 10) -> float:
    """A round time step (s) giving about ``target`` marks over ``span`` seconds."""
    for step in (1, 2, 5, 10, 20, 25, 50, 100, 200, 500):
        if span / step <= target:
            return float(step)
    return 1000.0


def _draw_direction(ax, t: np.ndarray, X: np.ndarray, i: int, j: int) -> None:
    """Time marks on the live track: an arrowhead in the direction of travel and the time.

    The tracks loop over themselves; these marks tell which way the AUV went
    and tie a place on the path to the time panels below. The arrow points
    from the mark to the pose ~1 s later; where the AUV barely moved in this
    plane (< 2 cm), only the label is drawn.
    """
    ok = np.isfinite(X[:, i]) & np.isfinite(X[:, j])
    if ok.sum() < 2:
        return
    tt, P = t[ok], X[ok][:, [i, j]]
    step = _mark_step(tt[-1] - tt[0])
    first = math.ceil((tt[0] + 0.25 * step) / step) * step     # clear of the start marker
    for n, tm in enumerate(np.arange(first, tt[-1] - 0.25 * step, step)):
        a = int(np.searchsorted(tt, tm))
        c = int(np.searchsorted(tt, tm + 1.0))
        if c >= len(tt):
            break
        p, q = P[a], P[c]
        if np.linalg.norm(q - p) > 0.02:
            ax.annotate("", xy=q, xytext=p, zorder=10,
                        arrowprops=dict(arrowstyle="-|>", color="black", lw=1.5,
                                        mutation_scale=26, shrinkA=0, shrinkB=0))
        ax.annotate(f"{tm:.0f} s", p, xytext=(7, 7), textcoords="offset points", fontsize=10,
                    zorder=10, bbox=dict(facecolor="white", edgecolor="0.6", alpha=0.85,
                                         pad=1.2, lw=0.5))
    ax.plot([], [], c="black", marker=r"$\rightarrow$", ms=14, ls="",
            label=f"direction of travel, every {step:g} s (time label)")


def _draw_tracks(ax, s: Series, b: Batch, setup, i: int, j: int, windows: List[dict],
                 E: np.ndarray, X: np.ndarray, L: np.ndarray, R: np.ndarray) -> None:
    """The tracks, the tag-factor band, station and tags and window segments in plane (i, j).

    Shared by the bird's-eye view and the side views, so every plane reads
    the same: EKF green dashed, live fixed-lag blue (wide), the replay with
    this YAML's settings reddish purple (only when it is not the live run),
    batch LM vermillion on top, a yellow band where the keyframes had tag
    factors (of the graph being solved: the replay's when it is drawn) and
    sky-blue triangles where an attached camera added tag factors.
    """
    band, mask = (R, s.tag_rp) if s.show_rp else (X, s.tag)
    ax.scatter(band[mask, i], band[mask, j], s=260, c=C_TAGKF, alpha=0.9, linewidths=0, zorder=1,
               label="keyframe with tag factors (yellow band)")
    ax.plot(E[:, i], E[:, j], ls="--", c=C_E, lw=LW_E, zorder=2, label="EKF E_k (recorded)")
    ax.plot(X[:, i], X[:, j], c=C_RT, lw=LW_RT, zorder=3,
            label=f"fixed-lag smoother {b.lag:g} s, X_rt (recorded live)")
    if s.show_rp:
        ax.plot(R[:, i], R[:, j], c=C_RP, lw=LW_RP, zorder=3.5,
                label="fixed-lag, this YAML's settings / detections, X_rp (replayed)")
    if b.X_lm:
        ax.plot(L[:, i], L[:, j], c=C_LM, lw=LW_LM, zorder=4,
                label="batch LM over the whole run, X_lm")
    if s.attached.any():
        cams = sorted({c for v in b.attached.values() for c, _ in v})
        P = R if s.show_rp else X
        ax.scatter(P[s.attached, i], P[s.attached, j], s=70, marker="v", c=C_ATT,
                   edgecolors="black", linewidths=0.6, zorder=8,
                   label=f"tag factors from {', '.join(cams)} (attached, "
                         f"{int(s.attached.sum())} keyframes)")
    if s.seeded.any():
        ax.scatter(X[s.seeded, i], X[s.seeded, j], s=60, marker="D", facecolors="none",
                   edgecolors="magenta", linewidths=1.6, zorder=8,
                   label=f"tag back after a gap: X_k seeded from the tag ({s.seeded.sum()})")

    # Station: S with its x axis, and the observed tags (live S with the URDF
    # tag pose; LM S with the LM tag pose), in the colour of their track.
    S_rt = _last(s.S_rt)
    seen = [(key, tag) for key, tag in setup.tags.items() if key in b.seen_tags]
    stations = [(S_rt, C_RT, "live, final")]
    if s.show_rp:
        stations.append((_last(s.S_rp), C_RP, "replay, final"))
    for S, col, name in (*stations, (b.S_lm, C_LM, "batch LM")):
        if S is None:
            continue
        p, d = S.translation(), S.rotation().matrix()[:, 0]
        ax.scatter([p[i]], [p[j]], s=220, c="#cfe3f3" if S is S_rt else "none", edgecolors=col,
                   linewidths=2.0, zorder=9, label=f"S, dock_point ({name})")
        ax.annotate("", (p[i] + 0.4 * d[i], p[j] + 0.4 * d[j]), (p[i], p[j]),
                    arrowprops=dict(arrowstyle="->", color=col, lw=1.8), zorder=9)
    if S_rt is not None:
        for n, (key, tag) in enumerate(seen):
            q = S_rt.compose(tag.T_dock_tag).translation()
            ax.scatter([q[i]], [q[j]], marker="s", s=80, facecolors="none", edgecolors="black",
                       linewidths=1.6, zorder=9, label="tag (live S, URDF pose)" if n == 0 else None)
            ax.annotate(f"{key[0]}:{key[1]}", (q[i], q[j]), xytext=(8, -4),
                        textcoords="offset points", fontsize=11)
    for n, T in enumerate(b.T_lm.values()):
        q = T.translation()
        ax.scatter([q[i]], [q[j]], marker="s", s=150, facecolors="none", edgecolors=C_LM,
                   linewidths=1.4, zorder=9, label="tag (batch LM)" if n == 0 else None)

    for w in windows:
        m = (s.t >= w["from"]) & (s.t <= w["to"])
        if m.any():
            ax.plot(X[m, i], X[m, j], c=C_HL, lw=7.0, alpha=0.55, zorder=5,
                    solid_capstyle="round")
    if windows:
        ax.plot([], [], c=C_HL, lw=7.0, alpha=0.55, label="window segment (live track)")
    _draw_direction(ax, s.t, X, i, j)

    # Where the tracks start and end (they coincide at the start until TRACKING).
    ok = np.flatnonzero(np.isfinite(X[:, i]))
    ok_e = np.flatnonzero(np.isfinite(E[:, i]))
    ax.scatter([X[ok[0], i]], [X[ok[0], j]], s=110, c="white", edgecolors="black",
               linewidths=1.8, zorder=11, label=f"start, {s.t[ok[0]]:.0f} s")
    ax.scatter([X[ok[-1], i]], [X[ok[-1], j]], s=130, marker="X", c=C_RT, edgecolors="black",
               linewidths=1.0, zorder=11, label=f"end, {s.t[ok[-1]]:.0f} s (live)")
    ax.scatter([E[ok_e[-1], i]], [E[ok_e[-1], j]], s=130, marker="X", c=C_E, edgecolors="black",
               linewidths=1.0, zorder=11, label="end (EKF)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(alpha=0.3)


def plot_overview(cfg: Dict[str, Any], s: Series, b: Batch, setup, png: str, title: str,
                  lines: List[str]) -> None:
    """One figure for the whole analysis (see the module docstring).

    Layout: the bird's-eye view (left, large) and the side views (right,
    stacked), with one shared legend above them; below, full-width time
    panels on a common time axis: drift correction, depth, update time.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _style()
    from matplotlib.patches import Circle, Rectangle

    E, X, L, R = _xyz(s.E), _xyz(s.X_rt), _xyz(s.X_lm), _xyz(s.X_rp)
    has_lm = bool(b.X_lm)
    planes = _planes(cfg)
    timing = cfg["timing"]["enabled"] and np.isfinite(s.upd_live).any()
    n_time = 3 if timing else 2

    fig = plt.figure(figsize=(20, 15 + 3.6 * n_time), layout="constrained")
    # Rows: shared legend of the planes | planes | time panels.
    outer = fig.add_gridspec(3, 1, height_ratios=[0.42, 3.2, 1.15 * n_time])
    lax = fig.add_subplot(outer[0])
    lax.axis("off")
    top = outer[1].subgridspec(max(1, len(planes)), 2, width_ratios=[1.7, 1.0])
    ax = fig.add_subplot(top[:, 0])
    sides = [fig.add_subplot(top[r, 1]) for r in range(len(planes))]
    bottom = outer[2].subgridspec(n_time, 1)
    tx = [fig.add_subplot(bottom[0])]
    tx += [fig.add_subplot(bottom[r], sharex=tx[0]) for r in range(1, n_time)]

    # ------------------------------------------------------------ bird's-eye view
    _draw_tracks(ax, s, b, setup, 0, 1, cfg["windows"], E, X, L, R)
    # Tag clouds, per tag: every observation chained onto each track. Filled
    # circle = RMS spread, dashed ring = max (a single outlier sets it).
    tracks = [("E", C_E, "EKF", "E_k"), ("X_rt", C_RT, "live", "X_rt")]
    if s.show_rp:
        tracks.append(("X_rp", C_RP, "replay", "X_rp"))
    tracks.append(("X_lm", C_LM, "LM", "X_lm"))
    cams = {}
    for _, tag, cam, _ in b.tag_obs:
        cams.setdefault(tag, set()).add(cam)
    for which, col, name, expr in tracks:
        per_tag = []
        for tag, q in _tag_clouds(s, b, which).items():
            if len(q) < 3:
                continue
            c, rms, rmax = _spread(q)
            ax.scatter(q[:, 0], q[:, 1], s=10, c=col, alpha=0.35 if which == "E" else 0.6,
                       linewidths=0, zorder=6)
            ax.add_patch(Circle(c, rms, facecolor=col, edgecolor=col, alpha=0.15, lw=1.2,
                                zorder=4))
            ax.add_patch(Circle(c, rmax, fill=False, edgecolor=col, alpha=0.8, lw=1.2, ls="--",
                                zorder=4))
            per_tag.append(f"{tag[1]} {rms * 100:.1f}")
            lines.append(f"tag {tag[0]}:{tag[1]} ({'+'.join(sorted(cams[tag]))}) via {name}: "
                         f"{len(q)} obs, horizontal spread RMS {rms * 100:.1f} cm, "
                         f"max {rmax * 100:.1f} cm")
        if per_tag:
            ax.scatter([], [], s=40, c=col, alpha=0.7, linewidths=0,
                       label=f"tag via {name} ({expr} T_bc T_ct), RMS per tag (cm): "
                             + ", ".join(per_tag))

    for w in cfg["windows"]:
        m = (s.t >= w["from"]) & (s.t <= w["to"]) & np.isfinite(X[:, 0]) & np.isfinite(E[:, 0])
        if not m.any():
            print(f"warning: no keyframes in window {w['label']} ({w['from']}-{w['to']} s)")
            continue
        pts = np.vstack([X[m, :2], E[m, :2]])
        lo, hi = pts.min(axis=0), pts.max(axis=0)
        size = np.maximum(hi - lo + 0.24, 0.35)          # never smaller than 35 cm
        c = 0.5 * (lo + hi)
        ax.add_patch(Rectangle(c - size / 2, size[0], size[1], fill=False, ec="#1a202c", lw=1.4,
                               zorder=7))
        ax.text(c[0] - size[0] / 2, c[1] + size[1] / 2 + 0.04,
                f"{w['label']}: {w['from']:g}–{w['to']:g} s", fontsize=12, fontweight="bold",
                va="bottom", zorder=10,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.5))
    ax.set_xlabel("odom x (m)")
    ax.set_ylabel("odom y (m)")
    ax.set_title("bird's-eye view (x–y)")

    # ------------------------------------------------------------ side views
    for sx, (p, i, j) in zip(sides, planes):
        _draw_tracks(sx, s, b, setup, i, j, cfg["windows"], E, X, L, R)
        for w in cfg["windows"]:
            m = (s.t >= w["from"]) & (s.t <= w["to"]) & np.isfinite(X[:, 0])
            if m.any():
                sx.annotate(w["label"], (X[m, i][0], X[m, j][0]), xytext=(-14, 8),
                            textcoords="offset points", fontsize=12, fontweight="bold", zorder=10)
        sx.set_xlabel(f"odom {p[0]} (m)")
        sx.set_ylabel(f"odom {p[1]} (m)")
        sx.set_title(f"side view ({p[0]}–{p[1]})")

    # One legend for all the planes, in its own row above them (the planes
    # share every handle), grouped: tracks, markers, station, tag clouds.
    handles, labels = ax.get_legend_handles_labels()
    order = ("EKF", "fixed-lag smoother", "fixed-lag,", "batch LM", "direction", "keyframe",
             "tag factors from", "tag back", "window",
             "start", "end",
             "S,", "tag (", "tag via")
    rank = {lab: next((n for n, o in enumerate(order) if lab.startswith(o)), len(order))
            for lab in labels}
    pairs = sorted(zip(handles, labels), key=lambda hl: rank[hl[1]])
    lax.legend([h for h, _ in pairs], [lab for _, lab in pairs], loc="center", ncol=3,
               frameon=False, fontsize=12)

    # ------------------------------------------------------------ time panels
    def frame(t_ax, ylabel: str) -> None:
        _shade_dvl(t_ax, s.dvl_gaps, legend=t_ax is tx[0])
        _shade_windows(t_ax, cfg["windows"], None)
        t_ax.set_ylabel(ylabel)
        t_ax.grid(alpha=0.3, which="both")
        # Legend in one row above the panel, clear of the data and the labels.
        t_ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), fontsize=11, frameon=False,
                    ncol=6, borderaxespad=0.2)
        if t_ax is not tx[-1]:
            t_ax.tick_params(labelbottom=False)

    # Drift correction: 3D distance between the graph's and the EKF's AUV position.
    bx = tx[0]
    D_rt = np.linalg.norm(X - E, axis=1)
    pre = ~s.tracking
    if pre.any():
        bx.axvspan(s.t[0], s.t[pre][-1], color="#fbd38d", alpha=0.45, lw=0,
                   label="not yet TRACKING")
    bx.plot(s.t, D_rt, c=C_RT, lw=LW_T + 0.8, label="live ‖t(X_rt) − t(E_k)‖")
    ymax = np.nanmax(D_rt)
    if s.show_rp:
        D_rp = np.linalg.norm(R - E, axis=1)
        bx.plot(s.t, D_rp, c=C_RP, lw=LW_T, label="replay ‖t(X_rp) − t(E_k)‖")
        ymax = max(ymax, np.nanmax(D_rp))
    if has_lm:
        D_lm = np.linalg.norm(L - E, axis=1)
        bx.plot(s.t, D_lm, c=C_LM, lw=LW_T, label="LM ‖t(X_lm) − t(E_k)‖")
        ymax = max(ymax, np.nanmax(D_lm))
    ymax = max(float(ymax) * 1.25, 0.05)
    bx.vlines(s.t[s.tag_rp if s.show_rp else s.tag], ymax * 0.93, ymax, color=C_TICK, lw=1.2,
              label="keyframe with tag factors")
    if s.attached.any():
        bx.vlines(s.t[s.attached], ymax * 0.89, ymax * 0.925, color=C_ATT, lw=1.2,
                  label="… from the attached camera")
    for w in cfg["windows"]:
        bx.text(0.5 * (w["from"] + w["to"]), ymax * 0.86, w["label"], ha="center",
                fontsize=12, fontweight="bold")
    bx.set_ylim(0, ymax)
    frame(bx, "drift correction (m)")

    # Depth: odom z of the three estimates.
    zx = tx[1]
    zx.plot(s.t, E[:, 2], ls="--", c=C_E, lw=LW_E, label="EKF E_k")
    zx.plot(s.t, X[:, 2], c=C_RT, lw=LW_T + 0.8, label="live X_rt")
    if s.show_rp:
        zx.plot(s.t, R[:, 2], c=C_RP, lw=LW_T, label="replay X_rp")
    if has_lm:
        zx.plot(s.t, L[:, 2], c=C_LM, lw=LW_T, label="LM X_lm")
    frame(zx, "odom z (m)")

    if timing:
        cx = tx[2]

        def st(u: np.ndarray) -> str:
            u = u[np.isfinite(u)]
            return (f"median {np.median(u):.1f}, p95 {np.percentile(u, 95):.1f}, "
                    f"max {u.max():.1f} ms")
        cx.plot(s.t, s.upd_live, c="#4a5568", lw=0.9,
                label=f"{cfg['timing']['live_label']}, live ({st(s.upd_live)})")
        lines.append(f"update time, live: {st(s.upd_live)}")
        lo = np.nanmin(s.upd_live)
        if cfg["timing"]["replay"]:
            cx.plot(s.t, s.upd_replay, c=C_RT, lw=0.9,
                    label=f"replay on {_cpu_name()} ({st(s.upd_replay)})")
            lines.append(f"update time, replay on {_cpu_name()}: {st(s.upd_replay)}")
            lo = min(lo, np.nanmin(s.upd_replay))
        budget = 1000.0 * float(np.median(np.diff(s.t))) if len(s.t) > 1 else 200.0
        cx.axhline(budget, c="#c53030", lw=1.4, ls="--",
                   label=f"keyframe period {budget:.0f} ms")
        cx.set_yscale("log")
        cx.set_ylim(max(1e-3, 0.7 * lo), budget * 4)
        frame(cx, "update / keyframe (ms)")
    tx[0].set_xlim(s.t[0], s.t[-1])
    tx[-1].set_xlabel("time since recording start (s)")

    fig.suptitle(title, fontsize=16)
    fig.savefig(png, dpi=cfg["dpi"], bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {png}")


def plot_rt_vs_batch(cfg: Dict[str, Any], s: Series, b: Batch, png: str, title: str) -> None:
    """Live fixed-lag estimate minus the batch optimum, over time.

    The docking pose ``X^-1 S`` (dock_point in base_link) is what the docking
    controller uses; live it is ``X_rt^-1 S_rt`` with ``S_rt`` as known at that
    keyframe, smoothed it is ``X_lm^-1 S_lm`` with the one LM station.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _style()

    def diffs(Xs, Ss):
        """Position, docking-pose translation / yaw and station differences to LM, per keyframe."""
        n = len(s.t)
        d_pos, d_dock, d_dyaw, d_S = (np.full(n, np.nan) for _ in range(4))
        for i in range(n):
            Xr, Xl, Sr = Xs[i], s.X_lm[i], Ss[i]
            if Xr is None or Xl is None:
                continue
            d_pos[i] = 100 * np.linalg.norm(Xr.translation() - Xl.translation())
            if Sr is not None and b.S_lm is not None and s.tracking[i]:
                p, c = Xr.between(Sr), Xl.between(b.S_lm)
                d_dock[i] = 100 * np.linalg.norm(p.translation() - c.translation())
                d_dyaw[i] = math.degrees(p.rotation().between(c.rotation()).yaw())
                d_S[i] = 100 * np.linalg.norm(Sr.translation() - b.S_lm.translation())
        return d_pos, d_dock, d_dyaw, d_S

    # The real-time estimates compared with LM: the recorded live run, and
    # the replay with this YAML's settings when that is not the live run.
    sources = [("live", "rt", s.X_rt, s.S_rt, C_RT)]
    if s.show_rp:
        sources.append(("replay", "rp", s.X_rp, s.S_rp, C_RP))
    fig, (ax, bx, cx) = plt.subplots(3, 1, figsize=(16, 14), sharex=True)
    for name, sub, Xs, Ss, col in sources:
        d_pos, d_dock, d_dyaw, d_S = diffs(Xs, Ss)
        ax.plot(s.t, d_pos, c=col, lw=1.5, label=f"{name}: AUV position ‖t(X_{sub}) − t(X_lm)‖")
        ax.plot(s.t, d_dock, c=col, lw=1.5, ls="--",
                label=f"{name}: docking pose ‖t(X_{sub}⁻¹ S_{sub}) − t(X_lm⁻¹ S_lm)‖")
        bx.plot(s.t, d_dyaw, c=col, lw=1.5, label=f"{name}: docking pose yaw − LM")
        cx.plot(s.t, d_S, c=col, lw=1.5, label=f"{name}: station ‖t(S_{sub}) − t(S_lm)‖ "
                                                 "(S as known then)")
    ax.set_ylabel("real time vs batch LM (cm)")
    bx.axhline(0, c="0.5", lw=0.6)
    bx.set_ylabel("yaw (deg)")
    cx.set_ylabel("station (cm)")
    cx.set_xlabel("time since recording start (s)")
    for x in (ax, bx, cx):
        top = x.get_ylim()[1]
        x.vlines(s.t[s.tag_rp if s.show_rp else s.tag], top * 0.95, top, color=C_TICK, lw=0.8)
        if s.attached.any():
            x.vlines(s.t[s.attached], top * 0.91, top * 0.945, color=C_ATT, lw=0.8)
        for t in s.t[s.seeded]:
            x.axvline(t, color="magenta", ls=":", lw=0.8)
        _shade_dvl(x, s.dvl_gaps, legend=x is ax)
        _shade_windows(x, cfg["windows"], None)
        x.grid(alpha=0.3)
        x.legend(fontsize=11, loc="upper left")
    ax.set_xlim(s.t[0], s.t[-1])
    fig.suptitle(title + "\nyellow ticks: keyframes with tag factors; sky blue: from an attached "
                 "camera; magenta: X_k seeded from the tag", fontsize=11)
    fig.tight_layout()
    fig.savefig(png, dpi=cfg["dpi"], bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {png}")


C_DVL = "#90cdf4"         # DVL bottom lock lost (light blue band)


def _shade_dvl(ax, gaps: List[Tuple[float, float]], legend: bool = False) -> None:
    """DVL dropouts as light-blue bands (the EKF dead-reckons there)."""
    for i, (a, c) in enumerate(gaps):
        ax.axvspan(a, c, color=C_DVL, alpha=0.35, lw=0,
                   label="DVL bottom lock lost" if legend and i == 0 else None)


def _shade_windows(ax, windows: List[dict], label_y: Optional[float]) -> None:
    for w in windows:
        ax.axvspan(w["from"], w["to"], color="#1a202c", alpha=0.18, lw=0)
        if label_y is not None:
            ax.text(0.5 * (w["from"] + w["to"]), label_y, w["label"], ha="center", fontsize=10,
                    fontweight="bold")


def _cpu_name() -> str:
    import platform
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as f:
            return next(ln.split(":", 1)[1].strip() for ln in f if ln.startswith("model name"))
    except (OSError, StopIteration):
        return platform.processor() or platform.machine()


# ---------------------------------------------------------------------------- outputs


def write_csv(s: Series, path: str) -> None:
    cols = ["k", "t", "tag", "tracking"]
    names = ("E", "X_rt", "X_lm", "dock_rt")
    cols += [f"{n}_{a}" for n in names for a in ("x", "y", "z", "yaw_deg")]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for i in range(len(s.t)):
            dock = (s.X_rt[i].between(s.S_rt[i])
                    if s.X_rt[i] is not None and s.S_rt[i] is not None else None)
            row = [int(s.k[i]), f"{s.t[i]:.4f}", int(s.tag[i]), int(s.tracking[i])]
            for P in (s.E[i], s.X_rt[i], s.X_lm[i], dock):
                if P is None:
                    row += [""] * 4
                else:
                    t = P.translation()
                    row += [f"{t[0]:.5f}", f"{t[1]:.5f}", f"{t[2]:.5f}",
                            f"{math.degrees(P.rotation().yaw()):.3f}"]
            w.writerow(row)
    print(f"wrote {path}")


def summarize(s: Series, b: Batch) -> List[str]:
    n = len(s.t)
    trk = [i for i in range(n) if s.tracking[i]]
    state = ("never TRACKING" if not trk else "TRACKING throughout" if trk[0] == 0
             else f"TRACKING from {s.t[trk[0]]:.1f} s")
    lines = [f"{s.t[0]:.1f}-{s.t[-1]:.1f} s: {n} keyframes, {int(s.tag.sum())} with tag "
             f"factors, {state}"]
    # With the live run's settings and detections the replay must reproduce
    # it, so the LM graph is "the" live graph. With other settings (e.g. a
    # redetect_bag.py bag) it is a different run, drawn as its own track.
    rp = [(s.X_rt[i], b.X_replay[int(s.k[i])]) for i in range(n) if s.X_rt[i] is not None]
    lines.append(f"replay vs live X_rt ({'differs: other settings / detections' if s.show_rp else 'should be ~0'}): "
                 f"{_stats_cm_deg(rp)}")
    if b.attached:
        att = Counter(f"{c}:{t[0]}:{t[1]}" for v in b.attached.values() for c, t in v)
        lines.append(f"attached-camera tag factors: {sum(att.values())} on {len(b.attached)} "
                     f"keyframes ({', '.join(f'{k} x{n}' for k, n in sorted(att.items()))})")
    if b.X_lm:
        st = b.stats
        lines.append(f"batch LM: {st['n_factors']} factors, {len(b.X_lm)} poses, "
                     f"{st['iterations']} iterations in {st['seconds']:.1f} s; graph error "
                     f"{st['error_init']:.1f} -> {st['error_lm']:.1f}")
        pairs = [(s.X_rt[i], s.X_lm[i]) for i in trk
                 if s.X_rt[i] is not None and s.X_lm[i] is not None]
        lines.append(f"X live vs LM (tracking): {_stats_cm_deg(pairs)}")
        if s.show_rp:
            pairs = [(s.X_rp[i], s.X_lm[i]) for i in trk if s.X_lm[i] is not None]
            lines.append(f"X replay vs LM (tracking): {_stats_cm_deg(pairs)}")
        if b.S_lm is not None:
            dock = [(s.X_rt[i].between(s.S_rt[i]), s.X_lm[i].between(b.S_lm)) for i in trk
                    if s.X_rt[i] is not None and s.X_lm[i] is not None and s.S_rt[i] is not None]
            lines.append(f"docking pose X^-1 S live vs LM (tracking): {_stats_cm_deg(dock)}")
            if s.show_rp:
                dock = [(s.X_rp[i].between(s.S_rp[i]), s.X_lm[i].between(b.S_lm)) for i in trk
                        if s.X_lm[i] is not None and s.S_rp[i] is not None]
                lines.append(f"docking pose X^-1 S replay vs LM (tracking): {_stats_cm_deg(dock)}")
            for name, Ss in (("live", s.S_rt), *((("replay", s.S_rp),) if s.show_rp else ())):
                S_f = _last(Ss)
                if S_f is not None:
                    dt, dr = pose_distance(S_f, b.S_lm)
                    lines.append(f"final {name} S vs LM S: {dt * 100:.2f} cm, "
                                 f"{math.degrees(dr):.2f} deg")
    return lines


def main(argv: List[str] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("run_yaml", help="analysis parameters (see config/analyze_run_example.yaml)")
    args = ap.parse_args(argv)
    cfg = load_run_config(args.run_yaml)
    os.makedirs(cfg["out_dir"], exist_ok=True)
    write_overrides(cfg)

    get = getter_from_yaml(cfg["dock_fuser_config"])
    setup = build_setup(get, lambda m: print("warning:", m))
    print("tags: " + ", ".join(f"{f}:{i} ({t.size * 100:g} cm)"
                               for (f, i), t in setup.tags.items()))
    t0 = source_start_time(cfg["bag"])              # zero of every time axis
    shift = bag_start_time(cfg["bag"]) - t0

    topics = dict(cfg["topics"])
    if topics["dvl"] is None:
        topics["dvl"] = get("dvl_topic") or ""
    rec = read_recorded(cfg["bag"], topics)
    gaps = dvl_gaps(rec.dvl, float(get("estimator.dvl_timeout")), t0)
    b = run_batch(cfg, setup, get, shift)
    s = align(rec, b, t0, cfg["view_start"], cfg["view_end"], gaps, cfg["replay_track"])

    name = cfg["title"] or bag_source(cfg["bag"])
    span = f"{s.t[0]:.0f}–{s.t[-1]:.0f} s"
    stride = f", frame stride {cfg['frame_stride']}" if int(cfg["frame_stride"]) > 1 else ""
    what = "EKF vs fixed-lag smoother (live)" + (" vs batch LM" if b.X_lm else "")
    lines = summarize(s, b)
    lines += [f"DVL bottom lock lost {a:.1f}-{c:.1f} s ({c - a:.1f} s)" for a, c in s.dvl_gaps]

    out = cfg["out_dir"]
    if cfg["outputs"]["overview"]:
        plot_overview(cfg, s, b, setup, os.path.join(out, "overview.png"),
                      f"{name}, {span}{stride}: {what}\nfixed-lag smoother {b.lag:g} s",
                      lines)
    if cfg["outputs"]["rt_vs_batch"] and b.X_lm:
        plot_rt_vs_batch(cfg, s, b, os.path.join(out, "rt_vs_batch.png"),
                         f"{name}, {span}: real-time fixed-lag smoother ({b.lag:g} s)"
                         f"{' (live and replayed)' if s.show_rp else ''} vs batch LM")
    if cfg["outputs"]["graphs"]:
        from .plot_graph import collect, plot
        for w in cfg["windows"]:
            if not w["graph"]:
                continue
            win = collect(cfg["bag"], cfg["dock_fuser_config"], cfg["start"], w["from"], w["to"],
                          int(cfg["frame_stride"]))
            plot(win, os.path.join(out, f"graph_{w['label']}.png"),
                 f"{w['label']}: Factor graph, {name}, {w['from']:g}–{w['to']:g} s{stride}\n"
                 f"fixed-lag smoother {b.lag:g} s, real-time estimates (replay)")
    if cfg["outputs"]["csv"]:
        write_csv(s, os.path.join(out, "keyframes.csv"))

    print("\n".join(lines))
    with open(os.path.join(out, "summary.txt"), "w", encoding="utf-8") as f:
        f.write(f"{name}\nrun yaml: {os.path.abspath(args.run_yaml)}\n"
                f"dock_fuser config: {' '.join(cfg['dock_fuser_config'])}\n\n")
        f.write("\n".join(lines) + "\n")
    print(f"wrote {os.path.join(out, 'summary.txt')}")


if __name__ == "__main__":
    main()
