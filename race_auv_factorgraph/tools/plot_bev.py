"""Bird's-eye-view plot of a dock_fuser run recorded by ``launch/bag_test.launch.py``.

Usage::

    ros2 run race_auv_factorgraph plot_bev OUTPUT_BAG [--config tank_dock_fuser.yaml]
        [--png bev.png] [--time-offset 95]

The figure has two panels.

**Top: BEV (odom x-y plane, metres, equal aspect)**

* grey line: raw EKF trajectory (``/race_auv/odometry/filtered``);
* blue line: dock_fuser real-time track ``X_k`` (``/dock_fuser/auv_pose``),
  the graph's drift-corrected pose *as estimated at each keyframe*. It
  coincides with the EKF until the station is initialized, then separates
  by the estimated drift;
* markers on the blue track at every keyframe that carried a tag detection:
  green = accepted as a factor, orange = buffered for initialization (before
  TRACKING), grey = held back by the temporal consistency gate (recordings
  made before the gate was removed), red = rejected;
* magenta diamonds: keyframes whose ``X_k`` started at the tag-implied pose
  instead of the odometry prediction (after drift; see
  ``DockEstimator._seed_from_tags``). Bags recorded before the lean gate
  show relocalizations here;
* black star: AUV position at the keyframe where the station was initialized;
* tag reprojections: each detection's tag position (from PnP on its 4
  corners) mapped into odom, once through the raw EKF pose ``E(t)`` (grey
  ``x``) and once through the graph pose ``X(t)`` (blue dots). EKF drift
  spreads the grey cloud; the graph's correction should collapse the blue one
  onto the estimated tag;
* station: the final ``S`` (dock_point, with its x axis as an arrow), the
  station base_link, and each tag at ``S * T_j``, drawn to scale as a
  segment with a short arrow on the side it faces. A faint coloured trail
  shows how ``S`` evolved while tracking.

**Middle: time series**

* drift correction ``D = X_k * E_k^-1``: translation (m) and yaw (deg);
* detection raster (same colours as above) and the init time.

**With ``--batch``: batch Levenberg-Marquardt post-processing** (``tools/batch_lm.py``).
The same graph the real-time run built (same keyframes, same accepted
factors) is re-optimized all at once, so every pose also uses later tag
observations (full smoothing). It is drawn as an orange dashed track with its
station, added to the drift-correction panel (dashed), and a bottom panel
shows how far the real-time estimate was from it over time: robot position,
and the docking pose ``X^-1 * S`` that docking actually uses.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from typing import Dict, List, Tuple

import gtsam
import numpy as np

from ..core.common.geometry import pose_from_quat_xyzw
from ..core.common.model import Detection, normalize_family
from ..core.frontend.pnp import ippe_tag
from ..core.frontend.pose_buffer import PoseBuffer
from ..ros.params import build_setup, getter_from_yaml
from .replay_bag import load_camera_info

TOPIC_ODOM = "/race_auv/odometry/filtered"
TOPIC_X = "/dock_fuser/auv_pose"
TOPIC_S = "/dock_fuser/station_pose"
TOPIC_KF = "/dock_fuser/keyframe"


def _stamp(header) -> float:
    return float(header.stamp.sec) + 1e-9 * float(header.stamp.nanosec)


def _pose(p, q) -> gtsam.Pose3:
    return pose_from_quat_xyzw((p.x, p.y, p.z), (q.x, q.y, q.z, q.w))


def read_bag(path: str, tag_topics: List[str]):
    """Read everything the plot needs from the output bag."""
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message

    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=path, storage_id="mcap"),
                rosbag2_py.ConverterOptions("cdr", "cdr"))
    types = {t.name: t.type for t in reader.get_all_topics_and_types()}
    wanted = [TOPIC_ODOM, TOPIC_X, TOPIC_S, TOPIC_KF, *tag_topics]
    cls = {t: get_message(types[t]) for t in wanted if t in types}
    missing = [t for t in wanted if t not in types]
    if missing:
        print("warning: not in the bag:", ", ".join(missing))

    ekf, X, S, kfs, frames = [], [], [], [], []
    while reader.has_next():
        topic, data, _ = reader.read_next()
        if topic not in cls:
            continue
        msg = deserialize_message(data, cls[topic])
        if topic == TOPIC_ODOM:
            ekf.append((_stamp(msg.header), _pose(msg.pose.pose.position, msg.pose.pose.orientation)))
        elif topic == TOPIC_X:
            X.append((_stamp(msg.header), _pose(msg.pose.position, msg.pose.orientation)))
        elif topic == TOPIC_S:
            S.append((_stamp(msg.header), _pose(msg.pose.position, msg.pose.orientation)))
        elif topic == TOPIC_KF:
            kfs.append(json.loads(msg.data))
        else:
            frames.append((topic, _stamp(msg.header), msg))
    for lst in (ekf, X, S):
        lst.sort(key=lambda e: e[0])
    return ekf, X, S, kfs, frames


def main(argv: List[str] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("bag", help="output bag of bag_test.launch.py")
    ap.add_argument("--config", default=[], nargs="+",
                    help="dock_fuser parameter file(s) used for the run, merged in order "
                         "(default: the installed tank_dock_fuser.yaml)")
    ap.add_argument("--png", default="", help="output image (default: <bag>/bev.png)")
    ap.add_argument("--time-offset", type=float, default=0.0,
                    help="added to the time axis, e.g. the start_offset of the replay, so times "
                         "read as seconds into the original bag")
    ap.add_argument("--batch", action="store_true",
                    help="also run batch LM post-processing on the same data and overlay it")
    args = ap.parse_args(argv)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    config = args.config
    if not config:
        from ament_index_python.packages import get_package_share_directory
        config = os.path.join(get_package_share_directory("race_auv_factorgraph"),
                              "config", "tank_dock_fuser.yaml")
    setup = build_setup(getter_from_yaml(config), lambda m: print("warning:", m))
    load_camera_info(args.bag, setup)
    cam_by_topic = {topic: cam for cam, topic in setup.camera_topics.items()}

    ekf, X, S, kfs, frames = read_bag(args.bag, list(cam_by_topic))
    if not ekf:
        raise SystemExit("no EKF odometry in the bag")
    t0 = ekf[0][0]
    rel = lambda t: t - t0 + args.time_offset  # noqa: E731

    E_buf = PoseBuffer(max_age=1e9)
    for t, E in ekf:
        E_buf.add(t, E)
    X_by_t = {round(t, 6): p for t, p in X}
    X_buf = PoseBuffer(max_age=1e9)
    for t, p in X:
        X_buf.add(t, p)

    def X_at(t: float):
        """Graph pose at a keyframe stamp (exact), else interpolated on the X track."""
        p = X_by_t.get(round(t, 6))
        return p if p is not None else X_buf.interpolate(t)

    # ---------------------------------------------------------------- per-detection reprojection
    raw_pts, graph_pts = [], []
    for topic, t, msg in frames:
        cam = setup.cameras[cam_by_topic[topic]]
        E, Xt = E_buf.interpolate(t), X_at(t)
        for d in msg.detections:
            key = (normalize_family(d.family), int(d.id))
            tag = setup.tags.get(key)
            if tag is None or E is None:
                continue
            det = Detection(cam.name, key, np.array([[c.x, c.y] for c in d.corners], dtype=float))
            sols = ippe_tag(det, cam, tag)
            if not sols:
                continue
            T_ct = sols[0][0]
            raw_pts.append(E.compose(cam.T_bc).compose(T_ct).translation())
            if Xt is not None:
                graph_pts.append(Xt.compose(cam.T_bc).compose(T_ct).translation())
    raw_pts, graph_pts = np.array(raw_pts).reshape(-1, 3), np.array(graph_pts).reshape(-1, 3)

    # ---------------------------------------------------------------- keyframe categories
    cats: Dict[str, List[Tuple[float, gtsam.Pose3]]] = {
        "accepted": [], "buffered": [], "held": [], "rejected": []}
    init_kf = None
    reloc: List[Tuple[float, gtsam.Pose3]] = []
    for kf in kfs:
        Xk = X_at(kf["stamp"])
        if Xk is None:
            continue
        if kf.get("initialized_now"):
            init_kf = (kf["stamp"], Xk)
        if kf.get("seeded_from_tag", kf.get("relocalized")):
            reloc.append((kf["stamp"], Xk))
        # Odometry-only keyframe (detector message without tags). "n_tags" is
        # missing in recordings made before every detector message became a
        # keyframe; there, a keyframe without camera was a tag-free tick.
        if not kf.get("camera") or kf.get("n_tags", 1) == 0:
            continue
        if kf["accepted"]:
            cats["accepted"].append((kf["stamp"], Xk))
        elif kf["rejected"]:
            cats["rejected"].append((kf["stamp"], Xk))
        elif kf.get("held"):
            cats["held"].append((kf["stamp"], Xk))
        else:
            cats["buffered"].append((kf["stamp"], Xk))
    colors = {"accepted": "tab:green", "buffered": "tab:orange", "held": "0.6",
              "rejected": "tab:red"}

    # ---------------------------------------------------------------- batch LM (optional)
    sol, LM_buf, lm_pts = None, None, np.zeros((0, 3))
    if args.batch:
        from .batch_lm import batch_solve
        print("running batch LM post-processing (replays the bag)...")
        sol = batch_solve(args.bag, config)
        LM_buf = PoseBuffer(max_age=1e9)
        for k, P in sol.X_lm.items():
            LM_buf.add(sol.stamps[k], P)
        # Tag positions via the smoothed poses, for the spread statistic.
        pts = []
        for topic, t, msg in frames:
            cam = setup.cameras[cam_by_topic[topic]]
            Xl = LM_buf.interpolate(t)
            for d in msg.detections:
                key = (normalize_family(d.family), int(d.id))
                if Xl is None or key not in setup.tags:
                    continue
                det = Detection(cam.name, key, np.array([[c.x, c.y] for c in d.corners], dtype=float))
                sols = ippe_tag(det, cam, setup.tags[key])
                if sols:
                    pts.append(Xl.compose(cam.T_bc).compose(sols[0][0]).translation())
        lm_pts = np.array(pts).reshape(-1, 3)

    # ---------------------------------------------------------------- figure
    if sol is None:
        fig = plt.figure(figsize=(11, 12))
        gs = fig.add_gridspec(2, 1, height_ratios=[3.2, 1.0], hspace=0.18)
    else:
        fig = plt.figure(figsize=(11, 15))
        gs = fig.add_gridspec(3, 1, height_ratios=[3.2, 1.0, 1.0], hspace=0.22)
    ax = fig.add_subplot(gs[0])
    exy = np.array([E.translation()[:2] for _, E in ekf])
    ax.plot(exy[:, 0], exy[:, 1], color="0.55", lw=1.2, label="EKF (raw)")
    if X:
        xxy = np.array([p.translation()[:2] for _, p in X])
        ax.plot(xxy[:, 0], xxy[:, 1], color="tab:blue", lw=1.4, label="dock_fuser X_k (real time)")
    if sol is not None:
        lxy = np.array([sol.X_lm[k].translation()[:2] for k in sorted(sol.X_lm)])
        ax.plot(lxy[:, 0], lxy[:, 1], color="darkorange", lw=1.4, ls="--", zorder=4,
                label="batch LM (post-processing)")
    ax.plot(*exy[0], "o", color="0.3", ms=6)
    ax.annotate("start", exy[0], textcoords="offset points", xytext=(5, 5), fontsize=8)
    for name, pts in cats.items():
        if pts:
            p = np.array([P.translation()[:2] for _, P in pts])
            ax.scatter(p[:, 0], p[:, 1], s=12, color=colors[name], zorder=4,
                       label=f"tag frame: {name} ({len(pts)})")
    if reloc:
        p = np.array([P.translation()[:2] for _, P in reloc])
        ax.scatter(p[:, 0], p[:, 1], s=60, marker="D", facecolors="none", edgecolors="magenta",
                   lw=1.5, zorder=5, label=f"X_k seeded from tag ({len(reloc)})")
    if len(raw_pts):
        ax.scatter(raw_pts[:, 0], raw_pts[:, 1], marker="x", s=10, lw=0.8, color="0.45",
                   label="tag via EKF pose  E(t) T_bc T_ct", zorder=3)
    if len(graph_pts):
        ax.scatter(graph_pts[:, 0], graph_pts[:, 1], s=6, color="tab:blue", alpha=0.6,
                   label="tag via graph pose  X(t) T_bc T_ct", zorder=3)

    if S:
        sxy = np.array([p.translation()[:2] for _, p in S])
        ax.scatter(sxy[:, 0], sxy[:, 1], c=[rel(t) for t, _ in S], cmap="viridis", s=4,
                   alpha=0.5, zorder=2)
        Sf = S[-1][1]
        d = Sf.translation()
        ax.plot(d[0], d[1], "s", color="k", ms=7, zorder=6, label="dock_point (final S)")
        xa = Sf.rotation().matrix()[:, 0]
        ax.arrow(d[0], d[1], 0.3 * xa[0], 0.3 * xa[1], width=0.01, color="k", zorder=6)
        base = Sf.compose(setup.T_dock_base).translation()
        ax.plot(base[0], base[1], "D", color="purple", ms=6, zorder=6, label="station base_link")
        for key, tag in setup.tags.items():
            W = Sf.compose(tag.T_dock_tag)
            c = W.translation()
            ex = W.rotation().matrix()[:, 0] * tag.size / 2
            out = -W.rotation().matrix()[:, 2]          # tag z points INTO the tag
            ax.plot([c[0] - ex[0], c[0] + ex[0]], [c[1] - ex[1], c[1] + ex[1]],
                    color="crimson", lw=3, zorder=7)
            ax.arrow(c[0], c[1], 0.2 * out[0], 0.2 * out[1], width=0.006, color="crimson", zorder=7)
            ax.annotate(f"tag {key[1]}", c[:2], textcoords="offset points", xytext=(6, -12),
                        fontsize=8, color="crimson")
    if sol is not None and sol.S_lm is not None:
        d = sol.S_lm.translation()
        ax.plot(d[0], d[1], "s", mfc="none", mec="darkorange", mew=2, ms=11, zorder=7,
                label="dock_point (batch LM)")
        for key, tag in setup.tags.items():
            c = sol.S_lm.compose(tag.T_dock_tag).translation()
            ax.plot(c[0], c[1], "o", mfc="none", mec="darkorange", mew=1.5, ms=9, zorder=8)
    if init_kf is not None:
        p = init_kf[1].translation()
        ax.plot(p[0], p[1], "*", color="k", ms=16, zorder=8,
                label=f"station init (t = {rel(init_kf[0]):.1f} s)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("odom x [m]")
    ax.set_ylabel("odom y [m]")
    what = "iSAM2 real time" + (" vs batch LM" if sol is not None else "")
    ax.set_title(f"BEV: EKF vs dock_fuser ({what})\n{os.path.basename(os.path.normpath(args.bag))}",
                 fontsize=10)
    ax.legend(fontsize=7, loc="best")

    # ---------------------------------------------------------------- time series
    ax2 = fig.add_subplot(gs[1])
    tD, dtr, dyaw = [], [], []
    for t, p in X:
        E = E_buf.interpolate(t)
        if E is None:
            continue
        D = p.compose(E.inverse())
        tD.append(rel(t))
        dtr.append(float(np.linalg.norm(D.translation()[:2])))
        dyaw.append(math.degrees(D.rotation().yaw()))
    ax2.plot(tD, dtr, color="tab:blue", label="|D| xy [m]")
    ax2.set_ylabel("drift correction xy [m]", color="tab:blue")
    ax2.set_xlabel("time [s]" + (" (into the original bag)" if args.time_offset else ""))
    ax3 = ax2.twinx()
    ax3.plot(tD, dyaw, color="tab:purple", lw=1.0, label="D yaw [deg]")
    ax3.set_ylabel("drift correction yaw [deg]", color="tab:purple")
    ymax_extra: List[float] = []
    if sol is not None:
        tl, ltr, lyaw = [], [], []
        for k in sorted(sol.X_lm):
            D = sol.X_lm[k].compose(sol.E[k].inverse())
            tl.append(rel(sol.stamps[k]))
            ltr.append(float(np.linalg.norm(D.translation()[:2])))
            lyaw.append(math.degrees(D.rotation().yaw()))
        ax2.plot(tl, ltr, color="darkorange", ls="--", lw=1.2, label="|D| xy, batch LM")
        ax3.plot(tl, lyaw, color="plum", ls="--", lw=1.0, label="D yaw, batch LM")
        ymax_extra = ltr
        ax2.legend(fontsize=7, loc="upper left")
    ymax = max(dtr + ymax_extra + [0.05])
    for i, (name, pts) in enumerate(cats.items()):
        ts = [rel(t) for t, _ in pts]
        ax2.vlines(ts, ymax * (1.02 + 0.06 * i), ymax * (1.06 + 0.06 * i), color=colors[name], lw=0.8)
    for t, _ in reloc:
        ax2.axvline(rel(t), color="magenta", ls=":", lw=1)
    if init_kf is not None:
        ax2.axvline(rel(init_kf[0]), color="k", ls="--", lw=1)
        ax2.annotate("init", (rel(init_kf[0]), ymax * 0.9), textcoords="offset points",
                     xytext=(4, 0), fontsize=8)
    ax2.set_ylim(0, ymax * 1.25)
    ax2.grid(True, alpha=0.3)

    # ---------------------------------------------------------------- real time vs batch
    if sol is not None:
        ax4 = fig.add_subplot(gs[2], sharex=ax2)
        S_by_t = {round(t, 6): P for t, P in S}
        tt, dpos, ddock = [], [], []
        for t, P in X:
            Xl = LM_buf.interpolate(t)
            if Xl is None:
                continue
            tt.append(rel(t))
            dpos.append(100 * float(np.linalg.norm(P.translation()[:2] - Xl.translation()[:2])))
            St = S_by_t.get(round(t, 6))
            ddock.append(np.nan if St is None or sol.S_lm is None else 100 * float(
                np.linalg.norm(P.between(St).translation() - Xl.between(sol.S_lm).translation())))
        ax4.plot(tt, dpos, color="tab:blue", lw=1.0, label="robot position |X_rt - X_LM| (xy)")
        ax4.plot(tt, ddock, color="crimson", lw=1.0, label="docking pose |X_rt^-1 S_rt - X_LM^-1 S_LM|")
        for t, _ in reloc:
            ax4.axvline(rel(t), color="magenta", ls=":", lw=1)
        if init_kf is not None:
            ax4.axvline(rel(init_kf[0]), color="k", ls="--", lw=1)
        ax4.set_ylabel("real time vs batch [cm]")
        ax4.set_xlabel(ax2.get_xlabel())
        ax4.grid(True, alpha=0.3)
        ax4.legend(fontsize=7, loc="upper left")

    png = args.png or os.path.join(args.bag, "bev.png")
    fig.savefig(png, dpi=150, bbox_inches="tight")
    print(f"wrote {png}")

    # ---------------------------------------------------------------- summary
    n_frames = sum(len(v) for v in cats.values())
    print(f"window: {rel(ekf[0][0]):.1f}-{rel(ekf[-1][0]):.1f} s, {len(kfs)} keyframes, "
          f"{n_frames} tag keyframes ({', '.join(f'{k} {len(v)}' for k, v in cats.items())})")
    print(f"tag detections in bag: {sum(len(m.detections) for _, _, m in frames)} "
          f"in {sum(1 for _, _, m in frames if m.detections)} non-empty messages")
    if init_kf is not None:
        print(f"station initialized at t = {rel(init_kf[0]):.2f} s")
    if reloc:
        print("X_k seeded from tag at t = " + ", ".join(f"{rel(t):.1f}" for t, _ in reloc) + " s")
    if tD:
        print(f"final drift correction: {dtr[-1]:.3f} m xy, {dyaw[-1]:.2f} deg yaw")
    if init_kf is not None and len(raw_pts):
        spread = lambda P: float(np.sqrt(np.mean(np.sum((P[:, :2] - P[:, :2].mean(0)) ** 2, 1))))  # noqa: E731
        print(f"tag reprojection spread (RMS about the mean, xy): via EKF {spread(raw_pts):.3f} m, "
              f"via graph (real time) {spread(graph_pts):.3f} m"
              + (f", via batch LM {spread(lm_pts):.3f} m" if len(lm_pts) else ""))
    if sol is not None:
        from .batch_lm import summarize
        print("\n".join(summarize(sol)))


if __name__ == "__main__":
    main()
