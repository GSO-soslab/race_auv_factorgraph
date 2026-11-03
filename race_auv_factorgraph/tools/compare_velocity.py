"""Compare body-frame velocities: raw DVL, the EKF, and the factor graph (batch LM).

Usage::

    ros2 run race_auv_factorgraph compare_velocity BAG \\
        --config station_tags.yaml race_auv_factorgraph.yaml --start 200 --end 574 \\
        --window 365.5 370.5 --window 474.5 481 --zoom 360 380 --out velocity.png

All times are seconds into the original recording, as in ``plot_graph``. The
bag is replayed from ``--start`` to ``--end`` through the same scheduler and
estimator as the node (``replay_bag.replay``), and the graph it built is
re-solved once with Levenberg-Marquardt (``batch_lm.optimize_full_graph``),
exactly as ``plot_graph --batch`` does. The same ``--start`` therefore gives
the same LM track as the ``batch_*`` figures.

Frame: everything in ``base_link``
----------------------------------
Every trace is the linear velocity of the ``base_link`` origin, expressed in
``base_link`` (body frame, x forward, y left, z up). Comparing in body frame,
not in odom, avoids picking a yaw: the EKF's yaw and the graph's yaw differ
(the graph corrects it), and rotating body velocities into odom with either
one would mix a heading difference into the velocity difference.

The four traces
---------------
**DVL** (``dvl_topic``, ~5 Hz, only while it has bottom lock). The message is
stamped in the ``dvl`` frame and measures the velocity of the DVL's own
location P. For a rigid body, the velocity of the base_link origin O is

    v_O = v_P - omega x r_OP,     v_P = R_bd v_d

with ``R_bd``, ``r_OP`` (= ``r_bd``) the rotation and position of ``dvl`` in
``base_link`` from the AUV URDF (roll = pi: y and z flip; r = (-0.038, 0,
-0.138) m), and ``omega`` the body angular rate, taken from the EKF twist
interpolated to the DVL stamp. At 0.3 rad/s of yaw the lever-arm term is
about 1 cm/s in y.

**EKF twist** (``odom_topic``, 20 Hz). ``nav_msgs/Odometry`` gives the pose
in ``header.frame_id`` (odom) but the twist in ``child_frame_id``
(base_link), so ``twist.linear`` is used as recorded. It is the EKF's
velocity *state*, which the DVL updates directly.

**EKF delta-pose** and **batch LM delta-pose** (one per keyframe interval).
The graph has no velocity variable (``x<k>`` are poses only), so its velocity
is the finite difference of consecutive poses on SE(3):

    xi_k = [omega; v] = Log(X_{k-n}^-1 X_k) / (t_k - t_{k-n})

``X_{k-n}^-1 X_k`` is the motion from keyframe k-n to k *expressed in the
body frame at k-n*: the odom frame cancels, so the result is a body-frame
quantity like the DVL's. ``Log`` (gtsam ``Pose3.Logmap``, rotation first)
returns the constant body twist whose exponential produces that motion, so
``v`` is the average body velocity over the interval under a constant-twist
assumption. It is plotted at the interval midpoint. The same formula on the
EKF poses ``E_k`` gives the EKF delta-pose. ``n`` is ``--span`` (default 1:
consecutive keyframes, 0.2 s at 5 Hz); larger spans average out tag jitter.

What the comparison shows
-------------------------
The graph's odometry factor *measures* ``E_{k-1}^-1 E_k``
(``core/factors/odometry.py``), so the LM delta-pose is the EKF delta-pose
plus whatever the tag factors pull on top of it: LM minus EKF delta-pose is
the rate of change of the drift correction. It is not an independent
velocity estimate; a graph velocity independent of the EKF would need
velocity states with DVL factors (or DVL/IMU preintegration) in the graph.
EKF twist vs EKF delta-pose shows where the EKF's position and velocity
disagree, e.g. the position snap on a DVL re-lock (a spike in the
delta-pose, none in the twist). The same happens in z on every depth
update: robot_localization fuses the pressure sensor's absolute z
(``odom0``), which steps the z position, so both delta-pose traces spike in
``v_z`` where the DVL and the twist do not. The residual panel is therefore
horizontal only.

Tank run 2026-10-22 20:46:55, 200-574 s (no DVL dropout in it): EKF
delta-pose vs DVL 0.08 cm/s median. With no position input, the EKF's x-y
position is the integrated DVL velocity, lever arm included: the residual is
smallest with ``-omega x r`` (y RMS 0.10 cm/s, against 0.37 without the
lever arm and 0.71 with ``+omega x r``), which confirms the sign above. The
EKF twist trails the DVL by one EKF cycle (20 Hz): evaluated 0.05 s after
the DVL stamp its residual drops from 0.33 to 0.10 cm/s median.

The bottom panel gives the horizontal residual of each trace against the DVL
(at the DVL stamps, the others linearly interpolated there; no residual while
the DVL has no bottom lock), plus LM minus EKF delta-pose at the keyframes.
Median and RMS over the drawn range are in the legend and printed.

``--zoom FROM TO`` (repeatable) writes one more figure per span,
``<out>_zoom_<FROM>-<TO>.png``, with the samples marked and the delta-pose
traces drawn as steps (one constant velocity per keyframe interval).
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
from ..core.common.urdf import Urdf
from ..ros.msgs import stamp_to_sec
from ..ros.params import build_setup, getter_from_yaml, resolve_urdf
from .analyze_run import C_DVL, dvl_gaps
from .bag import bag_source, open_reader, source_start_time
from .batch_lm import optimize_full_graph
from .replay_bag import replay

# Okabe-Ito colours (colour-blind safe); EKF green and LM vermillion as in analyze_run.
C_DVL_V = "#3a3a3a"       # raw DVL: dark grey, drawn underneath
C_TWIST = "#009E73"       # EKF twist: bluish green
C_EKF_D = "#0072B2"       # EKF delta-pose: blue
C_LM_D = "#D55E00"        # batch LM delta-pose: vermillion
C_LM_EKF = "#CC79A7"      # LM minus EKF delta-pose: reddish purple
C_WIN = "#1a202c"         # --window bands (grey), as in plot_graph

AXES = ("x", "y", "z")


# ---------------------------------------------------------------------------- legend math
# The derivation of every trace, as mathtext (matplotlib's built-in LaTeX
# subset, no TeX install needed). Used for the legend labels and the header.

def _lbl_dvl() -> str:
    return (r"DVL: $\mathbf{v} = R_{bd}\,\mathbf{v}_{d} - \boldsymbol{\omega}\times\mathbf{r}_{bd}$"
            r"   (5 Hz, bottom lock)")


def _lbl_twist() -> str:
    # Message field names stay outside $...$: mathtext spaces out the dot.
    return r"EKF twist: $\mathbf{v}$ = twist.linear   (20 Hz, child frame base_link)"


def _lbl_delta(P: str, n: int) -> str:
    """``[omega; v] = Log(P_{k-n}^-1 P_k) / dt`` for pose symbol ``P`` and span ``n``."""
    km = "k-1" if n == 1 else f"k-{n}"
    return (rf"$[\boldsymbol{{\omega}};\,\mathbf{{v}}] = "
            rf"\operatorname{{Log}}\left({P}_{{{km}}}^{{-1}}{P}_{{k}}\right)\,/\,\Delta t_{{k}}$")


# Superscript of each residual's trace, as mathtext (inside \mathrm{}).
RES_TEX = {"EKF twist": r"EKF\,twist", "EKF Δpose": r"EKF\,\Delta", "LM Δpose": r"LM\,\Delta"}


def _lbl_res(name: str, med: float, rms: float) -> str:
    """``||v_xy^name - v_xy^DVL||, median a, RMS b cm/s``."""
    return (rf"$\Vert\mathbf{{v}}^{{\mathrm{{{name}}}}}_{{xy}} - \mathbf{{v}}^{{\mathrm{{DVL}}}}_{{xy}}\Vert$"
            f"   median {med:.1f}, RMS {rms:.1f} cm/s")


# ---------------------------------------------------------------------------- data

@dataclass
class Trace:
    """One velocity source: sample times (s into the recording) and body-frame velocities."""

    t: np.ndarray                         # (n,)
    v: np.ndarray                         # (n, 3) linear velocity, base_link, m/s
    w: Optional[np.ndarray] = None        # (n, 3) angular velocity, base_link, rad/s
    max_gap: float = 1.0                  # break the drawn line where samples are further apart (s)


@dataclass
class Velocities:
    dvl: Trace
    twist: Trace
    ekf_d: Trace
    lm_d: Trace
    gaps: List[Tuple[float, float]] = field(default_factory=list)   # DVL bottom lock lost
    T_bd: gtsam.Pose3 = field(default_factory=gtsam.Pose3)          # dvl in base_link
    span: int = 1


def body_twist(P0: gtsam.Pose3, P1: gtsam.Pose3, dt: float) -> Tuple[np.ndarray, np.ndarray]:
    """Constant body twist ``(omega, v)`` that carries ``P0`` to ``P1`` in ``dt`` seconds.

    ``P0^-1 P1`` is the relative motion in the body frame of ``P0``. Its
    SE(3) logarithm ``xi = [omega; v]`` (gtsam: rotation first) satisfies
    ``P1 = P0 Exp(xi)``; dividing by ``dt`` turns it into the twist that,
    held constant over ``dt``, produces that motion. ``v`` is therefore a
    body-frame linear velocity, comparable with the DVL's.
    """
    xi = gtsam.Pose3.Logmap(P0.between(P1)) / dt
    return xi[:3], xi[3:]


def delta_pose_velocity(stamps: np.ndarray, poses: List[gtsam.Pose3], span: int,
                        max_dt: float) -> Trace:
    """Finite-difference body velocity of a pose track, one sample per ``span``-keyframe interval.

    Sample ``k`` is :func:`body_twist` of poses ``k-span`` and ``k``, stamped
    at the interval midpoint. Intervals longer than ``max_dt`` (the detector
    stopped) are skipped; the drawn line breaks there.
    """
    t, v, w = [], [], []
    for i in range(span, len(stamps)):
        dt = stamps[i] - stamps[i - span]
        if dt <= 0.0 or dt > max_dt:
            continue
        wi, vi = body_twist(poses[i - span], poses[i], dt)
        t.append(0.5 * (stamps[i] + stamps[i - span]))
        v.append(vi)
        w.append(wi)
    return Trace(np.array(t), np.array(v).reshape(-1, 3), np.array(w).reshape(-1, 3),
                 max_gap=1.5 * max_dt)


def interp_rows(t_src: np.ndarray, y_src: np.ndarray, t_dst: np.ndarray,
                max_gap: float) -> np.ndarray:
    """Linear interpolation of every column of ``y_src`` at ``t_dst``.

    NaN where ``t_dst`` lies outside ``t_src`` or between two samples more
    than ``max_gap`` apart, so a dropout is never bridged by a straight line.
    """
    out = np.full((len(t_dst), y_src.shape[1]), np.nan)
    if len(t_src) < 2:
        return out
    j = np.searchsorted(t_src, t_dst)
    ok = (j > 0) & (j < len(t_src))
    ok[ok] &= (t_src[j[ok]] - t_src[j[ok] - 1]) <= max_gap
    for c in range(y_src.shape[1]):
        out[ok, c] = np.interp(t_dst[ok], t_src, y_src[:, c])
    return out


def dvl_in_base(v_d: np.ndarray, w_b: np.ndarray, T_bd: gtsam.Pose3) -> np.ndarray:
    """DVL velocities (n x 3, ``dvl`` frame) -> velocity of the base_link origin, in base_link.

    ``R_bd v_d`` is the velocity of the DVL's location P, expressed in
    base_link. Rigid body: ``v_P = v_O + omega x r_OP``, so the base_link
    origin O moves with ``v_O = R_bd v_d - omega x r_bd``. ``w_b`` (n x 3)
    is the body angular rate at each DVL stamp; NaN rows (no rate known)
    fall back to no lever-arm term.
    """
    R = T_bd.rotation().matrix()
    r = np.asarray(T_bd.translation())
    w = np.nan_to_num(w_b)
    return v_d @ R.T - np.cross(w, r)


def read_recorded(bag: str, odom_topic: str, dvl_topic: str, t_bag0: float,
                  t0: float, t1: float):
    """EKF twist and raw DVL from the bag, header stamps in ``[t0, t1]`` (s into the recording).

    Returns ``(t_ekf, v_ekf, w_ekf, t_dvl, v_dvl, dvl_frame)``; the DVL
    velocity is still in its own frame (``dvl_frame``, from the header).
    """
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message

    reader = open_reader(bag)
    types = {t.name: t.type for t in reader.get_all_topics_and_types()}
    for needed in (odom_topic, dvl_topic):
        if needed not in types:
            raise RuntimeError(f"topic {needed} is not in the bag")
    cls = {t: get_message(types[t]) for t in (odom_topic, dvl_topic)}
    reader.set_filter(rosbag2_py.StorageFilter(topics=[odom_topic, dvl_topic]))
    ekf, dvl, frame = [], [], ""
    while reader.has_next():
        topic, data, _ = reader.read_next()
        msg = deserialize_message(data, cls[topic])
        t = stamp_to_sec(msg.header.stamp) - t_bag0
        if not t0 <= t <= t1:
            continue
        lin = msg.twist.twist.linear
        if topic == odom_topic:
            ang = msg.twist.twist.angular
            ekf.append((t, lin.x, lin.y, lin.z, ang.x, ang.y, ang.z))
        else:
            dvl.append((t, lin.x, lin.y, lin.z))
            frame = msg.header.frame_id
    ekf = np.array(sorted(ekf)).reshape(-1, 7)
    dvl = np.array(sorted(dvl)).reshape(-1, 4)
    return ekf[:, 0], ekf[:, 1:4], ekf[:, 4:7], dvl[:, 0], dvl[:, 1:4], frame


def collect(bag: str, config, start: Optional[float], end: float, span: int,
            frame_stride: int = 1, max_dt: float = 1.0) -> Velocities:
    """Replay, batch-solve, read the recorded velocities, and bring all four into base_link."""
    get = getter_from_yaml(config)
    setup = build_setup(get, lambda m: None)
    # Covariance does not change the estimates; skipping it is faster.
    cfg = dataclasses.replace(setup.cfg, covariance_mode="none")
    t_bag0 = source_start_time(bag)
    stamps: Dict[int, float] = {}
    kf_E: Dict[int, gtsam.Pose3] = {}
    kf_X: Dict[int, gtsam.Pose3] = {}

    def on_kf(est, r, held) -> None:
        stamps[r.k] = r.stamp - t_bag0
        kf_E[r.k] = r.E
        kf_X[r.k] = r.X

    est, _ = replay(bag, get, setup, cfg=cfg, on_keyframe=on_kf, start=start, end=end,
                    frame_stride=frame_stride)
    if est.n_recoveries:
        print(f"warning: {est.n_recoveries} backend rebuilds during the replay; the batch graph "
              "only covers the part after the last one")
    result, st = optimize_full_graph(est, kf_X)
    print(f"batch LM: {st['n_factors']} factors, {len(kf_X)} poses, {st['iterations']} iterations "
          f"in {st['seconds']:.1f} s; graph error real-time {st['error_init']:.1f} -> "
          f"LM {st['error_lm']:.1f}")

    ks = sorted(k for k in stamps if result.exists(K.X(k)))
    t_kf = np.array([stamps[k] for k in ks])
    ekf_d = delta_pose_velocity(t_kf, [kf_E[k] for k in ks], span, span * max_dt)
    lm_d = delta_pose_velocity(t_kf, [result.atPose3(K.X(k)) for k in ks], span, span * max_dt)

    t0 = t_kf[0] if start is None else start
    t_e, v_e, w_e, t_d, v_d, frame = read_recorded(bag, get("odom_topic"), get("dvl_topic"),
                                                   t_bag0, t0, end)
    # DVL frame -> base_link from the AUV URDF. The header carries the
    # namespaced TF name ("race_auv/dvl"); the URDF link is the last part.
    urdf = Urdf.from_file(resolve_urdf(get, "auv"))
    T_bd = urdf.transform("base_link", frame.rsplit("/", 1)[-1])
    timeout = float(get("estimator.dvl_timeout"))
    w_at_dvl = interp_rows(t_e, w_e, t_d, max_gap=0.2)
    dvl = Trace(t_d, dvl_in_base(v_d, w_at_dvl, T_bd), max_gap=timeout)
    twist = Trace(t_e, v_e, w_e, max_gap=0.2)
    # dvl_gaps takes absolute stamps and an origin; ours are already relative.
    gaps = dvl_gaps(list(t_d), timeout, 0.0)
    return Velocities(dvl, twist, ekf_d, lm_d, gaps, T_bd, span)


# ---------------------------------------------------------------------------- residuals

def _xy_norm_cm(d: np.ndarray) -> np.ndarray:
    return 100.0 * np.linalg.norm(d[:, :2], axis=1)


def residuals(vel: Velocities) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Horizontal residuals (cm/s): each trace minus the DVL at the DVL stamps, and LM minus EKF delta-pose.

    The other traces are interpolated to the DVL stamps (never across their
    own gaps); the delta-pose ones are interval averages, so a fast change
    also shows up as a residual of up to half the change.
    """
    td = vel.dvl.t
    out = {}
    for name, tr in (("EKF twist", vel.twist), ("EKF Δpose", vel.ekf_d), ("LM Δpose", vel.lm_d)):
        out[name] = (td, _xy_norm_cm(interp_rows(tr.t, tr.v, td, tr.max_gap) - vel.dvl.v))
    # Same keyframes, same midpoints: no interpolation needed.
    out["LM − EKF Δpose"] = (vel.lm_d.t, _xy_norm_cm(vel.lm_d.v - vel.ekf_d.v))
    return out


def _med_rms(y: np.ndarray, t: np.ndarray, lo: float, hi: float) -> Tuple[float, float]:
    m = np.isfinite(y) & (t >= lo) & (t <= hi)
    if not m.any():
        return float("nan"), float("nan")
    return float(np.median(y[m])), float(np.sqrt(np.mean(y[m] ** 2)))


# ---------------------------------------------------------------------------- plot

def _broken(t: np.ndarray, y: np.ndarray, max_gap: float) -> Tuple[np.ndarray, np.ndarray]:
    """Insert a NaN wherever consecutive samples are more than ``max_gap`` apart (breaks the line)."""
    if len(t) < 2:
        return t, y
    cut = np.flatnonzero(np.diff(t) > max_gap) + 1
    return np.insert(t, cut, np.nan), np.insert(y.astype(float), cut, np.nan, axis=0)


def _header(vel: Velocities) -> str:
    """The derivation block under the title: frame convention and every formula, in mathtext."""
    r = np.asarray(vel.T_bd.translation())
    rpy = np.degrees(vel.T_bd.rotation().rpy())
    rpy[np.abs(rpy) < 0.5] = 0.0          # no "-0°"
    km = "k-1" if vel.span == 1 else f"k-{vel.span}"
    return "\n".join([
        r"Every trace: linear velocity of the base_link origin, expressed in base_link "
        r"(body frame: $x$ forward, $y$ left, $z$ up).",
        rf"DVL: $R_{{bd}}$ = URDF rpy ({rpy[0]:.0f}°, {rpy[1]:.0f}°, {rpy[2]:.0f}°), "
        rf"$\mathbf{{r}}_{{bd}}$ = ({r[0]:.3f}, {r[1]:.3f}, {r[2]:.3f}) m (dvl in base_link); "
        r"$\boldsymbol{\omega}$ = EKF twist.angular at the DVL stamp.",
        rf"Δpose: $P_{{{km}}}^{{-1}}P_k$ is the motion over the interval in the body frame at "
        rf"${km}$ (the odom frame cancels); $\operatorname{{Log}}$ gives the constant body twist "
        r"producing it; plotted at the midpoint $\frac{1}{2}(t_{k-n}+t_k)$.",
        r"The odometry factor measures $E_{k-1}^{-1}E_k$, so LM − EKF Δpose is the velocity "
        r"the tag factors add on top of the EKF (rate of change of the drift correction).",
    ])


def plot(vel: Velocities, png: str, title: str, lo: float, hi: float,
         windows: List[Tuple[float, float]], zoom: bool = False) -> None:
    """Three velocity panels (body x, y, z) and the horizontal residual panel over ``[lo, hi]``."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(4, 1, figsize=(14, 15), sharex=True,
                            gridspec_kw={"height_ratios": [1, 1, 1, 1.1]})
    fig.suptitle(title, fontsize=12, y=0.995)
    fig.text(0.06, 0.962, _header(vel), fontsize=9, va="top", ha="left", linespacing=1.6)

    # In the zoom the samples are marked and the delta-pose traces drawn as
    # steps: each sample is one constant velocity over its keyframe interval,
    # and steps-mid around the midpoint stamps spans (about) that interval.
    ms = 3 if zoom else 0
    step = "steps-mid" if zoom else "default"
    traces = [
        (vel.dvl, C_DVL_V, _lbl_dvl(), dict(lw=0.8, marker="o", ms=ms + 1, zorder=2)),
        (vel.twist, C_TWIST, _lbl_twist(), dict(lw=1.0, zorder=3)),
        (vel.ekf_d, C_EKF_D, "EKF Δpose: " + _lbl_delta("E", vel.span),
         dict(lw=1.1, marker=".", ms=ms, drawstyle=step, zorder=4)),
        (vel.lm_d, C_LM_D, "batch LM Δpose: " + _lbl_delta("X", vel.span),
         dict(lw=1.3, marker=".", ms=ms, drawstyle=step, zorder=5)),
    ]
    for i, ax in enumerate(axs[:3]):
        for tr, c, label, kw in traces:
            m = (tr.t >= lo - 1) & (tr.t <= hi + 1)
            t, y = _broken(tr.t[m], tr.v[m, i], tr.max_gap)
            ax.plot(t, y, c=c, label=label if i == 0 else None, **kw)
        ax.set_ylabel(rf"$v_{AXES[i]}$ (m/s, base_link)")

    res = residuals(vel)
    cx = axs[3]
    res_style = {"EKF twist": (C_TWIST, 1.0), "EKF Δpose": (C_EKF_D, 1.0),
                 "LM Δpose": (C_LM_D, 1.2), "LM − EKF Δpose": (C_LM_EKF, 1.4)}
    for name, (t, y) in res.items():
        med, rms = _med_rms(y, t, lo, hi)
        if name == "LM − EKF Δpose":
            label = (r"$\Vert\mathbf{v}^{\mathrm{LM}}_{xy} - \mathbf{v}^{\mathrm{EKF\,\Delta}}_{xy}\Vert$"
                     f"   median {med:.1f}, RMS {rms:.1f} cm/s   (what the tags add)")
            gap = vel.lm_d.max_gap
        else:
            label = _lbl_res(RES_TEX[name], med, rms)
            gap = vel.dvl.max_gap
        c, lw = res_style[name]
        m = (t >= lo - 1) & (t <= hi + 1)
        tt, yy = _broken(t[m], y[m], gap)
        cx.plot(tt, yy, c=c, lw=lw, label=label, marker="." if zoom else None, ms=ms)
    cx.set_ylabel("horizontal residual (cm/s)")
    cx.set_xlabel("time since recording start (s)")
    cx.set_ylim(bottom=0)

    for j, ax in enumerate(axs):
        for i, (a, b) in enumerate(vel.gaps):
            if b >= lo and a <= hi:
                ax.axvspan(a, b, color=C_DVL, alpha=0.35, lw=0,
                           label="DVL bottom lock lost" if j == 0 and i == 0 else None)
        for i, (a, b) in enumerate(windows):
            if b >= lo and a <= hi:
                ax.axvspan(a, b, color=C_WIN, alpha=0.15, lw=0)
                if j == 0:
                    ax.text(0.5 * (a + b), 1.0, chr(ord("A") + i), transform=ax.get_xaxis_transform(),
                            ha="center", va="bottom", fontsize=10, fontweight="bold")
        ax.grid(alpha=0.3)
        ax.axhline(0.0, c="black", lw=0.5, zorder=1)
    axs[0].set_xlim(lo, hi)
    axs[0].legend(loc="lower left", bbox_to_anchor=(0.0, 1.04), ncol=2, fontsize=9,
                  frameon=False, handlelength=2.5)
    cx.legend(loc="upper left", fontsize=9, framealpha=0.9)
    # Top of the rect: below the derivation block (tight_layout fits the legend under it).
    fig.tight_layout(rect=(0, 0, 1, 0.905))
    fig.savefig(png, dpi=130)
    plt.close(fig)
    print(f"wrote {png}")


def summarize(vel: Velocities, lo: float, hi: float) -> List[str]:
    lines = [f"horizontal residuals {lo:g}-{hi:g} s (cm/s):"]
    for name, (t, y) in residuals(vel).items():
        med, rms = _med_rms(y, t, lo, hi)
        vs = "" if name == "LM − EKF Δpose" else "vs DVL"
        lines.append(f"  {name:>15} {vs:6} median {med:5.2f}  RMS {rms:5.2f}")
    for a, b in vel.gaps:
        if b >= lo and a <= hi:
            lines.append(f"DVL bottom lock lost {a:.1f}-{b:.1f} s ({b - a:.1f} s)")
    return lines


def main(argv: List[str] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("bag", help="rosbag2 directory with EKF odometry, DVL and tag detections")
    ap.add_argument("--config", required=True, nargs="+", help="node parameter file(s), merged in order")
    ap.add_argument("--start", type=float, default=None,
                    help="replay start, s into the original recording (the estimator starts fresh here)")
    ap.add_argument("--end", type=float, required=True, help="replay end, s into the original recording")
    ap.add_argument("--view-start", type=float, default=None, metavar="FROM",
                    help="draw only FROM..END; the replay still starts at --start")
    ap.add_argument("--window", type=float, nargs=2, action="append", default=[],
                    metavar=("FROM", "TO"), help="shade and label a window (A, B, ...), as in plot_graph")
    ap.add_argument("--zoom", type=float, nargs=2, action="append", default=[],
                    metavar=("FROM", "TO"), help="also write a zoomed figure of FROM..TO (repeatable)")
    ap.add_argument("--span", type=int, default=1,
                    help="difference poses over this many keyframes (1 = consecutive, 0.2 s at 5 Hz)")
    ap.add_argument("--frame-stride", type=int, default=1,
                    help="keep every Nth detector image, e.g. 3 to replay a 15 Hz bag at 5 Hz")
    ap.add_argument("--out", default="velocity.png", help="output image")
    args = ap.parse_args(argv)
    if args.span < 1:
        ap.error("--span must be >= 1")

    vel = collect(args.bag, args.config, args.start, args.end, args.span, args.frame_stride)
    lo = (args.view_start if args.view_start is not None
          else args.start if args.start is not None else float(vel.lm_d.t[0]))
    windows = [tuple(w) for w in args.window]
    name = bag_source(args.bag)
    span = "" if args.span == 1 else f", Δpose over {args.span} keyframes"
    print("\n".join(summarize(vel, lo, args.end)))
    plot(vel, args.out, f"{name}, {lo:g}–{args.end:g} s: body-frame velocity, DVL vs EKF vs "
         f"batch LM factor graph{span}", lo, args.end, windows)
    stem, ext = os.path.splitext(args.out)
    for a, b in args.zoom:
        plot(vel, f"{stem}_zoom_{a:g}-{b:g}{ext}",
             f"{name}, zoom {a:g}–{b:g} s: body-frame velocity{span}", a, b, windows, zoom=True)


if __name__ == "__main__":
    main()
