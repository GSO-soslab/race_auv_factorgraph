"""Plot helpers shared by the figure tools (``plot_graph``, ``analyze_run``).

Every time axis in these figures is seconds into the original recording
(:func:`~race_auv_factorgraph.tools.bag.source_start_time`).
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np


def cloud_spread(q: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """Centre, RMS and max horizontal distance (m) of a cloud of tag positions.

    Each tag observation chained onto an AUV track puts the tag somewhere;
    the tag does not move, so ideally all land on one point. The centre is
    the cloud's own mean (not the graph's tag estimate), so clouds from
    different tracks are measured the same way: consistency only.
    Horizontal (x-y), to match the bird's-eye views the circles are drawn in.
    """
    c = q[:, :2].mean(axis=0)
    r = np.linalg.norm(q[:, :2] - c, axis=1)
    return c, float(np.sqrt(np.mean(r ** 2))), float(r.max())


def timing_stats(u: np.ndarray) -> str:
    """``"median a, p95 b, max c ms"`` of update times (ms), NaNs ignored."""
    u = np.asarray(u, dtype=float)
    u = u[np.isfinite(u)]
    return f"median {np.median(u):.1f}, p95 {np.percentile(u, 95):.1f}, max {u.max():.1f} ms"


def cpu_name() -> str:
    """This computer's CPU model, for timing labels."""
    import platform
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as f:
            return next(ln.split(":", 1)[1].strip() for ln in f if ln.startswith("model name"))
    except (OSError, StopIteration):
        return platform.processor() or platform.machine()


def time_mark_indices(times: np.ndarray, every: float) -> List[Tuple[float, int]]:
    """``(mark time, track index)`` for every multiple of ``every`` along a track.

    The index is the track point closest to that time, kept one away from
    either end so it has neighbours for a tangent. Marks closer than half a
    period to either end of the track are skipped so they cannot collide with
    the start/end markers the callers already draw. Shared by
    :func:`time_labels` and :func:`direction_marks`, so the triangles sit on
    the ticks.
    """
    times = np.asarray(times, dtype=float)
    if len(times) < 3:
        return []
    cand = np.arange(np.ceil(times[0] / every) * every, times[-1] + 1e-9, every)
    cand = [c for c in cand if c - times[0] >= 0.5 * every and times[-1] - c >= 0.5 * every]
    return [(float(c), min(max(int(np.argmin(np.abs(times - c))), 1), len(times) - 2))
            for c in cand]


def direction_marks(ax, pts: np.ndarray, times: np.ndarray, every: float, color: str,
                    alpha: float = 1.0, size: float = 140, zorder: int = 6) -> None:
    """Triangles on a track at the time marks, each pointing the way the vehicle travels there.

    ``pts`` (n x >=2) is a track in metres, ``times`` its per-point time; the
    marks are those of :func:`time_labels` with the same ``every``. Each
    triangle is oriented along the local direction of travel (the track's
    tangent, through the neighbouring points), which can differ from the
    heading when the AUV sways or backs up. The marker is an isosceles
    triangle, so the tip reads as "forward"; with the equal aspect of the
    BEVs the data angle is also the angle on screen.
    """
    from matplotlib.path import Path

    pts = np.asarray(pts)[:, :2]
    tip = np.array([[1.0, 0.0], [-0.7, 0.55], [-0.7, -0.55], [1.0, 0.0]])
    for _, i in time_mark_indices(times, every):
        d = pts[i + 1] - pts[i - 1]
        if float(np.linalg.norm(d)) < 1e-9:
            continue
        a = np.arctan2(d[1], d[0])
        rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
        marker = Path(tip @ rot.T, [Path.MOVETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY])
        ax.scatter([pts[i, 0]], [pts[i, 1]], marker=marker, s=size, c=color, edgecolors="white",
                   linewidths=0.6, alpha=alpha, zorder=zorder)


def time_labels(ax, pts: np.ndarray, times: np.ndarray, every: float, color: str,
                fontsize: int = 8, zorder: int = 6) -> None:
    """Tick the track at regular times and write the time beside each tick.

    ``pts`` (n x >=2) is a track in metres, ``times`` its per-point time. A
    tick (short line perpendicular to the track) and an ``"abc s"`` label go
    at the track point closest to each multiple of ``every``. Candidates
    closer than half a period to either end of the track are skipped so they
    cannot collide with the start/end markers the callers already draw. The
    track can cross the same place several times; the label says which pass
    is which.
    """
    pts = np.asarray(pts)[:, :2]
    diag = float(np.hypot(*(pts.max(axis=0) - pts.min(axis=0)))) or 1.0
    for c, i in time_mark_indices(times, every):
        d = pts[i + 1] - pts[i - 1]
        n = np.array([-d[1], d[0]])
        nn = float(np.linalg.norm(n))
        if nn < 1e-9:
            continue
        n = n / nn * 0.025 * diag
        p = pts[i]
        ax.plot([p[0], p[0] + n[0]], [p[1], p[1] + n[1]], c=color, lw=0.9, zorder=zorder)
        ax.text(p[0] + 1.35 * n[0], p[1] + 1.35 * n[1], f"{c:.0f} s", fontsize=fontsize,
                color=color, ha="center", va="center", zorder=zorder + 1,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=0.8))


def window_box(ax, X: np.ndarray, E: np.ndarray, label: str, fontsize: int = 10,
               above: bool = True, zorder: int = 6) -> None:
    """Box around one time window's stretch of the tracks in a bird's-eye view.

    ``X``, ``E``: the window's graph and EKF positions (n x 2 or n x 3). The
    box covers both tracks with a 12 cm margin and is never smaller than
    35 cm, so it stays visible; ``label`` is written above it (below if
    ``above`` is False).
    """
    from matplotlib.patches import Rectangle
    pts = np.vstack([X[:, :2], E[:, :2]])
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    size = np.maximum(hi - lo + 0.24, 0.35)
    c = 0.5 * (lo + hi)
    ax.add_patch(Rectangle(c - size / 2, size[0], size[1], fill=False, ec="#1a202c", lw=1.4,
                           zorder=zorder))
    y = c[1] + size[1] / 2 + 0.04 if above else c[1] - size[1] / 2 - 0.04
    ax.text(c[0] - size[0] / 2, y, label, fontsize=fontsize, fontweight="bold",
            va="bottom" if above else "top", zorder=zorder + 2,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.5))
