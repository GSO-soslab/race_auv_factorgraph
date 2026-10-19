"""Evaluate a ``replay_bag`` run: error vs sim truth, held-out residuals (§13).

Usage::

    ros2 run race_auv_factorgraph evaluate /tmp/run1 [--gt-offset "0.7 0 0 0 0 0"]

Reads ``keyframes.csv``, and ``ground_truth.csv`` / ``heldout.csv`` when they
exist, from the replay output directory, and prints:

* time from the first keyframe to TRACKING;
* docking error vs ground truth (median / 95th percentile / max), translation
  and rotation, over all tracking keyframes and over the final 20 %;
* held-out reprojection RMS per tag (held-out frames or leave-one-tag-out).

Ground truth: ``ground_truth_docking_node`` publishes the station
``base_link`` in the AUV ``base_link`` (``PoseStamped``). It is compared with
the replay's ``station_base`` columns, ``X_k^-1 * S * T_dock_base``.

``--gt-offset "x y z roll pitch yaw"`` (m, rad) pre-multiplies every
ground-truth pose, ``T_base_station = offset * T_gt``. Use it when the
ground-truth source measures from a different AUV frame, e.g. the Stonefish
``Base`` link that coincides with the URDF ``nose_tip_link``, 0.7 m ahead of
``base_link``: ``--gt-offset "0.7 0 0 0 0 0"``.
"""

from __future__ import annotations

import argparse
import bisect
import collections
import csv
import math
import os
from typing import Dict, List

import numpy as np

from ..core.common.geometry import interpolate, pose_distance, pose_from_quat_xyzw, pose_from_xyz_rpy


def _read(path: str) -> List[Dict[str, str]]:
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _stats(values: List[float]) -> str:
    if not values:
        return "n/a"
    a = np.asarray(values)
    return (f"median {np.median(a):.4g}  p95 {np.percentile(a, 95):.4g}  "
            f"max {a.max():.4g}  (n={len(a)})")


def main(argv: List[str] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("run_dir", help="output directory of replay_bag")
    ap.add_argument("--gt-offset", default="0 0 0 0 0 0",
                    help='"x y z roll pitch yaw" applied as offset * T_gt')
    args = ap.parse_args(argv)

    kfs = _read(os.path.join(args.run_dir, "keyframes.csv"))
    if not kfs:
        raise SystemExit("keyframes.csv not found or empty")
    t0 = float(kfs[0]["stamp"])
    tracking = [r for r in kfs if r["state"] == "TRACKING" and r["station_base_x"] != ""]
    print(f"{len(kfs)} keyframes over {float(kfs[-1]['stamp']) - t0:.1f} s")
    if tracking:
        print(f"TRACKING after {float(tracking[0]['stamp']) - t0:.2f} s")
    else:
        print("never reached TRACKING")
    print(f"update time [ms]: {_stats([float(r['update_ms']) for r in kfs])}")

    # ------------------------------------------------------------ ground truth
    gt = _read(os.path.join(args.run_dir, "ground_truth.csv"))
    if gt and tracking:
        off = [float(v) for v in args.gt_offset.split()]
        offset = pose_from_xyz_rpy(off[:3], off[3:])
        gt_t = [float(r["stamp"]) for r in gt]
        gt_p = [offset.compose(pose_from_quat_xyzw(
            (float(r["x"]), float(r["y"]), float(r["z"])),
            (float(r["qx"]), float(r["qy"]), float(r["qz"]), float(r["qw"])))) for r in gt]
        errs = []
        for r in tracking:
            t = float(r["stamp"])
            i = bisect.bisect_left(gt_t, t)
            if i == 0 or i >= len(gt_t):
                continue
            a = (t - gt_t[i - 1]) / (gt_t[i] - gt_t[i - 1])
            truth = interpolate(gt_p[i - 1], gt_p[i], a)
            est = pose_from_xyz_rpy(
                [float(r[f"station_base_{k}"]) for k in "xyz"],
                [math.radians(float(r[f"station_base_{k}"])) for k in ("roll", "pitch", "yaw")])
            errs.append((t,) + pose_distance(est, truth))
        if errs:
            tail = [e for e in errs if e[0] >= errs[0][0] + 0.8 * (errs[-1][0] - errs[0][0])]
            print("docking error vs ground truth (station base_link in AUV base_link):")
            print(f"  translation [m]   all:  {_stats([e[1] for e in errs])}")
            print(f"  rotation [deg]    all:  {_stats([math.degrees(e[2]) for e in errs])}")
            print(f"  translation [m]   last 20%: {_stats([e[1] for e in tail])}")
            print(f"  rotation [deg]    last 20%: {_stats([math.degrees(e[2]) for e in tail])}")
        else:
            print("ground truth does not overlap the tracking period")

    # ------------------------------------------------------------ held-out residuals
    held = _read(os.path.join(args.run_dir, "heldout.csv"))
    if held:
        by_tag = collections.defaultdict(list)
        for r in held:
            by_tag[r["tag"]].append(float(r["rms_px"]))
        print("held-out reprojection RMS [px] (§13: detections the graph never saw):")
        for tag, vals in sorted(by_tag.items()):
            print(f"  {tag:>10s}: {_stats(vals)}")


if __name__ == "__main__":
    main()
