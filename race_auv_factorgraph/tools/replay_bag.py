"""Replay a rosbag2 recording through the core estimator, offline and deterministically (§12).

Usage::

    ros2 run race_auv_factorgraph replay_bag BAG_DIR \\
        --config $(ros2 pkg prefix race_auv_factorgraph)/share/race_auv_factorgraph/config/dock_fuser_sim.yaml \\
        --out /tmp/run1 [--holdout-every 5] [--exclude-tag 36h11:146] \\
        [--gt-topic /race_auv/docking_station/ground_truth/pose] [--frame-stride 3]

The bag must contain the EKF odometry and the per-camera
``apriltag_msgs/AprilTagDetectionArray`` topics named in the config. Messages
are fed in recorded order through the same :class:`KeyframeScheduler` and
:class:`DockEstimator` the node uses, built from the same parameter file
(``ros/params.py``). The result is therefore what the node would have
computed live, minus scheduling jitter, and is identical on every run.

Outputs (in ``--out``):

``keyframes.csv``
    One row per keyframe: state, accepted/rejected tag counts, the docking
    pose ``X_k^-1 * S`` (dock_point in base_link), the station base_link in
    base_link, ``S`` in odom, the drift correction ``D = X_k E_k^-1`` and the
    update time.
``heldout.csv``
    Validation residuals (§13): reprojection RMS of detections withheld from
    the graph, either every Nth tag frame (``--holdout-every``) or all
    observations of a tag (``--exclude-tag``, leave-one-tag-out).
``ground_truth.csv``
    If ``--gt-topic`` is given: the ground-truth poses (``PoseStamped``,
    e.g. from ``ground_truth_docking_node``), for ``evaluate``.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from typing import Any, Callable, Dict, List, Optional, Set

import gtsam
import numpy as np

from ..core.common.config import EstimatorConfig
from ..core.backend.estimator import DockEstimator, KeyframeResult
from ..core.common.geometry import pose_from_quat_xyzw
from ..core.common.model import Detection, TagFrame, normalize_family, tag_key_str
from ..core.frontend.scheduler import KeyframeScheduler
from ..ros.params import build_setup, getter_from_yaml


def _stamp(msg) -> float:
    return float(msg.header.stamp.sec) + 1e-9 * float(msg.header.stamp.nanosec)


def _pose_cols(prefix: str, T: gtsam.Pose3) -> Dict[str, float]:
    """x, y, z, roll, pitch, yaw columns for a pose (angles in degrees)."""
    if T is None:
        return {f"{prefix}_{k}": "" for k in ("x", "y", "z", "roll", "pitch", "yaw")}
    t, R = T.translation(), T.rotation()
    return {f"{prefix}_x": t[0], f"{prefix}_y": t[1], f"{prefix}_z": t[2],
            f"{prefix}_roll": math.degrees(R.roll()), f"{prefix}_pitch": math.degrees(R.pitch()),
            f"{prefix}_yaw": math.degrees(R.yaw())}


def bag_source(bag: str) -> str:
    """Display name of the recording a bag came from, for plot titles.

    ``bag_test.launch.py`` writes ``source.yaml`` into its output bag (the
    original recording and the replayed window). Without it the bag's own
    directory name is used.
    """
    name = os.path.basename(os.path.normpath(bag))
    path = os.path.join(bag, "source.yaml")
    if not os.path.exists(path):
        return name
    import yaml
    with open(path, encoding="utf-8") as f:
        src = yaml.safe_load(f) or {}
    out = str(src.get("source_bag", name))
    if float(src.get("start_offset", 0.0)):
        out += f" (from {float(src['start_offset']):g} s)"
    return out


def source_start_time(bag: str) -> float:
    """Start (s) of the original recording a bag came from, the zero of the plot time axes.

    From ``source.yaml`` (``bag_test.launch.py``): the start of the recording
    at ``source_path`` if it is still there, else this bag's start minus the
    replayed ``start_offset``. Without ``source.yaml``: this bag's own start.
    Times on that axis then read as seconds into the original recording,
    whichever window of it the bag holds.
    """
    path = os.path.join(bag, "source.yaml")
    if not os.path.exists(path):
        return bag_start_time(bag)
    import yaml
    with open(path, encoding="utf-8") as f:
        src = yaml.safe_load(f) or {}
    original = src.get("source_path", "")
    if original and os.path.isdir(original):
        return bag_start_time(original)
    return bag_start_time(bag) - float(src.get("start_offset", 0.0))


def bag_start_time(bag: str) -> float:
    """Recording start of the bag (seconds), from its metadata."""
    return _open_reader(bag).get_metadata().starting_time.nanoseconds * 1e-9


def _open_reader(bag: str):
    import rosbag2_py
    reader = rosbag2_py.SequentialReader()
    storage_id = "mcap" if any(f.endswith(".mcap") for f in os.listdir(bag)) else "sqlite3"
    reader.open(rosbag2_py.StorageOptions(uri=bag, storage_id=storage_id),
                rosbag2_py.ConverterOptions("cdr", "cdr"))
    return reader


def load_camera_info(bag: str, setup) -> None:
    """Complete the cameras without config intrinsics from the bag's recorded camera_info.

    Same source as the live node: the tag detector's latched, rectified
    camera_info (``bag_test.launch.py`` records it). Older detection bags
    have none; their config must then set ``cameras.<name>.fx ...``.
    """
    from rclpy.serialization import deserialize_message
    from sensor_msgs.msg import CameraInfo
    import rosbag2_py

    missing = {setup.camera_info_topics[n]: n for n in setup.missing_intrinsics()}
    if not missing:
        return
    reader = _open_reader(bag)
    reader.set_filter(rosbag2_py.StorageFilter(topics=list(missing)))
    while missing and reader.has_next():
        topic, data, _ = reader.read_next()
        if topic in missing:
            msg = deserialize_message(data, CameraInfo)
            setup.add_camera_info(missing.pop(topic), msg.k, msg.width, msg.height)
    if missing:
        raise RuntimeError(
            f"no intrinsics for camera(s) {sorted(missing.values())}: the config sets no "
            f"cameras.<name>.fx/fy/cx/cy/width/height and the bag has no {sorted(missing)}")


def replay(bag: str, get: Callable[[str], Any], setup, cfg: Optional[EstimatorConfig] = None,
           holdout_every: int = 0, excluded: Set = frozenset(), gt_topic: str = "",
           on_keyframe: Optional[Callable] = None, on_gt: Optional[Callable] = None,
           start: Optional[float] = None, end: Optional[float] = None,
           frame_stride: int = 1):
    """Feed a bag through the scheduler and a fresh estimator. Returns ``(estimator, scheduler)``.

    Parameters
    ----------
    get, setup
        Parameter accessor and models from ``ros/params.py``. The bag
        topics come from them.
    cfg
        Estimator config override (default: ``setup.cfg``). ``compare_backends``
        uses it to vary ``smoother_lag`` and switch the covariance off.
    holdout_every, excluded
        Validation (§13): withhold every Nth tag frame, or every observation
        of the tags in ``excluded``, from the graph.
    on_keyframe(est, result, held_detections)
        Called after every keyframe. ``held_detections`` are the withheld
        detections at exactly that keyframe's stamp.
    on_gt(stamp, pose_msg)
        Called for every message on ``gt_topic``.
    start, end
        Optional window in seconds from the start of the bag (by receive
        time). Messages outside it are skipped, so the estimator starts
        fresh at ``start``, exactly as if the bag began there.
    frame_stride
        Keep only every Nth detector message (1 = all). Emulates a slower
        detector on a bag recorded at a higher rate: the tank bags have the
        detector at 15 Hz, the vehicle runs it at 5 Hz, so ``frame_stride=3``
        replays them as the vehicle would see them. A 5 Hz detector processes
        the newest image every 0.2 s, i.e. every 3rd 15 Hz image.
        Re-published messages (same image stamp as the previous one) are
        removed *before* counting, so the stride counts distinct images.
    """
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message

    load_camera_info(bag, setup)
    est = DockEstimator(cfg or setup.cfg, setup.cameras, setup.tags, setup.T_dock_base)
    sched = KeyframeScheduler(ekf_buffer_seconds=float(get("ekf_buffer_seconds")))
    odom_topic = get("odom_topic")
    cam_by_topic = {topic: cam for cam, topic in setup.camera_topics.items()}
    reader = _open_reader(bag)
    t_bag0 = reader.get_metadata().starting_time.nanoseconds * 1e-9
    types = {t.name: t.type for t in reader.get_all_topics_and_types()}
    for needed in [odom_topic, *cam_by_topic]:
        if needed not in types:
            print(f"warning: topic {needed} is not in the bag")
    # DVL bottom lock (only the stamps are used). Without it in the bag the
    # DVL dropout handling stays inactive, as in the node with dvl_topic "".
    dvl_topic = get("dvl_topic") or ""
    if dvl_topic and dvl_topic not in types:
        print(f"warning: DVL topic {dvl_topic} is not in the bag; DVL dropout handling is off")
    msg_cls = {name: get_message(types[name]) for name in types
               if name in (odom_topic, gt_topic, dvl_topic, *cam_by_topic)}

    n_images = 0                     # distinct detector images seen (for frame_stride)
    last_image_stamp = None
    frame_count = 0                  # detector messages fed to the scheduler (for holdout)
    holdout_dets: Dict[float, List[Detection]] = {}
    while reader.has_next():
        topic, data, t_recv_ns = reader.read_next()
        if topic not in msg_cls:
            continue
        t_rel = t_recv_ns * 1e-9 - t_bag0
        if start is not None and t_rel < start:
            continue
        if end is not None and t_rel > end:
            break
        msg = deserialize_message(data, msg_cls[topic])
        if topic == odom_topic:
            p, q = msg.pose.pose.position, msg.pose.pose.orientation
            sched.add_odometry(_stamp(msg), pose_from_quat_xyzw((p.x, p.y, p.z),
                                                                 (q.x, q.y, q.z, q.w)))
        elif topic == gt_topic:
            if on_gt is not None:
                on_gt(_stamp(msg), msg)
            continue
        elif topic == dvl_topic:
            est.dvl.add(_stamp(msg))
            continue
        else:
            cam = cam_by_topic[topic]
            t = _stamp(msg)
            if frame_stride > 1:
                if last_image_stamp is not None and t <= last_image_stamp:
                    continue  # re-published image: does not count towards the stride
                last_image_stamp = t
                n_images += 1
                if (n_images - 1) % frame_stride != 0:
                    continue  # a slower detector would never have processed this image
            dets, held = [], []
            frame_count += 1
            hold_frame = holdout_every > 0 and frame_count % holdout_every == 0
            for d in msg.detections:
                key = (normalize_family(d.family), int(d.id))
                if key not in setup.tags:
                    continue
                det = Detection(cam, key, np.array([[c.x, c.y] for c in d.corners], dtype=float))
                (held if hold_frame or key in excluded else dets).append(det)
            if held:
                holdout_dets[t] = held
            # Every detector message becomes a keyframe, also an empty one or
            # one whose detections were all withheld; a withheld detection's
            # residual is then evaluated at a pose at exactly its stamp.
            sched.add_frame(TagFrame(t, cam, dets))
        for kf in sched.poll():
            res = est.process_keyframe(kf.stamp, kf.E, kf.frame)
            if on_keyframe is not None:
                on_keyframe(est, res, holdout_dets.pop(res.stamp, []))
    return est, sched


def main(argv: List[str] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("bag", help="rosbag2 directory")
    ap.add_argument("--config", required=True, nargs="+",
                    help="dock_fuser parameter file(s), merged in order (e.g. station_tags.yaml "
                         "dock_fuser.yaml)")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--holdout-every", type=int, default=0,
                    help="withhold every Nth tag frame from the graph and report its residual")
    ap.add_argument("--exclude-tag", action="append", default=[],
                    help="leave-one-tag-out: never give this tag (family:id) to the graph")
    ap.add_argument("--gt-topic", default="", help="ground-truth PoseStamped topic to export")
    ap.add_argument("--start", type=float, default=None, help="window start, s from bag start")
    ap.add_argument("--end", type=float, default=None, help="window end, s from bag start")
    ap.add_argument("--frame-stride", type=int, default=1,
                    help="keep every Nth detector image, e.g. 3 to replay a 15 Hz bag at 5 Hz")
    args = ap.parse_args(argv)

    get = getter_from_yaml(args.config)
    setup = build_setup(get, lambda m: print("warning:", m))
    excluded: Set = set()
    for s in args.exclude_tag:
        fam, tid = s.split(":")
        excluded.add((normalize_family(fam), int(tid)))

    os.makedirs(args.out, exist_ok=True)
    kf_rows, held_rows, gt_rows = [], [], []

    def on_keyframe(est: DockEstimator, res: KeyframeResult, held: List[Detection]) -> None:
        row = {"stamp": res.stamp, "state": res.state.value, "n_accepted": len(res.accepted),
               "n_rejected": len(res.rejected), "initialized_now": int(res.initialized_now),
               "recovered": int(res.recovered),
               "seeded_from_tag": int(res.seeded_from_tag),
               "update_ms": res.update_time * 1000.0, "init_note": res.init_note}
        dock = res.dock_in_base
        row.update(_pose_cols("dock", dock))
        row.update(_pose_cols("station_base", None if dock is None
                              else dock.compose(setup.T_dock_base)))
        row.update(_pose_cols("S", res.S))
        row.update(_pose_cols("D", res.drift_correction))
        kf_rows.append(row)
        # Residuals of withheld detections, at this keyframe's corrected pose.
        for d in held:
            rms = est.predict_rms(d, res.X)
            if rms is not None:
                held_rows.append({"stamp": res.stamp, "camera": d.camera,
                                  "tag": tag_key_str(d.tag), "rms_px": rms})

    def on_gt(t: float, msg) -> None:
        p, q = msg.pose.position, msg.pose.orientation
        gt_rows.append({"stamp": t, "x": p.x, "y": p.y, "z": p.z,
                        "qx": q.x, "qy": q.y, "qz": q.z, "qw": q.w})

    est, sched = replay(args.bag, get, setup, holdout_every=args.holdout_every,
                        excluded=excluded, gt_topic=args.gt_topic,
                        on_keyframe=on_keyframe, on_gt=on_gt, start=args.start, end=args.end,
                        frame_stride=args.frame_stride)

    def write(name: str, rows: List[dict]) -> None:
        if not rows:
            return
        path = os.path.join(args.out, name)
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {path} ({len(rows)} rows)")

    write("keyframes.csv", kf_rows)
    write("heldout.csv", held_rows)
    write("ground_truth.csv", gt_rows)
    print(f"final state {est.state.value}; {est.n_tag_factors} tag factors, "
          f"{est.n_rejected} rejections, {est.n_recoveries} backend recoveries, "
          f"{sched.n_dropped} detector messages dropped (re-published / out of order)")


if __name__ == "__main__":
    main()
