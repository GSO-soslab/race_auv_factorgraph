"""Re-run the tag detectors on a recorded bag's images, offline and deterministically.

Usage (source ROS first)::

    python3 race_auv_factorgraph/tools/redetect_bag.py BAG --out OUT_BAG \\
        [--detector-config station_tags.yaml tag_factorgraph_detector.yaml] \\
        [--cameras cam_front cam_down] [--camera-tags cam_front=54 cam_down=176,185]

Why
---
A live run records the detectors' output (``/<cam>/apriltag_detection/tags``)
made with the tag configuration of that day. To try a *different* one (e.g.
per-camera tags, ``camera_tag_ids``) on the same run, the detection has to
be redone on the recorded images. ``bag_test.launch.py`` does that by
playing the bag in real time, but the live detector's 5 Hz timer then
picks whichever image is newest, so the keyframes differ from the live run
and from one replay to the next.

This tool re-detects **exactly the images the live detectors processed**:
the header stamps of the recorded ``tags`` messages are the stamps of the
images they came from. Each of those images is decoded, rectified and
detected with the detector package's own code
(``race_auv_camera_tag_factorgraph``: ``Rectifier``, ``TagDetector``,
``select_tags``) and parameters (``--detector-config``, the same files the
launch uses), in the same order as ``detector_node._tick`` (colour decode
when that camera had ``publish_image``, rectify, then grey). So the
keyframe camera's keyframes keep the live stamps, and only the tags differ.

Output bag
----------
* ``/<cam>/apriltag_detection/tags`` (re-detected) and ``.../camera_info``
  (the rectifier's intrinsics), for every camera in ``--cameras``;
* copied unchanged: the EKF, the DVL and the live dock_fuser topics
  (``/dock_fuser/auv_pose``, ``station_pose``, ``keyframe``), so
  ``analyze_run`` can still compare against the live estimates;
* ``source.yaml`` naming the input bag, so the time axis of every tool stays
  seconds into the original recording.

Every message keeps its original receive time, so a replay sees the
detections and the EKF in the same order as the live node did.

Synced cameras (``sync_to`` in the detector config, e.g. cam_down synced to
cam_front): which of that camera's images the live detector happened to
process does not matter. For every recorded message on ``sync_to`` (the
reference detector's images), the camera's nearest image is matched with
the detector's own :class:`race_auv_camera_tag_factorgraph.sync.ImageSync`
and detected, and the result is written when the match is decided, as the
synced node would publish it. So a bag recorded with free-running detectors
shows what the synced ones give.

``--camera-tags cam=id,id`` overrides a camera's ``camera_tag_ids``. With the
tags the live detectors used it reproduces the recorded detections, which
is how the tool checks itself (``--compare``: corner differences against the
recorded ``tags``).
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections import Counter
from typing import Any, Dict, List, Sequence

import cv2
import numpy as np

if __package__ in (None, ""):
    # Run as a file: make the package importable from this source tree.
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    __package__ = "race_auv_factorgraph.tools"

from ..ros.params import flatten_params_yaml
from .replay_bag import _open_reader

# Copied unchanged into the output bag (when present in the input).
COPY_TOPICS = [
    "/race_auv/odometry/filtered",
    "/race_auv/nucleus_node/bottom_lock_velocity_common",
    "/dock_fuser/auv_pose",
    "/dock_fuser/station_pose",
    "/dock_fuser/keyframe",
    "/race_auv/dock_point/pose",
]

# The detector node's parameter defaults that matter here (detector_node.py).
_DETECTOR_DEFAULTS = {"threads": 4, "quad_decimate": 2.0, "refine_edges": True,
                      "min_edge_px": 10.0, "min_side_px": 15.0, "max_fit_rms_px": 1.5,
                      "publish_image": False, "camera_tag_ids": [-1], "camera_frame": "",
                      "sync_to": "", "sync_max_dt": 0.05, "sync_wait": 0.1,
                      "intrinsics.distortion": [0.0, 0.0, 0.0, 0.0], "intrinsics.fisheye": True}


def default_detector_config() -> List[str]:
    """The vehicle's detector parameter files, as tag_factorgraph_detection.launch.py loads them."""
    from ament_index_python.packages import get_package_share_directory
    d = os.path.join(get_package_share_directory("race_auv_bringup"), "config")
    return [os.path.join(d, "station_tags.yaml"), os.path.join(d, "tag_factorgraph_detector.yaml")]


def camera_params(paths: Sequence[str], camera: str) -> Dict[str, Any]:
    """One camera's detector parameters: ``/**`` then ``/<camera>/tag_detector``, files in order."""
    p: Dict[str, Any] = dict(_DETECTOR_DEFAULTS)
    for path in paths:
        p.update(flatten_params_yaml(path, node_name=f"{camera}/tag_detector"))
    return p


class CameraDetector:
    """Rectifier + detector of one camera, built exactly as ``TagDetectorNode.__init__`` does."""

    def __init__(self, camera: str, p: Dict[str, Any]) -> None:
        from race_auv_camera_tag_factorgraph.rectify import Rectifier
        from race_auv_camera_tag_factorgraph.tags import TagDetector, select_tags
        fx, fy, cx, cy = (float(p[f"intrinsics.{k}"]) for k in ("fx", "fy", "cx", "cy"))
        size = (int(p["intrinsics.width"]), int(p["intrinsics.height"]))
        self.camera = camera
        self.image_topic = str(p["image_topic"])
        self.frame_id = str(p["camera_frame"])
        self.colour = bool(p["publish_image"])     # the live node decoded in colour then
        self.sync_to = str(p["sync_to"] or "")
        if self.sync_to:
            from race_auv_camera_tag_factorgraph.sync import ImageSync
            self.sync = ImageSync(float(p["sync_max_dt"]), float(p["sync_wait"]))
        self.sizes = select_tags(list(p["tag_ids"]), list(p["tag_sizes"]),
                                 list(p["camera_tag_ids"]))
        self.rectifier = Rectifier(np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]]),
                                   p["intrinsics.distortion"], size, bool(p["intrinsics.fisheye"]))
        r = self.rectifier
        self.detector = TagDetector(
            self.sizes, r.K, r.width, r.height, threads=int(p["threads"]),
            decimate=float(p["quad_decimate"]), refine_edges=bool(p["refine_edges"]),
            min_edge_px=float(p["min_edge_px"]), min_side_px=float(p["min_side_px"]),
            max_fit_rms_px=float(p["max_fit_rms_px"]))

    def detect(self, jpeg: bytes):
        img = cv2.imdecode(np.frombuffer(jpeg, np.uint8),
                           cv2.IMREAD_COLOR if self.colour else cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None
        rect = self.rectifier.rectify(img)
        gray = cv2.cvtColor(rect, cv2.COLOR_BGR2GRAY) if rect.ndim == 3 else rect
        return self.detector.detect(gray)

    def tags_msg(self, res, stamp):
        """``AprilTagDetectionArray`` exactly as ``TagDetectorNode._publish`` fills it."""
        from apriltag_msgs.msg import AprilTagDetection, AprilTagDetectionArray, Point
        from race_auv_camera_tag_factorgraph.tags import FAMILY
        msg = AprilTagDetectionArray()
        msg.header.stamp = stamp
        msg.header.frame_id = self.frame_id
        for t in res.tags:
            d = AprilTagDetection()
            d.family = FAMILY
            d.id = t.tag_id
            d.hamming = t.hamming
            d.decision_margin = float(t.margin)
            d.centre = Point(x=float(t.corners[:, 0].mean()), y=float(t.corners[:, 1].mean()))
            d.corners = [Point(x=float(u), y=float(v)) for u, v in t.corners]
            msg.detections.append(d)
        return msg

    def camera_info_msg(self, stamp):
        from sensor_msgs.msg import CameraInfo
        r = self.rectifier
        info = CameraInfo()
        info.header.stamp = stamp
        info.header.frame_id = self.frame_id
        info.width, info.height = r.width, r.height
        info.distortion_model = "plumb_bob"
        info.d = [0.0] * 5
        info.k = [float(v) for v in r.K.ravel()]
        info.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        info.p = [float(r.K[0, 0]), 0.0, float(r.K[0, 2]), 0.0,
                  0.0, float(r.K[1, 1]), float(r.K[1, 2]), 0.0, 0.0, 0.0, 1.0, 0.0]
        return info


def redetect(bag: str, out: str, config: Sequence[str], cameras: Sequence[str],
             camera_tags: Dict[str, List[int]], compare: bool = False) -> Dict[str, Counter]:
    """Write ``out`` (see the module docstring). Returns per-camera tag counts."""
    import rosbag2_py
    from apriltag_msgs.msg import AprilTagDetectionArray
    from rclpy.serialization import deserialize_message, serialize_message
    from sensor_msgs.msg import CompressedImage

    dets: Dict[str, CameraDetector] = {}
    for cam in cameras:
        p = camera_params(config, cam)
        if cam in camera_tags:
            p["camera_tag_ids"] = camera_tags[cam]
        dets[cam] = CameraDetector(cam, p)
        print(f"{cam}: tags {dets[cam].sizes} from {dets[cam].image_topic}; "
              f"{dets[cam].rectifier.describe()}"
              + (f"; synced to {dets[cam].sync_to}" if dets[cam].sync_to else ""))
    tags_topic = {cam: f"/{cam}/apriltag_detection/tags" for cam in cameras}
    image_cam = {d.image_topic: cam for cam, d in dets.items()}

    reader = _open_reader(bag)
    types = {t.name: t.type for t in reader.get_all_topics_and_types()}
    # A synced camera's images follow its reference's messages, not its own record.
    sync_refs = {d.sync_to: cam for cam, d in dets.items() if d.sync_to}
    missing = [t for t in [*tags_topic.values(), *image_cam, *sync_refs] if t not in types]
    if missing:
        raise SystemExit(f"not in {bag}: {', '.join(missing)} (needs the live detector output and "
                         "the camera images)")
    copy = [t for t in COPY_TOPICS if t in types]
    reader.set_filter(rosbag2_py.StorageFilter(
        topics=sorted({*copy, *tags_topic.values(), *image_cam, *sync_refs})))
    t0_ns = reader.get_metadata().starting_time.nanoseconds

    if os.path.exists(out):
        raise SystemExit(f"{out} exists; choose a new --out")
    writer = rosbag2_py.SequentialWriter()
    writer.open(rosbag2_py.StorageOptions(uri=out, storage_id="mcap"),
                rosbag2_py.ConverterOptions("cdr", "cdr"))
    topic_id = 0

    def create(name: str, type_: str) -> None:
        nonlocal topic_id
        writer.create_topic(rosbag2_py.TopicMetadata(topic_id, name, type_, "cdr"))
        topic_id += 1
    for t in copy:
        create(t, types[t])
    for cam in cameras:
        create(tags_topic[cam], "apriltag_msgs/msg/AprilTagDetectionArray")
        create(f"/{cam}/apriltag_detection/camera_info", "sensor_msgs/msg/CameraInfo")
        # camera_info first: replay_bag.load_camera_info reads the first one.
        from builtin_interfaces.msg import Time
        stamp = Time(sec=t0_ns // 1_000_000_000, nanosec=t0_ns % 1_000_000_000)
        writer.write(f"/{cam}/apriltag_detection/camera_info",
                     serialize_message(dets[cam].camera_info_msg(stamp)), t0_ns)

    images: Dict[str, Dict[tuple, bytes]] = {cam: {} for cam in cameras}  # stamp -> jpeg
    counts = {cam: Counter() for cam in cameras}
    n_frames = Counter()
    n_missing = Counter()
    corner_err: Dict[str, List[float]] = {cam: [] for cam in cameras}
    tags_cam = {v: k for k, v in tags_topic.items() if not dets[k].sync_to}
    sync_dt: Dict[str, List[float]] = {cam: [] for cam in sync_refs.values()}
    t_start = time.perf_counter()

    def emit_synced(cam: str, ready, t_recv: int) -> None:
        """Detect and write a synced camera's matched images (stamped with their own image time)."""
        from builtin_interfaces.msg import Time
        for t_ref, t_img, jpeg in ready:
            res = dets[cam].detect(jpeg)
            if res is None:
                n_missing[cam] += 1
                continue
            n_frames[cam] += 1
            sync_dt[cam].append(t_img - t_ref)
            for t in res.tags:
                counts[cam][t.tag_id] += 1
            sec = int(np.floor(t_img))
            stamp = Time(sec=sec, nanosec=int(round((t_img - sec) * 1e9)) % 1_000_000_000)
            writer.write(tags_topic[cam], serialize_message(dets[cam].tags_msg(res, stamp)), t_recv)

    while reader.has_next():
        topic, data, t_recv = reader.read_next()
        if topic in sync_refs:            # a reference detection: ask the synced camera
            ref = deserialize_message(data, AprilTagDetectionArray)
            cam = sync_refs[topic]
            dets[cam].sync.add_reference(ref.header.stamp.sec + 1e-9 * ref.header.stamp.nanosec)
            emit_synced(cam, dets[cam].sync.pop_ready(), t_recv)
            if topic not in tags_cam:     # the reference camera itself is not re-detected
                continue
        if topic in image_cam:
            cam = image_cam[topic]
            msg = deserialize_message(data, CompressedImage)
            if dets[cam].sync_to:
                dets[cam].sync.add_image(msg.header.stamp.sec + 1e-9 * msg.header.stamp.nanosec,
                                         bytes(msg.data))
                emit_synced(cam, dets[cam].sync.pop_ready(), t_recv)
                continue
            buf = images[cam]
            buf[(msg.header.stamp.sec, msg.header.stamp.nanosec)] = bytes(msg.data)
            if len(buf) > 200:           # ~13 s at 15 Hz: far beyond the detector latency
                del buf[next(iter(buf))]
            continue
        if topic in tags_topic.values() and topic not in tags_cam:
            continue                      # a synced camera's own live output: replaced
        if topic in tags_cam:
            cam = tags_cam[topic]
            live = deserialize_message(data, AprilTagDetectionArray)
            key = (live.header.stamp.sec, live.header.stamp.nanosec)
            jpeg = images[cam].get(key)
            n_frames[cam] += 1
            res = dets[cam].detect(jpeg) if jpeg is not None else None
            if res is None:
                # The image the live detector used is not in the bag (dropped by
                # the recorder): keep the keyframe, with no detections.
                n_missing[cam] += 1
                res = type("Empty", (), {"tags": []})()
            for t in res.tags:
                counts[cam][t.tag_id] += 1
            if compare:
                ref = {d.id: np.array([[c.x, c.y] for c in d.corners]) for d in live.detections}
                for t in res.tags:
                    if t.tag_id in ref:
                        corner_err[cam].append(float(np.abs(ref[t.tag_id] - t.corners).max()))
            writer.write(topic, serialize_message(dets[cam].tags_msg(res, live.header.stamp)), t_recv)
            if sum(n_frames.values()) % 500 == 0:
                print(f"  {sum(n_frames.values())} frames, {(t_recv - t0_ns) * 1e-9:.0f} s into "
                      f"the bag, {time.perf_counter() - t_start:.0f} s elapsed")
            continue
        writer.write(topic, data, t_recv)
    for cam in sync_refs.values():
        emit_synced(cam, dets[cam].sync.flush(), t_recv)
    writer.close()        # flush and write the metadata before source.yaml goes in

    with open(os.path.join(out, "source.yaml"), "w", encoding="utf-8") as f:
        f.write(f"source_bag: {os.path.basename(os.path.normpath(bag))}\n"
                f"source_path: {os.path.abspath(bag)}\n"
                f"start_offset: 0.0\n"
                f"redetected: {', '.join(f'{c}={sorted(dets[c].sizes)}' for c in cameras)}\n")
    for cam in cameras:
        line = (f"{cam}: {n_frames[cam]} frames, tags {dict(sorted(counts[cam].items()))}, "
                f"{n_missing[cam]} images missing")
        if cam in sync_dt and sync_dt[cam]:
            d = np.abs(np.array(sync_dt[cam])) * 1000
            line += (f"; synced to {dets[cam].sync_to}: |image - reference| median {np.median(d):.0f}"
                     f" / max {d.max():.0f} ms, {dets[cam].sync.n_unmatched} references unmatched")
        if compare and corner_err[cam]:
            e = np.array(corner_err[cam])
            line += (f"; vs recorded corners: median {np.median(e):.3g} / max {e.max():.3g} px "
                     f"({len(e)} matched)")
        print(line)
    print(f"wrote {out} in {time.perf_counter() - t_start:.0f} s")
    return counts


def main(argv: List[str] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("bag", help="recorded bag with the camera images and the live detector output")
    ap.add_argument("--out", required=True, help="output bag directory (must not exist)")
    ap.add_argument("--detector-config", nargs="+", default=None,
                    help="detector parameter files, merged in order (default: race_auv_bringup's "
                         "station_tags.yaml tag_factorgraph_detector.yaml)")
    ap.add_argument("--cameras", nargs="+", default=["cam_front", "cam_down"])
    ap.add_argument("--camera-tags", nargs="*", default=[], metavar="CAM=ID,ID",
                    help="override a camera's camera_tag_ids, e.g. cam_down=176,185")
    ap.add_argument("--compare", action="store_true",
                    help="report corner differences against the recorded detections")
    args = ap.parse_args(argv)
    overrides = {}
    for s in args.camera_tags:
        cam, ids = s.split("=")
        overrides[cam] = [int(i) for i in ids.split(",") if i]
    redetect(os.path.expanduser(args.bag), os.path.expanduser(args.out),
             args.detector_config or default_detector_config(), args.cameras, overrides,
             args.compare)


if __name__ == "__main__":
    main()
