"""Parameter defaults and model construction shared by the node and the bag replay.

The node reads parameters from ROS. ``tools/replay_bag.py`` reads the same
YAML file directly. Both go through :func:`build_setup` with a ``get(name)``
accessor, so live and offline runs build identical cameras, tags and
estimator configuration.

Parameter names are flat and dotted, exactly as ROS 2 sees them
(``cameras.cam_front.fx``, ``estimator.sigma_px``, ...).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from typing import Any, Callable, Dict, List, Sequence, Union

import gtsam
import yaml

from ..core.common.config import EstimatorConfig, config_from_dict
from ..core.common.model import CameraModel, TagKey, TagModel, normalize_family, tag_key_str
from ..core.common.urdf import Urdf

# Top-level node parameters and their defaults (see config/dock_fuser_sim.yaml
# for what each one means).
NODE_DEFAULTS: Dict[str, Any] = {
    "odom_topic": "/race_auv/odometry/filtered",
    # DVL bottom-lock velocity (published only while the DVL has bottom
    # lock); "" disables the DVL dropout handling (core/frontend/dvl_monitor.py).
    "dvl_topic": "",
    "graph_frame": "race_auv/odom",
    "base_frame": "race_auv/base_link",
    "station_base_frame": "race_station/base_link",
    "dock_frame": "race_station/dock_point",
    "publish_station_static_tf": True,
    "dock_pose_topic": "dock_point/pose",
    # URDFs: an explicit path, or package + file inside its share directory.
    "auv_urdf_path": "",
    "auv_urdf_package": "race_auv_description",
    "auv_urdf_file": "urdf/base.urdf",
    "auv_base_link": "base_link",
    "station_urdf_path": "",
    "station_urdf_package": "race_station_description",
    "station_urdf_file": "urdf/base.urdf",
    "station_base_link": "base_link",
    "station_dock_link": "dock_point",
    # Tags used by the graph: the URDF gives their poses, these lists their
    # black-square sizes (m). A tag missing here is ignored even if detected.
    "tag_family": "36h11",
    "tag_ids": [146, 176, 185, 541, 558],
    "tag_sizes": [0.12, 0.12, 0.12, 0.04, 0.04],
    # Exactly one camera: its detector messages are the keyframe clock
    # (core/frontend/scheduler.py), see build_setup.
    "camera_names": ["cam_front"],
    # Keyframe scheduling: seconds of EKF history kept for interpolating E(t)
    # at each image stamp. Must exceed the worst detector + transport delay.
    "ekf_buffer_seconds": 30.0,
    # Rate limit on the published odom -> station correction (§10).
    "output_max_trans_rate": 0.5,   # m/s
    "output_max_rot_rate": 0.5,     # rad/s
    # Extra per-keyframe topics for analysis and live viewing (Foxglove):
    # ~/auv_pose, ~/station_pose, ~/keyframe, ~/odometry, ~/path and
    # ~/ekf_path. See ros/node.py.
    "publish_debug": False,
    # ~/path and ~/ekf_path: re-published at path_rate Hz, holding the last
    # path_length keyframes (3000 = 10 min at 5 Hz, ~250 kB per message each).
    "path_rate": 1.0,
    "path_length": 3000,
}


def camera_defaults(name: str) -> Dict[str, Any]:
    """Per-camera parameters, ``cameras.<name>.<key>``.

    The camera matrix (``fx .. height``) is that of the RECTIFIED image the
    corners are in. Leave it at 0 (the default) to take it from the tag
    detector's latched ``camera_info``, the single source of truth; set it
    only where no ``camera_info`` exists (simulation with the legacy
    detector, replays of old detection bags).
    """
    return {
        f"cameras.{name}.detections_topic": f"/{name}/apriltag_detection/tags",
        # "" = the detections topic's sibling, /<cam>/apriltag_detection/camera_info
        f"cameras.{name}.camera_info_topic": "",
        f"cameras.{name}.frame": f"race_auv/{name}",   # header.frame_id of the detections
        f"cameras.{name}.urdf_link": "",               # default: frame after the last '/'
        f"cameras.{name}.fx": 0.0,
        f"cameras.{name}.fy": 0.0,
        f"cameras.{name}.cx": 0.0,
        f"cameras.{name}.cy": 0.0,
        f"cameras.{name}.width": 0,
        f"cameras.{name}.height": 0,
    }


def estimator_defaults() -> Dict[str, Any]:
    """Every :class:`EstimatorConfig` field as ``estimator.<field>``."""
    d = EstimatorConfig()
    return {f"estimator.{f.name}": getattr(d, f.name) for f in fields(EstimatorConfig)}


@dataclass
class Setup:
    cfg: EstimatorConfig
    # camera name -> model. A camera whose intrinsics come from camera_info
    # is missing here until add_camera_info() has run for it.
    cameras: Dict[str, CameraModel]
    camera_frames: Dict[str, str]     # camera name -> expected header.frame_id
    camera_topics: Dict[str, str]     # camera name -> detections topic
    camera_info_topics: Dict[str, str]  # camera name -> rectified camera_info topic
    camera_extrinsics: Dict[str, gtsam.Pose3]  # camera name -> T_bc from the AUV URDF
    tags: Dict[TagKey, TagModel]
    T_dock_base: gtsam.Pose3          # station base_link in dock_point
    # tag -> its station URDF link name (e.g. "apriltag36h11_146"), for the
    # node's tag TF frames.
    tag_links: Dict[TagKey, str] = field(default_factory=dict)

    def missing_intrinsics(self) -> List[str]:
        """Cameras still waiting for their camera_info."""
        return [n for n in self.camera_topics if n not in self.cameras]

    def add_camera_info(self, name: str, k: Sequence[float], width: int, height: int) -> CameraModel:
        """Complete camera ``name`` from a rectified ``sensor_msgs/CameraInfo`` (``k``, size).

        ``k`` is the row-major 3x3 camera matrix. The detector's camera_info
        describes its rectified image (zero distortion), so ``k`` is the
        pinhole model the corners follow. The model is inserted into
        ``self.cameras`` in place, so an estimator built with that dict sees it.
        """
        cam = CameraModel(name, float(k[0]), float(k[4]), float(k[2]), float(k[5]),
                          int(width), int(height), self.camera_extrinsics[name])
        self.cameras[name] = cam
        return cam


def resolve_urdf(get: Callable[[str], Any], which: str) -> str:
    """Path of the ``auv`` or ``station`` URDF from ``<which>_urdf_path`` or package + file."""
    path = get(f"{which}_urdf_path")
    if path:
        return path
    from ament_index_python.packages import get_package_share_directory
    return os.path.join(get_package_share_directory(get(f"{which}_urdf_package")),
                        get(f"{which}_urdf_file"))


def build_setup(get: Callable[[str], Any], warn: Callable[[str], None]) -> Setup:
    """Load URDFs and parameters into the core model objects.

    ``get(name)`` returns a parameter value (already defaulted);
    ``warn(msg)`` reports non-fatal problems.
    """
    # One camera only. Each detector message becomes one keyframe at its image
    # stamp; a second, unsynchronized camera would interleave its frames
    # with the first one's, and out-of-order arrivals would be dropped.
    # Supporting it needs a reordering step in the scheduler (§6).
    if len(get("camera_names")) != 1:
        raise ValueError(f"camera_names must list exactly one camera, got {list(get('camera_names'))}: "
                         "keyframes follow a single detector's messages (see core/frontend/scheduler.py)")
    cfg = config_from_dict({k.split(".", 1)[1]: get(k) for k in estimator_defaults()})
    auv = Urdf.from_file(resolve_urdf(get, "auv"))
    station = Urdf.from_file(resolve_urdf(get, "station"))
    dock_link = get("station_dock_link")

    # ---- cameras: T_bc from the AUV URDF; rectified intrinsics from the
    #      parameters if given, else later from the detector's camera_info
    cameras, frames, topics, info_topics, extrinsics = {}, {}, {}, {}, {}
    for name in get("camera_names"):
        v = lambda key: get(f"cameras.{name}.{key}")  # noqa: E731
        link = v("urdf_link") or v("frame").split("/")[-1]
        extrinsics[name] = auv.transform(get("auv_base_link"), link)
        frames[name] = v("frame")
        topics[name] = v("detections_topic")
        info_topics[name] = v("camera_info_topic") or (
            topics[name].rsplit("/", 1)[0] + "/camera_info")
        if min(v("fx"), v("fy"), v("width"), v("height")) > 0:
            cameras[name] = CameraModel(name, float(v("fx")), float(v("fy")), float(v("cx")),
                                        float(v("cy")), int(v("width")), int(v("height")),
                                        extrinsics[name])

    # ---- tags: sizes from the parameters, poses from the station URDF
    family = normalize_family(get("tag_family"))
    ids: List[int] = list(get("tag_ids"))
    sizes: List[float] = list(get("tag_sizes"))
    if len(ids) != len(sizes):
        raise ValueError("tag_ids and tag_sizes must have the same length")
    poses = station.tag_poses(dock_link)
    tags = {}
    for tag_id, size in zip(ids, sizes):
        key = (family, int(tag_id))
        if key not in poses:
            # Without a URDF pose the tag cannot be used. Warn instead of
            # failing, so one config can list tags of several station builds.
            warn(f"tag {tag_key_str(key)} has no 'apriltag{family}_<id>' link in the station "
                 "URDF; it will be ignored")
            continue
        tags[key] = TagModel(key, float(size), poses[key])
    if not tags:
        raise ValueError("none of tag_ids exists in the station URDF")

    links = station.tag_links()
    return Setup(cfg, cameras, frames, topics, info_topics, extrinsics, tags,
                 station.transform(dock_link, get("station_base_link")),
                 {key: links[key] for key in tags})


def flatten_params_yaml(path: str, node_name: str = "dock_fuser") -> Dict[str, Any]:
    """Read a ROS 2 parameter file into flat dotted names (as the node would see them).

    Reads the ``/**`` section (every node, e.g. the shared station tag list)
    and the node's own section, the node's own winning, as ROS does.
    """
    with open(path, "r", encoding="utf-8") as f:
        doc = yaml.safe_load(f) or {}
    params: Dict[str, Any] = {}
    for section in ("/**", node_name, f"/{node_name}"):
        params.update((doc.get(section) or {}).get("ros__parameters", {}))
    flat: Dict[str, Any] = {}

    def walk(prefix: str, node: Any) -> None:
        if isinstance(node, dict):
            for k, v in node.items():
                walk(f"{prefix}.{k}" if prefix else str(k), v)
        else:
            flat[prefix] = node

    walk("", params)
    return flat


def getter_from_yaml(paths: Union[str, Sequence[str]]) -> Callable[[str], Any]:
    """``get(name)`` over one or more parameter files, falling back to the node defaults.

    Several files are merged in order, later ones winning, exactly like
    ``--params-file a.yaml --params-file b.yaml`` on the node (e.g. the
    shared station_tags.yaml, then dock_fuser.yaml).
    """
    flat: Dict[str, Any] = {}
    for path in ([paths] if isinstance(paths, str) else paths):
        flat.update(flatten_params_yaml(path))
    defaults = {**NODE_DEFAULTS, **estimator_defaults()}
    for name in flat.get("camera_names", NODE_DEFAULTS["camera_names"]):
        defaults.update(camera_defaults(name))
    return lambda name: flat.get(name, defaults.get(name))
