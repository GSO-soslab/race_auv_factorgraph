"""Plain data types shared by the core modules (no ROS, no logic).

These are the only objects that cross the boundary between the ROS node (or
the offline bag replay) and the estimator core:

* :class:`CameraModel`: one rectified pinhole camera plus its mounting on the AUV.
* :class:`TagModel`: one AprilTag on the station (size + URDF mounting).
* :class:`Detection`: one tag seen in one image (4 corner pixels).
* :class:`TagFrame`: all detections from one camera image.

Tags are identified by a :data:`TagKey` ``(family, id)``, with the family
normalized to the short form used in the station URDF link names (``"36h11"``,
not ``"tag36h11"``). See :func:`normalize_family`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import gtsam
import numpy as np

# (family, id), e.g. ("36h11", 146)
TagKey = Tuple[str, int]


def normalize_family(family: str) -> str:
    """Return the short family name: ``"tag36h11"`` -> ``"36h11"``.

    The detector publishes the apriltag library's family name (``tag36h11``),
    while the station URDF names its links ``apriltag36h11_<id>``. Normalizing
    both sides to ``36h11`` lets them be matched.
    """
    family = family.strip()
    return family[3:] if family.lower().startswith("tag") else family


def tag_key_str(key: TagKey) -> str:
    """Human-readable tag name for logs, e.g. ``"36h11:146"``."""
    return f"{key[0]}:{key[1]}"


def tag_corner_points(size: float) -> np.ndarray:
    """The 4 corners of a tag in its own frame, as a ``(4, 3)`` array.

    ``size`` is the edge length of the **black square**: that is where the
    detector places the corners, not at the outer white border.

    The order is the canonical corner order used everywhere in this package
    (and published by ``apriltag_detector_node`` on its ``tags`` topic):

    ====== ====================== ==============================
    index  tag-frame point        meaning (tag frame: x right,
                                  y down, z into the tag)
    ====== ====================== ==============================
    0      (-s/2, +s/2, 0)        left-bottom
    1      (+s/2, +s/2, 0)        right-bottom
    2      (+s/2, -s/2, 0)        right-top
    3      (-s/2, -s/2, 0)        left-top
    ====== ====================== ==============================

    This is the order of the AprilTag 3 library's ``det->p[]`` corners
    (``lb-rb-rt-lt``) and of the object points its ``estimate_tag_pose``
    uses. It is also the order ``cv2.SOLVEPNP_IPPE_SQUARE`` requires, so the
    same array feeds both the graph's projection factor and the PnP
    initialization.
    """
    h = 0.5 * float(size)
    return np.array([
        [-h, +h, 0.0],
        [+h, +h, 0.0],
        [+h, -h, 0.0],
        [-h, -h, 0.0],
    ])


@dataclass
class CameraModel:
    """A rectified pinhole camera rigidly mounted on the AUV.

    Attributes
    ----------
    name
        Config name, e.g. ``"cam_front"``.
    fx, fy, cx, cy
        Camera matrix of the **rectified** image the detector runs on (§3).
        The detector undistorts before detecting, so the graph uses a plain
        pinhole model with no distortion terms.
    width, height
        Rectified image size in pixels.
    T_bc
        ``base_link -> camera`` extrinsic (pose of the camera optical frame
        in ``base_link``), from the AUV URDF. Held constant in the graph.
        The camera frame is the optical convention: z forward, x right,
        y down.
    """

    name: str
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    T_bc: gtsam.Pose3

    @property
    def K(self) -> np.ndarray:
        """3x3 camera matrix, in the form OpenCV expects."""
        return np.array([[self.fx, 0.0, self.cx],
                         [0.0, self.fy, self.cy],
                         [0.0, 0.0, 1.0]])


@dataclass
class TagModel:
    """One AprilTag on the docking station.

    Attributes
    ----------
    key
        ``(family, id)``.
    size
        Black-square edge length in metres.
    T_dock_tag
        Pose of the tag frame in the station's ``dock_point`` frame, from the
        station URDF. It is the *prior* value of the graph variable ``T_j``
        (the real mounting may differ by a few mm / degrees).
    """

    key: TagKey
    size: float
    T_dock_tag: gtsam.Pose3
    corners: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        # Cache the (4, 3) corner layout in the tag frame.
        self.corners = tag_corner_points(self.size)

    def corners_in_dock(self) -> np.ndarray:
        """Corner points expressed in the ``dock_point`` frame, ``(4, 3)``.

        Used for multi-tag PnP, where all corners must share one object frame.
        """
        return np.array([self.T_dock_tag.transformFrom(c) for c in self.corners])


@dataclass
class Detection:
    """One tag observed in one rectified image.

    ``corners`` is a ``(4, 2)`` array of pixel coordinates in the canonical
    order of :func:`tag_corner_points`.
    """

    camera: str
    tag: TagKey
    corners: np.ndarray

    def area(self) -> float:
        """Area of the corner quadrilateral in px^2 (shoelace formula)."""
        x, y = self.corners[:, 0], self.corners[:, 1]
        return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))

    def min_side_px(self) -> float:
        """Length (px) of the shortest of the 4 sides of the corner quadrilateral.

        PnP range noise scales with its inverse (range ~ f * size / side).
        """
        c = self.corners
        return float(np.min(np.linalg.norm(c - np.roll(c, -1, axis=0), axis=1)))


@dataclass
class TagFrame:
    """All tag detections from one camera image.

    ``stamp`` is the image capture time (seconds), which is the time the
    keyframe is created at (§6).
    """

    stamp: float
    camera: str
    detections: List[Detection]
