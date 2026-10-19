"""Minimal URDF reader: rigid link-to-link transforms and station tag poses.

The node needs three things from URDFs (§3):

* ``T_bc`` for each camera, from the AUV URDF (``base_link -> cam_front`` ...).
* The pose of every tag relative to ``dock_point``, from the station URDF.
* ``T_dock_base``: the station ``base_link`` expressed in ``dock_point``,
  so the node can publish ``odom -> station base_link`` (§10).

Only ``<joint>`` origins are used. Every joint is taken at its zero position,
so a chain through a moving joint is evaluated at that joint's zero
configuration. The camera and tag chains here are all ``fixed`` joints. The
reader uses the standard-library XML parser and does not run xacro: point it
at a plain ``.urdf`` file.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import gtsam

from .geometry import pose_from_xyz_rpy
from .model import TagKey

# Station tag links are named "apriltag<family>_<id>", e.g. "apriltag36h11_146".
_TAG_LINK_RE = re.compile(r"^apriltag(?P<family>\d+h\d+)_(?P<id>\d+)$")


@dataclass
class _Joint:
    name: str
    parent: str
    child: str
    T_parent_child: gtsam.Pose3


class Urdf:
    """A parsed URDF kinematic tree (joint origins only)."""

    def __init__(self, xml_text: str) -> None:
        root = ET.fromstring(xml_text)
        self.links: List[str] = [l.get("name") for l in root.findall("link")]
        # child link -> the joint that attaches it to its parent. A URDF is a
        # tree, so every link except the root has exactly one parent joint.
        self._parent_joint: Dict[str, _Joint] = {}
        for j in root.findall("joint"):
            origin = j.find("origin")
            xyz = (0.0, 0.0, 0.0)
            rpy = (0.0, 0.0, 0.0)
            if origin is not None:
                xyz = tuple(float(v) for v in origin.get("xyz", "0 0 0").split())
                rpy = tuple(float(v) for v in origin.get("rpy", "0 0 0").split())
            joint = _Joint(
                name=j.get("name"),
                parent=j.find("parent").get("link"),
                child=j.find("child").get("link"),
                T_parent_child=pose_from_xyz_rpy(xyz, rpy),
            )
            self._parent_joint[joint.child] = joint

    @classmethod
    def from_file(cls, path: str) -> "Urdf":
        with open(path, "r", encoding="utf-8") as f:
            return cls(f.read())

    def _pose_in_root(self, link: str) -> Tuple[str, gtsam.Pose3]:
        """Walk up from ``link`` to the tree root.

        Returns ``(root_name, T_root_link)``.
        """
        if link not in self.links and link not in self._parent_joint:
            raise KeyError(f"link '{link}' not found in URDF")
        T = gtsam.Pose3()
        current = link
        # Prepend each parent joint: T_root_link = T_root_a * T_a_b * ... * T_y_link
        while current in self._parent_joint:
            joint = self._parent_joint[current]
            T = joint.T_parent_child.compose(T)
            current = joint.parent
        return current, T

    def transform(self, base: str, link: str) -> gtsam.Pose3:
        """Pose of ``link`` expressed in ``base`` (``T_base_link``).

        Both links are expressed in the common tree root, then
        ``T_base_link = T_root_base^-1 * T_root_link``. This also works when
        ``base`` is not an ancestor of ``link`` (e.g. ``dock_point -> base_link``).
        """
        root_a, T_root_base = self._pose_in_root(base)
        root_b, T_root_link = self._pose_in_root(link)
        if root_a != root_b:
            raise ValueError(f"links '{base}' and '{link}' are in different trees")
        return T_root_base.between(T_root_link)

    def tag_links(self) -> Dict[TagKey, str]:
        """All links named ``apriltag<family>_<id>``, keyed by ``(family, id)``."""
        out: Dict[TagKey, str] = {}
        for link in self.links:
            m = _TAG_LINK_RE.match(link or "")
            if m:
                out[(m.group("family"), int(m.group("id")))] = link
        return out

    def tag_poses(self, frame: str, families: Optional[List[str]] = None
                  ) -> Dict[TagKey, gtsam.Pose3]:
        """Pose of every tag link in ``frame`` (normally ``dock_point``)."""
        return {
            key: self.transform(frame, link)
            for key, link in self.tag_links().items()
            if families is None or key[0] in families
        }
