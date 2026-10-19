"""Symbol helpers and timestamp bookkeeping for the graph variables (§4).

======== ============ ============================================================
Symbol   Type         Meaning
======== ============ ============================================================
``x<k>`` ``Pose3``    AUV ``base_link`` in ``odom`` at keyframe k (one per keyframe).
``s0``   ``Pose3``    Station ``dock_point`` in ``odom`` (one, permanent).
``t<i>`` ``Pose3``    Pose of tag i relative to ``dock_point`` (one per tag, permanent).
======== ============ ============================================================

Tag variables are numbered by order of first sighting, not by AprilTag id,
because ids are only unique within a family. :class:`TagIndex` keeps the
``(family, id) -> index`` mapping.
"""

from __future__ import annotations

from typing import Dict, Iterable, List

import gtsam

from .model import TagKey


def X(k: int) -> int:
    """Key of the AUV pose at keyframe ``k``."""
    return gtsam.symbol("x", k)


# Key of the station pose S. There is exactly one station.
S: int = gtsam.symbol("s", 0)


class TagIndex:
    """Assigns a stable gtsam key ``t<i>`` to every tag that enters the graph."""

    def __init__(self) -> None:
        self._index: Dict[TagKey, int] = {}

    def key(self, tag: TagKey) -> int:
        """Key of tag ``tag``; allocates a new index on first use."""
        if tag not in self._index:
            self._index[tag] = len(self._index)
        return gtsam.symbol("t", self._index[tag])

    def __contains__(self, tag: TagKey) -> bool:
        return tag in self._index

    def tags(self) -> List[TagKey]:
        """All tags that have a key, in order of first use."""
        return list(self._index)

    def keys(self) -> List[int]:
        return [gtsam.symbol("t", i) for i in self._index.values()]

    def clear(self) -> None:
        self._index.clear()


def describe(key: int) -> str:
    """Readable key name for logs, e.g. ``"x12"``."""
    return str(gtsam.Symbol(key).string())


def restamp(timestamps: Dict[int, float], keys: Iterable[int], stamp: float) -> None:
    """Set ``timestamps[key] = stamp`` for every key.

    The fixed-lag smoother marginalizes any variable whose timestamp is older
    than ``now - lag``. ``S`` and the ``T_j`` must never be marginalized, so
    they are re-stamped with the current time on every update (§9). A far
    future timestamp must *not* be used instead: the smoother treats the
    newest timestamp as "now" and would marginalize everything else at once.
    """
    for key in keys:
        timestamps[key] = stamp
