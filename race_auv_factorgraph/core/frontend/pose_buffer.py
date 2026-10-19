"""Time-indexed buffer of EKF poses with interpolation (§6 step 1).

Keyframes are created at camera timestamps, which never coincide with EKF
message times. The estimator needs ``E(t)``, the EKF pose of ``base_link`` in
``odom`` at exactly the image stamp ``t``. This buffer keeps the recent EKF
history and interpolates between the two messages that bracket ``t``.

There is no extrapolation. If ``t`` is newer than the newest EKF message,
:meth:`PoseBuffer.interpolate` returns ``None`` and the caller waits for the
next EKF message (§6: "wait one EKF period").
"""

from __future__ import annotations

import bisect
from typing import List, Optional

import gtsam

from ..common.geometry import interpolate


class PoseBuffer:
    """Sorted ``(stamp, Pose3)`` history, pruned to ``max_age`` seconds."""

    def __init__(self, max_age: float = 30.0) -> None:
        self.max_age = float(max_age)
        self._stamps: List[float] = []
        self._poses: List[gtsam.Pose3] = []

    def __len__(self) -> int:
        return len(self._stamps)

    @property
    def newest_stamp(self) -> Optional[float]:
        return self._stamps[-1] if self._stamps else None

    @property
    def oldest_stamp(self) -> Optional[float]:
        return self._stamps[0] if self._stamps else None

    def add(self, stamp: float, pose: gtsam.Pose3) -> None:
        """Insert a pose. Out-of-order inserts are kept sorted; exact duplicates replace."""
        stamp = float(stamp)
        i = bisect.bisect_left(self._stamps, stamp)
        if i < len(self._stamps) and self._stamps[i] == stamp:
            self._poses[i] = pose
        else:
            self._stamps.insert(i, stamp)
            self._poses.insert(i, pose)
        # Drop history older than max_age relative to the newest sample.
        cutoff = self._stamps[-1] - self.max_age
        n_old = bisect.bisect_left(self._stamps, cutoff)
        if n_old > 0:
            del self._stamps[:n_old]
            del self._poses[:n_old]

    def interpolate(self, stamp: float) -> Optional[gtsam.Pose3]:
        """Return ``E(stamp)``, or ``None`` if ``stamp`` is outside the buffered range."""
        if not self._stamps:
            return None
        stamp = float(stamp)
        if stamp < self._stamps[0] or stamp > self._stamps[-1]:
            return None
        i = bisect.bisect_left(self._stamps, stamp)
        if self._stamps[i] == stamp:
            return self._poses[i]
        # stamps[i-1] < stamp < stamps[i]
        t0, t1 = self._stamps[i - 1], self._stamps[i]
        alpha = (stamp - t0) / (t1 - t0)
        return interpolate(self._poses[i - 1], self._poses[i], alpha)

    def clear(self) -> None:
        self._stamps.clear()
        self._poses.clear()
