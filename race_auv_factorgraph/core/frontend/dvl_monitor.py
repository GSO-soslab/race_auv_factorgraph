"""DVL bottom-lock monitor: how much to distrust the EKF odometry over a keyframe step (§5).

Why
---
robot_localization fuses the DVL's bottom-lock velocity (``twist0``). The
DVL driver publishes ``bottom_lock_velocity_common`` only while it has bottom
lock, so a dropout is simply **no messages**. During a dropout the EKF dead-
reckons: it integrates the IMU accelerations into velocity and velocity into
position, and its position drifts (an accelerometer bias b gives about
b * t^2 / 2). When the DVL comes back, the Kalman update sees a large velocity
innovation, and because position and velocity became strongly correlated
during the dropout, it corrects the **position** too, in one step: the EKF
"snaps" by roughly the drift it accumulated.

Tank run 2026-09-16 16:16:47: every EKF step over 10 cm comes 0.01-0.04 s
after the first DVL message that ends a silence, and is at most about
0.1 m/s times the silence length (11.2 s -> 1.05 m, 4.8 s -> 14 cm, 3.6 s ->
14 cm, 2.0 s -> 4 cm).

The individual EKF steps over a dropout are wrong both ways (the drift,
then the snap that undoes it). So:

* steps **inside** a silence (the drift) get extra translation sigma
  ``dvl_blind_speed * (time of the step spent blind)``;
* the step(s) containing the **re-lock** (the snap) get extra sigma
  ``dvl_blind_speed * (length of the whole silence)``, because the snap
  undoes the drift of the whole silence at once. A step counts as
  containing the re-lock if it overlaps ``[relock, relock + dvl_relock_margin]``
  (the EKF reacts within ~0.05 s; a small follow-up step came 0.2 s later).

De-weighting only changes the estimate where tags compete with the odometry.
Without tags the graph follows the EKF's mean motion, snap included, so it
still ends up where the EKF snapped back to. (A factor spanning the whole
dropout, carrying the EKF's net motion, was tried: ~0.3 cm better median in
the synthetic tests, no change at p95 or max, so it was left out.)

Definitions: a *silence* is a gap between two DVL messages longer than
``dvl_timeout``; the AUV is *blind* from ``dvl_timeout`` after the last
message until the next one. Only header stamps are used (the DVL is stamped
on the same front-seat computer as the EKF).

Plain data in, plain data out (no ROS), shared by the node, the bag replay
and the synthetic runs.
"""

from __future__ import annotations

import bisect
from typing import List, Tuple

from ..common.config import EstimatorConfig


class DvlMonitor:
    """History of DVL message stamps, and the extra odometry sigma per keyframe step."""

    def __init__(self, cfg: EstimatorConfig, buffer_seconds: float = 120.0) -> None:
        self.cfg = cfg
        self.buffer_seconds = float(buffer_seconds)
        self._stamps: List[float] = []      # sorted DVL header stamps

    def add(self, stamp: float) -> None:
        """Record one DVL bottom-lock message by its header stamp."""
        if self._stamps and stamp <= self._stamps[-1]:
            if stamp < self._stamps[-1]:
                bisect.insort(self._stamps, stamp)   # out of order: keep the list sorted
            return
        self._stamps.append(stamp)
        cutoff = stamp - self.buffer_seconds
        if self._stamps[0] < cutoff:
            del self._stamps[:bisect.bisect_left(self._stamps, cutoff)]

    @property
    def active(self) -> bool:
        """True once any DVL message has arrived (otherwise the monitor does nothing)."""
        return bool(self._stamps)

    def _silences(self, t0: float, t1: float) -> List[Tuple[float, float, bool]]:
        """Silences relevant to the step ``(t0, t1]``: ``(last msg, next msg or t1, ended)``."""
        cfg, s = self.cfg, self._stamps
        out = []
        lo = max(0, bisect.bisect_left(s, t0 - self.buffer_seconds) - 1)
        hi = bisect.bisect_right(s, t1)
        for i in range(lo, hi - 1):
            a, b = s[i], s[i + 1]
            if b - a > cfg.dvl_timeout and b + cfg.dvl_relock_margin > t0 and a + cfg.dvl_timeout < t1:
                out.append((a, b, True))
        # A silence still going on at t1 (no message since more than the timeout).
        last = s[hi - 1] if hi > 0 else None
        if last is not None and t1 - last > cfg.dvl_timeout:
            out.append((last, t1, False))
        return out

    def extra_sigma(self, t0: float, t1: float) -> float:
        """Extra translation sigma (m) for the EKF odometry over the keyframe step ``(t0, t1]``.

        0 when the DVL had bottom lock throughout (or no DVL message has
        ever arrived: monitor inactive, e.g. no DVL topic configured).
        """
        if not self.active or t1 <= t0:
            return 0.0
        cfg = self.cfg
        extra = 0.0
        for a, b, ended in self._silences(t0, t1):
            blind_from, blind_to = a + cfg.dvl_timeout, b
            overlap = max(0.0, min(t1, blind_to) - max(t0, blind_from))
            extra += cfg.dvl_blind_speed * overlap                  # drift while blind
            if ended and b <= t1 and b + cfg.dvl_relock_margin > t0:
                extra += cfg.dvl_blind_speed * (b - a)               # the snap at re-lock
        return extra
