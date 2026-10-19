"""Station state machine and the initialization buffer (§7).

::

    NO_STATION --first usable tag frame--> CANDIDATE --batch solve accepted--> TRACKING
                                               |
                                               +--window expires without a winner: slide the window

* ``NO_STATION``: no tag seen yet. The graph holds only the ``X`` chain and
  tracks the EKF.
* ``CANDIDATE``: tag frames are buffered (not dropped) and, after each new
  frame, voting + batch solve are attempted (:func:`init.try_initialize`).
* ``TRACKING``: ``S`` and the ``T_j`` exist in the graph. Tag observations
  go through gates 3 (and 4) and become projection factors directly.

The window slides two ways: frames older than ``init_window_duration`` are
dropped, and so are the oldest frames beyond ``init_window_max_frames``. A
rejected batch solve also drops the oldest frame, so a single bad frame
cannot block initialization forever.
"""

from __future__ import annotations

import enum
from typing import Dict, List

import gtsam

from ..common.config import EstimatorConfig
from ..common.model import CameraModel, TagKey, TagModel
from .solver import BufferedFrame, InitDecision, try_initialize


class StationState(enum.Enum):
    NO_STATION = "NO_STATION"
    CANDIDATE = "CANDIDATE"
    TRACKING = "TRACKING"


class StationInitializer:
    """Holds the buffered tag frames and drives the state machine up to TRACKING."""

    def __init__(self, cfg: EstimatorConfig) -> None:
        self.cfg = cfg
        self.state = StationState.NO_STATION
        self.frames: List[BufferedFrame] = []
        self.last_reason = ""

    def reset(self) -> None:
        self.state = StationState.NO_STATION
        self.frames.clear()
        self.last_reason = ""

    def _slide(self, now: float) -> None:
        """Drop frames that fell out of the time window or the frame budget."""
        cutoff = now - self.cfg.init_window_duration
        self.frames = [f for f in self.frames if f.stamp >= cutoff]
        excess = len(self.frames) - int(self.cfg.init_window_max_frames)
        if excess > 0:
            self.frames = self.frames[excess:]

    def add_frame(self, frame: BufferedFrame, X_est: Dict[int, gtsam.Pose3],
                  cams: Dict[str, CameraModel], tags: Dict[TagKey, TagModel]
                  ) -> InitDecision:
        """Buffer one frame and try to initialize.

        ``X_est`` maps keyframe index -> current backend estimate of ``X_k``
        for every buffered frame (plus the new one). The estimator refreshes
        it before each call, because iSAM2 keeps refining those poses.
        """
        if self.state == StationState.TRACKING:
            raise RuntimeError("add_frame called while TRACKING")
        self.frames.append(frame)
        self._slide(frame.stamp)
        if self.state == StationState.NO_STATION and frame.hypotheses:
            self.state = StationState.CANDIDATE

        usable = [f for f in self.frames if f.hypotheses]
        decision = try_initialize(usable, X_est, cams, tags, self.cfg)
        self.last_reason = decision.reason
        if decision.accepted:
            self.state = StationState.TRACKING
        elif decision.reason.startswith(("batch", "ambiguous")):
            # Voting was conclusive but the batch solve said no: forget the
            # oldest frame so the window can move past a bad stretch.
            self.frames = self.frames[1:]
        if not self.frames and self.state == StationState.CANDIDATE:
            self.state = StationState.NO_STATION
        return decision

    def buffered_keyframes(self) -> List[int]:
        return [f.k for f in self.frames]
