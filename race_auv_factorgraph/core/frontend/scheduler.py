"""Keyframe scheduling: one keyframe per detector message, at its image stamp (§6).

The rule
--------
``apriltag_detector_node`` runs on a fixed timer (``publish_rate``, 5 Hz on
the vehicle). Every tick it detects on the newest image and publishes the
result, *including when no tag was found*. Each such message becomes one
keyframe ``X_k``, stamped with the message's ``header.stamp``, which is the
image capture time. The detector's timer is therefore the graph's clock:

* tags in view   -> keyframe with projection factors (plus odometry);
* no tags in view -> keyframe with odometry only, which keeps the ``X``
  chain alive through blind stretches.

There is no separate keyframe timer, and only one camera drives it: the
*keyframe camera* (node parameter ``keyframe_camera``, the front camera on
the vehicle). A single detector's stamps only increase, so keyframes never
need reordering (§6).

Attached cameras
----------------
Any other camera (e.g. ``cam_down``) is *attached*: its messages never
become keyframes, so the graph's variables stay the keyframe camera's
``X_k`` and its clock stays the keyframe rate. A message with tags waits
only for the EKF to cover its stamp (:meth:`KeyframeScheduler.poll_attached`)
and is then attached to an existing keyframe by the estimator
(``DockEstimator.add_attached_frame``). Nothing waits for the other camera,
so attaching adds no delay to the keyframes. A message without tags carries
nothing for the graph and is discarded on arrival.

Timing across machines
----------------------
On the vehicle the EKF (``robot_localization``) runs on the front-seat Pi
and the camera, detector and this estimator run on the Jetson; messages
cross over zenoh. Only **header stamps** are ever used, never arrival
times, so transport latency and jitter cannot shift a keyframe:

* A frame is held until the EKF buffer holds a pose stamped at or after
  the frame (``E(t)`` is then interpolated between two real EKF poses,
  never extrapolated). If the EKF stream from the Pi is late, frames just
  wait in ``_pending``.
* What stamps cannot fix is a clock *offset* between the two computers: a
  50 ms offset places every ``X_k`` 50 ms off along the trajectory. The
  clocks must be synchronized (chrony / PTP). The node reports
  ``ekf_age_s`` / ``frame_age_s`` in ``/diagnostics`` to expose an offset.

The scheduler is shared by the ROS node, the bag replay and the synthetic
runs, so all three create exactly the same keyframes from the same inputs.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Iterable, List, Optional

import gtsam

from ..common.model import TagFrame
from .pose_buffer import PoseBuffer


@dataclass
class Keyframe:
    stamp: float       # image stamp of the detector message [s]
    E: gtsam.Pose3     # EKF pose odom -> base_link, interpolated at ``stamp``
    frame: TagFrame    # the detector message; ``frame.detections`` may be empty


@dataclass
class AttachedFrame:
    stamp: float       # image stamp of the attached camera's message [s]
    E: gtsam.Pose3     # EKF pose interpolated at ``stamp``
    frame: TagFrame    # the detector message, with at least one detection


class KeyframeScheduler:
    """Turn detector messages + EKF odometry into time-ordered keyframes.

    Usage (identical in the node, the bag replay and the synthetic runs)::

        sched.add_odometry(t, E)     # every EKF message
        sched.add_frame(frame)       # every detector message, empty or not, any camera
        for kf in sched.poll():      # after either of the two
            estimator.process_keyframe(kf.stamp, kf.E, kf.frame)
        for af in sched.poll_attached():   # then the attached cameras' frames
            estimator.add_attached_frame(af.stamp, af.E, af.frame)

    ``attached_cameras`` names the cameras whose frames are attached instead
    of becoming keyframes; every other camera's frames are keyframes.
    """

    def __init__(self, ekf_buffer_seconds: float = 30.0, max_pending: int = 50,
                 attached_cameras: Iterable[str] = ()) -> None:
        self.ekf = PoseBuffer(ekf_buffer_seconds)
        self.attached_cameras = frozenset(attached_cameras)
        # Attached frames waiting for EKF coverage, and the newest stamp seen
        # per attached camera (re-published images are dropped, as for keyframes).
        self._attached: Deque[TagFrame] = deque(maxlen=int(max_pending))
        self._attached_last: Dict[str, float] = {}
        # Frames waiting for the EKF to catch up, oldest first. ``maxlen`` caps
        # memory if the EKF stream dies: the oldest frame is then discarded
        # (50 frames = 10 s at 5 Hz, far beyond any normal EKF latency).
        self._pending: Deque[TagFrame] = deque(maxlen=int(max_pending))
        self.last_stamp: Optional[float] = None   # newest frame accepted by add_frame
        self.n_dropped = 0                        # frames rejected (see add_frame / poll)

    # ------------------------------------------------------------------ inputs
    def add_odometry(self, stamp: float, pose: gtsam.Pose3) -> None:
        """Buffer one EKF pose (``odom -> base_link``) by its header stamp."""
        self.ekf.add(stamp, pose)

    def add_frame(self, frame: TagFrame) -> bool:
        """Queue one detector message. Returns False if it was dropped.

        A frame is dropped when its stamp is not newer than the previous
        frame's. That covers two cases with one comparison:

        * a re-published image: the detector's timer fired before a new
          image arrived, so the same image stamp comes out twice (the same
          ``X_k`` must not be created twice);
        * an out-of-order message, which would break the keyframe order the
          smoother needs.
        """
        if frame.camera in self.attached_cameras:
            return self._add_attached(frame)
        if self.last_stamp is not None and frame.stamp <= self.last_stamp:
            self.n_dropped += 1
            return False
        self.last_stamp = frame.stamp
        if len(self._pending) == self._pending.maxlen:  # EKF stalled: the oldest frame falls out
            self.n_dropped += 1
        self._pending.append(frame)
        return True

    def _add_attached(self, frame: TagFrame) -> bool:
        """Queue an attached camera's frame if it has detections (see the module docstring)."""
        last = self._attached_last.get(frame.camera)
        if last is not None and frame.stamp <= last:
            self.n_dropped += 1
            return False
        self._attached_last[frame.camera] = frame.stamp
        if not frame.detections:
            return False      # nothing for the graph: not a drop, just not used
        if len(self._attached) == self._attached.maxlen:
            self.n_dropped += 1
        self._attached.append(frame)
        return True

    # ------------------------------------------------------------------ output
    @property
    def n_pending(self) -> int:
        """Frames still waiting for EKF coverage (grows if the EKF stalls), attached ones included."""
        return len(self._pending) + len(self._attached)

    def poll(self) -> List[Keyframe]:
        """Release every queued frame the EKF already covers, oldest first.

        A frame at stamp ``t`` is released once the newest EKF stamp is
        ``>= t``. Frames are released strictly in order, so a frame never
        overtakes an older one that is still waiting.
        """
        out: List[Keyframe] = []
        newest = self.ekf.newest_stamp
        while self._pending and newest is not None and self._pending[0].stamp <= newest:
            f = self._pending.popleft()
            E = self.ekf.interpolate(f.stamp)
            if E is None:  # older than the EKF buffer (e.g. frames before the first EKF message)
                self.n_dropped += 1
                continue
            out.append(Keyframe(f.stamp, E, f))
        return out

    def poll_attached(self) -> List[AttachedFrame]:
        """Release every queued attached frame the EKF already covers.

        Call it after :meth:`poll` and after processing the keyframes it
        returned, so the keyframe an attached frame anchors to already
        exists. Order among attached frames does not matter: each one is
        anchored on its own.
        """
        out: List[AttachedFrame] = []
        newest = self.ekf.newest_stamp
        keep: Deque[TagFrame] = deque(maxlen=self._attached.maxlen)
        for f in self._attached:
            if newest is None or f.stamp > newest:
                keep.append(f)
                continue
            E = self.ekf.interpolate(f.stamp)
            if E is None:
                self.n_dropped += 1
                continue
            out.append(AttachedFrame(f.stamp, E, f))
        self._attached = keep
        return out
