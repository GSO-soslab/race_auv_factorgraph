"""PnP helpers: tag and station poses in the camera frame from corner pixels.

PnP is used only where the graph has no estimate yet to linearize at:

* initialization (§7.1): each tag frame yields station hypotheses;
* the tag-implied initial value of a keyframe (translation only).

PnP is never a measurement: the projection factors use the corner pixels
directly, and are linearized at the graph's own estimate.

Single tag: ``cv2.SOLVEPNP_IPPE_SQUARE`` returns **both** solutions of the
planar pose problem. A single square seen roughly face-on has two poses that
fit its corners almost equally well (the "flip ambiguity"): the tag tilted
toward or away from the camera by about the same angle. Which one is right
cannot be decided from one frame. §7 resolves it with several frames.

Several tags: ``cv2.SOLVEPNP_SQPNP`` over all corners expressed in one common
frame (``dock_point``). Non-coplanar tags remove the ambiguity, so one
solution is returned.

All images are rectified, so the distortion coefficients are ``None``.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import cv2
import gtsam
import numpy as np

from ..common.geometry import pose_from_rvec_tvec
from ..common.model import CameraModel, Detection, TagModel

# One PnP solution: (pose, RMS reprojection error in px)
Solution = Tuple[gtsam.Pose3, float]


def ippe_tag(det: Detection, cam: CameraModel, tag: TagModel) -> List[Solution]:
    """Both IPPE solutions ``T_camera_tag`` for one tag, best first.

    The object points must be in the order ``IPPE_SQUARE`` requires, which is
    the package's canonical corner order (:func:`model.tag_corner_points`).
    Solutions that put the tag behind the camera are dropped.
    """
    obj = tag.corners.astype(np.float64)
    img = det.corners.astype(np.float64).reshape(4, 1, 2)
    try:
        n, rvecs, tvecs, errs = cv2.solvePnPGeneric(
            obj, img, cam.K, None, flags=cv2.SOLVEPNP_IPPE_SQUARE)
    except cv2.error:
        return []
    out: List[Solution] = []
    for i in range(int(n)):
        if float(np.ravel(tvecs[i])[2]) <= 0.0:  # tag behind the camera
            continue
        out.append((pose_from_rvec_tvec(rvecs[i], tvecs[i]), float(np.ravel(errs)[i])))
    out.sort(key=lambda s: s[1])
    return out


def station_solutions(dets: Sequence[Detection], cam: CameraModel,
                      tags: dict) -> List[Solution]:
    """Candidate ``T_camera_dock`` poses from all tags in one frame.

    * One tag: run IPPE in the tag frame, then move to the station frame with
      the URDF mounting, ``T_C_dock = T_C_tag * T_dock_tag^-1``. Up to 2
      solutions.
    * Several tags: express every corner in ``dock_point`` (via the URDF tag
      poses) and run SQPNP on all of them at once. 1 solution, with the RMS
      error over all corners.
    """
    if len(dets) == 1:
        det = dets[0]
        tag = tags[det.tag]
        return [(T_ct.compose(tag.T_dock_tag.inverse()), err)
                for T_ct, err in ippe_tag(det, cam, tag)]

    obj = np.vstack([tags[d.tag].corners_in_dock() for d in dets]).astype(np.float64)
    img = np.vstack([d.corners for d in dets]).astype(np.float64).reshape(-1, 1, 2)
    try:
        n, rvecs, tvecs, errs = cv2.solvePnPGeneric(
            obj, img, cam.K, None, flags=cv2.SOLVEPNP_SQPNP)
    except cv2.error:
        return []
    out: List[Solution] = []
    for i in range(int(n)):
        T_cd = pose_from_rvec_tvec(rvecs[i], tvecs[i])
        out.append((T_cd, float(np.ravel(errs)[i])))
    out.sort(key=lambda s: s[1])
    return out
