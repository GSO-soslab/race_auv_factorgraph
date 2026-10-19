"""Leaf layer shared by all of ``core``: no factors, no graph logic.

* ``config``   -- :class:`EstimatorConfig`, every tuning knob
* ``model``    -- plain data types (camera, tag, detection, tag frame)
* ``keys``     -- gtsam symbols for ``X_k``, ``S``, ``T_j`` and timestamp bookkeeping (§4)
* ``geometry`` -- Pose3 helpers and the frame/tangent-space conventions
* ``urdf``     -- minimal URDF reader for extrinsics and tag poses (§3)
"""
