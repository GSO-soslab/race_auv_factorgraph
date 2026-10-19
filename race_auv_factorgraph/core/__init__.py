"""Estimator core: pure numpy + gtsam, no ROS (shared by the node and the offline tools).

Sub-packages, in dependency order (each imports only from the ones above it).
Section numbers refer to ``docs/DOCK_FUSER_PIPELINE.md``.

============ ================================================================
``common``   data types, config, graph keys, Pose3 helpers, URDF reader (§3, §4)
``factors``  factor builders and noise models (§5)
``frontend`` EKF pose buffer, keyframe scheduling, PnP (§6)
``init``     station initialization: solver and state machine (§7)
``backend``  fixed-lag smoother and the per-keyframe estimator (§9, §10)
============ ================================================================
"""
