"""Measurement front end: turns raw EKF poses and tag detections into keyframes (§6).

* ``pose_buffer`` -- EKF pose history with interpolation at image stamps
* ``scheduler``   -- one keyframe per detector message, at its image stamp
* ``pnp``         -- IPPE tag / station poses from corner pixels
* ``dvl_monitor`` -- DVL bottom-lock dropouts -> extra odometry sigma
"""
