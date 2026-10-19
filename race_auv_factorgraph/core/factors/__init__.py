"""Factor builders and noise models (§5).

================== ============== =================================== ================
Factor             Keys           Residual                            Module
================== ============== =================================== ================
tag projection     X_k, S, T_j    corner pixels, 8 rows               ``projection``
odometry           X_{k-1}, X_k   (X_{k-1}^-1 X_k) (-) E_{k-1}^-1 E_k ``odometry``
EKF anchor prior   X_0            X_0 (-) E_0                         ``priors``
tag URDF prior     T_j            T_j (-) T_j^URDF                    ``priors``
================== ============== =================================== ================

``noise`` holds the shared Huber / diagonal-sigma helpers.

No absolute roll/pitch or depth factors: they did not change the docking
pose measurably (ablation, tank run 2026-09-16), so they were removed. They
belong back in once the EKF is replaced by IMU/DVL preintegration and a
pressure factor.
"""
