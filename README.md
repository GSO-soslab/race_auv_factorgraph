# race_auv_factorgraph

Factor-graph estimators for the RACE AUV. Used as a submodule of [race_auv](https://github.com/GSO-soslab/race_auv) (branch `jazzy-devel-perception`).

## Packages

- `dock_fuser`: gtsam iSAM2 graph that estimates the docking station pose from AprilTag corner detections and the `robot_localization` EKF odometry. Publishes `odom → station` TF and `/dock_point/pose`. **Status: scaffold only.**

## Docs

- [docs/DOCK_FUSER_PIPELINE.md](docs/DOCK_FUSER_PIPELINE.md): full pipeline design (factors, initialization, gating, backend, output, build order).
- [docs/DOCK_FUSER_FACTOR_GRAPH.md](docs/DOCK_FUSER_FACTOR_GRAPH.md): example factor graph.

## Dependencies

- ROS 2 Jazzy, Ubuntu 24.04
- gtsam Python bindings (`import gtsam`), e.g. `pip install gtsam`

## Build

```bash
cd ~/ros2_ws
colcon build --packages-select dock_fuser
colcon test --packages-select dock_fuser
```
