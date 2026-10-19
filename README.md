# race_auv_factorgraph

Factor-graph estimators for the RACE AUV. Used as a submodule of [race_auv](https://github.com/GSO-soslab/race_auv) (branch `jazzy-devel-perception`).

## Package

The repository root is the `race_auv_factorgraph` ament_python package. Its `dock_fuser_node` runs the gtsam fixed-lag smoother graph that estimates the docking station pose from AprilTag corner detections and the `robot_localization` EKF odometry. Publishes `odom → station` TF and `/dock_point/pose`. Status: implemented; tested on synthetic data and the live node. See the design doc §12 for what is left.

## Docs

- [docs/DOCK_FUSER_PIPELINE.md](docs/DOCK_FUSER_PIPELINE.md): full pipeline design (factors, initialization, detections used, backend, output, build order).
- [docs/DOCK_FUSER_FACTOR_GRAPH.md](docs/DOCK_FUSER_FACTOR_GRAPH.md): example factor graph.

## Dependencies

- ROS 2 Jazzy, Ubuntu 24.04
- gtsam Python bindings (`import gtsam`), e.g. `pip install gtsam` (tested with 4.3a0)
- `ros-jazzy-apriltag-msgs`; corner detections from `tag_detector_node` (`race_auv_perception/race_auv_camera_tag_factorgraph`)

## Build

```bash
cd ~/ros2_ws
colcon build --packages-select race_auv_factorgraph
colcon test --packages-select race_auv_factorgraph
```

## Run

See [docs/DOCK_FUSER_PIPELINE.md §15](docs/DOCK_FUSER_PIPELINE.md#15-running) for the live node, the bag-replay test with BEV plot, and the offline tools.
