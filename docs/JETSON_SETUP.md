# Jetson setup: tag detector + dock_fuser

How to set up a Jetson (the vehicle's back-seat computer) to run the factor-graph docking pipeline:
- the two camera drivers;
- the `race_auv_camera_tag_factorgraph` tag detectors;
- `dock_fuser`.

Most dependencies come from rosdep. A few do not, and those are the steps that are easy to miss. Pipeline design: [DOCK_FUSER_PIPELINE.md](DOCK_FUSER_PIPELINE.md).

Tested on a Jetson Orin Nano Super (JetPack R39, Ubuntu 24.04, ROS 2 Jazzy, Python 3.12, system OpenCV 4.6).

## 1. Clone

`race_station` is a separate repository, not a submodule. `dock_fuser` needs its station URDF (tag poses, `dock_point`).

```bash
cd ~/ros2_ws/src
git clone -b jazzy-devel-perception https://github.com/GSO-soslab/race_auv.git
cd race_auv && git submodule update --init      # top-level submodules only (no --recursive)
cd .. && git clone -b jazzy-devel-docking-sim https://github.com/GSO-soslab/race_station.git
```

`git submodule update --init` without `--recursive` fetches the three submodules:
- `race_auv_factorgraph`;
- `race_auv_perception`;
- `race_auv_sim`.

It leaves out the nested `race_auv_perception/third_party/isaac_ros_nitros` on purpose.

**`isaac_ros_nitros` is optional.** It is only needed by `race_auv_apriltag_cuda`, the CUDA detector of the older `race_auv_camera_pkg` pipeline (`bringup_camera_perception.launch.py`).
- **Not needed here:** the factor-graph detector (`race_auv_camera_tag_factorgraph`, CPU apriltag3) and `dock_fuser` do not use it.
- **Builds without it:** `race_auv_apriltag_cuda` still builds. CMake warns "cuAprilTags library not found … Skipping the CUDA detector shim", and that package falls back to its CPU backend.
- **Needs git-lfs:** do not clone with `--recurse-submodules` without git-lfs. That checkout stops with an error on this submodule.

If you do want the CUDA detector, install git-lfs and run `race_auv_perception/scripts/setup_third_party.sh`.

## 2. Dependencies rosdep does not cover

These two are imported from Python but not declared in any `package.xml`, so `rosdep install` will not install them.

**gtsam (Python).** It has no apt package; aarch64 wheels are on PyPI. Use the same version on every machine (the desktop and the first Jetson run 4.3a0):

```bash
pip3 install --user --break-system-packages gtsam==4.3a0
python3 -c "import gtsam; print(gtsam.IncrementalFixedLagSmoother)"   # check
```

**apriltag Python binding** (AprilRobotics `apriltag`, used by the detector as `from apriltag import apriltag`). Build it from source into `~/.local`. Pass the install rpath: without it the module installs fine but fails to import with `libapriltag.so.3: cannot open shared object file`.

```bash
git clone https://github.com/AprilRobotics/apriltag.git ~/third_party/apriltag
cd ~/third_party/apriltag
cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$HOME/.local \
      -DCMAKE_INSTALL_RPATH=$HOME/.local/lib -DBUILD_PYTHON_WRAPPER=ON \
      -DBUILD_EXAMPLES=OFF -DBUILD_TESTING=OFF
cmake --build build -j6 && cmake --install build
python3 -c "from apriltag import apriltag; apriltag('tag36h11'); print('ok')"   # check
```

## 3. ROS dependencies (rosdep)

```bash
cd ~/ros2_ws
rosdep install --from-paths src --ignore-src -y
```

This installs, among others, `ros-jazzy-apriltag-msgs`, `cv_bridge`, `vision_msgs` and `robot_localization`.

**Without sudo:** build `apriltag_msgs` from source in the workspace instead:

```bash
cd ~/ros2_ws/src && git clone https://github.com/christianrauch/apriltag_msgs.git
```

## 4. Build and test

```bash
cd ~/ros2_ws && source /opt/ros/jazzy/setup.bash
colcon build --symlink-install \
    --packages-up-to race_auv_factorgraph race_auv_camera_tag_factorgraph race_auv_bringup
source install/setup.bash
python3 -m pytest -q src/race_auv/race_auv_factorgraph/test \
    src/race_auv/race_auv_perception/race_auv_camera_tag_factorgraph/test
```

All 45 tests should pass: 35 `race_auv_factorgraph`, 10 `race_auv_camera_tag_factorgraph`. The detector tests also run on the Jetson's OpenCV 4.6. On that version the rectified camera matrix differs from the desktop's by about 0.04 px. This is harmless: `dock_fuser` takes the matrix from the detector's own `camera_info`.

## 5. System setup

**Same ROS network as the rest of the vehicle.**
- Set the same `ROS_DOMAIN_ID` in `~/.bashrc` (the desktop uses 53; a fresh Jetson has none, i.e. 0), or use the same zenoh router setup.
- Check from the Jetson that the front-seat topics are visible:

  ```bash
  ros2 topic echo /race_auv/odometry/filtered --once
  ```

**Clock sync (chrony) with the front-seat computer.** `dock_fuser` pairs each detection with the EKF pose by **header stamp**. The EKF runs on the front-seat Pi, the detector on the Jetson, so the two clocks must agree. An offset shows up as `ekf_age_s` in `/diagnostics`.

**Camera USB paths.** `race_auv_bringup/launch/include/multi_camera.launch.py` selects the two cameras by `video.product_name`:
- `usb-3610000.usb-2.3`: stellar2, `cam_front`;
- `usb-3610000.usb-2.1`: stellar1, `cam_down`.

A different Jetson or USB port can change these names.

**Power mode.** The real-time numbers below were measured in 25 W mode with the CPU clocks left to scale. Running `sudo jetson_clocks` locks the clocks at maximum for more headroom.

## 6. Run

```bash
ros2 launch race_auv_bringup bringup_camera_tag_factorgraph.launch.py
```

This starts, in order:
1. the cameras;
2. the detectors, after 3 s;
3. `dock_fuser`, after 8 s.

The vehicle configs are in `race_auv_bringup/config/`:

| File | Contents |
|---|---|
| `station_tags.yaml` | tag IDs and sizes, shared by the detectors and the fuser |
| `tag_factorgraph_detector.yaml` | raw camera calibration, detector settings |
| `dock_fuser.yaml` | factor-graph settings |

**Checks once running:**
- `ros2 topic hz /cam_front/apriltag_detection/tags`: 5 Hz, one message per processed image, with or without tags.
- `/diagnostics` (`dock_fuser`): `frames_dropped` and `frames_waiting` stay 0. The state goes `NO_STATION` → `CANDIDATE` → `TRACKING` once the station tags have been seen.

## 7. Topside viewing

On the topside computer (same ROS network):

```bash
sudo apt install ros-jazzy-compressed-image-transport   # RViz decodes the compressed camera topics
ros2 launch race_auv_bringup bringup_rviz_factorgraph.launch.py
```

The RViz window shows:
- the graph's and the EKF's live tracks;
- the AUV and station frames;
- the two camera feeds (`cam_down` turned 180° on the topside);
- the `cam_front` detector image.

This needs `publish_debug: true` in `dock_fuser.yaml` and the `cam_front` detector's `publish_image: true`; both are set in the vehicle configs. See §10 of the [pipeline doc](DOCK_FUSER_PIPELINE.md).

## 8. Expected real-time margin

Measured on the Orin Nano Super (25 W, clocks not locked): one detector plus `dock_fuser`, fed the tank recording `rosbag2_2026_09_16-16_16_47` (30–420 s) from a desktop.

| | median | p95 | max |
|---|---|---|---|
| detector, per frame | — | — | 83–96 ms (its log) |
| `dock_fuser` update, per keyframe | 6.6 ms | 77 ms | 120 ms |
| image → docking pose | 140 ms | 211 ms | 274 ms |

- The keyframe budget at 5 Hz is 200 ms. All 1949 detector messages became keyframes; none was dropped or queued.
- CPU load was 26 % on average (6 cores), 6.1 W, 1.1 GB RAM.
- The full vehicle load (two camera drivers, two detectors) has not been measured yet.

The figure `dock_fuser_graph_overview_timing.png` shows this run.
