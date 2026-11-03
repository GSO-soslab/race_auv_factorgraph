# race_auv_factorgraph: ROS interface and architecture

What the factor-graph node publishes, how the vehicle stack should use it, and how its topics map to those of the former `dock_fuser` node. Audience: whoever writes the docking guidance against it, and whoever runs or watches it.

Estimation design (factors, initialization, backend): [DOCK_FUSER_PIPELINE.md](DOCK_FUSER_PIPELINE.md). Code: `race_auv_factorgraph/ros/node.py`.

Contents:
1. [Architecture](#1-architecture)
2. [Frames](#2-frames)
3. [How a 5 Hz graph gives a 20 Hz pose](#3-how-a-5-hz-graph-gives-a-20-hz-pose)
4. [Topics](#4-topics)
5. [The DockState message](#5-the-dockstate-message)
6. [Using it for docking guidance](#6-using-it-for-docking-guidance)
7. [Foxglove](#7-foxglove)
8. [Migration from dock_fuser](#8-migration-from-dock_fuser)
9. [Launch, parameters, files](#9-launch-parameters-files)
10. [Verification](#10-verification)

---

## 1. Architecture

```
 IMU, DVL, depth ──▶ robot_localization EKF ──▶ /race_auv/odometry/filtered   (race_auv/odom, 20 Hz, drifts)
                                                   │                    │
 cam_front, cam_down ──▶ tag detectors (5 Hz)      │                    ▼
          corners ──────────────▶ race_auv_factorgraph            mvp_control (low-level control, EKF in odom)
                                         │                              ▲
                                         │  ~/odometry  (race_auv/odom_graph, 20 Hz)
                                         │  ~/dock_state (5 Hz), TF     │ setpoints stamped race_auv/odom_graph;
                                         ▼                              │ mvp_helm converts them through TF
                                  docking guidance (race_docking_mission) ─┘
```

Three layers, each in the frame that suits it:

| Layer | Frame | Pose it uses | Rate |
|---|---|---|---|
| Low-level control (`mvp_control`) | `race_auv/odom` | the EKF, `/race_auv/odometry/filtered` | 20 Hz |
| Docking guidance (`race_docking_mission`) | `race_auv/odom_graph` | AUV: `~/odometry`; dock: `~/dock_state` | 20 Hz / 5 Hz |
| Display (Foxglove) | `race_auv/odom_graph` | TF, `~/path`, `~/ekf_path`, `~/dock_pose` | |

- **The low-level controller keeps the EKF.** It is smooth, high rate and continuous, which is what inner loops need. It never sees the graph's corrections directly.
- **Guidance works in `odom_graph`.** The dock is fixed there, so a target taken once stays valid. The guidance must not use the EKF's *pose*: relative to the dock the EKF drifted 20 cm to 1 m per tank run (§10). It still gets the EKF's velocity, which `~/odometry` carries unchanged.
- **The two meet through TF.** A setpoint stamped in `race_auv/odom_graph` is converted to the controller's frame by `mvp_helm` with the current `odom → odom_graph` transform, so the controller's error is the drift-free error.
- **Nothing tag-derived is fed back into the EKF.** The graph uses the EKF as its odometry; a tag pose fused into `robot_localization` would count the tags twice.

The node runs as `/race_auv/race_auv_factorgraph`: every topic below is `/race_auv/race_auv_factorgraph/<name>` (written `~/<name>`). The exceptions are `/tf`, `/tf_static` and `/diagnostics`, which are shared by all nodes.

## 2. Frames

| Frame | Who publishes it | What it is |
|---|---|---|
| `race_auv/odom` | robot_localization (`odom → base_link`) | The EKF's frame. Continuous, but drifts relative to the station. |
| `race_auv/odom_graph` | this node (`odom → odom_graph`) | The factor graph's drift-corrected odom frame. The station is fixed in it. |
| `race_station/base_link` | this node (`odom_graph → station`) | The station, at the graph's estimate. |
| `race_station/dock_point` | this node (static, from the station URDF) | The docking target on the station. |
| `race_station/apriltag36h11_<id>` | this node (`dock_point → tag`) | Each station tag at the graph's estimate. |

```
race_auv/odom ──EKF──▶ race_auv/base_link ──static──▶ cameras, cg_link, auv_dock_point, ...
     │
     └──▶ race_auv/odom_graph ──▶ race_station/base_link ──static──▶ race_station/dock_point ──▶ race_station/apriltag36h11_<id>
```

**`odom_graph` starts equal to `odom`.** The graph pins its first keyframe to the EKF (`X_0 = E_0`), and until the station is found it has nothing to correct, so the transform `odom → odom_graph` stays the identity. Once tags are seen, it moves by however far the EKF has drifted. The graph frame never re-anchors during a run; only a node restart starts it again at the EKF.

Through TF, any frame pair works: `race_auv/auv_dock_point → race_station/dock_point` is the drift-corrected relative pose between the two dock points.

The station frames are published only with `publish_station_frames: true` (vehicle config: on). Set it false if the station's own `robot_state_publisher` runs, since it publishes the same frame names.

## 3. How a 5 Hz graph gives a 20 Hz pose

The graph solves once per keyframe (one per `cam_front` detector message, 5 Hz). A keyframe's result is ready ~140 ms after its image stamp. Used directly as the AUV pose, it would be 260 ms old on average when the guidance reads it, and 1.1 cm (5.2 cm p95) behind the AUV (2026-10-22 tank runs).

What the graph adds to the EKF changes far more slowly: the correction between the two frames at keyframe `k`,

```
C_k = X_k · E_k⁻¹          X_k: graph's AUV pose (odom_graph), E_k: EKF's (odom), at the keyframe stamp
```

It moves only as fast as the EKF drifts: 0.02 cm median per keyframe on the same runs. So the node applies it to every EKF message:

```
pose(t) = C_k · E(t) = X_k · (E_k⁻¹ · E(t))
```

That is the graph's keyframe pose moved by the motion the EKF measured since. It is how the graph itself predicts its next keyframe, and the usual ROS `map → odom` pattern (slow correction, fast odometry).

| Situation | `~/odometry` | `~/dock_state` |
|---|---|---|
| No tag seen yet | the EKF (`odom_graph` = `odom`) | `NO_STATION` |
| Tags seen, station initializing (≥ 5 consistent frames within 3 s) | the EKF | `CANDIDATE` |
| Tracking, tags in view | graph pose, 20 Hz | `TRACKING`, `last_tag_stamp` current |
| Tracking, tags out of view | graph prediction (EKF motion since the last tag), drifting like the EKF | `TRACKING`, `last_tag_stamp` ages, covariance grows |
| DVL dropout or re-lock since the keyframe | the keyframe pose `X_k`, updated at 5 Hz; the EKF's jumps are not followed. Also before the station exists: then `X_k` is the EKF's keyframe pose | as above |
| Tag back after a gap | one step of the size of the drift during the gap (7–11 cm on the 09-16 run) | `last_tag_stamp` updates |
| Node restart | the EKF again | `NO_STATION` |

**DVL dropouts.** Without bottom lock the EKF dead-reckons in jumps, and when the lock returns it snaps back (1.65 m on the 10-20 run, 23 m on the 10-22 20:46 run). The node watches the DVL topic; across a dropout or re-lock it does not use the EKF's motion, and holds the graph's keyframe pose until a keyframe after the snap is solved. The graph itself de-weights those odometry steps, so `X_k` follows the tags.

**No smoothing.** The outputs are the estimate as it is. The step when a tag returns after a gap is the correction arriving; smoothing it is the guidance's or controller's choice.

## 4. Topics

### Inputs

| Topic | Type | Use |
|---|---|---|
| `/race_auv/odometry/filtered` (`odom_topic`) | `nav_msgs/Odometry` | EKF pose `odom → base_link` and body-frame twist |
| `/race_auv/nucleus_node/bottom_lock_velocity_common` (`dvl_topic`) | `geometry_msgs/TwistWithCovarianceStamped` | DVL bottom lock: only the stamps are used (a silence = dropout) |
| `/cam_front/apriltag_detection/tags` | `apriltag_msgs/AprilTagDetectionArray` | corners, rectified image; every message is one keyframe |
| `/cam_down/apriltag_detection/tags` | `apriltag_msgs/AprilTagDetectionArray` | attached camera (tag factors on the cam_front keyframes); off in the vehicle config |
| `/cam_front/apriltag_detection/camera_info`, `/cam_down/...` | `sensor_msgs/CameraInfo`, latched | the detector's rectified intrinsics |

Only header stamps are used, never arrival times. The front-seat (EKF) and Jetson (detector, this node) clocks must be synchronized (chrony); `ekf_age_s` in `/diagnostics` shows an offset.

### Outputs

| Topic | Type | Frame | Rate | Content |
|---|---|---|---|---|
| `~/odometry` | `nav_msgs/Odometry` | `race_auv/odom_graph` → `race_auv/base_link` | every EKF message (20 Hz), from node start | AUV pose (§3: `KeyframeResult.auv_pose_at`); twist and twist covariance copied from the EKF (base_link axes); pose covariance not estimated (all zero) |
| `~/dock_state` | `race_auv_msgs/DockState` | `race_auv/odom_graph` | every keyframe (5 Hz), from node start | station state, dock pose + covariance, last time a tag was seen (§5) |
| `~/dock_pose` | `geometry_msgs/PoseWithCovarianceStamped` | `race_auv/odom_graph` | every keyframe while `TRACKING` | the same dock pose and covariance as a standard message: Foxglove draws it with its covariance ellipsoid; the offline tools read it |
| TF `race_auv/odom → race_auv/odom_graph` | | | every EKF message, from node start | consistent with `~/odometry`: TF `odom_graph → base_link` is the published pose |
| TF `race_auv/odom_graph → race_station/base_link` | | | every EKF message while `TRACKING` | the station |
| TF `race_station/base_link → race_station/dock_point` | static | | once | from the station URDF (`publish_station_frames`) |
| TF `race_station/dock_point → race_station/apriltag36h11_<id>` | | | every EKF message while `TRACKING` | each tag at the graph's estimate, its URDF pose until seen (`publish_station_frames`) |
| `/diagnostics` | `diagnostic_msgs/DiagnosticArray` | | 1 Hz | status `race_auv_factorgraph`: state, tags accepted / rejected, update time, input ages, `last_tag_age_s`, drift correction |

### Analysis and display outputs (`publish_debug: true`, on in the vehicle config)

| Topic | Type | Rate | Content |
|---|---|---|---|
| `~/auv_pose` | `geometry_msgs/PoseStamped` (`odom_graph`) | every keyframe | `X_k`, the graph's AUV pose as estimated at that keyframe; read by the offline tools |
| `~/keyframe` | `std_msgs/String` (JSON) | every keyframe | `stamp`, `state`, `camera`, `n_tags` (station tags in the detector message), `n_tag_factors`, `accepted`, `rejected` (with reasons), `initialized_now`, `recovered`, `init_shift_m`, `dvl_sigma`, `init_note`, `update_ms`, and `attached` with attached cameras; read by the offline tools |
| `~/path` | `nav_msgs/Path` (`odom_graph`) | `path_rate` (1 Hz) | the last `path_length` (3000) keyframes of `X_k`: the graph's track |
| `~/ekf_path` | `nav_msgs/Path` (`odom_graph`) | `path_rate` (1 Hz) | the EKF's `E_k` at the same keyframes, drawn in `odom_graph`: where the EKF thinks the AUV went, in the frame both started in. The gap between the two tracks is the EKF's drift. |

The paths are re-sent whole (~250 kB each at 3000 poses), hence the 1 Hz rate; lower `path_length` for a slow link.

## 5. The DockState message

`race_auv_msgs/msg/DockState` (package `race_auv_msgs` in `race_auv`):

| Field | Type | Meaning |
|---|---|---|
| `header.stamp` | `builtin_interfaces/Time` | keyframe time (the camera image stamp) |
| `header.frame_id` | `string` | `race_auv/odom_graph` |
| `state` | `uint8` | `NO_STATION` = 0: no station tag seen yet. `CANDIDATE` = 1: tags seen, collecting consistent views. `TRACKING` = 2: station initialized |
| `dock` | `geometry_msgs/PoseWithCovariance` | `race_station/dock_point` in `odom_graph`. Valid only in `TRACKING` (identity otherwise). Covariance in ROS order (x, y, z, rot x, rot y, rot z) along the `odom_graph` axes: the uncertainty of the dock **relative to the AUV** at this keyframe. All zero = not computed (`covariance_mode: none`) |
| `last_tag_stamp` | `builtin_interfaces/Time` | stamp of the newest keyframe that added a tag factor (any camera); zero if none yet |
| `n_tag_factors` | `uint8` | tag factors this keyframe added; 0 before `TRACKING` (detections are buffered for initialization) |

The estimator never leaves `TRACKING` on its own. It goes back to `NO_STATION` only when the node restarts or its optimizer has to be reset.

## 6. Using it for docking guidance

For the rework of `race_docking_mission` (its current inputs are listed in §8):

1. **AUV pose:** subscribe to `/race_auv/race_auv_factorgraph/odometry` instead of `/race_auv/odometry/filtered`, and plan in `race_auv/odom_graph` (`odom_frame` parameter). Before the station is found it equals the EKF, so the search phase is unchanged. The twist is the EKF's (body frame).
2. **Dock target:** from `/race_auv/race_auv_factorgraph/dock_state`, `dock.pose` when `state == TRACKING`. It is fixed in `odom_graph`, so it can be stored and reused: no composition with the EKF pose, no "record at each detection, hold when lost". Add the AUV's own `auv_dock_point` offset as before (TF `race_auv/base_link → race_auv/auv_dock_point`).
3. **Tag seen / lost:** `state` replaces the detection debounce (`CANDIDATE` = tags in view, `TRACKING` = target valid). How long the dock has gone unseen is `stamp(latest ~/odometry) − last_tag_stamp`; use message stamps, not the local clock, so a clock offset between computers cannot matter. The `dock` covariance grows while no tag is seen, if a threshold on uncertainty is preferred to one on time; it is conservative (§10), so calibrate such a threshold on a recorded run.
4. **Setpoints to `mvp_helm`:** stamp the `ControlProcess` with `header.frame_id = race_auv/odom_graph`. `mvp_helm` already transforms every setpoint from its `header.frame_id` into its world frame through TF (`Helm::f_transform_control_process_msg`, lookup at the latest time), so the PID, which runs on the EKF, gets a setpoint that includes the current correction. The RBF path (`Z1`, `z2`) is computed inside the mission and can stay entirely in `odom_graph`.
5. **Restart:** when `state` drops back to `NO_STATION` (node restarted), drop any stored target: the new `odom_graph` is a new frame.
6. **Remove** the EKF-drift workarounds the graph replaces: `dock_point_odom` recording/holding, the `dock_ref` freeze and `apriltag_absolute_pose`.

## 7. Foxglove

Through `foxglove_bridge` on the vehicle, or a recorded bag. Suggested layout:

**3D panel**
- Display frame: `race_auv/odom_graph`. Follow mode: fixed, or follow `race_auv/base_link`.
- Topics:
  - `/tf`, `/tf_static`: the AUV (`race_auv/base_link`, cameras), the station, `dock_point` and the tag frames.
  - `/race_auv/race_auv_factorgraph/path`: the graph's track.
  - `/race_auv/race_auv_factorgraph/ekf_path`: the EKF's track (a second colour). It separates from the graph's track by the EKF's drift.
  - `/race_auv/race_auv_factorgraph/dock_pose`: the dock with its covariance ellipsoid (grows while no tag is seen).
  - optional `/race_auv/race_auv_factorgraph/odometry`: the 20 Hz pose arrow.
- AUV model: a URDF layer from the topic `/race_auv/robot_description`; its links are placed by TF, so the model sits at the graph's pose.

**Other panels**
- Image: `/cam_front/apriltag_detection/image/compressed` (rectified, tags drawn; needs the detector's `publish_image: true`), and the camera feeds `/race/stellar2/image/compressed` (cam_front) and `/race/stellar1/image/compressed` (cam_down, mounted upside down: rotate 180° in the panel settings).
- Raw Messages: `/race_auv/race_auv_factorgraph/dock_state` (state, last tag time).
- Plot: `/race_auv/race_auv_factorgraph/odometry.pose.pose.position.x` against `/race_auv/odometry/filtered.pose.pose.position.x` (and `.y`): the drift correction over time.
- Diagnostics: `/diagnostics`, status `race_auv_factorgraph`.

## 8. Migration from dock_fuser

The node was `/dock_fuser` (executable `dock_fuser_node`). Its outputs kept the old `apriltag_fuser`'s interface for the docking mission, which is reworked now, so they were replaced.

### Topics and frames

| Before (`dock_fuser`) | After (`race_auv_factorgraph`) | Note |
|---|---|---|
| TF `race_auv/odom → race_station/base_link` (`E·X⁻¹·S`, rate-limited) | TF `race_auv/odom_graph → race_station/base_link` (`S`) | the station is fixed in `odom_graph`; `base_link → station` through TF is the same relative pose. No rate limit |
| TF `race_auv/odom → race_auv/odom_graph` (only with `corrected_frame`, from the first keyframe) | same, always, from the first EKF message | same value between keyframes; across DVL dropouts it keeps the AUV at the graph's keyframe pose |
| TF `race_station/base_link → race_station/dock_point`, `dock_point → apriltag frames` | same | parameter `publish_station_static_tf` renamed `publish_station_frames`; tag frames now at the EKF rate |
| `/race_auv/dock_point/pose` (`PoseWithCovarianceStamped`, dock in `base_link`) | `~/dock_state.dock` or `~/dock_pose` (dock in `odom_graph`) | the base_link-relative pose is a TF lookup `race_auv/base_link → race_station/dock_point` |
| `/race_station/dock_point/pose` (`PoseStamped`, only on keyframes with tags; the mission counted these as detections) | `~/dock_state` (`state`, `last_tag_stamp`, `n_tag_factors`) | published every keyframe; freshness is explicit |
| `/dock_fuser/odometry` (`X_k` at 5 Hz, labelled `race_auv/odom`, no twist) | `~/odometry` (20 Hz, `race_auv/odom_graph`, EKF twist) | |
| `/dock_fuser/auv_pose` (labelled `race_auv/odom`) | `~/auv_pose` (`race_auv/odom_graph`) | same values |
| `/dock_fuser/station_pose` (`PoseStamped`) | `~/dock_pose` (`PoseWithCovarianceStamped`) | with covariance |
| `/dock_fuser/keyframe` | `~/keyframe` | adds `n_tag_factors`; otherwise the same fields |
| `/dock_fuser/path`, `/dock_fuser/path_graph` | `~/path` (`odom_graph`) | one path |
| `/dock_fuser/ekf_path` (`race_auv/odom`) | `~/ekf_path` (drawn in `odom_graph`) | shows the drift in the graph's frame |
| `/diagnostics` status `dock_fuser` | status `race_auv_factorgraph` | the "old apriltag_fuser running" check is gone with the compat topic |

The offline tools (`analyze_run`, `plot_bev`, `redetect_bag`) read both the new names and the old `/dock_fuser/...` ones, so recorded runs still work.

### What the docking mission uses today

`race_docking_mission/search_mission_node.py` (`config/search_mission.yaml`):

| Today | Replace with |
|---|---|
| `ekf_odom_topic: /race_auv/odometry/filtered` | `/race_auv/race_auv_factorgraph/odometry` |
| `odom_frame: race_auv/odom` | `race_auv/odom_graph` |
| `dock_pose_topic: /race_station/dock_point/pose` (arrival = detection; dock in `base_link`) | `/race_auv/race_auv_factorgraph/dock_state` (`race_auv_msgs/DockState`) |
| `dock_point_odom` (EKF pose ∘ detection, recorded per detection, held when lost) | `dock_state.dock.pose`: already fixed |
| `apriltag_absolute_pose` and `dock_ref` (tag-derived yaw for the EKF) | remove (must not be fused into the EKF) |
| `tag_debounce_count`, `tag_detection_gap`, tag-lost timeouts on message arrival | `state` and `last_tag_stamp` |
| `ControlProcess` setpoints with `frame_id: race_auv/odom` | `frame_id: race_auv/odom_graph` (converted by `mvp_helm` via TF) |

### Removed

- RViz support: `race_auv_bringup/launch/bringup_rviz_factorgraph.launch.py` and `rviz/factorgraph.rviz`; Foxglove replaces them (§7). `image_rotate_node` (`race_auv_camera_tag_factorgraph`), which turned the cam_down feed for that RViz view, is removed too; Foxglove's Image panel rotates.
- Parameters `dock_pose_topic`, `compat_pose_topic`, `corrected_frame` (now `odom_graph_frame`, always on), `graph_frame` (now `odom_frame`), `publish_station_static_tf` (now `publish_station_frames`), `output_max_trans_rate`, `output_max_rot_rate`.

## 9. Launch, parameters, files

| | Before | After |
|---|---|---|
| Node | `/dock_fuser` | `/race_auv/race_auv_factorgraph` |
| Executable | `dock_fuser_node` | `factorgraph_node` |
| Vehicle launch | `race_auv_bringup/launch/include/dock_fuser.launch.py` | `race_auv_bringup/launch/include/factorgraph.launch.py` (via `bringup_camera_tag_factorgraph.launch.py`) |
| Vehicle config | `race_auv_bringup/config/dock_fuser.yaml` | `race_auv_bringup/config/race_auv_factorgraph.yaml` (with `station_tags.yaml`) |
| Package launch | `launch/dock_fuser.launch.py` | `launch/factorgraph.launch.py` |
| Package configs | `config/dock_fuser_sim.yaml`, `config/tank_dock_fuser.yaml` | `config/factorgraph_sim.yaml`, `config/tank_factorgraph.yaml` |

Parameter files address the node as `/**/race_auv_factorgraph` (any namespace). The offline tools still read files whose section is `dock_fuser`.

Node parameters (`ros/params.py`; estimator knobs `estimator.*` in `core/common/config.py`):

| Parameter | Default | |
|---|---|---|
| `odom_topic` | `/race_auv/odometry/filtered` | EKF |
| `dvl_topic` | `""` (vehicle: the Nucleus bottom-lock topic) | `""` = no DVL dropout handling |
| `odom_frame` | `race_auv/odom` | the EKF's frame |
| `odom_graph_frame` | `race_auv/odom_graph` | the graph's frame |
| `base_frame`, `station_base_frame`, `dock_frame` | `race_auv/base_link`, `race_station/base_link`, `race_station/dock_point` | |
| `publish_station_frames` | `true` | station `base_link → dock_point` and the tag frames |
| `camera_names`, `keyframe_camera`, `cameras.<name>.*` | `[cam_front]`, first enabled | see the vehicle config |
| `publish_debug`, `path_rate`, `path_length` | `false`, 1.0 Hz, 3000 | analysis and display topics |

Build: `colcon build --packages-select race_auv_msgs race_auv_factorgraph race_auv_bringup`.

## 10. Verification

**The node on a recorded run.** The 2026-10-22 20:46 tank run (910 s, DVL dropouts of 9, 71 and 52 s, EKF jumps of 23 m and 5.5 m when the lock returned) was replayed through the node as on the vehicle: vehicle config, the recorded EKF, DVL, detections and camera_info played at 4× speed, outputs recorded.

| Check | Result |
|---|---|
| Names | every topic under `/race_auv/race_auv_factorgraph/`; the vehicle config's `/**/race_auv_factorgraph` section was applied |
| Rates | `~/odometry`: 18207 messages for 18208 EKF messages; `~/dock_state`: 4551, one per keyframe |
| TF against `~/odometry` | `odom_graph → base_link` through TF equals `~/odometry` to 0.000 mm on every message |
| Twist | identical to the EKF's |
| Before the station | the EKF, except during a 0.5 s DVL dropout at 40 s, where it held the keyframe pose (≤ 4.6 cm from the EKF) |
| EKF jumps | 23 m at 759.9 s and 5.5 m at 862.3 s: `~/odometry` moved 0.0 cm |
| Steps of `~/odometry` | median 0.44 cm per message (the EKF's: 0.52 cm). The largest, 34 cm, come during the 71 s dropout, while the EKF dead-reckoned in 0.5–1.4 m jumps; the former node's `X_k` stepped as much there |
| Station | `NO_STATION → CANDIDATE` at 47.9 s, `TRACKING` at 48.9 s; afterwards the dock moved ≤ 4.2 cm in `odom_graph` over 860 s |
| `last_tag_stamp` | never goes back; the dock went unseen for up to 112 s |
| Dock covariance | identical to the former `/race_auv/dock_point/pose` (position 37 cm, rotation 2.7°, square root of the trace) |
| Offline tools | `analyze_run`, `plot_bev` and `plot_graph --timing` read this recording and the 10-22 recordings of the former node |

**The dock covariance is conservative.** With tags in view its position part is ~37 cm (1σ), while the docking pose agrees with the batch solution to 0.3 cm median (09-16 run, [pipeline doc §8](DOCK_FUSER_PIPELINE.md#8-which-detections-are-used)). It is useful as a relative measure (it grows while the dock is unseen); calibrate any threshold on it against a recorded run first.

**The EKF's drift relative to the dock**, from the former node's recorded outputs on the seven 10-22 runs. The dock in the EKF's `odom` wandered 22, 40, 50, 53, 88 and 98 cm per run, and 17.5 m in the 20:46 run after the EKF diverged. Over the same runs `S` moved 0.4–4.1 cm in the graph frame after its first 5 s.

**20 Hz against 5 Hz**, same runs. When a new keyframe lands, the 20 Hz pose steps by 0.02 cm median, 0.7 cm p95 and 3.7 cm p99 (EKF jumps excluded). The 5 Hz keyframe pose was 260 ms old on average when used (356 ms p95), and 1.1 cm (5.2 cm p95) behind the AUV.

**Tests:** `colcon test --packages-select race_auv_factorgraph` (45 tests: factors, geometry, scheduler, the 20 Hz pose rule, the dock covariance frame, the parameter-file sections, and synthetic end-to-end scenarios).
