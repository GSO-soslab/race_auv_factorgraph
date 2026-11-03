# Factor Graph (race_auv_factorgraph)

![Factor graph example](graphs/example/factor_graph.svg)

One tag j, four keyframes, through initialization. Circles are variables (solved for); squares are factors (constraints on the variables they connect).

- <i>X</i><sub>0</sub>: no tag in view (NO_STATION). Odometry only.
- <i>X</i><sub>1</sub>: tag j seen, but the station is not known yet (CANDIDATE). The frame is only **buffered** (PnP hypotheses for the initializer); no projection factor yet, because <i>S</i> does not exist.
- <i>X</i><sub>2</sub>: tag j seen again, and the initializer accepts. One smoother update then inserts <i>S</i>, <i>T</i><sub>j</sub> with its URDF prior, and the projection factors of **all** buffered frames, <i>X</i><sub>1</sub>'s and <i>X</i><sub>2</sub>'s (orange: "at init").
- <i>X</i><sub>3</sub>: TRACKING. Its projection factor goes in at its own keyframe (red).

(The real initializer waits for more frames and checks that their PnP hypotheses agree, §7 of [DOCK_FUSER_PIPELINE.md](DOCK_FUSER_PIPELINE.md); two frames keep the picture small.)

## Where each symbol lives

All variables live in the **graph frame**, which the node publishes as `odom_graph`. It is fixed to the tank: the station does not move in it. The EKF's `odom` coincides with it at the first keyframe (the EKF init factor) and drifts away from it afterwards; TF `odom → odom_graph` is that drift, <i>E</i><sub>k</sub><i>X</i><sub>k</sub><sup>−1</sup>.

| Symbol | Solved? | Pose of … in … | In ROS |
|---|---|---|---|
| <i>E</i><sub>k</sub> | no, measured (EKF input) | `base_link` in `odom` | TF `odom → base_link`, `/race_auv/odometry/filtered` |
| <i>X</i><sub>k</sub> | yes | `base_link` in the graph frame | `~/odometry` (`odom_graph → base_link`, 20 Hz: the latest <i>X</i><sub>k</sub> moved by the EKF since) |
| <i>S</i> | yes | `dock_point` in the graph frame | TF `odom_graph → race_station/base_link` (= <i>S</i>·<i>T</i><sub>dock→base</sub>), `~/dock_pose` |
| <i>T</i><sub>j</sub> | yes | tag j in `dock_point` | TF `race_station/dock_point → tag frame` |
| <i>T</i><sub>bc</sub> | no, fixed (AUV URDF) | camera optical frame in `base_link` | TF `base_link → cam_front` (URDF) |

<i>E</i><sub>k</sub> is not a circle in the figure: it is the measurement inside the EKF init and odom factors, as their labels show.

## Factors

| Symbol | Kind | Connects | Meaning |
|---|---|---|---|
| **EKF init** (black) | factor | <i>X</i><sub>0</sub> | <i>X</i><sub>0</sub> = <i>E</i><sub>0</sub>, σ = 10<sup>−4</sup> (rad, m). Added once. Ties the graph frame to `odom` at the first keyframe, so `odom_graph` = `odom` there. |
| **odom** (blue) | factor | <i>X</i><sub>k−1</sub>, <i>X</i><sub>k</sub> | EKF motion <i>E</i><sub>k−1</sub><sup>−1</sup><i>E</i><sub>k</sub>, all 6 DOF. Relative only, because the EKF's absolute pose drifts. Random-walk σ (grows with time and distance); extra translation σ while the DVL has no bottom lock. Huber, k = 3. |
| **proj** (red; orange if inserted at initialization) | factor | <i>X</i><sub>k</sub>, <i>S</i>, <i>T</i><sub>j</sub> | One sighting of tag j: its 4 corner pixels vs. the corners of <i>S</i>·<i>T</i><sub>j</sub> projected into the camera at <i>X</i><sub>k</sub>·<i>T</i><sub>bc</sub> (8 rows). σ = 1 px, Huber, k = 2. |
| **URDF** (green) | factor | <i>T</i><sub>j</sub> | <i>T</i><sub>j</sub> near its station-URDF pose; σ = mounting tolerance (2°, 1 cm). Added once per tag, when the tag first enters the graph. |

The proj residual, for corner i of tag j (<i>c</i><sub>i</sub> on the tag, <i>uv</i><sub>i</sub> detected, π the pinhole projection with the camera matrix <i>K</i>):

  π(<i>K</i>, (<i>X</i><sub>k</sub>·<i>T</i><sub>bc</sub>)<sup>−1</sup>·<i>S</i>·<i>T</i><sub>j</sub>·<i>c</i><sub>i</sub>) − <i>uv</i><sub>i</sub>

In words: the tag reached through the station (<i>S</i>·<i>T</i><sub>j</sub>) must land where the camera on the AUV (<i>X</i><sub>k</sub>·<i>T</i><sub>bc</sub>) saw it.

<i>T</i><sub>j</sub> hangs off <i>S</i> geometrically: it is tag j's pose in the `dock_point` frame, starting at (and held near) the station URDF. That is why it is drawn under <i>S</i>. But no factor connects the two, so there is no edge: they meet only inside each proj factor, as the product <i>S</i>·<i>T</i><sub>j</sub> (tag j's pose in the graph frame). The URDF factor holds <i>T</i><sub>j</sub> alone, not <i>S</i>·<i>T</i><sub>j</sub>. See [DOCK_FUSER_PIPELINE.md](DOCK_FUSER_PIPELINE.md); figure source [graphs/example/factor_graph.dot](graphs/example/factor_graph.dot).
