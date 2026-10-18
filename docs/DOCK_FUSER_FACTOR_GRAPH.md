# Dock Fuser Factor Graph

![Factor graph example](dock_fuser_factor_graph.svg)

One tag, three keyframes: <i>X</i><sub>0</sub> and <i>X</i><sub>2</sub> see tag a, <i>X</i><sub>1</sub> sees none. Circles are variables (solved for); squares are factors (constraints on the variables they connect).

| Symbol | Kind | Connects | Meaning |
|---|---|---|---|
| <i>X</i><sub>k</sub> (blue) | variable | | AUV pose in odom at keyframe k. Tags pull it off the EKF when the EKF drifts. |
| <i>S</i> (orange) | variable | | Station dock_point pose in odom; one for the whole run. Docking uses <i>X</i><sub>k</sub><sup>−1</sup><i>S</i>. |
| <i>T</i><sub>a</sub> (green) | variable | | Tag a's pose relative to <i>S</i>. Stays at its URDF value with one tag. |
| **EKF init** (black) | factor | <i>X</i><sub>0</sub> | <i>X</i><sub>0</sub> = <i>E</i><sub>0</sub>, added once. Ties the graph to odom. |
| **odom** (blue) | factor | <i>X</i><sub>k−1</sub>, <i>X</i><sub>k</sub> | EKF motion <i>E</i><sub>k−1</sub><sup>−1</sup><i>E</i><sub>k</sub>; relative only, as x, y, yaw drift. Huber. |
| **roll/pitch**, **depth** | factor | <i>X</i><sub>k</sub> | Roll/pitch (gravity, yaw free) and depth (pressure) from the EKF. |
| **proj** (red) | factor | <i>X</i><sub>k</sub>, <i>S</i>, <i>T</i><sub>a</sub> | 4 tag corner pixels vs. prediction of <i>S</i>·<i>T</i><sub>a</sub> seen from <i>X</i><sub>k</sub>·<i>T</i><sub>bc</sub>. 1 px, Huber. |
| **URDF** (green) | factor | <i>T</i><sub>a</sub> | <i>T</i><sub>a</sub> near its station-URDF pose; σ = mounting tolerance. |

<i>S</i> and <i>T</i><sub>a</sub> link only through proj, as <i>S</i>·<i>T</i><sub>a</sub> (the tag's pose in odom). <i>E</i><sub>k</sub>: EKF pose at keyframe k. <i>T</i><sub>bc</sub>: camera extrinsic. See [DOCK_FUSER_PIPELINE.md](DOCK_FUSER_PIPELINE.md); figure source [dock_fuser_factor_graph.dot](dock_fuser_factor_graph.dot).
