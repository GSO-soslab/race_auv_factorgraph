# Dock Fuser: gtsam Pipeline

This document describes the factor-graph pipeline that estimates the docking station's pose from AprilTag corner detections and the `robot_localization` EKF odometry. It is implemented as a Python ROS 2 node using gtsam's iSAM2.

---

## 1. Overview

```
 /odometry/filtered ──► EKF pose buffer ───────────────┐   (interpolate E(t) at any stamp)
 cam_front corners  ─┐                                 │
 cam_down  corners  ─┴► detection queue ─► pre-filter ─┤
                                                       ▼
                                            ┌─────────────────────┐
                                            │  Keyframe manager   │
                                            └──────────┬──────────┘
                                                       ▼
                          ┌──────────── Factor builders ────────────┐
                          │  odometry · roll/pitch · depth · tags   │
                          └────────────────────┬─────────────────────┘
                                               ▼
                          ┌──────────────── Gates ──────────────────┐
                          │  multi-tag consistency → reprojection    │
                          │  → local solve before admitting factors  │
                          └────────────────────┬─────────────────────┘
                                               ▼
                        Station state (NO_STATION / CANDIDATE / TRACKING)
                                               ▼
                                         iSAM2 backend
                                               ▼
                       Output: odom → station TF  +  /dock_point/pose
```

The EKF owns `odom → base_link` and drives control. The graph estimates the station's pose and publishes `odom → station`, including the correction for EKF drift (§10). The TF tree composes the two, so `base_link → dock_point` is always available at the EKF rate.

---

## 2. Conventions

| Symbol | Meaning |
|---|---|
| `odom` | Graph frame. Same as the EKF's `odom` frame. |
| `E_k` | EKF pose of `base_link` in `odom` at keyframe k, interpolated to the keyframe stamp. |
| `T_bc` | Camera extrinsic (`base_link → camera`) from the AUV URDF. Held constant. |
| `K_C` | Pinhole camera matrix of the **rectified** image for camera C, read from the node's YAML config (§3). No distortion terms. |
| `s_j` | Black-square edge length of tag j (where the detector places the corners). |
| `c_i` | Corner i of a tag in the tag frame: `(±s_j/2, ±s_j/2, 0)`. |

Camera model: the detector runs on rectified images, so corner pixels are already distortion-free. The graph uses a plain pinhole projection with `K_C`. The raw calibration is used only by the rectifier.

### Frames: what is known where

```
            odom  (origin = where robot_localization started, (0,0,0))
           /    \
   X_k    /      \   S          ← the one unknown that links the two worlds
 (EKF)   /        \
   base_link     dock_point  (station frame)
       |              |
  T_bc |              | T_j  (tag j's pose in the station frame; URDF value as prior)
       |              |
    camera ─────────► tag j
          observation
        (4 corner pixels)
```

- **AUV side, known in `odom`.** The EKF gives `X_k` (`base_link` in `odom`). `T_bc` comes from the AUV URDF. So the camera's pose in `odom` is known.
- **Station side, known in `dock_point`.** Every tag's pose `T_j` and corner layout are known in the station frame (station URDF). Nothing about the station is known in `odom`.
- **The link is `S`**, the station's pose in `odom`. It is the single unknown that connects the two sides. A tag is never placed directly in `odom`: its `odom` pose is always `S · T_j`.
- **The `odom` origin is arbitrary.** `robot_localization` starts at `(0,0,0)` wherever the AUV happens to be, and `S` simply absorbs that offset. The station ends up at some `S` in `odom`, and docking only ever uses the relative pose `X_k⁻¹ · S`.

---

## 3. Inputs

| Input | Type | Use |
|---|---|---|
| `/odometry/filtered` | `nav_msgs/Odometry` | Pose buffer for odometry, roll/pitch and depth factors. |
| Per-camera tag detections | Tag ID + 4 corner pixels, stamped | Tag factors and initialization. |
| Node YAML config | — | `K_C` for each camera, detection topics, noise parameters. |
| AUV URDF | — | `T_bc` for each camera. |
| Station URDF | — | Tag poses relative to `dock_point`, tag sizes. |

The node does not subscribe to `camera_info`. Camera intrinsics are defined in the YAML config:

```yaml
dock_fuser:
  ros__parameters:
    cameras:
      cam_front:
        detections_topic: /cam_front/tag_corners
        frame: race_auv/cam_front          # T_bc looked up from the AUV URDF
        # Pinhole matrix of the RECTIFIED image the detector runs on
        fx: 0.0
        fy: 0.0
        cx: 0.0
        cy: 0.0
        width: 1600
        height: 1200
      cam_down:
        detections_topic: /cam_down/tag_corners
        frame: race_auv/cam_down
        fx: 0.0
        fy: 0.0
        cx: 0.0
        cy: 0.0
        width: 1600
        height: 1200
```

`fx, fy, cx, cy` must be the camera matrix the rectifier maps into, not the raw calibration. If the rectifier uses a new camera matrix (for example from `cv2.fisheye.estimateNewCameraMatrixForUndistortRectify`), that is the matrix that belongs here. A mismatch scales every range estimate.

---

## 4. Variables

| Key | Type | Lifetime | Meaning |
|---|---|---|---|
| `X_k` | `Pose3` | one per keyframe | AUV `base_link` in `odom`. |
| `S` | `Pose3` | permanent | Station `dock_point` in `odom`. The station is static, so this is one variable shared by all keyframes. |
| `T_j` | `Pose3` | permanent, created at first sighting | Pose of tag j relative to `S` (its mounting offset). |

No velocity or IMU-bias variables: the EKF already fuses IMU, DVL and pressure, and the graph only consumes its pose.

---

## 5. Factors

| Factor | Keys | Residual | Noise | Added |
|---|---|---|---|---|
| EKF init prior | `X_0` | `X_0 ⊖ E_0` | very tight | first keyframe only |
| Odometry | `X_{k-1}, X_k` | `(X_{k-1}⁻¹ X_k) ⊖ (E_{k-1}⁻¹ E_k)` | random walk (below), Huber | every keyframe |
| Roll/pitch prior | `X_k` | roll and pitch of `X_k` vs `E_k`; yaw free (`Pose3AttitudeFactor`) | `σ_rp` | every keyframe |
| Depth | `X_k` | `X_k.z − E_k.z` (`CustomFactor`) | `σ_z` | every keyframe |
| Tag URDF prior | `T_j` | `T_j ⊖ T_j^URDF` | `σ_mount` | once per tag, at creation |
| Tag projection | `X_k, S, T_j` | `π(K_C, (X_k · T_bc)⁻¹ · S · T_j · c_i) − uv_i` for i = 1..4 (8 rows) | 1 px per coordinate, Huber | each accepted tag observation |

### Example factor graph

Also available on its own page: [DOCK_FUSER_FACTOR_GRAPH.md](DOCK_FUSER_FACTOR_GRAPH.md).

![Factor graph example](dock_fuser_factor_graph.svg)

One tag, three keyframes during an approach. `X_0` and `X_2` see tag a; `X_1` is a fixed-tick keyframe with no tag in view.

Circles are variables (solved for); squares are factors (measurements or priors on the variables they connect to).

| Symbol | Kind | Connects | Meaning |
|---|---|---|---|
| `X_k` (blue) | variable | | AUV pose in `odom` at keyframe k. Starts at the EKF-predicted pose; tags pull it off the EKF when the EKF drifts. |
| `S` (orange) | variable | | Station `dock_point` pose in `odom`. One for the whole run (the station is static). Docking uses `X_k⁻¹ · S`. |
| `T_a` (green) | variable | | Tag a's pose on the station, relative to `S`. Stays at its URDF value while only one tag is seen. |
| `EKF init` (black) | factor | `X_0` | `X_0 = E_0`, very tight. Added once; ties the graph to `odom`. |
| `odom` (blue) | factor | `X_{k-1}`, `X_k` | EKF motion `E_{k-1}⁻¹E_k` (relative only, since x, y, yaw drift). Huber. |
| `roll/pitch` (purple) | factor | `X_k` | Roll and pitch from the EKF (gravity); yaw free. Every keyframe. |
| `depth` (pink) | factor | `X_k` | Depth from the EKF (pressure): `X_k.z − E_k.z`. Every keyframe. |
| `proj` (red) | factor | `X_k`, `S`, `T_a` | One tag sighting: 4 observed corner pixels vs. corners predicted from `S · T_a` seen by the camera at `X_k · T_bc`. 1 px, Huber. |
| `URDF` (green) | factor | `T_a` | Keeps `T_a` near its station-URDF pose; σ = mounting tolerance. Added once. |

`S` and `T_a` have no direct link: the `proj` factors use them only as the product `S · T_a`, the tag's pose in `odom`.

How to read it:

- Every tag observation is one red square with three edges. All of them meet at `S`, which is how observations from different keyframes combine into one station estimate.
- The two observations are linked through the `odom` chain `X_0 – X_1 – X_2`. This is what makes single-tag initialization over several frames work (§7): the odometry says how the camera moved between the two views, and the static `S` must explain both.
- `X_1` has no projection factor. Its pose comes from odometry, roll/pitch and depth alone, and `S` keeps the value the observations gave it.
- With a single tag, its mounting error cannot be separated from the station pose: `S` absorbs it, and `T_a` stays at its prior. The dock point estimate is then only as accurate as that tag's URDF mounting. Additional tags work the same way: each one adds its own `T_j` with a URDF prior and its own projection factors, all meeting at the same `S`.

Source: [dock_fuser_factor_graph.dot](dock_fuser_factor_graph.dot). Re-render with `neato -n2 -Tsvg docs/dock_fuser_factor_graph.dot -o docs/dock_fuser_factor_graph.svg`.

Notes:

- **Why the EKF init prior is needed.** Odometry and projection factors only constrain poses relative to each other. Without it, the whole graph (every `X_k` together with `S`) could slide or rotate in x, y and yaw without changing any residual. Roll/pitch and depth priors already fix roll, pitch and z. It pins `X_0` to the EKF's first pose, which makes the graph frame equal to `odom`. It is the only factor that ties the graph to `odom`, and it is added once.
- **Why relative odometry plus absolute roll/pitch and depth.** The EKF's x, y and yaw drift, so they enter only as relative motion. Roll, pitch (gravity) and depth (pressure) do not drift, so they enter as absolute priors. Without them, noisy far-range corners can tilt `X_k` to fit the pixels.
- **Odometry noise.** `σ² = q_t·Δt + q_d·Δd` per axis, so the total uncertainty over an interval does not depend on the keyframe rate. The Huber kernel lets tags win a disagreement with the odometry.
- **One factor per tag.** Grouping the four corners into one 8-row `CustomFactor` keeps the four corners rigid and makes 4× fewer Python callbacks than one factor per corner.
- **Jacobians.** Chain them from `Pose3.compose`, `Pose3.transformFrom` / `transformTo` and the pinhole projection, all of which return Jacobians in the Python bindings. Every `CustomFactor` gets a finite-difference unit test.

---

## 6. Per-keyframe cycle

A keyframe is created at every camera frame that carries tag detections, and on a fixed tick (5 Hz) otherwise.

1. **Stamp.** Take the camera stamp t (or the tick). Interpolate `E(t)` from the EKF buffer; if t is newer than the newest EKF message, wait one EKF period.
2. **Create `X_k`.** Initial value `X̂_k = X_{k-1} · (E_{k-1}⁻¹ E_k)`.
3. **Motion factors.** Add odometry, roll/pitch and depth.
4. **Tag factors.**
   - Before `S` exists: buffer the detections and run initialization (§7).
   - In TRACKING: pre-filter detections, run the gates (§8), and add a projection factor for each accepted tag. A tag seen for the first time gets a new `T_j` with its URDF prior.
5. **Update iSAM2** with the new factors and values.
6. **Read back** `X_k`, `S` and the `T_j`.
7. **Publish** (§10).

---

## 7. Station initialization

Projection factors are non-convex and diverge from a poor initial guess, so `S`, `T_j` and tag factors enter the graph only after a reliable first estimate.

Initialization must work from a **single tag**: on approach, usually one tag is visible, and several tags appear only close in. A single tag in one frame is unreliable. It has two near-equal pose solutions (the flip ambiguity), and its range and orientation are poorly constrained at distance. Initialization therefore combines several frames of the same tag, linked by odometry.

**Why this works.** Over a few seconds the odometry chain `X̂_k` is accurate, and the station is static. The correct station hypothesis therefore stays at the same place in `odom` from frame to frame. The flipped hypothesis moves as the viewpoint changes. The AUV's motion also adds parallax, which constrains range and orientation better than any single frame.

```
NO_STATION ──first usable tag frame──► CANDIDATE ──batch solve accepted──► TRACKING
                                            │
                                            └──window expires without a winner: slide the window
```

Before `S` exists, the graph carries only the `X` chain and tracks the EKF. Tag frames are buffered, not dropped.

### 7.0 From one tag observation to a station pose

Tags are known only in the station frame, and the AUV only in `odom` (§2, Frames). One observation of tag j closes the loop between the two:

```
   tag j in odom, via the AUV:      X_k · T_bc · T_C_tag      (odometry · extrinsic · PnP)
   tag j in odom, via the station:  S · T_j                    (unknown station · URDF)

   Both describe the same physical tag, so:

        S · T_j = X_k · T_bc · T_C_tag
   ⇒    S       = X_k · T_bc · T_C_tag · T_j⁻¹
```

- `X_k`: AUV pose in `odom` from the EKF (known).
- `T_bc`: camera extrinsic from the AUV URDF (known).
- `T_C_tag`: tag pose in the camera frame, from PnP on the 4 corners with the known tag size (measured).
- `T_j`: tag pose in the station frame, from the station URDF (known).

This gives a first value for `S`. The tag offsets start at their URDF values, `T_j₀ = T_j^URDF`, because they are defined relative to `S`, not to `odom`. After that, the optimizer refines `S` and the `T_j` from all projection factors.

**What is actually solved with one tag.** The projection factors only see the product `S · T_a`, which is tag a's pose in `odom`. That product is what the data determines. How it splits between `S` and `T_a` is decided only by the URDF prior, so `T_a` stays at its URDF value and `S = (S · T_a) · T_a⁻¹`. In effect the graph solves for the tag's pose in `odom` and converts it to the station through the known URDF offset. `T_a` becomes informative only when a second tag is seen.

**Drift before the first sighting is harmless.** The tag may first appear long after `robot_localization` started, when the EKF's x, y and yaw have already drifted. `S₀` is computed from that drifted `X_k`, so it is placed where the station is *in the EKF's current, drifted `odom`*, not where it would be in a perfect `odom`. That is the frame everything else uses. The quantity docking needs, the station relative to the AUV, is

```
X_k⁻¹ · S₀ = T_bc · T_C_tag · T_a⁻¹
```

which contains no odometry at all: the drift cancels. Before `S` exists there are no factors that could disagree with the odometry, so the drift causes no conflict in the graph either.

**Drift after initialization is what the graph corrects.** Once `S` is in the graph it is fixed in `odom`, but the EKF keeps drifting. Each new tag observation then disagrees slightly with the odometry chain, and the optimizer resolves it by moving `X_k` away from `E_k`. The difference between `X_k` and `E_k` is the estimated drift since initialization. The output must account for it (§10). Only the drift *within* the initialization window matters for initialization itself, and over a few seconds it is small.

Running PnP with the corners expressed directly in the `dock_point` frame (as below) gives `T_C_dock = T_C_tag · T_j⁻¹` in one step, so `S = X_k · T_bc · T_C_dock`. With several tags visible, all their corners go into one PnP.

### 7.1 Per-frame hypotheses

For each buffered tag frame that passes gate 1 (§8):

1. Run PnP on the corners, expressed in the `dock_point` frame:
   - **Single tag:** `cv2.SOLVEPNP_IPPE_SQUARE` returns both planar solutions.
   - **Several tags:** `cv2.SOLVEPNP_SQPNP` over all corners gives one solution. Run gate 2 first.
2. Turn each PnP solution into a station hypothesis: `S_h = X̂_k · T_bc · T_C_dock`.
3. Prune hypotheses using information the single frame already has:
   - **Reprojection ratio.** If the better solution's reprojection error is below 0.3 × the other's, the frame is unambiguous: keep only the better one. Otherwise keep both.
   - **Station tilt.** Roll and pitch of `X̂_k` are absolute (gravity, from the EKF), so each hypothesis implies a station roll and pitch in `odom`. Drop a hypothesis whose station tilt exceeds what the seabed allows (for example 15°). A flipped solution tilts the tag plane, so it usually implies a visibly tilted station. This check often resolves the flip from one frame.

### 7.2 Multi-frame voting

Collect the surviving hypotheses over a sliding window (for example up to 3 s, or 5–10 tag frames). Each frame votes for at most one of its hypotheses. Find the station pose supported by the most frames within a tolerance of a few cm and a few degrees.

Move to the batch solve when:

- the winner is supported by at least N frames (for example 5),
- the supporting frames have **viewpoint diversity**: the viewing direction to the tag changes by at least a few degrees, or the camera moves laterally by at least a few cm. Without this, extra frames only average noise and cannot separate the flip.
- any competing cluster has clearly fewer votes.

### 7.3 Batch solve

Build a small graph from the window:

- the `X_k` of the buffered frames, held near their current estimates by loose priors, plus the odometry, roll/pitch and depth factors between them,
- `S`, initialized to the winning hypothesis,
- `T_j` for each tag seen in the window, with its URDF prior,
- projection factors for every buffered detection.

Run Levenberg–Marquardt. If a competing hypothesis exists, run the same graph again initialized from it.

Accept when:

- the maximum whitened residual is below a threshold, and
- the competing solution, if any, ends with a clearly higher error (for example ≥ 2×).

**On accept:** insert `S`, the `T_j`, the URDF priors and **all** buffered projection factors into iSAM2 (the `X_k` already exist there), and switch to TRACKING.

**On reject:** drop the oldest frames and keep collecting.

### 7.4 After initialization

- A station initialized from one tag can still be loose, mostly in range and in rotation about the tag's vertical axis. For a short period after initialization (for example 5 s), use a looser threshold for gate 3.
- New tags enter through the normal tracking path. Each one gets a `T_j` initialized from `S · T_j^URDF` with its URDF prior, and its observations pass the usual gates. Every additional tag tightens `S`.
- With the fixed-lag backend, the init window must be shorter than the lag so the buffered `X_k` still exist when the factors are inserted.

---

## 8. Tag gating

Gates run from cheapest to most expensive. A tag that fails any gate is dropped for that frame only.

1. **Pre-filter.** Minimum tag area, minimum viewing angle, minimum distance from the image border, and all four corners in front of the camera at the predicted pose. A factor whose corners fall behind the camera must never be added.
2. **Multi-tag consistency.** With two or more tags in one frame, compare their relative poses from PnP with the URDF. Drop a tag that disagrees with the others. This check uses no odometry.
3. **Predicted reprojection.** Project the tag's corners from `X̂_k`, `S` and `T_j`. Reject if the RMS error exceeds a few pixels.
4. **Local solve.** Build a small graph from the new factors plus tight priors holding the current estimates, run a few Levenberg–Marquardt iterations, and admit the factors only if the maximum whitened error is below a threshold. The initialization batch solve (§7.3) is this gate applied to a whole window; in steady tracking gate 3 is usually enough. (This is TagSLAM's subgraph-initialization idea.)

The flip ambiguity matters only when a pose is initialized from PnP (§7). In TRACKING, tag factors are linearized at the graph's own estimate of `X_k`, `S` and `T_j`, never at a PnP solution, so a flipped detection cannot enter directly. If it disagrees with the estimate, gate 3 rejects it.

---

## 9. Backend

The backend is iSAM2, wrapped so the variant is a single setting:

| | `gtsam.ISAM2` | `gtsam.IncrementalFixedLagSmoother` |
|---|---|---|
| What it is | Plain iSAM2 | The same iSAM2, plus marginalization of variables older than the lag |
| Graph size | Grows with time | Bounded by the lag (e.g. 30 s) |
| Use for | Development, offline bag replay, short runs | Long runs on the vehicle |

Fixed-lag specifics:

- The smoother treats the newest timestamp as "now". Re-stamp `S` and every `T_j` with the current time on each update so they are never marginalized. Do not give them a far-future timestamp: that marginalizes everything else immediately.

Error handling:

- Catch exceptions from the update (for example `IndeterminantLinearSystemException`). Rebuild the backend from the last good estimate, with priors on `X_k`, `S` and `T_j` taken from their marginal covariances, and keep publishing. The node never exits on an optimizer error.

---

## 10. Output

| Output | Value | Rate |
|---|---|---|
| TF `odom → station base_link` | `E_k · X_k⁻¹ · S · T_dock_base`, where `T_dock_base` is the inverse of the station URDF `dock_point_joint` origin | EKF rate (holds the last value between solves) |
| `/dock_point/pose` | `X_k⁻¹ · S` in `base_link`, with the marginal covariance of `S` | each keyframe |
| Diagnostics | station state, tags accepted / rejected, update time, residuals per factor type | each keyframe |

The station's own `robot_state_publisher` hangs `dock_point` and the tag frames below its `base_link`, so they appear in RViz without extra code.

**Why `E_k · X_k⁻¹ · S` and not just `S`.** The EKF keeps publishing `odom → base_link = E_k`, which drifts. The graph's corrected AUV pose is `X_k`, and its station is `S`. The TF tree composes `odom → base_link` with `odom → station`, so `base_link → station = E_k⁻¹ · (odom → station)`. Publishing `E_k · X_k⁻¹ · S` makes that equal `X_k⁻¹ · S`, the graph's corrected relative pose. Publishing `S` alone would give `E_k⁻¹ · S`, which mixes the drifted EKF pose with the graph's station and is wrong whenever `X_k ≠ E_k`. This is the same role `map → odom` plays in REP-105: the edge absorbs the drift correction, so it moves slowly as drift accumulates even though the station itself is static.

Corrections to `odom → station` are rate-limited (interpolated toward the new value) so the controller never sees a jump.

---

## 11. Code layout

The `dock_fuser` ament_python package in this repo:

```
dock_fuser/
  package.xml, setup.py, setup.cfg
  config/
    dock_fuser.yaml       # cameras and K_C (§3), noise parameters
  dock_fuser/
    core/                 # pure Python + gtsam, no ROS imports
      keys.py             # symbol helpers, timestamp bookkeeping
      factors.py          # tag projection (8-row), depth
      odometry.py         # relative motion and its noise
      init.py             # PnP hypotheses, ratio and tilt pruning, multi-frame voting, batch solve
      gating.py           # gates 1–4
      station.py          # NO_STATION / CANDIDATE / TRACKING, detection buffer
      estimator.py        # owns the backend, runs the per-keyframe cycle
    ros/
      node.py             # subscriptions, EKF buffer and interpolation, params, TF and pose output
    tools/
      replay_bag.py       # bag → core estimator, offline and deterministic
      evaluate.py         # error vs sim truth, leave-one-tag-out, held-out frames
  test/
    test_jacobians.py     # finite-difference check for every CustomFactor
    test_scenarios.py     # synthetic: clean approach, single tag, tags lost and reacquired
```

`core/` has no ROS dependency, so the same code runs live in the node, offline on bags, and in unit tests.

---

## 12. Build order

1. `factors.py` and `test_jacobians.py`.
2. `estimator.py` with odometry, roll/pitch and depth only, plus `replay_bag.py`. Check that `odom → station` stays fixed and `X_k` tracks the EKF.
3. Tag factors, gates 1–3, and single-tag multi-frame initialization (`init.py`, NO_STATION → CANDIDATE → TRACKING). Validate in simulation against Stonefish ground truth, including approaches where only one tag is visible for a long time.
4. Gate 4 and error recovery.
5. `node.py`, then profile on the vehicle computer and choose the backend variant.

---

## 13. Validation

A residual on detections the graph was fitted to measures self-consistency, not accuracy.

- **Simulation:** compare `S` and `X_k` with Stonefish ground truth.
- **Leave-one-tag-out:** solve without tag j's factors, predict its corners, and compare with its detections.
- **Held-out frames:** withhold every Nth tag frame and report its reprojection error.
- **Mounting residuals:** whitened residuals of the tag URDF priors above 3 flag a tag whose real mounting differs from the URDF.
- **Initialization:** in simulation, record how often a single-tag initialization picks the flipped solution, and how long after the first tag sighting it takes to reach TRACKING.

---

## 14. Relation to TagSLAM

The pipeline uses the same measurement model as TagSLAM (Pfrommer & Daniilidis, 2019) but is narrower in scope and differs in anchoring, priors and initialization.

**Taken from TagSLAM**

| Idea | TagSLAM | Here |
|---|---|---|
| Tag projection factor | 4 corners at `(±l/2, ±l/2, 0)`, pinhole projection, ~1 px noise | Same, one factor per tag |
| Tag pose relative to its body, with optional prior | Tag pose w.r.t. body | `T_j` relative to `S`, with URDF prior |
| Odometry as a relative pose prior | Relative pose prior between body poses | Odometry factor |
| Every frame is a keyframe, iSAM2 backend | Yes | Yes |
| Admit new measurements only after a local solve passes an error threshold | Subgraph optimized non-incrementally, accepted below a threshold | Gate 4 and the initialization batch solve |
| Use earlier measurements once their variables become solvable | Full graph keeps all measurements; they enter the optimized graph when determinable | Tag frames before `S` exists are buffered and inserted at initialization |
| Ambiguity test before a tag pose is trusted | Ambiguity error ratio and viewing angle | Reprojection ratio of the two IPPE solutions, at initialization |

**Different**

| | TagSLAM | Here |
|---|---|---|
| Scope | General: any number of static and dynamic bodies, tags and cameras | One station, one AUV |
| World frame | Anchored by an absolute prior on a static body | `odom`, anchored by `X_0 = E_0`; the station `S` is unknown |
| Absolute priors on the moving body | None | Roll, pitch and depth from the EKF at every keyframe |
| Camera extrinsics | Variables (can be calibrated) | Fixed from the URDF |
| Distortion | Inside the projection factor | Rectified image, plain pinhole, `K_C` from YAML |
| Noise model | Gaussian | Huber on tag and odometry factors; random-walk odometry noise |
| Initial pose estimate | Homographies from a minimal set of tags; factor ordering heuristic with retries | PnP hypotheses, tilt pruning, multi-frame voting over an odometry-linked window, batch solve |
| Graph growth | Full history (option to keep only the last two time steps) | Plain iSAM2 or fixed-lag smoother |
| Output | Body poses in the world frame | `odom → station` TF composed with the EKF, plus `/dock_point/pose` |

**Not included:** multiple moving bodies, extrinsic calibration, distance and coordinate measurement factors, and loop closure over large tag maps.

**Note on PnP.** The TagSLAM paper reports that PnP is not robust when tag poses in the map are wrong, and uses homographies instead. Here tag poses come from the URDF, and the multi-tag consistency gate catches a wrong one. If initialization proves fragile in testing, switch to TagSLAM's homography approach.
