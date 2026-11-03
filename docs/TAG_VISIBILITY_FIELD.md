# Tag Visibility Field

This note models, from first principles:

- where the AUV can see the docking-station tags;
- how likely a camera frame is to detect them;
- what each detection is worth to the factor graph;
- how to use all three for guidance (path and heading planning) and for choosing where the tags go.

**The method is built in levels.** Level 0 is plain geometry with two unknown thresholds and explicitly stated assumptions. Each higher level adds one physical effect, and should be added only when the data shows the level below it is not enough. The *explanations* are complete at every level: every equation is derived, interpreted in words, and evaluated with our hardware's numbers.

The numbers are for this vehicle and station:

- `cam_front`: Stellar camera, underwater calibration, rectified;
- the station's tag36h11 tags;
- the detector settings in `race_auv_bringup/config/tag_factorgraph_detector.yaml`.

Values marked **(assumed)** are placeholders until they're measured. Appendix B lists all hardware numbers with their sources.

**How to read this.** Part I (§4–9) is self-contained: it is the model to use first. Part II (§10–14) adds effects, information and planning; read a section when its effect becomes relevant.

## Contents

1. [Goal and approach](#1-goal-and-approach)
2. [Key results](#2-key-results)
3. [Frames and notation](#3-frames-and-notation)

**Part I: Level 0, geometry**

4. [Assumptions](#4-assumptions)
5. [Geometry of one camera and one tag](#5-geometry-of-one-camera-and-one-tag)
6. [Pixels per cell](#6-pixels-per-cell)
7. [The Level-0 visibility rule](#7-the-level-0-visibility-rule)
8. [The visibility volume](#8-the-visibility-volume)
9. [Checking Level 0 against the bags](#9-checking-level-0-against-the-bags)

**Part II: extensions, added only when needed**

10. [Level 1: detection probability](#10-level-1-detection-probability)
11. [Level 2: information](#11-level-2-information)
12. [Level 3: guidance and placement](#12-level-3-guidance-and-placement)
13. [Evaluation plan](#13-evaluation-plan)
14. [Where learning fits](#14-where-learning-fits)
15. [Related work](#15-related-work)
16. [Recipe](#16-recipe)

- [Appendix A: symbols](#appendix-a-symbols)
- [Appendix B: hardware numbers](#appendix-b-hardware-numbers)
- [Appendix C: Fisher information of one tag](#appendix-c-fisher-information-of-one-tag)

---

## 1. Goal and approach

### 1.1 Goal

**Dock reliably.** The factor graph estimates the docking pose $X^{-1}S$, the station's pose relative to the AUV. That estimate must stay within the docking tolerance throughout the approach, and especially in the last few metres, where the vehicle commits to the dock. Between tag sightings the estimate drifts with odometry. So the vehicle must see tags often enough, and from good enough viewpoints, that the drift never exceeds the tolerance.

The visibility model serves four uses:

1. **Diagnose.** Explain the gaps in tag detections seen in the bags. For each gap: was the camera pointed away, was the vehicle too far, was the viewing angle too steep, was a tag blocked, or did detection fail where geometry says it should have worked? The 2026-10-22 run had a 101.6 s stretch with no tags; Level 0 can say why.
2. **Visibility-aware guidance.** Plan heading and path so the tags stay in view, and the time between detections stays below what the drift allows.
3. **Information-aware guidance.** Plan heading and path so the detections actually pin down the docking pose. Visibility alone isn't enough: a path can see a tag constantly and still be poor for estimation, because a single far tag fixes its bearing well but its orientation badly (§11.2). Good viewpoints give parallax, line two tags up one behind the other (§11.3), or have several tags in view when the vehicle commits to the dock.
4. **Station layout.** Decide which tags, of what sizes, and where.

### 1.2 Why levels

A model can only be trusted as far as the data can pin down its parameters. Right now the bags contain detections out to the far end of the path, but no recorded *misses* there (§9.4). A five-parameter detection model fitted to those bags would produce numbers, but the data could not tell them apart: the fit would look confident and mean nothing. So the method starts with the fewest unknowns that can explain what we see, and adds a parameter only when the data both demands it and can determine it.

| Level | Model | What it adds | Unknowns | Move up when |
|---|---|---|---|---|
| **0** | geometry plus two thresholds (§4–9) | visible yes/no, the visibility volume, allowed headings, the cause of each gap | $\rho_\text{min}$, $\theta_\text{max}$ | Level 0 misclassifies many keyframes, or planning needs a risk measure |
| **1** | logistic detection probability (§10) | soft edges, motion blur, water, off-axis and steep-angle losses | $a_0..a_5$, plus the exposure $t_e$ and water attenuation $c$ | visibility is fine but the docking-pose error is not |
| **2** | information (§11) | which detections fix which part of the docking pose | $\sigma_\text{px}$, odometry noise $Q$ | a fixed guidance rule isn't good enough |
| **3** | optimization (§12) | heading schedules, paths, tag layout | none new | n/a |

Uses 1 and 2 need only Level 0 or 1. Use 3 needs Level 2. Use 4 needs Level 2 for a good answer, but Level 0 already explains why tags come in nested sizes (§8.6).

---

## 2. Key results

1. **One number mostly decides whether a tag decodes: pixels per tag cell.**

   $$\rho = \frac{f\,s\cos\theta}{b\,d}$$

   Here $f$ is the focal length (px), $s$ the tag side, $b = 8$ the number of cells across a tag36h11 black square, $d$ the range and $\theta$ the viewing angle. This is eq. (5), derived from the pinhole model in §6.1: weak perspective, then foreshortening by $\cos\theta$, then the $f/d$ scale. Tag size, camera and range all enter only through $\rho$, so a threshold on $\rho$ carries over to other tags and cameras (§6.3).

2. **The "visibility cone" of a tag is really a ball** (§8.2).
   - The set of camera positions where $\rho \ge \rho_\text{min}$ is a ball of diameter $f s/(b\rho_\text{min})$. It touches the tag plane at the tag centre and extends along the tag's normal: 16 m for tag 54 if $\rho_\text{min} = 2$.
   - The limit from rotational motion blur is a ball of the same kind, which shrinks as the turn rate grows (§10.2).
   - The steep-angle limit and translational blur are cones around the normal.
   - Turbidity caps range with a ball centred on the tag, and that cap doesn't grow with tag size.

3. **The camera's field of view mostly restricts heading, not position** (§8.3). At any position you can point the camera at the tag, as long as the tag's elevation fits the vertical field of view. Close to the dock, that elevation condition makes depth decide visibility.

4. **The field lives in 4D: camera position plus heading** (§8.4).
   - With a free heading (sway thruster), maximizing over heading leaves a 3D volume.
   - With heading tied to the path, it doesn't.

5. **One far tag says where the tag is, not where you are relative to its axis** (§11.2).
   - Tag 54 at 6 m, with 1 px corner noise, gives its bearing to about 4 mm but its tilt only to about 73°.
   - That leaves the vehicle's position relative to the tag's axis uncertain by about 7.7 m.
   - Tags 54 and 146 sit 1.27 m apart along the line of sight, so seen together they act like a ship's **leading lights** (§11.3). That gives about 2.3 cm lateral and 3.9 cm vertical precision at 6 m.

6. **Guidance and placement maximize the same objective** (§12).
   - Guidance: maximize detections, bound the gaps, or maximize information along a path.
   - Placement: the same objective with tag poses as the variables. This is a submodular selection problem, so greedy selection carries a guarantee.

---

## 3. Frames and notation

$T_{AB}$ is the pose of frame $B$ in frame $A$: it maps coordinates in $B$ to coordinates in $A$. This is the factor graph's convention (`core/factors/projection.py`).

| Symbol | Pose of | In frame | Source |
|---|---|---|---|
| $X = T_{OB}$ | AUV `base_link` | odom | graph variable $X_k$ |
| $S = T_{OS}$ | station `dock_point` | odom | graph variable $S$ |
| $T_{SJ}$ | tag $j$ | station | station URDF, link `apriltag36h11_<id>` |
| $T_{BC}$ | camera | `base_link` | AUV URDF, links `cam_front`, `cam_down` |
| $T_{CJ}$ | tag $j$ | camera | derived, eq. (1) |

The tag as seen by the camera is the chain camera ← AUV ← odom ← station ← tag:

$$T_{CJ} = T_{BC}^{-1}\,X^{-1}\,S\,T_{SJ} \tag{1}$$

This is the same chain as `project_tag`. Docking uses the relative pose $Y = X^{-1}S$, the station in the AUV frame. $Y$ is the quantity that matters because it does not drift: odometry drift moves $X$ and $S$ together in the odom frame, and cancels in $X^{-1}S$.

**Camera.** The graph uses the rectified pinhole camera from `camera_info`. Its axes: $z$ forward along the optical axis, $x$ right, $y$ down. A point $\mathbf p = (x, y, z)$ in the camera frame projects to the pixel

$$u = f_x\,\frac{x}{z} + c_x,\qquad v = f_y\,\frac{y}{z} + c_y \tag{2}$$

$(c_x, c_y)$ is the principal point, where the optical axis meets the image. The image is $W \times H$ pixels. Below, $f$ means $f_x \approx f_y = 1165.2$ px. Since $v$ grows downward, a point above the optical axis has $v < c_y$.

**Tag.** A tag36h11 marker is a grid of square **cells**:

- 6×6 data bits in the middle;
- a 1-cell black border around them, making $b = 8$ cells across the black square of side $s$;
- a 1-cell white quiet zone outside that, making 10 cells across the printed sheet.

The tag side $s$ in `station_tags.yaml` is the side of the black square, so each cell has side $s/b$. Check: the URDF gives 22 cm black and 27.5 cm printed for tag 54, and $0.22 \cdot 10/8 = 0.275$.

The tag frame has its origin at the tag centre and its $x$, $y$ axes in the tag plane. The four corners are $\mathbf q_i = (\pm s/2, \pm s/2, 0)$. $\hat{\mathbf n}$ is the **outward** unit normal, pointing from the printed face toward a viewer. In apriltag3's convention the tag's $z$ axis points *into* the tag, so $\hat{\mathbf n} = -\hat{\mathbf z}_J$. Check which way `TagModel` uses before coding.

**Field vs. distribution.** Two different things are called "probability" below, and they must not be confused:

- the visibility **field** gives, at each vehicle configuration, the probability that a frame taken there detects the tag. Each point carries its own yes/no chance, and the field does *not* integrate to 1 over space;
- the vehicle's **belief** about its own pose is a probability distribution, which does integrate to 1. It enters only in §11.5, where the field is averaged over it.

---

# Part I: Level 0, geometry

## 4. Assumptions

Level 0 rests on seven assumptions. Each one says what is ignored, why that is acceptable for now, how to tell when it stops being acceptable, and where a later level removes it.

**A1. Rectified pinhole camera, with $f = 1165.2$ px everywhere.** The detector works on the rectified image, and the graph uses its pinhole model. The real lens is a fisheye, so the pinhole slightly understates resolution away from the image centre: by at most 8% radially and 13% tangentially, at the image edge (§6.1). The error is always an underestimate, so the predicted volume is slightly too small, never too large. A 13% error in $\rho$ moves the visibility boundary by at most 13% of its size, which is small next to our uncertainty about $\rho_\text{min}$ (2–4). *Removed in:* §10.3, through the off-axis term.

**A2. Roll = pitch = 0.** The AUV is passively stable: its centre of buoyancy sits above its centre of gravity, which creates a righting moment. This removes two of the six pose dimensions. If the vehicle pitches noticeably while manoeuvring, the camera's vertical field of view tilts by the same angle: 1° of pitch moves the elevation cone (§8.3) by 1°. *Check:* the IMU's pitch during approaches.

**A3. No motion blur.** Valid while the vehicle turns and moves sideways slowly. For example, with a 10 ms exposure **(assumed)**, a yaw rate below 0.1 rad/s smears tag 54 at 6 m by less than a quarter of a cell (eq. 10). *Check:* once the exposure is logged, evaluate eq. (10) along the bag. *Removed in:* §10.2.

**A4. Clear water.** Turbidity limits range independently of tag size (§10.2). In the tank, that limit is far beyond the tank's size, so only resolution matters. Open water may be different. *Removed in:* §10.2.

**A5. Hard thresholds.** A tag is detected if and only if the rule in §7 holds. In reality detection fades out over a band near the boundary. A hard rule is accurate everywhere except inside that band, so it predicts well as long as the band is thin compared with the volume. *Check:* the unexplained misses and hits in §9.1 cluster near the boundary. *Removed in:* §10.

**A6. Tag poses are as in the URDF, and tags are flat.** The URDF tag poses were measured on the tank station on 2026-10-21. Occlusion is checked explicitly (G4 in §7). The graph also estimates the tag poses $T_j$, so a large URDF error would show up there.

**A7. For the bag checks, the batch-LM pose is accurate enough to predict visibility.** This fails near the volume's boundary during long gaps, when the pose estimate is least certain. §9.2 handles it with margins.

---

## 5. Geometry of one camera and one tag

Everything in Level 0 is computed from the relative pose $T_{CJ}$ of eq. (1). From it, take the tag centre in the camera frame and the camera centre in the tag frame:

$$\mathbf p = \mathbf t(T_{CJ}),\qquad d = \lVert\mathbf p\rVert,\qquad \hat{\mathbf u} = \mathbf p/d,\qquad \mathbf c_J = -R_{CJ}^{\top}\mathbf p \tag{3}$$

$\mathbf t(\cdot)$ and $R$ are the translation and rotation parts of a pose. $d$ is the range, and $\hat{\mathbf u}$ is the unit line of sight from the camera to the tag. $\mathbf c_J$ is where the camera is, seen from the tag.

Two angles do most of the work:

$$\cos\theta = \hat{\mathbf n}\cdot(-\hat{\mathbf u}),\qquad \cos\alpha = \hat{\mathbf z}_C\cdot\hat{\mathbf u} \tag{4}$$

- **Viewing angle $\theta$** is measured at the tag, between its normal and the direction back to the camera. $\theta = 0$ means the camera sees the tag head-on; $\theta = 90°$ means edge-on.
- **Off-axis angle $\alpha$** is measured at the camera, between its optical axis and the direction to the tag. $\alpha = 0$ puts the tag at the image centre; at the image's left or right edge, $\alpha \approx 34.5°$.

```
   tag plane
       │
       │   n̂  (outward normal)
       ●───────────────────────▶
        ╲  θ
         ╲
          ╲   line of sight, length d
           ╲
            ◉ camera ────────▶ ẑ_C  (optical axis)
              α = angle between ẑ_C and the line back to the tag
```

**The split that structures everything below:**

- $d$ and $\theta$ depend only on **where the camera is** relative to the tag. Turning the camera in place changes neither.
- $\alpha$, and whether the tag lands inside the image at all, also depend on **which way the camera points**.

So "can this tag be read from here?" (resolution, viewing angle) and "is the camera looking at it?" (field of view) are separate questions. §8 answers them separately and then combines them.

The predicted corner pixels are $\pi(R_{CJ}\mathbf q_i + \mathbf p)$ for $i = 1..4$: each tag-frame corner is moved into the camera frame, then projected with eq. (2) (written $\pi$).

---

## 6. Pixels per cell

### 6.1 Deriving $\rho$

To decode a tag, the detector must tell every cell apart from its neighbours. So the key question is: how many pixels does one cell cover in the image?

**Step 1: weak perspective.** Perspective projection divides each point's position by that point's depth. A tag is small compared with its range ($s \ll d$; 0.22 m against 1–10 m), so all its points have nearly the same depth $d$, and dividing by one common $d$ is accurate to about $s/d$ (4% for tag 54 at 6 m). With one common depth, the projection of the tag is a scaled version of its shadow on the plane perpendicular to the line of sight. This is the **weak-perspective** approximation.

**Step 2: foreshortening.** Tilt a square so its normal makes angle $\theta$ with the line of sight, and look at its shadow on the plane perpendicular to that line:

- lengths along the tilt axis are unchanged;
- lengths across the tilt axis shrink by $\cos\theta$.

A coin seen at an angle looks like an ellipse for the same reason.

**Step 3: scale.** The pinhole scales lengths at range $d$ by $f/d$ pixels per metre.

Together: a cell of side $s/b$ images to a parallelogram with sides $\frac{f}{d}\frac{s}{b}$ and $\frac{f}{d}\frac{s}{b}\cos\theta$. Decoding needs every cell resolved in every direction, so the shorter side is what limits it:

$$\boxed{\;\rho = \frac{f\,s\cos\theta}{b\,d}\;}\qquad[\text{pixels per cell}] \tag{5}$$

**In words:** pixels per cell grows with focal length and tag size, and falls with range and with viewing angle.

**Example.** Tag 54 at 6 m: head-on, $\rho = 1165.2 \cdot 0.22/(8 \cdot 6) = 5.3$, so the whole tag is $b\rho = 43$ px wide. At 45°, $\rho = 3.8$.

**Use eq. (5) everywhere**: in the Level-0 rule, as a calibration feature, and in the field. Don't replace it with the smallest singular value of the rectified projection Jacobian, $(s/b)\,\sigma_\text{min}(A)$, even though that looks more exact. Here is why.

A rectified pinhole image stretches objects away from the centre: by $1/\cos\alpha$ tangentially and $1/\cos^2\alpha$ radially. At the 34.5° image edge, that's ×1.21 and ×1.47. But the lens behind the rectified image is a fisheye. Its distortion polynomial is within 0.25% of the equidistant model $r = f_\text{raw}\vartheta$ out to 0.6 rad (34°), with $f_\text{raw} = 1262.7$ px/rad. An equidistant lens puts the same number of pixels on every degree of view *radially* (toward or away from the image centre), so the raw sensor samples about 1263 px per radian radially everywhere. *Tangentially* (around the centre) it samples $f_\text{raw}\,\alpha/\sin\alpha$, which grows slowly off-axis: 1263 at the centre, 1342 at the 34.5° edge. Rectification then resamples the sensor:

- at the centre, the rectified image has 1165 px/rad, slightly *fewer* than the sensor recorded (×0.92). Here rectification throws away a little real detail, so the rectified image is the limit;
- off-axis, it has up to 1716 px/rad radially and 1414 tangentially at the edge, *more* than the sensor recorded. The extra pixels are interpolated between real ones and carry no new detail, so here the sensor is the limit.

So the detail the detector actually receives, per radian of view, is the *smaller* of the two:

| $\alpha$ | 0° | 10° | 16° | 20° | 25° | 30° | 34.5° |
|---|---|---|---|---|---|---|---|
| radial (px/rad) | 1165 | 1201 | 1261 | 1263 | 1263 | 1263 | 1263 |
| tangential (px/rad) | 1165 | 1183 | 1212 | 1240 | 1286 | 1322 | 1342 |
| eq. (5) understates by (radial / tangential) | 0% | 3% / 1.5% | 8% / 4% | 8% / 6% | 8% / 9% | 8% / 12% | 8% / 13% |

The Jacobian would count the interpolated pixels as real resolution. For a tag tilted away from the image centre, it would also partly cancel the real $\cos\theta$ foreshortening. Eq. (5) with $f = 1165.2$ never counts interpolated pixels. It is exact at the image centre and **always errs low**: by up to 8% radially (reached at $\alpha \approx 16°$, where $1165.2/\cos^2\alpha$ meets $f_\text{raw}$) and up to 13% tangentially (at the 34.5° edge). Erring low is the safe direction: a tag predicted visible is visible, and the visibility volume is slightly too small, never too large. The Level-1 off-axis term (§10.3) absorbs the difference.

**So $\rho$ depends on where the camera is, not where it points.** Eq. (5) uses only the range $d$ and the viewing angle $\theta$, both fixed by the camera's *position* relative to the tag. The off-axis angle $\alpha$, set by the camera's *orientation*, is left out on purpose: as the table shows, it changes the real resolution by at most 13%, and always upward. Read $\rho$ as: "from this point in space, a camera pointed so that the tag is in view sees at least $\rho$ pixels per cell." Whether the camera *is* pointed that way is a separate question, answered by the field-of-view gates G1 and G2 (§7). That split, position in eq. (5) and orientation in the gates, is what makes the visibility volume factor into a tag side and a camera side (§8). The one gate that sees the off-axis stretch is G5 (`min_side_px`), because the detector applies it in the rectified image; it corresponds to $\rho \ge 1.875$ at the centre and is easier to pass off-axis, so it rarely binds once $\rho_\text{min} \ge 2$.

The **one place rectified pixels do matter** is the detector's G5 check (`min_side_px`, §7): the detector applies it in the rectified image, so compute that check from the projected corners.

**Where eq. (5) comes from in the literature.** Eq. (5) is derived here from the pinhole model; no paper we found states it in this form. Each ingredient, and the idea of thresholding on it, has a precedent (sources in §15):

- **Weak perspective** (Step 1) is the textbook affine-camera approximation: Hartley and Zisserman, *Multiple View Geometry*, ch. 6, "cameras at infinity". They give its error as $x_\text{affine} - x_\text{proj} = \frac{\Delta}{d_0}(x_\text{proj} - x_0)$: the depth relief over the range, times the offset from the principal point. For a tag the depth relief is at most $s\sin\theta$, which gives the "about $s/d$" accuracy of Step 1.
- **Resolution across the target decides the task.** Johnson (1958) found that observers detect, recognize and identify a target with 50% probability once about 1, 4 and 6.4 line pairs (2, 8 and 13 pixels) span it. "Pixels per cell" is the same idea for a machine reading a binary grid: the task is decoding one cell, so the count is taken across one cell.
- **AprilTag range scales with cell size.** Olson (2011) argues that, for the same printed size, a 16-bit tag reads only about 25% farther than a 36-bit one "due to the 4 pixel overhead of the borders". That is the cell-count ratio 10/8, i.e. range at a fixed pixels-per-cell threshold. Krogius et al. (2019) make the same argument for their 9×9 layout: printed at the same size, it has "a larger bit pitch, which improves its detection range."
- **AprilTag detection has been measured against range and against viewing angle, but separately**, in simulation, each with the other held fixed:
  - Olson (2011): detection rate and accuracy against viewing angle at a fixed range (Fig. 9), and against range with the tag facing the camera (Fig. 10).
  - Wang and Olson (2016): detection rate against range (Fig. 7), and orientation error (not detection) against viewing angle (Fig. 6). Their "off-axis angle" is the angle between the tag normal and the camera axis, i.e. our $\theta$, not our $\alpha$.

  Neither paper combines range, tag size, focal length and viewing angle into one variable. Eq. (5) and the collapse test (§10.4) do, and that combination is a small contribution of this work.

### 6.2 What the detector needs from $\rho$

The AprilTag detector (apriltag3) works in three stages. Each stage needs enough pixels per cell, and the strictest one sets the threshold $\rho_\text{min}$.

1. **Finding quads.** The detector downsamples the image by `quad_decimate` $q = 2$ (for speed), then looks for dark quadrilaterals: it thresholds the image, groups connected dark pixels, and fits four lines. The black border is one cell wide, so in the downsampled image it is $\rho/q$ pixels wide. Below about one downsampled pixel, the border breaks up and no quad is found. So $\rho \gtrsim q$ to $2q$, i.e. **2–4**.
2. **Refining corners.** With `refine_edges`, the detector refines the quad's edges on the full-resolution image. This needs the edges to be visible, which the previous stage already ensures.
3. **Decoding bits.** The detector reads each of the 36 data bits by sampling the full-resolution image at the predicted cell centres. The finest pattern is alternating black and white cells, which repeats every 2 cells, i.e. every $2\rho$ pixels. The sampling theorem needs at least two samples per repeat, so $\rho \ge 1$ in theory. In practice, lens blur, colour demosaicing, the JPEG compression of the image transport, and rectification interpolation each smear fine detail, which raises this to about **2–3**.

On top of these, our detector rejects any tag whose shortest projected side is under `min_side_px` = 15 px. That side spans $b = 8$ cells, so this check means $\rho \ge 15/8 = 1.875$.

| Mechanism | Condition | Value here |
|---|---|---|
| Software check `min_side_px` (G5) | $b\rho \gtrsim 15$ | $\rho \ge 1.875$ |
| Quad finding on the image downsampled by $q$ | border $\rho/q \gtrsim 1$ downsampled pixel | $\rho \gtrsim 2\text{–}4$ |
| Bit decoding at full resolution | sampling theorem plus blur and compression | $\rho \gtrsim 2\text{–}3$ |

The threshold $\rho_\text{min}$ is the largest of these, **probably 2–4 (assumed; start Level 0 at 2)**. The bags then bound it (§9.3).

### 6.3 Why $\rho$ generalizes

Take the logarithm of eq. (5):

$$\log\rho = \log f + \log s + \log\cos\theta - \log b - \log d \tag{6}$$

Every hardware choice and geometric quantity adds a separate term. So doubling the tag size has exactly the same effect on decoding as halving the range, or doubling the focal length.

**Example.** Tag 146 (12 cm) at 3.27 m has the same $\rho$ as tag 54 (22 cm) at 6 m. If resolution is what limits detection, both should be detected equally reliably.

**Example from the AprilTag paper.** The cell count enters the same way. A tag16h5 is 8 cells across including the white border, and a tag36h11 is 10. Printed at the same size, the 16h5 cell is $10/8 = 1.25$ times larger, so eq. (6) predicts 25% more range. That is the figure Olson (2011) gives.

This is why a threshold, or later a detection model, expressed in $\rho$ carries over to other tag sizes, lenses and downsampling settings without refitting. It holds *if* resolution is the limiting factor; §10.4 shows how to test that.

---

## 7. The Level-0 visibility rule

A tag is visible from camera configuration $(\mathbf c, \psi)$, camera position $\mathbf c$ and heading $\psi$, if and only if **all** of the following hold:

| Condition | Test | Why |
|---|---|---|
| G1, in front | every corner has camera-frame depth $> 0$ | a tag behind the camera can't be imaged |
| G2, in image | every corner is at least $m = 10$ px inside the image | the detector drops tags with a corner closer than `min_edge_px` to the border, because a clipped tag gives an unreliable pose |
| G3, not too steep | $\theta \le \theta_\text{max}$ | near edge-on, the cells are squeezed and the face may catch glare, so decoding fails even at high resolution. **$\theta_\text{max} = 60°$ to start (assumed)**; the data so far reaches 33° |
| G4, line of sight | the segment from camera to tag misses the station frame and the hull | the tag is hidden |
| G5, detector check | shortest projected side ≥ 15 px (rectified image) | `min_side_px` |
| R, resolution | $\rho \ge \rho_\text{min}$, eq. (5) | §6.2 |

The detector's other check, pose-fit RMS ≤ 1.5 px (`max_fit_rms_px`), depends on image noise and blur rather than geometry, so Level 0 can't predict it. Misses it causes show up as unexplained misses in §9.

Following the split in §5, write the result as a 0/1 field. One factor depends only on the camera position; the other also depends on heading:

$$V_0(\mathbf c, \psi) = \underbrace{\mathbb 1[\text{G3, G4, R}]}_{\text{position only}}\cdot\underbrace{\mathbb 1[\text{G1, G2, G5}]}_{\text{position and heading}} \tag{7}$$

$\mathbb 1[\cdot]$ is 1 when all listed conditions hold and 0 otherwise. (G4 depends slightly on heading through the hull, and G5 slightly through the image position; both are small effects.)

**Level 0 has two unknowns: $\rho_\text{min}$ and $\theta_\text{max}$.** Everything else is geometry or a configuration value.

---

## 8. The visibility volume

### 8.1 Configuration space

The full pose has 6 degrees of freedom. With roll = pitch = 0 (A2), 4 remain:

- the camera-centre position $\mathbf c$, in the tag or station frame;
- the heading $\psi$.

Using the camera centre, rather than `base_link`, as the coordinate makes the split in eq. (7) exact: resolution and viewing angle depend only on $\mathbf c$. The `base_link` position follows as $\mathbf c - R_z(\psi)\,\mathbf t_{BC}$, with $\mathbf t_{BC} = (0.696, 0, 0.035)$ m for `cam_front`. The camera is 0.7 m ahead of `base_link`, so turning in place swings the camera through a 0.7 m arc.

### 8.2 The tag side: the resolution ball

Which camera positions give enough resolution? From eq. (5):

$$\rho \ge \rho_\text{min} \iff d \le D_\rho\cos\theta,\qquad D_\rho = \frac{f\,s}{b\,\rho_\text{min}} \tag{8}$$

**In words:** head-on, the tag can be read out to range $D_\rho$; at angle $\theta$, only out to $D_\rho\cos\theta$.

**Claim:** in the tag frame, the surface $d = D\cos\theta$ is a sphere of diameter $D$, centred at $(D/2)\,\hat{\mathbf n}$. It touches the tag plane at exactly one point, the tag centre.

**Proof.** The camera is at $\mathbf c_J$, so $d = \lVert\mathbf c_J\rVert$, and $d\cos\theta = \mathbf c_J\cdot\hat{\mathbf n}$ is its distance in front of the tag plane. Expand the sphere's equation:

$$\Big\lVert\mathbf c_J - \tfrac D2\hat{\mathbf n}\Big\rVert^2 = \Big(\tfrac D2\Big)^2 \iff d^2 - D\,\mathbf c_J\cdot\hat{\mathbf n} = 0 \iff d^2 = D\,d\cos\theta \iff d = D\cos\theta. \qquad\blacksquare$$

Points inside the sphere have $d < D\cos\theta$, i.e. more than enough resolution.

```
   tag plane
       │
       │          . ─ ─ ─ ─ .
       │      .        ╱         .
       │   .       ╱               .
       │ .     ╱  θ_max cone        .
       ●──────────────────────────────────▶ n̂
       │ .     ╲                      .
       │   .       ╲                .
       │      .        ╲         .
       │          . ─ ─ ─ ─ .
       │
       ◀─────────────── D_ρ ──────────────▶
   visible (tag side) = inside the ball AND inside the θ_max cone
```

So the familiar "visibility cone" of a tag is really a ball: the reach falls off as $\cos\theta$. At 60° off the normal, the tag reaches half as far as head-on. The steep-angle limit G3 then cuts a cone out of the ball around the tag plane (the region near the touching point, where $\theta$ is large).

| Tag | $s$ | $fs/b$ (px·m) | $D_\rho$ at $\rho_\text{min} = 1.875$ (G5) | 2 | 3 | 5 |
|---|---|---|---|---|---|---|
| 54 | 0.22 m | 32.0 | 17.1 m | 16.0 m | 10.7 m | 6.4 m |
| 146, 176, 185 | 0.12 m | 17.5 | 9.3 m | 8.7 m | 5.8 m | 3.5 m |
| 541, 558 | 0.04 m | 5.8 | 3.1 m | 2.9 m | 1.9 m | 1.2 m |

The ball's diameter is proportional to the tag size: a tag twice as large can be read from twice as far, at every viewing angle.

### 8.3 The camera side: a heading interval and an elevation cone

Now the other factor: is the tag inside the image? Take `cam_front`: its optical axis is along body $+x$ with no pitch (URDF). Work in a level frame at the camera, and describe the tag's position by:

- its horizontal range $r_h$;
- its bearing $\beta$ (the horizontal direction to it, same angle convention as the heading $\psi$);
- its height difference $\Delta h$, tag minus camera, positive when the tag is shallower (above).

Let $\phi = \beta - \psi$ be how far the tag is to the left of where the camera points. With no roll or pitch, the tag centre's camera-frame coordinates are

$$z_c = r_h\cos\phi\ \ (\text{forward}),\qquad x_c = -r_h\sin\phi\ \ (\text{right}),\qquad y_c = -\Delta h\ \ (\text{down})$$

Put these into eq. (2):

$$u - c_x = -f\tan\phi,\qquad v - c_y = -\frac{f\,\Delta h}{r_h\cos\phi}$$

The pixel must stay inside the image: $0 \le u \le W$ and $0 \le v \le H$. That gives

$$\lvert\tan(\beta - \psi)\rvert \le \tan\alpha_h, \qquad -\tan\alpha_\text{dn} \le \frac{\Delta h}{r_h\cos(\beta - \psi)} \le \tan\alpha_\text{up} \tag{9}$$

The half-angles are the angles from the optical axis to each image edge. The principal point isn't at the image centre, so they differ:

- horizontal: $\alpha_h = \arctan(c_x/f) = 34.5°$ to the left, $\arctan((W - c_x)/f) = 34.4°$ to the right;
- up: $\alpha_\text{up} = \arctan(c_y/f) = 30.3°$;
- down: $\alpha_\text{dn} = \arctan((H - c_y)/f) = 24.1°$.

There is more room above the optical axis than below, because $c_y = 679.7$ is below the image's middle row (600).

Eq. (9) is for the tag's centre. All four corners must also be at least 10 px from the edge, so shrink each half-angle by the tag's angular half-diagonal, $s/(\sqrt2\,d)$, plus $m/f \approx 0.5°$. For tag 54 that's 2.0° at 6 m and 9.4° at 1 m.

**Consequence 1: position.** Pointing the camera straight at the tag ($\psi = \beta$, so $\phi = 0$) always satisfies the horizontal condition, and makes $\cos\phi = 1$, which is the easiest case for the vertical one. So whatever the position, the best heading is "point at the tag". The only condition left on *position* is the vertical one at that best heading:

$$-r_h\tan\alpha_\text{dn} \le \Delta h \le r_h\tan\alpha_\text{up}$$

This is a double cone with its apex at the tag and a vertical axis: the tag's elevation, seen from the camera, must be within 30.3° up or 24.1° down.

**Example.** At $r_h = 6$ m, the tag may be up to 3.5 m above or 2.7 m below the camera. At $r_h = 1$ m, only 0.58 m above or 0.45 m below. **Close to the dock, depth control decides visibility.**

**Consequence 2: heading.** At a given position, the headings that keep the tag in view are

$$\lvert\beta - \psi\rvert \le \min\!\Big(\alpha_h,\ \arccos\frac{\lvert\Delta h\rvert}{r_h\tan\alpha_v}\Big)$$

where $\alpha_v$ is $\alpha_\text{up}$ or $\alpha_\text{dn}$, depending on whether the tag is above or below. The first bound is the horizontal edge. The second comes from the vertical condition: turning away from the tag shortens its forward distance $z_c = r_h\cos\phi$, which pushes it toward the top or bottom edge. At the tag's depth ($\Delta h = 0$), any heading within ±34° of the tag works. Above or below it, the interval narrows. The width of this interval is how much heading error the approach can tolerate.

### 8.4 Putting the two sides together

$V_0(\mathbf c, \psi)$ is a 4D field: 3 position dimensions and 1 heading. It reduces to a 3D volume in two different ways, depending on how the vehicle steers.

- **Heading free (sway thruster).** The vehicle can point the camera anywhere while moving in any direction. So at each position, use the best heading:

  $$V_0^\ast(\mathbf c) = \max_\psi V_0(\mathbf c, \psi)$$

  By §8.3, this is the resolution ball ∩ the $\theta_\text{max}$ cone ∩ the elevation cone. It answers: "could the vehicle see the tag from here, if it looked?"
- **Heading tied to the path (no sway).** The vehicle points where it goes, so $\psi$ is the path's direction. Visibility then depends on the whole path, not just on positions, and there is no static 3D volume. Each candidate path has to be checked along its length. This is why guidance without heading control is harder, and why the sway thruster matters here.

### 8.5 `cam_down`

The same machinery applies with the optical axis pointing down. Tags 176 and 185 face up, so their resolution balls stand vertically above them. For 12 cm tags at $\rho_\text{min} = 2$, the ball is 8.7 m tall, far above any altitude used near the dock. So resolution never limits `cam_down`.

What limits it is the **footprint**: the patch of the station the camera sees. The tag must lie within altitude × $\tan(\text{half-angle})$ of the point directly below the camera:

- at 0.3 m above the tag, about ±0.21 m across the body (the image's long side maps to body $y$) and ±0.15 m along it;
- the footprint is a rectangle that rotates with the heading.

So the two cameras are limited by different things:

- `cam_front` by resolution far away, and by the vertical field of view close in;
- `cam_down` by its field of view everywhere.

### 8.6 Several tags: union of volumes, nested sizes

With several tags, the Level-0 volume is the union of the per-tag volumes: a frame counts as "visible" if any tag is.

Tag size sets the far reach (§8.2), but tag *position* sets the near reach. A tag offset by $o$ from the camera's axis appears at angle $\arctan(o/d)$, which grows as the vehicle closes in. The tag leaves the field of view once

$$d < \frac{o}{\tan(\alpha - \delta)}$$

where $\alpha$ is the relevant half-angle and $\delta$ the margin from §8.3.

**Example.** A tag 0.5 m below the docking axis leaves the bottom of the image inside about $0.5/\tan 24° = 1.1$ m.

So the far field needs big tags (large $D_\rho$), and the near field needs small tags close to the docking axis. That is the reason for the station's nested sizes:

- 22 cm (tag 54) for far range;
- 12 cm (146) for mid range;
- 4 cm (541, 558) for the last metre.

DFKI's AUV docking station uses the same nested design, arrived at empirically (§15). Krogius et al. (2019) build the same idea into a single marker: a "recursive" AprilTag with smaller tags printed inside its centre, so that "the larger/middle tag will be detected from longer ranges but will exceed the field of view of the camera at shorter ranges, which is when the middle/smaller tag will be detected."

---

## 9. Checking Level 0 against the bags

### 9.1 Attributing every gap to a cause

Replay a bag and solve it in batch (batch LM, as in `plot_graph --batch`), so every keyframe has its best pose estimate. For every keyframe and every tag:

1. evaluate the rule in §7 at the LM pose;
2. compare with what the detector actually reported.

Then put the keyframe in one of these classes:

| Prediction | Detected | Class | Meaning and fix |
|---|---|---|---|
| visible | yes | hit | the model and the detector agree |
| visible | no | **unexplained miss** | the geometry was fine but nothing was detected: blur, glare, water, pose error, or $\rho_\text{min}$/$\theta_\text{max}$ set too loose. Many of these mean Level 0 isn't enough → Level 1 |
| not visible | yes | **unexplained hit** | thresholds set too strict, or pose error |
| not visible: G1/G2, but $V_0^\ast > 0$ | no | **looking away** | some other heading would have shown the tag. Fixable by heading guidance |
| not visible: elevation (§8.3) | no | **wrong depth** | the tag was above or below the view. Fixable by depth guidance |
| not visible: R | no | **too far** | needs a bigger tag or a closer path |
| not visible: G3 | no | **too steep** | needs a different approach angle or tag placement |
| not visible: G4 | no | **occluded** | layout problem |

Then group consecutive keyframes with the same class into time intervals. Every gap in the detections becomes a short list of intervals with causes. For example: was the 101.6 s gap "looking away" (guidance could have fixed it) or "too far" (only a bigger tag could)?

The share of "looking away" and "wrong depth" time is the part of the gaps that visibility-aware guidance could close. That number is the first evidence for or against building the planner.

### 9.2 Avoiding false labels from pose error

Near the boundary of the volume, a small pose error can flip the prediction: a tag predicted 5 px inside the image may really have been 5 px outside it. Pose error is largest exactly during the long gaps we want to explain.

So use the graph's uncertainty. The batch solve gives a marginal covariance $\Sigma_Y$ for the docking pose at each keyframe. Push it through the projection: the predicted corner pixels then have covariance $H\Sigma_Y H^{\top}$, where $H$ is the projection Jacobian (§11.1). Count a keyframe in the table only if every condition holds, or fails, by more than about 3σ of that pixel uncertainty (or the equivalent margin in $\rho$ and $\theta$). Report the excluded keyframes separately, so it's visible how much of the data was too uncertain to classify.

### 9.3 Fitting the two thresholds

- **$\rho_\text{min}$:** every hit at some $\rho$ shows the threshold is at or below that value, so the smallest $\rho$ with a hit is an upper bound. Misses well inside the field of view at small $\rho$ (with nothing else wrong) give a lower bound.
- **$\theta_\text{max}$:** the same, using $\theta$.

If hits and misses overlap over a wide band of $\rho$, the threshold isn't sharp and A5 is failing: that's the signal to move to Level 1.

### 9.4 What the 2026-10-22 run covers (200–574 s)

- 62% of keyframes have at least one tag.
- The largest off-axis angle observed is 34.85°, at the image edge, so condition G2 is active in the data.
- Viewing angles reach only 33° (tag 54) and 24° (tag 146). Ranges reach 6.28 m and 4.95 m.
- Hits go down to $\rho \approx 4.9$ (tag 54) and $\rho \approx 3.4$ (tag 146), with no misses recorded farther out, because the path didn't go there.

So $\rho_\text{min} \le 3.4$, but its value isn't located yet, and $\theta_\text{max}$ isn't identified at all. A calibration run is needed to find them (§10.6).

---

# Part II: extensions, added only when needed

## 10. Level 1: detection probability

### 10.1 From thresholds to a probability

Level 0 says a tag is either visible or not. In reality, near the boundary, the same geometry sometimes decodes and sometimes doesn't. Level 1 replaces the hard yes/no with a probability.

**The idea.** Each mechanism in §6.2 is a threshold: detection works if the image clears it by some margin. Combine the physical quantities that set the margin into one number, and add the random frame-to-frame variation:

$$\text{detected} \iff \underbrace{a_0 + \textstyle\sum_k a_k z_k}_{\text{margin } m} + \varepsilon > 0$$

- the $z_k$ are physical features ($\log\rho$, blur, and so on; §10.3);
- the $a_k$ are coefficients to fit;
- the noise $\varepsilon$ collects everything that varies from frame to frame: sensor noise, JPEG artifacts, small changes in pose and blur, patches of water.

If $\varepsilon$ follows a logistic distribution, the detection probability is

$$p = P(m + \varepsilon > 0) = \sigma(m) = \frac{1}{1 + e^{-m}}$$

(A Gaussian $\varepsilon$ gives the probit model, which is almost the same curve.) This is the standard psychometric-function model from detection experiments. Published AprilTag detection-rate curves have this S shape.

### 10.2 Image-quality effects that Level 0 ignores

**Motion blur (removes A3).**

During the exposure, the image of the tag moves across the sensor, and each cell gets smeared into its neighbours.

*How fast does the image move?* The tag is fixed. The camera moves with linear velocity $\mathbf v_C$ and angular velocity $\boldsymbol\omega_C$, both in its own frame. Seen from the camera, the tag centre then moves as

$$\dot{\mathbf p} = -\mathbf v_C - \boldsymbol\omega_C\times\mathbf p$$

Near the image centre, $u \approx f x/z$, so the image moves at about $f/d$ times the part of $\dot{\mathbf p}$ across the line of sight. That part has two pieces:

- rotation: $\lvert\boldsymbol\omega_C\times\mathbf p\rvert = \omega_\perp d$, where $\omega_\perp$ is the turn rate about axes across the line of sight;
- translation: $v_\perp$, the speed across the line of sight.

So the image moves at $f\omega_\perp + f v_\perp/d$ pixels per second. The rotation part doesn't depend on range; the translation part falls as $1/d$.

*How much does that blur the cells?* Over an exposure $t_e$, the blur length is $\beta = t_e(f\omega_\perp + f v_\perp/d)$ pixels. What matters for decoding is the blur compared with the cell size, i.e. blur **in cells**:

$$\kappa = \frac{\beta}{\rho} = \underbrace{\frac{t_e\,\omega_\perp\,b\,d}{s\cos\theta}}_{\text{rotation}} + \underbrace{\frac{t_e\,v_\perp\,b}{s\cos\theta}}_{\text{translation}} \tag{10}$$

(Substitute eq. (5) for $\rho$ and simplify.) Three consequences:

- **Rotation blur grows with range.** Turning smears the image by the same number of pixels at any range, but far away each cell covers fewer pixels.
- **Translation blur doesn't depend on range.** Small tags are sensitive to sideways speed at any distance.
- **Driving straight at the tag causes almost no blur.** The tag is then at the focus of expansion, the one point in the image that doesn't move. Turning, sway and heave are what blur it.

**Example**, with **$t_e = 10$ ms (assumed)** and a yaw rate of 0.3 rad/s:

- the image moves $\beta = 0.01 \cdot 1165 \cdot 0.3 = 3.5$ px during the exposure;
- tag 54 at 6 m ($\rho = 5.3$): $\kappa = 0.65$, two thirds of a cell;
- tag 146 at 5 m ($\rho = 3.5$): $\kappa = 1.0$, so each cell is smeared over its whole neighbour;
- by comparison, sway at 0.3 m/s with the tag 3 m away blurs only 1.2 px.

Each limit $\kappa \le \kappa_\text{max}$ has the same shape as the resolution limit in §8:

| Limit | Condition | Shape |
|---|---|---|
| rotation blur | $d \le D_\omega\cos\theta$, $\ D_\omega = \dfrac{\kappa_\text{max}s}{t_e b\,\omega_\perp}$ | ball tangent at the tag, shrinking as the turn rate grows |
| translation blur | $\cos\theta \ge \dfrac{t_e v_\perp b}{\kappa_\text{max}s}$ | cone around $\hat{\mathbf n}$ |

With **$\kappa_\text{max} = 0.5$ (assumed)**:

- turning at 0.3 rad/s shrinks tag 54's ball from 16 m to $D_\omega = 0.5 \cdot 0.22/(0.01 \cdot 8 \cdot 0.3) = 4.6$ m, and tag 146's to 2.5 m. **Turning fast while far away is what loses the tag;**
- at $v_\perp = 0.3$ m/s, the cone limits tag 146 to $\theta \le 66°$. The 4 cm tags would need $\cos\theta \ge 1.2$, which is impossible: they can't be read while moving sideways that fast.

Exposure is currently automatic (`hardware_controls.yaml`: `auto_exposure: True`). At 15 fps, $t_e \le 67$ ms. **Log the actual exposure**, since every blur number scales with it. (The manual setting `exposure_time: 156` is in V4L2 units of 100 µs, i.e. 15.6 ms.) A rolling shutter skews a moving image rather than blurring it, by an amount proportional to the turn rate times the readout time; it's ignored here.

**Water (removes A4).**

*Refraction at a flat port.* Light bends where it passes from water through the flat window into the camera housing. Underwater, this multiplies the effective focal length by about $n_w \approx 1.33$ compared with air, and narrows the field of view by the same factor. Our two calibrations show it: 953.9 px in air (`apriltag.yaml`) and 1262.7 px underwater (`tag_factorgraph_detector.yaml`), a ratio of 1.32. The graph uses the underwater calibration, so refraction is already included in every number here.

*Contrast loss.* Water absorbs and scatters light. Its beam attenuation coefficient $c$ (per metre) says how fast a beam fades. Two things happen along the path from tag to camera:

- light from the tag fades as $e^{-cL}$ over the optical path length $L$;
- scattered light from the water itself (path radiance, the haze that makes distant things look washed out) is added.

The haze adds the same amount to a black cell and a white cell, so it cancels in their *difference*, but it raises the overall brightness. For a horizontal view in uniform water with ambient light, the result is Duntley's law of contrast reduction:

$$C = C_0\,e^{-cL} \tag{11}$$

where $C$ is the apparent contrast (difference relative to background brightness) and $C_0$ the contrast close up. With ambient light, $L = d$.

The camera's auto exposure holds the average grey level roughly constant. The average is dominated by the water's haze, so the grey-level *difference* between black and white cells tracks $C$. apriltag3 needs that difference to be at least `min_white_black_diff` (default 5 grey levels). If the difference close up is $\Delta I_0$, the tag stays detectable while $\Delta I_0 e^{-cL} \ge \Delta I_\text{min}$, i.e.

$$L \le \frac1c\ln\frac{\Delta I_0}{\Delta I_\text{min}} \tag{12}$$

**This cap doesn't depend on tag size.** A bigger tag extends the resolution limit (eq. 8) but not this one. With $\Delta I_0 = 150$, $\Delta I_\text{min} = 5$ (so the log is 3.4) and typical **(assumed)** water:

| Water | $c$ (1/m) | Range cap |
|---|---|---|
| clear | 0.2 | 17 m |
| coastal | 1 | 3.4 m |
| turbid harbour | 3 | 1.1 m |

So the tank is resolution-limited, while turbid open water will be contrast-limited. Contrast also affects accuracy: a corner is located where the image gradient is steepest, and its noise is roughly image noise divided by gradient. So the corner noise $\sigma_\text{px}$ grows roughly as $e^{cL}$ too.

*With lights on the vehicle*, the light travels out and back ($L \approx 2d$). The light reaching the tag also spreads out as $1/d^2$, and backscatter close to the camera dominates the auto exposure. Contrast then falls faster than $e^{-2cd}$. Treat lights-on as a separate condition and calibrate it separately.

**Lens.** Three effects grow with the off-axis angle $\alpha$:

- vignetting: the image gets darker toward the edges, roughly as $\cos^4\alpha$, to 0.46 of the centre brightness at 34.5°;
- interpolation in rectification (§6.1);
- residual distortion the calibration didn't capture.

Level 1 folds these into one term in $\alpha$. Specular glare from laminated tags depends on where the lights, camera and tag are; it is empirical, and the practical fix is matte tags.

**Steep viewing angles.** Even at high $\rho$, decoding fails near edge-on, usually somewhere between 60° and 75°. The perspective distortion of the cell grid (its shear) grows as $\tan\theta$, and glare appears. At 1 m, tag 54 still has $\rho \approx 11$ px per cell at 70°, so $\rho$ alone would predict reliable detection there. Level 1 therefore needs its own term in $\theta$, which replaces Level 0's hard cutoff $\theta_\text{max}$ with a smooth roll-off.

### 10.3 The model

$$\boxed{\;p_\text{det} = G\cdot\sigma\!\Big(a_0 + a_1\ln\frac{\rho}{\rho_\text{ref}} - a_2\kappa - a_3\,cL - a_4\Big(\frac{\alpha}{\alpha_\text{edge}}\Big)^2 - a_5\tan^2\theta\Big),\quad a_{1..5} \ge 0\;} \tag{13}$$

$G$ is the product of the hard gates G1, G2, G4 and G5. G3 is no longer a hard gate: the $a_5$ term makes it soft.

| Feature | Formula | Depends on | Physics |
|---|---|---|---|
| $\ln(\rho/\rho_\text{ref})$ | eq. (5), $\rho_\text{ref} = 2$ | position | resolution (§6) |
| $\kappa$ | eq. (10) | position, velocity, $t_e$ | blur |
| $cL$ | eq. (11) | position, water | contrast |
| $(\alpha/\alpha_\text{edge})^2$ | eq. (4) | position and heading | lens |
| $\tan^2\theta$ | eq. (4) | position | steep-angle decoding, glare |

**Why these features.** Every feature is a dimensionless physical quantity, so the coefficients mean the same thing for any tag, camera or water. The sign constraints ($a_{1..5} \ge 0$) come from the physics: more pixels per cell helps, while more blur, more water, a more off-axis image position and a steeper angle all hurt. Constraining the signs keeps a fit on limited data from learning physically impossible trends.

- **Why $\ln\rho$ and not $\rho$:** resolution effects are multiplicative. Going from 2 to 4 px per cell matters as much as going from 4 to 8. Using the log also makes eq. (6) appear inside the model: if resolution is the limit, a fit with separate terms for $\ln s$ and $\ln d$ would give them equal and opposite coefficients. That is another form of the collapse test (§10.4).
- **Precedent for this form.** Imaging-sensor range models turn Johnson's 50% resolution count $N_{50}$ into a probability with the *target transfer probability function* (TTPF): $P = (N/N_{50})^E / (1 + (N/N_{50})^E)$. That is exactly $\sigma\big(E\ln(N/N_{50})\big)$, the resolution term of eq. (13) with $a_1 = E$ and $\rho_{50} = N_{50}$. The ACQUIRE model sets $E = 2.7 + 0.7\,N/N_{50}$, so its slope grows with $N$; eq. (13) keeps the slope constant, which is the simpler choice until the data shows otherwise. (Sources in §15. That literature also warns that a single resolution count misses contrast and blur, which is why eq. (13) has separate terms for them.)
- **Why $\tan^2\theta$:** it stays small up to about 45° (1.0) and grows fast after 60° (3.0 at 60°, 7.5 at 70°, 13.9 at 75°), matching the usual roll-off.

**Reading the coefficients:**

- **50% point:** with the other features at zero, detection succeeds half the time at $\rho_{50} = \rho_\text{ref}e^{-a_0/a_1}$. This is Level 0's $\rho_\text{min}$ made precise.
- **Sharpness:** detection goes from 10% to 90% as $\rho$ changes by a factor of $e^{4.4/a_1}$. A large $a_1$ means a sharp threshold, i.e. Level 0's hard rule is nearly right.
- **Trade-offs:** ratios of coefficients are exchange rates. For example, $a_2/a_1$ says how much blur costs in units of $\ln\rho$, i.e. how much closer the vehicle must be to cancel a given turn rate.
- **Hard limit:** as $a_1 \to \infty$ with $\rho_{50}$ fixed, eq. (13) becomes the Level-0 rule again.

### 10.4 The collapse test

This tests whether resolution is really what limits detection, which is what lets the model carry over to other tags.

Plot the detection rate against range $d$ for tags 54 and 146: two separate curves, since the bigger tag reaches further. Then plot both against $\rho$:

- **They collapse onto one curve in $\rho$:** resolution limits detection, eq. (6) holds, and the model transfers to other tag sizes.
- **They collapse in $d$ instead, at long range:** water limits detection. The size-independent cap of eq. (12) is active.
- **Neither:** something not modelled is happening, such as glare or occlusion. Look at the misses one by one.

### 10.5 N tags and M cameras

Each (camera $c$, tag $j$) pair has its own probability $p_{cj}$ from eq. (13), using that pair's $T_{BC}$, $T_{SJ}$ and tag size.

**At least one detection.** If detections are independent given the pose, the chance that every pair fails is the product of the individual failure chances, so

$$P_\text{any} = 1 - \prod_{c,j}(1 - p_{cj}) \tag{14}$$

Without assuming independence, only bounds hold (Fréchet bounds): $\max_{c,j}p_{cj} \le P_\text{any} \le \min(1, \sum_{c,j} p_{cj})$. The lower bound holds because seeing the best tag is enough; the upper bound because the chance of a union never exceeds the sum. Blur, turbidity and pose error are shared by all tags in a frame, so detections tend to fail together, and the truth is nearer the **lower** bound. Two tags help less than eq. (14) suggests when the same blur defeats both.

**Expected number of detections** per frame is $\sum_{c,j}p_{cj}$, under any dependence.

**Two kinds of uncertainty.** $p = 0.5$ can mean two different things:

- **aleatoric:** "at this geometry, half the frames decode" (true frame-to-frame randomness);
- **epistemic:** "we don't know whether this geometry is above or below the threshold" (our ignorance).

The epistemic kind makes consecutive frames at nearly the same geometry fail *together*: if the threshold is really above this $\rho$, every frame here fails. So frames must not be treated as independent coin flips when bounding gaps (§11.5).

### 10.6 Calibration

**Dataset.** Built the same way as §9:

1. Replay and batch-solve a bag; take the LM poses and predict every tag's corners in every camera.
2. Compute the features of §10.3: $\rho$ from eq. (5), $\theta$, $\alpha$, $d$ from the geometry, and $\kappa$ from the EKF twist (velocity) and the exposure.
3. Label each (keyframe, camera, tag): detected or missed.
4. Keep only samples that pass the 3σ margin filter of §9.2.
5. Fit eq. (13) by logistic regression, with the sign constraints.
6. Cross-validate in blocks of time, not random splits. Consecutive frames are strongly correlated, so a random split would put near-copies of the test frames in the training set and overstate accuracy.

**Outputs:**

- the coefficients $a$, with uncertainties;
- the collapse plot (§10.4);
- a reliability diagram: group samples by predicted probability, and plot predicted against observed detection rate. A well-calibrated model lies on the diagonal.

**Calibration run.** The existing data lacks the regions that pin down the thresholds. A dedicated tank run should cover:

1. **Range steps:** hold depth and station-keep at several ranges, up to the tank's maximum, reaching $\rho \to 2$ for both tag sizes.
2. **Slow yaw sweeps** at each range: move the tag across the image (covers $\alpha$) and out of it (the G2 boundary).
3. **Lateral offsets:** push $\theta$ to 60–75°, to find the steep-angle roll-off.
4. **Fast yaw sweeps:** cover blur, with the exposure logged.
5. **Depth offsets:** test the elevation cone (§8.3).

---

## 11. Level 2: information

Level 0 and 1 say *whether* a tag is seen. Level 2 says *what the detection tells the graph*: which parts of the docking pose it pins down, and how precisely.

### 11.1 What one detection measures

A detection gives four corner pixels, eight numbers. Model them as the prediction from eq. (2) plus independent Gaussian noise of standard deviation $\sigma_\text{px}$ per coordinate. The graph uses $\sigma_\text{px} = 1$ px (`estimator.sigma_px`).

**Fisher information** measures how much a measurement says about an unknown. For Gaussian noise it is

$$\mathcal I_{cj} = \frac{1}{\sigma_\text{px}^2}H_{cj}^{\top}H_{cj},\qquad H_{cj} = \frac{\partial(u_1, v_1, \dots, u_4, v_4)}{\partial\xi}\quad(8\times6) \tag{15}$$

where $\xi$ is a small 6-DoF change of the docking pose (3 rotation, 3 translation).

How to read it:

- each column of $H$ says how far the eight pixels move when one pose component changes. A component that moves the pixels a lot, compared with the noise $\sigma_\text{px}$, is well determined;
- two components whose columns move the pixels in the same pattern can't be told apart. They come out correlated;
- the inverse $\mathcal I^{-1}$ is the best covariance any unbiased estimator can achieve (the Cramér–Rao bound). Least squares, which is what the graph solves, reaches it to first order;
- information from independent measurements adds.

$H$ is exactly the projection factor's own Jacobian, so it can be computed with existing code. Which perturbation $\xi$ to use, and in which frame, matters once information is added over a path (§11.4).

### 11.2 One tag

Take the simplest case: a tag seen head-on, at the image centre, at range $d$. Appendix C derives its information in closed form. From one frame:

| Component | How the pixels move | $\sigma$ | In terms of $\rho$ | Tag 54 at 6 m, $\sigma_\text{px} = 1$ |
|---|---|---|---|---|
| range | the tag grows or shrinks in the image | $\dfrac{\sigma_\text{px}d^2}{\sqrt2 fs}$ | $\dfrac{\sigma_\text{px}d}{\sqrt2\,b\rho}$ | 9.9 cm |
| roll about the line of sight | the tag rotates in the image | $\dfrac{\sigma_\text{px}d}{\sqrt2 fs}$ | $\dfrac{\sigma_\text{px}}{\sqrt2\,b\rho}$ | 0.95° |
| lateral (bearing) | the tag shifts in the image | $\dfrac{\sigma_\text{px}d}{\sqrt2 f}$ | $\dfrac{\sigma_\text{px}s}{\sqrt2\,b\rho}$ | 3.6 mm |
| tilt of the tag | the square becomes a slight trapezoid (keystone) | $\dfrac{2\sigma_\text{px}d^2}{fs^2}$ | $\dfrac{2\sigma_\text{px}d}{b\rho\,s}$ | **73°** |

**Why tilt is so poorly determined.** Tilting the tag moves one edge slightly closer to the camera and the opposite edge slightly further away. The near edge then appears a little longer than the far one, but only through perspective, an effect of size $(s/d)^2$. For tag 54 at 6 m, the whole tag is 43 px wide, and 5° of tilt moves its corners by only 0.03 px, far below the 1 px corner noise. On top of that, tilt also shifts the tag sideways in the image, just like a lateral move, so the two are correlated (correlation $1/\sqrt2$, Appendix C). This is the classic orientation ambiguity of a single marker, and it's why IPPE, the pose solver, returns two candidate poses.

**What that means for where the AUV is.** Docking needs the camera's position in the tag's frame, $\mathbf c_J = -R_{CJ}^{\top}\mathbf p$: where the AUV is relative to the tag's axis. That vector has length $d$, and a tilt error rotates it, so its sideways error is about $d$ times the tilt error:

$$\sigma_{c,\text{lat}} \approx d\,\sigma_\text{tilt} = \frac{2\sigma_\text{px}d^3}{fs^2} \approx 7.7\text{ m}\quad(\text{tag 54 at 6 m})$$

**One far tag says where the tag is, not where you are relative to its axis.** (This is a linearized number: in reality the error is bounded, because the camera must be in front of the tag.) The graph recovers this position in three ways:

- it combines many frames into the single station variable $S$;
- it combines them with the attitude from the IMU and the motion from odometry, so as the vehicle moves, bearings from different viewpoints build up parallax, like triangulation;
- it adds other tags (next section).

### 11.3 Two tags at different depths: leading lights

Ships enter harbour by lining up two lights placed one behind the other. When the lights appear stacked, the ship is on the line through them; any sideways offset makes them separate. Two tags at different depths work the same way.

**Geometry.** Describe the baseline between the two tags, as seen from the camera, by:

- $\Delta_\parallel$, its length along the line of sight;
- $\Delta_\perp$, its length across the line of sight, taken separately for the horizontal and the vertical direction.

Let the camera be offset by $y$ from the line of sight to the nearer tag's centre, in the same across direction, with the nearer tag at range $d$. A small angle is (offset across)/(range), so the two bearings are about $y/d$ and $(y - \Delta_\perp)/(d + \Delta_\parallel)$. Their difference is

$$\Delta\beta \approx \frac{y\,\Delta_\parallel + d\,\Delta_\perp}{d\,(d + \Delta_\parallel)}$$

Both bearings come from the same image, so the camera's own heading cancels out of the difference: no compass or IMU is needed for this measurement.

**Precision.** Each tag's bearing has precision $\sigma_\text{px}/(2f)$: its centre is the mean of four corners, which halves the noise. The difference of two bearings has $\sqrt2$ times that. There are two error sources for $y$:

- **bearing noise:** solve $\Delta\beta$ for $y$:

  $$\sigma_{y,\beta} = \frac{\sqrt2\,\sigma_\text{px}}{2f}\cdot\frac{d\,(d + \Delta_\parallel)}{\Delta_\parallel}$$
- **range error:** $\Delta\beta$ also depends on $d$, through the $\Delta_\perp$ term. An error $\delta d$ in the range is mistaken for an offset

  $$\delta y = \frac{d}{d + \Delta_\parallel}\cdot\frac{\Delta_\perp}{\Delta_\parallel}\,\delta d$$

These are independent, so they add in quadrature:

$$\sigma_y = \sqrt{\sigma_{y,\beta}^2 + \Big(\frac{d}{d + \Delta_\parallel}\frac{\Delta_\perp}{\Delta_\parallel}\,\sigma_d\Big)^2} \tag{16}$$

**Tags 54 and 146.** From the station URDF: $\Delta_\parallel = 1.27$ m along their normals, $\Delta_\perp = 0.165$ m horizontally and $0.54$ m vertically. At $d = 6$ m with $\sigma_\text{px} = 1$:

- bearing noise: $\sigma_{y,\beta} = 2.1$ cm;
- range: $\sigma_d = 9.3$ cm, combining tag 54 (9.9 cm at 6 m) and tag 146 (26.7 cm at 7.27 m) from §11.2;
- **lateral: $\sigma_y = 2.3$ cm. Vertical: 3.9 cm.** Compare 7.7 m from tag 54 alone.

The vertical number is worse because the tags are offset mostly vertically ($\Delta_\perp = 0.54$ m), so range error leaks into it more.

Caveats:

- Both tags must be detected in the same frame. Tag 146 at 7.3 m has $\rho \approx 2.4$, near the threshold.
- $\sigma_d$ here is from a single frame. The graph's range estimate improves over many frames, which shrinks the range term.

Tags placed directly behind one another ($\Delta_\perp = 0$) would remove the range term entirely. **This is the case for placing tags at different depths along the approach, not just side by side.**

### 11.4 Information along a path

Over a path, the graph combines detections from many keyframes. Information adds, but only if every term is information about **the same fixed quantity, expressed in the same frame**.

**Why the frame matters.** A small change of the docking pose can be described in the AUV's body frame ("the station moved 1 cm to my left") or in the station's frame ("the station moved 1 cm along its own $y$ axis"). As the AUV moves and turns, "1 cm to my left" refers to different physical directions at different keyframes. Adding body-frame information from two keyframes without converting it is like adding vectors written in two different coordinate systems. It also misses parallax, which is exactly the effect of the same station being seen from different directions.

**Recommended: let the factor graph do it.** The graph already handles frames correctly. Build it from *predicted* measurements along a candidate path:

1. odometry factors from the planned motion, with the EKF noise;
2. projection factors for the tags predicted visible.

Then read the marginal covariance of the docking pose at the dock entrance. To include the detection probabilities, either:

- **Expected information:** give each projection factor noise $\sigma_\text{px}/\sqrt{p_{cj}}$. Information scales as $1/\sigma^2$, so this multiplies the factor's information by $p_{cj}$, which is its expected information (it is $\mathcal I$ with probability $p_{cj}$ and $0$ otherwise).
- **Monte Carlo:** sample which detections happen, with probabilities $p_{cj}$, solve, and repeat. The spread of the final covariances shows the risk, including the effect of gaps.

`tools/synthetic.py` already builds such a world. Before using it here, update it: it uses an 82° horizontal field of view (ours is 68.9°), it has only the 12 cm and 4 cm tags (no tag 54), and it flies a built-in weaving approach instead of an arbitrary trajectory.

**If a closed form is needed** (for example inside an optimizer that can't call gtsam), use the station-frame perturbation $Y\,\mathrm{Exp}(\xi)$. Here $\mathrm{Exp}$ turns a 6-vector into a small pose change, and multiplying it on the right means $\xi$ is expressed in the station's own frame. That's how gtsam perturbs the $S$ key. The station doesn't move, so the same $\xi$ means the same physical change at every keyframe, and detections from different keyframes add directly.

The odometry noise $Q_B$ is defined in the AUV's body frame (per second). Between keyframes it enters the docking pose on the left, $Y_{k+1} = \mathrm{Exp}(-\eta)\,\Delta X^{-1}Y_k$. Moving it to the right side uses the identity $\mathrm{Exp}(\eta)\,Z = Z\,\mathrm{Exp}(\mathrm{Ad}_{Z^{-1}}\eta)$, where the adjoint $\mathrm{Ad}$ converts a small motion from one frame to another. The information then propagates as

$$\Lambda_{k+1} = \Big(\Lambda_k^{-1} + \mathrm{Ad}_{Y_{k+1}^{-1}}\,Q_B\Delta t\,\mathrm{Ad}_{Y_{k+1}^{-1}}^{\top}\Big)^{-1} + J_{k+1},\qquad J_{k+1} = \sum_{c,j}p_{cj}\,\mathcal I_{cj} \tag{17}$$

with every $H$ taken with respect to the same right perturbation. In words: between keyframes the covariance $\Lambda^{-1}$ grows by the odometry noise (converted to the station frame), and at each keyframe the expected information from the detections is added. Working in body coordinates instead is equivalent, but then the covariance must be carried along with $\mathrm{Ad}_{\Delta X_k^{-1}}$ at every step before adding the noise.

**Scalar criteria.** $\Lambda$ is a 6×6 matrix. To rank paths, reduce it to a number. First make the units comparable: rotations and translations have different units, and the docking tolerances differ per axis. With $W = \operatorname{diag}(1/\text{tol}_i)$, the normalized covariance $\tilde P = W\Lambda^{-1}W$ is 1 exactly at tolerance. Then use one of:

- **D-optimal:** $\log\det\tilde\Lambda$, the overall volume of the information;
- **A-optimal:** $\operatorname{tr}\tilde P$, the average variance;
- **task-specific:** $\tilde P$ restricted to the lateral and vertical offset and the yaw at the dock entrance. This is what docking actually needs, and the recommended choice.

### 11.5 Gaps

**How long can the vehicle go without a detection?** Between detections, the docking pose drifts with odometry. The longest acceptable gap, $T_\text{max}$, is the time it takes the drift to reach the docking tolerance $e_\text{tol}$:

- if the drift comes from a velocity bias, the error grows as $\sigma_v T$, so $T_\text{max} = e_\text{tol}/\sigma_v$;
- if it comes from random noise, it grows as $q\sqrt T$, so $T_\text{max} = (e_\text{tol}/q)^2$.

**Example.** During a DVL dropout, the graph assumes the velocity can be off by $\sigma_v = 0.1$ m/s (`dvl_blind_speed`). With $e_\text{tol} = 5$ cm (assumed), $T_\text{max} = 0.5$ s, about two keyframes. With the DVL locked, $\sigma_v$ is much smaller: it is the drift rate shown in the "LM − EKF Δpose" panel of `compare_velocity`.

**The gap constraint.** Require that, in every time window $\mathcal W$ of length $T_\text{max}$, the probability of no detection at all is below a small $\varepsilon$. Two ways to guarantee this:

- **Robust (any dependence).** "All frames in the window fail" can only happen if frame $k$ fails, for every $k$. So its probability is at most the failure probability of the *best* frame, $1 - \max_k P_\text{any}(\mathbf x_k)$. Requiring **at least one frame per window with $P_\text{any} \ge 1 - \varepsilon$** therefore guarantees the constraint, whatever the correlation between frames.
- **Independent frames only.** Then the failure probabilities multiply, and taking logs turns the product into a sum:

  $$\sum_{k\in\mathcal W}\log\big(1 - P_\text{any}(\mathbf x_k)\big) \le \log\varepsilon$$

Use the robust version. Because of the epistemic uncertainty in §10.5, consecutive frames tend to fail together, and many marginal frames don't add up to one good one. In words: stay deep enough inside the volume, often enough.

**Pose uncertainty.** The vehicle doesn't know its configuration exactly. Its belief is a distribution, say $\mathcal N(\boldsymbol\mu, \Sigma)$, and the detection probability it should expect is the field averaged over that belief:

$$\mathbb E[p] = \int V(\mathbf x)\,\mathcal N(\mathbf x;\boldsymbol\mu, \Sigma)\,d\mathbf x$$

Approximate the integral with a few sigma points (as in the unscented transform), or use a conservative rule: require visibility over the whole $2\sigma$ ellipsoid around the estimate. Either way, near the boundary of the volume a point that is visible at the estimate may not be visible at the true pose. **The less certain the pose, the further inside the volume the vehicle must stay.**

---

## 12. Level 3: guidance and placement

All of these problems use the same model. They differ only in what is free to choose.

### 12.1 Visibility-aware guidance

**The simple rule to beat.** "Point `cam_front` at the station whenever the vehicle is inside $V_0^\ast$, and turn slowly while far away." With the sway thruster, this rule already keeps the tag in view whenever the geometry allows it. Any planner must be compared against it.

**Optimal heading on a given path.** Fix the approach path, the sequence of positions $\mathbf c_1..\mathbf c_K$ at the keyframes, and choose the heading at each keyframe:

$$\max_{\psi_{1..K}}\ \sum_k V(\mathbf c_k, \psi_k;\ \dot\psi_k) - \lambda\sum_k(\psi_{k+1} - \psi_k)^2\quad\text{s.t.}\quad\lvert\psi_{k+1} - \psi_k\rvert \le \dot\psi_\text{max}\Delta t$$

- the first sum rewards visibility, using $V_0$ or Level 1's $p$;
- the second sum penalizes heading changes ($\lambda$ sets how much), for smoothness and energy;
- the constraint is the vehicle's maximum turn rate.

Two physical couplings are included automatically:

- **Blur.** The turn rate $\dot\psi_k = (\psi_{k+1} - \psi_k)/\Delta t$ enters $V$ through rotation blur (eq. 10), so fast turns lose visibility in the model just as they do in reality.
- **Lever arm.** If the path is given for `base_link` rather than for the camera, the heading also moves the camera: $\mathbf c_k = \mathbf p_k + R_z(\psi_k)\mathbf t_{BC}$. Turning also swings the camera sideways at $0.70\,\dot\psi$ m/s, which adds to the translation blur.

**Why it can be solved exactly.** Every term depends only on two consecutive headings, $(\psi_k, \psi_{k+1})$. A sum of terms linked in a chain like this can be maximized exactly by dynamic programming:

1. discretize the heading into $N_\psi$ bins (e.g. 5°, so $N_\psi = 72$);
2. for each keyframe $k$ and each heading bin, record the best total score of any heading sequence ending there. It's computed from keyframe $k-1$'s scores plus the transition term;
3. at the end, trace back the best sequence.

This gives the **exact optimum** on the heading grid in $O(KN_\psi^2)$ time. For a 60 s approach at 5 Hz ($K = 300$) that's about 1.6 million evaluations, cheap enough for the Jetson. The sway thruster is what makes the heading free to choose independently of the path. The gap constraint (§11.5) is not a chain term, but it can be added by extending the DP state with "time since the last confident detection".

### 12.2 Information-aware guidance

Visibility-aware guidance keeps tags in view. Information-aware guidance chooses viewpoints that make the detections count.

**Scoring paths.** Score each candidate approach path by the docking-pose covariance at the dock entrance, using the predicted-measurement graph of §11.4. Start with a handful of hand-designed paths:

- straight in along the docking axis;
- offset, then align;
- an arc;
- a sideways crossing at mid range, then straight in.

Then generalize to a lattice of motion primitives.

**What the information term makes the planner do**, which visibility alone can't express:

- **cross sideways at mid range:** bearings from different directions triangulate the station, fixing its yaw and range (§11.2);
- **come in along the line through tags 54 and 146:** the leading-lights effect gives centimetre-level lateral precision from metres away (§11.3);
- **have several tags in view when committing to the dock,** rather than relying on one;
- **slow down before the last metre,** so the 4 cm tags can be read without blur (§10.2).

**Full trajectory optimization.** The most general version optimizes the whole motion:

$$\begin{aligned}
\min_{\mathbf u(\cdot),\,t_f}\quad & \int_0^{t_f}\ell(\mathbf x, \mathbf u)\,dt + w\,\Phi\big(\Lambda(t_f)\big)\\
\text{s.t.}\quad & \dot{\mathbf x} = f(\mathbf x, \mathbf u),\quad \mathbf u \in \mathcal U\\
& \text{eq. (17)},\ \text{gap constraint},\ \mathbf x(t_f) \in \mathcal X_\text{dock},\ \text{obstacles}
\end{aligned}$$

- $\mathbf u(\cdot)$ is the thruster command over time and $t_f$ the arrival time;
- $\ell$ is the running cost (energy, time);
- $\Phi$ is a criterion from §11.4 applied to the information at arrival, weighted by $w$;
- $f$ is the vehicle's dynamics (a Fossen-type physics model, or a learned modeler), and $\mathcal U$ the thruster limits, including how much sway force is available;
- $\mathcal X_\text{dock}$ is the set of acceptable dock-entry states.

Two ways to solve it:

- **Gradient-based** (direct collocation or MPC, e.g. CasADi + IPOPT). These follow gradients, but the hard gates are flat (0 or 1) almost everywhere, giving no gradient to follow. So replace each gate with a smooth step (a sigmoid) of its margin in pixels or radians, so the optimizer can tell which way leads into the volume.
- **Sampling-based** (MPPI, or search over a lattice). These evaluate many candidate trajectories and keep the best, so they work with the non-smooth field directly, and they parallelize well on the Orin's GPU.

### 12.3 Tag placement

Choose tag poses $T_{SJ}$ and sizes $s_j$ (and possibly camera mounts) from a set of candidate options. Average the result over a distribution of approach paths, because the layout must work for every approach the vehicle might fly, not just one. For a set $\mathcal S$ of chosen tags:

$$F(\mathcal S) = \mathbb E_\text{path}\Big[\log\det\Big(\Lambda_0 + \sum_{j\in\mathcal S}M_j\Big) - \log\det\Lambda_0\Big],\qquad M_j = \sum_k p_j(\mathbf x_k)\,\mathcal I_j(\mathbf x_k)$$

- $M_j$ is the expected information tag $j$ contributes along a path;
- $\Lambda_0$ is the information without any tags (from odometry and priors);
- subtracting $\log\det\Lambda_0$ makes $F$ the information *gain*, so an empty layout scores zero.

**Why greedy selection works.** $F$ has **diminishing returns** (it is submodular): adding a tag to a small layout helps more than adding the same tag to a larger layout that already covers its views. Log-det of summed information has this property, and averaging over paths keeps it. For any function that is submodular, increasing, and zero for the empty set, the greedy algorithm (repeatedly add the tag with the largest gain, up to $k$ tags) is guaranteed to reach at least $(1 - 1/e) \approx 63\%$ of the best possible gain (Nemhauser, Wolsey and Fisher, 1978). In practice greedy is usually much closer than that.

The nested sizes (§8.6) and leading-light pairs (§11.3) should come out of this optimization without being designed in. That makes it a useful check on the model.

**Placement and guidance together:** the best layout depends on the paths and vice versa. Alternate: choose tags for the current set of paths, then paths for the current tags, and repeat.

---

## 13. Evaluation plan

Guidance is a contribution only if it beats a simple baseline on a docking metric.

**Baselines**, each adding to the previous:

1. current guidance;
2. "point the camera at the station" (§12.1);
3. visibility-aware heading planning (§12.1);
4. information-aware path planning (§12.2).

**Scenarios where the simple rule should fail.** In a tank, starting from an easy position, rule 2 may already keep the tags in view nearly all the time. Then no planner can show a gain, and that is a real risk for the evaluation. So include scenarios that stress it:

- a large lateral offset, or a start outside $V_0^\ast$;
- a DVL dropout, which shrinks the allowed gap to about 0.5 s;
- turbidity, in the Stonefish simulator;
- a tag layout with weak geometry, such as a single tag.

**Metrics:**

- longest gap between detections;
- fraction of keyframes with a detection;
- docking-pose covariance and error at the dock entrance;
- docking success rate;
- time and energy.

**Ground truth:**

- docking success (a mechanical fact);
- the batch-LM trajectory as a reference. It is not independent of the estimator being tested, so it can't validate the estimator itself;
- Stonefish ground truth, in simulation.

**Scope.** Level 0 plus the heading planner is a contained addition to a factor-graph docking paper: it needs one tank experiment showing fewer gaps than rule 2. Information-aware path planning (Level 2 scoring, Stonefish trials, comparison with a visibility-only planner such as AquaVis) is large enough for a follow-up paper.

---

## 14. Where learning fits

Keep the physics as the backbone. Learn only what the physics leaves unknown, or use learning to make the physics faster.

- **Calibration.** Fit eq. (13), about six coefficients, by logistic regression. If the residuals show structure the model misses, step up to a generalized additive model (a smooth curve per feature), or a small neural network that is monotone in each feature, with the same sign constraints. Keep the inputs physical and dimensionless, so the model still transfers to other tags and cameras.
- **Surrogates for speed.** Train a network to approximate something expensive: $V^\ast$, the heading planner's output, or a cost-to-go for trajectory optimization. The physics model is the teacher and can generate unlimited training labels. This is only for speed on the Orin; the physics model remains the reference the surrogate is checked against.
- **Learned policies.** For a learned controller, such as the modeler–actor of AUV_2026_REV, add the gap constraint or the information term to the actor's training loss. With the gates smoothed (§12.2), the field is differentiable and fits into the same training loop.
- **Uncertainty in the coefficients.** A Bayesian fit of $a$ gives credible bounds on $p$. Plan against the lower bound, to be cautious, and choose the next calibration run where the bounds are widest, which pins the model down fastest (active learning).
- **What not to do.** Don't train an end-to-end network from pose to "detected":
  - it doesn't transfer across tag size, camera, downsampling or water, while the physical model transfers through $\rho$ and $c$;
  - it needs data covering 4D configurations plus velocity, of which the tank covers a fraction;
  - it can't extrapolate past the ranges the tank allows;
  - it hides the parameters a designer actually changes: tag size, focal length, decimation, exposure, placement.

---

## 15. Related work

Entries with links were checked. The others are standard references cited from memory; check them before citing in a paper. This list comes from a few targeted searches, not a systematic review.

**Closest to this work:**

- **AquaVis**, IROS 2021 ([pdf](https://afrl.cse.sc.edu/afrl/publications/public_html/papers/IROS21_1551_FI.pdf)): online visibility-aware motion planning for AUVs, keeping multiple visual objectives in view. Not docking, and no information term.
- **MViDO** ([record](https://recipp.ipp.pt/entities/publication/fa1d14be-d771-457e-8f9f-a43e07e629eb/full)): a monocular docking aid whose guidance law keeps the markers in the camera's view during docking. Visibility only.
- **PAMPC**, Falanga et al., IROS 2018 ([arXiv](https://arxiv.org/pdf/1804.04811)): perception-aware MPC for drones. It maximizes the visibility of a point of interest and minimizes its speed in the image (i.e. blur), onboard. Closest to §12.1 with blur.
- Costante et al., *Perception-aware Path Planning* (2016); Zhang and Scaramuzza, ICRA 2018: perception-aware planning.

**Belief space and information:**

- Prentice and Roy, *The Belief Roadmap*, IJRR 2009;
- Indelman, Carlone and Dellaert, IJRR 2015.

**Placement:**

- Beinhofer, Müller and Burgard, *Effective landmark placement for accurate and reliable mobile robot navigation*, RAS 2013 ([dblp](https://dblp1.uni-trier.de/rec/journals/ras/Beinhofer0B13.html)): the fewest landmarks that keep the deviation from a planned path bounded with high confidence. This is essentially §12.3.
- Robust landmark selection, IROS 2013 ([record](https://researchr.org/publication/Beinhofer0KB13)): submodular selection.
- Krause, Singh and Guestrin, JMLR 2008: submodular sensor placement.
- Nemhauser, Wolsey and Fisher, 1978: the greedy guarantee for submodular maximization.

**Underwater fiducials:**

- dos Santos Cesar et al., OCEANS 2015 ([DFKI](https://robotik.dfki-bremen.de/en/research/publications/7953)): ARToolKit, AprilTag and ArUco compared at six turbidity levels.
- arXiv 2510.27151 ([pdf](https://arxiv.org/pdf/2510.27151)): AprilTag detection becomes sporadic at 3.74 NTU at 1.9 m.
- Risholm et al., ICCVW 2021 ([link](https://mlanthology.org/iccvw/2021/risholm2021iccvw-underwater)): marker pose estimation with uncertainty in turbidity.
- Duntley, *Light in the Sea*, JOSA 1963: contrast reduction, eq. (11).

**AUV docking with markers:**

- DFKI, OCEANS 2023 ([pdf](https://www-live.dfki.de/fileadmin/user_upload/import/13345_Oceans23_TomCreutz_221220-073.pdf)): USBL for long range, a ring of six AprilTags for 5–10 m, a small central tag under 5 m. Nested sizes, arrived at empirically.
- MBARI MOLA ([pdf](https://www.mbari.org/wp-content/uploads/Hemmi_Juri.pdf)): AprilTag poses fused into an EKF.

**Single-tag pose ambiguity:**

- Schweighofer and Pinz, TPAMI 2006;
- Collins and Bartoli (IPPE), IJCV 2014;
- TagSLAM, in `docs/`.

**AprilTag** (used in §6.1, §6.3, §8.6):

- Olson, *AprilTag: A robust and flexible visual fiducial system*, ICRA 2011 ([pdf](https://april.eecs.umich.edu/pdfs/olson2011a.pdf)). §V: the 25% range argument from cell count. Figs. 9–10: simulated detection against viewing angle and against range.
- Wang and Olson, *AprilTag 2: Efficient and robust fiducial detection*, IROS 2016, doi 10.1109/IROS.2016.7759617 ([pdf](https://april.eecs.umich.edu/pdfs/wang2016iros.pdf)). Fig. 7: detection against range, with and without decimation. Fig. 6: orientation error against viewing angle.
- Krogius, Haggenmiller and Olson, *Flexible layouts for fiducial tags*, IROS 2019 ([pdf](https://april.eecs.umich.edu/media/pdfs/krogius2019iros.pdf)). Larger bit pitch gives longer range; recursive (nested) tags; recall against distance (Fig. 8).

**Resolution and detection probability** (used in §6.1, §10.3):

- Hartley and Zisserman, *Multiple View Geometry in Computer Vision*, 2nd ed., 2004, ch. 6, "cameras at infinity": affine and weak-perspective cameras and the error of using them ([chapter slides](https://www.ece.mcmaster.ca/~shirani/vision/hartley_ch6.pdf); the section number is from memory, so check it against the book).
- Johnson, *Analysis of image forming systems*, Image Intensifier Symposium, Ft. Belvoir, 1958, pp. 244–273: line pairs across a target for 50% detection (1.0), recognition (4) and identification (6.4) ([summary](https://en.wikipedia.org/wiki/Johnson%27s_criteria); original not retrieved).
- Sjaardema, Smith and Birch, *History and Evolution of the Johnson Criteria*, Sandia report SAND2015-6368, 2015 ([record](https://www.sandia.gov/research/publications/details/history-and-evolution-of-the-johnson-criteria-2015-07-01/)): how the criteria grew into probability models, and their limits.
- TTPF and the ACQUIRE model, $P = (N/N_{50})^E/(1+(N/N_{50})^E)$ with $E = 2.7 + 0.7N/N_{50}$: seen in a U.S. patent on camouflaged-target detection ([US 7149652](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/7149652)). Find the primary ACQUIRE/NVESD report before citing this in a paper.
- Vollmerhausen et al.: the TTP metric, which replaces the Johnson count because a single resolution count ignores contrast below the limiting frequency ([DTIC](https://apps.dtic.mil/sti/pdfs/ADA422493.pdf); title and venue not checked).

**Where this work fits.** Existing AUV docking systems use staged sensors and nested tag sizes with simple homing rules. Perception-aware planning with an information objective comes mostly from drones. The gap is the combination of four things:

1. a physically derived detection model ($\rho$, blur, turbidity);
2. information from the estimator's own factor graph;
3. fiducial docking with multiple tags and cameras;
4. validation on tank data.

---

## 16. Recipe

**Level 0, at one configuration $(\mathbf c, \psi)$, for each camera $c$ and tag $j$:**

1. Compute $T_{CJ}$ from eq. (1); $\mathbf p$, $d$, $\hat{\mathbf u}$ from eq. (3); $\theta$ from eq. (4).
2. Project the four corners. Check G1 (in front), G2 (10 px margin), G5 (shortest side ≥ 15 px) and G4 (occlusion).
3. Compute $\rho$ from eq. (5).
4. The tag is visible if all gates pass, $\rho \ge \rho_\text{min}$ and $\theta \le \theta_\text{max}$.
5. For several tags: the frame is "visible" if any tag is.

**Level 1:** also compute the blur $\kappa$ (eq. 10), the optical depth $cL$ (eq. 11) and $\alpha$ (eq. 4). Then $p_{cj}$ from eq. (13), and $P_\text{any}$ from eq. (14).

**Level 2:** along a path, build a graph from predicted measurements (§11.4), or use eq. (17). Read the docking covariance at the dock entrance, and check the gap constraint (§11.5).

---

## Appendix A: symbols

| Symbol | Meaning |
|---|---|
| $X$, $S$, $Y = X^{-1}S$ | AUV pose, station pose (odom), docking pose |
| $T_{BC}$, $T_{SJ}$, $T_{CJ}$ | camera on AUV, tag on station, tag in camera |
| $\mathbf p$, $d$, $\hat{\mathbf u}$ | tag centre in camera frame, range, line of sight |
| $\mathbf c$, $\mathbf c_J$ | camera centre (station frame, tag frame) |
| $\hat{\mathbf n}$ | tag outward normal |
| $\theta$, $\alpha$ | viewing angle at the tag, off-axis angle at the camera |
| $\beta$, $\psi$, $\phi$ | bearing to the tag, heading, $\beta - \psi$ |
| $\Delta h$, $r_h$ | height difference (tag minus camera), horizontal range |
| $f$, $W$, $H$, $c_x$, $c_y$ | focal length (px), image size, principal point |
| $s$, $b$ | tag black-square side, cells across it (8) |
| $q$, $m$ | `quad_decimate`, edge margin (px) |
| $\rho$, $\rho_\text{min}$, $\theta_\text{max}$ | pixels per cell; the two Level-0 thresholds |
| $V_0$, $V_0^\ast$ | Level-0 field (4D); maximized over heading (3D) |
| $\kappa$, $t_e$ | blur in cells, exposure time |
| $c$, $L$ | beam attenuation, optical path length |
| $D_\rho$, $D_\omega$ | diameters of the resolution and rotation-blur balls |
| $p_{cj}$, $P_\text{any}$ | per-pair detection probability, at least one |
| $\mathcal I$, $J$, $\Lambda$, $Q_B$ | per-detection information, expected information per frame, belief information, odometry noise rate |
| $\Delta_\parallel$, $\Delta_\perp$ | tag-to-tag baseline along and across the line of sight |
| $\sigma_\text{px}$ | corner noise (graph: 1 px) |
| $T_\text{max}$, $e_\text{tol}$ | longest acceptable gap, docking tolerance |

## Appendix B: hardware numbers

| Quantity | Value | Source |
|---|---|---|
| Image | 1600 × 1200 px, 15 fps | `stellar_cam.yaml`, intrinsics |
| Rectified $f_x$, $f_y$ | 1165.24, 1164.92 px | `tag_factorgraph_detector.yaml` (underwater) |
| Principal point $c_x$, $c_y$ | 801.9, 679.7 px | same |
| Raw fisheye $f$ | 1262.7 px/rad (underwater); 953.9 (air) | `tag_factorgraph_detector.yaml`, `apriltag.yaml` |
| Half-FOV left / right | 34.5° / 34.4° | $\arctan(c_x/f)$, $\arctan((W - c_x)/f)$ |
| Half-FOV up / down | 30.3° / 24.1° | $\arctan(c_y/f)$, $\arctan((H - c_y)/f)$ |
| `cam_front` in `base_link` | (0.696, 0, 0.035) m, optical axis along $+x$ | AUV URDF |
| `cam_down` in `base_link` | (0.395, 0, −0.128) m, optical axis down, image width along body $y$ | AUV URDF |
| Tags | 54: 0.22 m; 146, 176, 185: 0.12 m; 541, 558: 0.04 m | `station_tags.yaml` |
| Tags 54 ↔ 146 | 1.27 m along normal; 0.165 m horizontal, 0.54 m vertical across | station URDF |
| Detector | `quad_decimate` 2, `min_edge_px` 10, `min_side_px` 15, `max_fit_rms_px` 1.5, 5 Hz | `tag_factorgraph_detector.yaml` |
| Graph corner noise | `sigma_px` 1.0 | `race_auv_factorgraph.yaml` |
| Exposure | auto; ≤ 67 ms at 15 fps | `hardware_controls.yaml` |

## Appendix C: Fisher information of one tag

**Setup.** The tag is seen head-on, at the image centre, at range $d$, with half-side $h = s/2$. Its corners in the camera frame are $(x_i, y_i, d)$ with $x_i, y_i = \pm h$. Each projects to $f\,(x/z,\ y/z)$ (eq. 2, with the principal point dropped).

**How the pixels move.** For each small change of the pose, the first-order change of corner $i$'s pixel:

| Change | $\Delta u_i$ | $\Delta v_i$ |
|---|---|---|
| lateral $\delta t$ (along $x$) | $f\,\delta t/d$ | $0$ |
| range $\delta d$ | $-fx_i\,\delta d/d^2$ | $-fy_i\,\delta d/d^2$ |
| roll $\delta\gamma$ about the line of sight | $-fy_i\,\delta\gamma/d$ | $fx_i\,\delta\gamma/d$ |
| tilt $\delta\tau$ about the tag's $y$ axis | $fh^2\,\delta\tau/d^2$ | $fx_iy_i\,\delta\tau/d^2$ |

(Lateral along $y$ and tilt about $x$ mirror the last two rows.)

Where the rows come from:

- **lateral:** every corner's $x$ grows by $\delta t$, so every $u = fx/d$ grows by $f\,\delta t/d$;
- **range:** $u = fx_i/d$, so $\partial u/\partial d = -fx_i/d^2$: the tag shrinks toward the image centre;
- **roll:** rotating by $\delta\gamma$ about the optical axis moves $(x_i, y_i)$ to $(x_i - y_i\delta\gamma,\ y_i + x_i\delta\gamma)$;
- **tilt:** rotating by $\delta\tau$ about the tag's $y$ axis moves each corner's depth to $d - x_i\delta\tau$. Then $u_i = fx_i/(d - x_i\delta\tau) \approx fx_i/d + fx_i^2\,\delta\tau/d^2$, and $x_i^2 = h^2$ at every corner. Likewise $v_i \approx fy_i/d + fx_iy_i\,\delta\tau/d^2$.

**Information.** Each entry of $\mathcal I = H^{\top}H/\sigma^2$ is the sum over the four corners of the products of two columns, divided by $\sigma^2 = \sigma_\text{px}^2$. Over the four corners, $\sum x_i = \sum y_i = \sum x_iy_i = \sum x_i^2y_i = 0$. So range and roll are orthogonal to everything else (their pixel patterns can't be confused with any other), and

$$\mathcal I_{dd} = \frac{8f^2h^2}{d^4\sigma^2},\qquad \mathcal I_{\gamma\gamma} = \frac{8f^2h^2}{d^2\sigma^2}$$

Lateral shift and tilt both move every $u$ by the same amount, so they overlap:

$$\mathcal I_{tt} = \frac{4f^2}{d^2\sigma^2},\quad \mathcal I_{\tau\tau} = \frac{8f^2h^4}{d^4\sigma^2},\quad \mathcal I_{t\tau} = \frac{4f^2h^2}{d^3\sigma^2}$$

Inverting this 2×2 block gives their covariance:

$$\Sigma_{tt} = \frac{d^2\sigma^2}{2f^2},\quad \Sigma_{\tau\tau} = \frac{d^4\sigma^2}{4f^2h^4},\quad \Sigma_{t\tau} = -\frac{d^3\sigma^2}{4f^2h^2},\quad \text{corr} = -\frac{1}{\sqrt2}$$

Substituting $h = s/2$ and taking square roots gives the table in §11.2. Without the overlap, the lateral error would be $\sigma d/(2f)$; the overlap with tilt makes it $\sqrt2$ times larger.

**Camera position in the tag frame.** To first order, the camera's lateral position in the tag frame is $-t + d\,\tau$: the tag's lateral offset, plus the sideways swing of a vector of length $d$ rotated by the tilt. Its variance is

$$\Sigma_{tt} + d^2\Sigma_{\tau\tau} - 2d\,\Sigma_{t\tau} = \Sigma_{tt} + d^2\Sigma_{\tau\tau} + 2d\,\lvert\Sigma_{t\tau}\rvert$$

The middle term is larger than the first by a factor of about $(d/h)^4$ and than the last by about $(d/h)^2$, so it dominates: $\sigma \approx d\,\sigma_\tau = 2\sigma_\text{px}d^3/(fs^2)$ (§11.2).
