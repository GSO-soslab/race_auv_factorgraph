# References

Local copies of the papers cited in [`../TAG_VISIBILITY_FIELD.md`](../TAG_VISIBILITY_FIELD.md) (§ numbers below refer to that doc), plus the underwater-visibility papers found while planning its validation. `../TagSLAM_Robust SLAM with Fiducial Markers.pdf` (single-tag ambiguity, §15) is one level up.

**Copyright.** The two files marked *IEEE (licensed copy)* were downloaded through a university IEEE Xplore subscription. Keep them out of any public repository. All other files are open-access copies from the authors' or publishers' sites.

## Resolution and AprilTag detection (§6.1, §6.3, §8.6)

| File | Citation | Used for |
|---|---|---|
| `olson2011_icra_apriltag.pdf` | E. Olson, *AprilTag: A robust and flexible visual fiducial system*, ICRA 2011 | §V: range scales with cell size ("only a 25% improvement … 16 bit tags instead of 36 bit"); Figs. 9–10: detection against viewing angle and against range (simulated) |
| `wang2016_iros_apriltag2.pdf` | J. Wang, E. Olson, *AprilTag 2: Efficient and robust fiducial detection*, IROS 2016, doi 10.1109/IROS.2016.7759617 | Fig. 7: detection against range, with decimation; Fig. 6: orientation error against off-axis angle (= our θ) |
| `krogius2019_iros_flexible_layouts.pdf` | M. Krogius, A. Haggenmiller, E. Olson, *Flexible layouts for fiducial tags*, IROS 2019 | Larger bit pitch gives longer range; recursive (nested) tags; Fig. 8: recall against distance |
| `hartley_zisserman_ch6_camera_models_slides.pdf` | Lecture slides of Hartley & Zisserman, *Multiple View Geometry*, 2nd ed., ch. 6 (not the book) | Affine and weak-perspective cameras; the error of the approximation, $\frac{\Delta}{d_0}(x_\text{proj}-x_0)$ |
| `us7149652_ttpf_patent.pdf` | U.S. Patent 7,149,652, method for modeling detection of camouflaged targets (scanned, no text layer) | The TTPF / ACQUIRE form $P = (N/N_{50})^E/(1+(N/N_{50})^E)$, $E = 2.7 + 0.7N/N_{50}$ (§10.3) |

## Underwater fiducial visibility (§10.2 water, §10.4 collapse test, validation)

| File | Citation | Used for |
|---|---|---|
| `cesar2015_oceans_underwater_fiducials.pdf` | D. B. dos Santos Cesar, C. Gaudig, M. Fritsche, M. A. dos Reis, F. Kirchner, *An evaluation of artificial fiducial markers in underwater environments*, OCEANS 2015 Genova, doi 10.1109/OCEANS-Genova.2015.7271491. *IEEE (licensed copy).* | Gantry tank, 6 turbidity levels (0.3–7 FTU), 2 lighting setups, 36h11 tags of 18.3, 5.5 and 2.0 cm. Smallest detected size (Figs. 5–6), largest range (Figs. 7–8), largest angle (Figs. 9–10) |
| `cheng2025_journal_confined_space_positioning.pdf` | X. Cheng, K. Yao, A. West, S. Watson, O. Marjanovic, B. Lennox, K. Groves, *Confined space underwater positioning using collaborative robots: a proposed system*, journal version (accepted, pagination pending). *IEEE (licensed copy).* | Turbidity summary only; refers to the arXiv version for the figure |
| `cheng2025_arxiv_confined_space_positioning.pdf` | Same authors, arXiv 2510.27151 | Fig. 9: AprilTag decision margin against turbidity (0.12, 2.58, 3.74 NTU) and distance (0.9, 1.4, 1.9 m); detection sporadic at 3.74 NTU and 1.9 m. Tag size and intrinsics not given, so ρ can't be computed |
| `risholm2021_iccvw_underwater_marker_pose.pdf` | P. Risholm et al., *Underwater marker-based pose-estimation with associated uncertainty*, ICCV Workshops 2021 | Marker pose uncertainty in turbidity |
| `cejka2019_remotesensing_underwater_markers.pdf` | J. Čejka, F. Bruno, D. Skarlatos, F. Liarokapis, *Detecting square markers in underwater environments*, Remote Sensing 11(4):459, 2019 | Underwater marker detector; method for generating synthetic underwater marker images |
| `westman2018_underwater_apriltag_slam.pdf` | E. Westman, M. Kaess, *Underwater AprilTag SLAM and calibration for high precision robot localization*, CMU tech. report 2018 | AprilTags in a tank at various depths and angles; no systematic range or angle sweep |

## AUV docking with markers (§8.6, §15)

| File | Citation | Used for |
|---|---|---|
| `creutz2023_oceans_dfki_docking.pdf` | T. Creutz et al. (DFKI), *Towards robust autonomous underwater docking for long-term under-ice exploration*, OCEANS 2023 | USBL far, AprilTag ring at 5–10 m, central small tag under 5 m: nested sizes found empirically |
| `hemmi_mbari_mola_apriltag.pdf` | J. Hemmi (MBARI intern report), *Docking MOLA: AprilTag detection, data fusion, and control of a 6-DOF AUV* | AprilTag poses fused into an EKF |
| `knutson2023_arxiv_adaptive_landmark_color.pdf` | Knutson, Cao, Sattar, *Adaptive landmark color for AUV docking in visually dynamic environments*, arXiv 2310.02944 | Active LED markers, about 5 m in clear water |
| `norman2023_iros_actag.pdf` | K. Norman, D. Butterfield, J. G. Mangelson, *AcTag: Opti-acoustic fiducial markers for underwater localization and mapping*, IROS 2023 | Acoustic and optical markers built because optical tags fail in turbidity (background) |
| `wang_acmarker_acoustic_marker.pdf` | Y. Wang et al., *ACMarker: Acoustic camera-based fiducial marker system in underwater environment* | Acoustic-camera markers (background) |

## Planning and placement (§12, §15)

| File | Citation | Used for |
|---|---|---|
| `falanga2018_iros_pampc.pdf` | D. Falanga et al., *PAMPC: Perception-aware model predictive control for quadrotors*, IROS 2018 (arXiv 1804.04811) | Perception-aware MPC: visibility plus image-plane speed (blur) |
| `krause2008_jmlr_sensor_placement.pdf` | A. Krause, A. Singh, C. Guestrin, *Near-optimal sensor placements in Gaussian processes*, JMLR 9, 2008 | Submodular sensor placement (§12.3) |

## Cited but not saved

| Reference | Why not saved |
|---|---|
| AquaVis, IROS 2021 (`afrl.cse.sc.edu/.../IROS21_1551_FI.pdf`) | The server's TLS certificate failed verification; download it in a browser |
| Vollmerhausen et al., TTP metric (DTIC ADA422493) | DTIC blocked the scripted download |
| Johnson 1958, *Analysis of image forming systems* | Not available online |
| Sjaardema, Smith, Birch, SAND2015-6368 | Record page only; PDF not located |
| Hartley & Zisserman (book), Duntley 1963, Nemhauser et al. 1978, Schweighofer & Pinz 2006, Collins & Bartoli 2014, Prentice & Roy 2009, Indelman et al. 2015, Costante et al. 2016, Zhang & Scaramuzza 2018, Beinhofer et al. 2013 (RAS and IROS), MViDO | Paywalled or no open PDF located; get them through the library |
