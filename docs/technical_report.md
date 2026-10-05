# roomscan — technical report

## 1. Architecture

```
capture ──► tier front end ──► gravity-aligned points + camera-facing normals + camera path
                                         │
                       drift.py  (Manhattan heading anchor + loop closure)
                                         │
                       plan2d.py (tier-agnostic geometry)
                         floor/ceiling levels ─ ray-carved free space ─ rooms
                         ─ rectilinear outlines ─ evidence-backed wall fits
                         ─ doors from walk crossings ─ windows from wall-face holes
                                         │
                       damage.py (CLIPSeg → surfaces)  scope.py (rules R1–R5, line items)
                                         │
                       output.py → result.json (schema-validated) + plan.png
```

The key design choice is that **all three tiers reduce to the same intermediate**:
points with normals that face the camera, in a gravity-aligned world, plus the
camera path. Everything after that is shared, so one improvement to wall fitting
or door detection lifts every tier, and the tiers differ only in how much they
can be trusted, which the error model expresses.

Camera-facing normals (from depth-image gradients, flipped towards the camera)
carry two facts used everywhere: whether a surface is floor, ceiling or wall,
and **which side of a wall it was seen from**. The second is what lets a room's
wall be fitted to its own inner face and not the neighbouring room's face 10 cm
away.

## 2. Tier design

**LiDAR** (Stray Scanner, or 3D Scanner App). Stray poses are camera-to-world in
OpenCV camera axes. This was verified on the samples: only that convention puts
the floor at the same height from every frame. Depth uses ARKit confidence = 2
only. Frames are strided to about 600 per capture.

**Video.** No depth, poses or IMU in the file. Metric depth per frame comes from
Depth Anything V2 (metric indoor, Small). Poses come from frame-to-frame ORB
matches lifted to 3D with the previous frame's depth, then PnP-RANSAC. A frame
that fails to track holds the previous pose, so it costs one frame instead of
the rest of the walk. Gravity is the dominant floor/ceiling normal, seeded by
image-up (the protocol holds the phone upright). From there it is the LiDAR
pipeline unchanged.

**Photo.** It is built around the protocol. In a *corner photo* (heels in a
corner, aimed at the opposite corner), the two far walls' distances plus the
camera-to-corner offset (0.30 ± 0.08 m) give the room's dimensions. Every corner
photo is an independent estimate, and their spread is part of the interval. A
*door photo* is a facing wall with a floor-to-1.8 m hole and points behind it;
this gives the door width and head height. For stitching, the view through
each door is SIFT-matched against every other room's photos to name the
neighbour. Rooms are then placed against their neighbours with no overlaps. A
door's position along its wall cannot be observed from one photo, so its offset
interval spans the wall.

## 3. Geometry (plan2d.py)

1. **Levels.** The floor is the lowest strong up-facing level. Each room's
   ceiling is its own strongest down-facing level, because lowered and false
   ceilings exist (the sample flat has 2.27–3.08 m). If no ceiling was scanned,
   the height is reported as a **lower bound** with a one-sided interval.
2. **Free space by 2D ray carving.** Every camera→point ray is empty in plan
   projection. This works whether or not floor or ceiling were scanned; the
   first real capture showed that floor-hit occupancy fails when the phone is
   held level.
3. **Rooms.** Free space is cut at thin door-head strips: down-facing surfaces
   below the ceiling, with broad areas (lowered ceilings, lofts) removed by a
   morphological opening. It is also cut at passages narrower than about 0.95 m.
   Seeds then grow back over free space. A seed the camera never entered (space
   seen through an opening) grows as its own region and is then dropped.
4. **Outlines.** A rectilinear trace with jogs under 16 cm removed. Each side is
   then fitted to the room-facing wall points above furniture height (0.9 m).
   A side with too little wall surface behind it is pruned into its better
   supported neighbour; any side left gets a ±25 cm interval. A merge may not
   collapse the outline.
5. **Openings.** Doors are found where the camera path crosses a wall line (the
   protocol walks through every doorway) and are measured as the hole in that
   wall's surface points. Edges are refined on jamb points; the head height
   comes from soffit points. Windows are holes with a sill below and a head
   above. Gaps outside 0.45–2.6 m are rejected as phantoms.

## 4. Drift handling (and the ablation)

ARKit is gravity-aligned, so heading and position drift.

* **Manhattan heading anchor.** Each 15-frame chunk's dominant wall direction
  (circular mean of 4× the normal angle) is compared with the first chunk's. The
  difference is the heading drift. It is removed incrementally along the path,
  re-rotating every step so that the position error it caused goes too.
* **Loop closure.** The protocol ends where it starts. The wall-point images of
  the first and last chunks are cross-correlated (FFT, ±0.6 m, sub-cell peak),
  the floor gives the vertical offset, and the offset is spread along the path
  by distance walked. The ramp is anchored at the chunk centres; anchoring at the
  first and last frames under-corrected by 15% in testing.
* **Ablation** (`--no-drift-correction`). On the synthetic drifted export, the
  stitched footprint error is 0.04 m² with correction and 3.45 m² (−7.7%)
  without. On the largest sample capture (`with_ceiling`, 98.9 m walk, loop
  closed with a 0.21 m offset), switching correction off loses 4.3 m² (−7.5%)
  of the stitched footprint (58.02 → 53.68 m²); see benchmark_results.md.
  "Poses used as-is" is only ever the ablation.

## 5. Error budget and intervals

Every number is a 90% interval (1.645σ):

| Term | σ | Source |
|---|---|---|
| wall plane, statistical | std/√n_eff of inlier points (n_eff ≤ 150) | per wall |
| wall plane, systematic | 5 mm | LiDAR range bias + residual drift |
| unsupported wall side | ±25 cm (90%) | no surface observed |
| level (floor/ceiling) | 4 mm systematic + statistical | per room |
| opening edge | 5 mm + jamb statistics, or point spacing | per edge |
| video error model | 25% + 2 cm (90%) | measured: 37% per-frame depth spread; first cross-tier run −24% outside a 3% interval |
| photo error model | 40% + 2 cm (90%) + corner spread + offset prior | measured per-photo scale spread; off-protocol proxy errors −56…−74% |

Lengths combine their two bounding wall planes in quadrature. Area adds Σ(Lₖ·hwₖ)²
from wall shifts.

## 6. Calibration analysis

The monocular depth model was measured against LiDAR on 72 frames from the three
sample captures (`scripts/calibrate_mono.py`):

* median model/LiDAR = **1.159**. This is a systematic over-estimate, now
  divided out (`depth_model.SCALE_CAL`). The Base model has less bias (1.054)
  but the same spread at 4× the cost, so Small is used.
* per-frame scale spread **37%**; within-frame error after removing per-frame
  scale **11%**.

The video and photo intervals therefore cannot honestly be LiDAR-like.
Cross-tier coverage (is the LiDAR footprint inside the video/photo interval?) is
tabulated in benchmark_results.md. Where coverage fails, the interval is too
narrow, and that is reported as a miss, not tuned away silently.

## 7. Known failure modes

* **Monocular scale** (video, photo): about 10% class errors; video intervals
  under-cover on the samples.
* **Photo tier off-protocol**: the proxy photos are walkthrough frames, not
  corner photos, so the corner model does not apply and dimensions are
  underestimated. The protocol's stance instruction exists for this reason.
* **Photo tier unvalidated on protocol input**: the sample footage has no
  corner-stance frames. Selecting the frames nearest each corner and facing the
  room (`stray_to_photos.py ... corners`) yields 0–1 per room, against the
  protocol's 4, and errors stay at −51…+75%. A room with no two measurable walls
  gets a labelled placeholder size, never a silent number.
* **Rooms only partly scanned** become rectangles bounded by the seen walls,
  with ±25 cm on unseen sides.
* **Doors the walk does not cross** are not found (LiDAR and video); real-data
  adjacency is incomplete.
* **Mirrors and glass**: LiDAR returns through glass or reflections. Low-confidence
  depth is dropped and wall fits use medians; a full-height mirror can still read
  as an opening. The protocol says to sweep past mirrors.
* **Wet-look surfaces / low light**: lower ARKit confidence means fewer points
  and wider intervals. Low light also weakens ORB tracking in the video tier.
* **Damage** is zero-shot CLIPSeg. On clean rooms its precision is the limiting
  factor (see fix loop). There is no staged-damage benchmark, so recall is
  unmeasured.
* **Non-Manhattan walls** (angled bay walls) are snapped to the dominant axes.

## 8. Repeatability

The two multi-room sample captures are the same flat, so the repeatability gate
was measured for real ([repeatability.md](repeatability.md)): 4 of 13 walls
measured in both scans agree within max(1 cm, 0.5%), with a median difference of
2.2 cm. The differences have mixed signs, so this is **unrepeatable, not biased**.
The large ones come from the two scans fitting different surfaces (a wardrobe
front against the wall behind it).

## 9. Fix loop

See [fix_loop.md](fix_loop.md): the worst measurable gate (damage phantoms), the
root cause, the fix, and the predicted and measured before/after, both
regenerable from this code.

## 10. What is missing and why

There is no tape or laser ground truth, no staged-damage room and no
consumer-app head-to-head. All of these need iPhone captures of
rooms we can physically access. The pipeline accepts them as-is:
`docs/ground_truth_template.csv` plus `scripts/run_all.sh`.
