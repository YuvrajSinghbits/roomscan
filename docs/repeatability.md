# Repeatability (same tier, two captures of the same property)

The sample captures `floor_only` (1a8384c3f6) and `with_ceiling` (c7d28f72c6)
are two LiDAR scans of the **same flat**: same rooms, same layout and orientation
(compare `plans/lidar_floor_only.png` and `plans/lidar_with_ceiling.png`). So the
repeatability gate can be measured for real. Regenerate with:

    python scripts/repeatability.py out/lidar_floor_only/result.json out/lidar_with_ceiling/result.json

Only walls measured in **both** captures (90% half-width ≤ 5 cm, i.e. backed by
scanned surface) are compared. Gate: two captures agree within max(1 cm, 0.5%)
per wall.

| room A | room B | wall A m | wall B m | difference cm | within ±max(1 cm, 0.5%) |
|---|---|---|---|---|---|
| room_1 | room_5 | 2.829 | 3.076 | +24.6 | NO |
| room_2 | room_4 | 2.846 | 2.869 | +2.3 | NO |
| room_2 | room_4 | 4.195 | 4.358 | +16.3 | NO |
| room_3 | room_6 | 2.527 | 2.522 | -0.5 | yes |
| room_4 | room_7 | 0.521 | 0.487 | -3.4 | NO |
| room_4 | room_7 | 1.165 | 1.172 | +0.8 | yes |
| room_4 | room_7 | 2.778 | 2.801 | +2.2 | NO |
| room_5 | room_2 | 2.334 | 2.280 | -5.4 | NO |
| room_5 | room_2 | 1.711 | 1.724 | +1.3 | NO |
| room_5 | room_2 | 1.651 | 1.587 | -6.4 | NO |
| room_5 | room_2 | 0.419 | 0.425 | +0.6 | yes |
| room_5 | room_2 | 0.684 | 0.694 | +1.0 | NO |
| room_5 | room_2 | 1.291 | 1.299 | +0.8 | yes |

4/13 walls within the gate; median |difference| 2.2 cm, max 24.6 cm.

## Verdict: unrepeatable at the 1 cm level (not "repeatable but biased")

The signs of the differences are mixed (+24.6 … −6.4 cm), so there is no common
offset to calibrate out. This is unrepeatability, not bias.

* **Large differences (5–25 cm, 5 walls):** the two scans chose different
  surfaces as "the wall". Wall fitting keeps the best-supported surface above
  0.9 m, and a wardrobe or shelving front can be that surface in one scan while
  the wall behind it wins in the other. The 16 cm on the 4.2 m wall and the
  24.6 cm on room_1/room_5 have this signature. Fix direction: prefer the
  outermost supported surface when two parallel candidates are both supported,
  and report furniture fronts as furniture.
* **Small differences (1–2.3 cm, 5 walls):** above the 1 cm gate but inside the
  sum of the two captures' intervals. Residual drift differs between a 54 m and
  a 99 m walk; the loop closure removes the end offset but spreads it linearly.
* **4 of 13 walls pass**, all short or bathroom walls fully seen in both scans.

Ceiling repeatability cannot be measured: `floor_only` never scanned the ceiling.
