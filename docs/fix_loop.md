# Fix loop declaration

## 1. Worst-performing gate (with the failing number)

Of the gates measurable on our benchmark (the three sample captures; no tape
truth), the worst is **damage detection on clean rooms**. The sample flat is a
clean, recently finished apartment with no visible damage (checked frame by
frame), yet the first shipped detector reported **19 damage regions on
`single_room`**. That includes physically impossible ones, such as "peeling
paint" and "mould" on a tiled floor and a 2.6 m² "water stain" on a plain wall.
Each one is a phantom. Under the gates' scoring rule, a phantom counts as a
miss, and each one also generates false concealed-damage flags (15) and scope
items (41), so the error propagates into the customer-facing output.

## 2. Root-cause hypothesis and evidence

**Hypothesis:** the zero-shot detector was accepted per *single view*, per
*pixel*, with a threshold below CLIPSeg's background activation on smooth
light surfaces, and with no physical prior on which defect can appear on which
surface.

Evidence:
* The phantom regions' mean confidence is 0.47–0.84, and 11 of 19 sit below 0.6,
  which is where CLIPSeg's sigmoid output lands on plain painted walls and tiles
  for any "wall defect" prompt.
* 10 of the 19 regions were seen in **one view only**. A real stain is a property
  of the surface, so it reappears when the camera passes again; a single-view
  activation is a lighting or texture artefact of that frame.
* 4 regions are class/surface combinations that cannot exist (peeling paint,
  mould or cracks on a ceramic floor).

## 3. The fix, and the predicted number

Fix (`src/roomscan/damage.py`, profile `after`):
1. Raise the acceptance threshold from 0.45 to 0.60.
2. Require the same class on the same surface in **≥ 2 views**.
3. Allow only physically valid class/surface pairs (floor: water stain only).

**Prediction, written before the after run:** on the three clean sample
captures, damage regions drop from double digits to **≤ 3 per capture**, with
**0 impossible class/surface pairs**. Concealed-damage flags fall
proportionally. Risk: the multi-view rule lowers recall for a defect seen in a
single frame. The protocol's damage photo (1.5 m away) and a walk that passes
the defect twice mitigate this.

## 4. Before / after (regenerable)

Both runs come from the same code:

```bash
ROOMSCAN_DAMAGE_PROFILE=before roomscan run "<capture>" --out out/fixloop_before_<name>
ROOMSCAN_DAMAGE_PROFILE=after  roomscan run "<capture>" --out out/fixloop_after_<name>
```

(`scripts/run_all.sh` runs both for all three captures.) The readable diff is the
`PROFILES` table in `damage.py` plus the commit that introduced it. The measured
results are in [benchmark_results.md](benchmark_results.md) under "Fix loop" and
summarised below.

## 5. Result

| capture | before: regions (impossible pairs) | after: regions | after: flags / scope items |
|---|---|---|---|
| single_room | 10 (2) | 2 | 2 / 4 |
| floor_only | 9 (2) | 0 | 0 / 0 |
| with_ceiling | 14 (3) | 0 | 0 / 0 |

**The prediction held:** at most 3 regions per capture (measured 2, 0, 0), with 0
impossible class/surface pairs. Phantom regions on the three clean captures fell
from 33 to 2, a 94% drop.

The "before" count for `single_room` is 10 here versus 19 in §1. The declaration
was written from the first shipped run, which sampled 40 views; both before and
after runs in the table use the current 20 views, so the comparison is like for
like.

What it did not prove: recall. With no staged-damage room there is no real defect
to find, so the multi-view rule's cost on true positives is unmeasured. The two
remaining `single_room` regions were not inspected individually; they may be real
marks or residual phantoms.
