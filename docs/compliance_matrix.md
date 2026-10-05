# Compliance matrix

Status: **Done** = implemented and exercised on data; **Partial** = implemented,
known gaps stated; **Not met** = missing, with the reason.

| # | Requirement (case study) | File path | Artifact | Status |
|---|---|---|---|---|
| 1 | Capture route: stock-capture one-page protocol | docs/capture_protocol.md | protocol | Done |
| 2 | Device matrix: tier → hardware → honest accuracy | docs/device_matrix.md | table | Done |
| 3 | Photo tier: per-room folders → stitched whole-property plan | src/roomscan/tiers/photo.py | `out/photo_*` | Partial — runs end to end; rooms are rectangles; door position along wall not observed; adjacency needs door photos (proxy data has none) |
| 4 | Video tier: handheld walkthrough → same contract | src/roomscan/tiers/video.py | `out/video_*` | Partial — runs end to end on real footage; monocular scale limits accuracy (see benchmark) |
| 5 | LiDAR tier: depth + poses + intrinsics | src/roomscan/tiers/lidar.py, lidar_io.py | `out/lidar_*` | Done |
| 6 | Same output contract from every tier, intervals widen as data thins | schema/capture_output.schema.json, output.py (`widen`) | `result.json` per run | Done |
| 7 | Per-room plan: walls, ceiling height, floor area, openings | plan2d.py, output.py | `rooms[]` | Done |
| 8 | Stitched multi-room plan with correct adjacency | plan2d.py (`find_doors`), photo.py (`layout`) | `stitched_plan` | Partial — adjacency on real LiDAR data misses doors the walk did not cross cleanly |
| 9 | Damage regions with class and metric extent per surface | damage.py | `damage[]` | Partial — zero-shot CLIPSeg; no staged-damage benchmark to score it |
| 10 | Concealed-damage flags with the rule that fired | scope.py (R1–R5) | `concealed_damage_flags[]` | Done |
| 11 | Scope line items keyed to surfaces | scope.py | `scope[]` | Done |
| 12 | Confidence interval on every measurement | output.py, plan2d.py error budget | every `{value, lo, hi}` | Done (test checks lo ≤ value ≤ hi everywhere) |
| 13 | One command per capture | cli.py | `roomscan run <capture>` | Done |
| 14 | JSON to the published schema | schema.py (`validate`) | validated on every run | Done |
| 15 | Rendered plan | output.py (`render`) | `plan.png` | Done |
| 16 | Drift accountability + on/off ablation | drift.py, `--no-drift-correction` | benchmark_results.md "Drift ablation" | Done |
| 17 | Gate: openings ≤ 2 cm on ≥ 85% | plan2d.py | synthetic test | Partial — met on synthetic (≤ 1 cm); unscored on real data (no ground truth) |
| 18 | Gate: ceiling ≤ 1.5 cm, repeat spread ≤ 1 cm | plan2d.py (`room_level`) | synthetic test | Partial — met on synthetic; no repeat capture available |
| 19 | Gate: repeatability (two captures, ≤ 1 cm / 0.5%) | scripts/repeatability.py | docs/repeatability.md | Measured, **fails**: 4/13 walls within gate, median 2.2 cm; diagnosed as unrepeatable (surface choice), not biased |
| 20 | Gate: photo-tier stitch, footprint ±8%, calibrated | photo.py | benchmark_results.md | Partial — runs; accuracy measured only against LiDAR on proxy photos |
| 21 | Benchmark set built by us (multi-room, damage room, all tiers, repeat, tape truth) | docs/ground_truth_template.csv, scripts/run_all.sh | benchmark_results.md | Partial — uses the provided sample captures; video/photo derived from the same footage; no tape truth, no staged damage, no repeat |
| 22 | Head-to-head vs a consumer app on 2 rooms | — | — | Not met — needs an iPhone capture with the consumer app on rooms we can access |
| 23 | Fix loop: worst gate, root cause, fix, before/after regenerable | docs/fix_loop.md, damage.py `PROFILES` | `out/fixloop_*`, benchmark_results.md | Done (on the gate we can measure: damage phantoms) |
| 24 | README to a fresh capture in < 15 min on a clean machine | README.md | — | Done |
| 25 | Reproduction bundle: regenerate every number from raw inputs | scripts/run_all.sh, scripts/report.py | docs/benchmark_results.md | Done |
| 26 | Technical report ≤ 6 pages | docs/technical_report.md | — | Done |
| 27 | Raw benchmark data | data/raw (fetched; large, not committed) | sample captures | Partial — sample data only |
| 28 | Mirrors, glass, wet-look surfaces, low light covered | docs/technical_report.md §7, protocol "Avoid" | — | Partial — handled by confidence filtering + protocol; not benchmarked |
| 29 | Process evidence: commit as you work | git history | — | Done |
| 30 | Pretrained models disclosed, nothing calls our infrastructure | README.md | — | Done |
