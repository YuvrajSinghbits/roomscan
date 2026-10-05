# roomscan

Phone capture → dimensioned, stitched whole-property floor plan, with a 90%
confidence interval on every measurement, damage regions, concealed-damage
flags and scope line items.

Three input tiers, one output contract ([schema/capture_output.schema.json](schema/capture_output.schema.json)):

| Tier  | Input                                              | Device              | Core method |
|-------|----------------------------------------------------|---------------------|-------------|
| lidar | Stray Scanner folder (or 3D Scanner App "All Data") | iPhone 15 Pro / Pro Max | ARKit poses + LiDAR depth, drift-corrected |
| video | one handheld walkthrough clip (.MOV/.mp4)          | any iPhone 15+      | monocular metric depth + visual odometry |
| photo | one folder per room, 2–8 stills each               | any iPhone 15+      | monocular metric depth per photo + door-view stitching |

Capture route: [docs/capture_protocol.md](docs/capture_protocol.md) (one page, stock apps only).
Device matrix and honest accuracy per tier: [docs/device_matrix.md](docs/device_matrix.md).

## Quick start (clean machine, ~10 min)

Requires Python 3.10–3.12.

```bash
py -m venv .venv
.venv\Scripts\activate                     # Windows  (source .venv/bin/activate on macOS/Linux)
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e .[dev]
python scripts/fetch_weights.py            # ~0.7 GB of pretrained weights into weights/
```

## One command per capture

```bash
roomscan run data/raw/<property>_lidar          # tier auto-detected
roomscan run data/raw/<property>_video.MOV
roomscan run data/raw/<property>_photo --tier photo
```

Output: `out/<id>/result.json` (validated against the schema) and `out/<id>/plan.png`.
Options: `--no-drift-correction` (ablation), `--no-damage` (geometry only, faster).

## Reproduce every reported number

```bash
# sample data: the three Stray Scanner captures from the assignment Drive folder,
# unzipped into data/raw/sample/Assignment - YC Startup/
bash scripts/run_all.sh        # all tiers, drift ablation, fix loop -> docs/benchmark_results.md
pytest                          # synthetic end-to-end test with known geometry and injected drift
```

Monocular depth predictions are cached in `.cache/depth` (keyed by model and
image bytes) so reruns replay deterministically; delete it to force the live path.

## Layout

| Path | What |
|---|---|
| `src/roomscan/lidar_io.py` | Stray Scanner / 3D Scanner App loaders, points + camera-facing normals |
| `src/roomscan/drift.py` | Manhattan heading anchor + loop closure (switchable for the ablation) |
| `src/roomscan/plan2d.py` | tier-agnostic: floor/ceiling, ray-carved free space, rooms, walls, openings |
| `src/roomscan/tiers/{lidar,video,photo}.py` | the three tiers |
| `src/roomscan/depth_model.py` | Depth Anything V2 metric indoor + calibration + cache |
| `src/roomscan/damage.py` | CLIPSeg zero-shot damage, lifted onto plan surfaces |
| `src/roomscan/scope.py` | concealed-damage rules R1–R5, scope line items |
| `src/roomscan/output.py` | schema document, error-model widening, plan rendering |
| `tests/` | synthetic 3-room + corridor export with injected drift |
| `docs/` | protocol, device matrix, compliance matrix, technical report, fix loop, results |

## Pretrained models (disclosure)

- Depth Anything V2 Metric Indoor Small — `depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf` (Apache-2.0)
- CLIPSeg — `CIDAS/clipseg-rd64-refined` (Apache-2.0)

Both run locally on CPU; nothing calls our infrastructure.
