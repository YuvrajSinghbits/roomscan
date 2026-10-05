# roomscan

Phone capture → dimensioned, stitched whole-property floor plan, with a
confidence interval on every measurement.

Three input tiers, one output contract ([schema/capture_output.schema.json](schema/capture_output.schema.json)):

| Tier  | Input                                         | Device           |
|-------|-----------------------------------------------|------------------|
| photo | one folder per room, 2–8 stills each          | any iPhone 15+   |
| video | one handheld walkthrough clip                 | any iPhone 15+   |
| lidar | depth + poses + intrinsics (raw app export)   | iPhone Pro (LiDAR) |

## Quick start

```bash
py -m venv .venv
.venv\Scripts\activate          # Windows  (source .venv/bin/activate on macOS/Linux)
pip install -e .[dev]
roomscan run path/to/capture --out out/my_capture
```

One command per capture. Output: `out/<id>/result.json` (validated against the
schema) and `out/<id>/plan.png`.

## Status

Work in progress — see commit history.
