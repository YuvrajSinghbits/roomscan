# Device matrix

| Tier | Hardware | Capture app | Sensors used | Runs on |
|---|---|---|---|---|
| lidar | iPhone 12 Pro or newer Pro/Pro Max (LiDAR); target iPhone 15 Pro / Pro Max | Stray Scanner (free) — or 3D Scanner App "All Data" | LiDAR depth 256×192, ARKit VIO poses, intrinsics, confidence | any laptop, CPU only |
| video | any iPhone 15 or newer (incl. Pro) | native Camera app, 1080p30, 1x | RGB only (no depth, no poses, no IMU in the file) | CPU (≈0.8 s/frame depth inference) |
| photo | any iPhone 15 or newer (incl. Pro) | native Camera app, 1x | RGB + EXIF focal length | CPU |

Android phones are not supported: the protocol, focal-length priors and the LiDAR
loader are iPhone-specific.

## What each tier honestly delivers

Every number is a 90% interval half-width; the pipeline writes the actual interval
into every measurement of every capture.

| Quantity | lidar | video | photo |
|---|---|---|---|
| Wall length | ±1.2 cm where the wall surface is scanned; ±25 cm on outline sides with no wall surface behind them | ±3% + 2 cm (error model) on top of fit spread | ±5% + 2 cm on top of the corner-photo spread and ±8 cm corner-offset prior |
| Ceiling height | ±1 cm when the ceiling is scanned; lower bound only when it is not | as lidar, + error model | median of per-photo floor-to-ceiling, + error model |
| Openings | width ±1.2–3.5 cm (jamb-refined edges), head height ±1 cm | as lidar, + error model | width from door photos; position along wall not observed (interval = whole wall) |
| Drift | Manhattan heading anchor + loop closure | same, on visual odometry (heading drift ~10× ARKit's) | n/a (independent stills) |

## Evidence behind these numbers

* LiDAR tier, synthetic export with known geometry and injected VIO drift
  (`pytest`): walls within 1 cm, ceiling within 1.5 cm, door widths within 1 cm,
  window within 2 cm; drift off inflates footprint error from 0.04 m² to 3.4 m².
* Monocular depth vs LiDAR on 72 frames of the sample captures
  (`scripts/calibrate_mono.py`): median scale bias ×1.159 (now calibrated out),
  per-frame scale spread 37%, within-frame error 11% after scale removal. This is
  why the video and photo intervals are wide and why the photo-tier ±8% gate is
  at risk.
* Cross-tier results on the sample captures: [benchmark_results.md](benchmark_results.md).
* No tape/laser ground truth exists for the sample captures; the LiDAR tier is
  the reference for the other two tiers there.
