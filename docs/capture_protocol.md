# Capture protocol (one page)

Follow this literally. You need an iPhone 15 or newer (Pro / Pro Max for the LiDAR tier).

## Before any capture (all tiers)
1. Turn **on every light** in every room. Open curtains in daytime.
2. Open **every interior door fully** (flat against the wall if it can).
3. Note which rooms you will capture and give each a short name: `kitchen`, `hall`, `bed1`...
4. Hold the phone **upright (portrait)**, at chest height, unless a step says otherwise.
5. Mirrors and large glass: do not stand still pointing at them. Sweep past them.

## Tier 1 — Photos (any iPhone 15+, native Camera app)
Per room, take **4 to 8 photos** with the **1x lens** (not 0.5x, no zoom, no Portrait mode, Live Photos off):
1. Stand in each **corner** and photograph the opposite corner, so floor and ceiling lines are both visible. (4 photos for a 4-corner room.)
2. For **every doorway that leads to another captured room**, stand 1–2 m back and take one photo with the **whole door frame visible, top to floor**. This is how rooms are joined together — do not skip it.
3. If a room has visible damage, add one photo of each damaged area from about 1.5 m away.

Hand-off: AirDrop / USB / Google Drive the photos into one folder per room:
```
data/raw/<property>_photo/kitchen/IMG_0001.HEIC ...
data/raw/<property>_photo/hall/...
```
Folder names are the room names. HEIC or JPG both fine; do not edit or crop.

## Tier 2 — Video (any iPhone 15+, native Camera app)
Settings → Camera → Record Video: **1080p at 30 fps**. 1x lens. Do not use Cinematic or Action mode.
1. Start recording in the first room. Walk **slowly** (half normal walking speed) along the walls, phone pointing at the **opposite wall**, so every wall is seen once.
2. In each room, do **one slow vertical tilt** from floor to ceiling.
3. Move to the next room **through the doorway** without stopping the recording. Do not cover the lens or turn around quickly.
4. Finish back in the room you started in, pointing at the same spot you started at (this lets the pipeline close the loop).
5. About **30–45 s per room**. One single clip for the whole property.

Hand-off: `data/raw/<property>_video.MOV`

## Tier 3 — LiDAR (iPhone Pro, "3D Scanner App" by Laan Labs, free)
1. Install **3D Scanner App** from the App Store. Allow camera access.
2. Choose the **LiDAR** mode. Settings: resolution 5 mm, max depth 5 m, confidence High.
3. Press record in the first room. Walk the same path as the video tier: along the walls, one floor-to-ceiling tilt per room, through doorways, end where you started.
4. Move slowly; if the app shows "Slow down", stop for one second.
5. Stop, let it process, then **Share → Export → All Data** (raw frames, depth, poses). Save the zip.

Hand-off: unzip into `data/raw/<property>_lidar/` (it contains `info.json`, `frame_*.jpg`, `frame_*.json`, `depth_*.png`).

## Then run
```
roomscan run data/raw/<capture>
```

## Avoid
- Walking fast, spinning in place, covering the lens with a finger.
- Capturing with people moving around in the room.
- Zoom, 0.5x lens, Portrait / Cinematic modes, filters.
- Closed doors between rooms you want stitched.
