"""LiDAR tier: 3D Scanner App raw export -> stitched, dimensioned plan."""
import json
from pathlib import Path

import numpy as np

from roomscan import drift as drift_mod
from roomscan.lidar_io import camera_points, find_capture, frame_count, load_frames, load_rgb
from roomscan.output import assemble, render
from roomscan.plan2d import Cloud, build_plan

MAX_POINTS = 8_000_000
TARGET_FRAMES = 600     # Stray records at 60 fps; neighbouring frames add little
DAMAGE_VIEWS = 20


def _device(capture: Path) -> str:
    info = capture / "info.json"
    if info.exists():
        meta = json.loads(info.read_text())
        for k in ("device", "deviceModel", "device_model", "model"):
            if k in meta:
                return str(meta[k])
    return "iPhone Pro (LiDAR)"


def process(capture: Path, out_dir: Path, drift: bool = True, stride: int | None = None,
            damage: bool = True, **_) -> dict:
    capture = find_capture(Path(capture))
    if stride is None:
        stride = max(1, -(-frame_count(capture) // TARGET_FRAMES))
    frames = load_frames(capture, stride=stride)
    cam = [camera_points(f, step=2) for f in frames]
    poses, drift_info = drift_mod.correct([f.pose for f in frames], cam, enabled=drift)

    pts, nrm, src = [], [], []
    for i, ((p, n), T) in enumerate(zip(cam, poses)):
        pts.append(p @ T[:3, :3].T + T[:3, 3])
        nrm.append(n @ T[:3, :3].T)
        src.append(np.full(len(p), i, np.int32))
    pts, nrm, src = np.concatenate(pts), np.concatenate(nrm), np.concatenate(src)
    if len(pts) > MAX_POINTS:
        keep = np.random.default_rng(0).choice(len(pts), MAX_POINTS, replace=False)
        pts, nrm, src = pts[keep], nrm[keep], src[keep]
    # ARKit world (x, y up, z) -> plan (X = x, Y = -z, h = y)
    cloud = Cloud(np.stack([pts[:, 0], -pts[:, 2]], 1).astype(np.float64), pts[:, 1].astype(np.float64),
                  np.stack([nrm[:, 0], -nrm[:, 2], nrm[:, 1]], 1).astype(np.float64), src)
    cams = np.array([[T[0, 3], -T[2, 3]] for T in poses])
    rooms, adjacency, theta, _, floor = build_plan(cloud, cams, np.array([f.index for f in frames]))
    if not rooms:
        raise RuntimeError("no rooms recovered from capture")

    doc = assemble(capture, "lidar", _device(capture), rooms, adjacency, drift_info)
    if damage:
        from roomscan.damage import View, run
        pick = np.linspace(0, len(frames) - 1, min(DAMAGE_VIEWS, len(frames))).astype(int)
        rgbs = load_rgb(capture, [frames[i].index for i in pick])
        views = [View(rgb, frames[i].depth, frames[i].K, poses[i]) for i, rgb in zip(pick, rgbs) if rgb is not None]
        run(doc, views, rooms, theta, floor)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    render(doc, out_dir / "plan.png")
    return doc
