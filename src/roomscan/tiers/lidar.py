"""LiDAR tier: 3D Scanner App raw export -> stitched, dimensioned plan."""
import json
from pathlib import Path

import numpy as np

from roomscan import drift as drift_mod
from roomscan.lidar_io import camera_points, load_frames
from roomscan.output import assemble, render
from roomscan.plan2d import Cloud, build_plan

MAX_POINTS = 8_000_000


def _device(capture: Path) -> str:
    info = capture / "info.json"
    if info.exists():
        meta = json.loads(info.read_text())
        for k in ("device", "deviceModel", "device_model", "model"):
            if k in meta:
                return str(meta[k])
    return "iPhone Pro (LiDAR)"


def process(capture: Path, out_dir: Path, drift: bool = True, stride: int = 1) -> dict:
    capture = Path(capture)
    frames = load_frames(capture, stride=stride)
    cam = [camera_points(f, step=2) for f in frames]
    poses, drift_info = drift_mod.correct([f.pose for f in frames], cam, enabled=drift)

    pts, nrm = [], []
    for (p, n), T in zip(cam, poses):
        pts.append(p @ T[:3, :3].T + T[:3, 3])
        nrm.append(n @ T[:3, :3].T)
    pts, nrm = np.concatenate(pts), np.concatenate(nrm)
    if len(pts) > MAX_POINTS:
        keep = np.random.default_rng(0).choice(len(pts), MAX_POINTS, replace=False)
        pts, nrm = pts[keep], nrm[keep]
    # ARKit world (x, y up, z) -> plan (X = x, Y = -z, h = y)
    cloud = Cloud(np.stack([pts[:, 0], -pts[:, 2]], 1).astype(np.float64), pts[:, 1].astype(np.float64),
                  np.stack([nrm[:, 0], -nrm[:, 2], nrm[:, 1]], 1).astype(np.float64))
    cams = np.array([[T[0, 3], -T[2, 3]] for T in poses])
    rooms, adjacency, _, _ = build_plan(cloud, cams, np.array([f.index for f in frames]))
    if not rooms:
        raise RuntimeError("no rooms recovered from capture")

    doc = assemble(capture, "lidar", _device(capture), rooms, adjacency, drift_info)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    render(doc, out_dir / "plan.png")
    return doc
