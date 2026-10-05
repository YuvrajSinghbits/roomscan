"""Loader for 3D Scanner App (Laan Labs) "All Data" exports.

Layout per frame i (5-digit index):
    frame_i.jpg    RGB
    frame_i.json   {"cameraPoseARFrame": 16 floats, "intrinsics": 9 floats, ...}
    depth_i.png    uint16 millimetres, low-res (256x192 on current devices)
    conf_i.png     uint8 ARKit confidence 0/1/2 (optional)

Conventions are ARKit's: world is gravity-aligned with +y up; the camera looks
down its -z axis with +y up, so pixel rows (v grows downward) map to -y.
"""
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


@dataclass
class Frame:
    index: int
    pose: np.ndarray        # 4x4 camera-to-world
    K: np.ndarray           # 3x3 intrinsics at depth resolution
    depth: np.ndarray       # HxW metres, 0 = invalid
    conf: np.ndarray | None  # HxW 0/1/2
    image: Path | None

    @property
    def center(self) -> np.ndarray:
        return self.pose[:3, 3]


def _pose(vals) -> np.ndarray:
    m = np.asarray(vals, dtype=np.float64).reshape(4, 4)
    # Row-major transforms end in [0,0,0,1]; if the translation sits in the
    # bottom row instead, the file was written column-major.
    if not np.allclose(m[3], [0, 0, 0, 1], atol=1e-6) and np.allclose(m[:, 3], [0, 0, 0, 1], atol=1e-6):
        m = m.T
    return m


def _read_depth(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        return np.load(path).astype(np.float32)
    return np.asarray(Image.open(path), dtype=np.float32) / 1000.0


def frame_indices(root: Path) -> list[int]:
    idx = []
    for p in root.glob("frame_*.json"):
        m = re.match(r"frame_(\d+)\.json", p.name)
        if m:
            idx.append(int(m.group(1)))
    return sorted(idx)


def load_frames(root: Path, stride: int = 1, min_conf: int = 2) -> list[Frame]:
    root = Path(root)
    frames = []
    for i in frame_indices(root)[::stride]:
        stem = f"{i:05d}"
        depth_path = next((p for p in (root / f"depth_{stem}.png", root / f"depth_{stem}.npy") if p.exists()), None)
        if depth_path is None:
            continue
        meta = json.loads((root / f"frame_{stem}.json").read_text())
        depth = _read_depth(depth_path)
        conf_path = root / f"conf_{stem}.png"
        conf = np.asarray(Image.open(conf_path)) if conf_path.exists() else None
        if conf is not None:
            depth = np.where(conf >= min_conf, depth, 0.0)

        K = np.asarray(meta["intrinsics"], dtype=np.float64).reshape(3, 3)
        # Intrinsics are given at RGB resolution; principal point ~ image centre,
        # so 2*cx recovers the RGB width and gives the scale to depth resolution.
        s = depth.shape[1] / (2.0 * K[0, 2])
        Kd = K.copy()
        Kd[:2] *= s
        img = root / f"frame_{stem}.jpg"
        frames.append(Frame(i, _pose(meta["cameraPoseARFrame"]), Kd, depth, conf,
                            img if img.exists() else None))
    if not frames:
        raise FileNotFoundError(f"no frames with depth found in {root}")
    return frames


def backproject(f: Frame, step: int = 2, max_depth: float = 5.0) -> np.ndarray:
    """World-space points (N,3) from one frame, sampling every `step` pixels."""
    d = f.depth[::step, ::step]
    h, w = d.shape
    v, u = np.mgrid[0:h, 0:w]
    u = u * step
    v = v * step
    ok = (d > 0.1) & (d < max_depth)
    fx, fy, cx, cy = f.K[0, 0], f.K[1, 1], f.K[0, 2], f.K[1, 2]
    z = d[ok]
    pc = np.stack([(u[ok] - cx) / fx * z, -(v[ok] - cy) / fy * z, -z], axis=1)
    return pc @ f.pose[:3, :3].T + f.pose[:3, 3]


def project(f: Frame, pts: np.ndarray):
    """World points -> (u, v, depth_along_view). depth<=0 means behind camera."""
    R, t = f.pose[:3, :3], f.pose[:3, 3]
    pc = (pts - t) @ R
    z = -pc[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = f.K[0, 0] * pc[:, 0] / z + f.K[0, 2]
        v = -f.K[1, 1] * pc[:, 1] / z + f.K[1, 2]
    return u, v, z
