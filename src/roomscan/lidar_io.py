"""Loaders for raw LiDAR exports: Stray Scanner and 3D Scanner App (Laan Labs).

Stray Scanner (one folder per scan):
    odometry.csv   timestamp, frame, x, y, z, qx, qy, qz, qw, fx, fy, cx, cy, ...
                   camera-to-world, camera axes in the OpenCV convention
                   (x right, y down, z forward); intrinsics at RGB resolution
    depth/NNNNNN.png       uint16 millimetres, 256x192
    confidence/NNNNNN.png  uint8 ARKit confidence 0/1/2
    rgb.mp4, imu.csv, camera_matrix.csv

3D Scanner App "All Data", per frame i (5-digit index):
    frame_i.jpg    RGB
    frame_i.json   {"cameraPoseARFrame": 16 floats, "intrinsics": 9 floats, ...}
    depth_i.png    uint16 millimetres, low-res (256x192 on current devices)
    conf_i.png     uint8 ARKit confidence 0/1/2 (optional)

Conventions are ARKit's: world is gravity-aligned with +y up; the camera looks
down its -z axis with +y up, so pixel rows (v grows downward) map to -y.
"""
import csv
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


# Stray poses use OpenCV camera axes; right-multiplying flips them to ARKit's.
CV_TO_ARKIT = np.diag([1.0, -1.0, -1.0, 1.0])
RGB_WIDTHS = (1920, 1440, 3840, 4032)


def _depth_intrinsics(K: np.ndarray, depth_w: int) -> np.ndarray:
    """Scale RGB-resolution intrinsics to the depth map.

    The RGB width is not stored per frame; 2*cx is within ~1% of it, so snap to
    the nearest standard ARKit video width (a 0.5% scale error would cost
    ~1 cm on a 2 m wall).
    """
    w = 2.0 * K[0, 2]
    snap = min(RGB_WIDTHS, key=lambda x: abs(x - w))
    if abs(snap - w) / snap < 0.03:
        w = snap
    Kd = K.copy()
    Kd[:2] *= depth_w / w
    return Kd


def find_capture(path: Path) -> Path:
    """The folder that actually holds the frames (unzipped exports nest one level)."""
    path = Path(path)
    for marker in ("odometry.csv", "info.json"):
        if (path / marker).exists():
            return path
    for marker in ("odometry.csv", "info.json"):
        hits = sorted(path.rglob(marker))
        if hits:
            return hits[0].parent
    return path


def is_stray(root: Path) -> bool:
    return (Path(root) / "odometry.csv").exists()


def _stray_rows(root: Path):
    with open(root / "odometry.csv", newline="") as fh:
        rows = list(csv.reader(fh))
    head = [h.strip() for h in rows[0]]
    return [{k: v.strip() for k, v in zip(head, r)} for r in rows[1:] if r]


def frame_count(root: Path) -> int:
    root = Path(root)
    return len(_stray_rows(root)) if is_stray(root) else len(frame_indices(root))


def _load_stray(root: Path, stride: int, min_conf: int) -> list[Frame]:
    from scipy.spatial.transform import Rotation

    frames = []
    for r in _stray_rows(root)[::stride]:
        i = int(r["frame"])
        dp = root / "depth" / f"{i:06d}.png"
        if not dp.exists():
            continue
        depth = _read_depth(dp)
        cp = root / "confidence" / f"{i:06d}.png"
        conf = np.asarray(Image.open(cp)) if cp.exists() else None
        if conf is not None:
            depth = np.where(conf >= min_conf, depth, 0.0)
        T = np.eye(4)
        T[:3, :3] = Rotation.from_quat([float(r[k]) for k in ("qx", "qy", "qz", "qw")]).as_matrix()
        T[:3, 3] = [float(r[k]) for k in ("x", "y", "z")]
        K = np.array([[float(r["fx"]), 0, float(r["cx"])], [0, float(r["fy"]), float(r["cy"])], [0, 0, 1]])
        frames.append(Frame(i, T @ CV_TO_ARKIT, _depth_intrinsics(K, depth.shape[1]), depth, conf, None))
    return frames


def frame_indices(root: Path) -> list[int]:
    idx = []
    for p in root.glob("frame_*.json"):
        m = re.match(r"frame_(\d+)\.json", p.name)
        if m:
            idx.append(int(m.group(1)))
    return sorted(idx)


def load_frames(root: Path, stride: int = 1, min_conf: int = 2) -> list[Frame]:
    root = find_capture(root)
    if is_stray(root):
        frames = _load_stray(root, stride, min_conf)
        if not frames:
            raise FileNotFoundError(f"no frames with depth found in {root}")
        return frames
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

        Kd = _depth_intrinsics(np.asarray(meta["intrinsics"], dtype=np.float64).reshape(3, 3), depth.shape[1])
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


def camera_points(f: Frame, step: int = 2, max_depth: float = 5.0, k: int = 2):
    """Camera-space points and unit normals (N,3)x2, every `step` pixels.

    Normals come from central differences `k` pixels apart in the depth image
    and are flipped to face the camera, so a wall point's normal says which
    side of the wall it was seen from. Pixels straddling a depth edge are dropped.
    """
    d = f.depth
    h, w = d.shape
    v, u = np.mgrid[0:h, 0:w]
    fx, fy, cx, cy = f.K[0, 0], f.K[1, 1], f.K[0, 2], f.K[1, 2]
    P = np.stack([(u - cx) / fx * d, -(v - cy) / fy * d, -d], axis=-1)
    valid = (d > 0.1) & (d < max_depth)

    c = (slice(k, h - k), slice(k, w - k))
    right, left = (slice(k, h - k), slice(2 * k, w)), (slice(k, h - k), slice(0, w - 2 * k))
    down, up = (slice(2 * k, h), slice(k, w - k)), (slice(0, h - 2 * k), slice(k, w - k))
    ok = valid[c] & valid[right] & valid[left] & valid[down] & valid[up]
    dc = d[c]
    for nb in (right, left, down, up):
        ok &= np.abs(d[nb] - dc) < 0.05 * dc + 0.02
    n = np.cross(P[right] - P[left], P[down] - P[up])
    norm = np.linalg.norm(n, axis=-1)
    ok &= norm > 0
    sub = (slice(None, None, step), slice(None, None, step))
    ok, n, norm, pts = ok[sub], n[sub], norm[sub], P[c][sub]
    n = n[ok] / norm[ok][:, None]
    pts = pts[ok]
    n[np.einsum("ij,ij->i", n, pts) > 0] *= -1
    return pts.astype(np.float32), n.astype(np.float32)


def to_world(pose: np.ndarray, pts: np.ndarray, normals: np.ndarray):
    R, t = pose[:3, :3], pose[:3, 3]
    return pts @ R.T + t, normals @ R.T
