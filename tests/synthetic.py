"""Synthetic 3D Scanner App "All Data" export with known ground truth.

Ray-casts axis-aligned boxes (walls, door headers, window sill/header, a sofa)
plus floor and ceiling planes into 256x192 z-depth maps, writes them in the same
layout lidar_io reads, and injects VIO-style drift into the stored poses:
heading bias growing along the path and a slow translation bias. The returned
ground truth is in plan coordinates (X east, Y north, metres).

Layout (wall centrelines, thickness 0.10 m, ceiling 2.50 m):

    y=4.9  +-----------A-----------+------B------+
           |       window          |             |
           |   sofa                |             |
    y=1.3  +----door A-------------+----door B---+
           |            corridor                 |
    y=0.0  +-----door C------------+-------------+
           |           C           |
    y=-3.1 +-----------------------+
          x=0                     x=4.1         x=7.1
"""
import json
from pathlib import Path

import numpy as np
from PIL import Image

T = 0.10            # wall thickness
H = 2.50            # floor-to-ceiling
DOOR_H = 2.03
FLOOR_Y = -1.45     # ARKit origin is where the phone started, ~chest height
YAW = np.deg2rad(23.0)   # building is not aligned with the ARKit world axes
OFFSET = np.array([-1.7, 0.0, 0.9])

# (axis, coord, from, to, gaps[(a, b, bottom, top)])
WALLS = [
    ("h", -3.1, 0.0, 4.1, []),
    ("h", 0.0, 0.0, 7.1, [(1.50, 2.35, 0.0, DOOR_H)]),
    ("h", 1.3, 0.0, 7.1, [(2.00, 2.90, 0.0, DOOR_H), (5.00, 5.80, 0.0, DOOR_H)]),
    ("h", 4.9, 0.0, 7.1, [(1.00, 2.20, 0.90, 2.10)]),   # window in A
    ("v", 0.0, -3.1, 4.9, []),
    ("v", 4.1, -3.1, 0.0, []),
    ("v", 4.1, 1.3, 4.9, []),
    ("v", 7.1, 0.0, 4.9, []),
]
SOFA = ((0.2, 2.2), (4.0, 4.0), (0.0, 0.8))   # x, y, height ranges

TRUTH = {
    "ceiling_height": H,
    "rooms": {   # interior extents (x0, x1, y0, y1)
        "A": (0.05, 4.05, 1.35, 4.85),
        "B": (4.15, 7.05, 1.35, 4.85),
        "corridor": (0.05, 7.05, 0.05, 1.25),
        "C": (0.05, 4.05, -3.05, -0.05),
    },
    "doors": [   # (room_a, room_b, width, height)
        ("C", "corridor", 0.85, DOOR_H),
        ("A", "corridor", 0.90, DOOR_H),
        ("B", "corridor", 0.80, DOOR_H),
    ],
    "windows": [("A", 1.20, 1.20)],
}


def _boxes():
    """Axis-aligned boxes in plan coords + height above floor: (lo[3], hi[3]) with (X, Y, h)."""
    out = []
    for axis, c, a, b, gaps in WALLS:
        cuts = [a - T / 2] + [g for gap in gaps for g in gap[:2]] + [b + T / 2]
        for i in range(0, len(cuts), 2):
            out.append((axis, c, cuts[i], cuts[i + 1], 0.0, H))
        for g0, g1, bot, top in gaps:
            if bot > 0:
                out.append((axis, c, g0, g1, 0.0, bot))
            out.append((axis, c, g0, g1, top, H))
    boxes = []
    for axis, c, s0, s1, h0, h1 in out:
        if axis == "h":
            boxes.append(((s0, c - T / 2, h0), (s1, c + T / 2, h1)))
        else:
            boxes.append(((c - T / 2, s0, h0), (c + T / 2, s1, h1)))
    (x0, x1), (y0, y1), (h0, h1) = SOFA
    boxes.append(((x0, y0, h0), (x1, y1, h1)))
    return np.array(boxes, dtype=np.float64)


BOXES = _boxes()
EXTENT = (-0.05, 7.15, -3.15, 4.95)


def _cast(origin, dirs):
    """origin (3,), dirs (N,3) in plan/height coords -> ray parameter t (N,), inf on miss."""
    t_best = np.full(len(dirs), np.inf)
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = 1.0 / dirs
        for lo, hi in BOXES:
            t1 = (lo - origin) * inv
            t2 = (hi - origin) * inv
            tmin = np.nanmax(np.minimum(t1, t2), axis=1)
            tmax = np.nanmin(np.maximum(t1, t2), axis=1)
            hit = (tmax >= np.maximum(tmin, 0)) & (tmin > 1e-6)
            t_best = np.where(hit & (tmin < t_best), tmin, t_best)
        for plane in (0.0, H):
            t = (plane - origin[2]) / dirs[:, 2]
            p = origin + t[:, None] * dirs
            inside = (p[:, 0] > EXTENT[0]) & (p[:, 0] < EXTENT[1]) & (p[:, 1] > EXTENT[2]) & (p[:, 1] < EXTENT[3])
            ok = (t > 1e-6) & inside & (t < t_best)
            t_best = np.where(ok, t, t_best)
    return t_best


def _rot_y(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _rot_x(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def plan_to_world(p):
    """(X, Y, h) plan coords -> ARKit world (x, y, z), y up, plan Y = -z."""
    p = np.atleast_2d(p)
    local = np.stack([p[:, 0], p[:, 2] + FLOOR_Y, -p[:, 1]], axis=1)
    return local @ _rot_y(YAW).T + OFFSET


def _trajectory():
    """(X, Y, heading_rad, pitch_rad) samples: corridor -> B -> A -> C -> back to start."""
    stops = [
        ("walk", [(0.6, 0.65), (5.4, 0.65)]),
        ("room", (5.6, 3.1)),
        ("walk", [(5.4, 0.65), (2.45, 0.65)]),
        ("room", (2.1, 3.0)),
        ("walk", [(2.45, 0.65), (1.92, 0.65)]),
        ("room", (2.0, -1.5)),
        ("walk", [(1.92, 0.65), (0.6, 0.65)]),
    ]
    out = []
    pos = np.array([0.6, 0.65])
    for kind, arg in stops:
        if kind == "walk":
            for p in arg:
                p = np.asarray(p, float)
                n = max(2, int(np.linalg.norm(p - pos) / 0.12))
                head = np.arctan2(*(p - pos)[::-1]) if np.linalg.norm(p - pos) > 1e-6 else 0.0
                for s in np.linspace(0, 1, n, endpoint=False):
                    out.append((*(pos + s * (p - pos)), head, np.deg2rad(-10)))
                pos = p
        else:
            c = np.asarray(arg, float)
            # step into the room through the door, then one slow lap with a sweeping tilt
            for s in np.linspace(0, 1, 10, endpoint=False):
                q = pos + s * (c - pos)
                out.append((*q, np.arctan2(*(c - pos)[::-1]), np.deg2rad(-10)))
            k = 90
            for i in range(k):
                a = 2 * np.pi * i / k
                q = c + 0.5 * np.array([np.cos(a), np.sin(a)])
                pitch = np.deg2rad(30 * np.sin(4 * a))
                out.append((*q, a + np.pi, pitch))
            for s in np.linspace(0, 1, 10, endpoint=False):
                q = c + s * (pos - c)
                out.append((*q, np.arctan2(*(pos - c)[::-1]), np.deg2rad(-10)))
    # close the loop facing the same way as the first frame
    out.append((0.6, 0.65, out[0][2], out[0][3]))
    return out


def make_capture(root: Path, drift: bool = True, seed: int = 0,
                 w: int = 256, h: int = 192) -> dict:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    rgb_w, rgb_h = 1920, 1440
    K = np.array([[1450.0, 0, rgb_w / 2], [0, 1450.0, rgb_h / 2], [0, 0, 1]])
    s = w / rgb_w
    fx, fy, cx, cy = K[0, 0] * s, K[1, 1] * s, K[0, 2] * s, K[1, 2] * s
    v, u = np.mgrid[0:h, 0:w]
    ray_cam = np.stack([(u - cx) / fx, -(v - cy) / fy, -np.ones_like(u, float)], -1).reshape(-1, 3)

    traj = _trajectory()
    # VIO drift model: heading bias grows along the path, translation bias accumulates slowly
    path = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(np.array([t[:2] for t in traj]), axis=0), axis=1))])
    yaw_bias = np.deg2rad(1.5) * path / path[-1] if drift else np.zeros(len(traj))
    trans_rate = np.array([0.012, 0.0, -0.008]) if drift else np.zeros(3)   # m per m walked

    est_pos = None
    prev_true = None
    for i, (X, Y, head, pitch) in enumerate(traj):
        # camera-to-plan rotation (plan axes X, Y, h with camera -z forward)
        R_world = _rot_y(YAW) @ _rot_y(head - np.pi / 2) @ _rot_x(pitch)
        p_true = plan_to_world([X, Y, 1.45])[0]
        # render with the true pose; directions in plan coords for the caster
        d_world = ray_cam @ R_world.T
        d_local = (d_world) @ _rot_y(YAW)   # undo building yaw
        d_plan = np.stack([d_local[:, 0], -d_local[:, 2], d_local[:, 1]], 1)
        t = _cast(np.array([X, Y, 1.45]), d_plan)
        z = t * 1.0   # ray_cam has unit -z component, so t is z-depth
        z = np.where(np.isfinite(z) & (z < 5.0), z, 0.0)
        noise = rng.normal(0, 0.003 + 0.004 * z)
        z = np.where(z > 0, z + noise, 0.0).reshape(h, w)

        # stored (drifted) pose
        if est_pos is None:
            est_pos = p_true.copy()
        else:
            step = p_true - prev_true
            est_pos = est_pos + _rot_y(yaw_bias[i]) @ step + trans_rate * np.linalg.norm(step)
        prev_true = p_true
        R_est = _rot_y(yaw_bias[i]) @ R_world
        T_est = np.eye(4)
        T_est[:3, :3] = R_est
        T_est[:3, 3] = est_pos

        stem = f"{i:05d}"
        Image.fromarray(np.round(z * 1000).astype(np.uint16)).save(root / f"depth_{stem}.png")
        Image.fromarray(np.full((h, w), 2, np.uint8)).save(root / f"conf_{stem}.png")
        (root / f"frame_{stem}.json").write_text(json.dumps({
            "cameraPoseARFrame": T_est.reshape(-1).tolist(),
            "intrinsics": K.reshape(-1).tolist(),
            "frame_index": i,
        }))
    (root / "info.json").write_text(json.dumps({"device": "synthetic iPhone Pro", "frames": len(traj)}))
    return TRUTH
