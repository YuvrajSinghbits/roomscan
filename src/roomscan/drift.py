"""Drift correction for a single continuous walkthrough.

ARKit poses are gravity-aligned, so roll/pitch drift is negligible; what
accumulates over a multi-room walk is heading (yaw) and position. Two
corrections, each switchable for the ablation:

1. Manhattan heading anchor. Walls in a property are (almost always) pairwise
   perpendicular, so every chunk of the walk should see the same dominant wall
   direction modulo 90 deg. The chunk-to-chunk change of that direction is the
   heading drift. It is removed incrementally: each frame's rotation is
   corrected and its *step* from the previous frame is re-rotated, so position
   error induced by the heading error goes too.

2. Loop closure. The capture protocol ends the walk where it started, facing
   the same way. Wall-point images of the first and last chunks are
   cross-correlated (translation only, after heading correction) to measure the
   accumulated offset, and the floor level of both chunks gives the vertical
   offset. The offset is distributed along the path in proportion to the
   distance walked.
"""
import numpy as np
from scipy import signal

from roomscan.plan2d import manhattan_angle

CHUNK = 15          # frames per heading estimate
LOOP_CHUNK = 15     # frames at each end used for loop closure
IMG_CELL = 0.02
SEARCH = 0.6        # max loop-closure offset searched, metres


def rot_y(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _plan(pts):
    return np.stack([pts[:, 0], -pts[:, 2]], 1)


def _world_cloud(poses, cam, idx):
    pts, nrm = [], []
    for i in idx:
        p, n = cam[i]
        R, t = poses[i][:3, :3], poses[i][:3, 3]
        pts.append(p @ R.T + t)
        nrm.append(n @ R.T)
    return np.concatenate(pts), np.concatenate(nrm)


def _chunk_angle(poses, cam, idx):
    _, n = _world_cloud(poses, cam, idx)
    vert = np.abs(n[:, 1]) < 0.2
    if vert.sum() < 200:
        return None
    return manhattan_angle(np.stack([n[vert, 0], -n[vert, 2]], 1))


def _apply_heading(poses, corr):
    """Re-integrate the trajectory with per-frame heading corrections (rad)."""
    out = [poses[0].copy()]
    for i in range(1, len(poses)):
        T = poses[i].copy()
        R = rot_y(corr[i])
        T[:3, :3] = R @ poses[i][:3, :3]
        T[:3, 3] = out[-1][:3, 3] + R @ (poses[i][:3, 3] - poses[i - 1][:3, 3])
        out.append(T)
    return out


def heading_correction(poses, cam):
    n = len(poses)
    starts = list(range(0, n, CHUNK))
    angles, centres = [], []
    for s in starts:
        idx = range(s, min(s + CHUNK, n))
        a = _chunk_angle(poses, cam, idx)
        if a is not None:
            angles.append(a)
            centres.append(s + (len(idx) - 1) / 2)
    if len(angles) < 2:
        return poses, np.zeros(n)
    angles = np.array(angles)
    # unwrap in the mod-90 domain, then anchor to the first chunk
    rel = np.unwrap(4 * (angles - angles[0])) / 4
    # median-of-3 smoothing: one chunk staring at furniture should not swing the walk
    sm = rel.copy()
    for i in range(1, len(rel) - 1):
        sm[i] = np.median(rel[i - 1:i + 2])
    corr = -np.interp(np.arange(n), centres, sm)
    return _apply_heading(poses, corr), corr


def _wall_image(pts, nrm, origin, shape):
    vert = np.abs(nrm[:, 1]) < 0.2
    xy = _plan(pts[vert])
    c = ((xy - origin) / IMG_CELL).astype(int)
    ok = (c[:, 0] >= 0) & (c[:, 0] < shape[1]) & (c[:, 1] >= 0) & (c[:, 1] < shape[0])
    img = np.zeros(shape)
    np.add.at(img, (c[ok, 1], c[ok, 0]), 1)
    return np.minimum(img, 5)


def _floor_level(pts, nrm):
    up = pts[nrm[:, 1] > 0.95, 1]
    if len(up) < 50:
        return None
    lo = np.percentile(up, 5)
    v = up[np.abs(up - lo) < 0.06]
    return float(np.median(v))


def loop_closure(poses, cam):
    """Returns (corrected poses, info dict). No-op when the walk does not close."""
    n = len(poses)
    k = min(LOOP_CHUNK, n // 4)
    head, tail = range(0, k), range(n - k, n)
    p0, p1 = poses[0][:3, 3], poses[-1][:3, 3]
    if np.linalg.norm(_plan((p1 - p0)[None])[0]) > 2.0:
        return poses, {"closed": False, "reason": "walk did not end near its start"}
    a_pts, a_n = _world_cloud(poses, cam, head)
    b_pts, b_n = _world_cloud(poses, cam, tail)
    both = np.concatenate([_plan(a_pts), _plan(b_pts)])
    origin = both.min(0) - SEARCH
    shape = tuple((np.ceil((both.max(0) + SEARCH - origin) / IMG_CELL)).astype(int)[::-1])
    A = _wall_image(a_pts, a_n, origin, shape)
    B = _wall_image(b_pts, b_n, origin, shape)
    if A.sum() < 200 or B.sum() < 200:
        return poses, {"closed": False, "reason": "too little wall overlap at loop ends"}
    xc = signal.fftconvolve(A, B[::-1, ::-1], mode="same")
    cy, cx = np.array(xc.shape) // 2
    r = int(SEARCH / IMG_CELL)
    win = xc[cy - r:cy + r + 1, cx - r:cx + r + 1]
    iy, ix = np.unravel_index(np.argmax(win), win.shape)
    # sub-cell peak by parabola fit
    def sub(v, i):
        if 0 < i < len(v) - 1:
            d = v[i - 1] - 2 * v[i] + v[i + 1]
            return i + (0.5 * (v[i - 1] - v[i + 1]) / d if d != 0 else 0)
        return i
    fy, fx = sub(win[:, ix], iy), sub(win[iy, :], ix)
    shift = np.array([fx - r, fy - r]) * IMG_CELL      # move B by this to match A
    peak_ratio = float(win.max() / (np.median(win[win > 0]) + 1e-9)) if (win > 0).any() else 0.0
    if peak_ratio < 3:
        return poses, {"closed": False, "reason": f"ambiguous loop match (peak ratio {peak_ratio:.1f})"}

    fa, fb = _floor_level(a_pts, a_n), _floor_level(b_pts, b_n)
    dy = (fa - fb) if fa is not None and fb is not None else 0.0
    delta = np.array([shift[0], dy, -shift[1]])          # world-frame correction at the end

    steps = [0.0] + [np.linalg.norm(poses[i][:3, 3] - poses[i - 1][:3, 3]) for i in range(1, n)]
    s = np.cumsum(steps)
    # the match measures drift accrued between the two chunk centres, not between
    # the first and last frame: anchor the linear ramp at those centres
    s_a, s_b = s[k // 2], s[n - k + k // 2]
    w = np.clip((s - s_a) / (s_b - s_a), 0, None) if s_b > s_a else np.zeros(n)
    out = []
    for i, T in enumerate(poses):
        T = T.copy()
        T[:3, 3] += w[i] * delta
        out.append(T)
    return out, {"closed": True, "offset_m": [round(float(x), 4) for x in delta],
                 "peak_ratio": round(peak_ratio, 1), "path_m": round(float(s[-1]), 2)}


def correct(poses, cam, enabled=True):
    if not enabled:
        return poses, {"method": "none", "heading_correction_deg_max": 0.0, "loop": {"closed": False}}
    poses, corr = heading_correction(poses, cam)
    poses, loop = loop_closure(poses, cam)
    method = "manhattan_heading_anchor" + ("+loop_closure" if loop["closed"] else "")
    return poses, {"method": method,
                   "heading_correction_deg_max": round(float(np.rad2deg(np.abs(corr).max())), 3),
                   "loop": loop}
