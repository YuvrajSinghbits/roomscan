"""Point cloud with normals -> dimensioned, stitched multi-room floor plan.

Tier-agnostic: anything that can produce gravity-aligned points with
camera-facing normals (LiDAR depth, or monocular depth + SfM poses) feeds this.

Coordinates: plan frame X/Y in metres (Y = -z of the ARKit world), h = height
above the floor. Walls are assumed Manhattan (pairwise perpendicular); the
dominant wall direction is estimated and the plan is rotated to align with it.

Pipeline:
  1. floor / ceiling levels from up- and down-facing surface points
  2. 5 cm grid: wall occupancy from vertical points at 1.0-1.95 m (above most
     furniture, below door heads), observed space from floor + ceiling hits
  3. rooms: free space cut at door soffits and at passages narrower than
     ~0.95 m, seeds grown back to fill free space
  4. per room: rectilinear outline traced from the mask, short jogs removed,
     then every wall re-fitted to the points on its room-facing surface; the
     fit residual sets that wall's interval
  5. openings: doors where two rooms touch, windows where a wall face has a
     hole with a sill below and a header above; edges refined on jamb points
"""
from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage

CELL = 0.05
Z90 = 1.645            # 90% two-sided normal quantile

# Error budget (1-sigma, metres). Systematic terms that do not average out
# over many points: LiDAR range bias, residual drift, normal-estimate bias.
# These are starting values to be calibrated against tape/laser ground truth.
SIG_WALL_SYS = 0.005
SIG_LEVEL_SYS = 0.004
SIG_EDGE_SYS = 0.005
N_EFF_MAX = 150        # neighbouring points are correlated; cap effective n
# A side of the outline with no wall surface behind it (unscanned, occluded,
# open to an unscanned space) is not a measured wall: its true position is
# unknown, so it carries this half-width instead of a fitted one.
UNSUPPORTED_HW = 0.25
MIN_WALL_PTS = 40
MIN_WALL_COVER = 0.35
# Wall fits use surfaces above typical furniture (beds, sofas, desks, counters)
WALL_FIT_MIN_H = 0.9
MIN_OPENING = 0.45     # narrower gaps are not walkable doorways
MAX_OPENING = 2.6      # wider "gaps" are unscanned wall, not an opening
MAX_DOOR_W = 1.3       # door vs open passage when the head was not scanned


@dataclass
class Cloud:
    xy: np.ndarray       # (N,2) plan coordinates
    h: np.ndarray        # (N,) height (absolute until floor is known)
    n: np.ndarray        # (N,3) normals (nX, nY, nh), facing the camera
    src: np.ndarray | None = None   # (N,) index of the camera that saw each point


@dataclass
class Line:
    axis: str            # "v": plane X=c ; "h": plane Y=c
    c: float
    inward: int          # +1 / -1: direction of room interior along the normal axis
    hw: float = 0.05     # 90% half-width of c
    n_pts: int = 0
    cover: float = 0.0   # fraction of the wall's length backed by surface points

    @property
    def supported(self) -> bool:
        return self.n_pts >= MIN_WALL_PTS and self.cover >= MIN_WALL_COVER


@dataclass
class Opening:
    kind: str
    line: int
    s0: float
    s1: float
    hw0: float
    hw1: float
    height: float | None = None
    height_hw: float | None = None
    other: str | None = None
    id: str = ""


@dataclass
class Room:
    id: str
    name: str
    mask: np.ndarray
    lines: list = field(default_factory=list)
    openings: list = field(default_factory=list)
    floor: float = 0.0
    ceiling: float = 0.0
    height_hw: float = 0.0
    ceiling_seen: bool = True
    first_frame: int = 0


# ---------------------------------------------------------------- orientation

def manhattan_angle(nxy: np.ndarray, weights=None) -> float:
    """Dominant wall direction (rad, in [-pi/4, pi/4)) from horizontal normals.

    Angles are multiplied by 4 so all four wall directions vote together;
    a second pass keeps only normals within 10 deg of the first estimate.
    """
    a = np.arctan2(nxy[:, 1], nxy[:, 0])
    w = np.ones(len(a)) if weights is None else weights
    theta = np.angle(np.sum(w * np.exp(4j * a))) / 4
    d = np.angle(np.exp(4j * (a - theta))) / 4
    keep = np.abs(d) < np.deg2rad(10)
    if keep.sum() > 20:
        theta = theta + np.angle(np.sum(w[keep] * np.exp(4j * d[keep]))) / 4
    return float(np.angle(np.exp(4j * theta)) / 4)


def rotate(xy, theta):
    c, s = np.cos(theta), np.sin(theta)
    return xy @ np.array([[c, -s], [s, c]])   # row vectors: rotate by -theta


# ----------------------------------------------------------------- levels

def _peak(vals, lo, hi, bin_=0.01):
    vals = vals[(vals > lo) & (vals < hi)]
    if len(vals) < 20:
        return None
    hist, edges = np.histogram(vals, bins=np.arange(lo, hi + bin_, bin_))
    hist = ndimage.uniform_filter1d(hist.astype(float), 3)
    return edges[np.argmax(hist)] + bin_ / 2


def _refine_level(vals, guess, win=0.03):
    """Trimmed mean around a level; returns (level, 1-sigma standard error)."""
    v = vals[np.abs(vals - guess) < 0.10]
    if len(v) < 20:
        return guess, 0.02
    m = np.median(v)
    v = v[np.abs(v - m) < win]
    return float(v.mean()), float(v.std() / np.sqrt(min(len(v), N_EFF_MAX)))


def floor_ceiling(cloud: Cloud):
    nh = cloud.n[:, 2]
    up, down = cloud.h[nh > 0.95], cloud.h[nh < -0.95]
    # floor: lowest substantial up-facing level; ceiling: highest down-facing
    hist_lo = _peak(up, up.min() if len(up) else -3, np.median(up) + 0.5 if len(up) else 3)
    if hist_lo is None:
        raise ValueError("no floor surface observed")
    floor = _refine_level(up, hist_lo)[0]
    ceil_guess = _peak(down, floor + 1.8, floor + 4.5)
    if ceil_guess is None or (np.abs(down - ceil_guess) < 0.05).sum() < 200:
        return floor, None
    ceiling = _refine_level(down, ceil_guess)[0]
    return floor, ceiling


# ------------------------------------------------------------------ grids

class Grid:
    def __init__(self, xy, pad=0.6):
        self.x0, self.y0 = xy.min(0) - pad
        x1, y1 = xy.max(0) + pad
        self.shape = (int(np.ceil((y1 - self.y0) / CELL)), int(np.ceil((x1 - self.x0) / CELL)))

    def idx(self, xy):
        c = np.floor((xy[:, 0] - self.x0) / CELL).astype(int)
        r = np.floor((xy[:, 1] - self.y0) / CELL).astype(int)
        ok = (r >= 0) & (r < self.shape[0]) & (c >= 0) & (c < self.shape[1])
        return r, c, ok

    def count(self, xy):
        g = np.zeros(self.shape, np.int32)
        r, c, ok = self.idx(xy)
        np.add.at(g, (r[ok], c[ok]), 1)
        return g

    def x(self, col):
        return self.x0 + col * CELL

    def y(self, row):
        return self.y0 + row * CELL


def _disk(radius_m):
    r = int(round(radius_m / CELL))
    y, x = np.mgrid[-r:r + 1, -r:r + 1]
    return x * x + y * y <= r * r


def carve_free(g: Grid, cloud: Cloud, cams_xy: np.ndarray, n_rays=300_000, stop=0.06, seed=0):
    """2D free-space carving: the plan projection of every camera->point ray is empty.

    Independent of whether floor or ceiling were scanned. Rays stop `stop` short
    of the hit so the hit surface itself is not carved.
    """
    count = np.zeros(g.shape[0] * g.shape[1], np.int32)
    if cloud.src is None:
        return count.reshape(g.shape)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(cloud.h), min(n_rays, len(cloud.h)), replace=False)
    a, b = cams_xy[cloud.src[idx]], cloud.xy[idx]
    L = np.linalg.norm(b - a, axis=1)
    ok = L > stop + CELL
    a, b, L = a[ok], b[ok], L[ok]
    steps = int(np.ceil(L.max() / (CELL / 2))) if len(L) else 0
    for i0 in range(0, len(L), 20_000):
        sa, sb, sl = a[i0:i0 + 20_000], b[i0:i0 + 20_000], L[i0:i0 + 20_000]
        t = np.linspace(0, 1, steps)[None, :]
        keep = t <= (1 - stop / sl)[:, None]
        pts = sa[:, None, :] + t[..., None] * (sb - sa)[:, None, :]
        pts = pts[np.broadcast_to(keep, pts.shape[:2])]
        r, c, ok2 = g.idx(pts)
        count += np.bincount(r[ok2] * g.shape[1] + c[ok2], minlength=count.size).astype(np.int32)
    return count.reshape(g.shape)


def segment_rooms(cloud: Cloud, ceiling: float, cams_xy: np.ndarray, door_max=0.95):
    g = Grid(cloud.xy)
    nh = cloud.n[:, 2]
    vert = np.abs(nh) < 0.3
    wall = g.count(cloud.xy[vert & (cloud.h > 1.0) & (cloud.h < 1.95)]) >= 3
    seen = g.count(cloud.xy[((nh > 0.9) & (np.abs(cloud.h) < 0.05)) |
                            ((nh < -0.9) & (np.abs(cloud.h - ceiling) < 0.08))]) >= 1
    soffit = g.count(cloud.xy[(nh < -0.9) & (cloud.h > 1.75) & (cloud.h < ceiling - 0.12)]) >= 2
    # door heads are thin strips (wall thickness x door width); broad low
    # surfaces are lowered ceilings, bulkheads or lofts and must not cut rooms
    soffit &= ~ndimage.binary_opening(soffit, _disk(0.20))

    free = ndimage.binary_closing(seen | (carve_free(g, cloud, cams_xy) >= 3), _disk(0.10), border_value=0)
    free = ndimage.binary_fill_holes(free)
    wall_d = ndimage.binary_dilation(wall, _disk(0.05))
    free &= ~wall_d

    cut = free & ~ndimage.binary_dilation(soffit, _disk(0.05))
    dist = ndimage.distance_transform_edt(cut) * CELL
    seeds, n = ndimage.label(dist > door_max / 2)
    # grow every seed back over free space; competing fronts meet mid-doorway
    labels = seeds.copy()
    struct = ndimage.generate_binary_structure(2, 1)
    for _ in range(400):
        grown = ndimage.grey_dilation(labels, footprint=struct)
        new = (labels == 0) & free & (grown > 0)
        if not new.any():
            break
        labels[new] = grown[new]
    # keep spaces the camera actually walked through; space only glimpsed
    # through an opening grew as its own seed and is dropped here
    r, c, ok = g.idx(cams_xy)
    visited = np.zeros(n + 1, int)
    np.add.at(visited, labels[r[ok], c[ok]], 1)
    sizes = ndimage.sum(np.ones_like(seeds), seeds, index=np.arange(n + 1)) * CELL ** 2
    keep = [i for i in range(1, n + 1) if visited[i] >= 3 and sizes[i] >= 0.15]
    lut = np.zeros(n + 1, labels.dtype)
    lut[keep] = np.arange(1, len(keep) + 1)
    labels = lut[labels]
    return g, labels, len(keep), wall


# --------------------------------------------------------------- outlines

def _trace(mask):
    """Outer boundary of a binary mask as CCW vertex loop in cell-corner (col,row) coords."""
    m = np.pad(mask, 1)
    edges = {}
    rows, cols = np.nonzero(m)
    for r, c in zip(rows, cols):
        if not m[r - 1, c]:
            edges.setdefault((c, r), []).append((c + 1, r))
        if not m[r, c + 1]:
            edges.setdefault((c + 1, r), []).append((c + 1, r + 1))
        if not m[r + 1, c]:
            edges.setdefault((c + 1, r + 1), []).append((c, r + 1))
        if not m[r, c - 1]:
            edges.setdefault((c, r + 1), []).append((c, r))
    loops = []
    while edges:
        start = next(iter(edges))
        loop, cur, prev_dir = [start], start, None
        while True:
            outs = edges[cur]
            if len(outs) > 1 and prev_dir is not None:
                # at a pinch vertex take the left-most turn to keep loops simple
                def turn(o):
                    d = (o[0] - cur[0], o[1] - cur[1])
                    return -(prev_dir[0] * d[1] - prev_dir[1] * d[0])
                outs.sort(key=turn)
            nxt = outs.pop(0)
            if not outs:
                del edges[cur]
            prev_dir = (nxt[0] - cur[0], nxt[1] - cur[1])
            cur = nxt
            if cur == start:
                break
            loop.append(cur)
        loops.append(np.array(loop, float) - 1)
    def area(p):
        x, y = p[:, 0], p[:, 1]
        return 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)
    return max(loops, key=area)


def _to_lines(loop, g: Grid):
    """Vertex loop -> alternating axis-aligned Lines (merging collinear runs)."""
    pts = np.stack([g.x(loop[:, 0]), g.y(loop[:, 1])], 1)
    lines = []
    n = len(pts)
    for i in range(n):
        a, b = pts[i], pts[(i + 1) % n]
        if abs(a[0] - b[0]) < 1e-9:      # vertical edge, plane X=c
            ln = Line("v", a[0], -1 if b[1] > a[1] else 1)
        else:
            ln = Line("h", a[1], 1 if b[0] > a[0] else -1)
        if lines and lines[-1].axis == ln.axis:
            continue
        lines.append(ln)
    if len(lines) > 1 and lines[0].axis == lines[-1].axis:
        lines.pop()
    return lines


def vertices(lines):
    """Corner k joins lines[k-1] and lines[k]."""
    out = []
    for k in range(len(lines)):
        a, b = lines[k - 1], lines[k]
        out.append((a.c, b.c) if a.axis == "v" else (b.c, a.c))
    return np.array(out)


def line_span(lines, k):
    """(start, end) along line k in traversal order."""
    return lines[k - 1].c, lines[(k + 1) % len(lines)].c


def remove_jogs(lines, tol=0.16):
    """Drop walls shorter than tol by merging their two parallel neighbours.

    Lines alternate v/h, so removing line k and fusing k-1 with k+1 keeps the
    alternation: the fused line's neighbours are k-2 and k+2, both perpendicular.
    """
    while len(lines) > 4:
        n = len(lines)
        lens = [abs(lines[(k + 1) % n].c - lines[k - 1].c) for k in range(n)]
        k = int(np.argmin(lens))
        if lens[k] >= tol:
            break
        i0, i2 = (k - 1) % n, (k + 1) % n
        a, b = lines[i0], lines[i2]
        la, lb = lens[i0] + 1e-6, lens[i2] + 1e-6
        merged = Line(a.axis, (a.c * la + b.c * lb) / (la + lb), a.inward if la >= lb else b.inward,
                      max(a.hw, b.hw), a.n_pts + b.n_pts)
        lines = [merged if i == i0 else lines[i] for i in range(n) if i not in (k, i2)]
    return lines


def _axis_vals(cloud, axis):
    """(normal coordinate, along-wall coordinate, normal component, along component)."""
    if axis == "v":
        return cloud.xy[:, 0], cloud.xy[:, 1], cloud.n[:, 0], cloud.n[:, 1]
    return cloud.xy[:, 1], cloud.xy[:, 0], cloud.n[:, 1], cloud.n[:, 0]


def refine_lines(lines, cloud: Cloud, ceiling: float, iters=2):
    """Fit each wall to its room-facing surface points; set its interval and support."""
    for _ in range(iters):
        new = []
        for k, ln in enumerate(lines):
            s0, s1 = sorted(line_span(lines, k))
            q, s, nq, _ = _axis_vals(cloud, ln.axis)
            sel = ((nq * ln.inward > 0.85) & (np.abs(q - ln.c) < 0.25) &
                   (s > s0 + 0.12) & (s < s1 - 0.12) & (cloud.h > WALL_FIT_MIN_H) & (cloud.h < ceiling - 0.12))
            v, vs = q[sel], s[sel]
            if len(v) < 30:
                new.append(Line(ln.axis, ln.c, ln.inward, UNSUPPORTED_HW, len(v), 0.0))
                continue
            # nearest surface on the room side of the current estimate wins
            m = np.median(v)
            inl = np.abs(v - m) < 0.03
            v, vs = v[inl], vs[inl]
            c = float(v.mean())
            se = v.std() / np.sqrt(min(len(v), N_EFF_MAX))
            nb = max(1, int((s1 - s0) / 0.1))
            hist, _ = np.histogram(vs, bins=nb, range=(s0, s1))
            ln2 = Line(ln.axis, c, ln.inward, Z90 * np.hypot(se, SIG_WALL_SYS), len(v), float((hist >= 2).mean()))
            if not ln2.supported:
                ln2.hw = UNSUPPORTED_HW
            new.append(ln2)
        lines = new
    return lines


def prune_unsupported(lines, cloud: Cloud, ceiling: float, min_area: float = 0.0, max_iter=60):
    """Remove outline sides that no wall surface backs, shortest first.

    The two parallel neighbours of a removed side merge onto the better
    supported one; on a tie the outer one, since unscanned floor next to a
    wall is usually furniture or occlusion rather than a missing room. A merge
    that would collapse the outline (area below `min_area`) is not taken.
    """
    lines = refine_lines(lines, cloud, ceiling)
    stuck = set()
    for _ in range(max_iter):
        n = len(lines)
        if n <= 4:
            break
        lens = [abs(lines[(k + 1) % n].c - lines[k - 1].c) for k in range(n)]
        cand = [k for k in range(n) if not lines[k].supported and (lines[k].axis, round(lines[k].c, 3)) not in stuck]
        if not cand:
            break
        k = min(cand, key=lambda j: lens[j])
        i0, i2 = (k - 1) % n, (k + 1) % n
        a, b = lines[i0], lines[i2]
        if a.supported != b.supported:
            order = [a, b] if a.supported else [b, a]
        elif a.supported and a.n_pts * a.cover != b.n_pts * b.cover:
            order = sorted([a, b], key=lambda x: -x.n_pts * x.cover)
        else:
            order = sorted([a, b], key=lambda x: x.c * x.inward)          # outer first
        for keep in order:
            merged = Line(keep.axis, keep.c, keep.inward, keep.hw, keep.n_pts, keep.cover)
            trial = [merged if i == i0 else lines[i] for i in range(n) if i not in (k, i2)]
            if polygon_area(vertices(trial)) >= min_area:
                lines = remove_jogs(refine_lines(trial, cloud, ceiling, iters=1), tol=0.10)
                break
        else:
            stuck.add((lines[k].axis, round(lines[k].c, 3)))
    return refine_lines(lines, cloud, ceiling)


# --------------------------------------------------------------- openings

def _edge(cloud_q, cloud_s, n_along, sel, s_guess, facing, fallback_hw):
    """Refine one opening edge on jamb points (normal along the wall, facing into the gap)."""
    jamb = sel & (n_along * facing > 0.85) & (np.abs(cloud_s - s_guess) < 0.06)
    s = cloud_s[jamb]
    if len(s) >= 15:
        m = np.median(s)
        s = s[np.abs(s - m) < 0.02]
        se = s.std() / np.sqrt(min(len(s), N_EFF_MAX))
        return float(s.mean()), Z90 * np.hypot(se, SIG_EDGE_SYS)
    return s_guess, fallback_hw


def measure_gap(cloud: Cloud, ln: Line, s_center: float, band=(0.3, 1.9), thickness=0.4):
    """Find the hole in a wall face around s_center; returns (s0, s1, hw0, hw1) or None."""
    q, s, nq, ns = _axis_vals(cloud, ln.axis)
    h = cloud.h
    face = (nq * ln.inward > 0.85) & (np.abs(q - ln.c) < 0.04) & (h > band[0]) & (h < band[1])
    fs = s[face]
    left, right = fs[fs < s_center], fs[fs > s_center]
    if not len(left) or not len(right):
        return None
    l0, r0 = left.max(), right.min()
    # spacing of face points near each edge bounds the gap-only estimate
    sp = 0.02
    in_wall = ((q - ln.c) * -ln.inward > -0.02) & ((q - ln.c) * -ln.inward < thickness) & \
              (h > band[0]) & (h < band[1])
    l, lhw = _edge(q, s, ns, in_wall, l0, +1, Z90 * np.hypot(sp, SIG_EDGE_SYS))
    r, rhw = _edge(q, s, ns, in_wall, r0, -1, Z90 * np.hypot(sp, SIG_EDGE_SYS))
    return l, r, lhw, rhw


def opening_height(cloud: Cloud, ln: Line, s0, s1, ceiling, thickness=0.4, from_h=1.5):
    """Head height of an opening: soffit (down-facing) points inside the wall over the gap."""
    q, s, nq, _ = _axis_vals(cloud, ln.axis)
    depth = (q - ln.c) * -ln.inward
    over = (s > s0 + 0.05) & (s < s1 - 0.05) & (depth > -0.03) & (depth < thickness)
    sof = cloud.h[over & (cloud.n[:, 2] < -0.9) & (cloud.h > from_h) & (cloud.h < ceiling - 0.05)]
    if len(sof) >= 10:
        lv, se = _refine_level(sof, np.median(sof), win=0.02)
        return lv, Z90 * np.hypot(se, SIG_LEVEL_SYS)
    face = cloud.h[over & (nq * ln.inward > 0.85) & (np.abs(q - ln.c) < 0.04) & (cloud.h > from_h)]
    if len(face) >= 10:
        return float(np.percentile(face, 2)), 0.03
    return None, None


def opening_sill(cloud: Cloud, ln: Line, s0, s1, below, thickness=0.4):
    """Sill level of a window: up-facing points inside the wall reveal under the gap."""
    q, s, _, _ = _axis_vals(cloud, ln.axis)
    depth = (q - ln.c) * -ln.inward
    over = (s > s0 + 0.05) & (s < s1 - 0.05) & (depth > -0.03) & (depth < thickness)
    up = cloud.h[over & (cloud.n[:, 2] > 0.9) & (cloud.h > 0.2) & (cloud.h < below - 0.2)]
    if len(up) < 10:
        return None, None
    lv, se = _refine_level(up, np.median(up), win=0.02)
    return lv, Z90 * np.hypot(se, SIG_LEVEL_SYS)


def _room_paths(rooms):
    from matplotlib.path import Path as MPath
    return [MPath(vertices(r.lines)) for r in rooms]


def find_doors(rooms, cloud: Cloud, ceiling: float, cams_xy: np.ndarray, look_ahead=3.0):
    """Doors where the walk crosses a room's wall line.

    The capture protocol walks through every doorway between captured rooms,
    so each crossing of a wall line by the camera path is a candidate. The
    opening itself is measured as the hole in that wall's surface points around
    the crossing; the room on the other side is the first room the path enters
    within `look_ahead` metres.
    """
    if not rooms:
        return []
    paths = _room_paths(rooms)
    inside = np.stack([p.contains_points(cams_xy) for p in paths], 1)     # (n_cams, n_rooms)
    step = np.r_[0, np.linalg.norm(np.diff(cams_xy, axis=0), axis=1)]
    walked = np.cumsum(step)
    adj = {}
    for ai, room in enumerate(rooms):
        for k, ln in enumerate(room.lines):
            s0, s1 = sorted(line_span(room.lines, k))
            q = cams_xy[:, 0] if ln.axis == "v" else cams_xy[:, 1]
            sv = cams_xy[:, 1] if ln.axis == "v" else cams_xy[:, 0]
            side = np.sign(q - ln.c)
            idx = np.nonzero(side[:-1] * side[1:] < 0)[0]
            crossings = []
            for i in idx:
                t = (ln.c - q[i]) / (q[i + 1] - q[i])
                sc = sv[i] + t * (sv[i + 1] - sv[i])
                if s0 - 0.05 < sc < s1 + 0.05:
                    # the camera is outside this room on one side of the crossing
                    out_j = i if (q[i] - ln.c) * ln.inward < 0 else i + 1
                    crossings.append((sc, i, out_j))
            done = []
            for sc, i, out_j in crossings:
                if any(a - 0.05 < sc < b + 0.05 for a, b in done):
                    continue
                gap = measure_gap(cloud, ln, sc)
                if gap is None or not MIN_OPENING <= gap[1] - gap[0] <= MAX_OPENING                         or not gap[0] - 0.05 < sc < gap[1] + 0.05:
                    continue
                done.append((gap[0], gap[1]))
                # follow the walk away from this room to find the neighbour
                direction = 1 if out_j == i + 1 else -1
                j, other = out_j, None
                while 0 <= j < len(cams_xy) and abs(walked[j] - walked[out_j]) < look_ahead:
                    hits = [r for r in np.nonzero(inside[j])[0] if r != ai]
                    if hits:
                        other = rooms[hits[0]]
                        break
                    if inside[j, ai] and j != out_j:
                        break
                    j += direction
                hgt, hhw = opening_height(cloud, ln, gap[0], gap[1], ceiling)
                if hgt is not None:
                    kind = "door" if hgt < ceiling - 0.1 else "open_passage"
                else:
                    kind = "door" if gap[1] - gap[0] <= MAX_DOOR_W else "open_passage"
                op = Opening(kind, k, *gap, height=hgt, height_hw=hhw,
                             other=other.id if other else None, id=f"{room.id}_o{len(room.openings) + 1}")
                room.openings.append(op)
                if other is not None:
                    adj.setdefault(frozenset((room.id, other.id)), (room.id, other.id, op.id))
    return list(adj.values())


def find_windows(room: Room, cloud: Cloud, ceiling: float, min_w=0.35):
    """Holes in a wall face with wall surface both below (sill) and above (head)."""
    for k, ln in enumerate(room.lines):
        s0, s1 = sorted(line_span(room.lines, k))
        q, s, nq, _ = _axis_vals(cloud, ln.axis)
        face = (nq * ln.inward > 0.85) & (np.abs(q - ln.c) < 0.04) & (s > s0) & (s < s1) & \
               (cloud.h > 0.15) & (cloud.h < ceiling - 0.1)
        if face.sum() < 50:
            continue
        fs, fh = s[face], cloud.h[face]
        nb = int(np.ceil((s1 - s0) / CELL))
        col = np.clip(((fs - s0) / CELL).astype(int), 0, nb - 1)
        mid = np.zeros(nb, bool)
        below = np.zeros(nb, bool)
        above = np.zeros(nb, bool)
        mid[col[(fh > 1.1) & (fh < 1.6)]] = True
        below[col[fh < 0.85]] = True
        above[col[fh > 1.95]] = True
        cand = ~mid & below & above
        runs, nr = ndimage.label(cand)
        taken = [(o.s0, o.s1) for o in room.openings if o.line == k]
        for ri in range(1, nr + 1):
            idx = np.nonzero(runs == ri)[0]
            if (idx[-1] - idx[0] + 1) * CELL < min_w:
                continue
            centre = s0 + (idx.mean() + 0.5) * CELL
            if any(a - 0.1 < centre < b + 0.1 for a, b in taken):
                continue
            cs = (fs > s0 + idx[0] * CELL) & (fs < s0 + (idx[-1] + 1) * CELL)
            sill = float(np.percentile(fh[cs & (fh < 1.1)], 98))
            head_guess = float(np.percentile(fh[cs & (fh > 1.6)], 2))
            gap = measure_gap(cloud, ln, centre, band=(sill + 0.1, head_guess - 0.1))
            if gap is None or gap[1] - gap[0] < min_w:
                continue
            head, head_hw = opening_height(cloud, ln, gap[0], gap[1], ceiling, from_h=sill + 0.3)
            sill_lv, sill_hw = opening_sill(cloud, ln, gap[0], gap[1], head if head else head_guess)
            if head is None:
                head, head_hw = head_guess, 0.03
            if sill_lv is None:
                sill_lv, sill_hw = sill, 0.03
            room.openings.append(Opening("window", k, *gap, height=head - sill_lv,
                                         height_hw=float(np.hypot(head_hw, sill_hw)),
                                         id=f"{room.id}_o{len(room.openings) + 1}"))


# ------------------------------------------------------------------- build

def room_level(cloud: Cloud, g: Grid, mask, floor_guess, ceil_guess):
    inner = ndimage.binary_erosion(mask, _disk(0.2))
    if inner.sum() < 10:
        inner = mask
    r, c, ok = g.idx(cloud.xy)
    inside = np.zeros(len(r), bool)
    inside[ok] = inner[r[ok], c[ok]]
    nh = cloud.n[:, 2]
    f, fse = _refine_level(cloud.h[inside & (nh > 0.95)], floor_guess)
    # each room's own ceiling: lowered/false ceilings differ from the global level
    down = cloud.h[inside & (nh < -0.95)]
    own = _peak(down, 1.9, ceil_guess + 0.3)
    c_, cse = _refine_level(down, own if own is not None else ceil_guess)
    return f, c_, Z90 * np.sqrt(fse ** 2 + cse ** 2 + 2 * SIG_LEVEL_SYS ** 2)


def build_plan(cloud: Cloud, cams_xy: np.ndarray, cam_frames: np.ndarray):
    """cloud and cams in plan coords (absolute heights).

    Returns (rooms, adjacency, theta, aligned cloud, floor level): theta and the
    floor level map world points into the frame the rooms live in.
    """
    vert = np.abs(cloud.n[:, 2]) < 0.3
    theta = manhattan_angle(cloud.n[vert, :2])
    cloud = Cloud(rotate(cloud.xy, theta), cloud.h.copy(),
                  np.concatenate([rotate(cloud.n[:, :2], theta), cloud.n[:, 2:]], 1), cloud.src)
    cams_xy = rotate(cams_xy, theta)

    floor, ceiling = floor_ceiling(cloud)
    cloud.h -= floor
    ceiling_seen = ceiling is not None
    if ceiling_seen:
        ceiling -= floor
    else:
        # ceiling never scanned: the top of the observed walls is a lower bound
        ceiling = float(np.percentile(cloud.h[vert & (cloud.h > 0.5)], 99.5)) + 0.02

    g, labels, n, _ = segment_rooms(cloud, ceiling, cams_xy)
    wall_pts = (np.abs(cloud.n[:, 2]) < 0.3) & (cloud.h > 0.1) & (cloud.h < ceiling - 0.12)
    rooms = []
    r, c, ok = g.idx(cams_xy)
    for i in range(1, n + 1):
        mask = labels == i
        sm = ndimage.binary_opening(mask, _disk(0.10))
        sm = ndimage.binary_fill_holes(ndimage.binary_closing(sm, _disk(0.10)))
        lab, nl = ndimage.label(sm)
        if nl == 0:
            continue
        sm = lab == (1 + np.argmax(ndimage.sum(sm, lab, range(1, nl + 1))))
        if sm.sum() * CELL ** 2 < 1.0:
            continue
        lines = remove_jogs(_to_lines(_trace(sm), g))
        v0 = vertices(lines)
        lo, hi = v0.min(0) - 0.5, v0.max(0) + 0.5
        near = wall_pts & np.all((cloud.xy > lo) & (cloud.xy < hi), axis=1)
        wc = Cloud(cloud.xy[near], cloud.h[near], cloud.n[near])
        mask_area = float(sm.sum()) * CELL ** 2
        traced = lines
        lines = prune_unsupported(lines, wc, ceiling, min_area=0.6 * mask_area)
        if polygon_area(vertices(lines)) < 0.6 * mask_area:
            # irregular space (typically a hall): keep the traced outline; its
            # unsupported sides already carry the wide interval
            lines = refine_lines(traced, wc, ceiling)
            if polygon_area(vertices(lines)) < 0.6 * mask_area:
                continue
        lines = remove_jogs(lines, tol=0.10)
        hit = ok.copy()
        hit[ok] = labels[r[ok], c[ok]] == i
        first = int(cam_frames[hit].min()) if hit.any() else 10 ** 9
        if ceiling_seen:
            f, cl, hhw = room_level(cloud, g, mask, 0.0, ceiling)
        else:
            f, cl, hhw = 0.0, ceiling - 0.02, 0.0
        rooms.append(Room("", "", mask, lines, floor=f, ceiling=cl, height_hw=hhw,
                          ceiling_seen=ceiling_seen, first_frame=first))

    # stable naming in walk order; long thin spaces are corridors
    order = sorted(range(len(rooms)), key=lambda j: rooms[j].first_frame)
    kept = []
    for new_i, j in enumerate(order, 1):
        room = rooms[j]
        room.id = f"room_{new_i}"
        w, d = np.ptp(vertices(room.lines), axis=0)
        room.name = ("Corridor " if min(w, d) < 1.8 and max(w, d) / max(min(w, d), 1e-6) > 2.5 else "Room ") + str(new_i)
        kept.append(room)
    adjacency = find_doors(kept, cloud, ceiling, cams_xy)
    for room in kept:
        find_windows(room, cloud, ceiling)
    return kept, adjacency, theta, cloud, floor


def polygon_area(v):
    x, y = v[:, 0], v[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))
