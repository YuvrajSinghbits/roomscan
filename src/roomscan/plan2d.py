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


@dataclass
class Cloud:
    xy: np.ndarray       # (N,2) plan coordinates
    h: np.ndarray        # (N,) height (absolute until floor is known)
    n: np.ndarray        # (N,3) normals (nX, nY, nh), facing the camera


@dataclass
class Line:
    axis: str            # "v": plane X=c ; "h": plane Y=c
    c: float
    inward: int          # +1 / -1: direction of room interior along the normal axis
    hw: float = 0.05     # 90% half-width of c
    n_pts: int = 0


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
    if ceil_guess is None:
        raise ValueError("no ceiling surface observed")
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


def segment_rooms(cloud: Cloud, ceiling: float, cams_xy: np.ndarray, door_max=0.95):
    g = Grid(cloud.xy)
    nh = cloud.n[:, 2]
    vert = np.abs(nh) < 0.3
    wall = g.count(cloud.xy[vert & (cloud.h > 1.0) & (cloud.h < 1.95)]) >= 3
    seen = g.count(cloud.xy[((nh > 0.9) & (np.abs(cloud.h) < 0.05)) |
                            ((nh < -0.9) & (np.abs(cloud.h - ceiling) < 0.08))]) >= 1
    soffit = g.count(cloud.xy[(nh < -0.9) & (cloud.h > 1.75) & (cloud.h < ceiling - 0.12)]) >= 2

    free = ndimage.binary_closing(seen, _disk(0.10), border_value=0)
    free = ndimage.binary_fill_holes(free)
    wall_d = ndimage.binary_dilation(wall, _disk(0.05))
    free &= ~wall_d

    cut = free & ~ndimage.binary_dilation(soffit, _disk(0.05))
    dist = ndimage.distance_transform_edt(cut) * CELL
    seeds, n = ndimage.label(dist > door_max / 2)
    # keep seeds the camera actually walked through (rooms only glimpsed through a door are dropped)
    r, c, ok = g.idx(cams_xy)
    visited = np.zeros(n + 1, int)
    np.add.at(visited, seeds[r[ok], c[ok]], 1)
    sizes = ndimage.sum(np.ones_like(seeds), seeds, index=np.arange(n + 1)) * CELL ** 2
    keep = [i for i in range(1, n + 1) if visited[i] >= 3 and sizes[i] >= 0.15]
    labels = np.zeros_like(seeds)
    for k, i in enumerate(keep, 1):
        labels[seeds == i] = k
    # grow seeds back over free space; competing fronts meet mid-doorway
    struct = ndimage.generate_binary_structure(2, 1)
    for _ in range(200):
        grown = ndimage.grey_dilation(labels, footprint=struct)
        new = (labels == 0) & free & (grown > 0)
        if not new.any():
            break
        labels[new] = grown[new]
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
    """Fit each wall to its room-facing surface points; set its interval."""
    for _ in range(iters):
        new = []
        for k, ln in enumerate(lines):
            s0, s1 = sorted(line_span(lines, k))
            q, s, nq, _ = _axis_vals(cloud, ln.axis)
            sel = ((nq * ln.inward > 0.85) & (np.abs(q - ln.c) < 0.25) &
                   (s > s0 + 0.12) & (s < s1 - 0.12) & (cloud.h > 0.1) & (cloud.h < ceiling - 0.12))
            v = q[sel]
            if len(v) < 30:
                new.append(Line(ln.axis, ln.c, ln.inward, max(ln.hw, 0.05), len(v)))
                continue
            # nearest surface on the room side of the current estimate wins
            m = np.median(v)
            v = v[np.abs(v - m) < 0.03]
            c = float(v.mean())
            se = v.std() / np.sqrt(min(len(v), N_EFF_MAX))
            new.append(Line(ln.axis, c, ln.inward, Z90 * np.hypot(se, SIG_WALL_SYS), len(v)))
        lines = new
    return lines


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


def find_doors(rooms, g: Grid, labels, cloud: Cloud, ceiling: float):
    adj = []
    by_label = {i + 1: r for i, r in enumerate(rooms)}
    n = len(rooms)
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            touch = (labels == a) & (
                ndimage.binary_dilation(labels == b, ndimage.generate_binary_structure(2, 1)))
            if touch.sum() < 3:
                continue
            clusters, nc = ndimage.label(ndimage.binary_dilation(touch, iterations=2))
            for ci in range(1, nc + 1):
                rr, cc = np.nonzero(touch & (clusters == ci))
                if len(rr) < 3:
                    continue
                centre = np.array([g.x(cc.mean() + 0.5), g.y(rr.mean() + 0.5)])
                ops = []
                for room in (by_label[a], by_label[b]):
                    k = _nearest_line(room.lines, centre)
                    if k is None:
                        break
                    ln = room.lines[k]
                    sc = centre[1] if ln.axis == "v" else centre[0]
                    gap = measure_gap(cloud, ln, sc)
                    if gap is None:
                        break
                    hgt, hhw = opening_height(cloud, ln, gap[0], gap[1], ceiling)
                    kind = "door" if hgt is not None and hgt < ceiling - 0.1 else "open_passage"
                    ops.append((room, Opening(kind, k, *gap, height=hgt, height_hw=hhw)))
                if len(ops) != 2:
                    continue
                (ra, oa), (rb, ob) = ops
                oa.other, ob.other = rb.id, ra.id
                for room, op in ops:
                    op.id = f"{room.id}_o{len(room.openings) + 1}"
                    room.openings.append(op)
                adj.append((ra.id, rb.id, oa.id))
    return adj


def _nearest_line(lines, p, max_d=0.4):
    best, bd = None, max_d
    for k, ln in enumerate(lines):
        s0, s1 = sorted(line_span(lines, k))
        q, s = (p[0], p[1]) if ln.axis == "v" else (p[1], p[0])
        if s0 - 0.1 <= s <= s1 + 0.1 and abs(q - ln.c) < bd:
            best, bd = k, abs(q - ln.c)
    return best


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
            if gap is None:
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
    c_, cse = _refine_level(cloud.h[inside & (nh < -0.95)], ceil_guess)
    return f, c_, Z90 * np.sqrt(fse ** 2 + cse ** 2 + 2 * SIG_LEVEL_SYS ** 2)


def build_plan(cloud: Cloud, cams_xy: np.ndarray, cam_frames: np.ndarray):
    """cloud and cams in plan coords (absolute heights). Returns (rooms, adjacency, theta)."""
    vert = np.abs(cloud.n[:, 2]) < 0.3
    theta = manhattan_angle(cloud.n[vert, :2])
    cloud = Cloud(rotate(cloud.xy, theta), cloud.h.copy(),
                  np.concatenate([rotate(cloud.n[:, :2], theta), cloud.n[:, 2:]], 1))
    cams_xy = rotate(cams_xy, theta)

    floor, ceiling = floor_ceiling(cloud)
    cloud.h -= floor
    ceiling -= floor

    g, labels, n, _ = segment_rooms(cloud, ceiling, cams_xy)
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
        lines = refine_lines(lines, cloud, ceiling)
        lines = remove_jogs(lines, tol=0.10)
        hit = ok.copy()
        hit[ok] = labels[r[ok], c[ok]] == i
        first = int(cam_frames[hit].min()) if hit.any() else 10 ** 9
        f, cl, hhw = room_level(cloud, g, mask, 0.0, ceiling)
        rooms.append(Room("", "", mask, lines, floor=f, ceiling=cl, height_hw=hhw, first_frame=first))

    # stable naming in walk order; long thin spaces are corridors
    order = sorted(range(len(rooms)), key=lambda j: rooms[j].first_frame)
    relabel = np.zeros(n + 1, int)
    kept = []
    for new_i, j in enumerate(order, 1):
        room = rooms[j]
        room.id = f"room_{new_i}"
        w, d = np.ptp(vertices(room.lines), axis=0)
        room.name = ("Corridor " if min(w, d) < 1.8 and max(w, d) / max(min(w, d), 1e-6) > 2.5 else "Room ") + str(new_i)
        kept.append(room)
    # labels must match the kept room order for door detection
    for new_i, room in enumerate(kept, 1):
        relabel_mask = room.mask
        labels = np.where(relabel_mask, -new_i, labels)
    labels = np.where(labels < 0, -labels, 0)

    adjacency = find_doors(kept, g, labels, cloud, ceiling)
    for room in kept:
        find_windows(room, cloud, ceiling)
    return kept, adjacency, theta, cloud


def polygon_area(v):
    x, y = v[:, 0], v[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))
