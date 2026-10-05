"""Photo tier: one folder per room, 2-8 stills each, no depth, no poses.

Built around the capture protocol (docs/capture_protocol.md):
  * corner photos: standing in a corner, photograph the opposite corner
  * door photos: whole door frame visible, from 1-2 m back

Per photo: metric depth (Depth Anything V2 metric indoor) -> points with
normals -> gravity from floor/ceiling normals -> Manhattan yaw. In that frame:

  * a corner photo sees two far walls; their distances from the camera plus the
    camera-to-corner offset (CORNER_OFFSET) give the room's two dimensions.
    Every corner photo is an independent estimate; their spread enters the
    interval together with the offset prior and the depth model's scale error.
  * a door photo has one dominant facing wall with a hole running from the
    floor to >= 1.8 m through which points lie behind the wall plane; the hole
    gives width and head height.

Stitching: the view through each doorway is matched (SIFT + RANSAC) against
the photos of every other room; the best match names the neighbour. Rooms are
then placed side by side across their shared door wall without overlaps.
Rooms are modelled as rectangles; a door's position along its wall is not
observable from one photo, so its offset carries the full wall as interval.
"""
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from roomscan.depth_model import predict
from roomscan.lidar_io import Frame, camera_points
from roomscan.output import assemble, render, widen
from roomscan.plan2d import Z90, Line, Opening, Room, manhattan_angle, rotate
from roomscan.tiers.video import _rot_to, _up_vector

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".heic", ".heif"}
WORK = 640
CORNER_OFFSET = 0.30        # camera to each back wall when standing in a corner, m
CORNER_OFFSET_SIG = 0.08
DEFAULT_F35 = 24.0          # iPhone 15 Pro 1x main camera, 35 mm equivalent
PHOTO_REL = 0.05            # depth-model scale error, 90% half-width (relative)
PHOTO_ABS = 0.02
WALL_T = 0.12               # assumed interior wall thickness when placing rooms


@dataclass
class Shot:
    path: Path
    rgb: np.ndarray
    kind: str = "other"
    dims: tuple | None = None         # (d1, d2) far-wall distances, corner photo
    floor: float | None = None
    ceiling: float | None = None
    door: dict | None = None
    door_px: tuple | None = None      # (u0, u1) image columns of the doorway
    extra: dict = field(default_factory=dict)


def _load(path: Path):
    from PIL import Image, ImageOps

    try:
        import pillow_heif
        pillow_heif.register_heif_opener()
    except ImportError:
        pass
    im = Image.open(path)
    f35 = None
    try:
        f35 = im.getexif().get_ifd(0x8769).get(41989)
    except Exception:
        pass
    im = ImageOps.exif_transpose(im).convert("RGB")
    s = WORK / max(im.size)
    im = im.resize((round(im.size[0] * s), round(im.size[1] * s)))
    return np.asarray(im), float(f35) if f35 else DEFAULT_F35


def _frame(rgb, f35):
    h, w = rgb.shape[:2]
    size = (256, 192) if w >= h else (192, 256)
    d = predict(rgb, size)
    f = f35 / 43.27 * np.hypot(w, h)          # 35 mm equivalent is defined on the diagonal
    K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1.0]])
    K[:2] *= size[0] / w
    return Frame(0, np.eye(4), K, d, None, None)


def _peak(vals, lo, hi, bin_=0.02, min_frac=0.15):
    """Strongest histogram peak in [lo, hi]; None if weak."""
    vals = vals[(vals > lo) & (vals < hi)]
    if len(vals) < 50:
        return None, 0
    hist, edges = np.histogram(vals, bins=np.arange(lo, hi + bin_, bin_))
    k = int(np.argmax(hist))
    sel = np.abs(vals - (edges[k] + bin_ / 2)) < 0.05
    return float(np.median(vals[sel])), int(sel.sum())


def analyse(shot: Shot, f35: float):
    fr = _frame(shot.rgb, f35)
    p, n = camera_points(fr, step=1)
    up = _up_vector([(None, n)])
    R = _rot_to(up)
    p, n = p @ R.T, n @ R.T
    # plan coordinates as in the LiDAR tier: X = x, Y = -z, h = y
    xy, hgt = np.stack([p[:, 0], -p[:, 2]], 1), p[:, 1]
    nxy, nh = np.stack([n[:, 0], -n[:, 2]], 1), n[:, 1]
    vert = np.abs(nh) < 0.3
    if vert.sum() < 200:
        return
    th = manhattan_angle(nxy[vert])
    xy, nxy = rotate(xy, th), rotate(nxy, th)
    f, _ = _peak(hgt[nh > 0.9], -3.0, -0.3)
    c, _ = _peak(hgt[nh < -0.9], 0.2, 3.0)
    shot.floor, shot.ceiling = f, c
    if f is not None:
        hgt = hgt - f

    # far walls on each half-axis: points whose normal faces back at the camera
    walls = {}
    for ax, sign in ((0, 1), (0, -1), (1, 1), (1, -1)):
        sel = vert & (nxy[:, ax] * sign < -0.85) & (xy[:, ax] * sign > 0.4)
        d, cnt = _peak(xy[sel, ax] * sign, 0.4, 8.0)
        if d is not None and cnt > 0.02 * vert.sum():
            walls[(ax, sign)] = (d, cnt)
    shot.extra["walls"] = walls

    door = _find_door(xy, hgt, nxy, vert, walls) if f is not None else None
    if door is not None:
        shot.kind, shot.door = "door", door
        return
    axes = {}
    for (ax, sign), (d, cnt) in walls.items():
        if ax not in axes or cnt > axes[ax][1]:
            axes[ax] = (d, cnt)
    if len(axes) == 2:
        shot.kind = "corner"
        shot.dims = (axes[0][0] + CORNER_OFFSET, axes[1][0] + CORNER_OFFSET)


def _find_door(xy, h, nxy, vert, walls):
    """Doorway in the dominant facing wall: a floor-to->=1.8 m hole with points behind it."""
    if not walls:
        return None
    (ax, sign), (d, cnt) = max(walls.items(), key=lambda kv: kv[1][1])
    q = xy[:, ax] * sign
    s = xy[:, 1 - ax]
    face = vert & (np.abs(q - d) < 0.05)
    behind = q > d + 0.25
    if face.sum() < 300 or behind.sum() < 100:
        return None
    lo, hi = np.percentile(s[face], [1, 99])
    bins = np.arange(lo, hi, 0.05)
    if len(bins) < 6:
        return None
    col_face = np.histogram(s[face & (h > 0.3) & (h < 1.8)], bins)[0]
    col_behind = np.histogram(s[behind & (h > 0.2) & (h < 1.8)], bins)[0]
    hole = (col_face < 3) & (col_behind > 5)
    best, run = None, []
    for i, v in enumerate(np.r_[hole, False]):
        if v:
            run.append(i)
        elif run:
            if best is None or len(run) > len(best):
                best = run
            run = []
    if best is None or len(best) * 0.05 < 0.5:
        return None
    c0, c1 = bins[best[0]], bins[best[-1] + 1]
    fs = s[face & (h > 0.3) & (h < 1.8)]
    left, right = fs[fs < (c0 + c1) / 2], fs[fs > (c0 + c1) / 2]
    if not len(left) or not len(right):
        return None
    s0, s1 = float(left.max()), float(right.min())
    if not 0.5 <= s1 - s0 <= 2.6:
        return None
    head = h[face & (s > s0 + 0.05) & (s < s1 - 0.05) & (h > 1.5)]
    return {"width": s1 - s0, "height": float(np.percentile(head, 2)) if len(head) > 20 else None,
            "distance": d}


# ------------------------------------------------------------------ stitching

def _sift():
    import cv2
    return cv2.SIFT_create(1500)


def _match_count(a, b):
    import cv2
    if a[1] is None or b[1] is None or len(a[0]) < 8 or len(b[0]) < 8:
        return 0
    m = cv2.BFMatcher().knnMatch(a[1], b[1], k=2)
    good = [x for x, y in (p for p in m if len(p) == 2) if x.distance < 0.75 * y.distance]
    if len(good) < 12:
        return 0
    pa = np.float32([a[0][x.queryIdx].pt for x in good])
    pb = np.float32([b[0][x.trainIdx].pt for x in good])
    _, inl = cv2.findFundamentalMat(pa, pb, cv2.FM_RANSAC, 2.0, 0.99)
    return int(inl.sum()) if inl is not None else 0


def neighbours(rooms_shots):
    """For every door photo, the other room whose photos best match the view through it."""
    import cv2
    sift = _sift()
    feats = {}
    for name, shots in rooms_shots.items():
        for sh in shots:
            g = cv2.cvtColor(sh.rgb, cv2.COLOR_RGB2GRAY)
            feats[id(sh)] = sift.detectAndCompute(g, None)
    links = []
    for name, shots in rooms_shots.items():
        for sh in shots:
            if sh.kind != "door":
                continue
            # keep only features in the central band, where the doorway is framed
            kp, des = feats[id(sh)]
            w = sh.rgb.shape[1]
            keep = [i for i, k in enumerate(kp) if 0.2 * w < k.pt[0] < 0.8 * w]
            mine = ([kp[i] for i in keep], des[keep] if des is not None and keep else None)
            best, score = None, 0
            for other, oshots in rooms_shots.items():
                if other == name:
                    continue
                sc = max((_match_count(mine, feats[id(o)]) for o in oshots), default=0)
                if sc > score:
                    best, score = other, sc
            sh.extra["neighbour"] = best if score >= 20 else None
            sh.extra["match_inliers"] = score
            links.append((name, sh))
    return links


def _rect_lines(x0, y0, x1, y1, hw):
    return [Line("h", y0, 1, hw, 999, 1.0), Line("v", x1, -1, hw, 999, 1.0),
            Line("h", y1, -1, hw, 999, 1.0), Line("v", x0, 1, hw, 999, 1.0)]


def _overlap(a, b, eps=0.01):
    return a[0] < b[2] - eps and b[0] < a[2] - eps and a[1] < b[3] - eps and b[1] < a[3] - eps


def layout(sizes, edges):
    """Place rectangles: BFS over door links, each new room against its placed neighbour."""
    names = list(sizes)
    placed = {names[0]: (0.0, 0.0, *sizes[names[0]])}
    side_used = {}
    order = [names[0]]
    pending = set(names[1:])
    while pending:
        progress = False
        for a, b in edges:
            for src, dst in ((a, b), (b, a)):
                if src in placed and dst in pending:
                    x0, y0, x1, y1 = placed[src]
                    W, L = sizes[dst][2], sizes[dst][3]
                    best = None
                    for side in ("E", "N", "W", "S"):
                        for (w_, l_) in ((W, L), (L, W)):
                            if side == "E":
                                r = (x1 + WALL_T, (y0 + y1 - l_) / 2, x1 + WALL_T + w_, (y0 + y1 + l_) / 2)
                            elif side == "W":
                                r = (x0 - WALL_T - w_, (y0 + y1 - l_) / 2, x0 - WALL_T, (y0 + y1 + l_) / 2)
                            elif side == "N":
                                r = ((x0 + x1 - w_) / 2, y1 + WALL_T, (x0 + x1 + w_) / 2, y1 + WALL_T + l_)
                            else:
                                r = ((x0 + x1 - w_) / 2, y0 - WALL_T - l_, (x0 + x1 + w_) / 2, y0 - WALL_T)
                            if any(_overlap(r, p) for p in placed.values()):
                                continue
                            allr = list(placed.values()) + [r]
                            bb = (max(p[2] for p in allr) - min(p[0] for p in allr)) * \
                                 (max(p[3] for p in allr) - min(p[1] for p in allr))
                            if best is None or bb < best[0]:
                                best = (bb, r, side)
                    if best:
                        placed[dst] = best[1]
                        side_used[(src, dst)] = best[2]
                        pending.discard(dst)
                        order.append(dst)
                        progress = True
        if not progress:
            # unconnected room: put it east of everything
            dst = sorted(pending)[0]
            xmax = max(p[2] for p in placed.values())
            W, L = sizes[dst][2], sizes[dst][3]
            placed[dst] = (xmax + 1.0, 0.0, xmax + 1.0 + W, L)
            pending.discard(dst)
            order.append(dst)
    return placed, side_used


SIDE_LINE = {"S": 0, "E": 1, "N": 2, "W": 3}
OPPOSITE = {"E": "W", "W": "E", "N": "S", "S": "N"}


def process(capture: Path, out_dir: Path, **_) -> dict:
    capture = Path(capture)
    room_dirs = sorted(p for p in capture.iterdir() if p.is_dir())
    if not room_dirs:
        room_dirs = [capture]
    rooms_shots = {}
    for rd in room_dirs:
        shots = []
        for p in sorted(rd.iterdir()):
            if p.suffix.lower() in IMAGE_EXT:
                rgb, f35 = _load(p)
                sh = Shot(p, rgb)
                analyse(sh, f35)
                shots.append(sh)
        if shots:
            rooms_shots[rd.name] = shots

    # per-room dimensions and ceiling
    sizes, info = {}, {}
    for name, shots in rooms_shots.items():
        corners = [s.dims for s in shots if s.dims]
        if not corners:
            # no usable corner photo: fall back to the farthest walls seen at all
            ds = sorted((d for s in shots for (d, _) in s.extra.get("walls", {}).values()), reverse=True)
            corners = [(ds[0] + CORNER_OFFSET, ds[1] + CORNER_OFFSET)] if len(ds) >= 2 else [(3.0, 3.0)]
            spread_floor = 0.5
        else:
            spread_floor = 0.0
        small = np.array([min(c) for c in corners])
        large = np.array([max(c) for c in corners])
        W, L = float(np.median(small)), float(np.median(large))
        n = len(corners)

        def hw(v):
            sp = np.std(v) / np.sqrt(n) if n > 1 else 0.10
            return float(Z90 * np.sqrt(sp ** 2 + CORNER_OFFSET_SIG ** 2) + spread_floor)
        hs = [s.ceiling - s.floor for s in shots if s.ceiling is not None and s.floor is not None]
        info[name] = {"W": W, "L": L, "hwW": hw(small), "hwL": hw(large), "n_corner": n,
                      "heights": hs, "kinds": [s.kind for s in shots]}
        sizes[name] = (0, 0, W, L)

    links = neighbours(rooms_shots)
    edges = sorted({tuple(sorted((a, sh.extra["neighbour"]))) for a, sh in links if sh.extra.get("neighbour")})
    placed, sides = layout(sizes, edges)

    rooms, adjacency = [], []
    ids = {name: f"room_{i + 1}" for i, name in enumerate(rooms_shots)}
    by_name = {}
    for name in rooms_shots:
        x0, y0, x1, y1 = placed[name]
        inf = info[name]
        line_hw = max(inf["hwW"], inf["hwL"]) / np.sqrt(2)
        hs = inf["heights"]
        room = Room(ids[name], name, None, _rect_lines(x0, y0, x1, y1, line_hw))
        if hs:
            room.floor, room.ceiling = 0.0, float(np.median(hs))
            room.height_hw = float(Z90 * np.sqrt((np.std(hs) / np.sqrt(len(hs)) if len(hs) > 1 else 0.05) ** 2))
        else:
            room.floor, room.ceiling, room.ceiling_seen = 0.0, 2.4, False
        rooms.append(room)
        by_name[name] = room

    # doors: on the wall facing the neighbour (or the first wall when unknown)
    done_pairs = {}
    for name, sh in links:
        room = by_name[name]
        nb = sh.extra.get("neighbour")
        side = sides.get((name, nb)) or (OPPOSITE[sides[(nb, name)]] if (nb, name) in sides else "S")
        k = SIDE_LINE[side]
        s_a, s_b = sorted([room.lines[k - 1].c, room.lines[(k + 1) % 4].c])
        wdt = sh.door["width"]
        mid = (s_a + s_b) / 2
        free = max((s_b - s_a - wdt) / 2, 0.0)
        op = Opening("door" if sh.door["height"] is not None else "open_passage", k,
                     mid - wdt / 2, mid + wdt / 2, 0.03, 0.03,
                     height=sh.door["height"], height_hw=0.05 if sh.door["height"] else None,
                     other=ids.get(nb), id=f"{room.id}_o{len(room.openings) + 1}")
        op.offset_override = (free, max(free, 0.05))    # position along the wall is not observed
        room.openings.append(op)
        if nb:
            key = frozenset((name, nb))
            if key not in done_pairs:
                done_pairs[key] = True
                adjacency.append((room.id, ids[nb], op.id))

    doc = assemble(capture, "photo", "iPhone (photos, no depth)", rooms, adjacency,
                   {"method": "not applicable: independent stills, rooms placed via door-view matching",
                    "photo_analysis": {n: {k: v for k, v in info[n].items()} for n in info}})
    widen(doc, PHOTO_REL, PHOTO_ABS)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    render(doc, out_dir / "plan.png")
    return doc
