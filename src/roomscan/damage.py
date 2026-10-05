"""Damage regions with class and metric extent, keyed to plan surfaces.

Detector: CLIPSeg (CIDAS/clipseg-rd64-refined, zero-shot text-prompted
segmentation, Apache-2.0). Every pixel is scored against each damage prompt and
against clean-surface prompts; a pixel is damage only where a damage prompt
beats every clean prompt and clears DAMAGE_THRESH, which keeps phantoms down on
clean rooms.

Extent: detected pixels are lifted to 3D with the view's depth and pose,
snapped to the surface they lie on (a room's wall plane, its floor or its
ceiling) and rasterised on a 5 cm grid in that surface's own (u, v) metres.
Detections of one class on one surface are merged across views (cell union).
"""
from dataclasses import dataclass

import numpy as np

from roomscan.plan2d import CELL, line_span, rotate
from roomscan.schema import measurement

MODEL_ID = "CIDAS/clipseg-rd64-refined"
PROMPTS = {
    "water_stain": "a brown water stain on a wall or ceiling",
    "crack": "a crack in a wall",
    "mold": "black mold on a wall",
    "hole": "a hole in a wall",
    "peeling_paint": "peeling paint on a wall",
}
CLEAN = ["a clean painted wall", "a clean ceiling", "a floor", "furniture", "a door", "a window"]
DAMAGE_THRESH = 0.60
MIN_PIXELS = 150          # at the 352x352 model resolution
MIN_AREA = 0.01           # m2
SURF_TOL = 0.10           # m from a surface plane
MIN_VIEWS = 2             # a real defect is seen from more than one position
# physically valid class/surface pairs (paint does not peel off floors)
VALID = {"wall": set(PROMPTS), "ceiling": {"water_stain", "crack", "mold", "peeling_paint", "hole"},
         "floor": {"water_stain"}}

_model = None


def _load():
    global _model
    if _model is None:
        from transformers import CLIPSegForImageSegmentation, CLIPSegProcessor
        from roomscan.depth_model import WEIGHTS
        _model = (CLIPSegProcessor.from_pretrained(MODEL_ID, cache_dir=WEIGHTS),
                  CLIPSegForImageSegmentation.from_pretrained(MODEL_ID, cache_dir=WEIGHTS).eval())
    return _model


def segment(rgb: np.ndarray):
    """-> dict class -> (H, W) bool mask at the image's resolution."""
    import torch
    from PIL import Image

    proc, model = _load()
    texts = list(PROMPTS.values()) + CLEAN
    img = Image.fromarray(rgb)
    with torch.no_grad():
        inp = proc(text=texts, images=[img] * len(texts), padding=True, return_tensors="pt")
        prob = torch.sigmoid(model(**inp).logits).numpy()        # (T, 352, 352)
    nd = len(PROMPTS)
    clean_best = prob[nd:].max(0)
    out = {}
    h, w = rgb.shape[:2]
    for i, cls in enumerate(PROMPTS):
        m = (prob[i] > DAMAGE_THRESH) & (prob[i] > clean_best) & (prob[i] == prob[:nd].max(0))
        if m.sum() >= MIN_PIXELS:
            ys = (np.arange(h) * m.shape[0] / h).astype(int)
            xs = (np.arange(w) * m.shape[1] / w).astype(int)
            out[cls] = (m[ys][:, xs], float(prob[i][m].mean()))
    return out


@dataclass
class View:
    rgb: np.ndarray          # HxWx3
    depth: np.ndarray        # hxw metres (any resolution)
    K: np.ndarray            # intrinsics at depth resolution
    pose: np.ndarray         # camera-to-world, ARKit axes


class SurfaceIndex:
    """Plan surfaces of the stitched rooms, in the build_plan frame."""

    def __init__(self, rooms, theta, floor):
        self.rooms, self.theta, self.floor = rooms, theta, floor

    def to_plan(self, pts_world):
        xy = rotate(np.stack([pts_world[:, 0], -pts_world[:, 2]], 1), self.theta)
        return xy, pts_world[:, 1] - self.floor

    def assign(self, xy, h):
        """-> list of (surface_id, u, v) arrays for points lying on a surface."""
        hits = {}
        for room in self.rooms:
            for k, ln in enumerate(room.lines):
                s0, s1 = line_span(room.lines, k)
                q, s = (xy[:, 0], xy[:, 1]) if ln.axis == "v" else (xy[:, 1], xy[:, 0])
                lo, hi = sorted((s0, s1))
                sel = (np.abs(q - ln.c) < SURF_TOL) & (s > lo) & (s < hi) & (h > 0.02) & (h < room.ceiling - room.floor - 0.02)
                if sel.any():
                    hits.setdefault(f"{room.id}_w{k + 1}", []).append((np.abs(s[sel] - s0), h[sel], np.nonzero(sel)[0]))
            from matplotlib.path import Path as MPath
            from roomscan.plan2d import vertices
            poly = vertices(room.lines)
            inside = MPath(poly).contains_points(xy)
            base = poly.min(0)
            for name, sel in (("floor", inside & (np.abs(h) < SURF_TOL)),
                              ("ceiling", inside & (np.abs(h - (room.ceiling - room.floor)) < SURF_TOL))):
                if sel.any():
                    hits.setdefault(f"{room.id}_{name}", []).append(
                        (xy[sel, 0] - base[0], xy[sel, 1] - base[1], np.nonzero(sel)[0]))
        return hits


def run(doc, views, rooms, theta, floor):
    """Fill doc['damage'], then the concealed-damage flags and scope derived from it."""
    from roomscan import scope
    doc["damage"] = detect(views, SurfaceIndex(rooms, theta, floor))
    return scope.apply(doc)


def detect(views, index: SurfaceIndex, max_views=20):
    """Run the detector over (a subset of) views; merge per class and surface."""
    if not views:
        return []
    pick = np.linspace(0, len(views) - 1, min(max_views, len(views))).astype(int)
    cells = {}       # (surface, class) -> set of (iu, iv); scores
    for vi in pick:
        v = views[vi]
        found = segment(v.rgb)
        if not found:
            continue
        dh, dw = v.depth.shape
        H, W = v.rgb.shape[:2]
        for cls, (mask, score) in found.items():
            md = mask[(np.arange(dh) * H / dh).astype(int)][:, (np.arange(dw) * W / dw).astype(int)]
            r, c = np.nonzero(md & (v.depth > 0.2))
            if len(r) < 5:
                continue
            z = v.depth[r, c]
            pc = np.stack([(c - v.K[0, 2]) / v.K[0, 0] * z, -(r - v.K[1, 2]) / v.K[1, 1] * z, -z], 1)
            pw = pc @ v.pose[:3, :3].T + v.pose[:3, 3]
            xy, h = index.to_plan(pw)
            for surf, parts in index.assign(xy, h).items():
                for u, vv, _ in parts:
                    key = (surf, cls)
                    entry = cells.setdefault(key, [set(), []])
                    entry[0].update(zip((u / CELL).astype(int), (vv / CELL).astype(int)))
                    entry[1].append(score)
    out = []
    for (surf, cls), (cs, scores) in sorted(cells.items()):
        area = len(cs) * CELL ** 2
        kind = "floor" if surf.endswith("_floor") else "ceiling" if surf.endswith("_ceiling") else "wall"
        if area < MIN_AREA or len(scores) < MIN_VIEWS or cls not in VALID[kind]:
            continue
        a = np.array(sorted(cs))
        # boundary cells are half in, half out: that is the extent uncertainty
        boundary = sum(1 for (i, j) in cs if any((i + di, j + dj) not in cs
                                                  for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1))))
        hw = 0.5 * boundary * CELL ** 2 + 0.2 * area
        out.append({
            "id": f"d{len(out) + 1}",
            "surface_id": surf,
            "class": cls,
            "area": measurement(area, hw, "m2"),
            "extent": [round(float(a[:, 0].min() * CELL), 3), round(float(a[:, 1].min() * CELL), 3),
                       round(float((a[:, 0].max() + 1) * CELL), 3), round(float((a[:, 1].max() + 1) * CELL), 3)],
            "confidence": round(float(np.mean(scores)), 3),
            "views": len(scores),
        })
    return out
