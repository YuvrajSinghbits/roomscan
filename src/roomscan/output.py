"""Rooms -> schema document, and schema document -> rendered plan.

Rendering reads only the JSON document, so every tier gets the same plan.
"""
from pathlib import Path

import numpy as np

from roomscan.plan2d import Z90, line_span, polygon_area, vertices
from roomscan.schema import measurement


def _hyp(*xs):
    return float(np.sqrt(sum(x * x for x in xs)))


def room_doc(room, origin):
    lines = room.lines
    n = len(lines)
    v = vertices(lines)
    walls = []
    for k in range(n):
        a, b = line_span(lines, k)
        walls.append({
            "id": f"{room.id}_w{k + 1}",
            "start": [round(float(x), 4) for x in v[k] - origin],
            "end": [round(float(x), 4) for x in v[(k + 1) % n] - origin],
            "length": measurement(abs(b - a), _hyp(lines[k - 1].hw, lines[(k + 1) % n].hw)),
        })
    area = polygon_area(v)
    # moving wall k by dc changes area by length_k * dc
    area_hw = _hyp(*[abs(line_span(lines, k)[1] - line_span(lines, k)[0]) * lines[k].hw for k in range(n)])
    openings = []
    for op in room.openings:
        start, end = line_span(lines, op.line)
        if end >= start:
            off, off_hw = op.s0 - start, _hyp(lines[op.line - 1].hw, op.hw0)
        else:
            off, off_hw = start - op.s1, _hyp(lines[op.line - 1].hw, op.hw1)
        o = {"id": op.id, "wall_id": f"{room.id}_w{op.line + 1}", "type": op.kind,
             "offset": measurement(off, off_hw), "width": measurement(op.s1 - op.s0, _hyp(op.hw0, op.hw1))}
        if op.height is not None:
            o["height"] = measurement(op.height, op.height_hw)
        if op.other:
            o["connects_to"] = op.other
        openings.append(o)
    return {
        "id": room.id,
        "name": room.name,
        "polygon": [[round(float(x), 4) for x in p - origin] for p in v],
        "walls": walls,
        "ceiling_height": _ceiling(room),
        "floor_area": measurement(area, area_hw, "m2"),
        "openings": openings,
    }


CEILING_UNSEEN_SPAN = 0.8


def _ceiling(room):
    h = room.ceiling - room.floor
    if room.ceiling_seen:
        return measurement(h, room.height_hw)
    return {"value": round(h, 4), "lo": round(h, 4), "hi": round(h + CEILING_UNSEEN_SPAN, 4), "unit": "m",
            "note": "ceiling not observed; value is a lower bound from the highest wall point"}


def assemble(capture: Path, tier: str, device: str, rooms, adjacency, drift: dict, render_name="plan.png"):
    all_v = np.concatenate([vertices(r.lines) for r in rooms])
    base = all_v.min(0)
    docs, placed = [], []
    for r in rooms:
        o = vertices(r.lines).min(0)
        docs.append(room_doc(r, o))
        placed.append({"room_id": r.id, "transform": [round(float(o[0] - base[0]), 4),
                                                       round(float(o[1] - base[1]), 4), 0.0]})
    areas = [d["floor_area"] for d in docs]
    total = sum(a["value"] for a in areas)
    total_hw = _hyp(*[(a["hi"] - a["value"]) for a in areas])
    return {
        "schema_version": "0.1",
        "capture": {"id": capture.stem, "tier": tier, "source": str(capture), "device": device},
        "rooms": docs,
        "stitched_plan": {
            "rooms": placed,
            "adjacency": [{"room_a": a, "room_b": b, "via_opening": o} for a, b, o in adjacency],
            "footprint_area": measurement(total, total_hw, "m2"),
            "drift_correction": drift["method"],
            "drift_detail": drift,
            "render": render_name,
        },
        "damage": [],
        "concealed_damage_flags": [],
        "scope": [],
    }


def _xf(t, pts):
    c, s = np.cos(t[2]), np.sin(t[2])
    p = np.asarray(pts, float)
    return p @ np.array([[c, s], [-s, c]]) + t[:2]


def render(doc: dict, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 10), dpi=130)
    placed = {p["room_id"]: np.array(p["transform"]) for p in doc["stitched_plan"]["rooms"]}
    colors = {"door": "#c0392b", "window": "#2e86de", "open_passage": "#27ae60"}
    for room in doc["rooms"]:
        t = placed[room["id"]]
        poly = _xf(t, room["polygon"])
        ax.fill(poly[:, 0], poly[:, 1], color="#f4efe6", zorder=1)
        c = poly.mean(0)
        a = room["floor_area"]
        ch = room["ceiling_height"]
        ax.text(c[0], c[1], f"{room['name']}\n{a['value']:.2f} m² ±{a['hi'] - a['value']:.2f}\n"
                f"h {ch['value']:.3f} ±{ch['hi'] - ch['value']:.3f}",
                ha="center", va="center", fontsize=8, zorder=5)
        walls = {w["id"]: w for w in room["walls"]}
        for w in room["walls"]:
            s, e = _xf(t, [w["start"], w["end"]])
            ax.plot([s[0], e[0]], [s[1], e[1]], color="#222", lw=3, solid_capstyle="projecting", zorder=2)
            m = (s + e) / 2
            d = (e - s) / (np.linalg.norm(e - s) + 1e-9)
            inward = np.array([-d[1], d[0]])         # CCW polygon: interior on the left
            L = w["length"]
            ax.text(*(m + 0.22 * inward), f"{L['value']:.2f}±{L['hi'] - L['value']:.2f}",
                    ha="center", va="center", fontsize=6.5, color="#444",
                    rotation=np.degrees(np.arctan2(d[1], d[0])) % 180 - (180 if 90 < np.degrees(np.arctan2(d[1], d[0])) % 360 <= 270 else 0),
                    zorder=5)
        for op in room["openings"]:
            w = walls[op["wall_id"]]
            s, e = _xf(t, [w["start"], w["end"]])
            d = (e - s) / (np.linalg.norm(e - s) + 1e-9)
            p0 = s + d * op["offset"]["value"]
            p1 = p0 + d * op["width"]["value"]
            ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color="white", lw=4, zorder=3)
            ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=colors[op["type"]], lw=1.8, zorder=4)
    ax.set_aspect("equal")
    ax.axis("off")
    fp = doc["stitched_plan"]["footprint_area"]
    ax.set_title(f"{doc['capture']['id']} [{doc['capture']['tier']}]  footprint {fp['value']:.2f} m² "
                 f"±{fp['hi'] - fp['value']:.2f}  drift: {doc['stitched_plan']['drift_correction']}",
                 fontsize=9)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
