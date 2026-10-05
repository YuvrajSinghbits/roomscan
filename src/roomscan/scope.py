"""Concealed-damage rules and scope line items, computed from the output document.

Both read only the JSON contract, so every tier gets the same logic.

Rules (each flag names the rule and the evidence that fired it):
  R1 ceiling_stain_over_wall   water stain on a ceiling and water damage/mold on a
                               wall of the same room -> leak path in that wall cavity
  R2 stain_at_wall_base        water stain or mold within 0.3 m of the floor ->
                               moisture in the floor / skirting below
  R3 mold_adjacent_walls       mold on a wall -> the walls meeting it at its corners
                               may hide growth behind the finish
  R4 ceiling_water             water stain on a ceiling -> plumbing/roof void above
  R5 long_crack                crack longer than 1 m -> structural check of that wall

Scope: one or more line items per damage region and per flag, quantities with
intervals propagated from the measured geometry.
"""
import numpy as np

from roomscan.schema import measurement


def _surfaces(doc):
    """surface_id -> dict(kind, room, length?, height?, area measurement)."""
    out = {}
    for room in doc["rooms"]:
        h = room["ceiling_height"]
        for k, w in enumerate(room["walls"]):
            L = w["length"]
            open_area = sum(o["width"]["value"] * (o["height"]["value"] if "height" in o else 2.0)
                            for o in room["openings"] if o["wall_id"] == w["id"])
            a = max(L["value"] * h["value"] - open_area, 0.0)
            rel = np.hypot((L["hi"] - L["value"]) / max(L["value"], 1e-6),
                           (h["hi"] - h["value"]) / max(h["value"], 1e-6))
            out[w["id"]] = {"kind": "wall", "room": room["id"], "index": k, "n": len(room["walls"]),
                            "area": measurement(a, rel * a, "m2")}
        out[f"{room['id']}_floor"] = {"kind": "floor", "room": room["id"], "area": room["floor_area"]}
        out[f"{room['id']}_ceiling"] = {"kind": "ceiling", "room": room["id"], "area": room["floor_area"]}
    return out


def concealed_flags(doc):
    surf = _surfaces(doc)
    dmg = doc["damage"]
    flags = []

    def add(sid, rule, ev):
        if sid in surf and not any(f["surface_id"] == sid and f["rule"] == rule for f in flags):
            flags.append({"surface_id": sid, "rule": rule, "evidence": ev})

    by_room = {}
    for d in dmg:
        s = surf.get(d["surface_id"])
        if s:
            by_room.setdefault(s["room"], []).append((d, s))
    for room, items in by_room.items():
        ceil_water = [d for d, s in items if s["kind"] == "ceiling" and d["class"] == "water_stain"]
        for d, s in items:
            if s["kind"] == "wall" and d["class"] in ("water_stain", "mold") and ceil_water:
                add(d["surface_id"], "R1_ceiling_stain_over_wall", [c["id"] for c in ceil_water] + [d["id"]])
            if s["kind"] == "wall" and d["class"] in ("water_stain", "mold") and d.get("extent") and d["extent"][1] < 0.3:
                add(f"{room}_floor", "R2_stain_at_wall_base", [d["id"]])
            if s["kind"] == "wall" and d["class"] == "mold":
                prefix = d["surface_id"].rsplit("_w", 1)[0]
                for nb in ((s["index"] - 1) % s["n"], (s["index"] + 1) % s["n"]):
                    add(f"{prefix}_w{nb + 1}", "R3_mold_adjacent_walls", [d["id"]])
            if s["kind"] == "ceiling" and d["class"] == "water_stain":
                add(d["surface_id"], "R4_ceiling_water", [d["id"]])
            if d["class"] == "crack" and d.get("extent"):
                e = d["extent"]
                if np.hypot(e[2] - e[0], e[3] - e[1]) > 1.0:
                    add(d["surface_id"], "R5_long_crack", [d["id"]])
    return flags


def _scale(m, f, unit=None):
    return {"value": round(m["value"] * f, 4), "lo": round(m["lo"] * f, 4), "hi": round(m["hi"] * f, 4),
            "unit": unit or m["unit"]}


def scope_items(doc):
    surf = _surfaces(doc)
    items = []

    def add(sid, code, desc, qty):
        items.append({"surface_id": sid, "code": code, "description": desc, "quantity": qty, "unit": qty["unit"]})

    for d in doc["damage"]:
        s = surf.get(d["surface_id"])
        if not s:
            continue
        sid, cls = d["surface_id"], d["class"]
        if cls == "water_stain":
            add(sid, "PNT-SEAL", f"Stain-block primer + 2 coats, whole {s['kind']} ({d['id']})", s["area"])
        elif cls == "mold":
            add(sid, "MLD-REM", f"Mould remediation incl. 0.3 m margin ({d['id']})", _scale(d["area"], 1.5))
            add(sid, "PNT-SEAL", f"Seal + repaint whole {s['kind']} after remediation ({d['id']})", s["area"])
        elif cls == "crack":
            e = d["extent"]
            L = float(np.hypot(e[2] - e[0], e[3] - e[1]))
            add(sid, "CRK-FILL", f"Rake out, fill and make good crack ({d['id']})", measurement(L, 0.1 + 0.1 * L))
            add(sid, "PNT-TOUCH", f"Repaint {s['kind']} after crack repair ({d['id']})", s["area"])
        elif cls == "hole":
            add(sid, "PATCH", f"Patch hole ({d['id']})", {"value": 1, "lo": 1, "hi": 1, "unit": "count"})
        elif cls == "peeling_paint":
            add(sid, "PNT-PREP", f"Scrape, prime and repaint whole {s['kind']} ({d['id']})", s["area"])
    one = {"value": 1, "lo": 1, "hi": 1, "unit": "count"}
    for f in doc["concealed_damage_flags"]:
        code = {"R1": "INSP-CAVITY", "R2": "INSP-MOIST", "R3": "INSP-MOLD", "R4": "INSP-VOID",
                "R5": "INSP-STRUCT"}[f["rule"][:2]]
        add(f["surface_id"], code, f"Inspection: {f['rule']} (evidence {', '.join(f['evidence'])})", one)
    return items


def apply(doc):
    doc["concealed_damage_flags"] = concealed_flags(doc)
    doc["scope"] = scope_items(doc)
    return doc
