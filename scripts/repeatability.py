"""Repeatability gate: two captures of the same property at the same tier.

    python scripts/repeatability.py out/lidar_floor_only/result.json out/lidar_with_ceiling/result.json

The two plans are aligned (best of 4 quarter-turns x translations that put a
room centre of B on a room centre of A, scored by room-overlap area), rooms are
matched by bounding-box IoU, and every wall measured in *both* captures (90%
half-width <= 5 cm, i.e. backed by scanned surface) is compared with the wall
of the same orientation nearest to it. Gate: |difference| <= max(1 cm, 0.5%).
"""
import json
import sys

import numpy as np

MEASURED_HW = 0.05


def rooms_in_property(doc):
    out = {}
    tf = {p["room_id"]: p["transform"] for p in doc["stitched_plan"]["rooms"]}
    for r in doc["rooms"]:
        t = tf[r["id"]]
        c, s = np.cos(t[2]), np.sin(t[2])
        R = np.array([[c, -s], [s, c]])
        poly = np.array(r["polygon"]) @ R.T + t[:2]
        walls = []
        for w in r["walls"]:
            a, b = np.array(w["start"]) @ R.T + t[:2], np.array(w["end"]) @ R.T + t[:2]
            walls.append((a, b, w["length"]))
        out[r["id"]] = (poly, walls)
    return out


def _rot(k):
    return np.linalg.matrix_power(np.array([[0, -1], [1, 0]]), k)


def _bbox(p):
    return np.r_[p.min(0), p.max(0)]


def _overlap(a, b):
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    return max(w, 0) * max(h, 0)


def align(A, B):
    best = (-1, None)
    for k in range(4):
        R = _rot(k)
        Bk = {i: (p @ R.T, [(a @ R.T, b @ R.T, L) for a, b, L in w]) for i, (p, w) in B.items()}
        for pa, _ in A.values():
            for pb, _ in Bk.values():
                t = pa.mean(0) - pb.mean(0)
                score = sum(_overlap(_bbox(qa), _bbox(qb + t)) for qa, _ in A.values() for qb, _ in Bk.values())
                if score > best[0]:
                    best = (score, (k, t))
    k, t = best[1]
    R = _rot(k)
    return {i: (p @ R.T + t, [(a @ R.T + t, b @ R.T + t, L) for a, b, L in w]) for i, (p, w) in B.items()}


def main(pa, pb):
    A = rooms_in_property(json.load(open(pa)))
    B = align(A, rooms_in_property(json.load(open(pb))))
    rows = []
    for ia, (qa, wa) in A.items():
        ba = _bbox(qa)
        ious = {ib: _overlap(ba, _bbox(qb)) / (np.prod(ba[2:] - ba[:2]) + np.prod(_bbox(qb)[2:] - _bbox(qb)[:2])
                                               - _overlap(ba, _bbox(qb)))
                for ib, (qb, _) in B.items()}
        ib = max(ious, key=ious.get)
        if ious[ib] < 0.5:
            continue
        for a0, a1, La in wa:
            if La["hi"] - La["value"] > MEASURED_HW:
                continue
            horiz = abs(a1[1] - a0[1]) < 1e-6
            mid = (a0 + a1) / 2
            cands = [(np.linalg.norm((b0 + b1) / 2 - mid), Lb) for b0, b1, Lb in B[ib][1]
                     if (abs(b1[1] - b0[1]) < 1e-6) == horiz and Lb["hi"] - Lb["value"] <= MEASURED_HW]
            if not cands:
                continue
            d, Lb = min(cands, key=lambda x: x[0])
            if d > 0.5:
                continue
            diff = Lb["value"] - La["value"]
            gate = max(0.01, 0.005 * La["value"])
            row = (ia, ib, La["value"], Lb["value"], diff, abs(diff) <= gate)
            if row not in rows:    # opposite sides of a rectangle share one measurement
                rows.append(row)
    print("| room A | room B | wall A m | wall B m | difference cm | within ±max(1 cm, 0.5%) |")
    print("|---|---|---|---|---|---|")
    for ia, ib, a, b, d, ok in rows:
        print(f"| {ia} | {ib} | {a:.3f} | {b:.3f} | {100 * d:+.1f} | {'yes' if ok else 'NO'} |")
    if rows:
        diffs = np.abs([r[4] for r in rows])
        print(f"\n{sum(r[5] for r in rows)}/{len(rows)} walls within the gate; "
              f"median |difference| {100 * np.median(diffs):.1f} cm, max {100 * diffs.max():.1f} cm.")


if __name__ == "__main__":
    main(*sys.argv[1:3])
