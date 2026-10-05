"""End-to-end LiDAR tier on a synthetic export with known geometry and injected drift.

The gate numbers mirror the case-study gates (openings <= 2 cm, ceiling
<= 1.5 cm); walls use 1 cm. Synthetic data only proves the geometry and the
drift handling are wired correctly -- real-capture accuracy is measured on the
benchmark set against tape/laser ground truth.
"""
import numpy as np
import pytest

from roomscan.cli import run
from synthetic import TRUTH, make_capture


@pytest.fixture(scope="session")
def capture(tmp_path_factory):
    root = tmp_path_factory.mktemp("syn") / "syn_lidar"
    make_capture(root, drift=True)
    return root


@pytest.fixture(scope="session")
def result(capture, tmp_path_factory):
    return run(capture, "lidar", tmp_path_factory.mktemp("out"))


@pytest.fixture(scope="session")
def ablation(capture, tmp_path_factory):
    return run(capture, "lidar", tmp_path_factory.mktemp("out_nodrift"), drift=False)


def _dims(room):
    p = np.array(room["polygon"])
    return sorted(np.ptp(p, axis=0))


def _truth_dims():
    return {k: sorted([x1 - x0, y1 - y0]) for k, (x0, x1, y0, y1) in TRUTH["rooms"].items()}


def _match(result):
    truth = _truth_dims()
    out = {}
    for room in result["rooms"]:
        d = _dims(room)
        name = min(truth, key=lambda k: np.abs(np.subtract(truth[k], d)).sum())
        out[name] = room
    return out


def test_all_rooms_found_and_rectangular(result):
    assert len(result["rooms"]) == len(TRUTH["rooms"])
    assert set(_match(result)) == set(TRUTH["rooms"])
    for room in result["rooms"]:
        assert len(room["walls"]) == 4


def test_wall_lengths(result):
    truth = _truth_dims()
    for name, room in _match(result).items():
        for w in room["walls"]:
            L = w["length"]
            err = min(abs(L["value"] - t) for t in truth[name])
            assert err < 0.01, (name, w["id"], L)


def test_ceiling_height(result):
    for room in result["rooms"]:
        assert abs(room["ceiling_height"]["value"] - TRUTH["ceiling_height"]) <= 0.015


def test_doors_and_adjacency(result):
    rooms = _match(result)
    ids = {r["id"]: n for n, r in rooms.items()}
    pairs = {frozenset((ids[a["room_a"]], ids[a["room_b"]])) for a in result["stitched_plan"]["adjacency"]}
    assert pairs == {frozenset(d[:2]) for d in TRUTH["doors"]}
    for a, b, width, height in TRUTH["doors"]:
        doors = [o for o in rooms[a]["openings"] if o["type"] == "door" and ids[o["connects_to"]] == b]
        assert len(doors) == 1
        assert abs(doors[0]["width"]["value"] - width) <= 0.02
        assert abs(doors[0]["height"]["value"] - height) <= 0.02


def test_window(result):
    rooms = _match(result)
    for name, width, height in TRUTH["windows"]:
        wins = [o for o in rooms[name]["openings"] if o["type"] == "window"]
        assert len(wins) == 1
        assert abs(wins[0]["width"]["value"] - width) <= 0.02
        assert abs(wins[0]["height"]["value"] - height) <= 0.02
    # no phantom windows anywhere else
    assert sum(o["type"] == "window" for r in result["rooms"] for o in r["openings"]) == len(TRUTH["windows"])


def test_intervals_present_and_ordered(result):
    def walk(x):
        if isinstance(x, dict):
            if {"value", "lo", "hi", "unit"} <= x.keys():
                assert x["lo"] <= x["value"] <= x["hi"]
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)
    walk(result)


def test_drift_ablation(result, ablation):
    """Correction on must recover the plan; poses as-is must visibly break it."""
    truth_area = sum((x1 - x0) * (y1 - y0) for x0, x1, y0, y1 in TRUTH["rooms"].values())
    on = result["stitched_plan"]
    off = ablation["stitched_plan"]
    assert on["drift_correction"].startswith("manhattan_heading_anchor")
    assert on["drift_detail"]["loop"]["closed"]
    assert off["drift_correction"] == "none"
    err_on = abs(on["footprint_area"]["value"] - truth_area)
    err_off = abs(off["footprint_area"]["value"] - truth_area)
    assert err_on < 0.1
    assert err_on < err_off
    assert len(on["adjacency"]) > len(off["adjacency"]) or \
        sum(len(r["walls"]) for r in ablation["rooms"]) > sum(len(r["walls"]) for r in result["rooms"])
