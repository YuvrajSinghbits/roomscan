"""One command per capture:

    roomscan run <capture_path> --tier {photo,video,lidar} --out out/<id>

Tier is auto-detected when omitted:
  * a folder of room sub-folders containing only images -> photo
  * a single .mov/.mp4 file                           -> video
  * a 3D Scanner App / Record3D raw export              -> lidar
"""
import argparse
import json
import sys
import time
from pathlib import Path

from roomscan import __version__
from roomscan.schema import validate

VIDEO_EXT = {".mov", ".mp4", ".m4v"}
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".heic"}


def detect_tier(path: Path) -> str:
    if path.is_file() and path.suffix.lower() in VIDEO_EXT:
        return "video"
    if path.is_dir():
        files = [p for p in path.rglob("*") if p.is_file()]
        if any(p.suffix.lower() in {".depth", ".exr", ".npy"} or p.name == "info.json" for p in files):
            return "lidar"
        if files and all(p.suffix.lower() in IMAGE_EXT for p in files):
            return "photo"
    raise SystemExit(f"cannot detect tier for {path}; pass --tier")


def run(capture: Path, tier: str, out_dir: Path, **opts) -> dict:
    from roomscan.tiers import dispatch

    t0 = time.perf_counter()
    result = dispatch(tier, capture, out_dir, **opts)
    result["capture"]["runtime_s"] = round(time.perf_counter() - t0, 2)
    result["capture"]["pipeline_version"] = __version__
    validate(result)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "result.json").write_text(json.dumps(result, indent=2, default=float))
    return result


def main(argv=None):
    ap = argparse.ArgumentParser(prog="roomscan")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="process one capture")
    r.add_argument("capture", type=Path)
    r.add_argument("--tier", choices=["photo", "video", "lidar"])
    r.add_argument("--out", type=Path)
    r.add_argument("--no-drift-correction", dest="drift", action="store_false",
                   help="ablation: use the device poses as recorded")
    args = ap.parse_args(argv)

    tier = args.tier or detect_tier(args.capture)
    out = args.out or Path("out") / args.capture.stem
    result = run(args.capture, tier, out, drift=args.drift)
    print(f"[{tier}] {len(result['rooms'])} rooms -> {out / 'result.json'}")


if __name__ == "__main__":
    sys.exit(main())
