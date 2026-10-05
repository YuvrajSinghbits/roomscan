from pathlib import Path


def dispatch(tier: str, capture: Path, out_dir: Path) -> dict:
    if tier == "lidar":
        from roomscan.tiers.lidar import process
    elif tier == "video":
        from roomscan.tiers.video import process
    elif tier == "photo":
        from roomscan.tiers.photo import process
    else:
        raise ValueError(tier)
    return process(capture, out_dir)
