"""Per-room photo folders from a Stray Scanner capture (photo-tier proxy input).

Rooms come from the LiDAR tier on the same capture. For each room, frames whose
camera is inside it are scored by how far they see (median LiDAR depth) and the
best few, spread over viewing directions, are written upright as JPEGs with the
true 35 mm-equivalent focal length in EXIF -- what an iPhone photo carries.
These are walkthrough frames, not the protocol's corner photos, so the photo
tier is exercised on a harder input than it is designed for.

    python -m scripts.stray_to_photos <stray capture> <out folder> [per_room] [corners]

mode "corners": per room corner, the frame taken closest to it (within
CORNER_R) that faces the room centre -- the nearest this footage gets to the
protocol's corner photo. The photo tier itself never sees LiDAR information.
"""
import sys
from pathlib import Path

import cv2
import numpy as np
from matplotlib.path import Path as MPath
from PIL import Image

from roomscan import drift as drift_mod
from roomscan.lidar_io import camera_points, find_capture, frame_count, load_frames, load_rgb
from roomscan.plan2d import Cloud, build_plan, rotate, vertices
from scripts.stray_to_video import upright_rotation


CORNER_R = 0.7


def main(src, dst, per_room=6, mode="far"):
    root = find_capture(Path(src))
    frames = load_frames(root, stride=max(1, -(-frame_count(root) // 600)))
    cam = [camera_points(f, step=2) for f in frames]
    poses, _ = drift_mod.correct([f.pose for f in frames], cam)
    pts = np.concatenate([p @ T[:3, :3].T + T[:3, 3] for (p, _), T in zip(cam, poses)])
    nrm = np.concatenate([n @ T[:3, :3].T for (_, n), T in zip(cam, poses)])
    src_i = np.concatenate([np.full(len(p), i, np.int32) for i, (p, _) in enumerate(cam)])
    cloud = Cloud(np.stack([pts[:, 0], -pts[:, 2]], 1), pts[:, 1].copy(),
                  np.stack([nrm[:, 0], -nrm[:, 2], nrm[:, 1]], 1), src_i)
    cams = np.array([[T[0, 3], -T[2, 3]] for T in poses])
    rooms, _, theta, _, _ = build_plan(cloud, cams, np.array([f.index for f in frames]))
    cams_r = rotate(cams, theta)
    heading = np.array([np.arctan2(*(rotate(np.array([[-T[0, 2], T[2, 2]]]), theta)[0][::-1])) for T in poses])
    rot = upright_rotation(root)
    f35 = round(frames[0].K[0, 0] / frames[0].depth.shape[1] * 1920 * 43.27 / np.hypot(1920, 1440))
    dst = Path(dst)
    for room in rooms:
        inside = np.nonzero(MPath(vertices(room.lines)).contains_points(cams_r))[0]
        if not len(inside):
            continue
        if mode == "corners":
            poly = vertices(room.lines)
            centre = poly.mean(0)
            chosen = []
            for corner in poly:
                d = np.linalg.norm(cams_r[inside] - corner, axis=1)
                to_c = np.arctan2(*(centre - cams_r[inside]).T[::-1])
                facing = np.abs(np.angle(np.exp(1j * (heading[inside] - to_c)))) < np.deg2rad(35)
                ok = (d < CORNER_R) & facing
                if ok.any():
                    i = inside[ok][np.argmin(d[ok])]
                    if i not in chosen:
                        chosen.append(i)
        else:
            depth = np.array([np.median(frames[i].depth[frames[i].depth > 0]) if (frames[i].depth > 0).any() else 0
                              for i in inside])
            order = inside[np.argsort(-depth)]
            chosen = []
            for i in order:   # far-seeing views, at least 60 degrees apart in heading
                if all(abs(np.angle(np.exp(1j * (heading[i] - heading[j])))) > np.deg2rad(60) for j in chosen):
                    chosen.append(i)
                if len(chosen) == per_room:
                    break
        out = dst / room.id
        out.mkdir(parents=True, exist_ok=True)
        for k, (i, rgb) in enumerate(zip(chosen, load_rgb(root, [frames[i].index for i in chosen], width=1920))):
            if rgb is None:
                continue
            if rot is not None:
                rgb = cv2.rotate(rgb, rot)
            exif = Image.Exif()
            exif.get_ifd(0x8769)[41989] = int(f35)
            Image.fromarray(rgb).save(out / f"IMG_{k:04d}.jpg", quality=92, exif=exif)
        print(room.id, len(chosen), "photos")


if __name__ == "__main__":
    args = sys.argv[3:]
    main(sys.argv[1], sys.argv[2], int(args[0]) if args else 6, args[1] if len(args) > 1 else "far")
