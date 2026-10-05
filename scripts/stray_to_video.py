"""Turn a Stray Scanner capture's rgb.mp4 into a native-camera-style clip.

Stray stores frames in sensor (landscape) orientation without a rotation tag;
the iPhone Camera app (what the video-tier protocol uses) writes upright
frames. The first odometry pose tells which way is up in the image, so the
clip is rotated by the matching multiple of 90 degrees. Same walk, same pixels:
a fair video-tier input to compare against the LiDAR tier on identical footage.

    python scripts/stray_to_video.py <stray capture> <out.mp4>
"""
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from roomscan.lidar_io import _stray_rows, find_capture


def upright_rotation(root: Path):
    r = _stray_rows(root)[0]
    R = Rotation.from_quat([float(r[k]) for k in ("qx", "qy", "qz", "qw")]).as_matrix()
    up_cam = R.T @ np.array([0.0, 1.0, 0.0])          # world up in OpenCV camera axes
    # image "up" is -y; pick the 90-degree rotation that brings up_cam closest to it
    u, v = up_cam[0], up_cam[1]
    options = {None: -v, cv2.ROTATE_90_CLOCKWISE: -u, cv2.ROTATE_180: v, cv2.ROTATE_90_COUNTERCLOCKWISE: u}
    # after ROTATE_90_CLOCKWISE the old -x direction points up, etc.
    return max(options, key=options.get)


def main(src, dst):
    root = find_capture(Path(src))
    rot = upright_rotation(root)
    cap = cv2.VideoCapture(str(root / "rgb.mp4"))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    out = None
    while True:
        ok, img = cap.read()
        if not ok:
            break
        if rot is not None:
            img = cv2.rotate(img, rot)
        if out is None:
            h, w = img.shape[:2]
            out = cv2.VideoWriter(str(dst), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        out.write(img)
    out.release()
    print(f"{dst}: rotation {rot}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
