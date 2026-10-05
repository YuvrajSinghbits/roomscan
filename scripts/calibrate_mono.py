"""Measure the monocular depth model against LiDAR on Stray Scanner captures.

    python scripts/calibrate_mono.py data/raw/<stray capture> [...]

Prints median(model/LiDAR) (the scale calibration in depth_model.SCALE_CAL),
the per-frame spread, and the within-frame error once per-frame scale is removed.
"""
import sys

import cv2
import numpy as np
from PIL import Image

from roomscan.depth_model import MODEL_ID, SCALE_CAL, predict
from roomscan.lidar_io import find_capture


def main(paths, per_capture=25):
    ratios, within = [], []
    for p in paths:
        root = find_capture(p)
        cap = cv2.VideoCapture(str(root / "rgb.mp4"))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for i in np.linspace(0, n - 1, per_capture).astype(int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ok, img = cap.read()
            if not ok:
                continue
            rgb = cv2.cvtColor(cv2.resize(img, (640, 480)), cv2.COLOR_BGR2RGB)
            d = predict(rgb, (256, 192)) / SCALE_CAL.get(MODEL_ID, 1.0)   # raw model output
            lid = np.asarray(Image.open(root / "depth" / f"{i:06d}.png")) / 1000.0
            conf = np.asarray(Image.open(root / "confidence" / f"{i:06d}.png"))
            ok = (lid > 0.3) & (conf >= 2)
            if ok.sum() < 1000:
                continue
            r = d[ok] / lid[ok]
            ratios.append(np.median(r))
            within.append(np.median(np.abs(r / np.median(r) - 1)))
    ratios = np.array(ratios)
    print(f"frames {len(ratios)}  median model/LiDAR {np.median(ratios):.3f}  "
          f"per-frame spread {ratios.std() / np.median(ratios):.3f}  "
          f"within-frame |rel err| {np.median(within):.3f}")


if __name__ == "__main__":
    main(sys.argv[1:])
