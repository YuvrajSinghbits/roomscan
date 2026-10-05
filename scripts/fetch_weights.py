"""Download all pretrained weights into weights/hf (run once; ~0.7 GB).

    python scripts/fetch_weights.py
"""
import numpy as np

from roomscan import damage, depth_model

if __name__ == "__main__":
    depth_model.predict(np.zeros((48, 64, 3), np.uint8), (64, 48), cache=False)
    damage._load()
    print("weights ready in", depth_model.WEIGHTS)
