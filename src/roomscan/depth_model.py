"""Monocular metric depth for the photo and video tiers.

Model: Depth Anything V2, metric indoor fine-tune, Small (Apache-2.0,
https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf).
Weights are fetched on first use (or by scripts/fetch_weights.py) into
weights/hf; nothing calls our infrastructure.

Predictions are cached under .cache/depth keyed by (model, image bytes, size),
so a benchmark replays bit-for-bit; delete the folder or pass cache=False to
force the live path.
"""
import hashlib
import os
from pathlib import Path

import numpy as np

MODEL_SMALL = "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"
MODEL_BASE = "depth-anything/Depth-Anything-V2-Metric-Indoor-Base-hf"
MODEL_ID = MODEL_SMALL
# Scale calibration: median(model depth / LiDAR depth) over 72 frames of the
# three sample Stray Scanner captures (scripts/calibrate_mono.py). Per-frame
# spread around it is ~37% and within-frame error after scale removal ~11%;
# the photo and video error models are set from those numbers.
SCALE_CAL = {MODEL_SMALL: 1 / 1.159, MODEL_BASE: 1 / 1.054}
ROOT = Path(__file__).resolve().parents[2]
WEIGHTS = ROOT / "weights" / "hf"
CACHE = ROOT / ".cache" / "depth"

_models = {}


def _load(model_id):
    if model_id not in _models:
        import torch
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        torch.set_num_threads(max(1, os.cpu_count() or 1))
        proc = AutoImageProcessor.from_pretrained(model_id, cache_dir=WEIGHTS)
        model = AutoModelForDepthEstimation.from_pretrained(model_id, cache_dir=WEIGHTS).eval()
        _models[model_id] = (proc, model)
    return _models[model_id]


def predict(rgb: np.ndarray, size=(256, 192), cache=True, model_id: str = MODEL_ID) -> np.ndarray:
    """rgb HxWx3 uint8 -> calibrated metric z-depth (metres) at size=(w, h)."""
    key = hashlib.sha1(model_id.encode() + rgb.tobytes() + repr((rgb.shape, size)).encode()).hexdigest()
    path = CACHE / f"{key}.npy"
    if cache and path.exists():
        return np.load(path) * SCALE_CAL.get(model_id, 1.0)
    import torch
    from PIL import Image

    proc, model = _load(model_id)
    with torch.no_grad():
        inputs = proc(images=Image.fromarray(rgb), return_tensors="pt")
        out = model(**inputs).predicted_depth            # (1, h', w')
        d = torch.nn.functional.interpolate(out[None], size=(size[1], size[0]), mode="bilinear",
                                            align_corners=False)[0, 0].numpy().astype(np.float32)
    if cache:
        CACHE.mkdir(parents=True, exist_ok=True)
        np.save(path, d)
    return d * SCALE_CAL.get(model_id, 1.0)
