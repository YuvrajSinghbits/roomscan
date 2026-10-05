import json
from functools import lru_cache
from pathlib import Path

import jsonschema

SCHEMA_PATH = Path(__file__).resolve().parents[2] / "schema" / "capture_output.schema.json"


@lru_cache
def load_schema() -> dict:
    return json.loads(SCHEMA_PATH.read_text())


def validate(doc: dict) -> None:
    jsonschema.validate(doc, load_schema())


def measurement(value: float, half_width: float, unit: str = "m") -> dict:
    """Symmetric 90% interval. Tiers widen half_width as sensor data thins."""
    return {"value": round(value, 4), "lo": round(value - half_width, 4),
            "hi": round(value + half_width, 4), "unit": unit}
