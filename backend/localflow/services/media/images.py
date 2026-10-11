"""Image helpers shared by the LLM features that send pictures (character describe, vision review)."""

from __future__ import annotations

import base64
import io
from pathlib import Path

from PIL import Image


def jpeg_data_url(path: Path, max_side: int = 1024) -> str:
    """A downscaled JPEG data URL (vision models don't need full resolution; it also caps cost)."""
    with Image.open(path) as im:
        img = im.convert("RGB")
        img.thumbnail((max_side, max_side))
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=88)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
