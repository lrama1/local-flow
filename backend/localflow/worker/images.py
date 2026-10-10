"""Torch-free image helpers for adapters: size snapping, reference sheets, PNG metadata."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PIL import Image
from PIL.PngImagePlugin import PngInfo


def snap(value: int, multiple: int, *, minimum: int = 256, maximum: int = 2048) -> int:
    """Round to the nearest multiple the model requires (Section 10.2), within sane bounds."""
    v = max(minimum, min(maximum, int(value)))
    return max(multiple, round(v / multiple) * multiple)


def reference_sheet(
    images: list[Image.Image], *, height: int = 1024, gutter: int = 24, max_width: int = 2560
) -> Image.Image:
    """Lays N references side by side on white, for edit models that accept a single reference image
    (Section 10.4). Each image is scaled to the same height; the sheet is downscaled if it gets too wide."""
    if not images:
        raise ValueError("reference_sheet needs at least one image")
    if len(images) == 1:
        return images[0].convert("RGB")
    scaled = []
    for im in images:
        im = im.convert("RGB")
        w = max(1, round(im.width * height / im.height))
        scaled.append(im.resize((w, height), Image.Resampling.LANCZOS))
    total = sum(im.width for im in scaled) + gutter * (len(scaled) - 1)
    sheet = Image.new("RGB", (total, height), (255, 255, 255))
    x = 0
    for im in scaled:
        sheet.paste(im, (x, 0))
        x += im.width + gutter
    if sheet.width > max_width:
        h = round(height * max_width / sheet.width)
        sheet = sheet.resize((max_width, h), Image.Resampling.LANCZOS)
    return sheet


def save_png(image: Image.Image, path: Path, meta: dict[str, Any]) -> Path:
    """Saves with generation parameters in a `localflow` PNG text chunk (handy outside the app)."""
    info = PngInfo()
    info.add_text("localflow", json.dumps(meta, ensure_ascii=False, default=str))
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, "PNG", pnginfo=info)
    return path
