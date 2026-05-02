import os
from typing import Union

import numpy as np
from PIL import Image


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def stem_with_suffix(input_path: str, suffix: str) -> str:
    base = os.path.basename(input_path)
    stem, _ = os.path.splitext(base)
    return f"{stem}_{suffix}"


def save_image(path: str, img: Union[np.ndarray, Image.Image]) -> None:
    """Save numpy arrays via Pillow (supports grayscale and RGB)."""
    ensure_dir(os.path.dirname(path) or ".")

    if isinstance(img, Image.Image):
        img.save(path)
        return

    arr = np.asarray(img)
    if arr.ndim == 2:
        im = Image.fromarray(arr.astype(np.uint8), mode="L")
    else:
        im = Image.fromarray(arr.astype(np.uint8), mode="RGB")
    im.save(path)
