import os
from typing import Optional

import cv2
import numpy as np


def _assert_supported(path: str) -> None:
    ext = os.path.splitext(path)[1].lower()
    if ext not in {".jpg", ".jpeg", ".png", ".tif", ".tiff"}:
        raise ValueError(f"Unsupported format: {ext}")


def load_image(path: str) -> np.ndarray:
    """Load image as numpy array (keeps original bands when possible)."""
    _assert_supported(path)
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Failed to read image: {path}")

    if img.ndim == 3 and img.shape[2] >= 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def apply_grayscale(img: np.ndarray) -> np.ndarray:
    """Convert to grayscale (2D uint8 image)."""
    if img.ndim == 2:
        return _to_uint8(img)
    rgb = img[:, :, :3]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return _to_uint8(gray)


def apply_contrast_enhancement(img: np.ndarray) -> np.ndarray:
    """Increase local contrast using CLAHE (per-channel for RGB)."""
    if img.ndim == 2:
        u8 = _to_uint8(img)
        return _clahe(u8)

    rgb = img[:, :, :3]
    u8 = _to_uint8(rgb)
    channels = [u8[:, :, i] for i in range(3)]
    enhanced = [_clahe(c) for c in channels]
    return np.stack(enhanced, axis=2)


def apply_edge_detection(img: np.ndarray) -> np.ndarray:
    """Detect edges using Canny on grayscale version."""
    gray = apply_grayscale(img)
    edges = cv2.Canny(gray, 80, 160)
    return edges


def apply_ndvi_like(img: np.ndarray) -> Optional[np.ndarray]:
    """
    NDVI-like visualization.
    - If 4+ bands: assumes band3 ~ NIR and band2 ~ Red.
    - If 3 bands (RGB): uses Green as pseudo-NIR and Red as Red.
    Returns an RGB visualization (uint8) or None if not possible.
    """
    if img.ndim != 3 or img.shape[2] < 3:
        return None

    bands = img.astype(np.float32)
    if bands.shape[2] >= 4:
        red = bands[:, :, 2]
        nir = bands[:, :, 3]
    else:
        red = bands[:, :, 0] if _looks_rgb(img) else bands[:, :, 2]
        nir = bands[:, :, 1]

    nd = (nir - red) / (nir + red + 1e-6)
    nd_norm = _normalize_01(nd)
    heat = (nd_norm * 255).astype(np.uint8)
    colored = cv2.applyColorMap(heat, cv2.COLORMAP_VIRIDIS)
    return cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)


def _normalize_01(x: np.ndarray) -> np.ndarray:
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    mn, mx = float(np.min(x)), float(np.max(x))
    if mx - mn < 1e-12:
        return np.zeros_like(x, dtype=np.float32)
    return (x - mn) / (mx - mn)


def _to_uint8(img: np.ndarray) -> np.ndarray:
    """Convert to uint8 safely (scales if needed)."""
    if img.dtype == np.uint8:
        return img
    img_f = img.astype(np.float32)
    mn, mx = float(np.min(img_f)), float(np.max(img_f))
    if mx - mn < 1e-12:
        return np.zeros(img.shape, dtype=np.uint8)
    scaled = (img_f - mn) / (mx - mn)
    return (scaled * 255).clip(0, 255).astype(np.uint8)


def _clahe(u8_gray: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(u8_gray)


def _looks_rgb(img: np.ndarray) -> bool:
    """Heuristic: assume already RGB if last dim is 3 and dtype is uint8."""
    return img.ndim == 3 and img.shape[2] >= 3
