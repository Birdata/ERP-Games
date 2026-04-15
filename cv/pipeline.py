"""
LEGO QC Computer Vision Pipeline
---------------------------------
Architecture
------------
1. Localise the figure with HSV colour masks → tight bounding-box ROI
2. Segment the ROI into N horizontal bands
3. Classify each band with LAB nearest-neighbour against a colour palette

The palette is either:
  a) Camera-calibrated  — built once from a reference image (preferred)
  b) Default            — ideal LEGO pigment values, camera-agnostic fallback

Why LAB nearest-neighbour instead of HSV ranges?
• LAB is perceptually uniform: equal Euclidean distance ≈ equal visual difference
• No thresholds to tune — just "which known colour is this band closest to?"
• If you provide a reference image the palette adapts to your camera and lighting
  automatically, so the same code works under any conditions.
"""

import json
import os

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# HSV colour ranges — used ONLY for figure localisation, NOT classification
# ---------------------------------------------------------------------------
COLOR_RANGES: dict[str, list[tuple[tuple, tuple]]] = {
    "blue":   [((88,  45,  30),  (138, 255, 255))],
    "yellow": [((16,  55,  55),  (42,  255, 255))],
    "green":  [((33,  35,  25),  (92,  255, 255))],
    "white":  [((0,   0,   120), (179, 75,  255))],
    "red":    [((0,   55,  35),  (13,  255, 255)),
               ((163, 55,  35),  (179, 255, 255))],
    "orange": [((7,   70,  70),  (28,  255, 255))],
    "purple": [((118, 28,  28),  (168, 255, 255))],
}

# ---------------------------------------------------------------------------
# Default LAB palette — ideal LEGO pigment colours in OpenCV LAB scale
# (L: 0-255, a: 0-255 centred at 128, b: 0-255 centred at 128)
# These are only used when NO reference image has been uploaded.
# ---------------------------------------------------------------------------
DEFAULT_LAB: dict[str, list[float]] = {
    "blue":   [60.0,  114.0,  65.0],
    "yellow": [210.0, 117.0, 196.0],
    "green":  [100.0,  80.0, 145.0],
    "white":  [245.0, 128.0, 128.0],
    "red":    [ 80.0, 195.0, 170.0],
    "orange": [165.0, 155.0, 185.0],
    "purple": [ 55.0, 158.0,  92.0],
}

CENTER_SAMPLE_FRACTION = 0.70
MIN_ROW_HEIGHT_PX      = 10
CANNY_LOW              = 20
CANNY_HIGH             = 60


# ===========================================================================
# Public API
# ===========================================================================

def detect_layers(image_path: str,
                  expected_colors: list[str],
                  palette: dict[str, list[float]] | None = None) -> list[str]:
    """
    Detect the colour of each brick layer (top to bottom).

    Parameters
    ----------
    image_path     : path to the captured frame
    expected_colors: ordered list of colour names (defines layer count)
    palette        : {colour_name: [L, a, b]} in OpenCV LAB scale.
                     Build with build_palette() from a reference image.
                     Falls back to DEFAULT_LAB if None.

    Returns an empty list if the image cannot be loaded.
    """
    img = cv2.imread(image_path)
    if img is None:
        return []

    img = _normalise_brightness(img)

    valid_colors   = list(dict.fromkeys(expected_colors))
    expected_count = len(expected_colors)

    roi   = _crop_to_color_region(img, valid_colors)
    bands = _segment_rows(roi, expected_count)

    active_palette = palette if palette else _default_palette(valid_colors)

    return [_classify_lab(_band_mean_lab(roi, top, bot), active_palette)
            for top, bot in bands]


def build_palette(ref_path: str,
                  expected_colors: list[str]) -> dict[str, list[float]]:
    """
    Sample each layer in the reference image and compute the mean LAB colour
    for every unique colour in expected_colors.

    The figure is split into equal bands — one per layer in expected_colors.
    If a colour appears in multiple layers (e.g. ["red","white","red","yellow"])
    its pixels from ALL matching bands are pooled before averaging, giving a
    more stable estimate.

    Returns {colour_name: [L, a, b]} ready to pass to detect_layers().
    """
    ref = cv2.imread(ref_path)
    if ref is None:
        return {}

    ref = _normalise_brightness(ref)
    valid_colors = list(dict.fromkeys(expected_colors))
    roi   = _crop_to_color_region(ref, valid_colors)
    bands = _equal_bands(roi.shape[0], len(expected_colors))

    # Collect LAB pixel arrays per colour (pool duplicate colours)
    buckets: dict[str, list[np.ndarray]] = {}
    for (top, bot), colour in zip(bands, expected_colors):
        band   = roi[top:bot, :]
        w      = band.shape[1]
        margin = max(1, int(w * (1 - CENTER_SAMPLE_FRACTION) / 2))
        sample = band[:, margin: w - margin] if w > margin * 2 else band
        sample = cv2.GaussianBlur(sample, (5, 5), 0)
        lab    = cv2.cvtColor(sample, cv2.COLOR_BGR2LAB)
        pixels = lab.reshape(-1, 3).astype(np.float64)
        buckets.setdefault(colour, []).append(pixels)

    palette: dict[str, list[float]] = {}
    for colour, arrays in buckets.items():
        all_px = np.vstack(arrays)
        palette[colour] = np.mean(all_px, axis=0).tolist()

    return palette


def save_palette(palette: dict[str, list[float]], json_path: str) -> None:
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(palette, f, indent=2)


def load_palette(json_path: str) -> dict[str, list[float]] | None:
    if not os.path.exists(json_path):
        return None
    with open(json_path, encoding="utf-8") as f:
        return json.load(f)


# ===========================================================================
# Internal helpers
# ===========================================================================

# ---------------------------------------------------------------------------
# Brightness normalisation
# ---------------------------------------------------------------------------

def _normalise_brightness(img: np.ndarray) -> np.ndarray:
    """CLAHE on the L channel of LAB — gentle and hue-safe."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


# ---------------------------------------------------------------------------
# Figure localisation (HSV masks → bounding box)
# ---------------------------------------------------------------------------

def _build_color_mask(img: np.ndarray, colors: list[str]) -> np.ndarray:
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    hsv  = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    for color in colors:
        for lo, hi in COLOR_RANGES.get(color, []):
            mask |= cv2.inRange(hsv, np.array(lo), np.array(hi))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


def _crop_to_color_region(img: np.ndarray, valid_colors: list[str]) -> np.ndarray:
    """Tight bounding box around valid-colour pixels; falls back to full image."""
    mask   = _build_color_mask(img, valid_colors)
    coords = cv2.findNonZero(mask)
    if coords is None:
        return img
    x, y, w, h = cv2.boundingRect(coords)
    margin = 8
    x = max(0, x - margin);  y = max(0, y - margin)
    w = min(img.shape[1] - x, w + 2 * margin)
    h = min(img.shape[0] - y, h + 2 * margin)
    return img[y:y + h, x:x + w]


# ---------------------------------------------------------------------------
# Segmentation
# ---------------------------------------------------------------------------

def _segment_rows(roi: np.ndarray, expected_count: int) -> list[tuple[int, int]]:
    h = roi.shape[0]
    if expected_count == 1:
        return [(0, h)]

    gray    = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges   = cv2.Canny(blurred, CANNY_LOW, CANNY_HIGH)
    profile = _smooth(edges.sum(axis=1).astype(np.float32), kernel=11)

    seams = _find_seams(profile, expected_count, h)
    boundaries = [0] + seams + [h]
    bands = [
        (boundaries[i], boundaries[i + 1])
        for i in range(len(boundaries) - 1)
        if boundaries[i + 1] - boundaries[i] >= MIN_ROW_HEIGHT_PX
    ]
    return bands if len(bands) == expected_count else _equal_bands(h, expected_count)


def _smooth(arr: np.ndarray, kernel: int) -> np.ndarray:
    return np.convolve(arr, np.ones(kernel) / kernel, mode="same")


def _find_seams(profile: np.ndarray, expected_count: int,
                total_height: int) -> list[int]:
    n_seams = expected_count - 1
    if n_seams <= 0:
        return []
    threshold = profile.max() * 0.10
    candidates = [(profile[y], y)
                  for y in range(1, len(profile) - 1)
                  if profile[y] > profile[y-1]
                  and profile[y] > profile[y+1]
                  and profile[y] > threshold]
    candidates.sort(reverse=True)
    selected: list[int] = []
    for _, y in candidates:
        if len(selected) >= n_seams:
            break
        if all(abs(y - s) >= MIN_ROW_HEIGHT_PX for s in selected):
            selected.append(y)
    selected.sort()
    return selected


def _equal_bands(height: int, count: int) -> list[tuple[int, int]]:
    step = height / count
    return [(int(i * step), int((i + 1) * step)) for i in range(count)]


# ---------------------------------------------------------------------------
# Classification — LAB nearest-neighbour
# ---------------------------------------------------------------------------

def _band_mean_lab(roi: np.ndarray, top: int, bot: int) -> np.ndarray:
    """Mean LAB colour of the centre strip of a band."""
    band = roi[top:bot, :]
    if band.size == 0:
        return np.zeros(3)
    w      = band.shape[1]
    margin = max(1, int(w * (1 - CENTER_SAMPLE_FRACTION) / 2))
    sample = band[:, margin: w - margin] if w > margin * 2 else band
    sample = cv2.GaussianBlur(sample, (5, 5), 0)
    lab    = cv2.cvtColor(sample, cv2.COLOR_BGR2LAB)
    return np.array(cv2.mean(lab)[:3])


def _classify_lab(mean_lab: np.ndarray,
                  palette: dict[str, list[float]]) -> str:
    """Return the palette colour with the smallest LAB distance."""
    return min(
        palette.keys(),
        key=lambda c: float(np.linalg.norm(mean_lab - np.array(palette[c])))
    )


def _default_palette(valid_colors: list[str]) -> dict[str, list[float]]:
    """Subset of DEFAULT_LAB containing only the colours needed for this figure."""
    return {c: DEFAULT_LAB[c] for c in valid_colors if c in DEFAULT_LAB}
