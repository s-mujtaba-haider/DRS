"""White pitch-marking extraction (used by the experimental crease registration).

Creases are thin bright lines on a darker pitch; see ``white_line_mask`` for how they are
separated from kit, pads, text and noise.
"""

from __future__ import annotations

import cv2
import numpy as np


def white_line_mask(bgr: np.ndarray, kernel_px: int = 15, min_contrast: float = 10.0, max_saturation: int = 90,
                    max_thickness_px: float | None = None) -> np.ndarray:
    """Thin, bright, low-saturation, elongated structures (painted lines).

    Horizontal and vertical white top-hats keep structures narrower than ``kernel_px`` across
    their length and brighter than their surroundings by ``min_contrast`` grey levels. A global
    contrast threshold is not used: on broadcast frames the strongest responses are white kit, pads and on-screen text, and faint
    creases would fall below any threshold set by them. Instead, connected components are kept
    only if they are long (at least half the kernel) and thin (mean thickness, area / length, at
    most ``max_thickness_px``, default a quarter of the kernel), which removes kit, pads, text
    and pixel noise while keeping line fragments.
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    # Directional top-hats: opening with a vertical line removes thin horizontal lines (so the
    # difference isolates them), and vice versa. They respond to faint 1-2 px creases that a round
    # kernel of the same size blurs into the pitch texture.
    th = [cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_RECT, sz))
          for sz in ((1, kernel_px), (kernel_px, 1))]
    tophat = np.maximum(th[0], th[1])
    sat = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[..., 1]
    raw = ((tophat >= min_contrast) & (sat <= max_saturation)).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(raw, connectivity=8)
    length = np.hypot(st[:, cv2.CC_STAT_WIDTH], st[:, cv2.CC_STAT_HEIGHT])
    thick = st[:, cv2.CC_STAT_AREA] / np.maximum(length, 1.0)
    keep = (length >= max(8, kernel_px // 2)) & (thick <= (max_thickness_px or max(2.5, kernel_px / 4)))
    keep[0] = False
    return keep[lab].astype(np.uint8) * 255
