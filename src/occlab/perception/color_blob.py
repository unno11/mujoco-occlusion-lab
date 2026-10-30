"""Colour-based detection of the red box and the green goal region.

No learned model: HSV thresholds plus the calibrated homography. That is enough to
answer the two questions the task needs -- where is the object in table
coordinates, and how much of it can the camera currently see.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np

from occlab.perception.calibration import pixel_to_world

log = logging.getLogger(__name__)


@dataclass
class Detection:
    xy: np.ndarray | None      # table coordinates (m)
    area_px: int               # red pixel count
    pixel: np.ndarray | None   # centroid in the image
    visible: bool
    partial: bool              # detected, but much smaller than the nominal blob

    def __str__(self) -> str:
        if not self.visible:
            return f"<not visible, {self.area_px}px>"
        return f"<xy={np.round(self.xy, 3)} area={self.area_px}px partial={self.partial}>"


class ColorBlobDetector:
    # thresholds on the rendered scene; the box is (0.86, 0.06, 0.06) and the goal
    # marker is (0.10, 0.80, 0.20)
    RED_LO = np.array([0, 120, 90])
    RED_HI = np.array([10, 255, 255])
    RED_LO2 = np.array([170, 120, 90])
    RED_HI2 = np.array([180, 255, 255])
    GREEN_LO = np.array([40, 90, 80])
    GREEN_HI = np.array([85, 255, 255])

    def __init__(self, homography: np.ndarray, nominal_red_area: int = 2600,
                 min_area: int = 60, center_offset: np.ndarray | None = None,
                 pixel_scale: float = 1.0) -> None:
        self.H = homography
        self.nominal_red_area = nominal_red_area
        self.min_area = min_area
        # The homography was fitted at the calibration resolution. Rendering smaller
        # for batch runs is much faster, so pixel coordinates are scaled back up
        # before the projection.
        self.pixel_scale = float(pixel_scale)
        # Fixed bias between the silhouette centroid and the box's footprint centre:
        # the camera sees a 10 cm cube from above, so the visible red region is not
        # the top-down footprint. Calibrated once in scripts/20_calibrate.py.
        self.center_offset = np.zeros(2) if center_offset is None else np.asarray(center_offset, float)
        self._visible_ratio = 0.55   # below this fraction of nominal: partial
        self._lost_ratio = 0.12      # below this: treat as not visible

    # ---------- helpers ----------
    def _mask(self, image: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        m = cv2.inRange(hsv, self.RED_LO, self.RED_HI) | cv2.inRange(hsv, self.RED_LO2, self.RED_HI2)
        return cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    def measure_nominal_area(self, image: np.ndarray) -> int:
        return int(cv2.countNonZero(self._mask(image)))

    # ---------- api ----------
    def detect_object(self, image: np.ndarray) -> Detection:
        # areas scale with the square of the resolution factor
        min_area = self.min_area / (self.pixel_scale ** 2)
        nominal = self.nominal_red_area / (self.pixel_scale ** 2)
        mask = self._mask(image)
        area = int(cv2.countNonZero(mask))
        if area < min_area:
            return Detection(None, area, None, visible=False, partial=False)

        n, _, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if n <= 1:
            return Detection(None, area, None, visible=False, partial=False)
        # largest red blob = the box
        biggest = int(np.argmax([stats[i, cv2.CC_STAT_AREA] for i in range(1, n)])) + 1
        blob_area = int(stats[biggest, cv2.CC_STAT_AREA])
        px = centroids[biggest]
        ratio = blob_area / max(1.0, nominal)
        visible = ratio >= self._lost_ratio
        return Detection(
            xy=pixel_to_world(self.H, px * self.pixel_scale) + self.center_offset if visible else None,
            area_px=blob_area,
            pixel=px,
            visible=visible,
            partial=visible and ratio < self._visible_ratio,
        )

    def detect_goal(self, image: np.ndarray) -> np.ndarray | None:
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(hsv, self.GREEN_LO, self.GREEN_HI)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        n, _, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if n <= 1:
            return None
        biggest = int(np.argmax([stats[i, cv2.CC_STAT_AREA] for i in range(1, n)])) + 1
        if stats[biggest, cv2.CC_STAT_AREA] < self.min_area:
            return None
        return pixel_to_world(self.H, centroids[biggest])
