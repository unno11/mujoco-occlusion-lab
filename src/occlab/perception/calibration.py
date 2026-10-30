"""One-time camera calibration: image pixels -> table coordinates.

Four magenta fiducials sit on the table at known positions (see `scene_push.xml`).
Magenta, not white: the robot is white, and a white threshold happily "finds" four
markers on the Panda's links.

The correspondence is fixed by construction, not searched:

* the camera has no roll, so image rows (increasing v) go from the far side of the
  table to the near side, i.e. from +y to -y;
* image columns (increasing u) follow +x.

Fitting with a permutation search was tried and is wrong: with four points the
homography has exactly as many unknowns as constraints, and a mirrored assignment
fits them perfectly, so a search picks the mirror image. Two extra known points
(the goal patch and the box's start position) are used only as *validation*, and
the fit is rejected if they do not land within tolerance.

This is the only place where known world coordinates enter the perception stack.
The matrix is written to `configs/calibration.json` and frozen; episodes only ever
see pixels.
"""

from __future__ import annotations

import json
import logging
import pathlib

import cv2
import numpy as np

log = logging.getLogger(__name__)

# fiducial positions on the table, in metres (must match scene_push.xml), listed in
# the order the detector returns them: image row (far first), then left to right
MARKERS = np.array(
    [
        [0.20, 0.28],   # far row, left
        [0.66, 0.28],   # far row, right
        [0.20, -0.28],  # near row, left
        [0.66, -0.28],  # near row, right
    ],
    dtype=float,
)
GOAL_WORLD = np.array([0.60, 0.00])
GOAL_TOL = 0.020     # m, validation only


def _hsv_mask(image: np.ndarray, lo, hi) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    m = cv2.inRange(hsv, np.array(lo), np.array(hi))
    return cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))


def _biggest_centroid(mask: np.ndarray, min_area: int) -> tuple[np.ndarray, int] | None:
    n, _, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return None
    areas = [stats[i, cv2.CC_STAT_AREA] for i in range(1, n)]
    biggest = int(np.argmax(areas)) + 1
    if int(stats[biggest, cv2.CC_STAT_AREA]) < min_area:
        return None
    return np.asarray(centroids[biggest], dtype=float), int(stats[biggest, cv2.CC_STAT_AREA])


def detect_marker_pixels(image: np.ndarray, min_area: int = 12, max_area: int = 900) -> np.ndarray:
    mask = _hsv_mask(image, (140, 120, 120), (170, 255, 255))
    n, _, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    blobs = [
        (centroids[i], stats[i, cv2.CC_STAT_AREA])
        for i in range(1, n)
        if min_area <= stats[i, cv2.CC_STAT_AREA] <= max_area
    ]
    if len(blobs) < 4:
        raise RuntimeError(f"expected 4 fiducials, found {len(blobs)} (areas {[b[1] for b in blobs]})")
    blobs.sort(key=lambda b: -b[1])
    pts = np.array([b[0] for b in blobs[:4]], dtype=float)

    # group into rows by image v (small v = far side of the table), then order each
    # row left to right. np.lexsort was tried here and quietly returned the rows in
    # the opposite order along u, which mirrored x and put the goal 340 mm off.
    by_v = pts[np.argsort(pts[:, 1])]
    far, near = by_v[:2], by_v[2:]
    far = far[np.argsort(far[:, 0])]
    near = near[np.argsort(near[:, 0])]
    return np.vstack([far, near])


def detect_goal_pixel(image: np.ndarray) -> np.ndarray:
    mask = _hsv_mask(image, (40, 90, 80), (85, 255, 255))
    found = _biggest_centroid(mask, min_area=200)
    if found is None:
        raise RuntimeError("goal patch not found")
    return found[0]


def detect_box_pixel(image: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, (0, 120, 90), (10, 255, 255)) | cv2.inRange(hsv, (170, 120, 90), (180, 255, 255))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    found = _biggest_centroid(mask, min_area=200)
    if found is None:
        raise RuntimeError("box not found during calibration")
    return found[0]


def pixel_to_world(H: np.ndarray, uv: np.ndarray) -> np.ndarray:
    p = np.array([uv[0], uv[1], 1.0])
    w = H @ p
    return (w[:2] / w[2]).astype(float)


def fit_homography(image: np.ndarray, box_start_xy: np.ndarray | None = None) -> tuple[np.ndarray, dict]:
    px = detect_marker_pixels(image)
    H, _ = cv2.findHomography(px, MARKERS)
    if H is None or not np.all(np.isfinite(H)):
        raise RuntimeError("homography fit failed")

    proj = cv2.perspectiveTransform(px.reshape(-1, 1, 2), H).reshape(-1, 2)
    marker_err = float(np.max(np.linalg.norm(proj - MARKERS, axis=1)))

    goal_px = detect_goal_pixel(image)
    goal_est = pixel_to_world(H, goal_px)
    goal_err = float(np.linalg.norm(goal_est - GOAL_WORLD))
    if goal_err > GOAL_TOL:
        raise RuntimeError(
            f"calibration rejected: goal maps to {np.round(goal_est, 3)} but is at "
            f"{GOAL_WORLD} ({goal_err * 1000:.0f} mm off) -- mirrored or rotated fiducial order"
        )

    info = {
        "marker_residual_mm": round(marker_err * 1000, 2),
        "goal_error_mm": round(goal_err * 1000, 2),
        "goal_estimate": np.round(goal_est, 4).tolist(),
        "markers_px": np.round(px, 2).tolist(),
        "goal_px": np.round(goal_px, 2).tolist(),
    }
    if box_start_xy is not None:
        box_est = pixel_to_world(H, detect_box_pixel(image))
        info["box_error_mm"] = round(float(np.linalg.norm(box_est - np.asarray(box_start_xy, float))) * 1000, 2)
    log.info("calibration: %s", info)
    return H, info


def save(H: np.ndarray, path: pathlib.Path, extra: dict | None = None) -> None:
    payload = {"homography": H.tolist()}
    if extra:
        payload.update(extra)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1), encoding="utf-8")


def load(path: pathlib.Path) -> tuple[np.ndarray, dict]:
    data = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    H = np.asarray(data["homography"], dtype=float)
    meta = {k: v for k, v in data.items() if k != "homography"}
    return H, meta
