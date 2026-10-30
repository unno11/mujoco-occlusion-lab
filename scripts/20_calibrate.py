"""Phase 2 gate: fit the camera homography, calibrate the silhouette bias, measure error.

    MUJOCO_GL=egl ./.venv/bin/python scripts/20_calibrate.py

Writes `configs/calibration.json` (homography + fixed centroid bias + residuals).
The detector never sees the simulator state; truth is used here, offline, only to
report the error and to fit one constant bias.
"""

from __future__ import annotations

import pathlib
import sys

import mujoco
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from occlab.config import load_config  # noqa: E402
from occlab.envs.push_env import PushEnv  # noqa: E402
from occlab.perception import calibration  # noqa: E402
from occlab.perception.color_blob import ColorBlobDetector  # noqa: E402

OUT = ROOT / "configs" / "calibration.json"


def main() -> None:
    cfg = load_config()
    env = PushEnv(cfg, render=True)
    env.reset(seed=0)
    m, d = env.model, env.data
    box_qadr = int(m.jnt_qposadr[m.joint("box_free").id])
    half = cfg.task.box_half_size

    def place(x: float, y: float) -> None:
        d.qpos[box_qadr : box_qadr + 3] = (x, y, half)
        d.qpos[box_qadr + 3 : box_qadr + 7] = (1, 0, 0, 0)
        mujoco.mj_forward(m, d)

    frame = env.render()
    H, info = calibration.fit_homography(frame, box_start_xy=env.object_truth_xy())
    print(
        f"homography: marker residual {info['marker_residual_mm']:.1f} mm, "
        f"goal error {info['goal_error_mm']:.1f} mm, raw box bias {info.get('box_error_mm', float('nan')):.1f} mm"
    )

    nominal = ColorBlobDetector(H).measure_nominal_area(frame)
    print(f"nominal red blob at reset: {nominal} px")

    # --- fit the constant silhouette bias on a coarse grid --------------------
    bias_samples = []
    for x in (0.36, 0.48, 0.60):
        for y in (-0.15, 0.0, 0.15):
            place(x, y)
            det = ColorBlobDetector(H, nominal_red_area=nominal).detect_object(env.render())
            if det.xy is not None:
                bias_samples.append(np.array([x, y]) - det.xy)
    bias = np.mean(bias_samples, axis=0)
    spread = float(np.max(np.linalg.norm(np.array(bias_samples) - bias, axis=1)))
    print(f"silhouette bias {np.round(bias * 1000, 1)} mm over {len(bias_samples)} samples "
          f"(residual spread {spread * 1000:.1f} mm)")

    detector = ColorBlobDetector(H, nominal_red_area=nominal, center_offset=bias)

    # --- error over a wider grid ---------------------------------------------
    xs = np.linspace(0.30, 0.72, 6)
    ys = np.linspace(-0.20, 0.20, 5)
    errs, missing, rows = [], 0, []
    for x in xs:
        for y in ys:
            place(x, y)
            det = detector.detect_object(env.render())
            if det.xy is None:
                missing += 1
                rows.append((x, y, None))
                continue
            e = float(np.linalg.norm(det.xy - np.array([x, y])))
            errs.append(e)
            rows.append((x, y, e))

    errs_arr = np.array(errs) if errs else np.array([np.nan])
    print(f"grid {len(xs)}x{len(ys)}: detected {len(errs)}/{len(rows)} (missed {missing})")
    print(f"  error mean {errs_arr.mean() * 1000:.1f} mm  p95 {np.percentile(errs_arr, 95) * 1000:.1f} mm "
          f"max {errs_arr.max() * 1000:.1f} mm")
    worst = sorted([r for r in rows if r[2] is not None], key=lambda r: -r[2])[:4]
    for x, y, e in worst:
        print(f"  worst ({x:.2f},{y:+.2f}) err {e * 1000:.1f} mm")

    calibration.save(
        H,
        OUT,
        extra={
            "nominal_red_area": nominal,
            "center_offset_m": bias.round(6).tolist(),
            "marker_residual_mm": info["marker_residual_mm"],
            "goal_error_mm": info["goal_error_mm"],
            "bias_spread_mm": round(spread * 1000, 2),
            "grid_error_mean_mm": round(float(errs_arr.mean()) * 1000, 2),
            "grid_error_p95_mm": round(float(np.percentile(errs_arr, 95)) * 1000, 2),
            "grid_error_max_mm": round(float(errs_arr.max()) * 1000, 2),
            "grid_detected": f"{len(errs)}/{len(rows)}",
            "camera": cfg.camera.name,
            "resolution": [cfg.camera.width, cfg.camera.height],
        },
    )
    print(f"wrote {OUT}")
    env.close()


if __name__ == "__main__":
    main()
