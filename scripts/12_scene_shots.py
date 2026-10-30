"""Render the scene for inspection: one overview + the RGB view under three occlusion states.

    MUJOCO_GL=egl ./.venv/bin/python scripts/12_scene_shots.py
"""

from __future__ import annotations

import pathlib
import sys

import cv2
import imageio.v2 as imageio
import mujoco
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from occlab.config import load_config  # noqa: E402
from occlab.envs.push_env import PushEnv  # noqa: E402

OUT = ROOT / "results" / "scene"


def label(img: np.ndarray, text: str) -> np.ndarray:
    img = img.copy()
    cv2.rectangle(img, (0, 0), (img.shape[1], 30), (20, 20, 20), -1)
    cv2.putText(img, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = load_config()
    env = PushEnv(cfg, render=True)
    env.reset(seed=0)

    box = env.object_truth_xy()
    goal = env.goal()
    print(f"box {np.round(box, 3)}  goal {np.round(goal, 3)}  half_size {cfg.task.box_half_size}")

    # --- overview from a free camera, occluder retracted -------------------
    env.set_occluder(cfg.occluder.retracted_z)
    renderer = mujoco.Renderer(env.model, height=640, width=960)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(env.model, cam)
    cam.lookat[:] = (0.42, 0.0, 0.10)
    cam.distance = 1.45
    cam.azimuth = 145.0
    cam.elevation = -18.0
    renderer.update_scene(env.data, camera=cam)
    overview = renderer.render().copy()
    imageio.imwrite(OUT / "00_overview.png", overview)  # imageio keeps RGB order
    print("wrote 00_overview.png")

    # --- the policy's RGB view in three occlusion states -------------------
    states = [
        ("A no occlusion (plate retracted)", cfg.occluder.retracted_z),
        ("B partial occlusion (plate hides lower half)", env.occlusion_z_for(0.55)),
        ("C full occlusion (plate stands on the table)", 0.0),
    ]
    shots = []
    for name, z in states:
        env.set_occluder(z)
        img = env.render()
        imageio.imwrite(OUT / f"{name.split()[0]}_rgb.png", img)
        shots.append(label(img, name))
        print(f"{name}: occluder body z = {z:.3f}")

    # --- contact sheet -----------------------------------------------------
    top = label(cv2.cvtColor(overview, cv2.COLOR_RGB2BGR), "scene overview")
    top = cv2.resize(top, (960, 640))
    bgr = [cv2.cvtColor(s, cv2.COLOR_RGB2BGR) for s in shots]
    row = np.hstack([cv2.resize(b, (320, 240)) for b in bgr])
    sheet = np.vstack([top, row])
    cv2.imwrite(str(OUT / "sheet.png"), sheet)
    print(f"wrote {OUT / 'sheet.png'}  {sheet.shape}")

    env.close()
    renderer.close()


if __name__ == "__main__":
    main()
