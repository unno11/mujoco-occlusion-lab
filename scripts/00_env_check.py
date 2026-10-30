"""Phase 0 smoke check: load the scene, step it, render one frame.

Run from the project root:
    MUJOCO_GL=egl ./.venv/bin/python scripts/00_env_check.py
"""

from __future__ import annotations

import os
import pathlib

import mujoco
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCENE = ROOT / "assets" / "panda" / "scene_push.xml"


def main() -> None:
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    data = mujoco.MjData(model)

    print(f"mujoco          {mujoco.__version__}")
    print(f"nq/nv/nu        {model.nq}/{model.nv}/{model.nu}")
    print(f"nbody/ngeom     {model.nbody}/{model.ngeom}")
    print(f"cameras         {[model.camera(i).name for i in range(model.ncam)]}")
    print(f"sites           {[model.site(i).name for i in range(model.nsite)]}")
    print(f"actuators       {[model.actuator(i).name for i in range(model.nu)]}")
    print(f"keyframes       {[model.key(i).name for i in range(model.nkey)]}")

    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    for _ in range(500):
        mujoco.mj_step(model, data)

    hand = data.body("hand").xpos
    box = data.body("box").xpos
    tip = 0.5 * (data.body("left_finger").xpos + data.body("right_finger").xpos)
    print(f"hand            {np.round(hand, 4)}")
    print(f"finger midpoint {np.round(tip, 4)}")
    print(f"box             {np.round(box, 4)}")
    print(f"box fell?       {'no' if abs(box[2] - 0.028) < 5e-3 else 'YES z=' + str(round(float(box[2]), 4))}")

    out = ROOT / "results" / "smoke.png"
    out.parent.mkdir(exist_ok=True)
    renderer = mujoco.Renderer(model, height=480, width=640)
    renderer.update_scene(data, camera="rgb")
    frame = renderer.render()
    renderer.close()

    import imageio.v2 as imageio

    imageio.imwrite(out, frame)
    print(f"rendered        {out}  mean_pixel={frame.mean():.1f}  os={os.name}")


if __name__ == "__main__":
    main()
