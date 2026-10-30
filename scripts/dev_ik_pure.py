"""Pure IK test: no simulator in the loop, just mink solving to a tip target.

Separates 'the pose is unreachable' from 'the simulated robot cannot follow'.
"""

from __future__ import annotations

import pathlib
import sys

import mink
import mujoco
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from occlab.config import load_config  # noqa: E402


def main() -> None:
    cfg = load_config()
    model = mujoco.MjModel.from_xml_path(str(cfg.scene_path))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    mujoco.mj_forward(model, data)

    conf = mink.Configuration(model, data.qpos)
    task = mink.FrameTask("hand", "body", position_cost=1.0, orientation_cost=0.0)
    limits = [mink.ConfigurationLimit(model), mink.VelocityLimit(model, {f"joint{i}": 2.0 for i in range(1, 8)})]

    for target in ((0.45, 0.0, 0.045), (0.45, 0.0, 0.16)):
        conf.update(data.qpos)  # always start from home
        tgt = np.asarray(target, float)
        for it in range(600):
            data.qpos[:] = conf.q
            mujoco.mj_forward(model, data)
            tip = 0.5 * (data.xpos[model.body("left_finger").id] + data.xpos[model.body("right_finger").id])
            hand = data.xpos[model.body("hand").id]
            offset = tip - hand
            task.set_target(mink.SE3.from_translation(tgt - offset))
            vel = mink.solve_ik(conf, [task], 0.01, solver=cfg.control.solver, limits=limits)
            conf.integrate_inplace(vel, 0.01)

        data.qpos[:] = conf.q
        mujoco.mj_forward(model, data)
        tip = 0.5 * (data.xpos[model.body("left_finger").id] + data.xpos[model.body("right_finger").id])
        R = data.xmat[model.body("hand").id].reshape(3, 3)
        err = np.linalg.norm(tip - tgt)
        print(f"target {np.round(tgt, 3)} -> tip {np.round(tip, 4)}  err {err * 1000:.0f} mm")
        print(f"   q {np.round(conf.q[:7], 2)}")
        print(f"   finger dir {np.round(R @ np.array([0, 0, 1.0]), 2)}")


if __name__ == "__main__":
    main()
