"""Print every end-effector geom: body, world position, lowest z, and its offset along hand +z."""

from __future__ import annotations

import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from occlab.config import load_config  # noqa: E402
from occlab.envs.push_env import PushEnv  # noqa: E402


def main() -> None:
    cfg = load_config()
    env = PushEnv(cfg, render=False)
    env.reset(seed=0)
    m, d = env.model, env.data

    # command a downward-looking pose for a while so the hand is where it stalls
    for _ in range(300):
        env.control.command_tool(np.array([0.38, -0.08, 0.012]))

    hand_id = m.body("hand").id
    R = d.xmat[hand_id].reshape(3, 3)
    finger_dir = R @ np.array([0.0, 0.0, 1.0])
    print(f"hand pos        {np.round(d.xpos[hand_id], 4)}")
    print(f"hand local +z -> world {np.round(finger_dir, 3)}  (finger direction?)")

    print("\nend-effector geoms:")
    for g in range(m.ngeom):
        if int(m.geom_bodyid[g]) not in (hand_id, m.body("left_finger").id, m.body("right_finger").id):
            continue
        if int(m.geom_contype[g]) == 0:
            continue
        rot = d.geom_xmat[g].reshape(3, 3)
        size = m.geom_size[g]
        corners = np.array([[sx * size[0], sy * size[1], sz * size[2]]
                            for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
        world = d.geom_xpos[g] + corners @ rot.T
        along = float((d.geom_xpos[g] - d.xpos[hand_id]) @ finger_dir)
        print(
            f"  {m.body(int(m.geom_bodyid[g])).name:13s} {m.geom(g).name or 'geom%d' % g:10s} "
            f"z=[{world[:, 2].min():+.4f},{world[:, 2].max():+.4f}]  along_finger={along:+.4f}"
        )

    env.close()


if __name__ == "__main__":
    main()
