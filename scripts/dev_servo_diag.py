"""Why does the simulated tip stop ~12 cm above the IK solution? Compare ctrl vs qpos and list contacts."""

from __future__ import annotations

import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from occlab.config import load_config  # noqa: E402
from occlab.envs.push_env import PushEnv  # noqa: E402

TARGET = (0.45, 0.0, 0.045)


def main() -> None:
    cfg = load_config()
    env = PushEnv(cfg, render=False)
    env.reset(seed=0)
    m, d = env.model, env.data
    for _ in range(120):
        env.control.command_tool(np.asarray(TARGET, float))

    print(f"target      {np.round(TARGET, 4)}")
    print(f"tip         {np.round(env.tool_position(), 4)}")
    print(f"hand        {np.round(d.xpos[m.body('hand').id], 4)}")
    print(f"qpos arm    {np.round(d.qpos[:7], 3)}")
    print(f"ctrl arm    {np.round(d.ctrl[:7], 3)}")
    print(f"actuator f  {np.round(d.actuator_force[:7], 1)}")
    print(f"gripper ctrl/dofadr {d.ctrl[7]:.0f}")

    print("\ncontacts:")
    for i in range(d.ncon):
        c = d.contact[i]
        b1 = m.body(int(m.geom_bodyid[c.geom1])).name
        b2 = m.body(int(m.geom_bodyid[c.geom2])).name
        print(f"  {b1:14s}<->{b2:14s} dist={c.dist:+.4f} pos={np.round(c.pos, 3)}")

    print("\nhand link chain positions:")
    for name in ("link4", "link5", "link6", "link7", "hand"):
        print(f"  {name:6s} {np.round(d.xpos[m.body(name).id], 4)}")

    env.close()


if __name__ == "__main__":
    main()
