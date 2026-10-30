"""Diagnostic: why does the tool stop descending? Dumps pose, limits and contacts."""

from __future__ import annotations

import pathlib
import sys

import mujoco
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from occlab.config import load_config  # noqa: E402
from occlab.envs.push_env import PushEnv  # noqa: E402
from occlab.planning.push_policy import PushPolicy  # noqa: E402


def contact_report(env) -> None:
    m, d = env.model, env.data
    names = []
    for i in range(d.ncon):
        c = d.contact[i]
        g1 = m.geom(c.geom1).name
        g2 = m.geom(c.geom2).name
        names.append((g1, g2))
    print(f"  contacts ({d.ncon}): {names}")


def main() -> None:
    cfg = load_config()
    env = PushEnv(cfg, render=True)
    env.reset(seed=0)
    pol = PushPolicy(cfg.task)
    pol.reset(env.object_truth_xy(), env.goal())

    for k in range(160):
        tgt = pol.act(env.object_truth_xy(), env.tool_position(), env.goal())
        env.step(tgt)
        if k in (60, 100, 150):
            m = env.model
            hand = env.data.body("hand").xpos
            tip = env.tool_position()
            print(f"step {k+1} phase={pol.phase.value}")
            print(f"  target {np.round(tgt, 4)}  tip {np.round(tip, 4)}  hand {np.round(hand, 4)}")
            print(f"  qpos arm {np.round(env.data.qpos[:7], 3)}")
            lo, hi = m.jnt_range[:7, 0], m.jnt_range[:7, 1]
            q = env.data.qpos[:7]
            print(f"  at_limit {np.round(np.minimum(q - lo, hi - q), 4)}")
            print(f"  box {np.round(env.object_truth_xy(), 4)}  boxz {env.object_truth_pose()[2]:.4f}")
            contact_report(env)
            import imageio.v2 as imageio

            out = ROOT / "results" / f"diag_{k+1}.png"
            out.parent.mkdir(exist_ok=True)
            imageio.imwrite(out, env.render())
            print(f"  frame -> {out}")

    env.close()


if __name__ == "__main__":
    main()
