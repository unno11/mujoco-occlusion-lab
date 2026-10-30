"""Can the tip track a commanded point at all? Sweeps a few targets, prints final error."""

from __future__ import annotations

import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from occlab.config import load_config  # noqa: E402
from occlab.envs.push_env import PushEnv  # noqa: E402

TARGETS = [
    (0.35, -0.20, 0.045),
    (0.62, 0.15, 0.045),
    (0.30, -0.15, 0.030),
    (0.45, -0.25, 0.045),
]


def main() -> None:
    cfg = load_config()
    env = PushEnv(cfg, render=False)
    for xyz in TARGETS:
        env.reset(seed=0)
        env.control.hold_orientation = None
        errs = []
        for _ in range(80):
            env.control.command_tool(np.asarray(xyz, float))
            errs.append(np.linalg.norm(env.tool_position() - np.asarray(xyz, float)))
        tip = env.tool_position()
        print(f"target {np.round(xyz, 3)} -> tip {np.round(tip, 4)}  err {errs[-1] * 1000:.0f} mm "
              f"(best {min(errs) * 1000:.0f} mm)")
        R = env.data.xmat[env.data.body("hand").id].reshape(3, 3)
        print(f"   finger dir {np.round(R @ np.array([0, 0, 1.0]), 2)}")
    env.close()


if __name__ == "__main__":
    main()
