"""Does the orientation task actually track? Probe it in free space vs near the table."""

from __future__ import annotations

import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from occlab.config import load_config  # noqa: E402
from occlab.envs.push_env import PushEnv  # noqa: E402


def rot_err(achieved: np.ndarray, target: np.ndarray) -> float:
    c = (np.trace(achieved.T @ target) - 1) / 2
    return float(np.degrees(np.arccos(np.clip(c, -1, 1))))


def probe(env, target_xyz, steps=150):
    env.reset(seed=0)
    for _ in range(steps):
        env.control.command_tool(np.asarray(target_xyz, float))
    d = env.data
    hand = d.body("hand").xpos
    R = d.xmat[d.body("hand").id].reshape(3, 3)
    print(f"target xyz {np.round(target_xyz, 3)} -> hand {np.round(hand, 4)}")
    print(f"  finger dir (hand +z) {np.round(R @ np.array([0,0,1.0]), 3)}")
    print(f"  orientation error vs commanded: {rot_err(R, env.control.hold_orientation):.1f} deg")


def main() -> None:
    cfg = load_config()
    env = PushEnv(cfg, render=False)
    env.reset(seed=0)

    # place the box far away so the probe is free space
    for xyz in ((0.45, 0.00, 0.30), (0.45, 0.00, 0.12), (0.45, 0.00, 0.05)):
        probe(env, xyz)
        print()

    print("commanded rotation:\n", env.control.hold_orientation)
    env.close()


if __name__ == "__main__":
    main()
