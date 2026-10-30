"""Trace one episode of a given policy: belief, visibility, phases, distance.

    MUJOCO_GL=egl ./.venv/bin/python scripts/dev_trace_policy.py --policy history --condition A --seed 0
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from occlab.config import load_config  # noqa: E402
from occlab.envs.push_env import PushEnv  # noqa: E402
from occlab.planning.policies import make_agent  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default="history")
    ap.add_argument("--condition", default="A")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--every", type=int, default=20)
    args = ap.parse_args()

    cfg = load_config()
    env = PushEnv(cfg, render=True, condition=args.condition)
    agent = make_agent(args.policy, cfg.task, cfg.control.control_dt)
    obs = env.reset(seed=args.seed)
    agent.reset(obs, env.goal(), truth_xy=env.object_truth_xy() if agent.uses_truth else None)
    print(f"start box {np.round(env.object_truth_xy(), 3)}  goal {np.round(env.goal(), 3)}")

    done = False
    while not done:
        truth = env.object_truth_xy()
        target = agent.act(obs, truth_xy=truth if agent.uses_truth else None)
        obs, _, done, truncated, info = env.step(target)
        st = agent.status()
        if env.step_count % args.every == 0 or done:
            b = st["belief_xy"]
            print(
                f"{env.step_count:3d} {agent.push.phase.value:8s} vis={str(info['det_visible'])[0]} "
                f"regime={str(st['belief_regime']):9s} belief={np.round(b, 3) if b else None} "
                f"truth={np.round(truth, 3)} box={np.round(env.object_truth_xy(), 3)} "
                f"dist={info['dist']:.3f} wait={st['waits']} rec={st['recoveries']} "
                f"plate={info['occluder_z']:+.2f} tgt={np.round(target, 3)}"
            )
        if truncated:
            break
    m = env.get_metrics()
    print(f"success={m['success']} steps={m['steps']} final={m['final_dist']} events={m['plate_events']}")
    env.close()


if __name__ == "__main__":
    main()
