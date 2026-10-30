"""Run the scripted push over N seeds and report the success rate.

Object position comes from the simulator here (oracle mode). This is the Phase 1
plant/controller check: it measures the environment and the pushing controller,
NOT a perception method.

    ./.venv/bin/python scripts/11_batch_oracle.py --seeds 10
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from occlab.config import load_config  # noqa: E402
from occlab.envs.push_env import PushEnv  # noqa: E402
from occlab.planning.push_policy import PushPolicy  # noqa: E402


def run_episode(env: PushEnv, policy: PushPolicy, seed: int) -> dict:
    env.reset(seed=seed)
    policy.reset(env.object_truth_xy(), env.goal())
    done = False
    truncated = False
    info: dict = {}
    while not done:
        target = policy.act(env.object_truth_xy(), env.tool_position(), env.goal())
        _, _, done, truncated, info = env.step(target)
    m = env.get_metrics()
    m.pop("trajectory", None)
    m["replans"] = policy.replans
    m["truncated"] = bool(truncated)
    return m


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--out", type=str, default="results/oracle_batch.json")
    args = ap.parse_args()

    cfg = load_config()
    env = PushEnv(cfg, render=False)
    policy = PushPolicy(cfg.task)

    rows = []
    for seed in range(args.start, args.start + args.seeds):
        rows.append(run_episode(env, policy, seed))
    env.close()

    n = len(rows)
    succ = sum(r["success"] for r in rows)
    steps = [r["steps"] for r in rows]
    print(f"seeds={args.start}..{args.start + n - 1}   success {succ}/{n} = {succ / n:.0%}")
    print(f"steps   mean={np.mean(steps):.0f}  min={min(steps)}  max={max(steps)}")
    print(f"replans mean={np.mean([r['replans'] for r in rows]):.1f}")
    for r in rows:
        print(
            f"  seed {r['seed']:3d}  success={str(r['success']):5s} steps={r['steps']:3d} "
            f"final={r['final_dist']:.4f} replans={r['replans']} box={r['final_box_xy']}"
        )

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=1), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0 if succ == n else 1


if __name__ == "__main__":
    raise SystemExit(main())
