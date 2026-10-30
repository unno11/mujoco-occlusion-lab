"""Phase 1 demo: one pushing episode with the oracle (ground-truth) object position.

This exists to validate the plant + controller + success check before any vision
is involved. The oracle is NOT a perception method: it reads the simulator state,
so its numbers are an upper bound, never a vision result.

    MUJOCO_GL=egl ./.venv/bin/python scripts/10_push_demo.py --seed 0 --video
"""

from __future__ import annotations

import argparse
import logging
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from occlab.config import load_config  # noqa: E402
from occlab.envs.push_env import PushEnv  # noqa: E402
from occlab.planning.push_policy import Phase, PushPolicy  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--video", action="store_true")
    ap.add_argument("--video-every", type=int, default=2)
    ap.add_argument("--out", type=str, default="results/phase1")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    cfg = load_config()

    env = PushEnv(cfg, render=args.video)
    policy = PushPolicy(cfg.task)

    obs = env.reset(seed=args.seed)
    policy.reset(env.object_truth_xy(), env.goal())
    print(f"start box   {np.round(env.object_truth_xy(), 3)}   goal {np.round(env.goal(), 3)}")

    frames: list[np.ndarray] = []
    if args.video:
        frames.append(env.render())

    done = False
    info: dict = {}
    while not done:
        target = policy.act(env.object_truth_xy(), env.tool_position(), env.goal())
        obs, reward, done, truncated, info = env.step(target)
        if args.video and env.step_count % args.video_every == 0:
            frames.append(env.render())
        if env.step_count % 25 == 0:
            print(
                f"step {env.step_count:3d}  phase={policy.phase.value:8s} "
                f"tool={np.round(env.tool_position(), 3)} box={np.round(env.object_truth_xy(), 3)} "
                f"dist={info['dist']:.3f}"
            )
        if done and not truncated:
            policy.finish()

    m = env.get_metrics()
    print("\n=== episode ===")
    for k, v in m.items():
        if k != "trajectory":
            print(f"  {k:12s} {v}")

    if args.video and frames:
        import imageio.v2 as imageio

        out_dir = ROOT / args.out
        out_dir.mkdir(parents=True, exist_ok=True)
        vid = out_dir / f"oracle_seed{args.seed}.mp4"
        imageio.mimwrite(vid, frames, fps=25, macro_block_size=None)
        print(f"video        {vid}  ({len(frames)} frames)")
        gif = out_dir / f"oracle_seed{args.seed}.gif"
        imageio.mimwrite(gif, frames[::3], fps=12, macro_block_size=None)
        print(f"gif          {gif}")

    env.close()
    return 0 if m["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
