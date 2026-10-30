"""Headline demo video: occluded pushing, full render resolution.

    MUJOCO_GL=egl ./.venv/bin/python scripts/40_demo_video.py --condition C --policy history --seed 0

Writes results/demo/<condition>_<policy>_seed<seed>.{mp4,gif} and prints the metrics.
The overlay is drawn on the frame: step counter, whether the object is currently
detected, and the current belief (history policy only).
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import cv2
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from occlab.config import load_config  # noqa: E402
from occlab.envs.push_env import PushEnv  # noqa: E402
from occlab.planning.policies import make_agent  # noqa: E402


def overlay(frame: np.ndarray, text: list[str], good: bool) -> np.ndarray:
    img = frame.copy()
    h, w = img.shape[:2]
    cv2.rectangle(img, (0, 0), (w, 26 * len(text) + 8), (18, 18, 18), -1)
    for i, line in enumerate(text):
        cv2.putText(img, line, (8, 22 + 26 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (120, 220, 160) if good else (200, 200, 200), 1, cv2.LINE_AA)
    return img


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", default="C")
    ap.add_argument("--policy", default="history")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--every", type=int, default=2)
    ap.add_argument("--fps", type=int, default=25)
    args = ap.parse_args()

    cfg = load_config()
    env = PushEnv(cfg, render=True, condition=args.condition, render_scale=1.0)
    agent = make_agent(args.policy, cfg.task, cfg.control.control_dt)
    obs = env.reset(seed=args.seed)
    agent.reset(obs, env.goal(), truth_xy=env.object_truth_xy() if agent.uses_truth else None)

    frames = []
    done = False
    while not done:
        truth = env.object_truth_xy()
        target = agent.act(obs, truth_xy=truth if agent.uses_truth else None)
        obs, _, done, truncated, info = env.step(target)
        if env.step_count % args.every == 0 or done:
            st = agent.status()
            lines = [
                f"step {env.step_count:3d}   object visible: {'yes' if info['det_visible'] else 'NO'}",
                f"policy: {args.policy}   condition: {args.condition}",
            ]
            if st["belief_xy"] is not None:
                lines.append(f"belief ({st['belief_xy'][0]:.3f}, {st['belief_xy'][1]:.3f})  "
                             f"sd {st['belief_sd_mm']:.0f} mm  {st['belief_regime']}")
            frames.append(overlay(cv2.cvtColor(env.render(), cv2.COLOR_RGB2BGR), lines, bool(info["det_visible"])))
        if truncated:
            break

    m = env.get_metrics()
    print(f"success={m['success']} steps={m['steps']} final_dist={m['final_dist']} "
          f"plate_events={m['plate_events']}")

    out = ROOT / "results" / "demo"
    out.mkdir(parents=True, exist_ok=True)
    stem = f"{args.condition}_{args.policy}_seed{args.seed}"
    if frames:
        h, w = frames[0].shape[:2]
        writer = cv2.VideoWriter(str(out / f"{stem}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                                 args.fps, (w, h))
        for f in frames:
            writer.write(f)
        writer.release()
        import imageio.v2 as imageio

        imageio.mimwrite(out / f"{stem}.gif",
                         [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames[::3]],
                         fps=12, macro_block_size=None)
        print(f"wrote {out / (stem + '.mp4')} and .gif  ({len(frames)} frames)")
    env.close()


if __name__ == "__main__":
    main()
