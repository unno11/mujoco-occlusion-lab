"""Log the first pusher-box contact: which step, where in z, and what the policy was doing."""

from __future__ import annotations

import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from occlab.config import load_config  # noqa: E402
from occlab.envs.push_env import PushEnv  # noqa: E402
from occlab.planning.push_policy import PushPolicy  # noqa: E402


def main() -> None:
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    cfg = load_config()
    env = PushEnv(cfg, render=False)
    env.reset(seed=seed)
    pol = PushPolicy(cfg.task)
    pol.reset(env.object_truth_xy(), env.goal())
    m, d = env.model, env.data
    pusher = m.geom("pusher").id
    box = m.geom("box").id
    half = cfg.task.box_half_size

    first = None
    for k in range(cfg.task.max_control_steps):
        phase_before = pol.phase
        tgt = pol.act(env.object_truth_xy(), env.tool_position(), env.goal())
        env.step(tgt)

        touching = False
        for i in range(d.ncon):
            c = d.contact[i]
            if {c.geom1, c.geom2} == {pusher, box}:
                touching = True
                if first is None:
                    first = (k + 1, phase_before.value, c.pos.copy(), float(env.tool_position()[2]))
                break

        if k % 20 == 0 or (touching and first and first[0] == k + 1):
            ball = env.tool_position()
            obj = env.object_truth_xy()
            gap = float(np.linalg.norm(ball[:2] - obj)) - half - 0.012
            print(
                f"{k + 1:3d} {phase_before.value:8s} ball={np.round(ball, 3)} "
                f"box={np.round(obj, 3)} gap={gap * 1000:6.1f}mm touch={touching}"
            )

    print()
    if first is None:
        print("NO CONTACT between pusher and box within the budget")
    else:
        step, phase, pos, bz = first
        print(f"first contact at step {step}, phase={phase}")
        print(f"  contact pos {np.round(pos, 4)}  (box centre z = {half:.3f}, box top z = {2 * half:.3f})")
        print(f"  ball z at that moment = {bz:.4f}")
        where = "SIDE" if pos[2] < 2 * half - 0.015 else "TOP / corner"
        print(f"  -> contact is on the {where}")

    m2 = env.get_metrics()
    print(f"\nresult: success={m2['success']} final_dist={m2['final_dist']} steps={m2['steps']}")
    env.close()


if __name__ == "__main__":
    main()
