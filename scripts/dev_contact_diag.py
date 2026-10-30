"""Focused diagnostic for a stalling seed: what is actually touching what, at what height."""

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


def geom_label(model, gid: int) -> str:
    body = model.body(int(model.geom_bodyid[gid])).name
    name = model.geom(gid).name
    return f"{body}:{name or 'geom%d' % gid}"


def low_z(model, data, gid: int) -> float:
    """Lowest world z of a box geom."""
    rot = data.geom_xmat[gid].reshape(3, 3)
    size = model.geom_size[gid]
    corners = np.array([[sx * size[0], sy * size[1], sz * size[2]]
                        for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    return float(np.min((data.geom_xpos[gid] + corners @ rot.T)[:, 2]))


def main() -> None:
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    n_steps = int(sys.argv[2]) if len(sys.argv) > 2 else 205
    cfg = load_config()
    env = PushEnv(cfg, render=False)
    env.reset(seed=seed)
    pol = PushPolicy(cfg.task)
    pol.reset(env.object_truth_xy(), env.goal())
    m, d = env.model, env.data

    for k in range(n_steps):
        tgt = pol.act(env.object_truth_xy(), env.tool_position(), env.goal())
        env.step(tgt)

    print(f"seed {seed}  phase={pol.phase.value}  step={env.step_count}")
    print(f"target {np.round(tgt, 4)}  tip {np.round(env.tool_position(), 4)}")
    print(f"hand   {np.round(d.body('hand').xpos, 4)}")
    print(f"box    {np.round(d.geom('box').xpos, 4)}  low_z={low_z(m, d, m.geom('box').id):.4f}")
    q = d.qpos[:7]
    lo, hi = m.jnt_range[:7, 0], m.jnt_range[:7, 1]
    print(f"qpos   {np.round(q, 3)}")
    print(f"margin {np.round(np.minimum(q - lo, hi - q), 3)}   (to joint limits)")

    print("\ncontacts:")
    for i in range(d.ncon):
        c = d.contact[i]
        print(f"  {geom_label(m, c.geom1):28s} <-> {geom_label(m, c.geom2):28s} "
              f"dist={c.dist:+.4f} pos={np.round(c.pos, 3)}")

    print("\nlowest z per body (world):")
    for bname in ("hand", "left_finger", "right_finger"):
        bid = m.body(bname).id
        zs = [low_z(m, d, g) for g in range(m.ngeom) if int(m.geom_bodyid[g]) == bid]
        if zs:
            print(f"  {bname:14s} min_z={min(zs):.4f}")

    import imageio.v2 as imageio

    png = ROOT / "results" / f"diag_seed{seed}_final.png"
    png.parent.mkdir(exist_ok=True)
    imageio.imwrite(png, env.render())
    print(f"\nframe -> {png}")

    env.close()


if __name__ == "__main__":
    main()
