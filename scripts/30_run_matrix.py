"""Batch evaluation: conditions x policies x seeds, with per-step bookkeeping.

    MUJOCO_GL=egl ./.venv/bin/python scripts/30_run_matrix.py --seeds 3          # smoke
    MUJOCO_GL=egl ./.venv/bin/python scripts/30_run_matrix.py --seeds 20 --out results/matrix.json

Every policy sees exactly the same start states (paired seeds), the same scene and
the same budget. Truth is read only for scoring and for the oracle baseline.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from occlab.config import load_config  # noqa: E402
from occlab.envs.push_env import PushEnv  # noqa: E402
from occlab.planning.policies import make_agent  # noqa: E402

log = logging.getLogger("matrix")


def run_episode(cfg, condition: str, policy: str, seed: int, render_scale: float = 0.5) -> dict:
    env = PushEnv(cfg, render=True, condition=condition, render_scale=render_scale)
    agent = make_agent(policy, cfg.task, cfg.control.control_dt)
    obs = env.reset(seed=seed)
    agent.reset(obs, env.goal(), truth_xy=env.object_truth_xy() if agent.uses_truth else None)

    rows = []
    done = False
    truncated = False
    info: dict = {}
    while not done:
        truth = env.object_truth_xy()
        target = agent.act(obs, truth_xy=truth if agent.uses_truth else None)
        obs, _, done, truncated, info = env.step(target)
        st = agent.status()
        rows.append(
            {
                "step": env.step_count,
                "truth_xy": truth.round(5).tolist(),
                "belief_xy": st["belief_xy"],
                "regime": st["belief_regime"],
                "visible": info["det_visible"],
                "occluder_z": round(info["occluder_z"], 4),
                "dist": round(info["dist"], 5),
            }
        )

    metrics = env.get_metrics()
    events = metrics["plate_events"]
    occluded = [r for r in rows if r["occluder_z"] > -0.2]
    occ_errors = [
        float(np.linalg.norm(np.array(r["belief_xy"]) - np.array(r["truth_xy"])))
        for r in occluded
        if r["belief_xy"] is not None
    ]
    # re-localisation: steps between the plate going down and the first measured frame
    reloc = None
    reveal = next((e["step"] for e in events if e["event"] == "revealed"), None)
    if reveal is not None:
        after = [r for r in rows if r["step"] > reveal and r["regime"] == "measured"]
        if after:
            reloc = after[0]["step"] - reveal

    out = {
        "condition": condition,
        "policy": policy,
        "seed": seed,
        "success": metrics["success"],
        "steps": metrics["steps"],
        "sim_time_s": metrics["sim_time_s"],
        "final_dist": metrics["final_dist"],
        "occluded_steps": len(occluded),
        "est_error_mean_mm": round(float(np.mean(occ_errors)) * 1000, 1) if occ_errors else None,
        "est_error_max_mm": round(float(np.max(occ_errors)) * 1000, 1) if occ_errors else None,
        "relocalise_steps": reloc,
        "waits": agent.waits,
        "recoveries": agent.recoveries,
        "plate_events": events,
        "timeline": rows if condition != "A" else None,
    }
    env.close()
    return out


def summarise(results: list[dict]) -> dict:
    summary: dict[str, dict] = {}
    for cond in sorted({r["condition"] for r in results}):
        for pol in sorted({r["policy"] for r in results}):
            sel = [r for r in results if r["condition"] == cond and r["policy"] == pol]
            if not sel:
                continue
            succ = [r for r in sel if r["success"]]
            errs = [r["est_error_mean_mm"] for r in sel if r["est_error_mean_mm"] is not None]
            summary.setdefault(cond, {})[pol] = {
                "n": len(sel),
                "success": len(succ),
                "success_rate": round(len(succ) / len(sel), 3),
                "steps_mean": round(float(np.mean([r["steps"] for r in sel])), 1),
                "sim_time_mean_s": round(float(np.mean([r["sim_time_s"] for r in sel])), 2),
                "est_error_mean_mm": round(float(np.mean(errs)), 1) if errs else None,
                "waits_mean": round(float(np.mean([r["waits"] for r in sel])), 1),
                "recoveries_mean": round(float(np.mean([r["recoveries"] for r in sel])), 1),
                "relocalise_mean_steps": (
                    round(float(np.mean([r["relocalise_steps"] for r in sel if r["relocalise_steps"] is not None])), 1)
                    if any(r["relocalise_steps"] is not None for r in sel)
                    else None
                ),
            }
    return summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--conditions", default="A,B,C")
    ap.add_argument("--policies", default="oracle,current-obs,history")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--start-seed", type=int, default=0)
    ap.add_argument("--out", default="results/matrix.json")
    ap.add_argument("--render-scale", type=float, default=0.5,
                    help="render resolution factor for detection (0.5 = 320x480 is 4x cheaper)")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.WARNING if args.quiet else logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")
    cfg = load_config()

    conditions = [c.strip().upper() for c in args.conditions.split(",") if c.strip()]
    policies = [p.strip() for p in args.policies.split(",") if p.strip()]
    seeds = list(range(args.start_seed, args.start_seed + args.seeds))

    results: list[dict] = []
    t0 = time.time()
    total = len(conditions) * len(policies) * len(seeds)
    done = 0
    for cond in conditions:
        for pol in policies:
            for seed in seeds:
                r = run_episode(cfg, cond, pol, seed, render_scale=args.render_scale)
                results.append(r)
                done += 1
                print(
                    f"[{done:3d}/{total}] cond {cond} policy {pol:11s} seed {seed:2d} "
                    f"success={str(r['success']):5s} steps={r['steps']:3d} "
                    f"est_err={r['est_error_mean_mm']} mm waits={r['waits']} rec={r['recoveries']}",
                    flush=True,
                )

    summary = summarise(results)
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "episodes": results}, indent=1), encoding="utf-8")

    csv_path = out.with_suffix(".csv")
    keys = ["condition", "policy", "seed", "success", "steps", "sim_time_s", "final_dist",
            "occluded_steps", "est_error_mean_mm", "est_error_max_mm", "relocalise_steps",
            "waits", "recoveries"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in results:
            w.writerow(r)

    print("\n=== summary ===")
    for cond, per_pol in summary.items():
        print(f"condition {cond}")
        for pol, s in per_pol.items():
            print(f"  {pol:11s} success {s['success']}/{s['n']} ({s['success_rate']:.0%})  "
                  f"steps {s['steps_mean']:.0f}  est_err {s['est_error_mean_mm']} mm  "
                  f"waits {s['waits_mean']}  rec {s['recoveries_mean']}  "
                  f"reloc {s['relocalise_mean_steps']}")
    print(f"\nwrote {out} and {csv_path}  ({time.time() - t0:.1f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
