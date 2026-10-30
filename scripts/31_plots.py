"""Plots for the comparison matrix: success rate, estimate error, waiting time.

    ./.venv/bin/python scripts/31_plots.py --in results/matrix.json --out figures
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]

POLICY_STYLE = {
    "oracle": ("#444444", "Oracle (truth)"),
    "current-obs": ("#d1495b", "Current observation"),
    "history": ("#2a9d8f", "History-aware"),
}
COND_LABEL = {
    "A": "A: no occlusion",
    "B": "B: partial",
    "C": "C: full, temporary",
    "D": "D: full, persistent",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="results/matrix.json")
    ap.add_argument("--out", dest="out", default="figures")
    args = ap.parse_args()

    data = json.loads((ROOT / args.inp).read_text(encoding="utf-8"))
    episodes = data["episodes"]
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)

    conditions = sorted({e["condition"] for e in episodes})
    policies = [p for p in ("oracle", "current-obs", "history")
                if any(e["policy"] == p for e in episodes)]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))

    # 1. success rate
    ax = axes[0]
    width = 0.8 / max(1, len(policies))
    for i, pol in enumerate(policies):
        rates, ns = [], []
        for cond in conditions:
            sel = [e for e in episodes if e["condition"] == cond and e["policy"] == pol]
            rates.append(sum(e["success"] for e in sel) / len(sel) if sel else np.nan)
            ns.append(len(sel))
        xs = np.arange(len(conditions)) + i * width - 0.4 + width / 2
        ax.bar(xs, rates, width, color=POLICY_STYLE[pol][0], label=POLICY_STYLE[pol][1])
    ax.set_xticks(np.arange(len(conditions)))
    ax.set_xticklabels([COND_LABEL.get(c, c) for c in conditions], fontsize=8)
    ax.set_ylabel("success rate")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8, frameon=False)

    # 2. estimate error while occluded
    ax = axes[1]
    for pol in policies:
        if pol == "oracle":
            continue
        vals, labels = [], []
        for cond in conditions:
            sel = [e["est_error_mean_mm"] for e in episodes
                   if e["condition"] == cond and e["policy"] == pol and e["est_error_mean_mm"] is not None]
            if sel:
                vals.append(np.mean(sel))
                labels.append(cond)
        if vals:
            ax.bar([f"{pol}\n{c}" for c in labels], vals, color=POLICY_STYLE[pol][0])
    ax.set_ylabel("estimate error while occluded (mm)")
    ax.tick_params(axis="x", labelsize=7)
    ax.grid(axis="y", alpha=0.3)

    # 3. steps: work done, split by condition
    ax = axes[2]
    for i, pol in enumerate(policies):
        means = []
        for cond in conditions:
            sel = [e["steps"] for e in episodes if e["condition"] == cond and e["policy"] == pol]
            means.append(np.mean(sel) if sel else np.nan)
        xs = np.arange(len(conditions)) + i * width - 0.4 + width / 2
        ax.bar(xs, means, width, color=POLICY_STYLE[pol][0], label=POLICY_STYLE[pol][1])
    ax.set_xticks(np.arange(len(conditions)))
    ax.set_xticklabels([COND_LABEL.get(c, c) for c in conditions], fontsize=8)
    ax.set_ylabel("control steps (mean)")
    ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Occlusion manipulation lab: pushing with a fixed RGB camera", fontsize=11)
    fig.tight_layout()
    path = out / "comparison.png"
    fig.savefig(path, dpi=150)
    print(f"wrote {path}")

    # text table as well, so numbers can be quoted without reading the plot
    rows = ["| condition | policy | success | steps | est error (mm) | waits | recoveries |",
            "| --- | --- | --- | --- | --- | --- | --- |"]
    for cond in conditions:
        for pol in policies:
            sel = [e for e in episodes if e["condition"] == cond and e["policy"] == pol]
            if not sel:
                continue
            succ = sum(e["success"] for e in sel)
            errs = [e["est_error_mean_mm"] for e in sel if e["est_error_mean_mm"] is not None]
            rows.append(
                f"| {COND_LABEL.get(cond, cond)} | {POLICY_STYLE[pol][1]} | {succ}/{len(sel)} | "
                f"{np.mean([e['steps'] for e in sel]):.0f} | "
                f"{(f'{np.mean(errs):.1f}' if errs else '-')} | "
                f"{np.mean([e['waits'] for e in sel]):.0f} | "
                f"{np.mean([e['recoveries'] for e in sel]):.1f} |"
            )
    table = out / "summary.md"
    table.write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(f"wrote {table}")


if __name__ == "__main__":
    main()
