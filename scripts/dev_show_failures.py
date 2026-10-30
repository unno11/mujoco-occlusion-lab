"""Print the failure rows of a batch result file."""

from __future__ import annotations

import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "results/oracle_batch.json")
rows = json.loads(path.read_text(encoding="utf-8"))
bad = [r for r in rows if not r["success"]]
print(f"total {len(rows)}  fail {len(bad)}")
for r in bad:
    print(f"  seed {r['seed']:3d} steps={r['steps']:3d} final={r['final_dist']:.4f} box={r['final_box_xy']}")
ok = sorted(r["steps"] for r in rows if r["success"])
print("success steps:", ok)
