import argparse
import json
from pathlib import Path

import numpy as np
import pcg_benchmark

SETTINGS = {
    "low":  {"enemies": 6,  "jumps": 15, "coins": 2},
    "mid":  {"enemies": 10, "jumps": 20, "coins": 8},
    "high": {"enemies": 15, "jumps": 25, "coins": 12},
}

ap = argparse.ArgumentParser()
ap.add_argument("--root", default="eval")
ap.add_argument("--out", default="results.json")
args = ap.parse_args()

env = pcg_benchmark.make("smb-v0")
problem = next(
    (getattr(env, attr, None) for attr in ("_problem", "problem", "_game", "game")
     if getattr(env, attr, None) is not None),
    None,
)
if problem is None:
    raise RuntimeError("Could not access the benchmark problem for per-level controllability")
records = []

for gen_dir in sorted(p for p in Path(args.root).iterdir() if p.is_dir()):
    for setting, control in SETTINGS.items():
        set_dir = gen_dir / setting
        if not set_dir.is_dir():
            continue
        for batch in sorted(p for p in set_dir.iterdir() if p.is_dir()):
            files = sorted(batch.glob("level_*.json"),
                           key=lambda p: int(p.stem.split("_")[1]))
            levels = [[int(x) for x in json.loads(f.read_text())] for f in files]
            if len(levels) < 2:
                print(f"skip {batch}: need at least 2 levels")
                continue
            controls = [control] * len(levels)
            q, d, k_raw, _, infos = env.evaluate(levels, controls)

            done = [i for i, info in enumerate(infos) if float(info["complete"]) >= 1.0]
            if not done:
                k_mean = float("nan")
            else:
                per_level = [
                    problem.controlability(infos[i], control)
                    for i in done
                ]
                k_mean = float(np.mean(per_level))
            rec = {
                "generator": gen_dir.name, "setting": setting, "batch": batch.name,
                "n": len(levels), "quality": float(q), "diversity": float(d),
                "controlability": k_mean,
                "controlability_mean": k_mean,
                "controlability_raw": float(k_raw),
                "controlability_completed": k_mean,
                "completion_rate": len(done) / len(levels),
            }
            records.append(rec)
            print(f"{gen_dir.name:8s} {setting:5s} {batch.name:5s} n={len(levels):3d} "
                  f"Q={q:.3f} D={d:.3f} K={k_mean:.3f} Kraw={k_raw:.3f} "
                  f"compl={rec['completion_rate']:.2f}")

Path(args.out).write_text(json.dumps(records, indent=2))
print(f"wrote {len(records)} batches to {args.out}")
