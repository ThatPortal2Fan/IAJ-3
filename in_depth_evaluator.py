# in_depth_evaluator.py — detailed diagnostics for quality and controlability
import argparse
import json
from pathlib import Path
import numpy as np
import pcg_benchmark
from pcg_benchmark.probs.utils import get_range_reward

# ------------------------------------------------------------------ #
# Parse command-line arguments
# ------------------------------------------------------------------ #
parser = argparse.ArgumentParser(description="Detailed diagnostics for level quality and controlability.")
parser.add_argument("--path", "-p", type=str, default="renders_nrl", help="Path to the directory containing level JSON files.")
args = parser.parse_args()

env = pcg_benchmark.make("smb-v0")

levels = [
    [int(x) for x in json.loads(p.read_text())]
    for p in sorted(Path(args.path).glob("level_*.json"),
                    key=lambda p: int(p.stem.split("_")[1]))
]
print(f"Loaded {len(levels)} levels from '{args.path}'\n")

# ------------------------------------------------------------------ #
# CONTROL — the target playthrough every level in this batch is scored
# against.  Edit these numbers to steer difficulty.
#
#   enemies = how many stomps ASTAR should make
#   jumps   = how many jumps ASTAR should make
#   coins   = how many coins ASTAR should collect
#
# Set CONTROL = None to fall back to the max of the env's control space.
# ------------------------------------------------------------------ #
CONTROL = {
    "enemies": 6, #6  10  15
    "jumps":   15, #15  20  25
    "coins":   2, #2  8  12
}

# ------------------------------------------------------------------ #
# Pull reference thresholds from the problem
# ------------------------------------------------------------------ #
problem = None
for attr in ("_problem", "problem", "_game", "game"):
    problem = getattr(env, attr, None)
    if problem is not None:
        break

empty_thresh    = getattr(problem, "_empty", 0.5)
fenemies_thresh = getattr(problem, "_fenemies", 0.1)
cerror          = getattr(problem, "_cerror",
                          {"enemies": 0, "jumps": 0, "coins": 0})

if CONTROL is None:
    cs = getattr(problem, "_control_space", None)
    space_dict = getattr(cs, "_value", None) if cs else None
    CONTROL = {
        k: int(getattr(v, "_max_value", getattr(v, "max_value", 0)))
        for k, v in space_dict.items()
    }
CONTROL = {k: int(v) for k, v in CONTROL.items()}

# Same target applied to every level
controls = [CONTROL] * len(levels)

# ------------------------------------------------------------------ #
# Evaluate (raw — includes incomplete levels; we correct below)
# ------------------------------------------------------------------ #
quality, diversity, controlability_raw, details, infos = env.evaluate(levels, controls)

# Identify which levels the solver actually completed.  Levels where
# ASTAR didn't finish have truncated playthroughs (fewer enemies / jumps
# / coins than the target), so including them skews controlability down.
completed_mask = [float(info["complete"]) >= 1.0 for info in infos]
completed_idxs = [i for i, c in enumerate(completed_mask) if c]
skipped_idxs   = [i for i, c in enumerate(completed_mask) if not c]

# ------------------------------------------------------------------ #
# Corrected controlability: re-run the env's own aggregation but only
# over the levels the solver completed.  This keeps whatever aggregation
# the wrapper uses (mean / min / threshold) intact.
# ------------------------------------------------------------------ #
if not completed_idxs:
    controlability = float("nan")
    controlability_source = "undefined (no completed levels)"
elif len(completed_idxs) == len(infos):
    controlability = controlability_raw
    controlability_source = f"all {len(infos)} levels completed"
else:
    sub_levels   = [levels[i]   for i in completed_idxs]
    sub_controls = [controls[i] for i in completed_idxs]
    _, _, controlability, _, _ = env.evaluate(sub_levels, sub_controls)
    controlability_source = f"completed-only ({len(completed_idxs)}/{len(infos)} levels)"

# Manual fallback on the completed subset only.
if problem is not None and completed_idxs:
    manual_per_level = [problem.controlability(infos[i], CONTROL) for i in completed_idxs]
    manual_mean = float(np.mean(manual_per_level))
    manual_min  = float(np.min(manual_per_level))
else:
    manual_per_level = []
    manual_mean = float("nan")
    manual_min  = float("nan")

print("=" * 78)
print(f"Quality:        {quality:.4f}  (all levels)")
print(f"Diversity:      {diversity:.4f}  (all levels)")
print(f"Controlability: {controlability:.4f}  — {controlability_source}")
if skipped_idxs:
    print(f"                (raw env.evaluate over all levels: {controlability_raw:.4f} — skewed by incomplete solves)")
print("=" * 78)
print(f"Control target (applied to every level): {CONTROL}")
print(f"Control tolerance:                       {cerror}")
print(f"Reference: empty > {empty_thresh}, fenemies < {fenemies_thresh}")
if skipped_idxs:
    print(f"Excluded (solver incomplete) level indices: {skipped_idxs}")
print()

if manual_per_level:
    print(f"Manual per-level controlability (completed only): {[round(x,3) for x in manual_per_level]}")
    print(f"Manual mean:                     {manual_mean:.4f}")
    print(f"Manual min:                      {manual_min:.4f}")
    if abs(manual_mean - float(controlability_raw)) > 1e-6:
        print(f"  NOTE: raw env.evaluate gave {float(controlability_raw):.4f} over all levels; "
              f"the corrected value above ({controlability:.4f}) excludes incomplete solves.")
    print()

# ------------------------------------------------------------------ #
# QUALITY — recompute MarioProblem.quality's arithmetic per level
# ------------------------------------------------------------------ #
def quality_breakdown(info):
    tube_r  = get_range_reward(info["tube"], 0, 0, 0, 10)
    empty_r = get_range_reward(info["empty"], 0, empty_thresh, 1)
    fenem_r = get_range_reward(info["fenemies"], 0, 0, fenemies_thresh, 1)
    noise_r = 0.0
    if empty_r >= 1.0 and tube_r >= 1.0:
        for n in info["noise"]:
            noise_r += get_range_reward(n, 0, 0, 0, 1)
    noise_r /= 16.0
    complete = float(info["complete"])
    q = (tube_r + empty_r + noise_r + fenem_r + 4 * complete) / 8.0
    return q, tube_r, empty_r, noise_r, fenem_r, complete

print("=== QUALITY per level ===")
hdr = (f"{'#':>3} {'score':>6} | {'tube':>4} {'empty':>5} {'noise':>5} "
       f"{'fenem':>5} {'compl':>5} | {'%empty':>6} {'%fen':>5} {'tube#':>5} | issue")
print(hdr); print("-" * len(hdr))
summary = {"tube": 0, "empty": 0, "noise": 0, "fenemies": 0, "solver": 0, "ok": 0}
for i, info in enumerate(infos):
    q, t, e, n, f, c = quality_breakdown(info)
    if   info["tube"] > 0:                    issue = "TUBE (odd run)";    summary["tube"] += 1
    elif info["empty"] <= empty_thresh:       issue = "EMPTY too low";     summary["empty"] += 1
    elif info["fenemies"] >= fenemies_thresh: issue = "FENEMIES too high"; summary["fenemies"] += 1
    elif n < 1.0:                             issue = "NOISE exceeded";    summary["noise"] += 1
    elif c < 1.0:                             issue = "SOLVER incomplete"; summary["solver"] += 1
    else:                                     issue = "ok";                summary["ok"] += 1
    print(f"{i:>3} {q:>6.3f} | {t:>4.2f} {e:>5.2f} {n:>5.2f} {f:>5.2f} {c:>5.2f} | "
          f"{info['empty']:>6.3f} {info['fenemies']:>5.3f} {info['tube']:>5} | {issue}")
print("-" * len(hdr))
print(f"{summary['ok']}/{len(infos)} fully valid.  Failures: "
      f"tube={summary['tube']} empty={summary['empty']} noise={summary['noise']} "
      f"fenemies={summary['fenemies']} solver={summary['solver']}")
print()

# ------------------------------------------------------------------ #
# CONTROLABILITY — breakdown against the CONTROL we just applied.
# Only completed levels contribute to the mean / weakest dimension.
# ------------------------------------------------------------------ #
def ctrl_breakdown(info, ctrl):
    r_e = get_range_reward(info["enemies"], 0,
                           max(0, ctrl["enemies"] - cerror["enemies"]),
                           ctrl["enemies"] + cerror["enemies"], 100)
    r_j = get_range_reward(info["jumps"], 0,
                           max(0, ctrl["jumps"] - cerror["jumps"]),
                           ctrl["jumps"] + cerror["jumps"], 100)
    r_c = get_range_reward(info["coins"], 0,
                           max(0, ctrl["coins"] - cerror["coins"]),
                           ctrl["coins"] + cerror["coins"], 100)
    return r_e, r_j, r_c, (r_e + r_j + r_c) / 3.0

print("=== CONTROLABILITY per level (completed levels only) ===")
print(f"  target: {CONTROL}   plateau = target ± cerror")
if skipped_idxs:
    print(f"  excluded (solver incomplete): indices {skipped_idxs}")
print()
h = (f"{'#':>3} {'score':>5} | {'enemies: actual∈[lo,hi]':>26} "
     f"{'jumps: actual∈[lo,hi]':>26} {'coins: actual∈[lo,hi]':>26}")
print(h); print("-" * len(h))
se = sj = sc = 0.0
n_used = 0
for i, info in enumerate(infos):
    if not completed_mask[i]:
        print(f"{i:>3} {'—':>5} | SKIPPED (solver incomplete — excluded from aggregates)")
        continue
    r_e, r_j, r_c, tot = ctrl_breakdown(info, CONTROL)
    se += r_e; sj += r_j; sc += r_c
    n_used += 1
    e_lo, e_hi = max(0, CONTROL["enemies"] - cerror["enemies"]), CONTROL["enemies"] + cerror["enemies"]
    j_lo, j_hi = max(0, CONTROL["jumps"]   - cerror["jumps"]),   CONTROL["jumps"]   + cerror["jumps"]
    c_lo, c_hi = max(0, CONTROL["coins"]   - cerror["coins"]),   CONTROL["coins"]   + cerror["coins"]
    print(f"{i:>3} {tot:>5.2f} | "
          f"{info['enemies']:>3} ∈ [{e_lo},{e_hi}] → {r_e:.2f}    "
          f"{info['jumps']:>3} ∈ [{j_lo},{j_hi}] → {r_j:.2f}    "
          f"{info['coins']:>3} ∈ [{c_lo},{c_hi}] → {r_c:.2f}")
print("-" * len(h))

if n_used == 0:
    print("No completed levels — controlability cannot be summarised.")
else:
    print(f"Mean component rewards (n={n_used} completed): "
          f"enemies={se/n_used:.3f}  jumps={sj/n_used:.3f}  coins={sc/n_used:.3f}")
    weakest = min((("enemies", se), ("jumps", sj), ("coins", sc)), key=lambda t: t[1])
    print(f"Weakest dimension: {weakest[0]}  →  this is what you need to fix first.")