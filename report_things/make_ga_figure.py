import numpy as np, matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path("eval/ga")
data = {}
for setting in ["low", "mid", "high"]:
    vals = []
    for s in range(10):
        p = ROOT / setting / f"s{s}" / "archive_scores.txt"
        lines = [l for l in p.read_text().splitlines() if l.strip()]
        vals.append(float(lines[0]))   # top of archive = best of run
    data[setting] = vals

fig, ax = plt.subplots(figsize=(5.5, 3.2))
positions = [1, 2, 3]
ax.boxplot([data[k] for k in ["low","mid","high"]], positions=positions,
           widths=0.5, showmeans=True)
for i, k in enumerate(["low","mid","high"]):
    ax.scatter([positions[i]]*len(data[k]), data[k], alpha=0.6, s=20, color='C0')
ax.set_xticks(positions); ax.set_xticklabels(["low","mid","high"])
ax.set_ylabel("final archive-best fitness $S$"); ax.grid(alpha=0.3)
plt.tight_layout()
Path("figs").mkdir(exist_ok=True)
plt.savefig("figs/ga_fitness.png", dpi=180)