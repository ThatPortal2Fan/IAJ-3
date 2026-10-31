import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

AXES = ["quality", "diversity", "controlability"]
LABELS = ["Quality", "Diversity", "Controllability"]
W_MAIN = (0.5, 0.3, 0.2)
W_SENS = {
    "ours (0.5,0.3,0.2)": (0.5, 0.3, 0.2),
    "equal": (1 / 3, 1 / 3, 1 / 3),
    "designer (0.4,0.2,0.4)": (0.4, 0.2, 0.4),
    "variety (0.4,0.5,0.1)": (0.4, 0.5, 0.1),
}
SETTINGS = ["low", "mid", "high"]
ORDER = ["nonrl", "rl", "ga"]                      # polygon / row order (others appended)
NICE = {"nonrl": "Non-RL", "rl": "RL", "ga": "GA"}


def composite(v, w):
    return sum(wi * xi for wi, xi in zip(w, v))


def composite_geo(v, w):
    return float(np.prod([max(xi, 0.0) ** wi for wi, xi in zip(w, v)]))


ap = argparse.ArgumentParser()
ap.add_argument("--results", default="results.json")
ap.add_argument("--outdir", default="figs")
args = ap.parse_args()
recs = json.loads(Path(args.results).read_text())
Path(args.outdir).mkdir(parents=True, exist_ok=True)

# ---- group: (generator, setting) -> list of batch records -----------------
groups = defaultdict(list)
for r in recs:
    groups[(r["generator"], r["setting"])].append(r)
gens = [g for g in ORDER if any(k[0] == g for k in groups)] + \
       sorted({k[0] for k in groups} - set(ORDER))


def stat(gen, setting, key):
    xs = np.array([r[key] for r in groups[(gen, setting)]], dtype=float)
    return float(np.nanmean(xs)), float(np.nanstd(xs))     # std = 0 if a single batch


# per (gen, setting): mean vector (Q, D, K)
mean = {(g, s): [stat(g, s, a)[0] for a in AXES]
        for g in gens for s in SETTINGS if (g, s) in groups}

# ---- print numbers for the LaTeX tables ------------------------------------
print("\n=== Table main: Q, D, K, Kcomp, completion %, C, C_geo (mean +- std over batches) ===")
for g in gens:
    avg_rows = []
    for s in SETTINGS:
        if (g, s) not in groups:
            continue
        q, qs = stat(g, s, "quality")
        d, ds = stat(g, s, "diversity")
        k, ks = stat(g, s, "controlability_mean")
        kc, _ = stat(g, s, "controlability_completed")
        cr, _ = stat(g, s, "completion_rate")
        c = composite((q, d, k), W_MAIN)
        cg = composite_geo((q, d, k), W_MAIN)
        n_b = len(groups[(g, s)])
        avg_rows.append((q, d, k, kc, cr, c, cg))
        print(f"{NICE.get(g, g):7s} {s:5s} "
              f"Q={q:.3f}±{qs:.3f}  D={d:.3f}±{ds:.3f}  K={k:.3f}±{ks:.3f}  "
              f"Kcomp={kc:.3f}  compl={100*cr:.0f}%  C={c:.3f}  Cgeo={cg:.3f}  (batches={n_b})")
    if avg_rows:
        a = np.nanmean(np.array(avg_rows), axis=0)
        print(f"{NICE.get(g, g):7s} AVG   Q={a[0]:.3f}  D={a[1]:.3f}  K={a[2]:.3f}  "
              f"Kcomp={a[3]:.3f}  compl={100*a[4]:.0f}%  C={a[5]:.3f}  Cgeo={a[6]:.3f}")

print("\n=== Sensitivity table: composite averaged over settings ===")
avg_vec = {g: np.mean([mean[(g, s)] for s in SETTINGS if (g, s) in mean], axis=0) for g in gens}
for name, w in W_SENS.items():
    vals = {g: composite(avg_vec[g], w) for g in gens}
    win = max(vals, key=vals.get)
    print(f"{name:26s} " + "  ".join(f"{NICE.get(g, g)}={vals[g]:.3f}" for g in gens)
          + f"   winner: {NICE.get(win, win)}")


# ---- radar -----------------------------------------------------------------
def radar(ax, series, title=None):
    ang = np.linspace(0, 2 * np.pi, len(AXES), endpoint=False).tolist()
    ang += ang[:1]
    for name, vec in series.items():
        v = list(vec) + [vec[0]]
        ax.plot(ang, v, marker="o", linewidth=2, label=name)
        ax.fill(ang, v, alpha=0.15)
    ax.set_xticks(ang[:-1])
    ax.set_xticklabels(LABELS, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.5", "0.75", "1"], fontsize=7)
    if title:
        ax.set_title(title, fontsize=11, pad=14)


fig = plt.figure(figsize=(5.2, 5.2))
ax = fig.add_subplot(111, polar=True)
radar(ax, {NICE.get(g, g): avg_vec[g] for g in gens})
ax.legend(loc="upper right", bbox_to_anchor=(1.28, 1.12), fontsize=9)
fig.tight_layout()
fig.savefig(Path(args.outdir) / "radar.png", dpi=200)

fig = plt.figure(figsize=(12, 4.4))
for i, s in enumerate(SETTINGS):
    ax = fig.add_subplot(1, 3, i + 1, polar=True)
    radar(ax, {NICE.get(g, g): mean[(g, s)] for g in gens if (g, s) in mean}, f"{s} setting")
    if i == 2:
        ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=9)
fig.tight_layout()
fig.savefig(Path(args.outdir) / "radar_by_setting.png", dpi=200)
print(f"\nwrote {args.outdir}/radar.png and {args.outdir}/radar_by_setting.png")
