# -*- coding: utf-8 -*-
"""先验偏差组合图（3行布局）：先验分布(4) + KL纠偏曲线 + SR对比。"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import csv
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.io_utils import read_json

plt.rcParams["font.family"] = ["Times New Roman", "Microsoft YaHei"]
plt.rcParams["font.size"] = 10

OUT = Path("results_v3/figures/schematic")
C1 = "#2E5E8C"; C2 = "#4A90D9"; C4 = "#E8A33D"; C5 = "#D9534F"; C6 = "#5CB85C"

fig = plt.figure(figsize=(13, 10), dpi=300)
gs = fig.add_gridspec(3, 2, hspace=0.5, wspace=0.3)
N = 16
tgt = 9

priors = [("uniform", C2), ("correct", C6), ("biased", C4), ("adversarial", C5)]
positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
for idx, (pt, col) in enumerate(priors):
    r, c = positions[idx]
    ax = fig.add_subplot(gs[r, c])
    if pt == "uniform":
        b0 = np.full(N, 1 / N)
    elif pt == "correct":
        d = np.minimum(np.abs(np.arange(N) - tgt), N - np.abs(np.arange(N) - tgt))
        b0 = np.exp(-(d ** 2) / (2 * 1.0 ** 2)); b0 /= b0.sum()
    elif pt == "biased":
        wrong = 3
        d = np.minimum(np.abs(np.arange(N) - wrong), N - np.abs(np.arange(N) - wrong))
        b0 = np.exp(-(d ** 2) / (2 * 1.5 ** 2)); b0 /= b0.sum()
    else:
        far = (tgt + N // 2) % N
        d = np.minimum(np.abs(np.arange(N) - far), N - np.abs(np.arange(N) - far))
        b0 = np.exp(-(d ** 2) / (2 * 1.0 ** 2)); b0 /= b0.sum()
    ax.bar(range(N), b0, color=col, alpha=0.88, edgecolor="white", lw=0.6)
    ax.axvline(tgt, color="k", ls="--", lw=1.5, label="true target")
    ax.set_title(f"({chr(97 + idx)}) prior: {pt}", fontsize=10.5, fontweight="bold")
    ax.set_xlabel("sector", fontsize=9)
    ax.set_ylabel("$b_0$", fontsize=9)
    ax.legend(fontsize=8.5, loc="upper left")
    ax.grid(alpha=0.3, axis="y")

ax_kl = fig.add_subplot(gs[2, 0])
kdir = Path("results_v3/processed/prior_bias_kl")
if kdir.exists():
    for f in sorted(kdir.glob("muzero_main_seed101_*.json")):
        d = read_json(f)
        curve = [v for v in d["kl_mean_curve"] if v is not None]
        if curve:
            ax_kl.plot(range(len(curve)), curve, marker="o", markersize=3,
                       lw=2, label=str(d["prior_type"]))
ax_kl.set_xlabel("ping step t", fontsize=10)
ax_kl.set_ylabel("$D_{KL}(P_{true} \\| b_t)$", fontsize=10)
ax_kl.set_title("(e) KL divergence correction over pings", fontsize=10.5,
                fontweight="bold", loc="left")
ax_kl.legend(fontsize=8.5)
ax_kl.grid(alpha=0.3)

ax_sr = fig.add_subplot(gs[2, 1])
rows = list(csv.DictReader(open("results_v3/processed/prior_bias_results.csv", encoding="utf-8")))
pts = ["uniform", "correct", "biased", "adversarial"]
mz_sr, bg_sr = [], []
for pt in pts:
    v = [float(r["SR"]) for r in rows if r.get("prior_type") == pt
         and r["algorithm"] == "muzero" and r["SR"]]
    mz_sr.append(float(np.mean(v)) if v else 0)
    v2 = [float(r["SR"]) for r in rows if r.get("prior_type") == pt
          and r["algorithm"] == "belief_greedy" and r["SR"]]
    bg_sr.append(float(np.mean(v2)) if v2 else 0)
x = np.arange(len(pts))
w = 0.38
ax_sr.bar(x - w / 2, bg_sr, w, color=C5, alpha=0.85, label="Belief-Greedy")
ax_sr.bar(x + w / 2, mz_sr, w, color=C1, alpha=0.9, label="MuZero")
ax_sr.set_xticks(x)
ax_sr.set_xticklabels(pts, fontsize=9)
ax_sr.set_ylabel("Success Rate", fontsize=10)
ax_sr.set_title("(f) SR under different priors", fontsize=10.5, fontweight="bold", loc="left")
ax_sr.legend(fontsize=8.5)
ax_sr.grid(alpha=0.3, axis="y")
ax_sr.set_ylim(0, 1.05)

fig.suptitle("Prior-bias robustness: belief distributions + KL correction + success rate",
             fontsize=14, fontweight="bold", y=0.99)
fig.savefig(OUT / "fig_prior_bias_combined.png", dpi=300, bbox_inches="tight", facecolor="white")
print("OK fig_prior_bias_combined.png")
