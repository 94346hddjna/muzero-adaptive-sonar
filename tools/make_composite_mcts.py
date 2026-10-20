# -*- coding: utf-8 -*-
"""重新生成 composite_mcts.png（修正版）：
(a) SR vs MCTS模拟次数（固定planning_depth=5）
(b) 决策时延 vs 模拟次数（固定planning_depth=5）
(c) SR vs 规划深度（固定num_simulations=100）
数据：results_v3/processed/ablation_results.csv，双重过滤避免扫描交叉污染。
"""
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = ROOT / "results_v3" / "figures" / "composite" / "composite_mcts.png"


def load():
    rows = list(csv.DictReader(
        open(ROOT / "results_v3" / "processed" / "ablation_results.csv",
             encoding="utf-8")))
    return [r for r in rows if r.get("algorithm") == "muzero"
            and r.get("condition") == "medium"]


def agg(rows, pred, metric):
    vals = []
    for r in rows:
        eo = str(r.get("eval_options", "")).replace(".0", "")
        if pred(eo) and r.get(metric):
            try:
                vals.append(float(r[metric]))
            except (TypeError, ValueError):
                pass
    return (float(np.mean(vals)), float(np.std(vals))) if vals else (None, None)


def main():
    rows = load()
    ms = [25, 50, 100, 200]
    ds = [1, 3, 5, 10]
    sr_m = [agg(rows, lambda eo, m=m: f'"num_simulations": {m}' in eo
                and '"planning_depth": 5' in eo, "SR") for m in ms]
    lat_m = [agg(rows, lambda eo, m=m: f'"num_simulations": {m}' in eo
                 and '"planning_depth": 5' in eo, "MeanDecisionLatencyMs")
             for m in ms]
    sr_d = [agg(rows, lambda eo, d=d: f'"planning_depth": {d}' in eo
                and '"num_simulations": 100' in eo, "SR") for d in ds]

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.0), dpi=300)
    C1, C4 = "#d94a4a", "#4a90d9"

    ax = axes[0]
    mu = [v[0] for v in sr_m]
    sd = [v[1] for v in sr_m]
    ax.errorbar(range(4), mu, yerr=sd, marker="o", ms=7, lw=2.2, capsize=5,
                color=C1, markerfacecolor="white", markeredgewidth=2)
    ax.set_xticks(range(4)); ax.set_xticklabels([str(m) for m in ms])
    ax.set_xlabel("MCTS simulations M", fontsize=10.5)
    ax.set_ylabel("Success Rate (Medium)", fontsize=10.5)
    ax.set_title("(a) SR vs simulation count (depth=5)", fontsize=11,
                 fontweight="bold", loc="left")
    ax.axhline(mu[2], color="gray", ls=":", lw=1)
    ax.annotate("default M=100", (2, mu[2]), textcoords="offset points",
                xytext=(-8, 10), fontsize=8.5, color="gray", ha="right")
    ax.grid(alpha=0.3); ax.set_ylim(0.15, 0.78)

    ax = axes[1]
    lmu = [v[0] for v in lat_m]
    lsd = [v[1] for v in lat_m]
    ax.errorbar(range(4), lmu, yerr=lsd, marker="s", ms=7, lw=2.2, capsize=5,
                color=C4, markerfacecolor="white", markeredgewidth=2)
    ax.set_xticks(range(4)); ax.set_xticklabels([str(m) for m in ms])
    ax.set_xlabel("MCTS simulations M", fontsize=10.5)
    ax.set_ylabel("Decision latency (ms)", fontsize=10.5)
    ax.set_title("(b) Latency vs simulation count (depth=5)", fontsize=11,
                 fontweight="bold", loc="left")
    ax.grid(alpha=0.3)

    ax = axes[2]
    mu = [v[0] for v in sr_d]
    sd = [v[1] for v in sr_d]
    ax.errorbar(range(4), mu, yerr=sd, marker="^", ms=7, lw=2.2, capsize=5,
                color="#4aa87c", markerfacecolor="white", markeredgewidth=2)
    ax.set_xticks(range(4)); ax.set_xticklabels([str(d) for d in ds])
    ax.set_xlabel("Planning depth D", fontsize=10.5)
    ax.set_ylabel("Success Rate (Medium)", fontsize=10.5)
    ax.set_title("(c) SR vs planning depth (M=100)", fontsize=11,
                 fontweight="bold", loc="left")
    ax.grid(alpha=0.3); ax.set_ylim(0.45, 0.78)

    fig.suptitle("MCTS ablations: performance–compute tradeoff "
                 "(5 seeds, 1000 paired episodes, Medium)", fontsize=12,
                 fontweight="bold", y=1.00)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight")
    print(f"✅ {OUT}")


if __name__ == "__main__":
    main()
