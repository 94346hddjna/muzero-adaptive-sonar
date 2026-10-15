# -*- coding: utf-8 -*-
"""重做两张不满意的图：认知闭环(图1-1)、先验偏差(图3-10)——饱满、丰富、舒服。"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Wedge, Rectangle
import numpy as np
from pathlib import Path

plt.rcParams["font.family"] = ["Times New Roman", "Microsoft YaHei"]
plt.rcParams["font.size"] = 10

OUT = Path("results_v3/figures/schematic")
C1 = "#2E5E8C"; C2 = "#4A90D9"; C3 = "#7FB3D5"; C4 = "#E8A33D"; C5 = "#D9534F"; C6 = "#5CB85C"
GRAY = "#666666"


def save(fig, name):
    fig.savefig(OUT / name, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"✅ {name}")


# ============ 图1-1: 认知声呐闭环（饱满版：内圈闭环+外圈细节） ============
def fig_cognitive_loop_full():
    fig = plt.figure(figsize=(13, 8.5), dpi=300)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.5, 1], wspace=0.25)

    # 左：闭环流程（四阶段+中心公式+箭头）
    ax = fig.add_subplot(gs[0])
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")

    # 中心：信念状态公式
    ax.text(5, 5.0, "Belief State\n$b_t(i)=P(x=i|H_t)$\n$\\sum_i b_t(i)=1$",
            ha="center", va="center", fontsize=13, fontweight="bold", color=C1,
            bbox=dict(boxstyle="round,pad=0.5", fc="#EAF3FB", ec=C1, lw=1.5))

    # 四阶段（菱形布置）
    stages = [
        (5.0, 8.9, "1. Transmit", "waveform params\n(f0, B, Tp, P)", C2),
        (8.6, 5.0, "2. Observe", "echo y(t)=a·s(t-τ)+c(t)+n(t)\nmatched filter → envelope", C4),
        (5.0, 1.1, "3. Cognize", "Bayesian belief update\nb(t+1) ∝ P(o|x=i,a)·b(t)", C5),
        (1.4, 5.0, "4. Adapt", "MCTS planning\nπ(a|s) from visit counts", C6),
    ]
    for x, y, title, sub, col in stages:
        box = FancyBboxPatch((x - 1.5, y - 0.72), 3.0, 1.45,
                             boxstyle="round,pad=0.05,rounding_size=0.1",
                             fc=col, ec=C1, lw=1.8, alpha=0.95, zorder=3)
        ax.add_patch(box)
        ax.text(x, y + 0.3, title, ha="center", va="center", fontsize=11,
                fontweight="bold", color="white", zorder=4)
        ax.text(x, y - 0.24, sub, ha="center", va="center", fontsize=7.8,
                color="white", zorder=4, linespacing=1.35)

    # 外圈循环箭头
    arrows = [((6.5, 8.2), (8.2, 6.0)), ((8.2, 4.0), (6.5, 1.9)),
              ((3.5, 1.9), (1.8, 4.0)), ((1.8, 6.0), (3.5, 8.2))]
    for (x1, y1), (x2, y2) in arrows:
        arr = FancyArrowPatch((x1, y1), (x2, y2),
                              connectionstyle="arc3,rad=-0.3",
                              arrowstyle="-|>", mutation_scale=24,
                              lw=2.6, color=C1, zorder=2)
        ax.add_patch(arr)

    ax.set_title("(a) Adaptive cognitive sonar closed-loop", fontsize=12,
                 fontweight="bold", loc="left", pad=6)

    # 右：声呐探测细节（扇区网格+信念热图+检测模型）
    gs_r = gs[1].subgridspec(2, 1, hspace=0.45)
    # 右上：扇区网格俯视（带信念热图+目标+波束）
    ax1 = fig.add_subplot(gs_r[0])
    ax1.set_xlim(-1.35, 1.35)
    ax1.set_ylim(-1.35, 1.35)
    ax1.set_aspect("equal")
    ax1.axis("off")
    N = 16
    rng = np.random.default_rng(5)
    belief = np.exp(-((np.arange(N) - 7) ** 2) / (2 * 2.2 ** 2))
    belief /= belief.sum()
    for i in range(N):
        a1 = i * 360 / N
        a2 = (i + 1) * 360 / N
        col = plt.cm.hot(belief[i] / belief.max())
        ax1.add_patch(Wedge((0, 0), 1.0, a1, a2, fc=col, ec="white", lw=0.6))
    tgt = 7
    ang = np.deg2rad((tgt + 0.5) * 360 / N)
    ax1.plot(0.68 * np.cos(ang), 0.68 * np.sin(ang), marker="*", ms=18,
             color="cyan", mec="k", mew=0.9, zorder=5)
    # 当前波束
    cur = 3
    a_beam = np.deg2rad((cur + 0.5) * 360 / N)
    ax1.add_patch(Wedge((0, 0), 1.02, (cur) * 360 / N, (cur + 1) * 360 / N,
                        fc="none", ec="blue", lw=2.4, ls="--"))
    ax1.add_patch(Circle((0, 0), 0.07, fc=C1, ec="k", zorder=6))
    ax1.annotate("target", (0.68 * np.cos(ang), 0.68 * np.sin(ang)),
                 xytext=(0.85, 0.95), fontsize=8.5, ha="center", color="k",
                 arrowprops=dict(arrowstyle="->", color="k", lw=1.1))
    ax1.annotate("current beam", (0.55 * np.cos(a_beam), 0.55 * np.sin(a_beam)),
                 xytext=(-1.2, 1.1), fontsize=8.5, ha="center", color="blue",
                 arrowprops=dict(arrowstyle="->", color="blue", lw=1.1))
    ax1.set_title("(b) Sector grid: belief heatmap + beam + target",
                  fontsize=10.5, fontweight="bold", loc="left", pad=4)

    # 右下：SNR→PD 检测模型
    ax2 = fig.add_subplot(gs_r[1])
    snr = np.linspace(-10, 20, 200)
    for pfa, c, lab in [(0.01, C2, "PFA=0.01"), (0.05, C4, "PFA=0.05"), (0.10, C5, "PFA=0.10")]:
        pd = np.exp(np.log(pfa) / (1 + 10 ** (snr / 10)))
        ax2.plot(snr, pd, color=c, lw=2.4, label=lab)
    ax2.set_xlabel("SNR (dB)", fontsize=10)
    ax2.set_ylabel("P_D", fontsize=10)
    ax2.set_title("(c) Swerling-I detection model", fontsize=10.5,
                  fontweight="bold", loc="left")
    ax2.legend(fontsize=8.5, loc="lower right", framealpha=0.9)
    ax2.grid(alpha=0.3)

    save(fig, "fig_cognitive_loop_full.png")


# ============ 图3-10: 先验偏差（饱满版：4类先验分布 + SR对比 + KL纠偏曲线） ============
def fig_prior_bias_full():
    import json
    fig = plt.figure(figsize=(14, 10), dpi=300)
    gs = fig.add_gridspec(2, 2, hspace=0.4, wspace=0.3)
    N = 16
    tgt = 9

    priors = [("uniform", C2), ("correct", C6), ("biased", C4), ("adversarial", C5)]
    for idx, (pt, col) in enumerate(priors):
        ax = fig.add_subplot(gs[idx // 2, idx % 2])
        if pt == "uniform":
            b0 = np.full(N, 1 / N)
        elif pt == "correct":
            d = np.minimum(np.abs(np.arange(N) - tgt), N - np.abs(np.arange(N) - tgt))
            b0 = np.exp(-(d ** 2) / (2 * 1.0 ** 2))
            b0 /= b0.sum()
        elif pt == "biased":
            wrong = 3
            d = np.minimum(np.abs(np.arange(N) - wrong), N - np.abs(np.arange(N) - wrong))
            b0 = np.exp(-(d ** 2) / (2 * 1.5 ** 2))
            b0 /= b0.sum()
        else:
            far = (tgt + N // 2) % N
            d = np.minimum(np.abs(np.arange(N) - far), N - np.abs(np.arange(N) - far))
            b0 = np.exp(-(d ** 2) / (2 * 1.0 ** 2))
            b0 /= b0.sum()
        ax.bar(range(N), b0, color=col, alpha=0.88, edgecolor="white", lw=0.6)
        ax.axvline(tgt, color="k", ls="--", lw=1.5, label="true target")
        ax.set_title(f"({chr(97 + idx)}) prior: {pt}", fontsize=11, fontweight="bold")
        ax.set_xlabel("sector", fontsize=10)
        ax.set_ylabel("initial belief $b_0$", fontsize=10)
        ax.legend(fontsize=8.5, loc="upper left")
        ax.grid(alpha=0.3, axis="y")

    fig.suptitle("Prior-bias experiment: four initial belief types over 16 sectors",
                 fontsize=14, fontweight="bold", y=0.99)
    save(fig, "fig_prior_bias_full.png")


# ============ 图3-10补充：先验偏差SR对比 + KL纠偏（放在同一张大图） ============
def fig_prior_bias_combined():
    import json
    import csv
    import sys
    from pathlib import Path as _P
    sys.path.insert(0, str(_P(__file__).resolve().parent.parent))
    from utils.io_utils import read_json
    fig = plt.figure(figsize=(14, 10), dpi=300)
    gs = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.32)
    N = 16
    tgt = 9

    # 上排：4类先验分布
    priors = [("uniform", C2), ("correct", C6), ("biased", C4), ("adversarial", C5)]
    for idx, (pt, col) in enumerate(priors):
        ax = fig.add_subplot(gs[0, idx % 2] if idx < 2 else gs[1, idx % 2])
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
        ax.axvline(tgt, color="k", ls="--", lw=1.5)
        ax.set_title(f"({chr(97 + idx)}) prior: {pt}", fontsize=10.5, fontweight="bold")
        ax.set_xlabel("sector", fontsize=9)
        ax.set_ylabel("$b_0$", fontsize=9)
        ax.grid(alpha=0.3, axis="y")

    # 下排左：KL纠偏曲线（真实数据）
    ax_kl = fig.add_subplot(gs[1, 0])
    kdir = Path("results_v3/processed/prior_bias_kl")
    if kdir.exists():
        for f in sorted(kdir.glob("muzero_main_seed101_*.json")):
            d = read_json(f)
            curve = [v for v in d["kl_mean_curve"] if v is not None]
            if curve:
                ax_kl.plot(range(len(curve)), curve, marker="o", markersize=3,
                           lw=2, label=str(d["prior_type"]))
    ax_kl.set_xlabel("ping step t", fontsize=10)
    ax_kl.set_ylabel("$D_{KL}(P_{true}\\|b_t)$", fontsize=10)
    ax_kl.set_title("(e) KL divergence correction over pings", fontsize=10.5,
                    fontweight="bold", loc="left")
    ax_kl.legend(fontsize=8.5)
    ax_kl.grid(alpha=0.3)

    # 下排右：SR对比（真实数据）
    ax_sr = fig.add_subplot(gs[1, 1])
    try:
        rows = list(csv.DictReader(open("results_v3/processed/prior_bias_results.csv",
                                        encoding="utf-8")))
        pts = ["uniform", "correct", "biased", "adversarial"]
        mz_sr = []
        bg_sr = []
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
        ax_sr.set_title("(f) SR under different priors", fontsize=10.5,
                        fontweight="bold", loc="left")
        ax_sr.legend(fontsize=8.5)
        ax_sr.grid(alpha=0.3, axis="y")
        ax_sr.set_ylim(0, 1.05)
    except Exception as e:
        ax_sr.text(0.5, 0.5, f"(data error: {e})", ha="center", transform=ax_sr.transAxes)

    fig.suptitle("Prior-bias robustness: belief distributions + KL correction + success rate",
                 fontsize=13, fontweight="bold", y=0.99)
    save(fig, "fig_prior_bias_combined.png")


fig_cognitive_loop_full()
fig_prior_bias_combined()
print("完成")
