# -*- coding: utf-8 -*-
"""生成内容丰富的技术图（替代单薄的AI示意图）。"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Wedge
from matplotlib.patches import ConnectionPatch
import numpy as np
from pathlib import Path

plt.rcParams["font.family"] = ["Times New Roman", "Microsoft YaHei"]
plt.rcParams["font.size"] = 10

OUT = Path("results_v3/figures/schematic")
OUT.mkdir(parents=True, exist_ok=True)

C1 = "#2E5E8C"  # 深蓝
C2 = "#4A90D9"  # 蓝
C3 = "#7FB3D5"  # 浅蓝
C4 = "#E8A33D"  # 橙
C5 = "#D9534F"  # 红
C6 = "#5CB85C"  # 绿
GRAY = "#666666"


def save(fig, name):
    fig.savefig(OUT / name, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"✅ {name}")


# ============ 图1: 认知声呐闭环（丰富版） ============
def fig_cognitive_loop():
    fig, ax = plt.subplots(figsize=(9.2, 7.6), dpi=300)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 9.6)
    ax.axis("off")

    # 中心标题
    ax.text(5, 5.0, "Adaptive\nCognitive Sonar Loop", ha="center", va="center",
            fontsize=14, fontweight="bold", color=C1)

    # 四个阶段框（上下左右布局，箭头走圆环外侧）
    stages = [
        (5.0, 8.5, "1. Transmit", "waveform selection\n(f0, B, Tp, P)", C2),
        (8.5, 5.0, "2. Observe", "echo y(t)=a s(t-tau)+c(t)+n(t)\nmatched filter + envelope", C4),
        (5.0, 1.5, "3. Cognize", "Bayesian belief update\nb(t+1)(i) ~ P(o|x=i,a) b(t)(i)", C5),
        (1.5, 5.0, "4. Adapt", "MCTS planning\npolicy from visit counts", C6),
    ]
    for x, y, title, sub, col in stages:
        box = FancyBboxPatch((x - 1.35, y - 0.62), 2.7, 1.24,
                             boxstyle="round,pad=0.05,rounding_size=0.1",
                             fc=col, ec=C1, lw=1.8, alpha=0.95, zorder=3)
        ax.add_patch(box)
        ax.text(x, y + 0.3, title, ha="center", va="center", fontsize=11,
                fontweight="bold", color="white", zorder=4)
        ax.text(x, y - 0.2, sub, ha="center", va="center", fontsize=7.8,
                color="white", zorder=4, linespacing=1.35)

    # 循环箭头（顺时针：上→右→下→左→上）
    arrows = [((6.35, 7.9), (8.0, 5.7)), ((8.0, 4.3), (6.35, 2.1)),
              ((3.65, 2.1), (2.0, 4.3)), ((2.0, 5.7), (3.65, 7.9))]
    for (x1, y1), (x2, y2) in arrows:
        arr = FancyArrowPatch((x1, y1), (x2, y2),
                              connectionstyle="arc3,rad=-0.32",
                              arrowstyle="-|>", mutation_scale=24,
                              lw=2.6, color=C1, zorder=2)
        ax.add_patch(arr)

    ax.text(5, 9.35, "Cognitive sonar: sense, observe, adapt, re-transmit",
            ha="center", fontsize=10.5, color=GRAY, style="italic")
    ax.text(5, 0.25, "each emission is adapted based on the previous echo",
            ha="center", fontsize=10, color=GRAY)

    save(fig, "fig_cognitive_loop_rich.png")


# ============ 图2: 声呐扇区搜索场景（丰富版） ============
def fig_sonar_scene():
    fig = plt.figure(figsize=(11, 8), dpi=300)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.5, 1], height_ratios=[1, 1],
                          hspace=0.35, wspace=0.3)
    # 左上：扇区网格俯视
    ax1 = fig.add_subplot(gs[0, 0], projection=None)
    N = 16
    ax1.set_xlim(-1.3, 1.3)
    ax1.set_ylim(-1.3, 1.3)
    ax1.set_aspect("equal")
    ax1.axis("off")
    # 信念热图作为扇区填充
    rng = np.random.default_rng(5)
    belief = np.exp(-((np.arange(N) - 7) ** 2) / (2 * 2.2 ** 2))
    belief /= belief.sum()
    for i in range(N):
        a1 = i * 360 / N
        a2 = (i + 1) * 360 / N
        col = plt.cm.hot(belief[i] / belief.max())
        w = Wedge((0, 0), 1.0, a1, a2, fc=col, ec="white", lw=0.5)
        ax1.add_patch(w)
    # 目标标记
    tgt = 7
    ang = np.deg2rad((tgt + 0.5) * 360 / N)
    ax1.plot(0.68 * np.cos(ang), 0.68 * np.sin(ang), marker="*", ms=16,
             color="cyan", mec="k", mew=0.8, zorder=5)
    # 声呐源
    ax1.add_patch(Circle((0, 0), 0.07, fc=C1, ec="k", zorder=6))
    ax1.annotate("sonar\nsource", (0, 0), xytext=(-0.5, -0.28), fontsize=8,
                 ha="center", color=C1)
    ax1.annotate("target\nsector", (0.68 * np.cos(ang), 0.68 * np.sin(ang)),
                 xytext=(0.9, 0.9), fontsize=8, ha="center", color="k",
                 arrowprops=dict(arrowstyle="->", color="k", lw=1))
    ax1.set_title("(a) Sector grid + belief heatmap", fontsize=11, fontweight="bold")

    # 右上：SNR→PD曲线
    ax2 = fig.add_subplot(gs[0, 1])
    snr = np.linspace(-10, 20, 200)
    for pfa, c, lab in [(0.01, C2, "PFA=0.01"), (0.05, C4, "PFA=0.05"), (0.10, C5, "PFA=0.10")]:
        pd = np.exp(np.log(pfa) / (1 + 10 ** (snr / 10)))
        ax2.plot(snr, pd, color=c, lw=2.2, label=lab)
    ax2.set_xlabel("SNR (dB)")
    ax2.set_ylabel("P_D")
    ax2.set_title("(b) Swerling-I detection model", fontsize=11, fontweight="bold")
    ax2.legend(fontsize=8, loc="lower right")
    ax2.grid(alpha=0.3)

    # 左下：波束方向图
    ax3 = fig.add_subplot(gs[1, 1])
    dth = np.linspace(-30, 30, 300)
    for sig, c, lab in [(8, C2, "Easy σ_b"), (4, C4, "Medium σ_b"), (2, C5, "Hard σ_b")]:
        g = np.exp(-(dth ** 2) / (2 * sig ** 2))
        ax3.plot(dth, g, color=c, lw=2.2, label=lab)
    ax3.set_xlabel("Δθ (deg)")
    ax3.set_ylabel("G(Δθ)")
    ax3.set_title("(c) Beam directivity (Gaussian)", fontsize=11, fontweight="bold")
    ax3.legend(fontsize=8)
    ax3.grid(alpha=0.3)

    # 右下：信念更新过程
    ax4 = fig.add_subplot(gs[1, 0])
    N_b = 16
    b = np.full(N_b, 1 / N_b)
    steps = [b.copy()]
    rng2 = np.random.default_rng(2)
    tgt2 = 9
    for t in range(8):
        a = rng2.integers(N_b)
        det = 1 if (a == tgt2 and rng2.random() < 0.85) else (1 if rng2.random() < 0.05 else 0)
        like = np.full(N_b, 1 - 0.05) if det == 0 else np.full(N_b, 0.05)
        like[a] = (1 - 0.85) if det == 0 else 0.85
        b = b * like
        b /= b.sum()
        steps.append(b.copy())
    for t, bb in enumerate(steps):
        ax4.plot(range(N_b), bb, color=plt.cm.viridis(t / len(steps)), lw=1.6,
                 label=f"t={t}" if t in (0, 4, 8) else None)
    ax4.set_xlabel("sector")
    ax4.set_ylabel("belief b_t(i)")
    ax4.set_title("(d) Belief evolution over pings", fontsize=11, fontweight="bold")
    ax4.legend(fontsize=8)
    ax4.grid(alpha=0.3)

    save(fig, "fig_sonar_scene_rich.png")


# ============ 图3: 先验偏差实验（丰富版） ============
def fig_prior_bias():
    import csv
    fig = plt.figure(figsize=(11, 8.5), dpi=300)
    gs = fig.add_gridspec(2, 2, hspace=0.4, wspace=0.32)
    N = 16
    tgt = 9

    priors = [("uniform", C2), ("correct", C6), ("biased", C4), ("adversarial", C5)]
    for idx, (pt, col) in enumerate(priors):
        ax = fig.add_subplot(gs[idx // 2, idx % 2])
        rng = np.random.default_rng(idx)
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
        ax.bar(range(N), b0, color=col, alpha=0.85, edgecolor="white")
        ax.axvline(tgt, color="k", ls="--", lw=1.4, label="true target")
        ax.set_title(f"({chr(97 + idx)}) prior: {pt}", fontsize=11, fontweight="bold")
        ax.set_xlabel("sector")
        ax.set_ylabel("initial belief b₀")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, axis="y")
    fig.suptitle("Prior-bias experiment: four initial belief types over 16 sectors",
                 fontsize=13, fontweight="bold")
    save(fig, "fig_prior_bias_rich.png")


fig_cognitive_loop()
fig_sonar_scene()
fig_prior_bias()
print("全部丰富图完成")
