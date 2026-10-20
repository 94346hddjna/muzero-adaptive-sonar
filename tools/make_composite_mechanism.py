# -*- coding: utf-8 -*-
"""重新生成 composite_mechanism.png（修正版，全真实机理数据）：
(a) 信念演化热图（单回合 belief_after 逐步）
(b) 搜索轨迹（动作 vs t，标注检测点与目标扇区）
(c) MCTS根访问分布（某步 visits）
(d) 原始网络策略 vs MCTS改进策略（同一步）
数据：results_v3/raw/mechanism/muzero_main_seed101_main_medium.json
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = ROOT / "results_v3" / "figures" / "composite" / "composite_mechanism.png"


def _arr(v):
    """字段可能是list或JSON字符串，统一成ndarray。"""
    if isinstance(v, str):
        v = json.loads(v)
    return np.array(v, dtype=float)


def parse(ep):
    """把字符串字段解析成数组。"""
    steps = []
    for r in ep:
        steps.append({
            "t": int(r["t"]),
            "action": int(r["action"]),
            "visits": _arr(r["visits"]),
            "raw_policy": _arr(r["raw_policy"]),
            "policy_target": _arr(r["policy_target"]),
            "detection": int(r["detection"]),
            "belief": _arr(r["belief_after"]),
            "reward": float(r["reward"]),
        })
    return steps


def main():
    d = json.load(open(ROOT / "results_v3" / "raw" / "mechanism" /
                       "muzero_main_seed101_main_medium.json", encoding="utf-8"))
    # 选一个信息量大的回合：步数>=6且有成功检测
    best = None
    for k, ep in d.items():
        steps = parse(ep)
        if len(steps) >= 6 and any(s["detection"] == 1 for s in steps):
            best = steps
            break
    if best is None:
        best = parse(d[list(d.keys())[0]])
    steps = best
    T = len(steps)
    N = len(steps[0]["belief"])
    # 目标扇区：由真实检测推断（detection==1 的 action）
    target = next((s["action"] for s in steps if s["detection"] == 1), None)

    fig = plt.figure(figsize=(12.5, 7.6), dpi=300)
    gs = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.30)

    # (a) 信念演化热图
    ax = fig.add_subplot(gs[0, 0])
    B = np.stack([s["belief"] for s in steps], axis=0)  # [T, N]
    im = ax.imshow(B.T, aspect="auto", cmap="viridis", origin="lower",
                   extent=[0.5, T + 0.5, 0, N])
    if target is not None:
        ax.axhline(target + 0.5, color="red", ls="--", lw=1.5)
        ax.text(T + 0.7, target + 0.5, "target", color="red", fontsize=8,
                va="center")
    ax.set_xlabel("ping step t", fontsize=10)
    ax.set_ylabel("sector", fontsize=10)
    ax.set_title("(a) Belief evolution over pings", fontsize=11,
                 fontweight="bold", loc="left")
    fig.colorbar(im, ax=ax, shrink=0.85, label="$b_t(i)$")

    # (b) 搜索轨迹
    ax = fig.add_subplot(gs[0, 1])
    ts = [s["t"] + 1 for s in steps]
    acts = [s["action"] for s in steps]
    dets = [s["detection"] for s in steps]
    ax.plot(ts, acts, "-o", color="#4a90d9", lw=2, ms=6, label="action (beam sector)")
    for t, a, dd in zip(ts, acts, dets):
        if dd == 1:
            ax.plot(t, a, "v", color="red", ms=13, zorder=5)
    if target is not None:
        ax.axhline(target, color="red", ls="--", lw=1.4, alpha=0.7,
                   label=f"true target sector = {target}")
    ax.set_xlabel("ping step t", fontsize=10)
    ax.set_ylabel("sector probed", fontsize=10)
    ax.set_title("(b) Search trajectory (▼ = detection)", fontsize=11,
                 fontweight="bold", loc="left")
    ax.legend(fontsize=8.5)
    ax.grid(alpha=0.3)
    ax.set_ylim(-0.5, N - 0.5)

    # (c) MCTS根访问分布（首步 vs 检测前一步）
    ax = fig.add_subplot(gs[1, 0])
    s0, s1 = steps[0], steps[min(T - 1, max(1, T - 2))]
    x = np.arange(N)
    w = 0.38
    v0 = s0["visits"] / max(1, s0["visits"].sum())
    v1 = s1["visits"] / max(1, s1["visits"].sum())
    ax.bar(x - w / 2, v0, w, color="#8c8c8c", alpha=0.85, label="t=1 (root visits)")
    ax.bar(x + w / 2, v1, w, color="#d94a4a", alpha=0.9,
           label=f"t={s1['t'] + 1} (root visits)")
    if target is not None:
        ax.axvline(target, color="red", ls="--", lw=1.2, alpha=0.6)
    ax.set_xlabel("sector", fontsize=10)
    ax.set_ylabel("visit fraction", fontsize=10)
    ax.set_title("(c) MCTS root visit distribution", fontsize=11,
                 fontweight="bold", loc="left")
    ax.legend(fontsize=8.5)
    ax.grid(alpha=0.3, axis="y")

    # (d) 原始策略 vs MCTS改进策略（同一步）
    ax = fig.add_subplot(gs[1, 1])
    rp = s1["raw_policy"] / max(1e-9, s1["raw_policy"].sum())
    pt = s1["policy_target"] / max(1e-9, s1["policy_target"].sum())
    ax.bar(x - w / 2, rp, w, color="#7a57c7", alpha=0.85, label="raw network policy $p_\\theta$")
    ax.bar(x + w / 2, pt, w, color="#d94a4a", alpha=0.9, label="MCTS-improved $\\pi$")
    if target is not None:
        ax.axvline(target, color="red", ls="--", lw=1.2, alpha=0.6)
    ax.set_xlabel("sector", fontsize=10)
    ax.set_ylabel("probability", fontsize=10)
    ax.set_title(f"(d) Raw vs MCTS-improved policy (t={s1['t'] + 1})", fontsize=11,
                 fontweight="bold", loc="left")
    ax.legend(fontsize=8.5)
    ax.grid(alpha=0.3, axis="y")

    fig.suptitle("Mechanism analysis of MuZero on a representative Medium episode "
                 "(real records, seed 101)", fontsize=12.5, fontweight="bold", y=0.98)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight")
    print(f"✅ {OUT}  episode_len={T} target={target}")


if __name__ == "__main__":
    main()
