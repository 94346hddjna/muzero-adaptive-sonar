# -*- coding: utf-8 -*-
"""典型案例可视化（用户第22章）。

自动从真实评测历史中选取四类典型回合并生成组合图（信念热图+动作轨迹+检出标记）：
  Case 1 正确先验+低噪声：Easy难度、correct先验、成功回合
  Case 2 错误先验+高噪声：prior_bias/biased或adversarial、成功纠偏回合
  Case 3 连续漏检：多步未检出后仍成功（长回合）
  Case 4 出现虚警：含false alarm的回合
每个案例记录：真实target、每步action/observation/belief、最终结果。
输出：results/figures/case{1..4}_{lang}.{png,pdf} + results/raw/case_studies.json
"""
from pathlib import Path
from typing import Dict, List

import numpy as np

from utils.io_utils import read_json


def _pick_cases(results_root: Path) -> Dict[str, Dict]:
    """从prior_bias histories与main histories中选取四类案例。"""
    cases = {}
    # Case 1: Easy+正确先验+成功（低噪声由Easy难度保证）
    f1 = results_root / "raw" / "histories" / "prior_bias" / "correct" / \
        "muzero_main_seed101_history.json"
    if not f1.exists():
        f1 = results_root / "raw" / "histories" / "prior_bias" / "correct" / \
            "belief_greedy_main_seed0_history.json"
    if f1.exists():
        recs = read_json(f1)
        succ = [r for r in recs if r["success"] and r["steps"] <= 6]
        if succ:
            cases["case1_correct_prior_low_noise"] = succ[0]
    # Case 2: 错误先验+成功纠偏
    for pt in ["biased", "adversarial"]:
        f2 = results_root / "raw" / "histories" / "prior_bias" / pt / \
            "muzero_main_seed101_history.json"
        if not f2.exists():
            f2 = results_root / "raw" / "histories" / "prior_bias" / pt / \
                "belief_greedy_main_seed0_history.json"
        if f2.exists():
            recs = read_json(f2)
            succ = [r for r in recs if r["success"]]
            if succ:
                cases["case2_wrong_prior_recovery"] = succ[0]
                break
    # Case 3/4: 从任一history集合找连续漏检/虚警回合
    pool = []
    hdir = results_root / "raw" / "histories" / "prior_bias"
    if hdir.exists():
        for pt_dir in hdir.iterdir():
            for f in pt_dir.glob("muzero_main_seed*_history.json"):
                pool.extend(read_json(f))
    if pool:
        miss3 = [r for r in pool if r["success"] and r["steps"] >= 10]
        if miss3:
            cases["case3_consecutive_misses"] = sorted(miss3, key=lambda r: -r["steps"])[0]
        fa = [r for r in pool if r.get("history")
              and any(h["detection"] == 1 for h in r["history"][:-1])
              and not r["success"]]
        if fa:
            cases["case4_false_alarm"] = fa[0]
    return cases


def run_case_studies(results_root: Path, smoke: bool = False) -> Dict:
    cases = _pick_cases(results_root)
    if not cases:
        return {"n_cases": 0}
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return {"n_cases": len(cases), "figures": "matplotlib unavailable"}
    from experiments.figures import _save, _setup_zh_font
    _setup_zh_font()
    fig_dir = results_root / "figures"
    case_desc_zh = {
        "case1_correct_prior_low_noise": "Case 1: 正确先验+低噪声（Easy）",
        "case2_wrong_prior_recovery": "Case 2: 错误先验+高噪声纠偏",
        "case3_consecutive_misses": "Case 3: 连续漏检后成功",
        "case4_false_alarm": "Case 4: 出现虚警",
    }
    case_desc_en = {
        "case1_correct_prior_low_noise": "Case 1: correct prior + low noise (Easy)",
        "case2_wrong_prior_recovery": "Case 2: wrong prior recovery",
        "case3_consecutive_misses": "Case 3: consecutive misses",
        "case4_false_alarm": "Case 4: false alarm episode",
    }
    manifest = {}
    for key, r in cases.items():
        hist = r.get("history") or []
        if not hist:
            continue
        N = r["n_sectors"]
        target = r["target"]
        bmat = np.zeros((len(hist) + 1, N))
        for t, h in enumerate(hist):
            bmat[t + 1] = np.asarray(h["belief"])
        acts = [h["action"] for h in hist]
        dets = [h["detection"] for h in hist]
        rewards = [h["reward"] for h in hist]
        for lang in ["zh", "en"]:
            fig, axes = plt.subplots(2, 1, figsize=(7.5, 6.5),
                                     gridspec_kw={"height_ratios": [2.4, 1]})
            im = axes[0].imshow(bmat, aspect="auto", cmap="hot", vmin=0, vmax=1)
            axes[0].plot(target, 0, marker="*", color="cyan", markersize=14)
            axes[0].set_xlabel("扇区编号" if lang == "zh" else "Sector index")
            axes[0].set_ylabel("探测步 t" if lang == "zh" else "Ping step t")
            axes[0].set_title((case_desc_zh if lang == "zh" else case_desc_en)[key])
            fig.colorbar(im, ax=axes[0], label="信念 b_t(i)", fraction=0.03)
            xs = range(len(acts))
            axes[1].step(xs, acts, where="mid", color="#4a90d9", linewidth=1.5,
                         label="动作" if lang == "zh" else "Action")
            det_x = [i for i, d in enumerate(dets) if d]
            axes[1].scatter(det_x, [acts[i] for i in det_x], marker="v", color="red",
                            zorder=5, s=55, label="检出" if lang == "zh" else "Detection")
            axes[1].axhline(y=target, color="green", linestyle="--", alpha=0.7,
                            label="真实目标扇区" if lang == "zh" else "True target sector")
            axes[1].set_xlabel("探测步 t" if lang == "zh" else "Ping step t")
            axes[1].set_ylabel("扇区编号" if lang == "zh" else "Sector index")
            axes[1].legend(fontsize=8)
            axes[1].grid(alpha=0.3)
            _save(fig, fig_dir, f"{key}_{lang}")
        manifest[key] = {
            "target": target, "n_sectors": N, "steps": r["steps"],
            "success": r["success"], "pd_ep": r.get("pd_ep"),
            "actions": acts, "detections": dets, "rewards": rewards,
            "beliefs": bmat.tolist(), "difficulty": r.get("difficulty"),
            "prior_type": r.get("prior_type"),
        }
    from utils.io_utils import write_json
    write_json(results_root / "raw" / "case_studies.json", manifest)
    return {"n_cases": len(manifest), "cases": list(manifest.keys())}
