# -*- coding: utf-8 -*-
"""Level 2信号级验证实验（开题报告第6章、第19章）。

训练完成后，代表性算法进入Level 2信号级环境（LFM/时延/传播/噪声/混响/MF检测）：
  Sequential、Belief-Greedy、PPO（无模型中较稳定者）、MuZero
验证Level 1与Level 2的实验趋势一致性（方法排序与结论）。
输出：results/raw/level2/ + processed/level2_results.csv + 一致性分析。
"""
from pathlib import Path
from typing import Dict, List

from env import scenarios as scen
from experiments import suite
from utils.config import load_config, load_section
from utils.io_utils import write_csv, write_json


def run_level2(results_root: Path, env_cfg: Dict, device: str, smoke: bool,
               seeds: List[int], resume: bool = True) -> List[Dict]:
    eval_cfg = load_section("evaluation", smoke)
    from utils.config import load_config as _lc
    l2_doc = _lc("robustness").get("level2_validation", {})
    if smoke:
        l2_doc = {**l2_doc, **_lc("robustness").get("smoke", {}).get("level2_validation", {})}
    l2cfg = load_config("environment").get("level2", {})
    train_cfg = load_section("training", smoke)
    test_seed = int(train_cfg["test_scenario_seed"])
    n_ep = int(l2_doc.get("episodes", 300))
    diffs = l2_doc.get("difficulties", ["easy", "medium", "hard"])
    ppo_algos = l2_doc.get("algorithms", ["sequential", "belief_greedy", "ppo", "muzero"])
    stage = "level2"
    rows: List[Dict] = []
    out_csv = results_root / "processed" / "level2_results.csv"
    if resume and out_csv.exists():
        import csv
        with open(out_csv, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    for diff in diffs:
        # Level2场景=Level1固定测试场景（相同目标/距离/先验，信号级检测实现）
        scs = scen.get_or_create_scenarios(env_cfg, diff, n_ep, test_seed, "test", results_root)
        entries = []
        for algo in ppo_algos:
            if algo in ("sequential", "belief_greedy"):
                entries.append({"algorithm": algo, "seed": 0})
            elif algo == "random":
                for seed in seeds:
                    entries.append({"algorithm": algo, "seed": seed})
            else:
                for seed in seeds:
                    entries.append({"algorithm": algo, "seed": seed,
                                    "alg_cfg": load_section(algo, smoke)})
        for entry in entries:
            rawf = results_root / "raw" / stage / diff / \
                f"{entry['algorithm']}_main_seed{entry['seed']}.jsonl"
            if resume and rawf.exists() and rawf.stat().st_size > 0:
                continue
            row = suite.evaluate_entry(
                entry, scs, env_cfg, eval_cfg, results_root, device,
                stage=stage, condition=diff, level=2, level2_cfg=l2cfg,
                batch_size=8)
            rows.append(row)
            write_csv(out_csv, rows)
    write_csv(out_csv, rows)
    # Level1 vs Level2 一致性分析（方法排序相关性）
    _consistency_analysis(results_root)
    return rows


def _consistency_analysis(results_root: Path):
    """计算Level1/Level2各算法SR排序的Spearman相关性（趋势一致性）。"""
    import csv
    import numpy as np
    from scipy import stats as sps
    l1_csv = results_root / "processed" / "main_results.csv"
    l2_csv = results_root / "processed" / "level2_results.csv"
    if not (l1_csv.exists() and l2_csv.exists()):
        return
    def _load(p):
        with open(p, encoding="utf-8") as f:
            return list(csv.DictReader(f))
    l1, l2 = _load(l1_csv), _load(l2_csv)
    report = {}
    for diff in ["easy", "medium", "hard"]:
        s1 = {r["algorithm"]: float(r["SR"]) for r in l1
              if r["condition"] == diff and r["seed"] in ("0", "101", "102", "103", "104", "105")}
        s2 = {r["algorithm"]: float(r["SR"]) for r in l2
              if r["condition"] == diff and r["seed"] in ("0", "101", "102", "103", "104", "105")}
        # 聚合到算法级（多seed均值）
        a1, a2 = {}, {}
        for r in l1:
            if r["condition"] == diff:
                a1.setdefault(r["algorithm"], []).append(float(r["SR"]))
        for r in l2:
            if r["condition"] == diff:
                a2.setdefault(r["algorithm"], []).append(float(r["SR"]))
        m1 = {k: float(np.mean(v)) for k, v in a1.items()}
        m2 = {k: float(np.mean(v)) for k, v in a2.items()}
        common = sorted(set(m1) & set(m2))
        if len(common) >= 3:
            rho, p = sps.spearmanr([m1[a] for a in common], [m2[a] for a in common])
            report[diff] = {
                "spearman_rho": float(rho), "p_value": float(p),
                "level1_SR": m1, "level2_SR": m2, "algorithms": common,
            }
    write_json(results_root / "processed" / "level2_consistency.json", report)
