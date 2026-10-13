# -*- coding: utf-8 -*-
"""先验偏差实验（开题报告第9章、式(36)、用户第18章）。

四类初始信念：uniform / correct / biased / adversarial
记录：信念随时间演化（完整轨迹）→ D_KL(belief‖true one-hot)演化→纠偏曲线
输出：results/raw/prior_bias/ + processed/prior_bias_results.csv
"""
from pathlib import Path
from typing import Dict, List

import numpy as np

from env import level1_env, scenarios as scen
from experiments import suite
from utils import metrics as mt
from utils.config import load_section
from utils.io_utils import write_csv, write_json


def run_prior_bias(results_root: Path, env_cfg: Dict, device: str, smoke: bool,
                   seeds: List[int], resume: bool = True) -> List[Dict]:
    eval_cfg = load_section("evaluation", smoke)
    from utils.config import load_config
    pb_cfg = load_config("robustness").get("prior_bias", {})
    if smoke:
        pb_cfg = {**pb_cfg, **(load_config("robustness").get("smoke", {}).get("prior_bias", {}))}
    train_cfg = load_section("training", smoke)
    n_ep = int(pb_cfg.get("episodes", 1000))
    n_traj = int(pb_cfg.get("belief_traj_episodes", 50))
    diff = pb_cfg.get("difficulty", "medium")
    ood_seed = int(train_cfg["ood_scenario_seed"])
    prior_types = pb_cfg.get("prior_types", ["uniform", "correct", "biased", "adversarial"])
    muzero_cfg = load_section("muzero", smoke)
    stage = "prior_bias"
    rows: List[Dict] = []
    out_csv = results_root / "processed" / "prior_bias_results.csv"
    if resume and out_csv.exists():
        import csv
        with open(out_csv, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    # 基准场景（Medium），各先验类型替换b0
    base = scen.get_or_create_scenarios(env_cfg, diff, n_ep, ood_seed, "ood", results_root,
                                        condition_label="priorbias_base")
    for pt in prior_types:
        scs = _reprior(base, pt, ood_seed)
        entries = []
        for algo in ["sequential", "belief_greedy"]:
            entries.append({"algorithm": algo, "seed": 0})
        for seed in seeds:
            entries.append({"algorithm": "random", "seed": seed})
        for algo in suite.RL_ALGOS:
            for seed in seeds:
                entries.append({"algorithm": algo, "seed": seed, "alg_cfg": load_section(algo, smoke)})
        for entry in entries:
            rawf = results_root / "raw" / stage / pt / \
                f"{entry['algorithm']}_main_seed{entry['seed']}.jsonl"
            if resume and rawf.exists() and rawf.stat().st_size > 0:
                continue
            # 记录完整历史（信念轨迹）以计算KL演化
            e2 = dict(entry)
            e2["record_history"] = True
            row = suite.evaluate_entry(e2, scs, env_cfg, eval_cfg, results_root, device,
                                       stage=stage, condition=pt)
            # KL演化分析（式36）与b0的KL
            kl_rows = _kl_curves(results_root, stage, pt, entry, scs)
            row["prior_type"] = pt
            row["kl_initial_mean"] = kl_rows.get("kl_initial_mean")
            row["kl_final_mean"] = kl_rows.get("kl_final_mean")
            row["kl_curve_mean"] = kl_rows.get("kl_curve_mean")
            rows.append(row)
            write_csv(out_csv, rows)
    write_csv(out_csv, rows)
    return rows


def _reprior(scs, prior_type, ood_seed):
    from env import belief as bl
    out = []
    for i, sc in enumerate(scs):
        rng = np.random.default_rng(ood_seed * 100003 + i + 77)
        b0 = bl.make_initial_belief(sc["n_sectors"], prior_type, sc["target"], rng=rng,
                                    sigma_sectors=sc.get("prior_sigma", 1.0))
        sc2 = dict(sc)
        sc2["b0"] = b0.tolist()
        sc2["prior_type"] = prior_type
        out.append(sc2)
    return out


def _kl_curves(results_root, stage, prior_type, entry, scs) -> Dict:
    """从history文件计算KL(belief‖true)逐episode逐step曲线（式36）。"""
    hist_path = results_root / "raw" / "histories" / stage / prior_type / \
        f"{entry['algorithm']}_main_seed{entry['seed']}_history.json"
    if not hist_path.exists():
        return {}
    from utils.io_utils import read_json
    recs = read_json(hist_path)
    curves = []
    for r in recs:
        N = r["n_sectors"]
        true = np.zeros(N)
        true[r["target"]] = 1.0
        # t=0: b0来自场景；之后每步belief
        pts = [0.0]
        for h in (r.get("history") or []):
            pts.append(mt.kl_divergence(true, np.asarray(h["belief"])))
        curves.append({"scenario_index": r["scenario_index"], "success": r["success"],
                       "steps": r["steps"], "kl": pts})
    if not curves:
        return {}
    L = max(len(c["kl"]) for c in curves)
    agg = []
    for t in range(L):
        vals = [c["kl"][t] for c in curves if t < len(c["kl"])]
        agg.append(float(np.mean(vals)) if vals else None)
    out = results_root / "processed" / "prior_bias_kl" / \
        f"{entry['algorithm']}_main_seed{entry['seed']}_{prior_type}.json"
    write_json(out, {"prior_type": prior_type, "algorithm": entry["algorithm"],
                     "seed": entry["seed"], "kl_mean_curve": agg, "curves": curves[:100]})
    return {
        "kl_initial_mean": agg[0] if agg else None,
        "kl_final_mean": agg[-1] if agg else None,
        "kl_curve_mean": float(np.mean([v for v in agg if v is not None])) if agg else None,
    }
