# -*- coding: utf-8 -*-
"""收集三轮实验全部真实数据 → 写JSON供Node构建docx（完整学术规格）。"""
import json
import csv
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def load_csv(p):
    if not p.exists():
        return []
    return list(csv.DictReader(open(p, encoding="utf-8")))


def fval(v):
    return float(v) if v not in (None, "", "None") else None


def agg(rows, algo, cond, metric):
    vals = [fval(r[metric]) for r in rows
            if r.get("algorithm") == algo and r.get("condition") == cond]
    vals = [v for v in vals if v is not None]
    if not vals:
        return None, None, 0
    return float(np.mean(vals)), float(np.std(vals)), len(vals)


def ads(rows, algo, cond):
    vals = [fval(r["ADS"]) for r in rows
            if r.get("algorithm") == algo and r.get("condition") == cond]
    vals = [v for v in vals if v is not None]
    return float(np.mean(vals)) if vals else None


def mets(rows, algo, cond):
    """返回完整指标字典：SR/ADS/EDC/FAR/Return/Energy/Latency"""
    out = {}
    for m in ["SR", "ADS", "EDC", "FAR", "ExpectedReturn", "EnergyPerStep",
              "MeanDecisionLatencyMs", "AvgSteps"]:
        out[m] = agg(rows, algo, cond, m)[0]
    return out


main1 = load_csv(ROOT / "results" / "processed" / "main_results.csv")
main2 = load_csv(ROOT / "results_v2" / "processed" / "main_results.csv")
main3 = load_csv(ROOT / "results_v3" / "processed" / "main_results.csv")
tests3 = load_csv(ROOT / "results_v3" / "statistics" / "statistical_tests.csv")
abl1 = load_csv(ROOT / "results" / "processed" / "ablation_results.csv")
abl3 = load_csv(ROOT / "results_v3" / "processed" / "ablation_results.csv")
rob3 = load_csv(ROOT / "results_v3" / "processed" / "robustness_results.csv")
ood3 = load_csv(ROOT / "results_v3" / "processed" / "ood_results.csv")
pb3 = load_csv(ROOT / "results_v3" / "processed" / "prior_bias_results.csv")
l2_3 = load_csv(ROOT / "results_v3" / "processed" / "level2_results.csv")
l2_consistency = json.load(open(ROOT / "results_v3" / "processed" / "level2_consistency.json"))
cal = json.load(open(ROOT / "results_v3" / "raw" / "calibration.json"))
# 训练曲线（MuZero主模型种子101）
train_log = load_csv(ROOT / "results_v3" / "checkpoints" / "muzero" / "main_seed101"
                     / "muzero_main_seed101_training_log.csv")

ALGOS = ["sequential", "random", "belief_greedy", "pomcp",
         "dqn", "ppo", "dqn_gru", "ppo_stack", "muzero"]


def agg_pb(rows, algo, pt):
    v = [fval(r["SR"]) for r in rows
         if r.get("algorithm") == algo and r.get("prior_type") == pt and r["SR"]]
    v = [x for x in v if x is not None]
    return float(np.mean(v)) if v else None


def full_metrics(rows):
    return {d: {a: mets(rows, a, d) for a in ALGOS} for d in ["easy", "medium", "hard"]}


def ablation_summary(abl, cond="medium"):
    mz_main = np.mean([fval(r["SR"]) for r in abl
                       if r["algorithm"] == "muzero" and r["condition"] == cond
                       and r.get("variant") == "main"])
    out = {"main": float(mz_main)}
    # A: w/o MCTS（eval_options）
    wo = [fval(r["SR"]) for r in abl if r["algorithm"] == "muzero"
          and r["condition"] == cond and "use_mcts" in str(r.get("eval_options", ""))]
    if wo:
        out["wo_mcts"] = {"sr": float(np.mean(wo)), "std": float(np.std(wo))}
    # B: simulations
    for M in [25, 50, 100, 200]:
        vals = [fval(r["SR"]) for r in abl if r["algorithm"] == "muzero"
                and r["condition"] == cond
                and f'"num_simulations": {M}' in str(r.get("eval_options", "")).replace(".0", "")
                and '"planning_depth": 5' in str(r.get("eval_options", "")).replace(".0", "")]
        vals = [v for v in vals if v is not None]
        if vals:
            out[f"M{M}"] = {"sr": float(np.mean(vals)), "std": float(np.std(vals))}
        lat = [fval(r["MeanDecisionLatencyMs"]) for r in abl if r["algorithm"] == "muzero"
               and r["condition"] == cond
               and f'"num_simulations": {M}' in str(r.get("eval_options", "")).replace(".0", "")
               and '"planning_depth": 5' in str(r.get("eval_options", "")).replace(".0", "")]
        lat = [v for v in lat if v is not None]
        if lat and f"M{M}" in out:
            out[f"M{M}"]["lat_ms"] = float(np.mean(lat))
    # C: depth
    for D in [1, 3, 5, 10]:
        vals = [fval(r["SR"]) for r in abl if r["algorithm"] == "muzero"
                and r["condition"] == cond
                and f'"planning_depth": {D}' in str(r.get("eval_options", "")).replace(".0", "")
                and '"num_simulations": 100' in str(r.get("eval_options", "")).replace(".0", "")]
        vals = [v for v in vals if v is not None]
        if vals:
            out[f"D{D}"] = {"sr": float(np.mean(vals)), "std": float(np.std(vals))}
    # D/E/F/G variants
    for variant in ["wo_belief", "wo_memory", "wo_domain_rand", "shaped"]:
        vals = [fval(r["SR"]) for r in abl if r["algorithm"] == "muzero"
                and r["condition"] == cond and r.get("variant") == variant and r["SR"]]
        vals = [v for v in vals if v is not None]
        if vals:
            out[variant] = {"sr": float(np.mean(vals)), "std": float(np.std(vals)),
                            "delta": float(mz_main - np.mean(vals))}
    return out


def get_stat(cond, algo_b):
    for r in tests3:
        if (r.get("stage") == "main" and r.get("condition") == cond
                and r.get("algo_a") == "muzero" and r.get("algo_b") == algo_b
                and r.get("metric") == "SR"):
            return {"p": float(r["p_holm"]), "effect": float(r["effect_size"]),
                    "diff": float(r["mean_diff"]), "sig": r["significant_holm_0.05"] == "True",
                    "test": r.get("test")}
    return None


# sweep数据
def sweep_data(rows, sweep_name):
    out = {}
    for r in rows:
        if r.get("algorithm") == "muzero" and str(r.get("sweep")) == sweep_name:
            cond = r["condition"]
            sr = fval(r["SR"])
            if sr is not None:
                out.setdefault(cond, []).append(sr)
    return {k: float(np.mean(v)) for k, v in out.items()}


# 训练成本（小时/种子，summary中为跨会话累计值）
import glob as _glob
train_cost = {}
for _a in ["dqn", "ppo", "dqn_gru", "ppo_stack", "muzero"]:
    _walls = []
    for _f in sorted(_glob.glob(str(ROOT / "results_v3" / "checkpoints" / _a / "main_seed*" / "*_summary.json"))):
        try:
            _d = json.load(open(_f))
            if _d.get("training_wall_clock_sec"):
                _walls.append(float(_d["training_wall_clock_sec"]))
        except Exception:
            pass
    if _walls:
        train_cost[_a] = float(np.mean(_walls)) / 3600.0

data = {
    "v1": full_metrics(main1),
    "v2": full_metrics(main2),
    "v3": full_metrics(main3),
    "abl1": ablation_summary(abl1),
    "abl3": ablation_summary(abl3),
    "l2_consistency": l2_consistency,
    "calibration": cal,
    "stats_v3": {f"{c}_{a}": get_stat(c, a) for c in ["easy", "medium", "hard"]
                  for a in ["sequential", "random", "belief_greedy", "pomcp",
                          "dqn", "ppo", "dqn_gru", "ppo_stack"]
                  if get_stat(c, a)},
    "sweep_snr": sweep_data(rob3, "snr"),
    "sweep_pfa": sweep_data(rob3, "pfa"),
    "sweep_n": sweep_data(rob3, "n_sectors"),
    "ood": {c: agg(ood3, "muzero", c, "SR") for c in
            ["id", "ood_snr", "ood_pfa", "ood_n", "ood_prior", "ood_reverb"]},
    "ood_all": {c: {a: agg(ood3, a, c, "SR") for a in ALGOS} for c in
                ["id", "ood_snr", "ood_pfa", "ood_n", "ood_prior", "ood_reverb"]},
    "prior_bias": {pt: {a: agg_pb(pb3, a, pt) for a in ALGOS}
                   for pt in ["uniform", "correct", "biased", "adversarial"]},
    "train_cost_h": train_cost,
    "train_log": [{"env_steps": int(r["env_steps"]), "success_rate": fval(r["success_rate"]),
                   "episode_return": fval(r["episode_return"])} for r in train_log],
}
out = ROOT / "tools" / "paper_data.json"
with open(out, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, default=float)
print(f"数据已写: {out}")
print(f"包含: {list(data.keys())}")
