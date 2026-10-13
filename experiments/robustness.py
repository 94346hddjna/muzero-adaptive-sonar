# -*- coding: utf-8 -*-
"""鲁棒性/OOD实验（开题报告第9章、式(35)）。

连续难度sweep：SNR −10→+10dB、PFA 0.01→0.15、N 8→64（性能-难度曲线）
OOD五类：OOD-SNR / OOD-PFA / OOD-N / OOD-Prior / OOD-Reverberation
（训练分布之外，B7决策；配对OOD场景种子与Test隔离）
输出：results/raw/robustness/、raw/ood/ + processed CSV + Drop(d)计算。
"""
from pathlib import Path
from typing import Dict, List

from env import level1_env, scenarios as scen
from experiments import suite
from utils.config import load_section
from utils.io_utils import write_csv, write_json


def _mk_entries(seeds, smoke, eval_all_algos=True):
    entries = []
    for algo in suite.BASELINES:
        if algo in ("sequential", "belief_greedy"):
            entries.append({"algorithm": algo, "seed": 0})
        else:
            for seed in seeds:
                entries.append({"algorithm": algo, "seed": seed})
    for algo in suite.RL_ALGOS:
        for seed in seeds:
            entries.append({"algorithm": algo, "seed": seed,
                            "alg_cfg": load_section(algo, smoke)})
    return entries


def run_robustness_sweeps(results_root: Path, env_cfg: Dict, device: str, smoke: bool,
                          seeds: List[int], resume: bool = True) -> List[Dict]:
    """连续难度sweep（SNR/PFA/N三轴，Medium基准）。"""
    eval_cfg = load_section("evaluation", smoke)
    rob_cfg = load_section("robustness", smoke)
    train_cfg = load_section("training", smoke)
    n_ep = int(rob_cfg.get("episodes_per_condition", 200))
    ood_seed = int(train_cfg["ood_scenario_seed"])
    sweeps = rob_cfg.get("sweeps", {})
    rows: List[Dict] = []
    out_csv = results_root / "processed" / "robustness_results.csv"
    if resume and out_csv.exists():
        import csv
        with open(out_csv, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    entries = _mk_entries(seeds, smoke)
    stage = "robustness"

    def _run_conditions(conditions, sweep_name):
        for cond_name, overrides in conditions:
            scs = _get_sweep_scenarios(env_cfg, overrides, n_ep, ood_seed,
                                       results_root, cond_name)
            for entry in entries:
                rawf = _raw_path(results_root, stage, cond_name, entry)
                if resume and rawf.exists() and rawf.stat().st_size > 0:
                    continue
                row = suite.evaluate_entry(entry, scs, env_cfg, eval_cfg, results_root,
                                           device, stage=stage, condition=cond_name)
                row["sweep"] = sweep_name
                rows.append(row)
                write_csv(out_csv, rows)

    _run_conditions([(f"snr_{o:+g}dB", {"snr_offset_db": float(o)})
                     for o in sweeps.get("snr_offsets_db", [])], "snr")
    _run_conditions([(f"pfa_{p:.2f}", {"pfa": float(p)})
                     for p in sweeps.get("pfa_values", [])], "pfa")
    _run_conditions([(f"N{n}", {"n_sectors": int(n)})
                     for n in sweeps.get("n_sectors_values", [])], "n_sectors")
    write_csv(out_csv, rows)
    return rows


def _get_sweep_scenarios(env_cfg, overrides, n_ep, ood_seed, results_root, label):
    """sweep/OOD场景：以Medium难度参数为基础施加覆盖（OOD种子空间，与Test隔离）。"""
    params = level1_env.difficulty_params(env_cfg, "medium", overrides)
    key = f"sweep_{label}_n{n_ep}"
    path = results_root / "raw" / "scenarios" / f"ood_{label}_n{n_ep}.json"
    from utils.io_utils import read_json
    if path.exists():
        return read_json(path)
    scs = []
    for i in range(n_ep):
        rng = _scenario_rng(ood_seed, i, label)
        scs.append(_scenario_from_params(params, env_cfg, rng, i, label))
    write_json(path, scs)
    return scs


def _raw_path(results_root, stage, cond, entry):
    return results_root / "raw" / stage / cond / \
        f"{entry['algorithm']}_main_seed{entry['seed']}.jsonl"


def _scenario_rng(seed, i, label):
    """按条件标签派生完全独立的64位种子流（修复：原小偏移叠加导致跨条件
    流为彼此移位副本；sweep/OOD场景需重新生成并重评测）。"""
    import hashlib
    import numpy as np
    full = int(hashlib.md5(f"{seed}|{label}|{i}".encode()).hexdigest()[:16], 16)
    return np.random.default_rng(full)


def _scenario_from_params(params, env_cfg, rng, i, label):
    import numpy as np
    from env.level1_env import make_scenario_from_params
    rng_seed = int(rng.integers(0, 2**31 - 1))
    return make_scenario_from_params(params, env_cfg, rng, rng_seed, scenario_index=i)


def run_ood(results_root: Path, env_cfg: Dict, device: str, smoke: bool,
            seeds: List[int], resume: bool = True) -> List[Dict]:
    """OOD五类测试（+ID基线）。Drop(d)=（P_nom−P_d)/P_nom（式35）在分析阶段计算。"""
    eval_cfg = load_section("evaluation", smoke)
    rob_cfg = load_section("robustness", smoke)
    train_cfg = load_section("training", smoke)
    n_ep = int(rob_cfg.get("episodes_per_condition", 1000))
    ood_seed = int(train_cfg["ood_scenario_seed"])
    ood_specs = rob_cfg.get("ood", {})
    rows: List[Dict] = []
    out_csv = results_root / "processed" / "ood_results.csv"
    if resume and out_csv.exists():
        import csv
        with open(out_csv, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    entries = _mk_entries(seeds, smoke)
    stage = "ood"
    for cond_name, spec in ood_specs.items():
        overrides = {k: v for k, v in spec.items() if k != "difficulty"}
        prior_type = overrides.pop("prior_type", None)
        scs = _get_sweep_scenarios(env_cfg, overrides, n_ep, ood_seed, results_root, cond_name)
        if prior_type:
            # OOD-Prior：替换场景先验类型
            scs = _reprior_scenarios(scs, prior_type, ood_seed, cond_name)
        for entry in entries:
            rawf = _raw_path(results_root, stage, cond_name, entry)
            if resume and rawf.exists() and rawf.stat().st_size > 0:
                continue
            row = suite.evaluate_entry(entry, scs, env_cfg, eval_cfg, results_root, device,
                                       stage=stage, condition=cond_name)
            row["is_ood"] = cond_name != "id"
            rows.append(row)
            write_csv(out_csv, rows)
    write_csv(out_csv, rows)
    return rows


def _reprior_scenarios(scs, prior_type, ood_seed, label):
    """按先验类型重生成b0（目标/距离不变，先验变化）。"""
    import numpy as np
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
