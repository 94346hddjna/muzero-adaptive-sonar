# -*- coding: utf-8 -*-
"""消融实验（开题报告表6，七组A-G全部实现）。

A w/o MCTS      ：评测期关闭搜索（同一训练模型直接用策略头）——B8决策
B 模拟次数       ：M∈{25,50,100,200}评测期扫描（性能-计算成本权衡）
C 规划深度       ：D_p∈{1,3,5,10}评测期扫描（多步前瞻边际收益）
D w/o belief    ：重训MuZero，obs_mode='raw'（移除信念地图）
E w/o memory    ：重训MuZero，去除GRU
F w/o 域随机化  ：重训MuZero，固定Medium
G 奖励设计       ：重训MuZero，稀疏式(16) vs 塑形PBRS

A/B/C为评测期开关（主模型不变）；D/E/F/G为独立重训（每种子独立训练）。
"""
from pathlib import Path
from typing import Dict, List

from env import scenarios as scen
from experiments import suite
from utils.config import load_section
from utils.io_utils import write_csv


def run_ablation(results_root: Path, env_cfg: Dict, device: str, smoke: bool,
                 seeds: List[int], resume: bool = True, difficulties=None) -> List[Dict]:
    eval_cfg = load_section("evaluation", smoke)
    train_cfg = load_section("training", smoke)
    abl_cfg = load_section("ablation", smoke)
    test_seed = int(train_cfg["test_scenario_seed"])
    n_ep = int(eval_cfg["test_episodes"])
    muzero_cfg = load_section("muzero", smoke)
    diffs = difficulties or ["easy", "medium", "hard"]
    rows: List[Dict] = []
    out_csv = results_root / "processed" / "ablation_results.csv"
    if resume and out_csv.exists():
        import csv
        with open(out_csv, encoding="utf-8") as f:
            rows = [_numify(r) for r in csv.DictReader(f)]
    stage = "ablation"

    def eval_entry_row(entry, scs, condition):
        suffix = entry.get("suffix", "")
        fname = f"{entry['algorithm']}_{entry['variant']}_seed{entry['seed']}" + \
            (f"_{suffix}" if suffix else "") + ".jsonl"
        rawf = results_root / "raw" / stage / condition / fname
        if resume and rawf.exists() and rawf.stat().st_size > 0:
            return None
        row = suite.evaluate_entry(entry, scs, env_cfg, eval_cfg, results_root, device,
                                   stage=stage, condition=condition)
        rows.append(row)
        write_csv(out_csv, rows)

    for diff in diffs:
        scs = scen.get_or_create_scenarios(env_cfg, diff, n_ep, test_seed, "test", results_root)
        # ---- A: w/o MCTS（评测期，主模型） ----
        for seed in seeds:
            eval_entry_row({"algorithm": "muzero", "variant": "main", "seed": seed,
                            "alg_cfg": muzero_cfg,
                            "eval_options": {"use_mcts": False}, "suffix": "woMCTS"},
                           scs, diff)
        # ---- B: 模拟次数扫描（仅Medium：性能-计算权衡在标称难度最清晰，协议文档化） ----
        if diff == "medium":
            for M in abl_cfg.get("simulations", [25, 50, 100, 200]):
                for seed in seeds:
                    eval_entry_row({"algorithm": "muzero", "variant": "main", "seed": seed,
                                    "alg_cfg": muzero_cfg,
                                    "eval_options": {"num_simulations": int(M),
                                                     "planning_depth": muzero_cfg.get("planning_depth", 5)},
                                    "suffix": f"M{M}"},
                                   scs, diff)
        # ---- C: 规划深度扫描（仅Medium，同B） ----
        if diff == "medium":
            for Dp in abl_cfg.get("planning_depths", [1, 3, 5, 10]):
                for seed in seeds:
                    eval_entry_row({"algorithm": "muzero", "variant": "main", "seed": seed,
                                    "alg_cfg": muzero_cfg,
                                    "eval_options": {"num_simulations": muzero_cfg.get("num_simulations", 50),
                                                     "planning_depth": int(Dp)},
                                    "suffix": f"D{Dp}"},
                               scs, diff)
        # ---- D/E/F/G: 重训变体（训练由run_all的train阶段完成，此处评测；
        # disabled_variants中的变体跳过，如v3盲搜索下D不适用） ----
        disabled = abl_cfg.get("disabled_variants", [])
        for variant, suffix in [("wo_belief", "woBelief"), ("wo_memory", "woMemory"),
                                ("wo_domain_rand", "woDomRand"), ("shaped", "shapedReward")]:
            if variant in disabled:
                continue
            for seed in seeds:
                eval_entry_row({"algorithm": "muzero", "variant": variant, "seed": seed,
                                "alg_cfg": muzero_cfg, "suffix": suffix},
                               scs, diff)
    write_csv(out_csv, rows)
    return rows


def _numify(r: Dict) -> Dict:
    out = {}
    for k, v in r.items():
        try:
            out[k] = float(v) if v not in (None, "") else v
        except (ValueError, TypeError):
            out[k] = v
    return out
