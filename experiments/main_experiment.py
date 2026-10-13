# -*- coding: utf-8 -*-
"""主实验编排（开题报告第7-8章、表4、用户第15章）。

流程：训练（DQN/PPO/MuZero×5种子，Sequential/Random/Belief-Greedy免训练）
→ 锁定best_val checkpoint → 在Easy/Medium/Hard固定测试场景上统一评测
（每条件≥1000回合，6算法，配对）→ results/raw/main/ + processed/main_results.csv
"""
import time
from pathlib import Path
from typing import Dict, List

from env import scenarios as scen
from experiments import suite
from experiments.train_rl import train_rl
from utils.config import load_config, load_section
from utils.io_utils import (done_marker, is_done, read_json, write_csv, write_json)
from utils import manifest as mf


def run_training_stage(results_root: Path, env_cfg: Dict, device: str,
                       smoke: bool, seeds: List[int], algorithms: List[str],
                       variants: List[str] = None, resume: bool = True) -> List[Dict]:
    """训练全部(algorithm, variant, seed)组合（表7超参/表6消融变体）。"""
    train_cfg = load_section("training", smoke)
    summaries = []
    variants = variants or ["main"]
    for algo in algorithms:
        alg_cfg = load_section(algo, smoke)
        for variant in variants:
            # 消融D/E/F/G仅MuZero（表6框架）
            if variant != "main" and algo != "muzero":
                continue
            for seed in seeds:
                run_dir = results_root / "checkpoints" / algo / f"{variant}_seed{seed}"
                exp_id = f"{algo}_{variant}_seed{seed}"
                if resume and is_done(run_dir, exp_id):
                    p = run_dir / f"{exp_id}_summary.json"
                    if p.exists():
                        summaries.append(read_json(p))
                    continue
                t0 = time.time()
                s = train_rl(algo, env_cfg, alg_cfg, train_cfg, seed=seed,
                             run_dir=run_dir, variant=variant, smoke=smoke,
                             resume=resume, device=device)
                s["wall_clock_this_session"] = round(time.time() - t0, 1)
                summaries.append(s)
                # manifest（第32章）
                m = mf.make_manifest(
                    experiment_id=f"train_{exp_id}", algorithm=algo, seed=seed,
                    difficulty="domain_rand" if variant == "main" else variant,
                    config={"env": env_cfg, "algo": alg_cfg, "training": train_cfg},
                    training_steps=s.get("env_steps"),
                    checkpoint_path=s.get("checkpoints", {}).get("best_val"),
                    stage="train", extra={"variant": variant})
                mf.save_manifest(m, results_root)
    return summaries


def run_main_test(results_root: Path, env_cfg: Dict, device: str, smoke: bool,
                  seeds: List[int], resume: bool = True) -> List[Dict]:
    """主测试：6算法×Easy/Medium/Hard（固定测试场景，配对评测）。"""
    eval_cfg = load_section("evaluation", smoke)
    train_cfg = load_section("training", smoke)
    n_ep = int(eval_cfg["test_episodes"])
    test_seed = int(train_cfg["test_scenario_seed"])
    stage = "main"
    rows: List[Dict] = []
    out_csv = results_root / "processed" / "main_results.csv"
    if resume and out_csv.exists():
        import csv
        with open(out_csv, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    for diff in ["easy", "medium", "hard"]:
        scs = scen.get_or_create_scenarios(env_cfg, diff, n_ep, test_seed, "test", results_root)
        # 场景中记录PD分布（用户要求：记录实际PD分布）
        entries = []
        for algo in suite.BASELINES:
            if algo in ("sequential", "belief_greedy"):
                entries.append({"algorithm": algo, "seed": 0})   # 确定性策略单次评测
            else:
                for seed in seeds:
                    entries.append({"algorithm": algo, "seed": seed})
        for algo in suite.RL_ALGOS:
            for seed in seeds:
                entries.append({"algorithm": algo, "seed": seed,
                                "alg_cfg": load_section(algo, smoke)})
        for entry in entries:
            key = f"{entry['algorithm']}_{entry['seed']}"
            rawf = results_root / "raw" / stage / diff / f"{entry['algorithm']}_main_seed{entry['seed']}.jsonl"
            if resume and rawf.exists() and rawf.stat().st_size > 0:
                continue  # 已评测（断点续跑）
            mech = set(range(min(30, n_ep))) if (entry["algorithm"] == "muzero" and diff == "medium") else None
            row = suite.evaluate_entry(
                entry, scs, env_cfg, eval_cfg, results_root, device,
                stage=stage, condition=diff, mechanism_indices=mech,
                batch_size=16)
            rows.append(row)
            write_csv(out_csv, rows)
    write_csv(out_csv, rows)
    return rows


def run_all(results_root: Path, smoke: bool = False, device: str = "auto",
            seeds: List[int] = None, resume: bool = True):
    env_full = load_config("environment")
    env_cfg = env_full["environment"]
    dev = device if device != "auto" else (
        "cuda" if __import__("torch").cuda.is_available() else "cpu")
    seeds = seeds or [101, 102, 103, 104, 105]
    summaries = run_training_stage(results_root, env_cfg, dev, smoke, seeds,
                                   ["dqn", "ppo", "muzero"])
    rows = run_main_test(results_root, env_cfg, dev, smoke, seeds, resume)
    return {"training_summaries": summaries, "main_rows": rows}


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    root = Path(__file__).resolve().parent.parent / "results"
    smoke = "--smoke" in sys.argv
    out = run_all(root, smoke=smoke)
    print(f"main experiment done: {len(out['main_rows'])} result rows")
