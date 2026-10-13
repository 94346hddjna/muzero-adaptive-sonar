# -*- coding: utf-8 -*-
"""条件评测引擎（主实验/消融/OOD/鲁棒性/Level2共用的统一评测核心）。

公平性保障（开题报告第7章）：
- 所有算法在完全相同的固定场景（目标位置/环境参数/先验/每回合RNG种子）下评测
- RL算法统一加载best_validation_checkpoint（第35章：Test前参数锁定）
- 逐回合原始记录落盘results/raw/<stage>/<condition>/<algo>_<variant>_seed<k>.jsonl
- 机理捕获：MCTS根访问分布/原始策略vs改进策略（图18-20数据）
"""
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from agents.dqn import DQNAgent
from agents.muzero import MuZeroAgent
from agents.ppo import PPOAgent
from baselines import BeliefGreedyPolicy, RandomPolicy, SequentialPolicy
from env import level1_env
from env.level1_env import SonarLevel1VecEnv, obs_dim
from env.level2_env import SonarLevel2VecEnv
from experiments.evaluate import AgentActWrapper, BaselineActWrapper, run_evaluation
from utils import metrics as mt
from utils.io_utils import JsonlWriter, write_json

RL_ALGOS = ["dqn", "ppo", "muzero"]
BASELINES = ["sequential", "random", "belief_greedy"]
ALL_ALGOS = BASELINES + RL_ALGOS


def algo_obs_mode(env_cfg: Dict, algorithm: str, variant: str = "main") -> str:
    """v3按算法分配观测模式（与train_rl._algo_obs_mode一致）。"""
    if variant == "wo_belief":
        return "raw"
    return env_cfg.get("obs_mode_by_algo", {}).get(algorithm, "belief")


def build_agent(algorithm: str, variant: str, alg_cfg: Dict, env_cfg: Dict,
                device: str, seed: int) -> object:
    """按算法/变体构建智能体（不含权重）。"""
    canonical_cells = int(env_cfg.get("canonical_cells", 192))
    obs_mode = algo_obs_mode(env_cfg, algorithm, variant)
    D = obs_dim(canonical_cells, obs_mode)
    if algorithm == "muzero":
        return MuZeroAgent(alg_cfg, D, canonical_cells, device=device,
                           obs_mode=obs_mode, use_memory=(variant != "wo_memory"),
                           seed=seed)
    if algorithm == "dqn":
        return DQNAgent(alg_cfg, D, canonical_cells, device=device, seed=seed,
                        obs_mode=obs_mode)
    if algorithm == "ppo":
        return PPOAgent(alg_cfg, D, canonical_cells, device=device, seed=seed)
    if algorithm == "sequential":
        return SequentialPolicy(seed=seed)
    if algorithm == "random":
        return RandomPolicy(seed=seed)
    if algorithm == "belief_greedy":
        return BeliefGreedyPolicy(seed=seed)
    raise ValueError(algorithm)


def load_trained_agent(algorithm: str, variant: str, seed: int, results_root: Path,
                       alg_cfg: Dict, env_cfg: Dict, device: str,
                       ckpt_kind: str = "best_val") -> object:
    """加载训练好的智能体（best_val/final/last）。"""
    agent = build_agent(algorithm, variant, alg_cfg, env_cfg, device, seed)
    ck = results_root / "checkpoints" / algorithm / f"{variant}_seed{seed}" / \
        f"{algorithm}_{variant}_seed{seed}_{ckpt_kind}.ckpt"
    if not ck.exists():
        # 回退final/last
        for alt in ["final", "last"]:
            alt_p = ck.parent / f"{algorithm}_{variant}_seed{seed}_{alt}.ckpt"
            if alt_p.exists():
                ck = alt_p
                break
    if not ck.exists():
        raise FileNotFoundError(f"checkpoint not found: {ck}")
    agent.load_checkpoint(ck, load_optimizer=False)
    return agent


def evaluate_entry(
    entry: Dict,
    scenarios: List[Dict],
    env_cfg: Dict,
    eval_cfg: Dict,
    results_root: Path,
    device: str,
    stage: str,
    condition: str,
    level: int = 1,
    level2_cfg: Optional[Dict] = None,
    batch_size: int = 16,
    mechanism_indices: Optional[set] = None,
    t_max: int = 20,
    condition_params: Optional[Dict] = None,
) -> Dict:
    """评测单个(algorithm, variant, seed)在给定场景集。返回聚合指标行。"""
    algorithm = entry["algorithm"]
    variant = entry.get("variant", "main")
    seed = entry.get("seed", 0)
    canonical_cells = int(env_cfg.get("canonical_cells", 192))
    obs_mode = algo_obs_mode(env_cfg, algorithm, variant)
    # 智能体
    if "agent" in entry:
        agent = entry["agent"]
    elif algorithm in RL_ALGOS:
        agent = load_trained_agent(
            algorithm, variant, seed, results_root, entry["alg_cfg"], env_cfg, device,
            ckpt_kind=entry.get("ckpt_kind", "best_val"))
    else:
        agent = build_agent(algorithm, variant, {}, env_cfg, device, seed)
    # 评测期选项（消融A/B/C）
    if algorithm == "muzero":
        opts = entry.get("eval_options", {})
        if "use_mcts" in opts:
            agent.set_policy_only(not opts["use_mcts"])       # 消融A
        if opts.get("num_simulations") is not None:
            agent.set_mcts_overrides(num_simulations=opts["num_simulations"])
        if opts.get("planning_depth") is not None:
            agent.set_mcts_overrides(planning_depth=opts["planning_depth"])
        if entry.get("eval_options", {}).get("reset_mcts"):
            agent.set_mcts_overrides(num_simulations=entry["alg_cfg"].get("num_simulations", 50),
                                     planning_depth=entry["alg_cfg"].get("planning_depth", 5))
    # 环境（Level 1/Level 2）
    if level == 2:
        env = SonarLevel2VecEnv(
            num_envs=min(batch_size, len(scenarios)), env_cfg=env_cfg, t_max=t_max,
            canonical_cells=canonical_cells, master_seed=seed, obs_mode=obs_mode,
            level2_cfg=level2_cfg or {},
            difficulty_overrides=condition_params or None,
            belief_pd_mode=env_cfg.get("belief_pd_mode", "exact"))
    else:
        env = SonarLevel1VecEnv(
            num_envs=min(batch_size, len(scenarios)), env_cfg=env_cfg, t_max=t_max,
            canonical_cells=canonical_cells, master_seed=seed, obs_mode=obs_mode,
            record_history=bool(entry.get("record_history", False)),
            difficulty_overrides=condition_params or None,
            belief_pd_mode=env_cfg.get("belief_pd_mode", "exact"))
    if algorithm in RL_ALGOS:
        wrapper = AgentActWrapper(agent, algorithm, training=False)
    else:
        wrapper = BaselineActWrapper(agent)
    mech_out: Dict = {}
    mech = mechanism_indices or set()
    outcomes, mech_out = run_evaluation(
        wrapper, env, scenarios, mechanism_indices=mech,
        tag={"algorithm": algorithm, "variant": variant, "seed": seed,
             "condition": condition, "stage": stage, "level": level})
    # 原始数据落盘
    suffix = entry.get("suffix", "")
    fname = f"{algorithm}_{variant}_seed{seed}" + (f"_{suffix}" if suffix else "") + ".jsonl"
    raw_path = results_root / "raw" / stage / condition / fname
    with JsonlWriter(raw_path) as w:
        for rec in outcomes:
            rec2 = {k: v for k, v in rec.items() if k != "history"}
            rec2["history_present"] = rec.get("history") is not None
            w.write(rec2)
    # 机理记录落盘
    if mech and mech_out:
        mech_path = results_root / "raw" / "mechanism" / \
            f"{algorithm}_{variant}_seed{seed}_{stage}_{condition}.json"
        write_json(mech_path, {str(k): v for k, v in mech_out.items()})
    # history单独落盘（先验偏差/案例研究用）
    hist_recs = [rec for rec in outcomes if rec.get("history")]
    if hist_recs:
        hist_path = results_root / "raw" / "histories" / stage / condition / \
            f"{algorithm}_{variant}_seed{seed}_history.json"
        write_json(hist_path, [
            {"scenario_index": r["scenario_index"], "target": r["target"],
             "n_sectors": r["n_sectors"], "success": r["success"], "steps": r["steps"],
             "pd_ep": r["pd_ep"], "pfa": r["pfa"], "final_belief": r["final_belief"],
             "difficulty": r.get("difficulty"), "history": r["history"],
             "prior_type": r.get("prior_type", None)}
            for r in hist_recs])
    # 聚合指标（式31-35）
    c_pen = float(eval_cfg.get("metrics", {}).get("edc_failure_penalty", 10.0))
    m = mt.aggregate_metrics(outcomes, c_pen, t_max)
    row = {
        "stage": stage, "condition": condition, "algorithm": algorithm,
        "variant": variant, "seed": seed, "level": level,
        **{k: v for k, v in m.items()},
        "n_episodes": len(outcomes),
    }
    if entry.get("eval_options"):
        row["eval_options"] = json.dumps(entry["eval_options"])
    return row
