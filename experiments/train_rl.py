# -*- coding: utf-8 -*-
"""RL训练器（开题报告算法1、第35章模型选择、第28章断点续训）。

三算法（muzero/dqn/ppo）统一训练管线：
  域随机化训练环境 → 算法特定交互/存储/更新 → 周期Validation评估（选best_val
  checkpoint，绝不用Test调参） → last/best checkpoint + 训练日志CSV + resume。

消融变体（variant，表6）：
  wo_belief: env obs_mode='raw'（表6-D，MuZero）
  wo_memory: MuZero use_memory=False（表6-E，MuZero）
  wo_domain_rand: 固定Medium训练（表6-F）
  shaped: 信念熵PBRS塑形奖励（表6-G）
"""
import time
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from agents.dqn import DQNAgent
from agents.muzero import MuZeroAgent, temperature_at
from agents.ppo import PPOAgent
from env import level1_env, scenarios as scen
from env.level1_env import SonarLevel1VecEnv, obs_dim
from experiments.evaluate import AgentActWrapper, run_evaluation
from utils import seeding as us
from utils.io_utils import (done_marker, is_done, read_json, write_csv, write_json)

VARIANTS = ["main", "wo_belief", "wo_memory", "wo_domain_rand", "shaped"]


def build_training_env(env_cfg, train_cfg, algorithm: str, variant: str, master_seed: int,
                       num_envs: int, t_max: int, canonical_cells: int) -> SonarLevel1VecEnv:
    domain_rand = env_cfg.get("domain_randomization", {})
    fixed_difficulty = None
    reward_mode = "sparse"
    obs_mode = "belief"
    if variant == "wo_domain_rand":
        domain_rand = {**domain_rand, "enabled": False}
        fixed_difficulty = "medium"
    if variant == "shaped":
        reward_mode = "shaped"
    obs_mode = _algo_obs_mode(env_cfg, algorithm, variant)
    return SonarLevel1VecEnv(
        num_envs=num_envs, env_cfg=env_cfg, t_max=t_max,
        canonical_cells=canonical_cells, domain_rand_cfg=domain_rand,
        master_seed=master_seed, reward_mode=reward_mode, obs_mode=obs_mode,
        belief_pd_mode=env_cfg.get("belief_pd_mode", "exact"))


def _algo_obs_mode(env_cfg: Dict, algorithm: str, variant: str = "main") -> str:
    """v3按算法分配观测模式：muzero=raw（GRU编码历史）、dqn/ppo=raw_stack
    （完整历史平铺，信息等价）、baselines=belief（贪心读env信念）。
    v1/v2配置无obs_mode_by_algo时默认belief（原行为）。"""
    if variant == "wo_belief":
        return "raw"
    return env_cfg.get("obs_mode_by_algo", {}).get(algorithm, "belief")


def validation_run(agent, kind: str, env_cfg, train_cfg, canonical_cells: int,
                   t_max: int, seed: int, n_episodes: int, results_dir: Path,
                   obs_mode: str = "belief") -> Dict:
    """Validation评估（固定验证场景，选模专用，不触碰Test）。"""
    scs = scen.get_or_create_scenarios(
        env_cfg, "medium", n_episodes, int(train_cfg["validation_scenario_seed"]),
        "validation", results_dir)
    env = SonarLevel1VecEnv(
        num_envs=min(16, len(scs)), env_cfg=env_cfg, t_max=t_max,
        canonical_cells=canonical_cells, master_seed=seed, obs_mode=obs_mode,
        belief_pd_mode=env_cfg.get("belief_pd_mode", "exact"))
    wrapper = AgentActWrapper(agent, kind, training=False)
    outcomes, _ = run_evaluation(wrapper, env, scs)
    from utils import metrics as mt
    return mt.aggregate_metrics(outcomes, c_pen=10.0, t_max=t_max)


def _find_results_root(run_dir: Path) -> Path:
    for p in [run_dir] + list(run_dir.parents):
        if p.name == "results":
            return p
    return run_dir.parent.parent


def train_rl(
    algorithm: str,
    env_cfg: Dict,
    alg_cfg: Dict,
    train_cfg: Dict,
    seed: int,
    run_dir: Path,
    variant: str = "main",
    smoke: bool = False,
    resume: bool = True,
    device: str = "auto",
) -> Dict:
    """训练单个(algorithm, seed, variant)。返回summary（含best_val路径）。"""
    assert algorithm in ("muzero", "dqn", "ppo"), algorithm
    assert variant in VARIANTS, variant
    t_max = 20
    canonical_cells = int(env_cfg.get("canonical_cells", 192))
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = run_dir  # run_dir已位于results/checkpoints/<algo>/<variant>_seed<k>/
    exp_id = f"{algorithm}_{variant}_seed{seed}"

    if resume and is_done(run_dir, exp_id):
        summary_path = run_dir / f"{exp_id}_summary.json"
        if summary_path.exists():
            return read_json(summary_path)
        return {"skipped": True, "exp_id": exp_id}

    results_root = _find_results_root(run_dir)
    progress_path = run_dir / f"{exp_id}_progress.json"
    log_path = run_dir / f"{exp_id}_training_log.csv"
    start_step = 0
    resume_count = 0
    best_val_sr = -1.0
    if resume and progress_path.exists():
        prog = read_json(progress_path)
        start_step = int(prog.get("env_steps", 0))
        resume_count = int(prog.get("resume_count", 0)) + 1
        best_val_sr = float(prog.get("best_val_sr", -1.0))

    total_steps = int(train_cfg["total_env_steps"])
    num_envs = int(train_cfg["num_parallel_envs"])
    obs_mode = _algo_obs_mode(env_cfg, algorithm, variant)
    D_obs = obs_dim(canonical_cells, obs_mode)
    us.set_global_seeds(seed + resume_count * 10000)

    env = build_training_env(
        env_cfg, train_cfg, algorithm, variant,
        master_seed=seed * 1000 + 17 + resume_count * 101, num_envs=num_envs,
        t_max=t_max, canonical_cells=canonical_cells)
    if algorithm == "muzero":
        agent = MuZeroAgent(alg_cfg, D_obs, canonical_cells, device=device,
                            obs_mode=obs_mode,
                            use_memory=(variant != "wo_memory"), seed=seed)
    elif algorithm == "dqn":
        agent = DQNAgent(alg_cfg, D_obs, canonical_cells, device=device, seed=seed,
                         obs_mode=obs_mode)
    else:
        agent = PPOAgent(alg_cfg, D_obs, canonical_cells, device=device, seed=seed)
    last_ckpt = ckpt_dir / f"{exp_id}_last.ckpt"
    if resume and last_ckpt.exists() and start_step > 0:
        agent.load_checkpoint(last_ckpt)
        agent.total_steps = start_step
    else:
        agent.total_steps = start_step

    # 训练日志（resume续写）
    log_rows: List[Dict] = []
    if log_path.exists() and start_step > 0:
        try:
            import csv as _csv
            with open(log_path, "r", encoding="utf-8") as f:
                log_rows = list(_csv.DictReader(f))
            log_rows = [{k: (float(v) if v not in ("", None) and k != "resume_count" else v)
                         for k, v in row.items()} for row in log_rows]
        except Exception:
            log_rows = []
    recent = deque(maxlen=200)
    sample_eff_curve: List[Dict] = []
    t_start = time.time()
    val_interval = max(1, int(train_cfg.get("validation_interval_steps", 25000)))
    ckpt_interval = max(1, int(train_cfg.get("checkpoint_interval_steps", 50000)))
    val_episodes = int(train_cfg.get("validation_episodes", 200))

    from agents.replay_buffer import EpisodeBuffer
    buffer = EpisodeBuffer(int(alg_cfg.get("buffer_capacity", 50000)), seed=seed) \
        if algorithm == "muzero" else None
    accum = _init_accum(num_envs) if algorithm == "muzero" else None
    obs = env.reset()
    if algorithm == "muzero":
        _capture_b0(env, accum)

    train_every = max(1, int(alg_cfg.get("train_every_steps", 32)))
    min_buffer_eps = int(alg_cfg.get("min_buffer_episodes", 32))
    last_loss_stats: Dict = {}
    log_every = max(100, val_interval // 5)
    last_val_step = start_step
    last_ckpt_step = start_step
    last_log_step = log_rows[-1]["env_steps"] if log_rows else -1

    while agent.total_steps < total_steps:
        progress = agent.total_steps / total_steps
        n_before = env.n.copy()
        # -- 交互（式17-26；ε-greedy/PPO采样） --
        if algorithm == "muzero":
            tau = temperature_at(progress, alg_cfg.get("temperature_initial", 1.0),
                                 alg_cfg.get("temperature_final", 0.25))
            actions, infos = agent.act(obs, n_before, env=env, training=True, temperature=tau)
        else:
            actions, infos = agent.act(obs, n_before, env=env, training=True)
        obs_next, rewards, dones, ep_infos = env.step(actions)
        agent.total_steps += num_envs
        steps = agent.total_steps
        # -- 算法特定存储与更新 --
        if algorithm == "muzero":
            _accumulate_muzero(env, accum, actions, rewards, ep_infos, infos)
            _finalize_done_episodes(env, accum, ep_infos, agent, buffer)
            agent.reset_slots(dones)
            if len(buffer) >= min_buffer_eps and steps % train_every < num_envs:
                st = agent.update(buffer)
                if st:
                    last_loss_stats = st
        elif algorithm == "dqn":
            for slot in range(num_envs):
                compact = compact_next = None
                if obs_mode == "raw_stack":
                    compact = (list(env.acts_hist[slot][:-1]), list(env.dets_hist[slot][:-1]),
                               max(env.t[slot] - 1, 0))
                    compact_next = (list(env.acts_hist[slot]), list(env.dets_hist[slot]),
                                    env.t[slot])
                agent.store(obs[slot], int(actions[slot]), float(rewards[slot]),
                            obs_next[slot], bool(dones[slot]), int(n_before[slot]),
                            compact=compact, compact_next=compact_next)
            if steps % agent.train_every < num_envs:
                st = agent.update()
                if st:
                    last_loss_stats = st
        else:  # ppo
            nv = agent.value_only(obs_next)
            for slot in range(num_envs):
                agent.store_transition(
                    obs[slot], int(actions[slot]), float(rewards[slot]), bool(dones[slot]),
                    float(infos[slot]["logp"]), float(infos[slot]["value"]),
                    int(n_before[slot]), slot, float(nv[slot]))
            if agent.ready_for_update():
                st = agent.update()
                if st:
                    last_loss_stats = st
        for slot in range(num_envs):
            if ep_infos[slot] is not None:
                recent.append({"success": ep_infos[slot]["success"],
                               "steps": ep_infos[slot]["steps"],
                               "ret": ep_infos[slot]["return"]})
        obs = obs_next
        # -- 日志 --
        if recent and (steps - last_log_step >= log_every):
            row = {
                "env_steps": steps,
                "episode_return": round(float(np.mean([r["ret"] for r in recent])), 4),
                "success_rate": round(float(np.mean([r["success"] for r in recent])), 4),
                "avg_steps": round(float(np.mean([r["steps"] for r in recent])), 4),
                "elapsed_sec": round(time.time() - t_start, 1),
                "resume_count": resume_count,
            }
            for k, v in (last_loss_stats or {}).items():
                row[f"train_{k}"] = v
            log_rows.append(row)
            last_log_step = steps
        # -- Validation选模（第35章：仅凭Validation指标；绝不用Test） --
        if steps - last_val_step >= val_interval:
            val_metrics = validation_run(
                agent, algorithm, env_cfg, train_cfg, canonical_cells, t_max,
                seed=seed, n_episodes=val_episodes, results_dir=results_root,
                obs_mode=obs_mode)
            sample_eff_curve.append({"env_steps": steps,
                                     "success_rate": val_metrics["SR"],
                                     "ADS": val_metrics["ADS"]})
            last_val_step = steps
            if val_metrics["SR"] > best_val_sr:
                best_val_sr = val_metrics["SR"]
                agent.save_checkpoint(ckpt_dir / f"{exp_id}_best_val.ckpt",
                                      extra={"val_SR": best_val_sr, "env_steps": steps})
        # -- checkpoint --
        if steps - last_ckpt_step >= ckpt_interval:
            agent.save_checkpoint(last_ckpt, extra={"env_steps": steps})
            write_json(progress_path, {
                "env_steps": steps, "resume_count": resume_count,
                "best_val_sr": best_val_sr, "exp_id": exp_id})
            last_ckpt_step = steps

    # 收尾
    agent.save_checkpoint(last_ckpt, extra={"env_steps": total_steps})
    agent.save_checkpoint(ckpt_dir / f"{exp_id}_final.ckpt", extra={"env_steps": total_steps})
    if not sample_eff_curve:
        val_metrics = validation_run(
            agent, algorithm, env_cfg, train_cfg, canonical_cells, t_max,
            seed=seed, n_episodes=val_episodes, results_dir=results_root,
            obs_mode=obs_mode)
        sample_eff_curve.append({"env_steps": total_steps,
                                 "success_rate": val_metrics["SR"],
                                 "ADS": val_metrics["ADS"]})
        if val_metrics["SR"] > best_val_sr:
            best_val_sr = val_metrics["SR"]
            agent.save_checkpoint(ckpt_dir / f"{exp_id}_best_val.ckpt",
                                  extra={"val_SR": best_val_sr, "env_steps": total_steps})
    write_csv(log_path, log_rows)
    elapsed = time.time() - t_start
    summary = {
        "exp_id": exp_id, "algorithm": algorithm, "variant": variant, "seed": seed,
        "env_steps": total_steps, "best_val_success_rate": best_val_sr,
        "training_wall_clock_sec": round(elapsed, 1),
        "train_updates": getattr(agent, "train_updates", 0),
        "sample_efficiency_curve": sample_eff_curve,
        "last_log": log_rows[-1] if log_rows else None,
        "checkpoints": {
            "last": str(last_ckpt),
            "best_val": str(ckpt_dir / f"{exp_id}_best_val.ckpt"),
            "final": str(ckpt_dir / f"{exp_id}_final.ckpt"),
        },
    }
    write_json(run_dir / f"{exp_id}_summary.json", summary)
    done_marker(run_dir, exp_id, {"env_steps": total_steps})
    return summary


# ---------------------------------------------------------------------------
# MuZero回合累积器（算法1第3-13行）
# ---------------------------------------------------------------------------
def _init_accum(B: int) -> Dict:
    return {"beliefs": [[] for _ in range(B)], "actions": [[] for _ in range(B)],
            "dets": [[] for _ in range(B)], "rewards": [[] for _ in range(B)],
            "pi": [[] for _ in range(B)], "rv": [[] for _ in range(B)]}


def _capture_b0(env, accum):
    for slot in range(env.B):
        if env.active[slot]:
            N = int(env.n[slot])
            accum["beliefs"][slot].append(env.belief[slot, :N].copy())


def _accumulate_muzero(env, accum, actions, rewards, ep_infos, act_infos):
    """每步累积（含done步：det取episode_info.last_detection，belief取final_belief）。"""
    for slot in range(env.B):
        info = ep_infos[slot]
        if info is not None:
            accum["beliefs"][slot].append(
                np.asarray(info["final_belief"], dtype=np.float32))
            accum["dets"][slot].append(int(info["last_detection"]))
        else:
            if not env.active[slot]:
                continue
            N = int(env.n[slot])
            accum["beliefs"][slot].append(env.belief[slot, :N].copy())
            accum["dets"][slot].append(int(env.prev_det[slot]))
        accum["actions"][slot].append(int(actions[slot]))
        accum["rewards"][slot].append(float(rewards[slot]))
        accum["pi"][slot].append(np.asarray(act_infos[slot]["policy_target"],
                                            dtype=np.float32))
        accum["rv"][slot].append(float(act_infos[slot]["root_value"]))


def _finalize_done_episodes(env, accum, ep_infos, agent, buffer):
    """回合结束：组装episode记录（式20/21目标）→回放缓冲；重置累积器。"""
    for slot in range(env.B):
        if ep_infos[slot] is None:
            continue
        info = ep_infos[slot]
        ep = {
            "n_sectors": int(info["n_sectors"]), "t_max": env.t_max,
            "beliefs": [np.asarray(b, dtype=np.float32) for b in accum["beliefs"][slot]],
            "actions": list(accum["actions"][slot]),
            "dets": list(accum["dets"][slot]),
            "rewards": list(accum["rewards"][slot]),
            "pi_targets": [np.asarray(p, dtype=np.float32) for p in accum["pi"][slot]],
            "root_values": list(accum["rv"][slot]),
        }
        T = len(ep["rewards"])
        assert len(ep["beliefs"]) == T + 1, (len(ep["beliefs"]), T)
        assert len(ep["pi_targets"]) == T and len(ep["actions"]) == T
        assert len(ep["root_values"]) == T
        ep = agent.compute_episode_targets(ep)
        # 紧凑序列化（ndarray→list）
        ep["beliefs"] = [b.tolist() for b in ep["beliefs"]]
        ep["pi_targets"] = [p.tolist() for p in ep["pi_targets"]]
        buffer.add(ep)
        # 新回合起点（env已重载新场景）
        N_new = int(env.n[slot])
        accum["beliefs"][slot] = [env.belief[slot, :N_new].copy()]
        accum["actions"][slot] = []
        accum["dets"][slot] = []
        accum["rewards"][slot] = []
        accum["pi"][slot] = []
        accum["rv"][slot] = []
