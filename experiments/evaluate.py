# -*- coding: utf-8 -*-
"""统一评测runner（开题报告第8章、第十一章公平比较协议）。

配对评测：同一固定场景列表（目标位置/环境参数/先验/每回合RNG种子）供所有算法使用；
逐回合记录全部字段→results/raw/；统计检验按scenario_index配对。

决策时延：仅计量策略计算（agent.act，含MCTS），不含环境步进。
机理捕获：指定场景索引集合时记录每步MCTS访问分布/原始策略/改进策略/信念（图18-20、
典型案例可视化数据来源）。
"""
import time
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np


class AgentActWrapper:
    """统一act接口包装（含MuZero的GRU状态重置钩子）。"""

    def __init__(self, agent, kind: str, training: bool = False,
                 temperature: Optional[float] = None):
        self.agent = agent
        self.kind = kind
        self.training = training
        self.temperature = temperature
        self.last_act_infos: List[Dict] = []

    def act(self, obs: np.ndarray, n_sectors: np.ndarray, env):
        if self.kind == "muzero":
            actions, infos = self.agent.act(
                obs, n_sectors, env=env, training=self.training,
                temperature=self.temperature)
        else:
            actions, infos = self.agent.act(obs, n_sectors, env=env, training=self.training)
        self.last_act_infos = infos
        return actions, infos

    def post_step(self, obs, rewards, dones, ep_infos):
        if self.kind == "muzero":
            self.agent.reset_slots(dones)


class BaselineActWrapper:
    def __init__(self, policy):
        self.policy = policy
        self.last_act_infos: List[Dict] = []

    def act(self, obs, n_sectors, env):
        actions, infos = self.policy.act(obs, n_sectors, env=env, training=False)
        self.last_act_infos = infos
        return actions, infos

    def post_step(self, obs, rewards, dones, ep_infos):
        pass


def run_evaluation(
    wrapper,
    env,
    scenarios: List[Dict],
    max_vector_steps: Optional[int] = None,
    mechanism_indices: Optional[set] = None,
    tag: Optional[Dict] = None,
) -> Tuple[List[Dict], Dict]:
    """在固定场景上评测。返回(逐回合outcomes, 机理记录{scenario_index: [step记录]})。"""
    B = env.B
    obs = env.reset(scenarios)
    n_scenarios = len(scenarios)
    outcomes: List[Dict] = []
    mech: Dict[int, List[Dict]] = {}
    cap_set = mechanism_indices or set()
    # 时延归账
    act_time_cum = np.zeros(B)
    steps_cum = np.zeros(B, dtype=np.int64)
    vsteps = 0
    max_vsteps = max_vector_steps or (n_scenarios * env.t_max + 1000)
    while len(outcomes) < n_scenarios and vsteps < max_vsteps:
        vsteps += 1
        n_active = int(env.active.sum())
        if n_active == 0:
            break
        t0 = time.perf_counter()
        actions, act_infos = wrapper.act(obs, env.n, env)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        obs, rewards, dones, ep_infos = env.step(actions)
        # 时延归账（按活跃槽位均摊）
        if n_active > 0:
            act_time_cum[:B] += dt_ms / n_active
        steps_cum += 1
        # 机理捕获（act阶段数据）
        pending_mech: Dict[int, Dict] = {}
        if cap_set:
            for slot in range(B):
                sidx = int(env.scenario_index[slot])
                if env.active[slot] and sidx in cap_set and act_infos:
                    info = act_infos[slot]
                    pending_mech[slot] = {
                        "scenario_index": sidx, "t": int(env.t[slot]),
                        "action": int(actions[slot]),
                        "visits": info.get("visits"),
                        "raw_policy": info.get("raw_policy"),
                        "policy_target": info.get("policy_target"),
                        "root_value": info.get("root_value"),
                    }
        wrapper.post_step(obs, rewards, dones, ep_infos)
        # 收集done回合 + 补充机理的观测/奖励
        for slot in range(B):
            if pending_mech and slot in pending_mech:
                rec = pending_mech[slot]
                rec["detection"] = int(env.prev_det[slot])
                N = int(env.n[slot])
                rec["belief_after"] = env.belief[slot, :N].tolist()
                rec["reward"] = float(rewards[slot])
                mech.setdefault(int(env.scenario_index[slot]), []).append(rec)
            if ep_infos[slot] is not None:
                rec = dict(ep_infos[slot])
                if steps_cum[slot] > 0:
                    rec["mean_decision_latency_ms"] = act_time_cum[slot] / steps_cum[slot]
                rec["actions_list"] = None  # history模式下由env提供
                if tag:
                    rec.update(tag)
                outcomes.append(rec)
    if len(outcomes) < n_scenarios:
        raise RuntimeError(
            f"evaluation incomplete: {len(outcomes)}/{n_scenarios} episodes "
            f"(vsteps={vsteps}) — 检查环境/场景配置")
    outcomes.sort(key=lambda r: r["scenario_index"])
    return outcomes, mech


def summarize_outcomes(outcomes: List[Dict], c_pen: float, t_max: int) -> Dict:
    """聚合指标（utils/metrics.aggregate_metrics）。"""
    from utils import metrics as mt
    return mt.aggregate_metrics(outcomes, c_pen, t_max)
