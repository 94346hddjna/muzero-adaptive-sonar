# -*- coding: utf-8 -*-
"""POMCP：部分可观测蒙特卡洛规划（评审点名的POMDP专用规划基线）。

基于 Silver & Veness 2010 (POMCP: Monte-Carlo planning in large POMDPs)。
实现要点：
- 粒子滤波维护信念状态（每个向量槽位独立一组粒子，目标扇区假设）
- MCTS 在信念树上展开：根节点粒子从当前粒子集采样（静态状态→树内转移为恒等）
- PUCT 选择（与 MuZero 相同的选择准则与折扣γ=0.99，保证公平对比）
- 生成模型 = 真实观测模型（PD/PFA 已知）——"已知模型规划器"参照：
  它回答"给定完美模型，在线规划能做到多好"，与 MuZero 的"学习模型规划"对照。

奖励模型与环境式(16)完全一致：r = -λs - λe·cost + R_hit·1{det∧hit} - λf·1{det∧¬hit}
（R_hit=10, λs=1.0, λf=10, λe·cost=0.1 → 基线-1.1）。
"""
import math
from typing import Dict, List, Optional

import numpy as np


class _Node:
    __slots__ = ("visits", "value_sum", "children")

    def __init__(self):
        self.visits = 0
        self.value_sum = 0.0
        self.children: Dict[int, "_Node"] = {}


# 与 configs_v3/environment.yaml reward 节一致
_R_HIT, _LAM_S, _LAM_F, _STEP_COST = 10.0, 1.0, 10.0, 0.1
_BASE_R = -_LAM_S - _STEP_COST  # -1.1


class POMCPAgent:
    """POMCP 规划器（按向量槽位维护独立粒子滤波与搜索状态）。"""

    def __init__(self, n_particles: int = 200, n_simulations: int = 50,
                 depth: int = 5, gamma: float = 0.99, seed: int = 0):
        self.K = n_particles
        self.M = n_simulations
        self.D = depth
        self.gamma = gamma
        self.rng = np.random.default_rng(seed)
        self.particles: Optional[np.ndarray] = None  # [B, K] 目标扇区假设
        self.Ns: Optional[np.ndarray] = None         # [B]
        self.pd: Optional[np.ndarray] = None         # [B]
        self.pfa: Optional[np.ndarray] = None        # [B]

    # ---------- 粒子滤波 ----------
    def _reset_slot(self, e: int, n_sectors: int, prior: np.ndarray,
                    pd: float, pfa: float):
        """新回合开始：从场景先验信念采样粒子。"""
        self.Ns[e] = n_sectors
        self.pd[e] = pd
        self.pfa[e] = pfa
        p = np.asarray(prior[:n_sectors], dtype=np.float64)
        p = p / p.sum()
        self.particles[e] = self.rng.choice(n_sectors, size=self.K, p=p)

    def _observe_slot(self, e: int, action: int, obs: int):
        """真实观测到达后对 slot e 的粒子做重要性重采样（系统重采样近似）。"""
        pd, pfa = float(self.pd[e]), float(self.pfa[e])
        like = np.full(self.K, pfa if obs == 1 else 1.0 - pfa)
        hit = self.particles[e] == action
        like[hit] = pd if obs == 1 else 1.0 - pd
        w = like / like.sum()
        idx = self.rng.choice(self.K, size=self.K, p=w)   # 在粒子索引上重采样
        self.particles[e] = self.particles[e][idx]

    # ---------- MCTS ----------
    def _simulate(self, node: _Node, particle: int, e: int, depth: int) -> float:
        """MCTS四阶段。静态状态→粒子沿路径不变；成功检测即回合终止。"""
        if depth >= self.D:
            return 0.0
        node.visits += 1
        N = int(self.Ns[e])
        pd, pfa = float(self.pd[e]), float(self.pfa[e])
        log_v = math.log(max(1, node.visits))
        best_a, best_score = 0, -1e18
        for a in range(N):
            child = node.children.get(a)
            if child is None or child.visits == 0:
                q = 0.0
                u = 1.25 * math.sqrt(log_v)          # 未访问动作给最大探索奖金
            else:
                q = child.value_sum / child.visits
                u = 1.25 * math.sqrt(log_v / child.visits)
            score = q + u
            if score > best_score:
                best_score, best_a = score, a
        child = node.children.get(best_a)
        if child is None:
            child = _Node()
            node.children[best_a] = child
        # 真实生成模型：式(13)(14)
        hit = particle == best_a
        det = 1 if (self.rng.random() < (pd if hit else pfa)) else 0
        r = _BASE_R
        if det:
            r += _R_HIT if hit else -_LAM_F
        if det and hit:
            v = r                                    # 成功→回合终止
        else:
            v = r + self.gamma * self._simulate(child, particle, e, depth + 1)
        child.visits += 1
        child.value_sum += v
        return v

    def _search(self, e: int) -> np.ndarray:
        """从 slot e 的当前粒子集做 M 次模拟，返回各动作访问数。"""
        N = int(self.Ns[e])
        root = _Node()
        for _ in range(self.M):
            particle = int(self.particles[e][self.rng.integers(self.K)])
            self._simulate(root, particle, e, depth=0)
        visits = np.zeros(N)
        for a, child in root.children.items():
            visits[a] = child.visits
        return visits

    # ---------- 统一接口 ----------
    def act(self, obs, n_sectors, env=None, training=False, temperature=0.0):
        """obs 不用（POMCP 维护内部粒子信念）；env 仅用于读取回合边界与真实观测
        （观测是POMDP定义下智能体合法可得的信息）。"""
        B = obs.shape[0]
        actions = np.zeros(B, dtype=np.int64)
        if self.particles is None or self.particles.shape[0] != B:
            self.particles = np.zeros((B, self.K), dtype=np.int64)
            self.Ns = np.zeros(B, dtype=np.int64)
            self.pd = np.zeros(B)
            self.pfa = np.zeros(B)
            for e in range(B):
                N = int(n_sectors[e])
                prior = env.belief[e, :N] if env is not None else np.full(N, 1 / N)
                pd = float(env.pd_ep[e]) if env is not None else 0.5
                pfa = float(env.pfa[e]) if env is not None else 0.01
                self._reset_slot(e, N, prior, pd, pfa)
        for e in range(B):
            N = int(n_sectors[e])
            if env is None:
                actions[e] = self.rng.integers(N)
                continue
            if env.t[e] == 0:
                # 槽位载入新回合（含首回合与中途换场景）：重置粒子
                self._reset_slot(e, N, env.belief[e, :N],
                                 float(env.pd_ep[e]), float(env.pfa[e]))
            elif env.prev_action[e] >= 0:
                # 上一步的真实观测到达：粒子滤波更新
                self._observe_slot(e, int(env.prev_action[e]), int(env.prev_det[e]))
            visits = self._search(e)
            if visits.sum() > 0:
                if training:
                    actions[e] = self.rng.choice(N, p=visits / visits.sum())
                else:
                    actions[e] = int(np.argmax(visits))
            else:
                # 兜底：粒子信念众数
                actions[e] = int(np.argmax(np.bincount(self.particles[e],
                                                       minlength=N)))
        return actions, [{"policy": None}] * B
