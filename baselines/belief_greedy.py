# -*- coding: utf-8 -*-
"""基线3：信念贪心Belief-Greedy（开题报告式(27)）。
a_t = argmax_{i∈S} b_t(i)。显式利用信念，是与MuZero最直接的
"单步贪心 vs 多步规划"对照组。

信念来源：直接读取环境维护的信念地图（观测向量式(15)的第一段即信念的
规范格重采样；为数值精度直接使用env.belief）。
"""
import numpy as np


class BeliefGreedyPolicy:
    name = "belief_greedy"

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def act(self, obs: np.ndarray, n_sectors: np.ndarray, env=None,
            training: bool = True):
        B = obs.shape[0]
        actions = np.zeros(B, dtype=np.int64)
        for e in range(B):
            N = int(n_sectors[e])
            if env is not None and getattr(env, "obs_mode", "belief") == "belief":
                b = env.belief[e, :N]
            else:
                # 从观测向量重建：前C格为信念的规范格重采样→按块求和
                C = obs.shape[1] - 3
                per = C // N
                bc = obs[e, :C].reshape(N, per).sum(axis=-1)
                b = bc
            actions[e] = int(np.argmax(b))     # 式(27)
        return actions, [{"policy": None}] * B
