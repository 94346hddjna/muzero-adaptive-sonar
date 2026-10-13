# -*- coding: utf-8 -*-
"""基线2：随机扫描Random（开题报告第7章）。
每步均匀随机选择扇区（用于验证学习策略确实学到有效搜索行为）。"""
import numpy as np


class RandomPolicy:
    name = "random"

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def act(self, obs: np.ndarray, n_sectors: np.ndarray, env=None,
            training: bool = True):
        B = obs.shape[0]
        actions = np.zeros(B, dtype=np.int64)
        for e in range(B):
            N = int(n_sectors[e])
            actions[e] = self.rng.integers(N)
        return actions, [{"policy": None}] * B
