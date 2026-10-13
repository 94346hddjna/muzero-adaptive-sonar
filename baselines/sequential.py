# -*- coding: utf-8 -*-
"""基线1：顺序扫描Sequential（开题报告第7章）。
按扇区编号固定顺序循环发射。"""
import numpy as np


class SequentialPolicy:
    name = "sequential"

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def act(self, obs: np.ndarray, n_sectors: np.ndarray, env=None,
            training: bool = True):
        """obs[B,D]，n_sectors[B]→actions[B]。顺序=当前步数t mod N。
        t/T_max分量位于obs[-1]（信念模式）/obs[-1]（raw模式一致）。"""
        B = obs.shape[0]
        actions = np.zeros(B, dtype=np.int64)
        for e in range(B):
            N = int(n_sectors[e])
            frac = float(obs[e][-1])          # t/T_max（两模式均在末位）
            t = int(round(frac * env.t_max)) if env is not None else int(frac * 20)
            actions[e] = t % N
        return actions, [{"policy": None}] * B
