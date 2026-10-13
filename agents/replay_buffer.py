# -*- coding: utf-8 -*-
"""回合式回放缓冲区（开题报告算法1第13行、表7|D|=5e4回合）。

存储episode记录（紧凑格式：信念轨迹而非完整观测向量，节省内存；
训练时由build_observation_vector重建式(15)观测向量）：
  n_sectors/pfa/pd_ep、beliefs[T+1,N]、actions[T]、dets[T]、rewards[T]、
  pi_targets[T,N]（式26 τ=1）、root_values[T]（MCTS根价值）、z[T]/u[T]（式20/21）
"""
import random
from collections import deque
from typing import Dict, List, Optional

import numpy as np


class EpisodeBuffer:
    """FIFO回合缓冲。"""

    def __init__(self, capacity: int, seed: int = 0):
        self.capacity = int(capacity)
        self._buf: deque = deque(maxlen=capacity)
        self._rng = random.Random(seed)

    def __len__(self):
        return len(self._buf)

    def add(self, episode: Dict):
        self._buf.append(episode)

    def sample_segments(self, batch: int) -> List[Dict]:
        """随机采样(batch, t0)片段：t0 ∈ [0, T-1]均匀。"""
        out = []
        n_ep = len(self._buf)
        if n_ep == 0:
            return out
        for _ in range(batch):
            ep = self._buf[self._rng.randrange(n_ep)]
            T = len(ep["rewards"])
            if T <= 0:
                continue
            t0 = self._rng.randrange(T)
            out.append({"episode": ep, "t0": t0})
        return out

    def sample_episodes(self, n: int) -> List[Dict]:
        n = min(n, len(self._buf))
        idxs = self._rng.sample(range(len(self._buf)), n)
        return [self._buf[i] for i in idxs]

    def stats(self) -> Dict:
        if not self._buf:
            return {"episodes": 0}
        rets = [float(np.sum(ep["rewards"])) for ep in self._buf]
        steps = [len(ep["rewards"]) for ep in self._buf]
        return {
            "episodes": len(self._buf),
            "mean_return": float(np.mean(rets)),
            "mean_steps": float(np.mean(steps)),
        }
