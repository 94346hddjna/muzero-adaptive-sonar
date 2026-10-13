# -*- coding: utf-8 -*-
"""DQN基线（开题报告式(28)、第7章基线4）。

输入=与MuZero一致的观测向量（式15），输出各动作Q值（规范格C logits→扇区Q=格均值）。
训练：经验回放+目标网络+ε-greedy；损失式(28)：
  L=E[(r_t + γ·max_a' Q^−(s_{t+1},a') − Q(s_t,a_t))²]
"""
import random
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from env.level1_env import sector_center_cell


class QNet(nn.Module):
    """MLP Q网络：obs→C格Q值。"""

    def __init__(self, obs_dim: int, canonical_cells: int = 192,
                 hidden_dims=(256, 256)):
        super().__init__()
        layers = []
        d = obs_dim
        for h in hidden_dims:
            layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        layers += [nn.Linear(d, canonical_cells)]
        self.net = nn.Sequential(*layers)
        self.C = canonical_cells

    def forward(self, obs):
        return self.net(obs)


class DQNAgent:
    def __init__(self, dqn_cfg: Dict, obs_dim: int, canonical_cells: int = 192,
                 device: str = "cpu", seed: int = 0, obs_mode: str = "belief",
                 t_max: int = 20):
        self.cfg = dqn_cfg
        self.C = canonical_cells
        self.device = device
        self.obs_mode = obs_mode
        self.t_max = t_max
        self.rng = np.random.default_rng(seed)
        self.q = QNet(obs_dim, canonical_cells,
                      tuple(dqn_cfg.get("hidden_dims", [256, 256]))).to(device)
        self.q_target = QNet(obs_dim, canonical_cells,
                             tuple(dqn_cfg.get("hidden_dims", [256, 256]))).to(device)
        self.q_target.load_state_dict(self.q.state_dict())
        self.optimizer = torch.optim.AdamW(
            self.q.parameters(), lr=dqn_cfg.get("learning_rate", 3e-4),
            weight_decay=dqn_cfg.get("weight_decay", 1e-4))
        self.buffer = deque(maxlen=int(dqn_cfg.get("buffer_size", 200000)))
        self.gamma = dqn_cfg.get("gamma", 0.99)
        self.batch = int(dqn_cfg.get("batch_size", 128))
        self.eps_start = dqn_cfg.get("epsilon_start", 1.0)
        self.eps_final = dqn_cfg.get("epsilon_final", 0.05)
        self.eps_decay = int(dqn_cfg.get("epsilon_decay_steps", 200000))
        self.target_sync = int(dqn_cfg.get("target_update_steps", 2000))
        self.train_every = int(dqn_cfg.get("train_every_steps", 4))
        self.warmup = int(dqn_cfg.get("warmup_steps", 5000))
        self.max_grad_norm = dqn_cfg.get("max_grad_norm", 5.0)
        self.total_steps = 0
        self.train_updates = 0
        self._sync_counter = 0

    def epsilon(self) -> float:
        p = min(1.0, self.total_steps / max(1, self.eps_decay))
        return self.eps_start + (self.eps_final - self.eps_start) * p

    def _sector_q(self, cell_q: torch.Tensor, N: int) -> torch.Tensor:
        """C格Q→N扇区Q（格均值）。cell_q [..., C]。"""
        per = self.C // N
        return cell_q[..., :N * per].reshape(*cell_q.shape[:-1], N, per).mean(dim=-1)

    @torch.no_grad()
    def act(self, obs: np.ndarray, n_sectors: np.ndarray, env=None, training: bool = True
            ) -> Tuple[np.ndarray, List[Dict]]:
        B = obs.shape[0]
        q_cells = self.q(torch.as_tensor(obs, dtype=torch.float32, device=self.device))
        q_np = q_cells.cpu().numpy()
        actions = np.zeros(B, dtype=np.int64)
        eps = self.epsilon() if training else 0.0
        for e in range(B):
            N = int(n_sectors[e])
            qs = self._sector_q_np(q_np[e], N)
            if training and self.rng.random() < eps:
                actions[e] = self.rng.integers(N)
            else:
                actions[e] = int(np.argmax(qs))
        return actions, [{"q_sector": None}] * B

    def _sector_q_np(self, cell_q: np.ndarray, N: int) -> np.ndarray:
        per = self.C // N
        return cell_q[:N * per].reshape(N, per).mean(axis=-1)

    def store(self, obs, action, reward, next_obs, done, n_sectors,
              compact=None, compact_next=None):
        """raw_stack模式下存紧凑(acts,dets,t)历史（观测采样时重建，防内存爆炸）。
        compact=(acts_list, dets_list, t)。"""
        if self.obs_mode == "raw_stack" and compact is not None:
            self.buffer.append((compact, int(action), float(reward),
                                compact_next, bool(done), int(n_sectors)))
            return
        self.buffer.append((obs.astype(np.float32), int(action), float(reward),
                            next_obs.astype(np.float32), bool(done), int(n_sectors)))

    def update(self) -> Optional[Dict]:
        if len(self.buffer) < max(self.batch, self.warmup):
            return None
        idxs = [random.randrange(len(self.buffer)) for _ in range(self.batch)]
        batch = [self.buffer[i] for i in idxs]
        if self.obs_mode == "raw_stack" and isinstance(batch[0][0], tuple):
            # 紧凑历史→重建3861维观测
            from env.level1_env import build_raw_stack_observation_vector
            obs = np.stack([build_raw_stack_observation_vector(b[0][0], b[0][1], b[0][2],
                                                                self.t_max, self.C)
                            for b in batch])
            nxt = np.stack([build_raw_stack_observation_vector(b[3][0], b[3][1], b[3][2],
                                                               self.t_max, self.C)
                            for b in batch])
        else:
            obs = np.stack([b[0] for b in batch])
            nxt = np.stack([b[3] for b in batch])
        obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        nxt = torch.as_tensor(nxt, dtype=torch.float32, device=self.device)
        act = torch.as_tensor([b[1] for b in batch], device=self.device)
        rew = torch.as_tensor([b[2] for b in batch], dtype=torch.float32, device=self.device)
        dns = torch.as_tensor([b[4] for b in batch], dtype=torch.float32, device=self.device)
        nsec = np.array([b[5] for b in batch], dtype=np.int64)
        # Q(s,a)：格Q聚合到扇区（取所选扇区格均值）
        q_cells = self.q(obs)                                   # [B, C]
        q_sel = self._gather_sector_q(q_cells, act, nsec)       # [B]
        # 目标：r + γ·max_a' Q^−(s',a')（式28；终态无自举）
        with torch.no_grad():
            qn_cells = self.q_target(nxt)
            max_next = self._max_sector_q(qn_cells, nsec)       # [B]
            target = rew + self.gamma * (1.0 - dns) * max_next
        loss = F.mse_loss(q_sel, target)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q.parameters(), self.max_grad_norm)
        self.optimizer.step()
        self.train_updates += 1
        self._sync_counter += 1
        if self._sync_counter >= self.target_sync:
            self.q_target.load_state_dict(self.q.state_dict())
            self._sync_counter = 0
        return {"loss_q": float(loss.detach()), "epsilon": self.epsilon(),
                "train_updates": self.train_updates}

    def _gather_sector_q(self, q_cells: torch.Tensor, act: torch.Tensor,
                         nsec: np.ndarray) -> torch.Tensor:
        B = q_cells.shape[0]
        out = torch.zeros(B, device=self.device)
        for N in np.unique(nsec):
            sel = np.where(nsec == N)[0]
            per = self.C // int(N)
            qs = q_cells[sel][:, :int(N) * per].view(len(sel), int(N), per).mean(-1)
            out[sel] = qs.gather(1, act[sel].unsqueeze(1)).squeeze(1)
        return out

    def _max_sector_q(self, q_cells: torch.Tensor, nsec: np.ndarray) -> torch.Tensor:
        B = q_cells.shape[0]
        out = torch.zeros(B, device=self.device)
        for N in np.unique(nsec):
            sel = np.where(nsec == N)[0]
            per = self.C // int(N)
            qs = q_cells[sel][:, :int(N) * per].view(len(sel), int(N), per).mean(-1)
            out[sel] = qs.max(dim=1).values
        return out

    def save_checkpoint(self, path: Path, extra: Optional[Dict] = None):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"q": self.q.state_dict(), "q_target": self.q_target.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "total_steps": self.total_steps,
                    "train_updates": self.train_updates, "extra": extra or {}}, path)

    def load_checkpoint(self, path: Path, load_optimizer: bool = True) -> Dict:
        ck = torch.load(Path(path), map_location=self.device, weights_only=False)
        self.q.load_state_dict(ck["q"])
        self.q_target.load_state_dict(ck["q_target"])
        if load_optimizer and "optimizer" in ck:
            self.optimizer.load_state_dict(ck["optimizer"])
        self.total_steps = ck.get("total_steps", 0)
        self.train_updates = ck.get("train_updates", 0)
        return ck.get("extra", {})
