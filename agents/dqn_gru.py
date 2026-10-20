# -*- coding: utf-8 -*-
"""DQN+GRU：记忆增强DQN（评审点名的记忆增强基线）。

在v3盲搜索环境下，DQN只用平铺历史（raw_stack）。这里给DQN加GRU编码器：
  观测序列 o_1..o_t → GRU → Q值
Q网络：obs(2N+3) → GRU(256) → MLP(256,256) → N Q值。
与DQN的区别：有记忆（GRU），能利用历史。这是"DQN+记忆"的最强反应式基线。
"""
import random
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from env.level1_env import obs_dim, sector_center_cell


class GRUQNet(nn.Module):
    """GRU编码历史 → Q值。"""

    def __init__(self, obs_dim: int, canonical_cells: int = 192,
                 gru_hidden: int = 256, hidden_dims=(256, 256)):
        super().__init__()
        self.C = canonical_cells
        self.gru = nn.GRUCell(obs_dim, gru_hidden)
        trunk = []
        d = gru_hidden
        for h in hidden_dims:
            trunk += [nn.Linear(d, h), nn.ReLU()]
            d = h
        trunk.append(nn.Linear(d, canonical_cells))
        self.trunk = nn.Sequential(*trunk)

    def forward(self, obs_seq: torch.Tensor, z: torch.Tensor):
        """obs_seq [B, L, D], z [B, gru_hidden] → (Q_cells [B, C], z' [B, gru_hidden])"""
        B, L, D = obs_seq.shape
        for t in range(L):
            z = self.gru(obs_seq[:, t], z)
        return self.trunk(z), z


class DQNGRUAgent:
    """记忆增强DQN：GRU编码历史 + Q值。"""

    def __init__(self, dqn_cfg: Dict, obs_dim: int, canonical_cells: int = 192,
                 device: str = "cpu", seed: int = 0, obs_mode: str = "raw",
                 t_max: int = 20):
        self.cfg = dqn_cfg
        self.C = canonical_cells
        self.device = device
        self.rng = np.random.default_rng(seed)
        self.q = GRUQNet(obs_dim, canonical_cells).to(device)
        self.q_target = GRUQNet(obs_dim, canonical_cells).to(device)
        self.q_target.load_state_dict(self.q.state_dict())
        self.optimizer = torch.optim.AdamW(self.q.parameters(),
                                            lr=dqn_cfg.get("learning_rate", 3e-4),
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
        self._z: Optional[torch.Tensor] = None  # GRU状态
        self.obs_mode = obs_mode
        self.t_max = t_max

    def reset_slots(self, mask):
        if self._z is not None and np.any(mask):
            m = torch.as_tensor(mask, dtype=torch.bool, device=self.device)
            self._z[m] = 0.0

    def epsilon(self) -> float:
        p = min(1.0, self.total_steps / max(1, self.eps_decay))
        return self.eps_start + (self.eps_final - self.eps_start) * p

    def _sector_q(self, cell_q: torch.Tensor, N: int) -> torch.Tensor:
        per = self.C // N
        return cell_q[..., :N * per].reshape(*cell_q.shape[:-1], N, per).mean(dim=-1)

    @torch.no_grad()
    def act(self, obs: np.ndarray, n_sectors: np.ndarray, env=None, training: bool = True
            ) -> Tuple[np.ndarray, List[Dict]]:
        B = obs.shape[0]
        # raw模式：obs为每步194维（a_prev onehot, d_prev, t/T）
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        if self._z is None or self._z.shape[0] != B:
            self._z = torch.zeros(B, 256, device=self.device)
        # GRU编码当前观测（GRUCell直接返回新隐状态）
        self._z = self.q.gru(obs_t, self._z)
        self._z = self._z.detach()
        q_cells = self.q.trunk(self._z)  # [B, C]
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
        """GRU版DQN不需要compact（GRU有记忆），参数仅用于兼容DQN接口。"""
        self.buffer.append((obs.astype(np.float32), int(action), float(reward),
                            next_obs.astype(np.float32), bool(done), int(n_sectors)))

    def update(self) -> Optional[Dict]:
        if len(self.buffer) < max(self.batch, self.warmup):
            return None
        idxs = [random.randrange(len(self.buffer)) for _ in range(self.batch)]
        batch = [self.buffer[i] for i in idxs]
        obs = torch.as_tensor(np.stack([b[0] for b in batch]), device=self.device)
        act = torch.as_tensor([b[1] for b in batch], device=self.device)
        rew = torch.as_tensor([b[2] for b in batch], dtype=torch.float32, device=self.device)
        nxt = torch.as_tensor(np.stack([b[3] for b in batch]), device=self.device)
        dns = torch.as_tensor([b[4] for b in batch], dtype=torch.float32, device=self.device)
        nsec = np.array([b[5] for b in batch], dtype=np.int64)
        # GRU编码观测序列（当前帧）：q_cells [B, C]
        z0 = torch.zeros(len(batch), 256, device=self.device)
        q_latent = self.q.gru(obs, z0)
        q_cells = self.q.trunk(q_latent)
        q_sel = self._gather_sector_q(q_cells, act, nsec)
        with torch.no_grad():
            q_latent_next = self.q_target.gru(nxt, torch.zeros(len(batch), 256, device=self.device))
            qn_cells = self.q_target.trunk(q_latent_next)
            max_next = self._max_sector_q(qn_cells, nsec)
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

    def _gather_sector_q(self, q_cells, act, nsec):
        B = q_cells.shape[0]
        out = torch.zeros(B, device=self.device)
        for N in np.unique(nsec):
            sel = np.where(nsec == N)[0]
            per = self.C // int(N)
            qs = q_cells[sel][:, :int(N) * per].view(len(sel), int(N), per).mean(-1)
            out[sel] = qs.gather(1, act[sel].unsqueeze(1)).squeeze(1)
        return out

    def _max_sector_q(self, q_cells, nsec):
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
