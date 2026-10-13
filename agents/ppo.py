# -*- coding: utf-8 -*-
"""PPO基线（开题报告式(29)(30)、第7章基线5）。

输入=与MuZero一致的观测向量（式15）。策略头输出C格logits→扇区边缘化概率：
  π_sector(a) = Σ_{cells∈a} softmax(cell logits)
价值头标量。训练：GAE优势+截断目标式(30)：
  L^CLIP = E_t[min(r_t(θ)A_t, clip(r_t(θ),1−ε,1+ε)A_t)]，r_t(θ)=π_θ/π_θold（式29）
"""
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from env.level1_env import sector_center_cell


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, canonical_cells: int = 192, hidden_dims=(256, 256)):
        super().__init__()
        self.C = canonical_cells
        layers = []
        d = obs_dim
        for h in hidden_dims:
            layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        self.trunk = nn.Sequential(*layers)
        self.policy_head = nn.Linear(d, canonical_cells)
        self.value_head = nn.Linear(d, 1)

    def forward(self, obs):
        t = self.trunk(obs)
        return self.policy_head(t), self.value_head(t)


class PPOAgent:
    def __init__(self, ppo_cfg: Dict, obs_dim: int, canonical_cells: int = 192,
                 device: str = "cpu", seed: int = 0):
        self.cfg = ppo_cfg
        self.C = canonical_cells
        self.device = device
        self.rng = np.random.default_rng(seed)
        self.net = ActorCritic(obs_dim, canonical_cells,
                               tuple(ppo_cfg.get("hidden_dims", [256, 256]))).to(device)
        self.optimizer = torch.optim.AdamW(
            self.net.parameters(), lr=ppo_cfg.get("learning_rate", 3e-4),
            weight_decay=ppo_cfg.get("weight_decay", 1e-4))
        self.gamma = ppo_cfg.get("gamma", 0.99)
        self.lam = ppo_cfg.get("gae_lambda", 0.95)
        self.clip_eps = ppo_cfg.get("clip_epsilon", 0.2)
        self.entropy_coef = ppo_cfg.get("entropy_coef", 0.01)
        self.value_coef = ppo_cfg.get("value_coef", 0.5)
        self.epochs = int(ppo_cfg.get("epochs_per_update", 4))
        self.minibatch = int(ppo_cfg.get("minibatch_size", 256))
        self.rollout_steps = int(ppo_cfg.get("rollout_steps", 1280))
        self.max_grad_norm = ppo_cfg.get("max_grad_norm", 0.5)
        self.total_steps = 0
        self.train_updates = 0
        # rollout缓存（含slot与next_value，向量化GAE所需）
        self._obs: List[np.ndarray] = []
        self._act: List[int] = []
        self._rew: List[float] = []
        self._done: List[bool] = []
        self._logp: List[float] = []
        self._val: List[float] = []
        self._nsec: List[int] = []
        self._slot: List[int] = []
        self._nval: List[float] = []

    @torch.no_grad()
    def act(self, obs: np.ndarray, n_sectors: np.ndarray, env=None, training: bool = True
            ) -> Tuple[np.ndarray, List[Dict]]:
        B = obs.shape[0]
        logits, v = self.net(torch.as_tensor(obs, dtype=torch.float32, device=self.device))
        logp_cells = F.log_softmax(logits, dim=-1).cpu().numpy()
        v_np = v.squeeze(-1).cpu().numpy()
        actions = np.zeros(B, dtype=np.int64)
        logps = np.zeros(B)
        for e in range(B):
            N = int(n_sectors[e])
            per = self.C // N
            lp = logp_cells[e]
            # 扇区边缘概率与对数概率
            sec_lp = np.logaddexp.reduce(lp[:N * per].reshape(N, per), axis=-1)  # [N]
            probs = np.exp(sec_lp)
            s = probs.sum()
            if s > 0:
                probs = probs / s
            else:
                probs = np.full(N, 1.0 / N)
            if training:
                a = int(self.rng.choice(N, p=np.clip(probs, 0, None) /
                                        np.clip(probs, 0, None).sum()))
            else:
                a = int(np.argmax(probs))
            actions[e] = a
            logps[e] = sec_lp[a]
        infos = [{"logp": float(logps[e]), "value": float(v_np[e])} for e in range(B)]
        return actions, infos

    def store_transition(self, obs, action, reward, done, logp, value, n_sectors, slot,
                         next_value=0.0):
        """存单步转移（next_value=该步后观测的价值估计，向量化GAE自包含所需）。"""
        self._obs.append(obs)
        self._act.append(int(action))
        self._rew.append(float(reward))
        self._done.append(bool(done))
        self._logp.append(float(logp))
        self._val.append(float(value))
        self._nsec.append(int(n_sectors))
        self._slot.append(int(slot))
        self._nval.append(float(next_value))

    def ready_for_update(self) -> bool:
        return len(self._rew) >= self.rollout_steps

    @torch.no_grad()
    def value_only(self, obs: np.ndarray) -> np.ndarray:
        """批量计算观测价值（GAE尾部自举）。"""
        _, v = self.net(torch.as_tensor(obs, dtype=torch.float32, device=self.device))
        return v.squeeze(-1).cpu().numpy()

    def update(self) -> Optional[Dict]:
        """GAE（按slot链式回溯，done处截断）+截断目标式(30)。"""
        n = len(self._rew)
        if n < self.minibatch:
            return None
        # GAE：delta自包含（r+γ·next_value−V），λ链按slot回溯、done截断
        rew = np.asarray(self._rew)
        val = np.asarray(self._val)
        nval = np.asarray(self._nval)
        dns = np.asarray(self._done, dtype=bool)
        slots = np.asarray(self._slot)
        adv = np.zeros(n, dtype=np.float64)
        gae = {}
        for t in reversed(range(n)):
            s = int(slots[t])
            if dns[t]:
                gae[s] = 0.0
            delta = rew[t] + self.gamma * (0.0 if dns[t] else nval[t]) - val[t]
            gae[s] = delta + self.gamma * self.lam * (0.0 if dns[t] else gae.get(s, 0.0))
            adv[t] = gae[s]
        ret = adv + val
        # 标准化优势
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        obs_np = np.stack(self._obs)
        act_np = np.asarray(self._act, dtype=np.int64)
        nsec_np = np.asarray(self._nsec, dtype=np.int64)
        old_logp = np.asarray(self._logp, dtype=np.float64)
        adv_t = torch.as_tensor(adv, dtype=torch.float32, device=self.device)
        ret_t = torch.as_tensor(ret, dtype=torch.float32, device=self.device)
        obs_t = torch.as_tensor(obs_np, dtype=torch.float32, device=self.device)
        act_t = torch.as_tensor(act_np, device=self.device)
        nsec_t = nsec_np
        old_logp_t = torch.as_tensor(old_logp, dtype=torch.float32, device=self.device)
        # 多轮minibatch更新（式30）
        stats_acc = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "approx_kl": 0.0}
        n_updates = 0
        for _epoch in range(self.epochs):
            perm = np.random.permutation(n)
            for start in range(0, n - self.minibatch + 1, self.minibatch):
                mb = perm[start:start + self.minibatch]
                logits, v = self.net(obs_t[mb])
                logp_cells = F.log_softmax(logits, dim=-1)
                # 扇区log概率与所选动作logp（按N分组）
                new_logp = self._sector_logp_gather(logp_cells, act_t[mb], nsec_np[mb])
                ratio = torch.exp(new_logp - old_logp_t[mb])          # 式(29)
                a_mb = adv_t[mb]
                l_clip = torch.minimum(ratio * a_mb,
                                       torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * a_mb)
                l_pg = -l_clip.mean()
                l_v = F.mse_loss(v.squeeze(-1), ret_t[mb])
                # 扇区级熵
                ent = self._sector_entropy(logp_cells, nsec_np[mb])
                loss = l_pg + self.value_coef * l_v - self.entropy_coef * ent
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
                self.optimizer.step()
                self.train_updates += 1
                n_updates += 1
                with torch.no_grad():
                    stats_acc["policy_loss"] += float(l_pg)
                    stats_acc["value_loss"] += float(l_v)
                    stats_acc["entropy"] += float(ent)
                    stats_acc["approx_kl"] += float((new_logp - old_logp_t[mb]).mean())
        # 清空rollout缓存
        self._obs, self._act, self._rew, self._done = [], [], [], []
        self._logp, self._val, self._nsec = [], [], []
        self._slot, self._nval = [], []
        if n_updates == 0:
            return None
        return {k: v / n_updates for k, v in stats_acc.items()} | {
            "train_updates": self.train_updates, "rollout_size": n}

    def _sector_logp_gather(self, logp_cells: torch.Tensor, act: torch.Tensor,
                            nsec: np.ndarray) -> torch.Tensor:
        """扇区边缘化log π(a)。logp_cells [B, C]。"""
        B = logp_cells.shape[0]
        out = torch.zeros(B, device=self.device)
        for N in np.unique(nsec):
            sel = np.where(nsec == N)[0]
            N = int(N)
            per = self.C // N
            lp = logp_cells[sel][:, :N * per].view(len(sel), N, per)
            sec_lp = torch.logsumexp(lp, dim=-1)
            out[sel] = sec_lp.gather(1, act[sel].unsqueeze(1)).squeeze(1)
        return out

    def _sector_entropy(self, logp_cells: torch.Tensor, nsec: np.ndarray) -> torch.Tensor:
        """扇区级熵（探索正则）。"""
        B = logp_cells.shape[0]
        ent = torch.zeros(B, device=self.device)
        for N in np.unique(nsec):
            sel = np.where(nsec == N)[0]
            N = int(N)
            per = self.C // N
            lp = logp_cells[sel][:, :N * per].view(len(sel), N, per)
            sec_lp = torch.logsumexp(lp, dim=-1)
            p = torch.exp(sec_lp)
            ent[sel] = -(p * sec_lp).sum(-1)
        return ent.mean()

    def save_checkpoint(self, path: Path, extra: Optional[Dict] = None):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"net": self.net.state_dict(), "optimizer": self.optimizer.state_dict(),
                    "total_steps": self.total_steps,
                    "train_updates": self.train_updates, "extra": extra or {}}, path)

    def load_checkpoint(self, path: Path, load_optimizer: bool = True) -> Dict:
        ck = torch.load(Path(path), map_location=self.device, weights_only=False)
        self.net.load_state_dict(ck["net"])
        if load_optimizer and "optimizer" in ck:
            self.optimizer.load_state_dict(ck["optimizer"])
        self.total_steps = ck.get("total_steps", 0)
        self.train_updates = ck.get("train_updates", 0)
        return ck.get("extra", {})
