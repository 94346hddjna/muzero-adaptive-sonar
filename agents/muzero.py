# -*- coding: utf-8 -*-
"""MuZero智能体（开题报告算法1、式(17)-(23)）。

流程（算法1）：
  表示：h_t^0=φ(GRU序列编码)（式17）
  规划：π_t←MCTS(h_t^0; g_θ,f_θ,M)（式24-26，想象轨迹，不调用真实环境）
  执行：a_t~π_t作用于真实声呐环境；信念更新（环境内式5）
  存储：(H_t, a_t, r_t, π_t)与根价值→回合结束计算式(20)(21)目标→回放缓冲
  训练：采样K步片段→隐空间unroll→式(22)(23)多目标损失→梯度裁剪→AdamW

价值/奖励标定：网络输出v̂=v/value_scale、r̂=u/reward_scale（MCTS回传与目标已换算）。
温度调度（B10决策）：训练前50%步τ=1.0→50%-90%线性降至0.25；评测τ=0（argmax）。
"""
import math
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from agents import mcts as mcts_mod
from agents.networks import MuZeroNetworks, build_muzero_networks
from agents.replay_buffer import EpisodeBuffer
from env.level1_env import (build_observation_vector, build_raw_observation_vector,
                            sector_center_cell)


def temperature_at(progress: float, t_init: float = 1.0, t_final: float = 0.25) -> float:
    """τ退火调度：前50%为t_init，50%-90%线性降至t_final，之后保持。"""
    if progress <= 0.5:
        return t_init
    if progress >= 0.9:
        return t_final
    return t_init + (t_final - t_init) * (progress - 0.5) / 0.4


class MuZeroAgent:
    """MuZero智能体：批量act（GRU+MCTS）+训练update。"""

    def __init__(
        self,
        muzero_cfg: Dict,
        obs_dim: int,
        canonical_cells: int = 192,
        device: str = "cpu",
        obs_mode: str = "belief",
        use_memory: bool = True,
        seed: int = 0,
    ):
        self.cfg = muzero_cfg
        self.C = canonical_cells
        self.device = device
        self.obs_mode = obs_mode
        self.use_memory = use_memory
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.nets: MuZeroNetworks = build_muzero_networks(
            muzero_cfg, obs_dim, canonical_cells, use_memory=use_memory).to(device)
        self.mcts = mcts_mod.BatchedPUCTMCTS(
            nets=self.nets,
            num_simulations=muzero_cfg.get("num_simulations", 50),
            planning_depth=muzero_cfg.get("planning_depth", 5),
            puct_c1=muzero_cfg.get("puct_c1", 1.25),
            puct_cbase=muzero_cfg.get("puct_cbase", 19652.0),
            gamma=muzero_cfg.get("gamma", 0.99),
            dirichlet_alpha=muzero_cfg.get("dirichlet_alpha", 0.3),
            dirichlet_weight=muzero_cfg.get("dirichlet_weight", 0.25),
            canonical_cells=canonical_cells,
            device=device,
            value_scale=muzero_cfg.get("value_scale", 10.0),
            reward_scale=muzero_cfg.get("reward_scale", 10.0),
        )
        self.optimizer = torch.optim.AdamW(
            self.nets.parameters(),
            lr=muzero_cfg.get("learning_rate", 3e-4),
            weight_decay=muzero_cfg.get("weight_decay", 1e-4),
        )
        self.gamma = muzero_cfg.get("gamma", 0.99)
        self.value_scale = muzero_cfg.get("value_scale", 10.0)
        self.reward_scale = muzero_cfg.get("reward_scale", 10.0)
        self.K = muzero_cfg.get("unroll_steps", 5)
        self.n_bootstrap = muzero_cfg.get("bootstrap_steps", 5)
        self.max_grad_norm = muzero_cfg.get("max_grad_norm", 5.0)
        self._z: Optional[torch.Tensor] = None   # [B, gru_hidden] GRU状态
        self.total_steps = 0
        self.train_updates = 0
        self._policy_only = False  # 消融A：w/o MCTS（评测期开关）
        self._override_m = None     # 消融B/C：评测期MCTS参数覆盖
        self._override_dp = None

    # -- 交互 ----------------------------------------------------------------
    def reset_slots(self, mask: np.ndarray):
        if self._z is not None and np.any(mask):
            m = torch.as_tensor(mask, dtype=torch.bool, device=self.device)
            self._z[m] = 0.0

    def set_policy_only(self, flag: bool):
        """消融A：决策期关闭MCTS，直接用预测网络策略头。"""
        self._policy_only = bool(flag)

    def set_mcts_overrides(self, num_simulations: Optional[int] = None,
                           planning_depth: Optional[int] = None):
        """消融B/C：评测期覆盖模拟次数/规划深度。"""
        self._override_m = num_simulations
        self._override_dp = planning_depth
        if num_simulations is not None:
            self.mcts.M = int(num_simulations)
        if planning_depth is not None:
            self.mcts.Dp = max(1, int(planning_depth))

    @torch.no_grad()
    def act(self, obs: np.ndarray, n_sectors: np.ndarray, env=None, training: bool = True,
            temperature: Optional[float] = None) -> Tuple[np.ndarray, List[Dict]]:
        """一次批量决策。返回(actions[B], per-env信息：policy_target/root_value/raw_policy)。"""
        B = obs.shape[0]
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        if self._z is None or self._z.shape[0] != B:
            self._z = self.nets.initial_state(B, self.device)
        # 式(17)：GRU单步
        z = self.nets.observe(obs_t, self._z)
        h0 = self.nets.represent(z)
        self._z = z.detach()
        infos: List[Dict] = []
        actions = np.zeros(B, dtype=np.int64)
        if self._policy_only:
            # 消融A：直接用预测网络策略头（式19）选动作
            logits, v = self.nets.prediction(h0)
            probs_cells = torch.softmax(logits, dim=-1).cpu().numpy()
            for e in range(B):
                N = int(n_sectors[e])
                probs = mcts_mod.aggregate_cells_to_sectors(probs_cells[e], N)
                s = probs.sum()
                probs = probs / s if s > 0 else np.full(N, 1.0 / N)
                if training:
                    a = int(self.rng.choice(N, p=probs))
                else:
                    a = int(np.argmax(probs))
                actions[e] = a
                infos.append({"policy_target": probs, "root_value": float(v[e, 0]) * self.value_scale,
                              "raw_policy": probs, "visits": None, "root": None})
            return actions, infos
        # MCTS规划（式24-26）
        add_noise = training
        visits_list, root_values, roots = self.mcts.search(
            h0, n_sectors, add_dirichlet=add_noise, rng=self.rng)
        for e in range(B):
            visits = visits_list[e]
            if training:
                tau = temperature if temperature is not None else 1.0
                a = mcts_mod.apply_temperature(visits, tau, self.rng)
            else:
                a = mcts_mod.apply_temperature(visits, 0.0, self.rng)
            actions[e] = a
            pi = mcts_mod.normalized_policy(visits)   # 式(26) τ=1，训练蒸馏目标
            # 原始策略（机理分析图20：raw vs improved）
            raw = roots[e].prior
            infos.append({"policy_target": pi, "root_value": float(root_values[e]),
                          "raw_policy": raw, "visits": visits.tolist(), "root": roots[e]})
        return actions, infos

    # -- 式(20)(21)目标 ------------------------------------------------------
    def compute_episode_targets(self, ep: Dict) -> Dict:
        """n步自举目标z_t（式20）与等效即时奖励u_t（式21）。

        z_t = Σ_{i=0}^{n-1} γ^i r_{t+i} + γ^n v_{t+n}（v取MCTS根价值；越界=终态0）
        u_t = z_t − γ z_{t+1}（z_T:=0）
        """
        rewards = np.asarray(ep["rewards"], dtype=np.float64)
        root_v = np.asarray(ep["root_values"], dtype=np.float64)
        T = len(rewards)
        n = self.n_bootstrap
        gamma = self.gamma
        z = np.zeros(T)
        for t in range(T):
            acc = 0.0
            n_eff = 0
            for i in range(n):
                if t + i < T:
                    acc += (gamma ** i) * rewards[t + i]
                    n_eff = i + 1
            # 自举：t+n < T用根价值；否则终态价值0
            if t + n < T:
                acc += (gamma ** n) * root_v[t + n]
            z[t] = acc
        u = np.zeros(T)
        for t in range(T):
            z_next = z[t + 1] if t + 1 < T else 0.0
            u[t] = z[t] - gamma * z_next
        ep["z"] = z.tolist()
        ep["u"] = u.tolist()
        return ep

    # -- 训练（式(22)(23)） --------------------------------------------------
    def _rebuild_prefix_obs(self, ep: Dict, t0: int) -> np.ndarray:
        """重建o_0..o_{t0}观测向量序列（式15 / raw模式）。"""
        N = int(ep["n_sectors"])
        beliefs = np.asarray(ep["beliefs"], dtype=np.float64)  # [T+1, N]
        actions = ep["actions"]
        dets = ep["dets"]
        T = len(actions)
        obs_list = []
        m = np.zeros(N)
        for t in range(t0 + 1):
            a_prev = actions[t - 1] if t > 0 else -1
            d_prev = dets[t - 1] if t > 0 else 0
            if self.obs_mode == "raw":
                obs_list.append(build_raw_observation_vector(
                    N, a_prev, d_prev, t, ep["t_max"], self.C))
            else:
                obs_list.append(build_observation_vector(
                    beliefs[t], m, N, a_prev, d_prev, t, ep["t_max"], self.C))
            if t < T:
                m[actions[t]] += 1.0
        return np.stack(obs_list)

    def update(self, buffer: EpisodeBuffer) -> Dict:
        """一批K步unroll训练（算法1第14-15行）。梯度贯穿整个隐空间展开链。"""
        segments = buffer.sample_segments(self.cfg.get("batch_size", 128))
        if not segments:
            return {}
        B = len(segments)
        device = self.device
        # 前缀编码（式17批量）
        prefix_list = [self._rebuild_prefix_obs(s["episode"], s["t0"]) for s in segments]
        L = max(p.shape[0] for p in prefix_list)
        D = prefix_list[0].shape[1]
        obs_seq = np.zeros((B, L, D), dtype=np.float32)
        lens = np.zeros(B, dtype=np.int64)
        for i, p in enumerate(prefix_list):
            obs_seq[i, :p.shape[0]] = p
            lens[i] = p.shape[0]
        obs_seq_t = torch.as_tensor(obs_seq, device=device)
        lens_t = torch.as_tensor(lens, device=device)
        h = self.nets.encode_sequence(obs_seq_t, lens_t)  # [B, 128]（t0位置）
        # 目标（式20/21/26）
        Ns = np.array([int(s["episode"]["n_sectors"]) for s in segments], dtype=np.int64)
        T_arr = np.array([len(s["episode"]["rewards"]) for s in segments], dtype=np.int64)
        t0_arr = np.array([int(s["t0"]) for s in segments], dtype=np.int64)
        z_all = [np.asarray(s["episode"]["z"], dtype=np.float64) for s in segments]
        pi_all = [np.asarray(s["episode"]["pi_targets"], dtype=np.float64) for s in segments]
        u_all = [np.asarray(s["episode"]["u"], dtype=np.float64) for s in segments]
        acts_all = [s["episode"]["actions"] for s in segments]
        K = self.K
        # k=0..K位置有效性：t0+k < T（k=0恒有效：t0<T）
        valid = np.stack([(t0_arr + k) < T_arr for k in range(K + 1)], axis=1)  # [B, K+1]
        # k=0：f(h^0)（式19）→ 目标 z_{t0}/π_{t0}
        logits, v_pred = self.nets.prediction(h)
        value_loss, policy_loss, n_v_terms = self._value_policy_loss(
            logits, v_pred, z_all, pi_all, t0_arr, np.ones(B, dtype=bool), Ns, offset=0)
        # k=1..K：g_θ(h,a)→(r,h')（式18）+f_θ(h')；全批量展开+损失掩码
        reward_loss = torch.tensor(0.0, device=device)
        n_r_terms = 0
        h_cur = h
        for k in range(1, K + 1):
            ok = valid[:, k]
            a_cells = np.zeros(B, dtype=np.int64)
            for i in range(B):
                if ok[i]:
                    a_cells[i] = sector_center_cell(
                        int(acts_all[i][t0_arr[i] + k - 1]), int(Ns[i]), self.C)
                else:
                    a_cells[i] = sector_center_cell(0, int(Ns[i]), self.C)  # 占位动作（损失掩蔽）
            oh = torch.zeros(B, self.C, device=device)
            oh[torch.arange(B, device=device),
               torch.as_tensor(a_cells, device=device)] = 1.0
            r_pred, h_cur = self.nets.dynamics(h_cur, oh)      # 梯度链保留
            logits_k, v_k = self.nets.prediction(h_cur)
            if np.any(ok):
                idxs = np.where(ok)[0]
                u_t = torch.as_tensor(
                    np.array([u_all[i][t0_arr[i] + k - 1] / self.reward_scale for i in idxs]),
                    dtype=torch.float32, device=device)
                reward_loss = reward_loss + F.mse_loss(r_pred[idxs].squeeze(-1), u_t)
                n_r_terms += len(idxs)
            loss_v, loss_p, cnt = self._value_policy_loss(
                logits_k, v_k, z_all, pi_all, t0_arr, ok, Ns, offset=k)
            value_loss = value_loss + loss_v
            policy_loss = policy_loss + loss_p
            n_v_terms += cnt
        total = value_loss + policy_loss + reward_loss
        self.optimizer.zero_grad(set_to_none=True)
        total.backward()
        torch.nn.utils.clip_grad_norm_(self.nets.parameters(), self.max_grad_norm)
        self.optimizer.step()
        self.train_updates += 1
        return {
            "loss_policy": float(policy_loss.detach()),
            "loss_value": float(value_loss.detach()),
            "loss_reward": float(reward_loss.detach()),
            "total_loss": float(total.detach()),
            "n_value_terms": n_v_terms,
            "n_reward_terms": n_r_terms,
            "learning_rate": self.optimizer.param_groups[0]["lr"],
            "train_updates": self.train_updates,
        }

    def _value_policy_loss(self, logits, v_pred, z_all, pi_all, t0s, ok_mask, Ns,
                           offset: int = 0) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """l_v（式23 MSE）与l_p（扇区级交叉熵，式23）。目标取位置t0s+offset。

        logits/v_pred为[B, ...]全批量预测；ok_mask屏蔽无效样本。
        """
        device = self.device
        idxs = np.where(ok_mask)[0]
        if len(idxs) == 0:
            z0 = torch.tensor(0.0, device=device)
            return z0, z0.clone(), 0
        # 价值损失：v̂ vs z/value_scale
        z_t = torch.as_tensor(
            np.array([z_all[i][t0s[i] + offset] / self.value_scale for i in idxs]),
            dtype=torch.float32, device=device)
        loss_v = F.mse_loss(v_pred[idxs].squeeze(-1), z_t)
        # 策略损失：扇区边缘化交叉熵（按N分组）
        logp_cells = F.log_softmax(logits[idxs], dim=-1)  # [g, C]
        gN = Ns[idxs]
        loss_p = torch.tensor(0.0, device=device)
        for N in np.unique(gN):
            sel = np.where(gN == N)[0]
            N = int(N)
            per = self.C // N
            assert self.C % N == 0, f"C={self.C} not divisible by N={N}"
            lp = logp_cells[sel].view(len(sel), N, per)
            sec_lp = torch.logsumexp(lp, dim=-1)  # [g, N] 扇区log概率
            pi_t = torch.as_tensor(
                np.stack([pi_all[i][t0s[i] + offset] for i in idxs[sel]]),
                dtype=torch.float32, device=device)
            loss_p = loss_p + (-(pi_t * sec_lp).sum(dim=-1)).mean()
        return loss_v, loss_p, len(idxs)

    # -- 检查点 --------------------------------------------------------------
    def save_checkpoint(self, path: Path, extra: Optional[Dict] = None):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "nets": self.nets.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "total_steps": self.total_steps,
            "train_updates": self.train_updates,
            "seed": self.seed,
            "cfg": self.cfg,
            "obs_mode": self.obs_mode,
            "use_memory": self.use_memory,
            "extra": extra or {},
        }, path)

    def load_checkpoint(self, path: Path, load_optimizer: bool = True) -> Dict:
        ck = torch.load(Path(path), map_location=self.device, weights_only=False)
        self.nets.load_state_dict(ck["nets"])
        if load_optimizer and "optimizer" in ck:
            self.optimizer.load_state_dict(ck["optimizer"])
        self.total_steps = ck.get("total_steps", 0)
        self.train_updates = ck.get("train_updates", 0)
        return ck.get("extra", {})
