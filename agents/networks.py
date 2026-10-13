# -*- coding: utf-8 -*-
"""MuZero三网络（开题报告表3、式(17)-(19)）。

表3结构（规范网格C=192，见design_notes B17）：
  表示网络h_θ: 输入2C+3 → FC(256) → GRU(256) → FC(128)         输出隐状态128
  动力学网络g_θ: 输入128+C(动作嵌入=扇区中心格one-hot) → FC(256)→FC(256)
                → 隐状态128 + 奖励1（奖励头线性）
  预测网络f_θ: 输入128 → FC(256)→FC(256) → 策略(C格logits)+价值1

式(17)：z_t=GRU(z_{t−1}, o_t)，h_t^0=φ(z_t)（φ为线性映射）
式(18)：(r^k, h^{k+1}) = g_θ(h^k, a^k)
式(19)：(p^k, v^k) = f_θ(h^k)

消融开关：
  use_memory=False（消融E w/o memory）：去除GRU，仅用当前观测向量MLP编码
  use_belief=False（消融D w/o belief）：观测向量不含b/m（原始观测模式，env侧配合）
价值/奖励标定（B决策）：网络输出v̂=v/value_scale、r̂=u/reward_scale，
外部备份/目标计算乘回尺度，保证数值稳定且与式(25)一致。
"""
from typing import Dict, Optional

import torch
import torch.nn as nn


def mlp(sizes, activation=nn.ReLU, out_linear_last=True):
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2 or not out_linear_last:
            layers.append(activation())
    return nn.Sequential(*layers)


class MuZeroNetworks(nn.Module):
    """三网络联合模块（表3）。"""

    def __init__(
        self,
        obs_dim: int,                # 2C+3（信念模式）或 C+2（原始模式）
        action_embed_dim: int,       # C（规范格one-hot）
        hidden_dim: int = 128,       # 隐状态维度 dim(h)
        gru_hidden: int = 256,       # GRU隐维 dim(z)
        representation_fc: int = 256,
        dynamics_fc=(256, 256),
        prediction_fc=(256, 256),
        use_memory: bool = True,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_embed_dim = action_embed_dim
        self.hidden_dim = hidden_dim
        self.gru_hidden = gru_hidden
        self.use_memory = use_memory

        # h_θ（式17）：FC(256)→GRU(256)→φ FC(128)
        self.rep_in = mlp([obs_dim, representation_fc])
        if use_memory:
            self.gru = nn.GRUCell(representation_fc, gru_hidden)
        else:
            self.no_mem = mlp([representation_fc, gru_hidden])
        self.rep_out = nn.Linear(gru_hidden, hidden_dim)

        # g_θ（式18）：concat(h, 动作one-hot) → FC(256)→FC(256) → h'128 + r 1
        dyn_in = hidden_dim + action_embed_dim
        self.dyn_trunk = mlp([dyn_in, *dynamics_fc])
        self.dyn_state_head = nn.Linear(dynamics_fc[-1], hidden_dim)
        self.dyn_reward_head = nn.Linear(dynamics_fc[-1], 1)   # 线性（表3：奖励头线性）

        # f_θ（式19）：h → FC(256)→FC(256) → 策略C logits + 价值1
        self.pred_trunk = mlp([hidden_dim, *prediction_fc])
        self.pred_policy_head = nn.Linear(prediction_fc[-1], action_embed_dim)  # Softmax外部
        self.pred_value_head = nn.Linear(prediction_fc[-1], 1)                  # 线性

    def initial_state(self, batch: int, device) -> torch.Tensor:
        dtype = next(self.parameters()).dtype
        return torch.zeros(batch, self.gru_hidden, device=device, dtype=dtype)

    def observe(self, obs: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """式(17)单步：z_t=GRU(z_{t−1}, FC(o_t))。"""
        x = self.rep_in(obs)
        if self.use_memory:
            return self.gru(x, z)
        return self.no_mem(x)

    def represent(self, z: torch.Tensor) -> torch.Tensor:
        """φ：z→隐状态h^0。"""
        return self.rep_out(z)

    def encode_sequence(self, obs_seq: torch.Tensor, lens: torch.Tensor) -> torch.Tensor:
        """批量编码变长前缀（训练unroll起点）：obs_seq [B, L, D]，lens [B]。"""
        B, L, D = obs_seq.shape
        z = self.initial_state(B, obs_seq.device)
        if not self.use_memory:
            # 无记忆：直接用每序列最后一帧
            idx = (lens - 1).clamp(min=0)
            last = obs_seq[torch.arange(B, device=obs_seq.device), idx]
            return self.represent(self.observe(last, z))
        out = torch.zeros(B, self.gru_hidden, device=obs_seq.device)
        for t in range(L):
            mask = (lens > t).float().unsqueeze(1)
            z_new = self.observe(obs_seq[:, t], z)
            z = torch.where(mask > 0, z_new, z)
            out = torch.where(mask > 0, z_new, out)
        return self.represent(out)

    def dynamics(self, h: torch.Tensor, action_cell_onehot: torch.Tensor):
        """式(18)：(r, h')=g_θ(h, a)。动作=扇区中心格one-hot。"""
        x = torch.cat([h, action_cell_onehot], dim=-1)
        t = self.dyn_trunk(x)
        return self.dyn_reward_head(t), self.dyn_state_head(t)

    def prediction(self, h: torch.Tensor):
        """式(19)：(policy_logits[C], value)。"""
        t = self.pred_trunk(h)
        return self.pred_policy_head(t), self.pred_value_head(t)

    def root_expansion(self, h: torch.Tensor):
        return self.prediction(h)


def action_cell_onehot(cell_indices: torch.Tensor, action_embed_dim: int) -> torch.Tensor:
    """扇区中心格one-hot动作嵌入。"""
    B = cell_indices.shape[0]
    oh = torch.zeros(B, action_embed_dim, device=cell_indices.device)
    oh[torch.arange(B, device=cell_indices.device), cell_indices] = 1.0
    return oh


def build_muzero_networks(muzero_cfg: Dict, obs_dim: int, canonical_cells: int,
                          use_memory: bool = True) -> MuZeroNetworks:
    return MuZeroNetworks(
        obs_dim=obs_dim,
        action_embed_dim=canonical_cells,
        hidden_dim=muzero_cfg.get("hidden_dim", 128),
        gru_hidden=muzero_cfg.get("gru_hidden", 256),
        representation_fc=muzero_cfg.get("representation_fc", 256),
        dynamics_fc=tuple(muzero_cfg.get("dynamics_fc", [256, 256])),
        prediction_fc=tuple(muzero_cfg.get("prediction_fc", [256, 256])),
        use_memory=use_memory,
    )
