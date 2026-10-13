# -*- coding: utf-8 -*-
"""MuZero三网络单元测试（表3、式17-19）：形状/前向/编码/unroll梯度链。"""
import numpy as np
import pytest
import torch

from agents.networks import MuZeroNetworks, build_muzero_networks, action_cell_onehot
from env.level1_env import obs_dim, sector_center_cell


@pytest.fixture
def nets(muzero_cfg):
    C = 192
    return build_muzero_networks(muzero_cfg, obs_dim(C, "belief"), C).double()


def test_network_shapes_match_table3(nets, muzero_cfg):
    """表3：h_θ输出128隐状态；g_θ输出128+1；f_θ输出策略+价值。"""
    C = 192
    hdim = muzero_cfg["hidden_dim"]
    obs = torch.randn(4, obs_dim(C, "belief"), dtype=torch.float64)
    z = nets.initial_state(4, "cpu")
    z1 = nets.observe(obs, z)
    assert z1.shape == (4, muzero_cfg["gru_hidden"])
    h = nets.represent(z1)
    assert h.shape == (4, hdim)
    oh = action_cell_onehot(torch.tensor([sector_center_cell(3, 16, C),
                                          sector_center_cell(7, 16, C),
                                          sector_center_cell(1, 8, C),
                                          sector_center_cell(2, 8, C)]), C).double()
    r, h2 = nets.dynamics(h, oh)
    assert h2.shape == (4, hdim)
    assert r.shape == (4, 1)
    logits, v = nets.prediction(h2)
    assert logits.shape == (4, C)
    assert v.shape == (4, 1)


def test_encode_sequence_prefix(nets):
    C = 192
    B, L, D = 3, 7, obs_dim(C, "belief")
    obs_seq = torch.randn(B, L, D, dtype=torch.float64)
    lens = torch.tensor([7, 3, 5])
    h = nets.encode_sequence(obs_seq, lens)
    assert h.shape == (3, 128)
    # 前缀长度不影响输出形状；与逐步编码一致（全长度样本）
    z = nets.initial_state(1, "cpu")
    for t in range(7):
        z = nets.observe(obs_seq[0:1, t], z)
    h_manual = nets.represent(z)
    assert torch.allclose(h[0:1], h_manual, atol=1e-8)


def test_wo_memory_variant(muzero_cfg):
    """消融E：去除GRU仍可前向（仅当前观测）。"""
    C = 192
    nets = build_muzero_networks(muzero_cfg, obs_dim(C, "belief"), C,
                                 use_memory=False).double()
    obs = torch.randn(2, obs_dim(C, "belief"), dtype=torch.float64)
    z = nets.initial_state(2, "cpu")
    z1 = nets.observe(obs, z)
    h = nets.represent(z1)
    assert h.shape == (2, 128)
    # 无记忆：两个不同长度序列的同一最后帧→相同隐状态
    seq = torch.randn(1, 5, obs_dim(C, "belief"), dtype=torch.float64)
    h1 = nets.encode_sequence(seq, torch.tensor([5]))
    h2 = nets.encode_sequence(seq[:, 4:5], torch.tensor([1]))
    assert torch.allclose(h1, h2, atol=1e-8)


def test_raw_obs_mode_variant(muzero_cfg):
    """消融D：raw观测模式（C+2维输入）可前向。"""
    C = 192
    nets = build_muzero_networks(muzero_cfg, obs_dim(C, "raw"), C).double()
    obs = torch.randn(2, obs_dim(C, "raw"), dtype=torch.float64)
    z = nets.observe(obs, nets.initial_state(2, "cpu"))
    assert nets.represent(z).shape == (2, 128)


def test_dynamics_gradient_chain(muzero_cfg):
    """K步unroll梯度贯穿隐状态链（式22训练要求）。"""
    C = 192
    nets = build_muzero_networks(muzero_cfg, obs_dim(C, "belief"), C).double()
    h = torch.randn(2, 128, dtype=torch.float64, requires_grad=True)
    oh = torch.zeros(2, C, dtype=torch.float64)
    oh[0, sector_center_cell(1, 16, C)] = 1.0
    oh[1, sector_center_cell(2, 16, C)] = 1.0
    total = 0.0
    hc = h
    for k in range(5):
        r, hc = nets.dynamics(hc, oh)
        logits, v = nets.prediction(hc)
        total = total + r.sum() + v.sum()
    total.backward()
    assert h.grad is not None
    assert torch.isfinite(h.grad).all()
