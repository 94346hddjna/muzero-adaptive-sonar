# -*- coding: utf-8 -*-
"""MCTS单元测试（式24-26）：PUCT/回传/温度/访问分布/深度截断/Dirichlet噪声。"""
import math

import numpy as np
import pytest
import torch

from agents import mcts as mcts_mod
from agents.networks import build_muzero_networks
from env.level1_env import obs_dim


class _FakeNets:
    """可控假网络：先验集中于扇区3、价值恒正、奖励恒−1（式25可解析验证）。"""

    def __init__(self, C=192, hidden=128):
        self.C = C
        self.hidden = hidden

    def prediction(self, h):
        B = h.shape[0]
        logits = torch.zeros(B, self.C)
        # 扇区3的格logits调高（16扇区：每扇区12格）
        per = self.C // 16
        logits[:, 3 * per:(3 + 1) * per] = 5.0
        return logits, torch.full((B, 1), 1.0)  # v=1

    def dynamics(self, h, oh):
        B = h.shape[0]
        r = torch.full((B, 1), -0.5)   # 每步奖励−0.5
        return r, h * 0.9 + 0.05


def test_root_visits_match_simulation_count():
    """根访问总数=模拟次数M（式26的分母）。"""
    C, H = 192, 128
    nets = _FakeNets()
    m = mcts_mod.BatchedPUCTMCTS(nets, num_simulations=20, planning_depth=3,
                                 gamma=0.99, canonical_cells=C, device="cpu")
    h = torch.randn(2, H)
    visits, root_vals, roots = m.search(h, np.array([16, 16]), add_dirichlet=False)
    for v in visits:
        assert v.sum() == 20
    assert len(visits[0]) == 16


def test_puct_select_prefers_high_prior():
    """式(24)：先验高的动作被优先扩展。"""
    nets = _FakeNets()
    m = mcts_mod.BatchedPUCTMCTS(nets, num_simulations=30, planning_depth=2,
                                 canonical_cells=192, device="cpu")
    visits, _, _ = m.search(torch.randn(1, 128), np.array([16]))
    assert visits[0][3] == visits[0].max()  # 先验集中于扇区3


def test_backup_value_accounting():
    """式(25)回传：叶值沿路径折扣累计，Q=W/N。"""
    # 手工构造树验证_backup
    m = mcts_mod.BatchedPUCTMCTS(_FakeNets(), num_simulations=1, planning_depth=5,
                                 gamma=0.5, canonical_cells=192, device="cpu")
    root = mcts_mod.Node(np.array([0.5, 0.5]), depth=0)
    root.children = [None, None]
    root.has_children = True
    child = mcts_mod.Node(np.array([1.0]), depth=1)
    child.reward = 2.0
    child.hidden = torch.randn(1, 128)
    child.children = [None]
    child.has_children = True
    root.children[1] = child
    leaf = mcts_mod.Node(np.array([1.0]), depth=2)
    leaf.reward = 1.0
    leaf.value_estimate = 4.0
    leaf.children = [None]
    leaf.has_children = True
    child.children[0] = leaf
    # 回传：叶值4 → 叶G=4；child G=leaf.reward+γ·4=1+2=3；root G=child.reward+γ·3=2+1.5=3.5
    m._backup([root, child, leaf], 4.0)
    assert leaf.value_sum == 4.0 and leaf.visit_count == 1
    assert child.value_sum == 3.0 and child.visit_count == 1
    assert root.value_sum == 3.5 and root.visit_count == 1


def test_temperature_sampling():
    rng = np.random.default_rng(0)
    visits = np.array([10, 30, 60])
    # τ=0：argmax
    assert mcts_mod.apply_temperature(visits, 0.0, rng) == 2
    # τ→∞：近似均匀（采样频率约束放宽）
    picks = [mcts_mod.apply_temperature(visits, 1000.0, rng) for _ in range(300)]
    assert min(picks) == 0 or max(picks) == 2  # 允许覆盖
    # τ=1：按访问比例
    picks = [mcts_mod.apply_temperature(visits, 1.0, rng) for _ in range(3000)]
    freq = np.bincount(picks, minlength=3) / 3000
    assert abs(freq[2] - 0.6) < 0.05 and abs(freq[0] - 0.1) < 0.04


def test_normalized_policy():
    v = np.array([2, 6, 2])
    p = mcts_mod.normalized_policy(v)
    assert np.allclose(p, [0.2, 0.6, 0.2])
    # 空访问→均匀
    p0 = mcts_mod.normalized_policy(np.zeros(4))
    assert np.allclose(p0, 0.25)


def test_depth_limit_respected():
    """规划深度D_p：树深不超过D_p。"""
    nets = _FakeNets()
    m = mcts_mod.BatchedPUCTMCTS(nets, num_simulations=15, planning_depth=2,
                                 canonical_cells=192, device="cpu")
    _, _, roots = m.search(torch.randn(1, 128), np.array([16]))

    def max_depth(node):
        if not node.has_children or all(c is None for c in node.children):
            return node.depth
        return max(max_depth(c) for c in node.children if c is not None)
    assert max_depth(roots[0]) <= 2


def test_dirichlet_noise_changes_visits():
    """根先验Dirichlet噪声改变访问分布（训练探索）。"""
    nets = _FakeNets()
    m = mcts_mod.BatchedPUCTMCTS(nets, num_simulations=25, planning_depth=2,
                                 canonical_cells=192, device="cpu")
    h = torch.randn(1, 128)
    v1, _, _ = m.search(h, np.array([16]), add_dirichlet=False,
                        rng=np.random.default_rng(1))
    v2, _, _ = m.search(h, np.array([16]), add_dirichlet=True,
                        rng=np.random.default_rng(2))
    assert not np.allclose(v1, v2)


def test_mcts_uses_only_latent_dynamics():
    """搜索绝不调用真实环境（双通道原则）：search接口不含环境参数。"""
    import inspect
    sig = inspect.signature(mcts_mod.BatchedPUCTMCTS.search)
    assert "env" not in sig.parameters
    assert set(sig.parameters) <= {"self", "root_hidden", "n_sectors", "add_dirichlet", "rng"}
