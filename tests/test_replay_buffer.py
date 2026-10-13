# -*- coding: utf-8 -*-
"""回放缓冲与MuZero训练目标单元测试（式20-21、算法1第12-15行）。"""
import numpy as np
import pytest

from agents.muzero import MuZeroAgent, temperature_at
from agents.replay_buffer import EpisodeBuffer


def _make_episode(T=8, N=16, gamma=0.99):
    rng = np.random.default_rng(0)
    rewards = [(-1.1) if i < T - 1 else 8.9 for i in range(T)]
    root_values = [0.5] * T
    ep = {
        "n_sectors": N, "t_max": 20,
        "beliefs": [rng.dirichlet(np.ones(N)).tolist() for _ in range(T + 1)],
        "actions": [int(rng.integers(N)) for _ in range(T)],
        "dets": [0] * (T - 1) + [1],
        "rewards": rewards,
        "pi_targets": [np.full(N, 1.0 / N).tolist() for _ in range(T)],
        "root_values": root_values,
    }
    return ep


def test_buffer_fifo_and_sampling():
    buf = EpisodeBuffer(capacity=5, seed=0)
    for i in range(8):
        buf.add({"id": i, "rewards": [0.0] * (i % 5 + 1)})
    assert len(buf) == 5
    assert buf._buf[-1]["id"] == 7  # FIFO淘汰最旧
    segs = buf.sample_segments(10)
    assert len(segs) == 10
    for s in segs:
        assert 0 <= s["t0"] < 8
        assert "rewards" in s["episode"]


def test_nstep_targets_formula():
    """式(20)：z_t = Σγ^i r_{t+i} + γ^n v_{t+n}；式(21)：u_t = z_t − γ z_{t+1}。"""
    cfg = {"gamma": 0.5, "bootstrap_steps": 2, "value_scale": 1.0, "reward_scale": 1.0}
    agent = MuZeroAgent.__new__(MuZeroAgent)  # 不初始化网络
    agent.gamma, agent.n_bootstrap = 0.5, 2
    agent.value_scale, agent.reward_scale = 1.0, 1.0
    ep = {
        "rewards": [1.0, 2.0, 4.0, 8.0],
        "root_values": [100.0, 200.0, 300.0, 400.0],
    }
    # n=2, γ=0.5
    # z0 = r0 + 0.5r1 + 0.25·v2 = 1 + 1 + 75 = 77
    # z1 = r1 + 0.5r2 + 0.25·v3 = 2 + 2 + 100 = 104
    # z2 = r2 + 0.5r3 (无自举: t+n=4>=T) = 4+4 = 8
    # z3 = r3 = 8
    ep = agent.compute_episode_targets(ep)
    z, u = np.array(ep["z"]), np.array(ep["u"])
    assert z[0] == pytest.approx(77.0)
    assert z[1] == pytest.approx(104.0)
    assert z[2] == pytest.approx(8.0)
    assert z[3] == pytest.approx(8.0)
    assert u[0] == pytest.approx(77.0 - 0.5 * 104.0)
    assert u[3] == pytest.approx(8.0 - 0.0)  # z_T:=0


def test_agent_update_runs_and_losses_finite(muzero_cfg):
    """MuZero训练一步可运行且损失有限（式22）。"""
    from env.level1_env import obs_dim
    agent = MuZeroAgent(muzero_cfg, obs_dim(192, "belief"), 192, device="cpu", seed=0)
    buf = EpisodeBuffer(capacity=100, seed=0)
    for _ in range(12):
        ep = _make_episode()
        ep = agent.compute_episode_targets(ep)
        ep["beliefs"] = [np.asarray(b) for b in ep["beliefs"]]
        ep["pi_targets"] = [np.asarray(p) for p in ep["pi_targets"]]
        buf.add(ep)
    stats = agent.update(buf)
    assert stats and np.isfinite(stats["total_loss"])
    assert "loss_policy" in stats and "loss_value" in stats and "loss_reward" in stats
    assert stats["train_updates"] == 1


def test_episode_record_consistency():
    """episode记录字段完备（beliefs长度=T+1等）。"""
    ep = _make_episode()
    T = len(ep["rewards"])
    assert len(ep["beliefs"]) == T + 1
    assert len(ep["pi_targets"]) == T
    assert len(ep["actions"]) == T


def test_temperature_schedule():
    """B10决策：前50%τ=1.0→90%线性至0.25。"""
    assert temperature_at(0.0) == 1.0
    assert temperature_at(0.5) == 1.0
    assert abs(temperature_at(0.7) - 0.625) < 1e-9  # 线性中点
    assert temperature_at(0.9) == 0.25
    assert temperature_at(1.0) == 0.25
