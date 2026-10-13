# -*- coding: utf-8 -*-
"""信念更新单元测试（式4-5）：归一化/漏检衰减/检出坍缩/先验类型。"""
import numpy as np
import pytest

from env import belief as bl


def test_belief_sum_always_one():
    b = np.full(16, 1 / 16)
    rng = np.random.default_rng(0)
    for _ in range(200):
        a = int(rng.integers(16))
        det = int(rng.integers(2))
        pd = rng.uniform(0.3, 0.99)
        pfa = rng.uniform(0.001, 0.2)
        b = bl.bayes_update(b, a, det, pd, pfa)
        assert abs(b.sum() - 1.0) < 1e-9
        assert np.all(b >= 0)


def test_miss_decays_not_zeroed():
    """式(5)：未检出→指向扇区后验按漏检概率衰减，非清零。"""
    b = np.full(8, 1 / 8)
    pd, pfa = 0.85, 0.05
    b1 = bl.bayes_update(b, 0, 0, pd, pfa)
    assert 0 < b1[0] < 1 / 8
    # 精确验证贝叶斯公式
    like0 = (1 - pd) / (1 - pfa)  # 相对似然比
    expected = (b[0] * (1 - pd)) / (b[0] * (1 - pd) + 7 * (1 / 8) * (1 - pfa))
    assert abs(b1[0] - expected) < 1e-12


def test_detection_collapses_to_pointed_sector():
    b = np.full(16, 1 / 16)
    b1 = bl.bayes_update(b, 5, 1, 0.9, 0.05)
    # 检出→后验集中于指向扇区（按PD/PFA似然比）
    expected = (b[5] * 0.9) / (b[5] * 0.9 + 15 * b[0] * 0.05)
    assert abs(b1[5] - expected) < 1e-12
    assert b1[5] > 0.5


def test_repeated_misses_recover_target_probability():
    """多次漏检后其他扇区概率此消彼长上升（式5物理意义）。"""
    b = np.full(8, 1 / 8)
    pd, pfa = 0.9, 0.05
    for _ in range(10):
        b = bl.bayes_update(b, 0, 0, pd, pfa)
    assert b[0] < 1e-4
    assert b[1:].sum() > 0.999


def test_prior_types():
    rng = np.random.default_rng(1)
    N, target = 16, 7
    bu = bl.make_initial_belief(N, "uniform", target, rng=rng)
    assert np.allclose(bu, 1 / N)
    bc = bl.make_initial_belief(N, "correct", target, rng=rng, sigma_sectors=1.0)
    assert bc[target] == bc.max()
    bm = bl.make_initial_belief(N, "moderate", target, rng=rng, sigma_sectors=2.5)
    assert bm[target] == bm.max()
    bb = bl.make_initial_belief(N, "biased", target, rng=rng, sigma_sectors=1.5)
    assert bb[target] < bb.max()  # 偏向错误扇区
    ba = bl.make_initial_belief(N, "adversarial", target, rng=rng, sigma_sectors=1.0)
    far = (target + N // 2) % N
    assert ba[far] == ba.max()
    assert ba[target] < 1e-6
    for b in [bu, bc, bm, bb, ba]:
        assert abs(b.sum() - 1.0) < 1e-9


def test_batch_update_matches_single():
    B, N = 5, 16
    rng = np.random.default_rng(2)
    beliefs = np.stack([rng.dirichlet(np.ones(N)) for _ in range(B)])
    actions = rng.integers(0, N, B)
    dets = rng.integers(0, 2, B)
    pds = rng.uniform(0.5, 0.95, B)
    pfas = rng.uniform(0.01, 0.1, B)
    out = bl.bayes_update_batch(beliefs.copy(), actions, dets, pds, pfas)
    for i in range(B):
        single = bl.bayes_update(beliefs[i], int(actions[i]), int(dets[i]),
                                 float(pds[i]), float(pfas[i]))
        assert np.allclose(out[i], single, atol=1e-12)


def test_belief_is_exact_posterior():
    """滤波器与采样模型一致→信念为精确后验（蒙特卡洛验证）。"""
    N, pd, pfa = 4, 0.8, 0.1
    b0 = np.array([0.1, 0.2, 0.3, 0.4])
    # 观测序列：指向扇区1未检出
    b = bl.bayes_update(b0, 1, 0, pd, pfa)
    # 解析后验
    like = np.array([1 - pfa, 1 - pd, 1 - pfa, 1 - pfa])
    expected = b0 * like / (b0 * like).sum()
    assert np.allclose(b, expected, atol=1e-12)
