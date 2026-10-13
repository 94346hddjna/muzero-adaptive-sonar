# -*- coding: utf-8 -*-
"""奖励函数单元测试（式16）：稀疏/塑形两种模式、真检出/漏检/虚警数值。"""
import numpy as np
import pytest

from env.level1_env import SonarLevel1VecEnv


def _reward_values(env_cfg):
    rc = env_cfg["reward"]
    return rc["R_hit"], rc["lambda_s"], rc["lambda_f"], rc["lambda_e"], rc["action_cost"]


def test_reward_formula_true_detection(env_cfg):
    """真检出：r = R_hit − λs − λe·C（终止，success）。"""
    env = SonarLevel1VecEnv(1, env_cfg, t_max=20, master_seed=1,
                            fixed_difficulty="easy",
                            difficulty_overrides={"pd_target": 0.999999})
    # 极高PD：首次指向目标必检出
    env.reset()
    # 强制成功路径：直接检查step返回reward与公式一致性（用伪场景）
    # 改为构造确定性检测：将pd_ep设为1.0
    env.pd_ep[0] = 1.0
    obs, r, done, infos = env.step(np.array([int(env.target[0])]))
    R_hit, lam_s, lam_f, lam_e, c = _reward_values(env_cfg)
    assert r[0] == pytest.approx(R_hit - lam_s - lam_e * c)
    assert done[0] and infos[0]["success"]


def test_reward_formula_miss(env_cfg):
    """漏检（指向非目标且无虚警）：r = −λs − λe·C。"""
    env = SonarLevel1VecEnv(1, env_cfg, t_max=20, master_seed=2,
                            fixed_difficulty="medium")
    env.reset()
    env.pd_ep[0] = 0.5
    env.pfa[0] = 0.0  # 无虚警
    x = int(env.target[0])
    a = (x + 8) % int(env.n[0])
    obs, r, done, infos = env.step(np.array([a]))
    R_hit, lam_s, lam_f, lam_e, c = _reward_values(env_cfg)
    assert r[0] == pytest.approx(-lam_s - lam_e * c)
    assert infos[0] is None  # 未检出不终止回合


def test_reward_formula_false_alarm(env_cfg):
    """虚警（o=1但x≠a，按式16字面）：r = R_hit − λs − λf − λe·C。"""
    env = SonarLevel1VecEnv(1, env_cfg, t_max=20, master_seed=3,
                            fixed_difficulty="medium")
    env.reset()
    env.pd_ep[0] = 0.5
    env.pfa[0] = 1.0  # 必虚警
    x = int(env.target[0])
    a = (x + 8) % int(env.n[0])
    obs, r, done, infos = env.step(np.array([a]))
    R_hit, lam_s, lam_f, lam_e, c = _reward_values(env_cfg)
    assert r[0] == pytest.approx(R_hit - lam_s - lam_f - lam_e * c)
    assert infos[0] is None  # 虚警不终止回合（目标未被真正发现）
    assert int(env.prev_det[0]) == 1  # 观测为检出（虚警）


def test_false_alarm_no_arbitrage(env_cfg):
    """设计决策B1：λf=R_hit使虚警步净收益=漏检步（无虚警套利）。"""
    rc = env_cfg["reward"]
    assert rc["lambda_f"] >= rc["R_hit"]


def test_shaped_reward_mode_runs(env_cfg):
    """塑形模式（消融G，PBRS）可运行且改变奖励数值。"""
    env_sparse = SonarLevel1VecEnv(2, env_cfg, t_max=20, master_seed=4,
                                   fixed_difficulty="medium", reward_mode="sparse")
    env_shaped = SonarLevel1VecEnv(2, env_cfg, t_max=20, master_seed=4,
                                   fixed_difficulty="medium", reward_mode="shaped",
                                   shaping_coef=1.0)
    o1 = env_sparse.reset()
    o2 = env_shaped.reset()
    np.testing.assert_allclose(o1, o2, atol=1e-6)
    rng = np.random.default_rng(0)
    same_count = 0
    for _ in range(10):
        a = np.array([rng.integers(env_sparse.n[s]) for s in range(2)])
        _, r1, _, _ = env_sparse.step(a)
        _, r2, _, _ = env_shaped.step(a)
        if not np.allclose(r1, r2):
            same_count += 1
    assert same_count > 0  # 塑形确实改变奖励


def test_budget_exhaustion(env_cfg):
    """达到T_max未发现→回合终止且success=False。"""
    env = SonarLevel1VecEnv(1, env_cfg, t_max=3, master_seed=5,
                            fixed_difficulty="medium")
    env.reset()
    env.pd_ep[0] = 0.0
    env.pfa[0] = 0.0
    for t in range(3):
        obs, r, done, infos = env.step(np.array([0]))
    assert done[0]
    assert infos[0] is not None and not infos[0]["success"]
    assert infos[0]["steps"] == 3
