# -*- coding: utf-8 -*-
"""种子隔离单元测试（第十一章）：四命名空间互不相交+场景可复现。"""
import numpy as np
import pytest

from utils import seeding as us


def test_namespace_isolation():
    res = us.check_namespace_isolation()
    assert res["ok"], res["overlaps"]


def test_train_val_test_ood_disjoint():
    train = set(us.TRAIN_SEEDS)
    val = {us.VALIDATION_SCENARIO_SEED}
    test = {us.TEST_SCENARIO_SEED}
    ood = {us.OOD_SCENARIO_SEED}
    assert not (train & val) and not (train & test) and not (train & ood)
    assert not (val & test) and not (val & ood) and not (test & ood)


def test_env_episode_rng_deterministic():
    r1 = us.env_episode_rng(30303, 5)
    r2 = us.env_episode_rng(30303, 5)
    assert r1.random() == r2.random()
    # 不同episode不同流
    r3 = us.env_episode_rng(30303, 6)
    assert r1.random(100).tolist() != r3.random(100).tolist()


def test_scenarios_reproducible(env_cfg):
    """同一场景种子→完全相同的场景（配对评测基础）。"""
    from env import scenarios as scen
    s1 = scen.generate_scenarios(env_cfg, "medium", 10, us.TEST_SCENARIO_SEED)
    s2 = scen.generate_scenarios(env_cfg, "medium", 10, us.TEST_SCENARIO_SEED)
    assert s1 == s2
    # 不同namespace→不同场景
    s3 = scen.generate_scenarios(env_cfg, "medium", 10, us.OOD_SCENARIO_SEED)
    assert s1[0]["rng_seed"] != s3[0]["rng_seed"]
    assert not (s1 == s3)


def test_training_env_stream_differs_from_scenarios(env_cfg):
    """训练环境master种子体系与固定场景种子空间结构分离。"""
    from env import scenarios as scen
    test_scn_seeds = set()
    for sc in scen.generate_scenarios(env_cfg, "medium", 50, us.TEST_SCENARIO_SEED):
        test_scn_seeds.add(sc["rng_seed"])
    # 训练env master seeds = train_seed*1000+17（训练命名空间派生）
    train_masters = [s * 1000 + 17 for s in us.TRAIN_SEEDS]
    assert not (set(train_masters) & test_scn_seeds)
    # 训练场景rng_seed由master_rng派生，重放验证前若干个与测试场景不重合
    import numpy as np
    for ms in train_masters:
        rng = np.random.default_rng(ms)
        # 跳过难度/先验采样后取一个episode种子区间（保守检查1024个）
        drawn = set(rng.integers(0, 2**31 - 1, 1024).tolist())
        assert not (drawn & test_scn_seeds)
