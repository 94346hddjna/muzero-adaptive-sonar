# -*- coding: utf-8 -*-
"""随机种子系统：训练/验证/测试/OOD四命名空间严格隔离（开题报告第十一章）。

种子命名空间（互不相交）：
- train      : 网络初始化与探索种子 [101..105]
- train-env  : 训练环境采样流 TR-<train_seed>
- validation : 验证场景 VS-20250901
- test       : 测试场景 TS-30303（锁定，不得用于调参）
- ood        : OOD场景 OS-40404
"""
import numpy as np
import torch

TRAIN_SEEDS = [101, 102, 103, 104, 105]
VALIDATION_SCENARIO_SEED = 20250901
TEST_SCENARIO_SEED = 30303
OOD_SCENARIO_SEED = 40404


def all_namespace_seeds():
    """返回所有命名空间占用的整数种子集合（用于隔离性测试）。"""
    s = set()
    s.update(TRAIN_SEEDS)                       # train
    s.update(range(1000, 1000 + 500))           # train-env 派生流区间 TR-<seed>→seed*1000+i
    s.add(VALIDATION_SCENARIO_SEED)
    s.update(range(VALIDATION_SCENARIO_SEED, VALIDATION_SCENARIO_SEED + 500))
    s.add(TEST_SCENARIO_SEED)
    s.update(range(TEST_SCENARIO_SEED, TEST_SCENARIO_SEED + 500))
    s.add(OOD_SCENARIO_SEED)
    s.update(range(OOD_SCENARIO_SEED, OOD_SCENARIO_SEED + 500))
    return s


def check_namespace_isolation() -> dict:
    """验证四个命名空间互不相交。"""
    train = set(TRAIN_SEEDS) | set(range(1000, 1000 + 500))
    val = {VALIDATION_SCENARIO_SEED} | set(range(VALIDATION_SCENARIO_SEED, VALIDATION_SCENARIO_SEED + 500))
    test = {TEST_SCENARIO_SEED} | set(range(TEST_SCENARIO_SEED, TEST_SCENARIO_SEED + 500))
    ood = {OOD_SCENARIO_SEED} | set(range(OOD_SCENARIO_SEED, OOD_SCENARIO_SEED + 500))
    pairs = {
        "train∩validation": len(train & val),
        "train∩test": len(train & test),
        "train∩ood": len(train & ood),
        "validation∩test": len(val & test),
        "validation∩ood": len(val & ood),
        "test∩ood": len(test & ood),
    }
    return {"ok": all(v == 0 for v in pairs.values()), "overlaps": pairs}


def env_episode_rng(namespace_seed: int, episode_index: int) -> np.random.Generator:
    """每个episode独立的RNG：保证同一场景下不同算法面对同一噪声流（配对评测）。"""
    return np.random.default_rng(namespace_seed * 100003 + episode_index)


def scenario_rng(namespace_seed: int, episode_index: int) -> np.random.Generator:
    """场景定义RNG（目标位置/距离/先验），与噪声流分离。"""
    return np.random.default_rng(namespace_seed * 100003 + episode_index + 500000000)


def set_global_seeds(seed: int):
    """设置python/numpy/torch全局种子（训练网络初始化）。"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
