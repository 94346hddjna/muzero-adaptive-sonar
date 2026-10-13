# -*- coding: utf-8 -*-
"""检测模型单元测试（式6-13）：TL/波束/SNR/PD/门限/Level2统计一致性。"""
import numpy as np
import pytest

from env import acoustic_model as am
from env import level1_env, level2_env


def test_transmission_loss_formula():
    # 式(9)：TL=20log10(R)+αf·R
    R = 1000.0
    assert abs(am.transmission_loss(np.array([R]), 0.0)[0] - 60.0) < 1e-9
    tl = am.transmission_loss(np.array([R]), 0.001)[0]
    assert abs(tl - 60.0 - 1.0) < 1e-9


def test_beam_gain_formula():
    # 式(10)：G=exp(−Δθ²/2σb²)
    assert am.beam_gain(np.array([0.0]), 10.0)[0] == 1.0
    g = am.beam_gain(np.array([10.0]), 10.0)[0]
    assert abs(g - np.exp(-0.5)) < 1e-9
    assert am.beam_gain(np.array([100.0]), 10.0)[0] < 1e-10


def test_beam_hpbw_matches_sector(env_cfg):
    N = 16
    sigma = am.beam_sigma_b(N, env_cfg["beam_hpbw_to_sector_width"])
    hpbw = 2 * np.sqrt(2 * np.log(2)) * sigma
    assert abs(hpbw - 360.0 / N) < 1e-6


def test_pd_swerling1_formula():
    # 式(13)：PD=PFA^(1/(1+SNR))
    pfa = 0.05
    # SNR→∞：PD→1
    assert am.pd_swerling1(pfa, np.array([60.0]))[0] > 0.999
    # SNR=0dB：PD=PFA^(1/2)
    assert abs(am.pd_swerling1(pfa, np.array([0.0]))[0] - np.sqrt(pfa)) < 1e-9
    # 单调性：SNR↑→PD↑
    snrs = np.linspace(-10, 20, 50)
    pds = am.pd_swerling1(pfa, snrs)
    assert np.all(np.diff(pds) > 0)
    # SNR→−∞：PD→PFA
    assert abs(am.pd_swerling1(pfa, np.array([-60.0]))[0] - pfa) < 1e-6


def test_threshold_from_pfa():
    # 式(12)：V_T=σn√(−2ln PFA)
    for pfa in [0.01, 0.05, 0.1]:
        vt = am.threshold_from_pfa(pfa, sigma_n=1.0)
        assert abs(np.exp(-vt ** 2 / 2) - pfa) < 1e-9


def test_snr_equation_structure(env_cfg):
    # 式(11)：相对结构（距离↑→SNR↓；波束偏差↑→SNR↓；混响↑→SNR↓）
    base = 20.0
    snr_near = float(am.snr_db_from_equation(base, 2000.0, 0.0, 10.0, 0.001, 3000.0, 0.0))
    snr_far = float(am.snr_db_from_equation(base, 4000.0, 0.0, 10.0, 0.001, 3000.0, 0.0))
    assert snr_near > snr_far
    snr_on = float(am.snr_db_from_equation(base, 3000.0, 0.0, 10.0, 0.001, 3000.0, 0.0))
    snr_off = float(am.snr_db_from_equation(base, 3000.0, 15.0, 10.0, 0.001, 3000.0, 0.0))
    assert snr_on > snr_off
    pen = am.reverberation_penalty(6.0)  # RNR 6dB → 背景抬升
    snr_rr = float(am.snr_db_from_equation(base, 3000.0, 0.0, 10.0, 0.001, 3000.0, pen))
    assert snr_on - snr_rr == pytest.approx(pen)


def test_calibration_matches_targets(env_cfg):
    for d in ["easy", "medium", "hard"]:
        p = level1_env.difficulty_params(env_cfg, d)
        assert abs(p["pd_distribution"]["mean"] - p["calibration"]["pd_target"]) < 0.02


def test_level2_statistics_consistency(env_cfg, l2cfg):
    """Level2波形级检测统计与Level1一致（式12-13实现验证）。"""
    l2 = level2_env.SonarLevel2VecEnv(4, env_cfg, t_max=20, master_seed=5,
                                      level2_cfg=l2cfg)
    p = level1_env.difficulty_params(env_cfg, "medium")
    n = 400
    fa = 0
    for i in range(n):
        rng = np.random.default_rng(1200 + i)
        sc = level1_env.make_scenario_from_params(p, env_cfg, rng, 1300 + i, scenario_index=i)
        sc["target"] = 8
        l2.reset([sc])
        l2.step(np.array([0]))
        fa += int(l2.prev_det[0])
    assert abs(fa / n - 0.05) < 0.05  # ±0.05 (400试验, 2σ≈0.022)
    det = 0
    pds = []
    for i in range(n):
        rng = np.random.default_rng(1500 + i)
        sc = level1_env.make_scenario_from_params(p, env_cfg, rng, 1600 + i, scenario_index=i)
        pds.append(sc["pd"])
        l2.reset([sc])
        l2.step(np.array([sc["target"]]))
        det += int(l2.prev_det[0])
    assert abs(det / n - float(np.mean(pds))) < 0.06
