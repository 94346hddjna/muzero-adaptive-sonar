# -*- coding: utf-8 -*-
"""评价指标单元测试（式31-36）。"""
import numpy as np
import pytest

from utils import metrics as mt
from utils import stats as ust


def _outcomes(n=100, sr=0.8, steps_succ=5, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        succ = rng.random() < sr
        out.append({
            "scenario_index": i, "success": bool(succ),
            "steps": int(steps_succ if succ else 20),
            "return": 10.0 if succ else -20.0,
            "energy": float(steps_succ if succ else 20.0),
            "n_negative_emissions": 10 if not succ else 3,
            "n_false_alarms": 1 if not succ else 0,
        })
    return out


def test_success_rate_formula31():
    assert mt.success_rate(_outcomes(100, sr=0.8, seed=1)) == pytest.approx(
        sum(o["success"] for o in _outcomes(100, sr=0.8, seed=1)) / 100)


def test_ads_formula32_only_successful():
    out = _outcomes(100, sr=0.8, seed=2)
    succ_steps = [o["steps"] for o in out if o["success"]]
    assert mt.average_detection_steps(out) == pytest.approx(np.mean(succ_steps))
    # 全失败→None
    assert mt.average_detection_steps(_outcomes(10, sr=0.0)) is None


def test_edc_formula33_censoring():
    """C_i = T_D·1{s=1} + (T_max+C_pen)·1{s=0}。"""
    out = [{"success": True, "steps": 4},
           {"success": False, "steps": 20}]
    edc = mt.censored_expected_detection_cost(out, c_pen=10.0, t_max=20)
    assert edc == pytest.approx((4 + 30) / 2)


def test_far_formula34():
    out = [{"n_negative_emissions": 100, "n_false_alarms": 5}] * 10
    assert mt.false_alarm_rate(out) == pytest.approx(0.05)
    assert mt.false_alarm_rate([{"n_negative_emissions": 0, "n_false_alarms": 0}]) is None


def test_drop_formula35():
    assert mt.relative_drop(0.9, 0.45) == pytest.approx(0.5)
    assert mt.relative_drop(0.9, 0.9) == pytest.approx(0.0)


def test_kl_divergence_formula36():
    # 独热 vs 独热：D_KL=0
    p = np.array([0, 0, 1, 0])
    assert mt.kl_divergence(p, p) == pytest.approx(0.0, abs=1e-9)
    # 均匀 vs 独热：D_KL = ln(N)
    N = 4
    q = np.full(N, 1 / N)
    p = np.array([0, 0, 1, 0])
    assert mt.kl_divergence(p, q) == pytest.approx(np.log(N), rel=1e-6)


def test_expected_return_and_energy():
    out = _outcomes(10, sr=0.5, seed=3)
    assert mt.expected_return(out) == pytest.approx(np.mean([o["return"] for o in out]))
    e = mt.energy_per_step(out)
    assert np.isfinite(e) and e > 0


def test_aggregate_metrics_complete():
    m = mt.aggregate_metrics(_outcomes(50, sr=0.7, seed=4), c_pen=10.0, t_max=20)
    for key in ["SR", "ADS", "EDC", "FAR", "ExpectedReturn", "EnergyPerStep",
                "AvgSteps", "episodes", "SR_ci95_halfwidth"]:
        assert key in m
    assert 0 <= m["SR"] <= 1


def test_zero_episode_raises():
    with pytest.raises(ValueError):
        mt.aggregate_metrics([], c_pen=10, t_max=20)


def test_sample_efficiency():
    curve = [{"env_steps": 100, "success_rate": 0.3},
             {"env_steps": 500, "success_rate": 0.91}]
    assert mt.sample_efficiency(curve, 0.9) == 500
    assert mt.sample_efficiency(curve, 0.95) is None  # 未达到→如实None


def test_paired_test_and_effect_size():
    rng = np.random.default_rng(0)
    x = rng.binomial(1, 0.8, 500).astype(float)
    y = rng.binomial(1, 0.5, 500).astype(float)
    res = ust.paired_test(x, y)
    assert res["n_pairs"] == 500
    assert res["p_value"] < 0.05
    assert res["effect_size_name"] in ("cohens_dz", "r_rank_biserial")
    assert abs(res["effect_size"]) > 0.1


def test_holm_bonferroni():
    holm = ust.holm_bonferroni([0.01, 0.04, 0.03])
    # 排序后0.01×3=0.03; 0.03×2=0.06; 0.04×1=0.04→max单调化0.06
    assert holm[0]["p_holm"] == pytest.approx(0.03)
    assert holm[2]["p_holm"] == pytest.approx(0.06)
    assert holm[1]["p_holm"] == pytest.approx(0.06)
