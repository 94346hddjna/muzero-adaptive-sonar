# -*- coding: utf-8 -*-
"""评价指标（开题报告式(31)-(36)）。

设评测回合数为N（N_succ成功）：
- SR        式(31) 成功探测概率
- ADS       式(32) 平均发现步数（仅成功回合）
- EDC       式(33) 删失期望探测代价 C_i = T_D·1{s=1} + (T_max+C_pen)·1{s=0}
- FAR       式(34) 虚警率 N_FA/N_neg（目标不在覆盖扇区内的发射中检出为1的比例）
- Drop      式(35) 相对性能退化
- D_KL      式(36) 先验偏差
- 期望回报、能量/步成本、样本效率、决策时延
"""
import math
from typing import Dict, List, Optional

import numpy as np


def success_rate(outcomes: List[Dict]) -> float:
    n = len(outcomes)
    if n == 0:
        raise ValueError("empty outcomes")
    return sum(1 for o in outcomes if o["success"]) / n


def average_detection_steps(outcomes: List[Dict]) -> Optional[float]:
    succ = [o["steps"] for o in outcomes if o["success"]]
    if not succ:
        return None
    return float(np.mean(succ))


def censored_expected_detection_cost(outcomes: List[Dict], c_pen: float, t_max: int) -> float:
    """式(33)。"""
    total = 0.0
    for o in outcomes:
        if o["success"]:
            total += o["steps"]
        else:
            total += t_max + c_pen
    return total / len(outcomes)


def false_alarm_rate(outcomes: List[Dict]) -> Optional[float]:
    """式(34)：N_neg = 目标不在覆盖扇区内的发射总次数；N_FA = 其中检出为1的次数。"""
    n_neg = sum(o.get("n_negative_emissions", 0) for o in outcomes)
    n_fa = sum(o.get("n_false_alarms", 0) for o in outcomes)
    if n_neg == 0:
        return None
    return n_fa / n_neg


def expected_return(outcomes: List[Dict]) -> float:
    return float(np.mean([o["return"] for o in outcomes]))


def energy_per_step(outcomes: List[Dict]) -> float:
    """能量/步：核心实验各动作成本相同（C(a)=1），能量与发射步数成正比（开题报告明确说明）。"""
    energies = [o.get("energy", o["steps"] * 1.0) for o in outcomes]
    steps = [o["steps"] for o in outcomes]
    total_e, total_s = sum(energies), sum(steps)
    return total_e / total_s if total_s > 0 else float("nan")


def relative_drop(p_nom: float, p_d: float) -> float:
    """式(35)。"""
    return (p_nom - p_d) / p_nom


def kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    """式(36)：D_KL(P‖Q)=Σ P(i)ln(P(i)/Q(i))。P为真实目标分布（单目标为独热）。"""
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    q = np.clip(q, eps, None)
    p = np.clip(p, eps, None)
    mask = p > 0
    return float(np.sum(p[mask] * np.log(p[mask] / q[mask])))


def belief_entropy(b: np.ndarray) -> float:
    b = np.asarray(b, dtype=np.float64)
    b = np.clip(b, 1e-12, None)
    return float(-np.sum(b * np.log(b)))


def aggregate_metrics(outcomes: List[Dict], c_pen: float, t_max: int) -> Dict:
    """单个(算法,难度,种子)条件下的全套指标。outcomes为逐回合记录（含success/steps/return/...）。"""
    n = len(outcomes)
    if n == 0:
        raise ValueError("zero-episode evaluation forbidden (quality check)")
    m = {
        "episodes": n,
        "SR": success_rate(outcomes),
        "ADS": average_detection_steps(outcomes),
        "EDC": censored_expected_detection_cost(outcomes, c_pen, t_max),
        "FAR": false_alarm_rate(outcomes),
        "ExpectedReturn": expected_return(outcomes),
        "EnergyPerStep": energy_per_step(outcomes),
        "AvgSteps": float(np.mean([o["steps"] for o in outcomes])),
        "MeanDecisionLatencyMs": _mean_latency(outcomes),
    }
    sr_se = math.sqrt(m["SR"] * (1 - m["SR"]) / n)
    m["SR_stderr"] = sr_se
    m["SR_ci95_halfwidth"] = 1.96 * sr_se
    return m


def _mean_latency(outcomes: List[Dict]) -> Optional[float]:
    lat = [o.get("mean_decision_latency_ms") for o in outcomes if o.get("mean_decision_latency_ms") is not None]
    if not lat:
        return None
    return float(np.mean(lat))


def sample_efficiency(step_to_sr: List[Dict], sr_threshold: float) -> Optional[int]:
    """达到指定成功率阈值所需环境交互步数；未达到返回None（如实报告）。"""
    for rec in step_to_sr:
        if rec["success_rate"] >= sr_threshold:
            return rec["env_steps"]
    return None
