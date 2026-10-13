# -*- coding: utf-8 -*-
"""声学模型（开题报告式(6)-(13)）。

物理链路：波束指向→方向性增益式(10)→传播损失式(9)→回波信噪比式(11)
→虚警概率式(12)→Swerling I检测概率式(13)→伯努利观测式(14)。

Level 1（检测概率抽象环境）使用本模块的统计映射；
Level 2（信号级环境）在 env/level2_env.py 中实现式(6)-(8)的LFM/时延/传播波形仿真，
其检测统计量与本模块严格一致（由一致性检验验证）。
"""
from typing import Dict, Optional, Tuple

import numpy as np

# Swerling I 检测概率数值稳定下限
_PD_EPS = 1e-12


def transmission_loss(R: np.ndarray, alpha_f: float) -> np.ndarray:
    """式(9)：TL(R)=20log10(R)+αf·R（球面扩散+吸收衰减，dB）。"""
    R = np.asarray(R, dtype=np.float64)
    return 20.0 * np.log10(R) + alpha_f * R


def beam_gain(delta_theta_deg: np.ndarray, sigma_b_deg: float) -> np.ndarray:
    """式(10)：G(Δθ)=exp(−Δθ²/(2σb²))（线性增益，≤1）。"""
    d = np.asarray(delta_theta_deg, dtype=np.float64)
    return np.exp(-(d ** 2) / (2.0 * sigma_b_deg ** 2))


def beam_sigma_b(n_sectors: int, hpbw_ratio: float = 1.0) -> float:
    """σb由半功率束宽HPBW=2√(2ln2)σb设定为扇区宽度的hpbw_ratio倍（设计决策B3）。"""
    sector_width = 360.0 / n_sectors
    hpbw = hpbw_ratio * sector_width
    return hpbw / (2.0 * np.sqrt(2.0 * np.log(2.0)))


def snr_db_from_equation(
    snr_base_db: float,
    R: np.ndarray,
    delta_theta_deg: np.ndarray,
    sigma_b_deg: float,
    alpha_f: float,
    R_ref: float,
    reverberation_penalty_db: float = 0.0,
) -> np.ndarray:
    """式(11)的参数化实现：SNR_t = SL−2TL(R)+TS+20log10 G(Δθ)−(NL−DI)。

    SL/TS/NL/DI为环境难度参数（表4未给绝对值），合并为校准的基准snr_base_db
    （在参考距离R_ref、波束对准、无混响时）。相对结构严格保留：
      SNR = snr_base − 2·[TL(R) − TL(R_ref)] + 20log10(G(Δθ)) − 混响罚项
    混响按信混比折算并入背景项（开题报告式(11)段说明）。
    """
    R = np.asarray(R, dtype=np.float64)
    tl = transmission_loss(R, alpha_f)
    tl_ref = transmission_loss(np.float64(R_ref), alpha_f)
    g_db = 20.0 * np.log10(np.clip(beam_gain(delta_theta_deg, sigma_b_deg), _PD_EPS, None))
    return snr_base_db - 2.0 * (tl - tl_ref) + g_db - reverberation_penalty_db


def reverberation_penalty(rnr_db: float) -> float:
    """混响/噪声功率比RNR(dB)折算为背景噪声抬升：ΔNL=10log10(1+10^(RNR/10)) dB。"""
    return 10.0 * np.log10(1.0 + 10.0 ** (rnr_db / 10.0))


def pd_swerling1(pfa: float, snr_db: np.ndarray) -> np.ndarray:
    """式(13)：PD = PFA^(1/(1+SNR))，Swerling I（瑞利起伏目标）。
    SNR为线性平均信噪比（输入为dB时转换）。"""
    snr_lin = 10.0 ** (np.asarray(snr_db, dtype=np.float64) / 10.0)
    pfa = np.clip(np.float64(pfa), _PD_EPS, 1.0 - _PD_EPS)
    # ln(PD) = ln(PFA)/(1+SNR)
    pd = np.exp(np.log(pfa) / (1.0 + snr_lin))
    return np.clip(pd, 0.0, 1.0)


def threshold_from_pfa(pfa: float, sigma_n: float = 1.0) -> float:
    """式(12)反解检测门限：V_T = σn·√(−2 ln PFA)。"""
    return sigma_n * np.sqrt(-2.0 * np.log(np.clip(pfa, _PD_EPS, 1.0)))


def pd_from_threshold(pfa: float, snr_lin: float) -> float:
    """式(12)(13)联合的直接计算（标量工具，供校准/测试）。"""
    return float(np.exp(np.log(np.clip(pfa, _PD_EPS, 1.0)) / (1.0 + snr_lin)))


def _mean_pd_at_target_sector(
    snr_base_db: float,
    pfa: float,
    n_sectors: int,
    sigma_b_deg: float,
    range_min: float,
    range_max: float,
    alpha_f: float,
    R_ref: float,
    reverberation_penalty_db: float,
    n_samples: int = 20000,
    seed: int = 0,
) -> float:
    """波束指向目标扇区（目标在扇区内均匀分布）时的平均检测概率。"""
    rng = np.random.default_rng(seed)
    R = rng.uniform(range_min, range_max, n_samples)
    sector_width = 360.0 / n_sectors
    dtheta = rng.uniform(-sector_width / 2.0, sector_width / 2.0, n_samples)
    snr = snr_db_from_equation(snr_base_db, R, dtheta, sigma_b_deg, alpha_f, R_ref,
                               reverberation_penalty_db)
    return float(np.mean(pd_swerling1(pfa, snr)))


def calibrate_snr_base(
    pd_target: float,
    pfa: float,
    n_sectors: int,
    sigma_b_deg: float,
    range_min: float,
    range_max: float,
    alpha_f: float,
    R_ref: float,
    reverberation_penalty_db: float = 0.0,
    snr_jitter_db: float = 0.0,
    seed: int = 0,
    n_samples: int = 20000,
) -> Tuple[float, Dict]:
    """二分法求解snr_base_db，使平均PD收敛到表4目标（设计决策B2：不覆盖物理计算）。

    返回 (snr_base_db, 校准记录)。snr_jitter_db>0时按抖动幅度中心校准。
    """
    lo, hi = -30.0, 40.0  # dB 二分区间

    def mean_pd(base_db: float) -> float:
        return _mean_pd_at_target_sector(
            base_db, pfa, n_sectors, sigma_b_deg, range_min, range_max,
            alpha_f, R_ref, reverberation_penalty_db, n_samples, seed,
        )

    for _ in range(60):
        mid = (lo + hi) / 2.0
        if mean_pd(mid) < pd_target:
            lo = mid
        else:
            hi = mid
    snr_base = (lo + hi) / 2.0
    achieved = mean_pd(snr_base)
    record = {
        "pd_target": pd_target,
        "pfa": pfa,
        "n_sectors": n_sectors,
        "snr_base_db": float(snr_base),
        "achieved_mean_pd": float(achieved),
        "abs_error": abs(achieved - pd_target),
        "snr_jitter_db": snr_jitter_db,
    }
    return float(snr_base), record


def pd_distribution_at_target_sector(
    snr_base_db: float,
    pfa: float,
    n_sectors: int,
    sigma_b_deg: float,
    range_min: float,
    range_max: float,
    alpha_f: float,
    R_ref: float,
    reverberation_penalty_db: float = 0.0,
    n_samples: int = 50000,
    seed: int = 1,
) -> Dict:
    """记录实际PD分布（均值/std/分位数）——用户要求：记录实际PD分布。"""
    rng = np.random.default_rng(seed)
    R = rng.uniform(range_min, range_max, n_samples)
    sector_width = 360.0 / n_sectors
    dtheta = rng.uniform(-sector_width / 2.0, sector_width / 2.0, n_samples)
    snr = snr_db_from_equation(snr_base_db, R, dtheta, sigma_b_deg, alpha_f, R_ref,
                               reverberation_penalty_db)
    pd = pd_swerling1(pfa, snr)
    return {
        "mean": float(np.mean(pd)),
        "std": float(np.std(pd)),
        "q05": float(np.quantile(pd, 0.05)),
        "q50": float(np.quantile(pd, 0.50)),
        "q95": float(np.quantile(pd, 0.95)),
        "snr_db_mean": float(np.mean(snr)),
        "snr_db_std": float(np.std(snr)),
    }
