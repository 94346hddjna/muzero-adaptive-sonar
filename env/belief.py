# -*- coding: utf-8 -*-
"""贝叶斯信念更新（开题报告式(4)(5)）。

式(4)：b_t(i)=P(x=i|H_t)，Σb_t(i)=1
式(5)：b_{t+1}(i) = P(o_{t+1}|x=i,a_t)b_t(i) / Σ_j P(o_{t+1}|x=j,a_t)b_t(j)

观测概率模型（式(14)）：
  P(o=1|x=a)=P_D   P(o=0|x=a)=1−P_D
  P(o=1|x≠a)=P_FA  P(o=0|x≠a)=1−P_FA

一次未检测到目标不把该扇区概率置0，而按漏检概率(1−P_D)衰减（式(5)的贝叶斯后验）。
初始信念支持：uniform / correct(concentrated) / moderate / biased / adversarial。
"""
from typing import Optional

import numpy as np

_EPS = 1e-15


def belief_entropy(b: np.ndarray) -> float:
    """信念熵H(b)（塑形奖励Φ与先验偏差分析使用）。"""
    b = np.clip(np.asarray(b, dtype=np.float64), 1e-12, None)
    return float(-np.sum(b * np.log(b)))


def circular_distance(i: np.ndarray, j: int, n: int) -> np.ndarray:
    """环形扇区距离 d(i,j)=min(|i−j|, N−|i−j|)。"""
    d = np.abs(np.asarray(i) - j)
    return np.minimum(d, n - d)


def make_initial_belief(
    n_sectors: int,
    prior_type: str,
    target_sector: int,
    rng: Optional[np.random.Generator] = None,
    sigma_sectors: Optional[float] = None,
) -> np.ndarray:
    """生成初始信念b_0（式(4)在t=0的特例）。

    prior_type:
      uniform     均匀 1/N
      correct     集中于真目标扇区（"较集中"，Easy）
      moderate    弥散但中心为真目标（"中等弥散"，Medium）
      biased      集中于随机错误扇区（"存在偏差"，Hard）
      adversarial 集中于距目标最远的扇区（对抗性错误先验）
    sigma_sectors为环形高斯先验宽度（扇区数），None时用默认。
    """
    rng = rng or np.random.default_rng()
    sectors = np.arange(n_sectors)
    if prior_type == "uniform":
        return np.full(n_sectors, 1.0 / n_sectors)
    if prior_type == "correct":
        sigma = sigma_sectors if sigma_sectors is not None else 1.0
        center = int(target_sector)
    elif prior_type == "moderate":
        sigma = sigma_sectors if sigma_sectors is not None else 2.5
        center = int(target_sector)
    elif prior_type == "biased":
        sigma = sigma_sectors if sigma_sectors is not None else 1.5
        # 随机错误扇区：与目标至少间隔N/4
        min_gap = max(1, n_sectors // 4)
        candidates = [i for i in range(n_sectors)
                      if circular_distance(np.int64(i), target_sector, n_sectors) >= min_gap]
        center = int(rng.choice(candidates)) if candidates else int((target_sector + n_sectors // 2) % n_sectors)
    elif prior_type == "adversarial":
        sigma = sigma_sectors if sigma_sectors is not None else 1.0
        center = int((target_sector + n_sectors // 2) % n_sectors)  # 最远扇区
    else:
        raise ValueError(f"unknown prior_type: {prior_type}")
    d = circular_distance(sectors, center, n_sectors)
    b = np.exp(-(d.astype(np.float64) ** 2) / (2.0 * sigma ** 2))
    b += 1e-8  # 数值兜底，避免全零
    return b / b.sum()


def bayes_update(
    belief: np.ndarray,
    action: int,
    detection: int,
    pd: float,
    pfa: float,
) -> np.ndarray:
    """式(5)贝叶斯后验更新（单个环境）。

    似然（式(14)）：
      detection=1: L(i)=P_D if i==a else P_FA
      detection=0: L(i)=1−P_D if i==a else 1−P_FA
    后验 ∝ L⊙b，归一化保证Σ=1。
    """
    b = np.asarray(belief, dtype=np.float64)
    n = b.shape[0]
    if detection:
        like = np.full(n, pfa)
        like[action] = pd
    else:
        like = np.full(n, 1.0 - pfa)
        like[action] = 1.0 - pd
    post = like * b
    s = post.sum()
    if s <= _EPS:
        # 极端数值情况（如pd=0且detection=1且belief集中于a）——回退均匀防止NaN
        return np.full(n, 1.0 / n)
    return post / s


def bayes_update_batch(
    beliefs: np.ndarray,      # [B, N]
    actions: np.ndarray,      # [B]
    detections: np.ndarray,   # [B] ∈{0,1}
    pds: np.ndarray,          # [B]
    pfas: np.ndarray,         # [B]
) -> np.ndarray:
    """向量化式(5)（B个并行环境，各自独立更新）。"""
    B, N = beliefs.shape
    idx = np.arange(B)
    if detections.ndim == 0:
        detections = np.broadcast_to(detections, (B,))
    # 似然矩阵 [B, N]
    if np.isscalar(pds) or (isinstance(pds, np.ndarray) and pds.ndim == 0):
        pds = np.full(B, float(pds))
    if np.isscalar(pfas) or (isinstance(pfas, np.ndarray) and pfas.ndim == 0):
        pfas = np.full(B, float(pfas))
    det = detections.astype(bool)
    pd_b = np.where(det, pds, 1.0 - pds)          # [B] 指向扇区似然
    pfa_b = np.where(det, pfas, 1.0 - pfas)       # [B] 非指向扇区似然
    like = np.broadcast_to(pfa_b[:, None], (B, N)).copy()
    like[idx, actions] = pd_b
    post = like * beliefs
    s = post.sum(axis=1, keepdims=True)
    bad = s.squeeze(1) <= _EPS
    if np.any(bad):
        post[bad] = 1.0 / N
        s = post.sum(axis=1, keepdims=True)
    return post / s
