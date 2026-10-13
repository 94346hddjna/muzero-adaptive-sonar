# -*- coding: utf-8 -*-
"""统计分析（开题报告第8章）：配对设计、分布检验、配对t/Wilcoxon、效应量、多重比较校正。"""
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats as sps


def describe(x: np.ndarray) -> Dict:
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    mean = float(np.mean(x))
    std = float(np.std(x, ddof=1)) if n > 1 else 0.0
    if n > 1:
        se = std / np.sqrt(n)
        ci = (mean - 1.96 * se, mean + 1.96 * se)
    else:
        ci = (mean, mean)
    return {"n": n, "mean": mean, "std": std, "ci95_low": ci[0], "ci95_high": ci[1]}


def paired_test(x: np.ndarray, y: np.ndarray, alpha: float = 0.05) -> Dict:
    """配对比较 x vs y（同场景配对）。自动检查差值分布正态性（Shapiro-Wilk），
    正态→配对t检验；否则→Wilcoxon符号秩检验。同时报告效应量与95%CI。"""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    assert len(x) == len(y) and len(x) > 0, "paired test requires equal-length non-empty samples"
    d = x - y
    n = len(d)
    result = {
        "n_pairs": n,
        "mean_diff": float(np.mean(d)),
        "mean_x": float(np.mean(x)),
        "mean_y": float(np.mean(y)),
    }
    # 正态性检查（差值）
    if n >= 3 and np.any(d != d[0]):
        try:
            sh_stat, sh_p = sps.shapiro(d)
        except Exception:
            sh_stat, sh_p = float("nan"), 0.0
    else:
        sh_stat, sh_p = float("nan"), 0.0
    result["shapiro_p"] = float(sh_p) if not np.isnan(sh_p) else None
    normal_enough = (result["shapiro_p"] is not None and result["shapiro_p"] > 0.05)
    if n >= 3 and np.any(d != d[0]):
        if normal_enough or n >= 30:
            t_stat, p_val = sps.ttest_rel(x, y)
            test_name, test_stat = "paired_t_test", t_stat
            sd = float(np.std(d, ddof=1))
            dz = float(np.mean(d) / sd) if sd > 0 else 0.0  # paired Cohen's d (dz)
            effect = {"name": "cohens_dz", "value": dz}
            se = sd / np.sqrt(n)
            ci = (float(np.mean(d) - 1.96 * se), float(np.mean(d) + 1.96 * se))
        else:
            try:
                w_stat, p_val = sps.wilcoxon(x, y, zero_method="wilcox")
            except Exception:
                w_stat, p_val = sps.wilcoxon(x, y, zero_method="pratt")
            test_name, test_stat = "wilcoxon_signed_rank", w_stat
            mu_w = n * (n + 1) / 4.0
            sigma_w = np.sqrt(n * (n + 1) * (2 * n + 1) / 24.0)
            z = (w_stat - mu_w) / sigma_w if sigma_w > 0 else 0.0
            r_rb = abs(z) / np.sqrt(n)  # matched-pairs rank-biserial近似
            effect = {"name": "r_rank_biserial", "value": float(r_rb)}
            ci = (None, None)
    else:
        p_val, test_name, effect = 1.0, "not_testable_constant_diff", {"name": "none", "value": 0.0}
        test_stat, ci = None, (None, None)
    result.update({
        "test": test_name,
        "statistic": None if test_stat is None else float(test_stat),
        "p_value": float(p_val),
        "significant_at_0.05": bool(p_val < alpha),
        "effect_size_name": effect["name"],
        "effect_size": effect["value"],
        "diff_ci95_low": ci[0],
        "diff_ci95_high": ci[1],
    })
    return result


def holm_bonferroni(p_values: List[float], alpha: float = 0.05) -> List[Dict]:
    """Holm-Bonferroni多重比较校正。返回逐条(排序位、校正p、是否显著)。"""
    m = len(p_values)
    idx = np.argsort(p_values)
    corrected = [0.0] * m
    prev = 0.0
    for rank, i in enumerate(idx):
        adj = (m - rank) * p_values[i]
        adj = max(adj, prev)  # 单调性
        prev = adj
        corrected[i] = min(1.0, adj)
    return [
        {"index": i, "p_raw": float(p_values[i]), "p_holm": float(corrected[i]),
         "significant": bool(corrected[i] < alpha)}
        for i in range(m)
    ]


def seed_level_summary(per_seed_values: List[float]) -> Dict:
    """跨种子聚合：mean±std与95%CI（种子数≥2时）。"""
    arr = np.asarray(per_seed_values, dtype=np.float64)
    d = describe(arr)
    return {"n_seeds": d["n"], "mean": d["mean"], "std": d["std"],
            "ci95_low": d["ci95_low"], "ci95_high": d["ci95_high"]}
