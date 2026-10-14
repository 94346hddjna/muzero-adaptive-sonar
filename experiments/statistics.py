# -*- coding: utf-8 -*-
"""统计分析（开题报告第8章、用户第20章）。

从results/raw/逐回合记录计算：
- 配对检验（同场景配对；Shapiro正态性检查→配对t或Wilcoxon，α=0.05）
- 效应量（paired Cohen's dz / rank-biserial r）+ 95%CI
- Holm-Bonferroni多重比较校正（每个难度×指标的比较族内）
- 跨种子聚合 mean±std+CI
- OOD相对退化Drop(d)（式35）
输出：results/statistics/statistical_tests.csv、effect_sizes.csv等
"""
import csv
from pathlib import Path
from typing import Dict, List

import numpy as np

from utils import stats as ust
from utils import metrics as umt
from utils.io_utils import read_jsonl, write_csv

PAIR_FAMILIES = [
    ("muzero", "sequential"), ("muzero", "random"), ("muzero", "belief_greedy"),
    ("muzero", "dqn"), ("muzero", "ppo"),
]


def _parse_stem(stem: str):
    """解析<algo>_<variant>_seed<k>[_suffix]（algo可含下划线）。"""
    import re
    m = re.match(r"(.+?)_(\w+)_seed(\d+)(?:_(\w+))?$", stem)
    if not m:
        return None
    return {"algo": m.group(1), "variant": m.group(2),
            "seed": int(m.group(3)), "suffix": m.group(4) or ""}


def _load_condition(raw_dir: Path, condition: str) -> Dict[str, Dict[int, List[Dict]]]:
    """读取condition下全部(algorithm, seed)的逐回合记录（主variant、无suffix）。"""
    out: Dict[str, Dict[int, List[Dict]]] = {}
    cdir = raw_dir / condition
    if not cdir.exists():
        return out
    for f in sorted(cdir.glob("*.jsonl")):
        p = _parse_stem(f.stem)
        if not p or p["suffix"] or p["variant"] != "main":
            continue
        out.setdefault(p["algo"], {})[p["seed"]] = read_jsonl(f)
    return out


def _load_condition_full(raw_dir: Path, condition: str):
    """读取全部文件（含消融variant/suffix），返回key=algo|variant|suffix。"""
    out = {}
    cdir = raw_dir / condition
    if not cdir.exists():
        return out
    for f in sorted(cdir.glob("*.jsonl")):
        p = _parse_stem(f.stem)
        if not p:
            continue
        key = f"{p['algo']}|{p['variant']}|{p['suffix']}"
        out.setdefault(key, {})[p["seed"]] = read_jsonl(f)
    return out


def _per_scenario_metric(records: List[Dict], metric: str) -> np.ndarray:
    """逐场景指标向量（按scenario_index对齐）。"""
    byidx = {int(r["scenario_index"]): r for r in records}
    n = max(byidx) + 1 if byidx else 0
    vals = np.full(n, np.nan)
    for i, r in byidx.items():
        vals[i] = float(1.0 if r["success"] else 0.0) if metric == "SR" else float(r[metric])
    return vals


def run_statistics(results_root: Path, stage: str = "main", smoke: bool = False) -> Dict:
    raw_dir = results_root / "raw" / stage
    stats_dir = results_root / "statistics"
    stats_dir.mkdir(parents=True, exist_ok=True)
    conditions = sorted([d.name for d in raw_dir.iterdir() if d.is_dir()]) if raw_dir.exists() else []
    test_rows: List[Dict] = []
    effect_rows: List[Dict] = []
    for cond in conditions:
        data = _load_condition(raw_dir, cond)
        if not data:
            continue
        for metric in ["SR"]:
            # 各算法的跨种子平均逐场景值（配对向量）
            vecs = {}
            for algo, seeds in data.items():
                per_seed = [_per_scenario_metric(recs, metric) for recs in seeds.values()]
                n = max(len(v) for v in per_seed)
                mat = np.full((len(per_seed), n), np.nan)
                for i, v in enumerate(per_seed):
                    mat[i, :len(v)] = v
                vec = np.nanmean(mat, axis=0)
                vecs[algo] = vec
            # 配对比较族（Holm族=本condition×metric的全部对）
            family_rows = []
            for a, b in PAIR_FAMILIES:
                if a not in vecs or b not in vecs:
                    continue
                va, vb = vecs[a], vecs[b]
                n = min(len(va), len(vb))
                if n < 5:
                    continue
                x, y = va[:n], vb[:n]
                if np.all(np.isnan(x)) or np.all(np.isnan(y)):
                    continue
                res = ust.paired_test(x, y)
                row = {
                    "stage": stage, "condition": cond, "metric": metric,
                    "algo_a": a, "algo_b": b, "n_pairs": res["n_pairs"],
                    "mean_a": res["mean_x"], "mean_b": res["mean_y"],
                    "mean_diff": res["mean_diff"],
                    "test": res["test"], "statistic": res["statistic"],
                    "p_value": res["p_value"],
                    "shapiro_p_on_diff": res["shapiro_p"],
                    "diff_ci95_low": res["diff_ci95_low"],
                    "diff_ci95_high": res["diff_ci95_high"],
                }
                family_rows.append((row, res))
            if not family_rows:
                continue
            pvals = [r["p_value"] for r, _ in family_rows]
            holm = ust.holm_bonferroni(pvals)
            for (row, res), h in zip(family_rows, holm):
                row["p_holm"] = h["p_holm"]
                row["significant_holm_0.05"] = h["significant"]
                row["effect_size_name"] = res["effect_size_name"]
                row["effect_size"] = res["effect_size"]
                test_rows.append(row)
                effect_rows.append({
                    "stage": stage, "condition": cond, "metric": metric,
                    "comparison": f"{row['algo_a']}_vs_{row['algo_b']}",
                    "effect_size_name": res["effect_size_name"],
                    "effect_size": res["effect_size"],
                    "mean_diff": res["mean_diff"],
                    "ci95_low": row["diff_ci95_low"], "ci95_high": row["diff_ci95_high"],
                    "n_pairs": row["n_pairs"],
                })
    # 消融阶段：MuZero主模型 vs 各消融配置（对应H2/H3/H4/RQ2/RQ4/RQ5）
    if stage == "ablation":
        main_raw_dir = results_root / "raw" / "main"
        for cond in conditions:
            full = _load_condition_full(raw_dir, cond)
            base_key = "muzero|main|"
            if base_key not in full:
                # 主模型基准取自main阶段（同一固定场景，可按scenario_index配对）
                base_seeds = _load_condition(main_raw_dir, cond).get("muzero")
                if not base_seeds:
                    continue
                vec_base = _mean_over_seeds(base_seeds)
            else:
                vec_base = _mean_over_seeds(full[base_key])
            family = []
            for key in full:
                if key == base_key or not key.startswith("muzero|"):
                    continue
                vec = _mean_over_seeds(full[key])
                if vec is None:
                    continue
                n = min(len(vec_base), len(vec))
                if n < 5:
                    continue
                res = ust.paired_test(vec_base[:n], vec[:n])
                label = key.split("|", 1)[1]
                row = {
                    "stage": stage, "condition": cond, "metric": "SR",
                    "algo_a": "muzero(full)", "algo_b": f"muzero({label})",
                    "n_pairs": res["n_pairs"],
                    "mean_a": res["mean_x"], "mean_b": res["mean_y"],
                    "mean_diff": res["mean_diff"],
                    "test": res["test"], "statistic": res["statistic"],
                    "p_value": res["p_value"],
                    "shapiro_p_on_diff": res["shapiro_p"],
                    "diff_ci95_low": res["diff_ci95_low"],
                    "diff_ci95_high": res["diff_ci95_high"],
                }
                family.append((row, res))
            if not family:
                continue
            holm = ust.holm_bonferroni([r["p_value"] for r, _ in family])
            for (row, res), h in zip(family, holm):
                row["p_holm"] = h["p_holm"]
                row["significant_holm_0.05"] = h["significant"]
                row["effect_size_name"] = res["effect_size_name"]
                row["effect_size"] = res["effect_size"]
                test_rows.append(row)
                effect_rows.append({
                    "stage": stage, "condition": cond, "metric": "SR",
                    "comparison": f"{row['algo_a']}_vs_{row['algo_b']}",
                    "effect_size_name": res["effect_size_name"],
                    "effect_size": res["effect_size"],
                    "mean_diff": res["mean_diff"],
                    "ci95_low": row["diff_ci95_low"], "ci95_high": row["diff_ci95_high"],
                    "n_pairs": row["n_pairs"],
                })
    # 每阶段独立落盘（修复：原实现各阶段覆盖同一文件，仅保留最后阶段）
    write_csv(stats_dir / f"statistical_tests_{stage}.csv", test_rows)
    write_csv(stats_dir / f"effect_sizes_{stage}.csv", effect_rows)
    # 跨种子聚合
    _seed_level_summary(results_root, stage)
    # OOD Drop（式35）
    if stage == "ood":
        _ood_drop(results_root)
    return {"n_tests": len(test_rows), "n_effects": len(effect_rows)}


def _mean_over_seeds(seeds: Dict[int, List[Dict]]) -> "np.ndarray | None":
    """跨种子的逐场景平均向量。"""
    per_seed = [_per_scenario_metric(recs, "SR") for recs in seeds.values()]
    per_seed = [v for v in per_seed if len(v)]
    if not per_seed:
        return None
    n = max(len(v) for v in per_seed)
    mat = np.full((len(per_seed), n), np.nan)
    for i, v in enumerate(per_seed):
        mat[i, :len(v)] = v
    return np.nanmean(mat, axis=0)


def _seed_level_summary(results_root: Path, stage: str):
    raw_dir = results_root / "raw" / stage
    if not raw_dir.exists():
        return
    rows = []
    for cond_dir in sorted([d for d in raw_dir.iterdir() if d.is_dir()]):
        data = _load_condition(raw_dir, cond_dir.name)
        for algo, seeds in data.items():
            for seed, recs in seeds.items():
                succ = [r["success"] for r in recs]
                steps = [r["steps"] for r in recs]
                rows.append({
                    "condition": cond_dir.name, "algorithm": algo, "seed": seed,
                    "episodes": len(recs),
                    **ust.describe(np.array(succ, dtype=float)),
                    **{f"steps_{k}": v for k, v in
                       ust.describe(np.array(steps, dtype=float)).items()},
                })
    if rows:
        write_csv(results_root / "statistics" / f"{stage}_seed_level.csv", rows)


def _ood_drop(results_root: Path):
    """式(35)：Drop(d)=(P_nom−P_d)/P_nom，P_nom=ID条件SR。"""
    raw_dir = results_root / "raw" / "ood"
    if not raw_dir.exists():
        return
    id_data = _load_condition(raw_dir, "id")
    if not id_data:
        return
    id_sr = {}
    for algo, seeds in id_data.items():
        allr = [r for recs in seeds.values() for r in recs]
        id_sr[algo] = float(np.mean([r["success"] for r in allr])) if allr else None
    rows = []
    for cond_dir in sorted([d for d in raw_dir.iterdir() if d.is_dir()]):
        if cond_dir.name == "id":
            continue
        data = _load_condition(raw_dir, cond_dir.name)
        for algo, seeds in data.items():
            if algo not in id_sr or id_sr[algo] is None:
                continue
            allr = [r for recs in seeds.values() for r in recs]
            if not allr:
                continue
            sr_d = float(np.mean([r["success"] for r in allr]))
            rows.append({
                "condition": cond_dir.name, "algorithm": algo,
                "SR_id": id_sr[algo], "SR_ood": sr_d,
                "Drop": umt.relative_drop(id_sr[algo], sr_d) if id_sr[algo] > 0 else None,
            })
    if rows:
        write_csv(results_root / "processed" / "ood_drop.csv", rows)


def merge_statistics(results_root: Path):
    """合并各阶段统计检验/效应量为单一总表（statistical_tests.csv等）。"""
    import csv as _csv
    stats_dir = results_root / "statistics"
    for name in ["statistical_tests", "effect_sizes"]:
        rows = []
        for f in sorted(stats_dir.glob(f"{name}_*.csv")):
            with open(f, encoding="utf-8") as fh:
                rows.extend(list(_csv.DictReader(fh)))
        if rows:
            write_csv(stats_dir / f"{name}.csv", rows)
            print(f"  merged {name}: {len(rows)} rows from {len(list(stats_dir.glob(name + '_*.csv')))} stage files")
