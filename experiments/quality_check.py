# -*- coding: utf-8 -*-
"""自动质量检查（用户第27章）：发现异常→中止最终报告生成并报错。

检查项：NaN/Inf、空结果、缺种子、缺算法、缺条件、重复测试样本、
train/test种子重叠、信念和异常、缺checkpoint、零回合评测、常数可疑输出。
"""
import csv
import math
from pathlib import Path
from typing import Dict, List

import numpy as np

from utils import seeding as us
from utils.io_utils import read_json, read_jsonl


def run_quality_check(results_root: Path, smoke: bool = False, strict: bool = True) -> Dict:
    issues: List[str] = []
    warnings: List[str] = []
    ok_all = True

    # 1 NaN/Inf/空结果/零回合/常数输出（扫描所有processed CSV）
    proc = results_root / "processed"
    for f in proc.glob("*.csv"):
        with open(f, encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        if not rows:
            warnings.append(f"empty csv: {f.name}")
            continue
        for i, r in enumerate(rows):
            for k, v in r.items():
                if v in ("", None):
                    continue
                try:
                    fv = float(v)
                    if math.isnan(fv) or math.isinf(fv):
                        issues.append(f"NaN/Inf in {f.name} row{i} key={k}")
                except (ValueError, TypeError):
                    pass
        # 零回合评测
        if "episodes" in rows[0]:
            for i, r in enumerate(rows):
                try:
                    if int(float(r["episodes"])) == 0:
                        issues.append(f"zero-episode eval in {f.name} row{i}")
                except (ValueError, TypeError):
                    pass
        # 常数可疑输出（一个算法在某条件所有seed的SR完全相同且非0/1边界）—仅警告
        if "algorithm" in rows[0] and "SR" in rows[0] and "condition" in rows[0] and "seed" in rows[0]:
            from collections import defaultdict
            grp = defaultdict(list)
            for r in rows:
                try:
                    grp[(r["algorithm"], r["condition"])].append(float(r["SR"]))
                except (ValueError, TypeError):
                    pass
            for (algo, cond), vals in grp.items():
                if len(vals) >= 3 and len(set(vals)) == 1 and vals[0] not in (0.0, 1.0):
                    warnings.append(
                        f"constant SR across seeds: {algo}@{cond} SR={vals[0]} "
                        f"(可能为确定性行为，请人工核查)")

    # 2 缺算法/缺条件/缺种子（main阶段）
    main = results_root / "raw" / "main"
    expected_algos = {"sequential", "random", "belief_greedy", "dqn", "ppo", "muzero"}
    for diff_dir in sorted([d for d in main.iterdir() if d.is_dir()]) if main.exists() else []:
        import re as _re
        algos = set()
        for f in diff_dir.glob("*.jsonl"):
            m = _re.match(r"(.+?)_main", f.name)
            if m:
                algos.add(m.group(1))
        missing = expected_algos - algos
        if missing and not smoke:
            issues.append(f"missing algorithms in main/{diff_dir.name}: {sorted(missing)}")
        seeds_by_algo: Dict[str, set] = {}
        for f in diff_dir.glob("*.jsonl"):
            m = _re.match(r"(.+?)_main_seed(\d+)", f.stem)
            if not m:
                continue
            seeds_by_algo.setdefault(m.group(1), set()).add(int(m.group(2)))
        for algo, seeds in seeds_by_algo.items():
            if algo in ("dqn", "ppo", "muzero") and not smoke:
                need = set(us.TRAIN_SEEDS)
                miss = need - seeds
                if miss:
                    issues.append(f"missing seeds in main/{diff_dir.name}/{algo}: {sorted(miss)}")

    # 3 重复测试样本（scenario_index在单文件内重复）
    for f in (results_root / "raw").rglob("*.jsonl"):
        if f.stat().st_size == 0:
            continue
        idxs = []
        n_read = 0
        for rec in read_jsonl(f):
            n_read += 1
            if "scenario_index" in rec:
                idxs.append(rec["scenario_index"])
        if idxs and len(idxs) != len(set(idxs)):
            issues.append(f"duplicate scenario_index in {f}")
        if not smoke and n_read == 0 and "training_log" not in f.name:
            issues.append(f"zero-episode evaluation file: {f}")

    # 4 train/test种子重叠
    iso = us.check_namespace_isolation()
    if not iso["ok"]:
        issues.append(f"seed namespace overlap: {iso['overlaps']}")

    # 5 信念和异常（raw里的final_belief）
    bad_belief = 0
    checked = 0
    for f in (results_root / "raw").rglob("*.jsonl"):
        if f.stat().st_size == 0:
            continue
        for rec in read_jsonl(f):
            fb = rec.get("final_belief")
            if fb is not None:
                checked += 1
                s = sum(fb)
                if abs(s - 1.0) > 1e-6:
                    bad_belief += 1
    if checked:
        (results_root / "processed").mkdir(exist_ok=True)
        write_note = {"checked": checked, "bad_belief_sums": bad_belief}
        from utils.io_utils import write_json
        write_json(results_root / "processed" / "belief_check.json", write_note)
        if bad_belief > 0:
            issues.append(f"abnormal belief sums: {bad_belief}/{checked}")

    # 6 缺checkpoint（RL算法每个seed应有best_val）
    for algo in ["dqn", "ppo", "muzero"]:
        for seed in us.TRAIN_SEEDS:
            p = results_root / "checkpoints" / algo / f"main_seed{seed}" / \
                f"{algo}_main_seed{seed}_best_val.ckpt"
            if not p.exists():
                # 允许final/last回退
                alt = p.parent / f"{algo}_main_seed{seed}_final.ckpt"
                if not alt.exists():
                    if not smoke:
                        issues.append(f"missing checkpoint: {algo} seed{seed}")
                    else:
                        warnings.append(f"missing checkpoint (smoke允许): {algo} seed{seed}")

    # 7 场景文件隔离（验证/测试/OOD不共享rng_seed）
    from env.scenarios import check_scenario_files_disjoint
    scen_check = check_scenario_files_disjoint(results_root)
    if not scen_check["ok"]:
        issues.append(f"scenario rng stream overlap: {scen_check.get('stream_level_overlaps', scen_check.get('duplicates'))}")

    ok_all = len(issues) == 0
    out = {"passed": ok_all, "issues": issues, "warnings": warnings,
           "n_checks": 7, "strict": strict}
    from utils.io_utils import write_json
    write_json(results_root / "processed" / "quality_check.json", out)
    return out


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    root = Path(__file__).resolve().parent.parent / "results"
    res = run_quality_check(root, smoke="--smoke" in sys.argv)
    print("QUALITY CHECK:", "PASSED" if res["passed"] else "FAILED")
    for i in res["issues"]:
        print("  ISSUE:", i)
    for w in res["warnings"]:
        print("  warn:", w)
    sys.exit(0 if res["passed"] else 1)
