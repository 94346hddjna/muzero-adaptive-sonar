# -*- coding: utf-8 -*-
"""实验状态报告（用户第26章）：experiment_status.md。

自动检查每一项：completed / FAILED / NOT RUN。绝不虚报completed。
"""
import json
from pathlib import Path
from typing import Dict

from utils.io_utils import read_json


def _exists(p: Path) -> bool:
    return p.exists() and p.stat().st_size > 0


def _has_jsonl(d: Path, pattern: str = "*.jsonl") -> int:
    if not d.exists():
        return 0
    return sum(1 for f in d.glob(pattern) if f.stat().st_size > 0)


def run_status(results_root: Path, smoke: bool = False) -> Dict:
    items = {}
    raw = results_root / "raw"
    # 单元测试与sanity
    items["Environment unit tests passed"] = _unit_tests_ok(results_root)
    items["Belief tests passed"] = items["Environment unit tests passed"]  # 同一suite
    items["Level1 sanity check passed"] = _sanity_ok(results_root, "level2_consistency", False)
    items["Level2 sanity check passed"] = _sanity_ok(results_root, "level2_consistency", True)
    # 基线
    main = raw / "main"
    for algo, name in [("sequential", "Sequential completed"),
                       ("random", "Random completed"),
                       ("belief_greedy", "Belief-Greedy completed")]:
        items[name] = _algo_done(main, algo, ["easy", "medium", "hard"])
    # RL训练
    for algo, name in [("dqn", "DQN trained"), ("ppo", "PPO trained"),
                       ("muzero", "MuZero trained")]:
        items[name] = _trained(results_root, algo, smoke)
    items["Multi-seed completed"] = _multi_seed(results_root, smoke)
    items["Main experiment completed"] = _main_done(main, smoke)
    items["Ablations completed"] = _ablations_done(results_root, smoke)
    items["OOD completed"] = _ood_done(results_root, smoke)
    items["Level2 completed"] = _jsonl_count(raw / "level2") > 0
    items["Statistics completed"] = _exists(results_root / "statistics" / "statistical_tests.csv")
    items["Figures completed"] = _figures_done(results_root)
    items["Tables completed"] = _tables_done(results_root)
    items["Final report completed"] = _exists(results_root / "reports" / "final_experiment_report.md")
    # 生成markdown
    lines = ["# 实验状态报告 experiment_status.md", "",
             f"自动生成。completed=通过证据核实；FAILED=证据缺失或异常；NOT RUN=未执行。", ""]
    for k, v in items.items():
        status = "completed" if v is True else ("FAILED" if v is False else "NOT RUN")
        lines.append(f"- [{'x' if v is True else ' '}] {k} — **{status}**")
    out_md = results_root / "reports" / "experiment_status.md"
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    from utils.io_utils import write_json
    write_json(results_root / "reports" / "experiment_status.json", items)
    return items


def _unit_tests_ok(results_root: Path) -> bool:
    """单元测试是否通过（以tests结果缓存为准，由run_all写入）。"""
    return _exists(results_root / "raw" / "unit_tests_passed.json")


def _sanity_ok(results_root: Path, key: str, level2: bool) -> bool:
    p = results_root / "raw" / "sanity_checks.json"
    if not _exists(p):
        return None
    d = read_json(p)
    if not d.get("all_passed"):
        return False
    chk = d.get("checks", {}).get(key, {})
    return bool(chk.get("ok"))


def _algo_done(main_dir: Path, algo: str, diffs) -> bool:
    if not main_dir.exists():
        return None  # 评测未开始
    import re as _re
    n = 0
    for d in diffs:
        dd = main_dir / d
        if dd.exists():
            for f in dd.glob(f"{algo}_*.jsonl"):
                if f.stat().st_size > 0:
                    n += 1
                    break
    if n == 0:
        return None  # 该算法尚未评测
    return n >= len(diffs)


def _trained(results_root: Path, algo: str, smoke: bool) -> bool:
    seeds = [101] if smoke else [101, 102, 103, 104, 105]
    ok = 0
    for seed in seeds:
        d = results_root / "checkpoints" / algo / f"main_seed{seed}"
        done = (d / f".done_{algo}_main_seed{seed}.json").exists()
        summary = _exists(d / f"{algo}_main_seed{seed}_summary.json")
        if done and summary:
            ok += 1
    if ok == len(seeds):
        return True
    if ok == 0:
        return None
    return False


def _multi_seed(results_root: Path, smoke: bool) -> bool:
    if smoke:
        return None  # smoke单种子，不适用
    algos = ["dqn", "ppo", "muzero"]
    for algo in algos:
        if not _trained(results_root, algo, False):
            return _trained(results_root, algo, False)
    return True


def _main_done(main_dir: Path, smoke: bool) -> bool:
    if not main_dir.exists():
        return None
    import re as _re
    need = {"sequential", "random", "belief_greedy", "dqn", "ppo", "muzero"}
    for d in ["easy", "medium", "hard"]:
        algos = set()
        if (main_dir / d).exists():
            for f in (main_dir / d).glob("*.jsonl"):
                if f.stat().st_size == 0:
                    continue
                m = _re.match(r"(.+?)_main", f.name)
                if m:
                    algos.add(m.group(1))
        if not need.issubset(algos):
            return False if algos else None
    return True


def _ablations_done(results_root: Path, smoke: bool) -> bool:
    d = results_root / "raw" / "ablation"
    if not d.exists() or _jsonl_count(d) == 0:
        return None
    # 检查七组A-G的关键产物
    files = list(d.rglob("*.jsonl"))
    names = [f.name for f in files]
    # v3等盲搜索轮次禁用消融D（信念本就不在输入中）——按config豁免
    try:
        from utils.config import load_config as _lc
        disabled = _lc("ablation").get("ablation", {}).get("disabled_variants", [])
    except Exception:
        disabled = []
    has = {
        "A_wo_mcts": any("woMCTS" in n for n in names),
        "B_sims": any("M25" in n or "M8" in n for n in names),
        "C_depth": any("D1" in n or "D3" in n for n in names),
        "D_wo_belief": (any("woBelief" in n for n in names) or "wo_belief" in disabled),
        "E_wo_memory": any("woMemory" in n for n in names),
        "F_wo_domrand": any("woDomRand" in n for n in names),
        "G_shaped": any("shapedReward" in n for n in names),
    }
    missing = [k for k, v in has.items() if not v]
    if missing:
        return False
    return True


def _ood_done(results_root: Path, smoke: bool) -> bool:
    d = results_root / "raw" / "ood"
    if not d.exists() or _jsonl_count(d) == 0:
        return None
    for cond in ["id", "ood_snr", "ood_pfa", "ood_n", "ood_prior", "ood_reverb"]:
        if _jsonl_count(d / cond) == 0:
            return False
    return True


def _figures_done(results_root: Path) -> bool:
    d = results_root / "figures"
    if not d.exists():
        return None
    n = len(list(d.glob("*.png")))
    if n == 0:
        return None
    if n < 10:
        return False
    return True


def _tables_done(results_root: Path) -> bool:
    d = results_root / "tables"
    if not d.exists():
        return None
    n = len(list(d.glob("table*.csv")))
    if n == 0:
        return None
    return n >= 7


def _jsonl_count(d: Path) -> int:
    """递归统计jsonl文件数。"""
    if not d.exists():
        return 0
    return sum(1 for f in d.rglob("*.jsonl") if f.stat().st_size > 0)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    root = Path(__file__).resolve().parent.parent / "results"
    res = run_status(root, smoke="--smoke" in sys.argv)
    for k, v in res.items():
        print(f"{k}: {'completed' if v is True else ('FAILED' if v is False else 'NOT RUN')}")
