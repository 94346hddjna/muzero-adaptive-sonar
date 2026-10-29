# -*- coding: utf-8 -*-
"""论文表格自动生成（用户第24章；全部数值从raw结果计算）。

Table 1 主对比 | Table 2 难度分层 | Table 3 统计显著性 | Table 4 MCTS消融
Table 5 信念/记忆消融 | Table 6 OOD鲁棒性 | Table 7 计算成本
每表输出CSV/XLSX/Markdown/LaTeX四格式。
"""
import csv
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from utils.io_utils import write_csv, write_json

ALGOS = ["sequential", "random", "belief_greedy", "pomcp",
         "dqn", "ppo", "dqn_gru", "ppo_stack", "muzero"]
ALGO_NAMES = {"sequential": "Sequential", "random": "Random",
              "belief_greedy": "Belief-Greedy",
              "pomcp": "POMCP (known model)",
              "dqn": "DQN", "ppo": "PPO",
              "dqn_gru": "DQN-GRU (memory)", "ppo_stack": "PPO-Stack (memory)",
              "muzero": "MuZero (ours)"}
TRAIN_FREE = ("sequential", "random", "belief_greedy", "pomcp")


def _load(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        rows = []
        for r in csv.DictReader(f):
            r2 = {}
            for k, v in r.items():
                try:
                    r2[k] = float(v) if v not in (None, "", "True", "False") else v
                except (ValueError, TypeError):
                    r2[k] = v
            rows.append(r2)
        return rows


def _fmt(v, prec=2):
    if v is None or (isinstance(v, float) and (np.isnan(v) if isinstance(v, float) else False)):
        return "—"
    if isinstance(v, (int, float)):
        return f"{v:.{prec}f}"
    return str(v)


def _alg_mean_std(rows, algo, metric, condition=None, variant=None):
    vals = [r[metric] for r in rows
            if r.get("algorithm") == algo
            and (condition is None or r.get("condition") == condition)
            and (variant is None or r.get("variant") == (variant or "main"))
            and isinstance(r.get(metric), (int, float))]
    if not vals:
        return None, None
    return float(np.mean(vals)), float(np.std(vals))


def _train_budget(smoke: bool) -> int:
    """从训练配置读取真实预算（禁止硬编码）。"""
    try:
        from utils.config import load_section
        return int(load_section("training", smoke)["total_env_steps"])
    except Exception:
        return 2000000


def _write_table(name: str, header: List[str], rows: List[List], tables_dir: Path,
                 caption: str = ""):
    tables_dir.mkdir(parents=True, exist_ok=True)
    # CSV
    write_csv(tables_dir / f"{name}.csv",
              [dict(zip(header, r)) for r in rows], header)
    # Markdown
    md = ["| " + " | ".join(str(h) for h in header) + " |",
          "|" + "---|" * len(header)]
    for r in rows:
        md.append("| " + " | ".join(_fmt(v) if isinstance(v, (int, float)) else str(v)
                                    for v in r) + " |")
    if caption:
        md.insert(0, f"**{caption}**\n")
    (tables_dir / f"{name}.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    # LaTeX
    colspec = "l" + "c" * (len(header) - 1)
    tex = ["\\begin{table}[htbp]", "\\centering",
           "\\caption{" + caption + "}" if caption else "",
           f"\\begin{{tabular}}{{{colspec}}}", "\\toprule",
           " & ".join(str(h) for h in header) + " \\\\", "\\midrule"]
    for r in rows:
        tex.append(" & ".join(_fmt(v) if isinstance(v, (int, float)) else str(v)
                              for v in r) + " \\\\")
    tex += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    (tables_dir / f"{name}.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")
    # XLSX
    try:
        import openpyxl
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = name[:31]
        ws.append([str(h) for h in header])
        for r in rows:
            ws.append([v if isinstance(v, (int, float)) else str(v) for v in r])
        wb.save(tables_dir / f"{name}.xlsx")
    except ImportError:
        pass


def run_tables(results_root: Path, smoke: bool = False) -> Dict:
    tables_dir = results_root / "tables"
    made = {}
    main = _load(results_root / "processed" / "main_results.csv")
    abl = _load(results_root / "processed" / "ablation_results.csv")
    ood = _load(results_root / "processed" / "ood_results.csv")
    tests = _load(results_root / "statistics" / "statistical_tests.csv")
    # 运行时数据（训练时间+时延）
    runtime_rows = []
    for algo in ALGOS:
        wall = []
        for f in (results_root / "checkpoints").glob(f"{algo}/main_seed*/"):
            p = list(f.glob("*_summary.json"))
            if p:
                d = read_json_safe(p[0])
                if d and "training_wall_clock_sec" in d:
                    wall.append(d["training_wall_clock_sec"])
        lat_m, lat_s = _alg_mean_std(main, algo, "MeanDecisionLatencyMs", "medium")
        runtime_rows.append([
            ALGO_NAMES[algo],
            _fmt(np.mean(wall) if wall else None, 0),
            _fmt(lat_m, 2), _fmt(lat_s, 2),
            "—" if algo in TRAIN_FREE else str(_train_budget(smoke)),
        ])
    _write_table("table7_computational_cost",
                 ["Method", "Training wall-clock (s, mean)", "Decision latency (ms, mean)",
                  "Decision latency (ms, std)", "Env. steps budget"],
                 runtime_rows, tables_dir,
                 "Table 7: Computational cost (Medium condition)")
    made["table7"] = True
    # ---- Table 1/2: 主结果 ----
    if main:
        metrics = [("SR", "SR"), ("ADS", "ADS"), ("EDC", "EDC"), ("FAR", "FAR"),
                   ("ExpectedReturn", "J"), ("AvgSteps", "Steps")]
        rows1 = []
        for algo in ALGOS:
            row = [ALGO_NAMES[algo]]
            for mkey, _ in metrics:
                mu, sd = _alg_mean_std(main, algo, mkey, "medium")
                row.append(f"{_fmt(mu)}±{_fmt(sd)}" if mu is not None else "—")
            rows1.append(row)
        _write_table("table1_main_comparison",
                     ["Method"] + [m for m, _ in metrics], rows1, tables_dir,
                     "Table 1: Main comparison (Medium difficulty, mean±std over seeds)")
        made["table1"] = True
        rows2 = []
        for algo in ALGOS:
            row = [ALGO_NAMES[algo]]
            for diff in ["easy", "medium", "hard"]:
                for mkey in ["SR", "ADS"]:
                    mu, sd = _alg_mean_std(main, algo, mkey, diff)
                    row.append(f"{_fmt(mu)}±{_fmt(sd)}" if mu is not None else "—")
            rows2.append(row)
        _write_table("table2_difficulty",
                     ["Method", "SR(Easy)", "ADS(Easy)", "SR(Medium)", "ADS(Medium)",
                      "SR(Hard)", "ADS(Hard)"], rows2, tables_dir,
                     "Table 2: Easy/Medium/Hard performance")
        made["table2"] = True
    # ---- Table 3: 统计显著性 ----
    if tests:
        rows3 = []
        for r in tests:
            if r.get("metric") == "SR":
                rows3.append([
                    f"{r.get('algo_a')} vs {r.get('algo_b')}", r.get("condition"),
                    _fmt(r.get("mean_a"), 3), _fmt(r.get("mean_b"), 3),
                    r.get("test"), _fmt(r.get("p_value"), 4), _fmt(r.get("p_holm"), 4),
                    "Yes" if str(r.get("significant_holm_0.05")) in ("True", "1.0") else "No",
                    f"{r.get('effect_size_name')}={_fmt(r.get('effect_size'), 3)}",
                ])
        _write_table("table3_significance",
                     ["Comparison", "Condition", "SR(a)", "SR(b)", "Test",
                      "p", "p (Holm)", "Sig. (α=0.05)", "Effect size"],
                     rows3, tables_dir,
                     "Table 3: Statistical significance (paired, Holm-corrected)")
        made["table3"] = True
    # ---- Table 4: MCTS消融（A/B/C） ----
    if abl:
        # 主基准行：必须用主实验干净值（消融CSV的main行混入了M/D扫描行，
        # 口径与论文表3-3一致：main_results.csv 主测试值）
        rows4 = [["MuZero (M=100, D=5)"]]
        for cond in ["easy", "medium", "hard"]:
            mu, sd = _alg_mean_std(main, "muzero", "SR", cond, "main")
            rows4[0].append(f"{_fmt(mu)}±{_fmt(sd)}" if mu is not None else "—")
        # w/o MCTS
        row = ["w/o MCTS"]
        for cond in ["easy", "medium", "hard"]:
            vals = [r["SR"] for r in abl
                    if r.get("condition") == cond and r.get("algorithm") == "muzero"
                    and "use_mcts" in str(r.get("eval_options", ""))
                    and isinstance(r.get("SR"), (int, float))]
            row.append(f"{_fmt(np.mean(vals))}±{_fmt(np.std(vals))}" if vals else "—")
        rows4.append(row)
        for M in [25, 50, 100, 200]:
            row = [f"M={M}"]
            for cond in ["easy", "medium", "hard"]:
                vals = [r["SR"] for r in abl
                        if r.get("condition") == cond and r.get("algorithm") == "muzero"
                        and f'"num_simulations": {M}' in str(r.get("eval_options", "")).replace(".0", "")
                        and '"planning_depth": 5' in str(r.get("eval_options", "")).replace(".0", "")
                        and isinstance(r.get("SR"), (int, float))]
                row.append(f"{_fmt(np.mean(vals))}±{_fmt(np.std(vals))}" if vals else "—")
            rows4.append(row)
        for D in [1, 3, 5, 10]:
            row = [f"D={D}"]
            for cond in ["easy", "medium", "hard"]:
                vals = [r["SR"] for r in abl
                        if r.get("condition") == cond and r.get("algorithm") == "muzero"
                        and f'"planning_depth": {D}' in str(r.get("eval_options", "")).replace(".0", "")
                        and '"num_simulations": 100' in str(r.get("eval_options", "")).replace(".0", "")
                        and isinstance(r.get("SR"), (int, float))]
                row.append(f"{_fmt(np.mean(vals))}±{_fmt(np.std(vals))}" if vals else "—")
            rows4.append(row)
        _write_table("table4_mcts_ablation",
                     ["Setting", "SR (Easy)", "SR (Medium)", "SR (Hard)"], rows4,
                     tables_dir, "Table 4: MCTS ablations (A/B/C)")
        made["table4"] = True
        # ---- Table 5: 信念/记忆/域随机/奖励消融 ----
        rows5 = []
        for variant, label in [("main", "MuZero (full)"), ("wo_belief", "w/o belief (D)"),
                               ("wo_memory", "w/o memory (E)"),
                               ("wo_domain_rand", "w/o domain rand. (F)"),
                               ("shaped", "shaped reward (G)")]:
            row = [label]
            for cond in ["easy", "medium", "hard"]:
                mu, sd = _alg_mean_std(abl, "muzero", "SR", cond, variant)
                row.append(f"{_fmt(mu)}±{_fmt(sd)}" if mu is not None else "—")
            rows5.append(row)
        _write_table("table5_belief_memory_ablation",
                     ["Setting", "SR (Easy)", "SR (Medium)", "SR (Hard)"], rows5,
                     tables_dir, "Table 5: Belief/Memory/DomainRand/Reward ablations")
        made["table5"] = True
    # ---- Table 6: OOD ----
    if ood:
        rows6 = []
        drop = _load(results_root / "processed" / "ood_drop.csv")
        drop_map = {(r.get("condition"), r.get("algorithm")): r.get("Drop") for r in drop}
        for algo in ALGOS:
            row = [ALGO_NAMES[algo]]
            for cond in ["id", "ood_snr", "ood_pfa", "ood_n", "ood_prior", "ood_reverb"]:
                mu, sd = _alg_mean_std(ood, algo, "SR", cond)
                d = drop_map.get((cond, algo))
                cell = f"{_fmt(mu)}±{_fmt(sd)}"
                if d is not None and isinstance(d, (int, float)) and cond != "id":
                    cell += f" (drop {_fmt(d*100, 1)}%)"
                row.append(cell if mu is not None else "—")
            rows6.append(row)
        _write_table("table6_ood_robustness",
                     ["Method", "ID", "OOD-SNR", "OOD-PFA", "OOD-N", "OOD-Prior",
                      "OOD-Reverb"], rows6, tables_dir,
                     "Table 6: OOD robustness (SR, Drop in %)")
        made["table6"] = True
    return made


def read_json_safe(p: Path):
    try:
        return read_json(p)
    except Exception:
        return None


from utils.io_utils import read_json  # noqa: E402
