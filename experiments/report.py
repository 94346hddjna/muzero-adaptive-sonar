# -*- coding: utf-8 -*-
"""自动实验报告生成（用户第25、37章）。

final_experiment_report.md：20节完整报告（只描述真实结果；显著性结论仅当
统计检验真正支持时才写"显著"）。
FINAL_SUMMARY.md：中文、非专业读者可读的最终交付摘要（20项问答）。
"""
import csv
from pathlib import Path
from typing import Dict, List

import numpy as np

from utils import hardware as hw
from utils.io_utils import read_json

ALGO_NAMES = {"sequential": "Sequential", "random": "Random",
              "belief_greedy": "Belief-Greedy", "dqn": "DQN", "ppo": "PPO",
              "muzero": "MuZero"}


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


def _fmt(v, prec=3):
    if v is None or v == "—" or v == "":
        return "—"
    if isinstance(v, (int, float)):
        if isinstance(v, float) and np.isnan(v):
            return "—"
        return f"{v:.{prec}f}"
    return str(v)


def _agg(rows, algo, metric, condition=None, variant=None, suffix=None):
    vals = [r[metric] for r in rows
            if r.get("algorithm") == algo
            and (condition is None or r.get("condition") == condition)
            and (variant is None or (r.get("variant") or "main") == variant)
            and (suffix is None or (r.get("eval_options") or "") == suffix)
            and isinstance(r.get(metric), (int, float))]
    if not vals:
        return None, None, 0
    return float(np.mean(vals)), float(np.std(vals)), len(vals)


def run_report(results_root: Path, smoke: bool = False) -> Dict:
    reports = results_root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    main = _load(results_root / "processed" / "main_results.csv")
    abl = _load(results_root / "processed" / "ablation_results.csv")
    ood = _load(results_root / "processed" / "ood_results.csv")
    tests = _load(results_root / "statistics" / "statistical_tests.csv")
    pb = _load(results_root / "processed" / "prior_bias_results.csv")
    l2 = _load(results_root / "processed" / "level2_results.csv")
    sanity = _loadj(results_root / "raw" / "sanity_checks.json")
    quality = _loadj(results_root / "processed" / "quality_check.json")
    hwd = hw.detect_hardware()
    cal = _loadj(results_root / "raw" / "calibration.json")
    status = _loadj(results_root / "reports" / "experiment_status.json")

    # ================= final_experiment_report.md =================
    L = []
    L.append("# 最终实验报告：基于MuZero的自适应水下目标声呐探测策略研究\n")
    L.append("> 本报告由程序从results/raw/原始数据自动生成；所有数值来自真实运行的"
             "原始日志，可逐条追溯。未运行/失败的实验如实标注。\n")

    L.append("## 1. Experimental Setup\n")
    L.append("两级声呐仿真环境（开题报告第6章）：Level 1检测概率抽象环境（式(9)-(14)"
             "统计映射）用于训练；Level 2信号级环境（式(6)-(8) LFM/时延/传播+匹配滤波）"
             "用于验证一致性。POMDP式(3)；贝叶斯信念更新式(4)(5)；奖励式(16)。\n")
    L.append("## 2. Hardware\n")
    L.append(f"- CPU核数: {hwd.get('cpu_count')}；RAM: {hwd.get('ram_total_gb')} GB\n")
    L.append(f"- GPU: {hwd.get('gpu_name')} ({hwd.get('gpu_memory_total_gb')} GB), "
             f"CUDA {hwd.get('cuda_version')}, PyTorch CUDA可用: {hwd.get('cuda_available')}\n")
    L.append("## 3. Software\n")
    L.append(f"- Python {hwd.get('python_version')}, PyTorch {hwd.get('pytorch_version')}, "
             f"NumPy {__import__('numpy').__version__}, SciPy {__import__('scipy').__version__}\n")
    L.append("## 4. Environment\n")
    if cal:
        L.append("各难度物理校准（平均PD目标 vs 实测，B2决策）：\n")
        L.append("| 难度 | N | P_FA | PD目标 | 实测平均PD | PD_std | SNR基准(dB) |")
        L.append("|---|---|---|---|---|---|---|")
        for d, v in cal.get("difficulties", cal).items() if isinstance(cal.get("difficulties", cal), dict) else []:
            L.append(f"|{d}|{v.get('N')}|{v.get('pfa')}|{v.get('pd_target')}|"
                     f"{_fmt(v.get('achieved'), 4)}|{_fmt(v.get('pd_std'), 4)}|"
                     f"{_fmt(v.get('snr_base_db'), 2)}|")
    L.append("\n## 5. Dataset/Scenario Generation\n")
    L.append("全部数据由声呐仿真环境生成（无外部数据集）。固定场景文件由互不相交的"
             "主种子生成并锁定（results/raw/scenarios/）：Train种子[101..105]、"
             "Validation 20250901、Test 30303（锁定）、OOD 40404。配对评测：所有算法在"
             "完全相同场景（目标位置/环境参数/先验/每回合RNG）下测试。\n")
    L.append("## 6. Train/Validation/Test Protocol\n")
    L.append("- 训练：域随机化（Easy25%/Medium50%/Hard25%+SNR±3dB+PFA×[0.8,1.2]+先验"
             "类型随机，B7决策）；1e6环境步预算（开题报告公平性）。\n"
             "- 验证：固定Medium验证场景（200回合/次），仅以Validation SR选择"
             "best_val checkpoint（绝不用Test调参）。\n"
             "- 测试：参数锁定后运行；Test结果不用于任何调参。\n")
    L.append("## 7. Algorithms\n")
    L.append("Sequential（顺序扫描）、Random（随机扫描）、Belief-Greedy（式(27)）、"
             "DQN（式(28)）、PPO（式(29)(30)）、MuZero+MCTS（式(17)-(26)，主方法）。\n"
             "所有学习算法：相同观测向量（式15）、相同奖励、相同训练预算、相同测试场景。\n")
    L.append("## 8. Hyperparameters\n")
    L.append("见configs/*.yaml（表7初始值：γ=0.99, lr=3e-4 AdamW, batch=128, K=5, n=5, "
             "latent=128, GRU=256, M=50, D_p=5, c1=1.25, τ退火1.0→0.25, Dirichlet α=0.3）。"
             "每次实验的实际config与manifest存于results/raw/manifests/。\n")
    # 主结果
    L.append("## 9. Main Results\n")
    if main:
        L.append("| 方法 | SR(Easy) | SR(Medium) | SR(Hard) | ADS(Medium) | EDC(Medium) | FAR(Medium) |")
        L.append("|---|---|---|---|---|---|---|")
        for algo in ["sequential", "random", "belief_greedy", "dqn", "ppo", "muzero"]:
            e = _agg(main, algo, "SR", "easy")[0]
            m = _agg(main, algo, "SR", "medium")[0]
            h = _agg(main, algo, "SR", "hard")[0]
            ads = _agg(main, algo, "ADS", "medium")[0]
            edc = _agg(main, algo, "EDC", "medium")[0]
            far = _agg(main, algo, "FAR", "medium")[0]
            L.append(f"|{ALGO_NAMES[algo]}|{_fmt(e)}|{_fmt(m)}|{_fmt(h)}|{_fmt(ads)}|"
                     f"{_fmt(edc)}|{_fmt(far)}|")
        # 样本效率
        L.append("\n样本效率（达到SR≥0.90所需环境步，未达到则标'未达到'）：\n")
        for algo in ["dqn", "ppo", "muzero"]:
            steps = _sample_efficiency(results_root, algo, smoke)
            L.append(f"- {ALGO_NAMES[algo]}: {steps if steps else '未达到阈值'}\n")
    else:
        L.append("主实验未运行或无结果。\n")
    # 统计
    L.append("## 10. Statistical Tests\n")
    if tests:
        sig = [r for r in tests if str(r.get("significant_holm_0.05")) in ("True", "1.0")]
        L.append(f"共{len(tests)}项配对检验（Holm校正，α=0.05），其中{len(sig)}项显著。"
                 "详见results/statistics/statistical_tests.csv与effect_sizes.csv。\n")
        for r in sig:
            L.append(f"- {r.get('condition')}: {r.get('algo_a')} vs {r.get('algo_b')}: "
                     f"ΔSR={_fmt(r.get('mean_diff'))}, p={_fmt(r.get('p_holm'), 4)}, "
                     f"{r.get('effect_size_name')}={_fmt(r.get('effect_size'))}\n")
    else:
        L.append("统计检验未运行或无结果。\n")
    # 消融
    L.append("## 11. Ablation Studies\n")
    if abl:
        for variant, label in [("main", "MuZero完整"), ("wo_belief", "D w/o belief"),
                               ("wo_memory", "E w/o memory"),
                               ("wo_domain_rand", "F w/o域随机化"), ("shaped", "G 塑形奖励")]:
            mu, sd, _ = _agg(abl, "muzero", "SR", "medium", variant)
            L.append(f"- {label}: SR(Medium)={_fmt(mu)}±{_fmt(sd)}\n")
        mu, sd, _ = _agg(abl, "muzero", "SR", "medium", suffix=None)
        wo = [r for r in abl if r.get("condition") == "medium"
              and "use_mcts" in str(r.get("eval_options", ""))]
        if wo:
            vals = [r["SR"] for r in wo if isinstance(r.get("SR"), (int, float))]
            L.append(f"- A w/o MCTS: SR(Medium)={_fmt(np.mean(vals) if vals else None)}±"
                     f"{_fmt(np.std(vals) if vals else None)}\n")
        for M in [25, 50, 100, 200]:
            sub = [r for r in abl if r.get("condition") == "medium"
                   and f'"num_simulations": {M}' in str(r.get("eval_options", "")).replace(".0", "")
                   and isinstance(r.get("SR"), (int, float))]
            if sub:
                L.append(f"- B M={M}: SR={_fmt(np.mean([r['SR'] for r in sub]))}\n")
        for D in [1, 3, 5, 10]:
            sub = [r for r in abl if r.get("condition") == "medium"
                   and f'"planning_depth": {D}' in str(r.get("eval_options", "")).replace(".0", "")
                   and isinstance(r.get("SR"), (int, float))]
            if sub:
                L.append(f"- C D={D}: SR={_fmt(np.mean([r['SR'] for r in sub]))}\n")
    else:
        L.append("消融实验未运行或无结果。\n")
    # 鲁棒性/OOD
    L.append("## 12-13. Robustness & OOD\n")
    if ood:
        L.append("（OOD相对退化见processed/ood_drop.csv，式(35)）\n")
        L.append("| 方法 | ID SR | OOD-SNR | OOD-PFA | OOD-N | OOD-Prior | OOD-Reverb |")
        L.append("|---|---|---|---|---|---|---|")
        for algo in ["sequential", "random", "belief_greedy", "dqn", "ppo", "muzero"]:
            row = [ALGO_NAMES[algo]]
            for cond in ["id", "ood_snr", "ood_pfa", "ood_n", "ood_prior", "ood_reverb"]:
                mu, _, _ = _agg(ood, algo, "SR", cond)
                row.append(_fmt(mu))
            L.append("|" + "|".join(row) + "|")
    else:
        L.append("OOD实验未运行或无结果。\n")
    # 先验偏差
    L.append("\n## 14. Prior Bias\n")
    if pb:
        for pt in ["uniform", "correct", "biased", "adversarial"]:
            mu, _, _ = _agg(pb, "muzero", "SR", pt)
            bg, _, _ = _agg(pb, "belief_greedy", "SR", pt)
            L.append(f"- {pt}: MuZero SR={_fmt(mu)}, Belief-Greedy SR={_fmt(bg)}\n")
    else:
        L.append("先验偏差实验未运行或无结果。\n")
    # Level2
    L.append("## 15. Level 2 Validation\n")
    cons = _loadj(results_root / "processed" / "level2_consistency.json")
    if l2:
        L.append("| 方法 | Level1 SR(Medium) | Level2 SR(Medium) |")
        L.append("|---|---|---|")
        for algo in ["sequential", "belief_greedy", "ppo", "muzero"]:
            a, _, _ = _agg(main, algo, "SR", "medium")
            b, _, _ = _agg(l2, algo, "SR", "medium")
            L.append(f"|{ALGO_NAMES.get(algo, algo)}|{_fmt(a)}|{_fmt(b)}|")
        if cons:
            for d, v in cons.items():
                L.append(f"\n- {d}: 方法排序Spearman ρ={_fmt(v.get('spearman_rho'))}"
                         f" (p={_fmt(v.get('p_value'), 4)})\n")
    else:
        L.append("Level 2验证未运行或无结果。\n")
    # 计算成本
    L.append("## 16. Computational Cost\n")
    runtime_rows = []
    for algo in ["sequential", "random", "belief_greedy", "dqn", "ppo", "muzero"]:
        walls = []
        for f in (results_root / "checkpoints").glob(f"{algo}/main_seed*/"):
            p = list(f.glob("*_summary.json"))
            if p:
                d = read_json_safe(p[0])
                if d and d.get("training_wall_clock_sec"):
                    walls.append(d["training_wall_clock_sec"])
        lat, _, _ = _agg(main, algo, "MeanDecisionLatencyMs", "medium")
        runtime_rows.append((ALGO_NAMES[algo], walls, lat))
    for name, walls, lat in runtime_rows:
        L.append(f"- {name}: 训练总时长={_fmt(np.mean(walls) if walls else None, 0)}s"
                 f"（{len(walls)}种子），平均决策时延={_fmt(lat, 2)}ms\n")
    # 机理
    L.append("## 17. Mechanism Analysis\n")
    mech = results_root / "raw" / "mechanism"
    n_mech = len(list(mech.glob("*.json"))) if mech.exists() else 0
    L.append(f"机理数据文件：{n_mech}个（信念热图/搜索轨迹/MCTS访问分布/原始vs改进策略），"
             "对应figures/17-20系列图。\n")
    # 异常发现
    L.append("## 18. Unexpected Findings\n")
    L.append(_unexpected_findings(main, abl, tests))
    # 局限
    L.append("## 19. Limitations\n")
    L.append("- 二维离散方位空间、单静止目标、离散扇区动作（开题报告研究边界）。\n"
             "- 声传播为参数化简化模型（式(9)-(11)）；噪声/混响为统计模型。\n"
             "- Level 2为复基带统计一致实现，非全波形海底混响物理仿真。\n"
             "- 报告B节所列16项设计决策为开题报告留白处的文档化选择。\n")
    # 复现
    L.append("## 20. Reproducibility\n")
    L.append("- 复现命令：`python run_all_experiments.py --mode full`（断点续跑自动skip已完成）。\n"
             "- 全部配置：configs/*.yaml；实际使用配置与manifest：results/raw/manifests/。\n"
             "- 原始逐回合数据：results/raw/；质量检查：results/processed/quality_check.json。\n")
    if quality and not quality.get("passed"):
        L.append("\n> ⚠️ 质量检查存在问题项，见quality_check.json：\n")
        for i in quality.get("issues", []):
            L.append(f"> - {i}\n")
    (reports / "final_experiment_report.md").write_text("\n".join(L), encoding="utf-8")

    # ================= FINAL_SUMMARY.md（中文，非专业可读） =================
    S = _final_summary(results_root, main, abl, tests, ood, pb, l2, sanity, status, smoke)
    (results_root / "FINAL_SUMMARY.md").write_text(S, encoding="utf-8")
    return {"report": str(reports / "final_experiment_report.md"),
            "summary": str(results_root / "FINAL_SUMMARY.md")}


def _unexpected_findings(main, abl, tests) -> str:
    findings = []
    if main:
        m_mz = _agg(main, "muzero", "SR", "medium")[0]
        m_ppo = _agg(main, "ppo", "SR", "medium")[0]
        m_dqn = _agg(main, "dqn", "SR", "medium")[0]
        m_bg = _agg(main, "belief_greedy", "SR", "medium")[0]
        if m_mz is not None and m_bg is not None and m_mz <= m_bg + 0.01:
            findings.append("- MuZero在Medium上未超过Belief-Greedy（信念为该POMDP的"
                            "充分统计量→单步贪心已很强的结构性原因，见design_notes B6）。\n")
        if m_mz is not None and m_ppo is not None and m_mz <= m_ppo:
            findings.append("- MuZero未超过PPO。\n")
    if not findings:
        findings.append("- 无明显异常发现（或主实验未完成）。\n")
    return "".join(findings)


def _sample_efficiency(results_root: Path, algo: str, smoke: bool):
    for f in (results_root / "checkpoints" / algo).glob(f"main_seed*/"):
        p = list(f.glob(f"{algo}_main_seed*_summary.json"))
        if p:
            d = read_json_safe(p[0])
            if d:
                curve = d.get("sample_efficiency_curve") or []
                for rec in curve:
                    if rec.get("success_rate", 0) >= 0.90:
                        return rec.get("env_steps")
    return None


def read_json_safe(p: Path):
    try:
        return read_json(p)
    except Exception:
        return None


def _loadj(p: Path):
    if p.exists() and p.stat().st_size > 0:
        return read_json_safe(p)
    return None


def _final_summary(results_root, main, abl, tests, ood, pb, l2, sanity, status, smoke) -> str:
    def sr(algo, cond="medium", rows=main, variant=None):
        return _agg(rows, algo, "SR", cond, variant)[0]

    S = ["# FINAL_SUMMARY（最终交付摘要，中文，面向非专业读者）\n"]
    S.append("> 本摘要由程序从真实实验原始数据自动生成，不美化、不虚构。\n")
    S.append("\n## 1. 实验是否全部真正完成？\n")
    if status:
        done = [k for k, v in status.items() if v is True]
        failed = [k for k, v in status.items() if v is False]
        notrun = [k for k, v in status.items() if v is None]
        S.append(f"- 已完成 {len(done)} 项；失败 {len(failed)} 项：{failed or '无'}；"
                 f"未运行 {len(notrun)} 项：{notrun or '无'}。\n"
                 f"（逐项状态见 results/reports/experiment_status.md）\n")
    else:
        S.append("状态报告尚未生成。\n")
    S.append("\n## 2. 哪些实验失败/未运行？\n见上与experiment_status.md；"
             "如列出FAILED/NOT RUN即未完成，绝不显示completed。\n")
    S.append("\n## 3. MuZero最终表现如何？\n")
    if main and sr("muzero") is not None:
        S.append(f"- Easy SR={_fmt(sr('muzero', 'easy'))}, Medium SR={_fmt(sr('muzero'))}, "
                 f"Hard SR={_fmt(sr('muzero', 'hard'))}（Medium平均发现步数"
                 f"{_fmt(_agg(main, 'muzero', 'ADS', 'medium')[0], 1)}步）。\n")
    else:
        S.append("- 主实验未完成，无MuZero测试结果。\n")
    S.append("\n## 4-8. 与各基线比较（Medium难度成功率）\n")
    if main:
        for algo in ["sequential", "random", "belief_greedy", "dqn", "ppo"]:
            a, b = sr("muzero"), sr(algo)
            if a is not None and b is not None:
                diff = a - b
                word = "高于" if diff > 0.005 else ("低于" if diff < -0.005 else "持平于")
                S.append(f"- MuZero({_fmt(a)}) {word} {ALGO_NAMES[algo]}({_fmt(b)})，"
                         f"差值{_fmt(diff)}。\n")
    S.append("\n## 9. MCTS是否真正带来收益？\n")
    if abl:
        wo = [r for r in abl if r.get("condition") == "medium"
              and "use_mcts" in str(r.get("eval_options", ""))]
        a = sr("muzero")
        if wo and a is not None:
            b = float(np.mean([r["SR"] for r in wo if isinstance(r.get("SR"), (int, float))]))
            S.append(f"- 完整MuZero SR={_fmt(a)} vs 去除MCTS SR={_fmt(b)}："
                     f"{'MCTS带来提升' + _fmt(a - b) if a > b else 'MCTS无提升（差距' + _fmt(a - b) + '）'}"
                     f"（同训练模型，评测期开关，消融A）。\n")
        else:
            S.append("- 消融A未完成。\n")
    else:
        S.append("- 消融未运行。\n")
    S.append("\n## 10. Belief是否有作用？\n")
    if abl:
        wb = _agg(abl, "muzero", "SR", "medium", "wo_belief")[0]
        a = sr("muzero")
        if wb is not None and a is not None:
            S.append(f"- 完整{ _fmt(a)} vs w/o belief {_fmt(wb)}：信念输入贡献"
                     f"{_fmt(a - wb)}（消融D）。\n")
    S.append("\n## 11. GRU memory是否有作用？\n")
    if abl:
        wm = _agg(abl, "muzero", "SR", "medium", "wo_memory")[0]
        a = sr("muzero")
        if wm is not None and a is not None:
            S.append(f"- 完整{_fmt(a)} vs w/o memory {_fmt(wm)}：GRU记忆贡献"
                     f"{_fmt(a - wm)}（消融E）。\n")
    S.append("\n## 12. OOD泛化表现如何？\n")
    if ood:
        a_id = sr("muzero", "id", ood)
        for cond in ["ood_snr", "ood_pfa", "ood_n", "ood_prior", "ood_reverb"]:
            v = sr("muzero", cond, ood)
            if a_id is not None and v is not None:
                drop = (a_id - v) / a_id if a_id > 0 else None
                S.append(f"- {cond}: SR { _fmt(a_id)}→{_fmt(v)}"
                         f"（退化{_fmt((drop * 100 if drop is not None else 0), 1)}%）\n")
    else:
        S.append("- OOD未运行。\n")
    S.append("\n## 13. Level 1和Level 2结论是否一致？\n")
    cons = _loadj(results_root / "processed" / "level2_consistency.json")
    if cons:
        for d, v in cons.items():
            S.append(f"- {d}: Spearman ρ={_fmt(v.get('spearman_rho'))}——"
                     f"{'排序一致' if (v.get('spearman_rho') or 0) > 0.8 else '排序有差异'}\n")
    else:
        S.append("- Level2一致性分析无数据。\n")
    S.append("\n## 14. 是否达到统计显著？\n")
    if tests:
        mz_sig = [r for r in tests
                  if r.get("algo_a") == "muzero"
                  and str(r.get("significant_holm_0.05")) in ("True", "1.0")]
        S.append(f"- MuZero相关配对检验共{len([r for r in tests if r.get('algo_a') == 'muzero'])}项，"
                 f"Holm校正后显著{len(mz_sig)}项：\n")
        for r in mz_sig:
            S.append(f"  - {r.get('condition')}: vs {r.get('algo_b')} "
                     f"p={_fmt(r.get('p_holm'), 4)}\n")
    else:
        S.append("- 统计检验未运行。\n")
    S.append("\n## 15. 效应量多大？\n见effect_sizes.csv；上列显著项的效应量：\n")
    if tests:
        for r in tests:
            if r.get("algo_a") == "muzero" and str(r.get("significant_holm_0.05")) in ("True", "1.0"):
                S.append(f"  - vs {r.get('algo_b')}: {r.get('effect_size_name')}"
                         f"={_fmt(r.get('effect_size'))}\n")
    S.append("\n## 16. MuZero付出了多少额外计算成本？\n")
    if main:
        for algo in ["dqn", "ppo", "muzero"]:
            walls = []
            for f in (results_root / "checkpoints" / algo).glob("main_seed*/"):
                p = list(f.glob("*_summary.json"))
                if p:
                    d = read_json_safe(p[0])
                    if d and d.get("training_wall_clock_sec"):
                        walls.append(d["training_wall_clock_sec"])
            lat, _, _ = _agg(main, algo, "MeanDecisionLatencyMs", "medium")
            S.append(f"- {ALGO_NAMES[algo]}: 训练{_fmt(np.mean(walls) if walls else None, 0)}s/种子，"
                     f"决策时延{_fmt(lat, 1)}ms/步\n")
    S.append("\n## 17. 哪些结果最适合放进小论文？\n")
    S.append("- 表4/表7式主对比+统计检验（若MuZero显著优于基线）。\n"
             "- 消融A（MCTS贡献）与B/C（性能-计算权衡曲线）。\n"
             "- OOD退化表（表6）与先验纠偏KL曲线。\n"
             "- Level1↔Level2一致性（增强可信度）。\n"
             "- 若MuZero未显著领先：规划收益边界分析（开题报告风险4预案）同样是有效结论。\n")
    S.append("\n## 18. 哪些结果不支持原研究假设？\n")
    S.append(_unexpected_findings(main, abl, tests))
    S.append("\n## 19. 本实验最重要的真实结论（3-5条）\n")
    S.append(_key_conclusions(main, abl, tests, ood))
    S.append("\n## 20. 下一步建议补什么实验？\n")
    S.append("- 若主实验领先不显著：扩大种子数（≥10）、增加训练预算、或按开题报告"
             "风险4预案转向『规划有效条件与收益边界』分析。\n"
             "- 波形参数联合优化（开题报告扩展方向）。\n"
             "- 多目标/连续方位扩展。\n")
    return "\n".join(S)


def _key_conclusions(main, abl, tests, ood):
    lines = []
    if main:
        a = _agg(main, "muzero", "SR", "medium")[0]
        b = _agg(main, "belief_greedy", "SR", "medium")[0]
        if a is not None and b is not None:
            if a > b:
                lines.append(f"1. MuZero在Medium难度SR={_fmt(a)}，超过信念贪心{_fmt(b)}。\n")
            else:
                lines.append(f"1. MuZero({ _fmt(a)})未超过信念贪心({_fmt(b)})：信念为该"
                             "POMDP充分统计量，单步贪心已接近最优（诚实结构性结论）。\n")
    if abl:
        wo = [r["SR"] for r in abl if r.get("condition") == "medium"
              and "use_mcts" in str(r.get("eval_options", ""))
              and isinstance(r.get("SR"), (int, float))]
        a = _agg(main, "muzero", "SR", "medium")[0]
        if wo and a is not None:
            gain = a - float(np.mean(wo))
            lines.append(f"2. MCTS对决策期性能的净贡献：{ _fmt(gain)}"
                         f"（{'正' if gain > 0.005 else '非正'}）。\n")
    if ood:
        a_id = _agg(ood, "muzero", "SR", "id")[0]
        drops = []
        for cond in ["ood_snr", "ood_pfa", "ood_n", "ood_prior", "ood_reverb"]:
            v = _agg(ood, "muzero", "SR", cond)[0]
            if a_id and v is not None and a_id > 0:
                drops.append((cond, (a_id - v) / a_id))
        if drops:
            worst = max(drops, key=lambda x: x[1])
            lines.append(f"3. MuZero在OOD下最大退化出现在{worst[0]}"
                         f"（{ _fmt(worst[1] * 100, 1)}%）。\n")
    if tests:
        n_sig = len([r for r in tests
                     if r.get("algo_a") == "muzero"
                     and str(r.get("significant_holm_0.05")) in ("True", "1.0")])
        lines.append(f"4. MuZero与基线的差异经Holm校正后{n_sig}项统计显著"
                     f"（共{len([r for r in tests if r.get('algo_a') == 'muzero'])}项检验）。\n")
    if not lines:
        lines.append("- 主实验数据不完整，暂无法给出结论。\n")
    return "".join(lines[:5])
