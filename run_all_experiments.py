# -*- coding: utf-8 -*-
"""一键实验总入口（用户第29-30章）。

用法：
  python run_all_experiments.py --mode smoke   # 极小规模验证全pipeline
  python run_all_experiments.py --mode full    # 正式论文实验
  python run_all_experiments.py --mode full --stage train   # 分阶段运行
  阶段：selfcheck|baselines|train|main|ablation|robustness|prior_bias|level2|analysis
  已完成实验自动skip（断点续跑，第28章）。

流水线：环境自检→单元测试→sanity→场景/seed生成→基线→DQN/PPO/MuZero训练→
Validation选模→Main Test→Ablation→Robustness/OOD→Prior Bias→Level2→
Statistics→Figures→Tables→Quality Check→Final Report。
"""
import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from utils.config import load_config, load_section  # noqa: E402
from utils import hardware as hw  # noqa: E402
from utils.io_utils import (write_json, write_csv, ensure_results_tree,
                            write_text, read_json)  # noqa: E402

STAGES = ["selfcheck", "baselines", "train", "main", "ablation",
          "robustness", "prior_bias", "level2", "analysis"]


def stage_selfcheck(results: Path, smoke: bool) -> bool:
    print("\n" + "=" * 60)
    print("[STAGE 1/9] 环境自检：硬件/依赖/单元测试/sanity check")
    print("=" * 60)
    ensure_results_tree(results)
    hwd = hw.save_hardware_report(results / "raw" / "hardware.json")
    print(f"  hardware: {hwd.get('gpu_name') or 'CPU-only'}, "
          f"CUDA={hwd.get('cuda_available')}, RAM={hwd.get('ram_total_gb')}GB")
    print(f"  worker adaptation: {hw.adapt_workers(hwd)}")
    # 保存实际使用配置（第31章：config复制保存到results）
    cfg_dir_copy = results / "raw" / "configs"
    if cfg_dir_copy.exists():
        shutil.rmtree(cfg_dir_copy)
    shutil.copytree(ROOT / "configs", cfg_dir_copy)
    # 单元测试（pytest子进程）
    print("  running unit tests...")
    r = subprocess.run([sys.executable, "-m", "pytest", str(ROOT / "tests"), "-q",
                        "--tb=short"],
                       capture_output=True, text=True, timeout=600)
    passed = r.returncode == 0
    tail = (r.stdout or r.stderr).strip().split("\n")[-3:]
    print("  " + "\n  ".join(tail))
    write_json(results / "raw" / "unit_tests_passed.json",
               {"passed": passed, "stdout_tail": tail, "returncode": r.returncode})
    if not passed:
        print("  !! 单元测试未通过，禁止进入正式实验（第33章）")
        return False
    # Sanity check（第34章）
    from experiments.sanity import run_sanity_checks
    print("  running sanity checks...")
    res = run_sanity_checks(results, smoke)
    for k, v in res["checks"].items():
        print(f"  {'PASS' if v['ok'] else 'FAIL'} {k}")
    if not res["all_passed"]:
        print("  !! Sanity check未通过，禁止进入正式训练")
        return False
    # 保存校准记录（报告第4节）
    from env import level1_env
    from utils.config import load_config as _lc
    env_cfg = _lc("environment")["environment"]
    cal = {}
    for d in ["easy", "medium", "hard"]:
        p = level1_env.difficulty_params(env_cfg, d)
        cal[d] = {"N": p["n_sectors"], "pfa": p["pfa"],
                  "pd_target": p["calibration"]["pd_target"],
                  "pd_std": p["pd_distribution"]["std"],
                  "achieved": p["pd_distribution"]["mean"],
                  "snr_base_db": p["snr_base_db"]}
    write_json(results / "raw" / "calibration.json", cal)
    return True


def _device() -> str:
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def stage_train(results: Path, smoke: bool, seeds, resume: bool):
    print("\n" + "=" * 60)
    print("[STAGE 3/9] RL训练：DQN/PPO/MuZero（含消融变体D/E/F/G）")
    print("=" * 60)
    from experiments.main_experiment import run_training_stage
    env_cfg = load_config("environment")["environment"]
    dev = _device()
    # main变体（3算法）+ 消融重训变体（MuZero，表6 D/E/F/G；
    # v3盲搜索模式D不适用→由ablation.yaml的retrain_variants控制）
    from utils.config import load_section as _ls
    abl_cfg = _ls("ablation", smoke)
    retrain_variants = abl_cfg.get("retrain_variants",
                                   ["wo_belief", "wo_memory", "wo_domain_rand", "shaped"])
    sums = run_training_stage(results, env_cfg, dev, smoke, seeds,
                              ["dqn", "ppo", "muzero", "dqn_gru", "ppo_stack"],
                              variants=["main"])
    if retrain_variants:
        sums += run_training_stage(results, env_cfg, dev, smoke, seeds, ["muzero"],
                                   variants=retrain_variants)
    print(f"  training runs (incl. skipped-resume): {len(sums)}")
    return sums


def stage_main(results: Path, smoke: bool, seeds, resume: bool):
    print("\n" + "=" * 60)
    print("[STAGE 4/9] 主实验测试：6算法×Easy/Medium/Hard×固定场景")
    print("=" * 60)
    from experiments.main_experiment import run_main_test
    env_cfg = load_config("environment")["environment"]
    rows = run_main_test(results, env_cfg, _device(), smoke, seeds, resume)
    print(f"  main result rows: {len(rows)}")


def stage_ablation(results: Path, smoke: bool, seeds, resume: bool):
    print("\n" + "=" * 60)
    print("[STAGE 5/9] 消融实验A-G（表6）")
    print("=" * 60)
    from experiments.ablation import run_ablation
    env_cfg = load_config("environment")["environment"]
    diffs = ["easy", "medium", "hard"] if not smoke else ["medium"]
    rows = run_ablation(results, env_cfg, _device(), smoke, seeds, resume, diffs)
    print(f"  ablation rows: {len(rows)}")


def stage_robustness(results: Path, smoke: bool, seeds, resume: bool):
    print("\n" + "=" * 60)
    print("[STAGE 6/9] 鲁棒性sweep + OOD五类")
    print("=" * 60)
    from experiments.robustness import run_robustness_sweeps, run_ood
    env_cfg = load_config("environment")["environment"]
    r1 = run_robustness_sweeps(results, env_cfg, _device(), smoke, seeds, resume)
    r2 = run_ood(results, env_cfg, _device(), smoke, seeds, resume)
    print(f"  robustness rows: {len(r1)}, ood rows: {len(r2)}")


def stage_prior_bias(results: Path, smoke: bool, seeds, resume: bool):
    print("\n" + "=" * 60)
    print("[STAGE 7/9] 先验偏差实验（4类先验+KL纠偏）")
    print("=" * 60)
    from experiments.prior_bias import run_prior_bias
    env_cfg = load_config("environment")["environment"]
    rows = run_prior_bias(results, env_cfg, _device(), smoke, seeds, resume)
    print(f"  prior_bias rows: {len(rows)}")


def stage_level2(results: Path, smoke: bool, seeds, resume: bool):
    print("\n" + "=" * 60)
    print("[STAGE 8/9] Level 2信号级验证")
    print("=" * 60)
    from experiments.level2_validation import run_level2
    env_cfg = load_config("environment")["environment"]
    rows = run_level2(results, env_cfg, _device(), smoke, seeds, resume)
    print(f"  level2 rows: {len(rows)}")


def stage_analysis(results: Path, smoke: bool):
    print("\n" + "=" * 60)
    print("[STAGE 9/9] 统计分析→图表→表格→质量检查→最终报告")
    print("=" * 60)
    from experiments import statistics as stat_mod
    from experiments.figures import run_figures
    from experiments.tables import run_tables
    from experiments.report import run_report
    from experiments.quality_check import run_quality_check
    from experiments.status import run_status
    for stage in ["main", "ablation", "ood", "robustness", "prior_bias"]:
        r = stat_mod.run_statistics(results, stage=stage, smoke=smoke)
        print(f"  statistics[{stage}]: {r}")
    stat_mod.merge_statistics(results)
    _build_runtime_csv(results)
    qc = run_quality_check(results, smoke=smoke)
    print(f"  quality check: {'PASSED' if qc['passed'] else 'FAILED — 见issues'}")
    if not qc["passed"]:
        for i in qc["issues"][:10]:
            print("    ISSUE:", i)
        print("  !! 质量检查未通过——最终报告仍生成但明确标注问题（第27章）")
    fig = run_figures(results, smoke)
    n_figs = sum(1 for v in fig.values() if v is True)
    print(f"  figures generated: {n_figs}")
    try:
        from experiments.case_studies import run_case_studies
        cs = run_case_studies(results, smoke)
        print(f"  case studies: {cs}")
    except Exception as e:
        print(f"  case studies skipped: {e}")
    tab = run_tables(results, smoke)
    print(f"  tables generated: {sorted(tab)}")
    _build_experiment_summary(results, smoke)
    run_status(results, smoke)
    run_report(results, smoke)
    print("  reports: results/reports/final_experiment_report.md, results/FINAL_SUMMARY.md")


def _build_runtime_csv(results: Path):
    """runtime_results.csv：训练时长/更新次数/时延（第13章公平性要求）。"""
    import csv
    rows = []
    for f in sorted((results / "checkpoints").glob("*/*/")):
        for s in f.glob("*_summary.json"):
            try:
                d = read_json(s)
            except Exception:
                continue
            if not d or "algorithm" not in d:
                continue
            rows.append({
                "algorithm": d.get("algorithm"), "variant": d.get("variant"),
                "seed": d.get("seed"), "env_steps": d.get("env_steps"),
                "train_updates": d.get("train_updates"),
                "training_wall_clock_sec": d.get("training_wall_clock_sec"),
                "best_val_success_rate": d.get("best_val_success_rate"),
            })
    if rows:
        write_csv(results / "processed" / "runtime_results.csv", rows)


def _build_experiment_summary(results: Path, smoke: bool):
    """experiment_summary.csv/xlsx：全部实验汇总（可追溯到raw）。"""
    import csv
    import numpy as np
    frames = {
        "main": results / "processed" / "main_results.csv",
        "ablation": results / "processed" / "ablation_results.csv",
        "ood": results / "processed" / "ood_results.csv",
        "robustness": results / "processed" / "robustness_results.csv",
        "prior_bias": results / "processed" / "prior_bias_results.csv",
        "level2": results / "processed" / "level2_results.csv",
    }
    all_rows = []
    for stage, p in frames.items():
        if not p.exists():
            continue
        with open(p, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                r["experiment_stage"] = stage
                all_rows.append(r)
    if all_rows:
        write_csv(results / "experiment_summary.csv", all_rows)
        try:
            import pandas as pd
            pd.DataFrame(all_rows).to_excel(results / "experiment_summary.xlsx",
                                            index=False)
        except ImportError:
            pass
    # raw文件清单（可追溯性）
    raw_files = [str(p.relative_to(results)) for p in (results / "raw").rglob("*")
                 if p.is_file()]
    write_json(results / "raw" / "raw_file_inventory.json",
               {"n_files": len(raw_files), "files": raw_files[:5000]})


def main():
    ap = argparse.ArgumentParser(description="MuZero自适应声呐探测策略实验总入口")
    ap.add_argument("--mode", choices=["smoke", "full"], default="full")
    ap.add_argument("--stage", choices=STAGES + ["all"], default="all")
    ap.add_argument("--seeds", type=int, nargs="*", default=None,
                    help="训练种子覆盖（默认5个）")
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--config-dir", default=None,
                    help="配置目录覆盖（默认configs/；强化轮用configs_v2/）")
    ap.add_argument("--results-root", default=None,
                    help="结果根目录覆盖（默认results/；强化轮用results_v2/）")
    args = ap.parse_args()
    smoke = args.mode == "smoke"
    resume = not args.no_resume
    seeds = args.seeds or ([101] if smoke else [101, 102, 103, 104, 105])
    if args.config_dir:
        from utils.config import set_config_dir
        set_config_dir(ROOT / args.config_dir)
    results = Path(args.results_root) if args.results_root else ROOT / "results"
    results = results if results.is_absolute() else ROOT / results
    ensure_results_tree(results)

    t0 = time.time()
    print(f"模式: {args.mode} | 阶段: {args.stage} | 种子: {seeds} | resume: {resume}")
    print(f"结果目录: {results}")
    if smoke:
        print("SMOKE模式：极小训练/评测量，仅验证pipeline端到端可运行")
    else:
        print("FULL模式：正式论文实验规模（1e6步×5种子×3算法+消融+OOD+统计+图表）")

    stages = STAGES if args.stage == "all" else [args.stage]
    if "selfcheck" in stages:
        if not stage_selfcheck(results, smoke):
            print("\n自检失败，中止。")
            sys.exit(1)
    if "baselines" in stages or "main" in stages:
        # 基线在main阶段内完成评测（免训练）
        if "baselines" in stages and "main" not in stages:
            stage_main(results, smoke, seeds, resume)
    if "train" in stages:
        stage_train(results, smoke, seeds, resume)
    if "main" in stages:
        stage_main(results, smoke, seeds, resume)
    if "ablation" in stages:
        stage_ablation(results, smoke, seeds, resume)
    if "robustness" in stages:
        stage_robustness(results, smoke, seeds, resume)
    if "prior_bias" in stages:
        stage_prior_bias(results, smoke, seeds, resume)
    if "level2" in stages:
        stage_level2(results, smoke, seeds, resume)
    if "analysis" in stages:
        stage_analysis(results, smoke)

    print(f"\n总耗时 {time.time() - t0:.1f}s 完成：{args.stage}")
    print("主要交付：results/FINAL_SUMMARY.md, results/experiment_summary.xlsx, "
          "results/figures/, results/tables/, results/reports/final_experiment_report.md")


if __name__ == "__main__":
    main()
