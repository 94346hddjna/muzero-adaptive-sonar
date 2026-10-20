# -*- coding: utf-8 -*-
"""补评测驱动：POMCP（规划器免训练）+ dqn_gru（已训完5种子）在 v3 主测试场景上评测。

- 场景/协议与 run_main_test 完全一致（同一 evaluate_entry 入口、同一测试场景集、
  同一 raw 落盘路径），因此后续 run_all_experiments 的 main 阶段会按 resume 跳过。
- 产出行直接并入 results_v3/processed/main_results.csv（保持schema一致）。
用法：python tools/eval_new_baselines.py --config-dir configs_v3 --results-root results_v3
"""
import argparse
import csv
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-dir", default="configs_v3")
    ap.add_argument("--results-root", default="results_v3")
    ap.add_argument("--algos", nargs="*", default=["pomcp", "dqn_gru"])
    args = ap.parse_args()

    from utils.config import set_config_dir, load_config, load_section
    set_config_dir(ROOT / args.config_dir)
    results = ROOT / args.results_root

    from env import scenarios as scen
    from experiments import suite
    from utils.io_utils import write_csv

    env_cfg = load_config("environment")["environment"]
    eval_cfg = load_section("evaluation", False)
    train_cfg = load_section("training", False)
    n_ep = int(eval_cfg["test_episodes"])
    test_seed = int(train_cfg["test_scenario_seed"])
    seeds = [101, 102, 103, 104, 105]

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"

    out_csv = results / "processed" / "main_results.csv"
    rows = []
    if out_csv.exists():
        with open(out_csv, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    have = {(r["algorithm"], r["condition"], str(r["seed"]), r.get("variant", "main"))
            for r in rows}

    for diff in ["easy", "medium", "hard"]:
        scs = scen.get_or_create_scenarios(env_cfg, diff, n_ep, test_seed,
                                           "test", results)
        for algo in args.algos:
            for seed in seeds:
                if (algo, diff, str(seed), "main") in have:
                    print(f"[skip] {algo} {diff} seed{seed} 已有行")
                    continue
                rawf = results / "raw" / "main" / diff / f"{algo}_main_seed{seed}.jsonl"
                if rawf.exists() and rawf.stat().st_size > 0:
                    print(f"[skip] {algo} {diff} seed{seed} raw已存在（行将并入）")
                    continue
                entry = {"algorithm": algo, "seed": seed}
                if algo in suite.RL_ALGOS:
                    entry["alg_cfg"] = load_section(algo, False)
                t0 = time.time()
                row = suite.evaluate_entry(entry, scs, env_cfg, eval_cfg, results,
                                           device, stage="main", condition=diff,
                                           batch_size=16)
                rows.append({k: (f"{v:.6f}" if isinstance(v, float) else v)
                             for k, v in row.items()})
                write_csv(out_csv, rows)
                print(f"[done] {algo} {diff} seed{seed}: SR={row['SR']:.4f} "
                      f"({time.time()-t0:.0f}s)")
    print(f"全部完成，main_results.csv 现共 {len(rows)} 行")


if __name__ == "__main__":
    main()
