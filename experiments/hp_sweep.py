# -*- coding: utf-8 -*-
"""MuZero超参数扫描（强化轮v2；验证集驱动，绝不触碰Test）。

候选网格（聚焦最敏感的3个旋钮）：
  learning_rate ∈ {1e-4, 3e-4}
  num_simulations ∈ {50, 100}
  temperature_final ∈ {0.25}（固定）
每个配置短程训练300k步（seed=101）→ 固定验证集评估SR → 选最优写入configs_v2/muzero.yaml。
扫描的环境交互（300k×4）单独记录于results_v2/raw/hp_sweep.json（透明报告）。
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import json
import time
from typing import Dict

from utils.io_utils import write_json, read_json
from utils.config import load_section, set_config_dir, load_config


def run_hp_sweep(results_root: Path, config_dir: Path, smoke: bool = False,
                 probe_steps: int = 300000) -> Dict:
    from experiments.train_rl import train_rl
    env_cfg = load_config("environment")["environment"]
    base_mz = load_section("muzero")
    grid = []
    for lr in [1e-4, 3e-4]:
        for M in [50, 100]:
            grid.append({"learning_rate": lr, "num_simulations": M})
    if smoke:
        grid = grid[:2]
        probe_steps = 2000
    outcomes = []
    for i, over in enumerate(grid):
        mz = {**base_mz, **over}
        tag = f"hp{over['learning_rate']:.0e}_M{over['num_simulations']}"
        run_dir = results_root / "checkpoints" / "hp_sweep" / tag
        exp_id = f"muzero_hp_{tag}"
        # 复用train_rl：variant传'hp'（main语义但目录独立）
        t0 = time.time()
        summary = train_rl(
            "muzero", env_cfg, mz, {
                **load_section("training"),
                "total_env_steps": probe_steps,
                "validation_interval_steps": max(50000, probe_steps // 4),
                "checkpoint_interval_steps": max(50000, probe_steps // 4),
                "validation_episodes": 200,
                "num_parallel_envs": 64,
            },
            seed=101, run_dir=run_dir, variant="main", smoke=smoke,
            resume=True, device="cuda" if _has_cuda() else "cpu")
        outcomes.append({
            "config": over, "tag": tag,
            "best_val_success_rate": summary.get("best_val_success_rate"),
            "wall_clock_sec": round(time.time() - t0, 1),
            "env_steps": probe_steps,
        })
        print(f"  [hp_sweep {i+1}/{len(grid)}] {tag}: val_SR="
              f"{summary.get('best_val_success_rate')}")
    best = max(outcomes, key=lambda o: (o["best_val_success_rate"] or 0.0))
    report = {"grid": grid, "outcomes": outcomes, "best": best,
              "selection_metric": "validation_success_rate",
              "note": "仅用验证集选择；测试集不参与。扫描交互量已透明记录。"}
    results_root.mkdir(parents=True, exist_ok=True)
    write_json(results_root / "raw" / "hp_sweep.json", report)
    # 将最优配置写回configs_v2/muzero.yaml（学习率与模拟次数）
    import yaml
    mz_path = config_dir / "muzero.yaml"
    with open(mz_path, encoding="utf-8") as f:
        doc = yaml.safe_load(f)
    doc.setdefault("muzero", {})
    doc["muzero"]["learning_rate"] = best["config"]["learning_rate"]
    doc["muzero"]["num_simulations"] = best["config"]["num_simulations"]
    doc["muzero"]["_v2_note"] = (
        f"v2强化轮：超参由验证集sweep选定（best val SR={best['best_val_success_rate']}, "
        f"lr={best['config']['learning_rate']}, M={best['config']['num_simulations']}）；"
        "sweep过程见results_v2/raw/hp_sweep.json。")
    with open(mz_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(doc, f, allow_unicode=True, sort_keys=False)
    print(f"  [hp_sweep] BEST: {best['tag']} val_SR={best['best_val_success_rate']}"
          f" → 已写入{mz_path}")
    return report


def _has_cuda():
    import torch
    return torch.cuda.is_available()


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--config-dir", default="configs_v2")
    ap.add_argument("--results-root", default="results_v2")
    ap.add_argument("--probe-steps", type=int, default=300000)
    args = ap.parse_args()
    set_config_dir(ROOT / args.config_dir)
    run_hp_sweep(ROOT / args.results_root, ROOT / args.config_dir,
                 smoke=args.smoke, probe_steps=args.probe_steps)
