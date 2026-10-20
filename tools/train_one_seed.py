# -*- coding: utf-8 -*-
"""单种子训练驱动：用于并行补跑指定(algorithm, seed)。

用法：
    python tools/train_one_seed.py --algo ppo_stack --seed 103 \
        --config-dir configs_v3 --results-root results_v3

与 run_all_experiments.py 的 stage_train 路径完全一致（同一 train_rl 入口、
同一配置加载、同一 run_dir 布局、resume 兼容），仅把"循环"换成命令行参数，
以便多种子并行。命令行含 configs_v3 → watchdog3 识别为实验进程存活。
"""
import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--config-dir", default="configs_v3")
    ap.add_argument("--results-root", default="results_v3")
    ap.add_argument("--variant", default="main")
    args = ap.parse_args()

    from utils.config import set_config_dir, load_config, load_section
    set_config_dir(ROOT / args.config_dir)
    results = ROOT / args.results_root

    env_cfg = load_config("environment")["environment"]
    train_cfg = load_section("training", False)
    alg_cfg = load_section(args.algo, False)

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"

    run_dir = results / "checkpoints" / args.algo / f"{args.variant}_seed{args.seed}"
    from experiments.train_rl import train_rl
    t0 = time.time()
    s = train_rl(args.algo, env_cfg, alg_cfg, train_cfg, seed=args.seed,
                 run_dir=run_dir, variant=args.variant, smoke=False,
                 resume=True, device=device)
    print(f"[train_one_seed] {args.algo} seed{args.seed} 完成 "
          f"best_val={s.get('best_val_success_rate')} "
          f"wall={time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
