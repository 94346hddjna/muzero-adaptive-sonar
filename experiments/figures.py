# -*- coding: utf-8 -*-
"""论文图表自动生成（用户第21、22章；全部来自真实实验raw数据）。

22张图×{中文版, 英文版}×{PNG(300dpi), PDF}。坐标轴/单位/legend完整。
数据来源：results/raw/ 与 results/processed/（可追溯到逐回合记录）。
"""
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from utils.io_utils import read_json, read_jsonl, write_json

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

ALGOS = ["sequential", "random", "belief_greedy", "pomcp",
         "dqn", "ppo", "dqn_gru", "ppo_stack", "muzero"]
ALGO_LABELS_ZH = {"sequential": "顺序扫描", "random": "随机扫描", "belief_greedy": "信念贪心",
                  "pomcp": "POMCP(已知模型)", "dqn": "DQN", "ppo": "PPO",
                  "dqn_gru": "DQN-GRU(记忆)", "ppo_stack": "PPO-Stack(记忆)",
                  "muzero": "MuZero"}
ALGO_LABELS_EN = {"sequential": "Sequential", "random": "Random",
                  "belief_greedy": "Belief-Greedy", "pomcp": "POMCP (known model)",
                  "dqn": "DQN", "ppo": "PPO",
                  "dqn_gru": "DQN-GRU (memory)", "ppo_stack": "PPO-Stack (memory)",
                  "muzero": "MuZero (ours)"}
COLORS = {"sequential": "#8c8c8c", "random": "#c7b78a", "belief_greedy": "#e28c4a",
          "pomcp": "#4aa87c", "dqn": "#4a90d9", "ppo": "#7a57c7",
          "dqn_gru": "#2f6fb8", "ppo_stack": "#5b3fa8", "muzero": "#d94a4a"}
DIFFS_ZH = {"easy": "Easy", "medium": "Medium", "hard": "Hard"}


def _lang_labels(lang: str):
    if lang == "zh":
        return ALGO_LABELS_ZH, {"SR": "成功率 SR", "ADS": "平均发现步数",
                                "steps": "步数", "episodes": "回合",
                                "sr": "成功率", "time": "训练时间",
                                "latency": "决策时延 (ms)"}
    return ALGO_LABELS_EN, {"SR": "Success Rate", "ADS": "Avg. Detection Steps",
                            "steps": "Steps", "episodes": "Episodes",
                            "sr": "Success Rate", "time": "Training time",
                            "latency": "Decision latency (ms)"}


def _save(fig, out_dir: Path, name: str, dpi: int = 300, formats=("png", "pdf")):
    out_dir.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        fig.savefig(out_dir / f"{name}.{fmt}", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _load_processed(path: Path) -> List[Dict]:
    import csv
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


def _agg_algo(rows, condition, metric, algos=ALGOS):
    """按算法聚合（多seed均值±std）。"""
    means, stds = {}, {}
    for algo in algos:
        vals = [r[metric] for r in rows
                if r.get("condition") == condition and r.get("algorithm") == algo
                and isinstance(r.get(metric), (int, float))]
        if vals:
            means[algo] = float(np.mean(vals))
            stds[algo] = float(np.std(vals))
    return means, stds


def fig_training_curves(results_root: Path, fig_dir: Path, lang: str):
    """图1：训练曲线（episode return / SR vs env steps，6算法...RL三条+基线水平线）。"""
    labels, _ = _lang_labels(lang)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    log_dir = results_root / "raw" / "training_logs"
    ckpt_dir = results_root / "checkpoints"
    found = False
    for algo in ["dqn", "ppo", "dqn_gru", "ppo_stack", "muzero"]:
        for f in sorted(ckpt_dir.glob(f"{algo}/main_seed*/")):
            log_f = f / f"{algo}_main_seed*.csv"
            csvs = sorted(f.glob("*_training_log.csv"))
            if not csvs:
                continue
            rows = _load_processed(csvs[0])
            if not rows:
                continue
            found = True
            steps = [r["env_steps"] for r in rows]
            for ax_idx, metric in [(0, "episode_return"), (1, "success_rate")]:
                vals = [r.get(metric) for r in rows]
                axes[ax_idx].plot(steps, vals, label=labels[algo],
                                  color=COLORS[algo], alpha=0.75, linewidth=1.2)
    if not found:
        plt.close(fig)
        return False
    tzh = "环境交互步数" if lang == "zh" else "Environment steps"
    axes[0].set_xlabel(tzh)
    axes[0].set_ylabel("回合回报" if lang == "zh" else "Episode return")
    axes[1].set_xlabel(tzh)
    axes[1].set_ylabel("训练成功率" if lang == "zh" else "Training success rate")
    axes[1].set_ylim(0, 1.05)
    for ax in axes:
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    _save(fig, fig_dir, f"01_training_curves_{lang}")
    return True


def _bar_compare(rows, conditions, metric, fig_dir, name, lang, ylabel, ylim=None):
    labels, _ = _lang_labels(lang)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    width = 0.12
    x = np.arange(len(conditions))
    plotted = False
    for i, algo in enumerate(ALGOS):
        means = []
        stds = []
        for cond in conditions:
            m, s = _agg_algo(rows, cond, metric)
            means.append(m.get(algo, np.nan))
            stds.append(s.get(algo, 0.0))
        if np.all(np.isnan(means)):
            continue
        plotted = True
        ax.bar(x + (i - 2.5) * width, means, width, yerr=stds, capsize=2,
               label=labels[algo], color=COLORS[algo], alpha=0.85)
    if not plotted:
        plt.close(fig)
        return False
    ax.set_xticks(x)
    ax.set_xticklabels([DIFFS_ZH.get(c, c) for c in conditions])
    ax.set_ylabel(ylabel)
    if ylim:
        ax.set_ylim(*ylim)
    ax.legend(fontsize=8, ncol=3)
    ax.grid(alpha=0.3, axis="y")
    _save(fig, fig_dir, name)
    return True


def fig_main_comparisons(results_root: Path, fig_dir: Path, lang: str):
    """图2/3：SR与ADS对比（Easy/Medium/Hard）。"""
    rows = _load_processed(results_root / "processed" / "main_results.csv")
    if not rows:
        return {}
    made = {}
    made["02"] = _bar_compare(rows, ["easy", "medium", "hard"], "SR", fig_dir,
                              f"02_sr_comparison_{lang}", lang,
                              "成功率 SR" if lang == "zh" else "Success Rate",
                              ylim=(0, 1.05))
    made["03"] = _bar_compare(rows, ["easy", "medium", "hard"], "ADS", fig_dir,
                              f"03_ads_comparison_{lang}", lang,
                              "平均发现步数（仅成功回合）" if lang == "zh"
                              else "Avg. detection steps (successful only)")
    return made


def _sweep_plot(rows, sweep_key, xvalues, x_label, fig_dir, name, lang):
    """SR vs sweep变量曲线。"""
    labels, _ = _lang_labels(lang)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    plotted = False
    for algo in ALGOS:
        xs, ys, es = [], [], []
        for xv in xvalues:
            sub = [r for r in rows if str(r.get("sweep")) == sweep_key
                   and r.get("algorithm") == algo
                   and _match_x(r.get("condition"), xv)]
            if sub:
                vals = [r["SR"] for r in sub if isinstance(r.get("SR"), (int, float))]
                if vals:
                    xs.append(xv)
                    ys.append(float(np.mean(vals)))
                    es.append(float(np.std(vals)))
        if xs:
            plotted = True
            ax.errorbar(xs, ys, yerr=es, label=labels[algo], color=COLORS[algo],
                        marker="o", capsize=3, linewidth=1.5)
    if not plotted:
        plt.close(fig)
        return False
    ax.set_xlabel(x_label)
    ax.set_ylabel("成功率 SR" if lang == "zh" else "Success Rate")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    _save(fig, fig_dir, name)
    return True


def _match_x(cond, xv):
    if cond is None:
        return False
    c = str(cond)
    if c.startswith("snr_"):
        try:
            return abs(float(c[4:-2]) - float(xv)) < 1e-9
        except ValueError:
            return False
    if c.startswith("pfa_"):
        try:
            return abs(float(c[4:]) - float(xv)) < 1e-9
        except ValueError:
            return False
    if c.startswith("N"):
        try:
            return int(c[1:]) == int(xv)
        except ValueError:
            return False
    return False


def fig_robustness_sweeps(results_root: Path, fig_dir: Path, lang: str, smoke: bool):
    """图4/5/6：SR vs SNR / PFA / N。"""
    rows = _load_processed(results_root / "processed" / "robustness_results.csv")
    if not rows:
        return {}
    from utils.config import load_config
    sweeps = load_config("robustness", smoke)["robustness"]["sweeps"]
    made = {}
    made["04"] = _sweep_plot(rows, "snr", sweeps["snr_offsets_db"],
                             "SNR偏置 (dB)" if lang == "zh" else "SNR offset (dB)",
                             fig_dir, f"04_sr_vs_snr_{lang}", lang)
    made["05"] = _sweep_plot(rows, "pfa", sweeps["pfa_values"],
                             "虚警概率 $P_{FA}$" if lang == "zh" else "False alarm prob. $P_{FA}$",
                             fig_dir, f"05_sr_vs_pfa_{lang}", lang)
    made["06"] = _sweep_plot(rows, "n_sectors", sweeps["n_sectors_values"],
                             "扇区数 N" if lang == "zh" else "Number of sectors N",
                             fig_dir, f"06_sr_vs_n_{lang}", lang)
    return made


def _ablation_fig(rows, cond, filter_fn, xlabels, metric, fig_dir, name, lang,
                  ylabel="成功率 SR" if True else "", ylim=(0, 1.05)):
    labels, _ = _lang_labels(lang)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    xs, ys, es = [], [], []
    for xi, xl in enumerate(xlabels):
        sub = [r for r in rows if r.get("condition") == cond and filter_fn(r, xi)]
        vals = [r[metric] for r in sub if isinstance(r.get(metric), (int, float))]
        if vals:
            xs.append(xi)
            ys.append(float(np.mean(vals)))
            es.append(float(np.std(vals)))
    if not xs:
        plt.close(fig)
        return False
    ax.errorbar(xs, ys, yerr=es, marker="o", capsize=3, color=COLORS["muzero"],
                linewidth=1.5)
    ax.set_xticks(list(xs))
    ax.set_xticklabels([xlabels[i] for i in xs])
    ax.set_ylabel("成功率 SR" if lang == "zh" else "Success Rate")
    if ylim:
        ax.set_ylim(*ylim)
    ax.grid(alpha=0.3)
    _save(fig, fig_dir, name)
    return True


def fig_ablations(results_root: Path, fig_dir: Path, lang: str, smoke: bool):
    """图7-14：MuZero vs w/o MCTS、模拟次数、深度、信念/记忆/域随机/奖励消融。"""
    rows = _load_processed(results_root / "processed" / "ablation_results.csv")
    if not rows:
        return {}
    from utils.config import load_config
    abl = load_config("ablation", smoke)["ablation"]
    made = {}
    # 7: MuZero vs w/o MCTS（medium条件，主模型 vs woMCTS后缀）
    cond = "medium"
    main_m = [r["SR"] for r in rows if r.get("condition") == cond
              and r.get("algorithm") == "muzero" and r.get("variant") == "main"
              and not r.get("eval_options")]
    if not main_m:
        # v3消融阶段的main行均带eval_options（扫描行）→主基准回退到主实验干净值
        main_rows = _load_processed(results_root / "processed" / "main_results.csv")
        main_m = [r["SR"] for r in main_rows if r.get("condition") == cond
                  and r.get("algorithm") == "muzero" and r.get("variant") == "main"]
    wo_m = [r["SR"] for r in rows if r.get("condition") == cond
            and r.get("algorithm") == "muzero" and "use_mcts" in str(r.get("eval_options", ""))]
    labels, _ = _lang_labels(lang)
    if main_m and wo_m:
        fig, ax = plt.subplots(figsize=(5, 4.2))
        ax.bar([0, 1], [np.mean(main_m), np.mean(wo_m)],
               yerr=[np.std(main_m), np.std(wo_m)], capsize=4,
               color=[COLORS["muzero"], "#b04ad9"], alpha=0.85)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["MuZero\n(MCTS)" if lang == "zh" else "MuZero\n(with MCTS)",
                            "w/o MCTS" if lang == "zh" else "w/o MCTS"])
        ax.set_ylabel("成功率 SR" if lang == "zh" else "Success Rate")
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.3, axis="y")
        _save(fig, fig_dir, f"07_muzero_vs_womcts_{lang}")
        made["07"] = True
    # 8/9: 模拟次数 M 性能与时延
    msims = abl.get("simulations", [25, 50, 100, 200])
    made["08"] = _ablation_fig(
        rows, cond, lambda r, xi: "num_simulations" in str(r.get("eval_options", ""))
        and f'"num_simulations": {msims[xi]}' in str(r.get("eval_options", "")).replace(".0", "")
        and '"planning_depth": 5' in str(r.get("eval_options", "")).replace(".0", ""),
        [f"M={m}" for m in msims], "SR", fig_dir, f"08_sims_vs_performance_{lang}", lang)
    # 9: sims vs latency
    made["09"] = _ablation_fig(
        rows, cond, lambda r, xi: "num_simulations" in str(r.get("eval_options", ""))
        and f'"num_simulations": {msims[xi]}' in str(r.get("eval_options", "")).replace(".0", "")
        and '"planning_depth": 5' in str(r.get("eval_options", "")).replace(".0", ""),
        [f"M={m}" for m in msims], "MeanDecisionLatencyMs", fig_dir,
        f"09_sims_vs_latency_{lang}", lang,
        ylabel=None, ylim=None)
    # 10: 深度
    depths = abl.get("planning_depths", [1, 3, 5, 10])
    made["10"] = _ablation_fig(
        rows, cond, lambda r, xi: "planning_depth" in str(r.get("eval_options", ""))
        and f'"planning_depth": {depths[xi]}' in str(r.get("eval_options", "")).replace(".0", "")
        and '"num_simulations": 100' in str(r.get("eval_options", "")).replace(".0", ""),
        [f"D={d}" for d in depths], "SR", fig_dir, f"10_depth_vs_performance_{lang}", lang)
    # 11-14: 信念/记忆/域随机化/奖励（variant对比）
    pairs = [("11_belief_ablation", ["main", "wo_belief"],
              ["MuZero", "w/o belief"], "#b04ad9"),
             ("12_memory_ablation", ["main", "wo_memory"],
              ["MuZero", "w/o memory"], "#4ad9b0"),
             ("13_domainrand_ablation", ["main", "wo_domain_rand"],
              ["MuZero", "w/o dom. rand."], "#d9b04a"),
             ("14_reward_ablation", ["main", "shaped"],
              ["sparse (式16)", "shaped (PBRS)"], "#4ad99a")]
    for name, variants, vlabels, color in pairs:
        means = []
        stds = []
        for v in variants:
            vals = [r["SR"] for r in rows if r.get("condition") == cond
                    and r.get("algorithm") == "muzero" and r.get("variant") == v]
            if not vals:
                means.append(np.nan)
                stds.append(0)
            else:
                means.append(float(np.mean(vals)))
                stds.append(float(np.std(vals)))
        if np.all(np.isnan(means)):
            continue
        fig, ax = plt.subplots(figsize=(5, 4.2))
        ax.bar(range(len(variants)), means, yerr=stds, capsize=4, color=color, alpha=0.85)
        ax.set_xticks(range(len(variants)))
        ax.set_xticklabels(vlabels)
        ax.set_ylabel("成功率 SR" if lang == "zh" else "Success Rate")
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.3, axis="y")
        _save(fig, fig_dir, f"{name}_{lang}")
        made[name[:2]] = True
    return made


def fig_ood_prior(results_root: Path, fig_dir: Path, lang: str):
    """图15: ID vs OOD；图16: 先验偏差鲁棒性。"""
    made = {}
    rows = _load_processed(results_root / "processed" / "ood_results.csv")
    labels, _ = _lang_labels(lang)
    if rows:
        conds = sorted({r["condition"] for r in rows})
        means_d = {}
        for cond in conds:
            m, _ = _agg_algo(rows, cond, "SR")
            means_d[cond] = m
        fig, ax = plt.subplots(figsize=(8.5, 4.5))
        width = 0.12
        x = np.arange(len(conds))
        for i, algo in enumerate(ALGOS):
            ys = [means_d.get(c, {}).get(algo, np.nan) for c in conds]
            if np.all(np.isnan(ys)):
                continue
            ax.bar(x + (i - 2.5) * width, ys, width, label=labels[algo],
                   color=COLORS[algo], alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels([c if c == "id" else c.replace("ood_", "OOD-") for c in conds],
                           fontsize=8)
        ax.set_ylabel("成功率 SR" if lang == "zh" else "Success Rate")
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=8, ncol=3)
        ax.grid(alpha=0.3, axis="y")
        _save(fig, fig_dir, f"15_id_vs_ood_{lang}")
        made["15"] = True
    # 16: 先验偏差
    pb = _load_processed(results_root / "processed" / "prior_bias_results.csv")
    if pb:
        conds = sorted({r.get("prior_type", r.get("condition")) for r in pb})
        fig, ax = plt.subplots(figsize=(8, 4.5))
        width = 0.12
        x = np.arange(len(conds))
        for i, algo in enumerate(ALGOS):
            ys = []
            for c in conds:
                vals = [r["SR"] for r in pb
                        if (r.get("prior_type", r.get("condition")) == c
                            and r.get("algorithm") == algo
                            and isinstance(r.get("SR"), (int, float)))]
                ys.append(float(np.mean(vals)) if vals else np.nan)
            if np.all(np.isnan(ys)):
                continue
            ax.bar(x + (i - 2.5) * width, ys, width, label=labels[algo],
                   color=COLORS[algo], alpha=0.85)
        ax.set_xticks(x)
        zh = {"uniform": "均匀先验", "correct": "正确先验", "biased": "有偏先验",
              "adversarial": "对抗先验"}
        ax.set_xticklabels([zh.get(c, str(c)) if lang == "zh" else str(c) for c in conds])
        ax.set_ylabel("成功率 SR" if lang == "zh" else "Success Rate")
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=8, ncol=3)
        ax.grid(alpha=0.3, axis="y")
        _save(fig, fig_dir, f"16_prior_bias_{lang}")
        made["16"] = True
    return made


def fig_belief_evolution(results_root: Path, fig_dir: Path, lang: str):
    """图17：信念演化热图（MuZero典型回合）；KL纠偏曲线。"""
    made = {}
    # 信念热图：来自histories
    hdir = results_root / "raw" / "histories" / "prior_bias"
    target_algo = "muzero"
    files = sorted(hdir.glob(f"{target_algo}_main_seed101_*.json")) if hdir.exists() else []
    if files:
        recs = read_json(files[0])
        succ = [r for r in recs if r["success"]]
        if succ:
            r = succ[0]
            hist = r["history"][:20]
            N = r["n_sectors"]
            bmat = np.zeros((len(hist) + 1, N))
            # t=0的b0：history[0].belief是第一步之后的信念；b0从scenario推断不可得→用第一步前
            for t, h in enumerate(hist):
                bmat[t + 1] = np.asarray(h["belief"])
            # t=0: 均匀近似（无b0记录时画空行0）
            fig, ax = plt.subplots(figsize=(7, 5))
            im = ax.imshow(bmat, aspect="auto", cmap="hot", vmin=0, vmax=1)
            ax.axhline(y=0, color="white", linewidth=0)
            ax.set_xlabel("扇区编号" if lang == "zh" else "Sector index")
            ax.set_ylabel("探测步 t" if lang == "zh" else "Ping step t")
            # 标注真实目标
            ax.plot(r["target"], 0, marker="*", color="cyan", markersize=12,
                    label="target" if lang == "en" else "目标")
            ax.legend()
            fig.colorbar(im, ax=ax, label="信念 b_t(i)")
            _save(fig, fig_dir, f"17_belief_heatmap_{lang}")
            made["17"] = True
    # KL纠偏曲线
    kdir = results_root / "processed" / "prior_bias_kl"
    if kdir.exists():
        kl_files = sorted(kdir.glob("muzero_main_seed101_*.json"))
        if kl_files:
            fig, ax = plt.subplots(figsize=(7, 4.5))
            for f in kl_files:
                d = read_json(f)
                curve = [v for v in d["kl_mean_curve"] if v is not None]
                if curve:
                    ax.plot(range(len(curve)), curve, marker="o", markersize=3,
                            label=str(d["prior_type"]), linewidth=1.5)
            ax.set_xlabel("探测步 t" if lang == "zh" else "Ping step t")
            ax.set_ylabel("$D_{KL}(P_{true}\\,\\|\\,b_t)$" if lang == "zh"
                          else "$D_{KL}(P_{true}\\|b_t)$")
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)
            _save(fig, fig_dir, f"17b_kl_correction_{lang}")
    return made


def fig_mechanism(results_root: Path, fig_dir: Path, lang: str):
    """图18/19/20：搜索轨迹/MCTS根访问分布/raw vs improved策略。"""
    made = {}
    mdir = results_root / "raw" / "mechanism"
    files = sorted(mdir.glob("muzero_main_seed*_main_medium.json")) if mdir.exists() else []
    if not files:
        return made
    mech = read_json(files[0])
    sidx = sorted(mech.keys(), key=lambda k: int(k))[:1]
    if not sidx:
        return made
    recs = mech[sidx[0]]
    # 18: 搜索轨迹（动作序列 + 检测）
    fig, ax = plt.subplots(figsize=(8, 3.8))
    acts = [r["action"] for r in recs]
    dets = [r["detection"] for r in recs]
    xs = range(len(acts))
    ax.step(xs, acts, where="mid", color="#4a90d9", linewidth=1.5,
            label="动作（波束指向扇区）" if lang == "zh" else "Action (beam sector)")
    det_x = [i for i, d in enumerate(dets) if d]
    det_y = [acts[i] for i in det_x]
    ax.scatter(det_x, det_y, marker="v", color="red", zorder=5, s=60,
               label="检出" if lang == "zh" else "Detection")
    ax.set_xlabel("探测步 t" if lang == "zh" else "Ping step t")
    ax.set_ylabel("扇区编号" if lang == "zh" else "Sector index")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    _save(fig, fig_dir, f"18_search_trajectory_{lang}")
    made["18"] = True
    # 19: MCTS根访问分布（首步+中步）
    steps_with_visits = [r for r in recs if r.get("visits")]
    if steps_with_visits:
        fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
        for ax_i, r in zip(axes, [steps_with_visits[0], steps_with_visits[-1]]):
            v = np.asarray(r["visits"], dtype=float)
            ax.bar(range(len(v)), v, color="#d94a4a", alpha=0.8)
            ax.set_xlabel("扇区编号" if lang == "zh" else "Sector index")
            ax.set_ylabel("访问次数 N(s,a)" if lang == "zh" else "Visits N(s,a)")
            ax.set_title(f"t={r['t']}")
        _save(fig, fig_dir, f"19_mcts_visits_{lang}")
        made["19"] = True
    # 20: raw policy vs improved policy
    steps_with_both = [r for r in recs if r.get("raw_policy") is not None
                       and r.get("policy_target") is not None]
    if steps_with_both:
        r = steps_with_both[0]
        N = len(r["raw_policy"])
        x = np.arange(N)
        fig, ax = plt.subplots(figsize=(8, 3.8))
        ax.bar(x - 0.2, r["raw_policy"], 0.4, label="网络先验 $p(a|s)$",
               color="#4a90d9", alpha=0.85)
        ax.bar(x + 0.2, r["policy_target"], 0.4, label="MCTS改进 $\\pi(a|s)$",
               color="#d94a4a", alpha=0.85)
        ax.set_xlabel("扇区编号" if lang == "zh" else "Sector index")
        ax.set_ylabel("概率" if lang == "zh" else "Probability")
        ax.legend(fontsize=8)
        _save(fig, fig_dir, f"20_raw_vs_improved_{lang}")
        made["20"] = True
    return made


def fig_tradeoff_and_level2(results_root: Path, fig_dir: Path, lang: str):
    """图21: 成功率-时延权衡；图22: Level1 vs Level2。"""
    made = {}
    rows = _load_processed(results_root / "processed" / "main_results.csv")
    labels, _ = _lang_labels(lang)
    if rows:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for algo in ALGOS:
            xs = [r.get("MeanDecisionLatencyMs") for r in rows
                  if r.get("algorithm") == algo and r.get("condition") == "medium"
                  and isinstance(r.get("MeanDecisionLatencyMs"), (int, float))]
            ys = [r.get("SR") for r in rows
                  if r.get("algorithm") == algo and r.get("condition") == "medium"
                  and isinstance(r.get("MeanDecisionLatencyMs"), (int, float))]
            if xs:
                ax.errorbar([np.mean(xs)], [np.mean(ys)],
                            xerr=[np.std(xs)], yerr=[np.std(ys)],
                            marker="o", capsize=4, label=labels[algo],
                            color=COLORS[algo], linestyle="none", markersize=9)
        ax.set_xlabel("平均决策时延 (ms)" if lang == "zh" else "Mean decision latency (ms)")
        ax.set_ylabel("成功率 SR" if lang == "zh" else "Success Rate")
        ax.set_xscale("log")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        _save(fig, fig_dir, f"21_sr_latency_tradeoff_{lang}")
        made["21"] = True
    l1 = rows
    l2 = _load_processed(results_root / "processed" / "level2_results.csv")
    if l2 and l1:
        algos = [a for a in ALGOS if any(r["algorithm"] == a for r in l2)]
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
        for ax, (data, title) in zip(axes, [
                (l1, "Level 1 (抽象环境)" if lang == "zh" else "Level 1 (abstract)"),
                (l2, "Level 2 (信号级)" if lang == "zh" else "Level 2 (signal-level)")]):
            means = []
            for a in algos:
                vals = [r["SR"] for r in data
                        if r.get("algorithm") == a and isinstance(r.get("SR"), (int, float))]
                means.append(np.mean(vals) if vals else np.nan)
            if not np.all(np.isnan(means)):
                ax.bar(range(len(algos)), means,
                       color=[COLORS[a] for a in algos], alpha=0.85)
                ax.set_xticks(range(len(algos)))
                ax.set_xticklabels([labels[a] for a in algos], fontsize=7, rotation=20)
            ax.set_title(title)
            ax.set_ylabel("成功率 SR" if lang == "zh" else "Success Rate")
            ax.set_ylim(0, 1.05)
            ax.grid(alpha=0.3, axis="y")
        _save(fig, fig_dir, f"22_level1_vs_level2_{lang}")
        made["22"] = True
    return made


def run_figures(results_root: Path, smoke: bool = False) -> Dict:
    if not HAS_MPL:
        return {"error": "matplotlib unavailable"}
    _setup_zh_font()
    fig_dir = results_root / "figures"
    report = {}
    for lang in ["zh", "en"]:
        prefix = "zh" if lang == "zh" else "en"
        report[f"training_{lang}"] = fig_training_curves(results_root, fig_dir, lang)
        report.update({f"{k}_{lang}": v for k, v in
                       fig_main_comparisons(results_root, fig_dir, lang).items()})
        report.update({f"{k}_{lang}": v for k, v in
                       fig_robustness_sweeps(results_root, fig_dir, lang, smoke).items()})
        report.update({f"{k}_{lang}": v for k, v in
                       fig_ablations(results_root, fig_dir, lang, smoke).items()})
        report.update({f"{k}_{lang}": v for k, v in
                       fig_ood_prior(results_root, fig_dir, lang).items()})
        report.update({f"{k}_{lang}": v for k, v in
                       fig_belief_evolution(results_root, fig_dir, lang).items()})
        report.update({f"{k}_{lang}": v for k, v in
                       fig_mechanism(results_root, fig_dir, lang).items()})
        report.update({f"{k}_{lang}": v for k, v in
                       fig_tradeoff_and_level2(results_root, fig_dir, lang).items()})
    write_json(results_root / "processed" / "figures_manifest.json", report)
    return report


def _setup_zh_font():
    try:
        from matplotlib import font_manager
        for name in ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC"]:
            try:
                font_manager.findfont(name, fallback_to_default=False)
                matplotlib.rcParams["font.sans-serif"] = [name]
                matplotlib.rcParams["axes.unicode_minus"] = False
                return
            except Exception:
                continue
    except Exception:
        pass
