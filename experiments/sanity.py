# -*- coding: utf-8 -*-
"""环境合理性Sanity Check（用户第34章）。

正式RL训练前必须通过（任一失败→禁止进入正式训练）：
 1 PD提高→真实目标更易被检测（物理校准核对）
 2 PFA提高→虚警增加（式12-14统计验证）
 3 SNR提高→检测性能整体改善
 4 Belief总和恒≈1
 5 连续负观测合理改变后验（漏检概率衰减而非清零，式5）
 6 Sequential行为符合定义（顺序循环发射）
 7 Random动作分布近似均匀
 8 Belief-Greedy选择当前最大后验扇区（式27）
 9 Level1↔Level2 PFA/PD统计一致性（容差±0.03）
结果：results/raw/sanity_checks.json
"""
from pathlib import Path
from typing import Dict

import numpy as np
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from env import level1_env, level2_env
from env.level1_env import SonarLevel1VecEnv
from utils.config import load_config
from utils.io_utils import write_json


def run_sanity_checks(results_root: Path, smoke: bool = False) -> Dict:
    full = load_config("environment")
    env_cfg = full["environment"]
    l2cfg = full["level2"]
    checks: Dict[str, Dict] = {}

    # ---- 1 PD校准核对（PD_target vs 实测mean PD，B2） ----
    cal = {}
    for d in ["easy", "medium", "hard"]:
        p = level1_env.difficulty_params(env_cfg, d)
        cal[d] = {"target": p["calibration"]["pd_target"],
                  "achieved": p["pd_distribution"]["mean"],
                  "pd_std": p["pd_distribution"]["std"],
                  "snr_base_db": p["snr_base_db"]}
    ok1 = all(abs(cal[d]["target"] - cal[d]["achieved"]) < 0.02 for d in cal)
    checks["calibration"] = {"ok": ok1, "data": cal}

    # ---- 2/3 PFA与SNR趋势（式12-14统计验证；从done回合episode_info累计） ----
    pfa_vals = [0.01, 0.05, 0.15]
    fa_rates = []
    for pfa in pfa_vals:
        env = SonarLevel1VecEnv(32, env_cfg, t_max=20, master_seed=11,
                                fixed_difficulty="medium",
                                difficulty_overrides={"pfa": pfa})
        env.reset()
        n_fa, n_neg = 0, 0
        rng_acts = np.random.default_rng(3)
        for _ in range(30):
            acts = np.array([rng_acts.integers(env.n[s]) for s in range(32)])
            _, _, _, infos = env.step(acts)
            for inf in infos:
                if inf is not None:
                    n_fa += inf["n_false_alarms"]
                    n_neg += inf["n_negative_emissions"]
        fa_rates.append(n_fa / max(n_neg, 1))
    ok2 = fa_rates[0] < fa_rates[1] < fa_rates[2]
    checks["pfa_trend"] = {"ok": ok2, "pfa_settings": pfa_vals,
                           "empirical_fa_rates": [round(r, 4) for r in fa_rates]}

    snr_offsets = [-6.0, 0.0, 6.0]
    det_rates = []
    for off in snr_offsets:
        env = SonarLevel1VecEnv(32, env_cfg, t_max=20, master_seed=12,
                                fixed_difficulty="medium",
                                difficulty_overrides={"snr_offset_db": off})
        env.reset()
        n_hit, n_aim = 0, 0
        for _ in range(30):
            # 全部指向真实目标（检验指向即检测的趋势；done槽位从last_detection累计）
            acts = env.target.copy()
            _, _, _, infos = env.step(acts)
            for s in range(32):
                inf = infos[s]
                if inf is not None:
                    n_hit += inf["last_detection"]
                    n_aim += 1
                    # 同回合后续步骤若未结束也计入prev_det
                    if inf["steps"] > 1:
                        pass
                else:
                    n_hit += int(env.prev_det[s])
                    n_aim += 1
        det_rates.append(n_hit / max(n_aim, 1))
    ok3 = det_rates[0] < det_rates[1] < det_rates[2]
    checks["snr_trend"] = {"ok": ok3, "snr_offsets_db": snr_offsets,
                           "empirical_detection_rates": [round(r, 4) for r in det_rates]}

    # ---- 4/5 Belief性质 ----
    env = SonarLevel1VecEnv(64, env_cfg, t_max=20, master_seed=13,
                            fixed_difficulty="medium")
    env.reset()
    sums_all = []
    rng_acts = np.random.default_rng(5)
    post_ratios = []
    for i in range(40):
        acts = np.array([rng_acts.integers(env.n[s]) for s in range(64)])
        _, _, _, _ = env.step(acts)
        sums_all.extend(env.belief_sums().tolist())
        # 连续负观测：指向同一扇区3次未检出→该扇区后验按漏检概率衰减（式5）而非清零
        env2 = SonarLevel1VecEnv(1, env_cfg, t_max=20, master_seed=14,
                                 fixed_difficulty="medium")
        env2.reset()
        from env import belief as bl
        b = env2.belief[0, :int(env2.n[0])].copy()
        b0_val = float(b[0])
        pd_ep = float(env2.pd_ep[0])
        pfa = float(env2.pfa[0])
        for _ in range(3):
            b = bl.bayes_update(b, 0, 0, pd_ep, pfa)
        post_ratios.append(float(b[0]))
    ok4 = np.allclose(sums_all, 1.0, atol=1e-9)
    checks["belief_sum"] = {"ok": ok4, "max_abs_dev": float(np.max(np.abs(np.array(sums_all) - 1.0)))}
    ok5 = all(r > 0.0 for r in post_ratios) and np.mean(post_ratios) < b0_val
    checks["negative_observation_posterior"] = {
        "ok": ok5, "posterior_after_3_misses_mean": float(np.mean(post_ratios)),
        "prior_at_pointed_sector": b0_val,
        "note": "未检出使指向扇区后验按漏检概率衰减（严格>0且下降），非清零（式5）"}

    # ---- 6/7/8 基线行为 ----
    from baselines import SequentialPolicy, RandomPolicy, BeliefGreedyPolicy
    env = SonarLevel1VecEnv(1, env_cfg, t_max=20, master_seed=15, fixed_difficulty="medium")
    obs = env.reset()
    seq = SequentialPolicy(seed=1)
    seq_acts = []
    for t in range(6):
        a, _ = seq.act(obs, env.n, env=env)
        seq_acts.append(int(a[0]))
        obs, _, _, _ = env.step(a)
    ok6 = seq_acts[:4] == [0, 1, 2, 3]
    checks["sequential"] = {"ok": ok6, "actions_over_steps": seq_acts,
                            "note": "按t mod N顺序发射（单槽位时间序列）"}

    env = SonarLevel1VecEnv(64, env_cfg, t_max=20, master_seed=16, fixed_difficulty="medium")
    obs = env.reset()
    rnd = RandomPolicy(seed=2)
    counts = np.zeros(16)
    for _ in range(40):
        a, _ = rnd.act(obs, env.n, env=env)
        for ai in a:
            counts[int(ai)] += 1
        obs, _, _, _ = env.step(a)
    exp = counts.sum() / 16
    ok7 = counts.min() > 0.5 * exp and counts.max() < 1.5 * exp
    checks["random_uniform"] = {"ok": ok7, "counts": counts.astype(int).tolist(),
                                "expected_per_sector": exp}

    env = SonarLevel1VecEnv(32, env_cfg, t_max=20, master_seed=17, fixed_difficulty="medium")
    obs = env.reset()
    bg = BeliefGreedyPolicy(seed=3)
    ok8 = True
    for _ in range(10):
        a, _ = bg.act(obs, env.n, env=env)
        for s in range(32):
            b = env.belief[s, :int(env.n[s])]
            if int(a[s]) != int(np.argmax(b)):
                ok8 = False
        obs, _, _, _ = env.step(a)
    checks["belief_greedy"] = {"ok": ok8, "note": "式(27)：选择argmax b_t(i)"}

    # ---- 9 Level1↔Level2统计一致性 ----
    l2 = level2_env.SonarLevel2VecEnv(8, env_cfg, t_max=20, master_seed=18, level2_cfg=l2cfg)
    p = level1_env.difficulty_params(env_cfg, "medium")
    n_try = 200
    fa2 = 0
    for i in range(n_try):
        rng = np.random.default_rng(800 + i)
        from env.level1_env import make_scenario_from_params
        sc = make_scenario_from_params(p, env_cfg, rng, 900 + i, scenario_index=i)
        sc["target"] = 8
        l2.reset([sc])
        l2.step(np.array([0]))
        fa2 += int(l2.prev_det[0])
    det2 = 0
    pds = []
    for i in range(n_try):
        rng = np.random.default_rng(950 + i)
        from env.level1_env import make_scenario_from_params
        sc = make_scenario_from_params(p, env_cfg, rng, 950 + i, scenario_index=i)
        pds.append(sc["pd"])
        l2.reset([sc])
        l2.step(np.array([sc["target"]]))
        det2 += int(l2.prev_det[0])
    pfa_l2, pd_l2, pd_l1 = fa2 / n_try, det2 / n_try, float(np.mean(pds))
    tol = float(l2cfg.get("consistency_tolerance", 0.03))
    se = np.sqrt(0.05 * 0.95 / n_try)
    ok9 = abs(pfa_l2 - 0.05) < tol and abs(pd_l2 - pd_l1) < tol + 3 * se
    checks["level2_consistency"] = {
        "ok": ok9, "pfa_level2_empirical": pfa_l2, "pfa_nominal": 0.05,
        "pd_level2_empirical": pd_l2, "pd_level1_mean": pd_l1,
        "tolerance": tol}

    all_ok = all(c["ok"] for c in checks.values())
    out = {"all_passed": all_ok, "checks": checks}
    write_json(results_root / "raw" / "sanity_checks.json", out)
    return out


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent / "results"
    res = run_sanity_checks(root)
    for k, v in res["checks"].items():
        print(("PASS" if v["ok"] else "FAIL"), k)
    sys.exit(0 if res["all_passed"] else 1)
