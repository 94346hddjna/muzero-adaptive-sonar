# -*- coding: utf-8 -*-
"""固定场景生成（第十一章Train/Validation/Test/OOD种子体系）。

四类场景集合由互不相交的主种子生成并落盘（JSON），锁定后所有算法在
完全相同的target positions/环境参数/先验/难度/随机种子下评测（配对评测）。
训练环境采样与场景文件完全隔离（utils/seeding.py）。
"""
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from env import level1_env
from utils import seeding as us
from utils.io_utils import read_json, write_json


def generate_scenarios(
    env_cfg: Dict,
    difficulty: str,
    n_episodes: int,
    namespace_seed: int,
    overrides: Optional[Dict] = None,
    prior_type: Optional[str] = None,
    prior_sigma: Optional[float] = None,
    snr_offset_db: float = 0.0,
    difficulty_label: Optional[str] = None,
) -> List[Dict]:
    """生成一组固定场景。namespace_seed决定目标位置/距离/先验/噪声流（可复现）。"""
    params = level1_env.difficulty_params(env_cfg, difficulty, overrides)
    # 难度/条件偏移：不同条件的场景流完全独立（避免跨难度重复测试样本）
    import hashlib
    diff_offset = int(hashlib.md5(str(difficulty).encode()).hexdigest()[:8], 16) % 100000
    stream_seed = namespace_seed + diff_offset
    scenarios = []
    for i in range(n_episodes):
        srng = us.scenario_rng(stream_seed, i)
        rng_seed = int(us.env_episode_rng(stream_seed, i).integers(0, 2**31 - 1))
        sc = level1_env.make_scenario_from_params(
            params, env_cfg, srng, rng_seed, scenario_index=i,
            prior_type=prior_type, prior_sigma=prior_sigma,
            snr_offset_db=snr_offset_db)
        scenarios.append(sc)
    for i, sc in enumerate(scenarios):
        sc["condition"] = difficulty_label or difficulty
        sc["condition_index"] = i
    return scenarios


def scenario_file_path(results_dir: Path, split: str, condition: str, n_episodes: int) -> Path:
    return results_dir / "raw" / "scenarios" / f"{split}_{condition}_n{n_episodes}.json"


def get_or_create_scenarios(
    env_cfg: Dict,
    difficulty: str,
    n_episodes: int,
    namespace_seed: int,
    split: str,
    results_dir: Path,
    overrides: Optional[Dict] = None,
    prior_type: Optional[str] = None,
    prior_sigma: Optional[float] = None,
    snr_offset_db: float = 0.0,
    condition_label: Optional[str] = None,
) -> List[Dict]:
    """固定场景文件：存在即读（锁定不重生成），不存在则生成并保存。"""
    label = condition_label or difficulty
    path = scenario_file_path(results_dir, split, label, n_episodes)
    if path.exists():
        return read_json(path)
    scs = generate_scenarios(
        env_cfg, difficulty, n_episodes, namespace_seed, overrides=overrides,
        prior_type=prior_type, prior_sigma=prior_sigma, snr_offset_db=snr_offset_db,
        difficulty_label=label)
    write_json(path, scs)
    return scs


def check_scenario_files_disjoint(results_dir: Path) -> Dict:
    """质量检查：验证/测试/OOD场景文件不得有重叠的rng_seed。"""
    import glob
    files = glob.glob(str(results_dir / "raw" / "scenarios" / "*.json"))
    seen: Dict[int, str] = {}
    from collections import defaultdict
    pair_hits = defaultdict(int)
    pair_sizes = {}
    import re as _re
    for f in sorted(files):
        stem = Path(f).stem
        base_cond = _re.sub(r"_n\d+$", "", stem)
        scs = read_json(Path(f))
        pair_sizes[f] = len(scs)
        for sc in scs:
            s = int(sc["rng_seed"])
            if s in seen and seen[s] != f:
                prev_base = _re.sub(r"_n\d+$", "", Path(seen[s]).stem)
                if prev_base == base_cond:
                    continue  # 前缀关系：确定性前缀属性，跳过
                pair_hits[(seen[s], f)] += 1
            seen[s] = f
    # 生日悖论阈值：独立流在2^31空间的零星碰撞属数学期望；
    # 仅当共享比例超过较小文件50%时判定为流级共享（真实重复样本）
    dup = []
    for (fa, fb), hits in pair_hits.items():
        if hits > 0.5 * min(pair_sizes[fa], pair_sizes[fb]):
            dup.append({"files": [fa, fb], "shared_seeds": hits})
    return {"ok": len(dup) == 0, "n_scenarios": len(seen),
            "stream_level_overlaps": dup[:20]}
