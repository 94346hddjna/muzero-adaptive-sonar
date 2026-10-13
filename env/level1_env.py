# -*- coding: utf-8 -*-
"""Level 1：检测概率抽象环境（开题报告第6章，训练主环境）。

严格实现式(11)-(14)统计映射链路（Gymnasium风格、向量化、随机种子受控）：
  动作a_t(波束指向) → Δθ → 方向性增益式(10) → 传播损失式(9)
  → 回波信噪比式(11) → P_D式(13)/P_FA式(12) → 伯努利观测式(14)
  → 贝叶斯信念更新式(5) → 奖励式(16) → 下一决策步

数学一致性关键（design_notes B6）：每回合在场景生成时计算PD_ep（由R、目标扇区内
偏移Δθ、波束σb、混响经式(9)-(13)决定，回合内恒定）。观测采样与信念滤波使用
同一似然模型：
  P(o=1|x=a)=PD_ep（波束指向目标扇区）   P(o=1|x≠a)=P_FA
使采样分布≡滤波模型→信念为精确后验（"声呐环境、信念更新与决策学习数学上严格自洽"）。

设计要点：
- B4：波束指向非目标扇区时严格按式(14)只以P_FA产生观测
- 规范网格C=192：N∈{8,16,24,32,48,64}均整除→单一固定维度网络处理全部扇区数
- 隐藏状态(target)/观测(o_t)/信念(b_t)/智能体输入(式15向量)严格分离
"""
from typing import Dict, List, Optional

import numpy as np

from env import acoustic_model as am
from env import belief as bl

_N_MAX_DEFAULT = 64


# ---------------------------------------------------------------------------
# 规范网格映射（N扇区 ↔ C规范格）
# ---------------------------------------------------------------------------
def cell_to_sector_map(n_sectors: int, canonical_cells: int) -> np.ndarray:
    """格j属于扇区 j*N//C（C整除N时每扇区恰C/N格）。"""
    return (np.arange(canonical_cells) * n_sectors) // canonical_cells


def sector_center_cell(action: int, n_sectors: int, canonical_cells: int) -> int:
    """扇区中心规范格索引（动力学网络动作嵌入使用）。"""
    per = canonical_cells // n_sectors
    return action * per + per // 2


def resample_belief_to_cells(belief_n: np.ndarray, canonical_cells: int) -> np.ndarray:
    """N维信念→C格（质量守恒：每格=扇区概率×N/C）。"""
    n = len(belief_n)
    cell_sector = cell_to_sector_map(n, canonical_cells)
    return np.asarray(belief_n, dtype=np.float64)[cell_sector] * (n / canonical_cells)


def resample_counts_to_cells(m_n: np.ndarray, canonical_cells: int, t_max: int) -> np.ndarray:
    """N维访问计数→C格（按扇区重复，/T_max归一化）。"""
    n = len(m_n)
    cell_sector = cell_to_sector_map(n, canonical_cells)
    return np.asarray(m_n, dtype=np.float64)[cell_sector] / float(t_max)


def aggregate_cells_to_sectors(cell_probs: np.ndarray, n_sectors: int) -> np.ndarray:
    """C格策略→N扇区策略（边缘化：扇区概率=其格概率之和）。"""
    cell_sector = cell_to_sector_map(n_sectors, len(cell_probs))
    return np.bincount(cell_sector, weights=np.asarray(cell_probs, dtype=np.float64),
                       minlength=n_sectors)


def build_observation_vector(
    belief_n: np.ndarray,
    m_n: np.ndarray,
    n_sectors: int,
    prev_action: int,
    prev_detection: int,
    t: int,
    t_max: int,
    canonical_cells: int,
) -> np.ndarray:
    """式(15)：o_t=(b_t, m_t, a_{t−1}, d_{t−1}, t/T_max)（规范网格化+归一化，B13决策）。"""
    C = canonical_cells
    b_c = resample_belief_to_cells(belief_n, C)
    m_c = resample_counts_to_cells(m_n, C, t_max)
    if t == 0 or prev_action is None or prev_action < 0:
        a_norm = 0.0
    else:
        a_norm = sector_center_cell(prev_action, n_sectors, C) / float(C)
    d = float(prev_detection)
    frac = min(t / float(t_max), 1.0)
    return np.concatenate([b_c, m_c, [a_norm, d, frac]]).astype(np.float32)


OBS_EXTRA_DIM = 3  # a_prev, d_prev, t/T_max
RAW_EXTRA_DIM = 2  # d_prev, t/T_max（raw模式：a_prev为C维one-hot）


def obs_dim(canonical_cells: int, obs_mode: str = "belief", t_max: int = 20) -> int:
    if obs_mode == "belief":
        return 2 * canonical_cells + OBS_EXTRA_DIM
    if obs_mode == "raw_stack":
        return t_max * (canonical_cells + 1) + 1
    return canonical_cells + RAW_EXTRA_DIM


def build_raw_stack_observation_vector(
    acts_hist: list,
    dets_hist: list,
    t: int,
    t_max: int,
    canonical_cells: int,
) -> np.ndarray:
    """v3盲搜索模式：完整历史平铺（每步=动作one-hot(C)+检出(1)），pad到T_max。

    与MuZero的raw模式信息等价（同样的(a,d)历史），供无记忆网络（DQN/PPO）
    以平铺形式获取全部信息——公平性：信息相同、架构不同（MLP平铺 vs GRU+模型）。
    """
    C = canonical_cells
    dim = t_max * (C + 1) + 1
    v = np.zeros(dim, dtype=np.float32)
    for i, (a, d) in enumerate(zip(acts_hist, dets_hist)):
        v[i * (C + 1) + (a % C)] = 1.0
        v[i * (C + 1) + C] = float(d)
    v[t_max * (C + 1)] = min(t / float(t_max), 1.0)
    return v


def build_raw_observation_vector(
    n_sectors: int,
    prev_action: int,
    prev_detection: int,
    t: int,
    t_max: int,
    canonical_cells: int,
) -> np.ndarray:
    """消融D（w/o belief）观测：仅最近原始观测(a_{t−1} one-hot, d_{t−1}, t/T_max)。"""
    C = canonical_cells
    v = np.zeros(C + RAW_EXTRA_DIM, dtype=np.float32)
    if t > 0 and prev_action is not None and prev_action >= 0:
        v[sector_center_cell(prev_action, n_sectors, C)] = 1.0
    v[C] = float(prev_detection)
    v[C + 1] = min(t / float(t_max), 1.0)
    return v


_NOMINAL_PD_CACHE: Dict = {}


def _nominal_pd(params: Dict, env_cfg: Optional[Dict] = None) -> float:
    """名义检测概率=该配置下的平均检测概率（智能体已知系统平均性能，
    但不知单回合真实PD——其随距离R与波束偏移浮动±0.15左右）。
    v2（nominal模式）信念滤波使用该值，建模"操作员不知确切单次检测条件"。"""
    pfa = float(params["pfa"])
    snr_base = float(params["snr_base_db"])
    sigma_b = float(params["sigma_b_deg"])
    rnr = float(params.get("reverb_rnr_db", 0.0))
    key = (round(pfa, 5), round(snr_base, 4), round(sigma_b, 4), round(rnr, 4))
    if key not in _NOMINAL_PD_CACHE:
        d = am.pd_distribution_at_target_sector(
            snr_base, pfa, 16, sigma_b, 2000.0, 4000.0, 0.001, 3000.0,
            am.reverberation_penalty(rnr), n_samples=8000, seed=7)
        _NOMINAL_PD_CACHE[key] = float(d["mean"])
    return _NOMINAL_PD_CACHE[key]


def episode_pd(scn: Dict, env_cfg: Dict) -> float:
    """回合检测概率PD_ep：式(11)-(13)（R、目标扇区内偏移、波束σb、混响）。"""
    snr = am.snr_db_from_equation(
        scn["snr_base_db"], scn["R"], scn["theta_off"], scn["sigma_b_deg"],
        env_cfg["absorption_coeff"], env_cfg["reference_range"],
        am.reverberation_penalty(scn.get("reverb_rnr_db", 0.0)))
    return float(am.pd_swerling1(scn["pfa"], snr))


# ---------------------------------------------------------------------------
# 难度参数与校准缓存
# ---------------------------------------------------------------------------
_CALIB_CACHE: Dict[str, Dict] = {}


def difficulty_params(env_cfg: Dict, diff_name: str, overrides: Optional[Dict] = None) -> Dict:
    """构建难度参数（含snr_base物理校准，B2决策）。overrides可覆盖N/PFA/先验等（OOD）。"""
    d = dict(env_cfg["difficulties"][diff_name])
    if overrides:
        d.update({k: v for k, v in overrides.items() if k in d or k == "snr_offset_db"})
    n = int(d["n_sectors"])
    pfa = float(d["pfa"])
    sigma_b = am.beam_sigma_b(n, env_cfg.get("beam_hpbw_to_sector_width", 1.0))
    rnr = float(d.get("reverb_rnr_db", 0.0))
    snr_offset = float(d.pop("snr_offset_db", 0.0)) if "snr_offset_db" in d else 0.0
    # 波束几何N不变性：Δθ/σb分布与N无关→任意N的校准结果一致（按N缓存避免重复二分）
    cal_key = f"{diff_name}|pfa={pfa}|rnr={rnr}|sig={sigma_b:.4f}|tgt={d['pd_target']}"
    if cal_key not in _CALIB_CACHE:
        base, rec = am.calibrate_snr_base(
            pd_target=float(d["pd_target"]), pfa=pfa, n_sectors=n, sigma_b_deg=sigma_b,
            range_min=env_cfg["range_min"], range_max=env_cfg["range_max"],
            alpha_f=env_cfg["absorption_coeff"], R_ref=env_cfg["reference_range"],
            reverberation_penalty_db=am.reverberation_penalty(rnr),
        )
        dist = am.pd_distribution_at_target_sector(
            base, pfa, n, sigma_b, env_cfg["range_min"], env_cfg["range_max"],
            env_cfg["absorption_coeff"], env_cfg["reference_range"],
            am.reverberation_penalty(rnr))
        _CALIB_CACHE[cal_key] = {"snr_base_db": base, "calibration": rec, "pd_distribution": dist}
    return {
        "difficulty": diff_name,
        "n_sectors": n,
        "pfa": pfa,
        "reverb_rnr_db": rnr,
        "snr_base_db": _CALIB_CACHE[cal_key]["snr_base_db"] + snr_offset,
        "sigma_b_deg": sigma_b,
        "prior_type": d.get("prior_type", "uniform"),
        "prior_sigma": float(d.get("prior_sigma_sectors", 1.0)),
        "calibration": _CALIB_CACHE[cal_key]["calibration"],
        "pd_distribution": _CALIB_CACHE[cal_key]["pd_distribution"],
    }


# ---------------------------------------------------------------------------
# Level 1 向量化环境
# ---------------------------------------------------------------------------
class SonarLevel1VecEnv:
    """B个并行episode的检测概率抽象环境。

    - 训练模式：done后按域随机化分布重采样场景（B7决策）
    - 评测模式：传入固定场景列表（配对评测：同一场景各算法面对同一噪声流）
    - 每slot独立episode RNG；episode结束自动装入下一个场景
    - reward_mode: 'sparse'（式(16)）或 'shaped'（式(16)+信念熵PBRS，消融G）
    """

    def __init__(
        self,
        num_envs: int,
        env_cfg: Dict,
        t_max: int = 20,
        canonical_cells: int = 192,
        domain_rand_cfg: Optional[Dict] = None,
        master_seed: int = 7,
        reward_mode: str = "sparse",
        shaping_coef: float = 1.0,
        fixed_difficulty: Optional[str] = None,
        difficulty_overrides: Optional[Dict] = None,
        record_history: bool = False,
        obs_mode: str = "belief",   # 'belief'（式15默认）或 'raw'（消融D）
        belief_pd_mode: str = "exact",  # 'exact'（滤波用真实PD）或'nominal'（v2：仅知名义PD）
    ):
        self.belief_pd_mode = belief_pd_mode
        self.B = num_envs
        self.env_cfg = env_cfg
        self.t_max = int(t_max)
        self.C = int(canonical_cells)
        self.obs_mode = obs_mode
        self.domain_rand = domain_rand_cfg
        self.master_rng = np.random.default_rng(master_seed)
        self.reward_mode = reward_mode
        self.shaping_coef = shaping_coef
        self.fixed_difficulty = fixed_difficulty
        self.difficulty_overrides = difficulty_overrides or {}
        self.record_history = record_history
        self.obs_d = obs_dim(self.C, obs_mode)
        self.n_max = env_cfg.get("max_sectors", _N_MAX_DEFAULT)
        # 难度参数模板（含校准）。difficulty_overrides固定应用于全部难度（OOD/消融）
        if self.difficulty_overrides:
            base_name = self.difficulty_overrides.pop("difficulty", None)
            if base_name:
                self.fixed_difficulty = base_name
            self.diff_templates = {
                name: difficulty_params(env_cfg, name, self.difficulty_overrides)
                for name in [self.fixed_difficulty or "medium"]
            }
        else:
            self.diff_templates = {
                name: difficulty_params(env_cfg, name) for name in env_cfg["difficulties"]
            }
        self._alloc()
        self.scenario_queue: List[Dict] = []
        self._queue_ptr = 0

    def calibration_records(self) -> Dict:
        return {k: v for k, v in _CALIB_CACHE.items()}

    # -- 内部分配 ------------------------------------------------------------
    def _alloc(self):
        B, Nm = self.B, self.n_max
        self.n = np.zeros(B, dtype=np.int64)
        self.target = np.zeros(B, dtype=np.int64)
        self.R = np.zeros(B)
        self.theta_off = np.zeros(B)
        self.pfa = np.zeros(B)
        self.pd_ep = np.zeros(B)
        self.pd_filter = np.zeros(B)  # 信念滤波似然用PD（exact=真实PD；nominal=名义PD）
        self.snr_base = np.zeros(B)
        self.sigma_b = np.zeros(B)
        self.rnr = np.zeros(B)
        self.difficulty = [""] * B
        self.scenario_index = np.full(B, -1, dtype=np.int64)
        self.belief = np.zeros((B, Nm))
        self.m = np.zeros((B, Nm))
        self.prev_action = np.full(B, -1, dtype=np.int64)
        self.prev_det = np.zeros(B, dtype=np.int64)
        self.t = np.zeros(B, dtype=np.int64)
        self.rngs: List[Optional[np.random.Generator]] = [None] * B
        self.ep_return = np.zeros(B)
        self.energy = np.zeros(B)
        self.n_neg = np.zeros(B, dtype=np.int64)
        self.n_fa = np.zeros(B, dtype=np.int64)
        self.prev_phi = np.zeros(B)
        self.hist: List[Optional[List[Dict]]] = [None] * B
        self.active = np.zeros(B, dtype=bool)
        self.acts_hist: List[list] = [[] for _ in range(B)]  # v3 raw_stack有序历史
        self.dets_hist: List[list] = [[] for _ in range(B)]

    # -- 场景 ----------------------------------------------------------------
    def _apply_scenario(self, slot: int, sc: Dict):
        N = int(sc["n_sectors"])
        self.n[slot] = N
        self.target[slot] = int(sc["target"])
        self.R[slot] = float(sc["R"])
        self.theta_off[slot] = float(sc["theta_off"])
        self.pfa[slot] = float(sc["pfa"])
        self.snr_base[slot] = float(sc["snr_base_db"])
        self.sigma_b[slot] = float(sc["sigma_b_deg"])
        self.rnr[slot] = float(sc.get("reverb_rnr_db", 0.0))
        self.pd_ep[slot] = float(sc.get("pd", episode_pd(sc, self.env_cfg)))
        # 信念滤波用检测概率（v2设计：nominal模式下仅知名义PD，检测仍用真实PD_ep）
        self.pd_filter[slot] = self.pd_ep[slot] if self.belief_pd_mode == "exact" \
            else float(sc.get("nominal_pd", self.pd_ep[slot]))
        self.difficulty[slot] = str(sc.get("difficulty", "?"))
        self.scenario_index[slot] = int(sc.get("scenario_index", -1))
        b0 = np.asarray(sc["b0"], dtype=np.float64)
        bpad = np.zeros(self.n_max)
        bpad[:N] = b0
        self.belief[slot] = bpad
        self.m[slot] = 0.0
        self.prev_action[slot] = -1
        self.prev_det[slot] = 0
        self.t[slot] = 0
        self.rngs[slot] = np.random.default_rng(int(sc["rng_seed"]))
        self.ep_return[slot] = 0.0
        self.energy[slot] = 0.0
        self.n_neg[slot] = 0
        self.n_fa[slot] = 0
        H0 = bl.belief_entropy(b0)
        self.prev_phi[slot] = -self.shaping_coef * H0 / np.log(N)
        self.hist[slot] = [] if self.record_history else None
        self.acts_hist[slot] = []
        self.dets_hist[slot] = []
        self.active[slot] = True

    def _sample_training_scenario(self, slot: int) -> Dict:
        """域随机化场景采样（B7决策：标称Medium为主+难度混合+参数抖动）。"""
        rng = self.master_rng
        if self.fixed_difficulty is not None:
            diff = self.fixed_difficulty
        elif self.domain_rand and self.domain_rand.get("enabled", True):
            names = list(self.domain_rand["difficulty_probs"].keys())
            probs = list(self.domain_rand["difficulty_probs"].values())
            diff = str(rng.choice(names, p=probs))
        else:
            diff = "medium"
        tpl = dict(self.diff_templates[diff])
        if self.domain_rand and self.domain_rand.get("enabled", True) and self.fixed_difficulty is None:
            jit = self.domain_rand.get("snr_jitter_db", 0.0)
            tpl["snr_base_db"] = float(tpl["snr_base_db"] + rng.uniform(-jit, jit))
            f = self.domain_rand.get("pfa_jitter_factor", 1.0)
            tpl["pfa"] = float(np.clip(tpl["pfa"] * rng.uniform(1.0 / f, f), 1e-4, 0.5))
            prior_types = self.domain_rand.get("prior_types", ["uniform"])
            tpl["prior_type"] = str(rng.choice(prior_types))
            sj = self.domain_rand.get("prior_sigma_jitter", 0.0)
            tpl["prior_sigma"] = max(0.3, float(tpl["prior_sigma"]) + rng.uniform(-sj, sj))
        N = tpl["n_sectors"]
        target = int(rng.integers(N))
        R = float(rng.uniform(self.env_cfg["range_min"], self.env_cfg["range_max"]))
        sector_width = 360.0 / N
        theta_off = float(rng.uniform(-sector_width / 2, sector_width / 2))
        b0 = bl.make_initial_belief(N, tpl["prior_type"], target, rng=rng,
                                    sigma_sectors=tpl["prior_sigma"])
        sc = {
            "n_sectors": N, "target": target, "R": R, "theta_off": theta_off,
            "pfa": tpl["pfa"], "snr_base_db": tpl["snr_base_db"],
            "sigma_b_deg": tpl["sigma_b_deg"], "reverb_rnr_db": tpl["reverb_rnr_db"],
            "b0": b0.tolist(), "rng_seed": int(rng.integers(0, 2**31 - 1)),
            "difficulty": diff,
            "nominal_pd": _nominal_pd(tpl, self.env_cfg),
        }
        self._apply_scenario(slot, sc)
        return sc

    def reset(self, scenarios: Optional[List[Dict]] = None) -> np.ndarray:
        """重置。scenarios非空=评测模式（固定场景配对）；否则训练模式。"""
        self.scenario_queue = list(scenarios) if scenarios else []
        self._queue_ptr = 0
        for slot in range(self.B):
            if scenarios:
                if self._queue_ptr < len(self.scenario_queue):
                    self._apply_scenario(slot, self.scenario_queue[self._queue_ptr])
                    self._queue_ptr += 1
                else:
                    self.active[slot] = False
            else:
                self._sample_training_scenario(slot)
        return self._obs_all()

    def load_next_scenario(self, slot: int):
        """done后装入下一个场景（评测）或重采样（训练）。"""
        if self.scenario_queue:
            if self._queue_ptr < len(self.scenario_queue):
                self._apply_scenario(slot, self.scenario_queue[self._queue_ptr])
                self._queue_ptr += 1
            else:
                self.active[slot] = False
        else:
            self._sample_training_scenario(slot)

    def queue_exhausted(self) -> bool:
        return bool(self.scenario_queue) and self._queue_ptr >= len(self.scenario_queue)

    # -- 观测 ----------------------------------------------------------------
    def _obs_slot(self, slot: int) -> np.ndarray:
        N = int(self.n[slot])
        if self.obs_mode == "raw_stack":
            return build_raw_stack_observation_vector(
                self.acts_hist[slot], self.dets_hist[slot],
                int(self.t[slot]), self.t_max, self.C)
        if self.obs_mode == "raw":
            return build_raw_observation_vector(
                N, int(self.prev_action[slot]), int(self.prev_det[slot]),
                int(self.t[slot]), self.t_max, self.C)
        return build_observation_vector(
            self.belief[slot, :N], self.m[slot, :N], N,
            int(self.prev_action[slot]), int(self.prev_det[slot]),
            int(self.t[slot]), self.t_max, self.C)

    def _obs_all(self) -> np.ndarray:
        out = np.zeros((self.B, self.obs_d), dtype=np.float32)
        for slot in range(self.B):
            if self.active[slot]:
                out[slot] = self._obs_slot(slot)
        return out

    # -- 检测实现（Level 2子类覆盖为波形级检测） ----------------------------
    def _detect(self, slot: int, a: int, rng: np.random.Generator) -> int:
        """式(14)伯努利观测采样：a==x用PD_ep；否则P_FA。"""
        x = int(self.target[slot])
        u = rng.random()
        if a == x:
            return 1 if u < float(self.pd_ep[slot]) else 0
        return 1 if u < float(self.pfa[slot]) else 0

    # -- 核心步进 ------------------------------------------------------------
    def step(self, actions: np.ndarray):
        """执行动作（波束指向）→式(14)伯努利观测→式(5)信念更新→式(16)奖励。"""
        rewards = np.zeros(self.B, dtype=np.float64)
        dones = np.zeros(self.B, dtype=bool)
        episode_infos: List[Optional[Dict]] = [None] * self.B
        rc = self.env_cfg["reward"]
        R_hit, lam_s, lam_f, lam_e = rc["R_hit"], rc["lambda_s"], rc["lambda_f"], rc["lambda_e"]
        cost_a = rc["action_cost"]
        for slot in range(self.B):
            if not self.active[slot]:
                dones[slot] = True  # 非活跃槽位占位（评测器按episode_info==None忽略）
                continue
            a = int(actions[slot])
            N = int(self.n[slot])
            assert 0 <= a < N, f"action {a} out of range [0,{N})"
            rng = self.rngs[slot]
            x = int(self.target[slot])
            pd_ep = float(self.pd_ep[slot])
            pfa = float(self.pfa[slot])
            # 式(14)：观测采样（Level 1=伯努利统计映射；Level 2=波形级匹配滤波检测）
            det = self._detect(slot, a, rng)
            # 式(16)：奖励（B1决策：按公式字面，λf=R_hit使虚警步净收益=漏检步）
            r = -lam_s - lam_e * cost_a
            if det:
                r += R_hit
                if a != x:
                    r -= lam_f
            # 式(5)：贝叶斯信念更新（exact模式：似然=采样模型→精确后验；
            # nominal模式（v2）：滤波用名义PD，真实PD未知→信念为近似后验，
            # 历史中蕴含检测条件信息→记忆/模型式方法获得真实优势）
            b_next = bl.bayes_update(self.belief[slot, :N], a, det,
                                     float(self.pd_filter[slot]), pfa)
            # 塑形奖励（消融G，PBRS：Φ(b)=−c·H(b)/lnN，B14决策）
            if self.reward_mode == "shaped":
                H_next = bl.belief_entropy(b_next)
                phi_next = -self.shaping_coef * H_next / np.log(N)
                r = r + phi_next - self.prev_phi[slot]
                self.prev_phi[slot] = phi_next
            # 状态推进
            self.belief[slot, :N] = b_next
            self.m[slot, a] += 1.0
            self.prev_action[slot] = a
            self.prev_det[slot] = det
            self.t[slot] += 1
            self.acts_hist[slot].append(int(a))
            self.dets_hist[slot].append(int(det))
            self.ep_return[slot] += r
            self.energy[slot] += cost_a
            if a != x:
                self.n_neg[slot] += 1
                if det:
                    self.n_fa[slot] += 1
            success = bool(det == 1 and a == x)
            done = success or self.t[slot] >= self.t_max
            if self.record_history and self.hist[slot] is not None:
                self.hist[slot].append({
                    "t": int(self.t[slot]), "action": a, "detection": int(det),
                    "pd": pd_ep, "reward": float(r), "belief": b_next.tolist(),
                })
            rewards[slot] = r
            dones[slot] = done
            if done:
                episode_infos[slot] = {
                    "slot": slot, "scenario_index": int(self.scenario_index[slot]),
                    "success": success, "steps": int(self.t[slot]),
                    "target": x, "n_sectors": N, "return": float(self.ep_return[slot]),
                    "energy": float(self.energy[slot]),
                    "n_negative_emissions": int(self.n_neg[slot]),
                    "n_false_alarms": int(self.n_fa[slot]),
                    "final_belief": self.belief[slot, :N].tolist(),
                    "pfa": pfa, "pd_ep": pd_ep,
                    "snr_base_db": float(self.snr_base[slot]),
                    "R": float(self.R[slot]),
                    "difficulty": self.difficulty[slot],
                    "last_action": a, "last_detection": int(det),
                    "history": self.hist[slot],
                }
                self.load_next_scenario(slot)
        obs = self._obs_all()
        return obs, rewards, dones, episode_infos

    # -- 便捷 ----------------------------------------------------------------
    def belief_sums(self) -> np.ndarray:
        """每个活跃slot的信念和（质量检查：应恒≈1）。"""
        s = np.zeros(self.B)
        for slot in range(self.B):
            if self.active[slot]:
                s[slot] = self.belief[slot, :int(self.n[slot])].sum()
        return s


def make_scenario_from_params(
    params: Dict, env_cfg: Dict, rng: np.random.Generator, rng_seed: int,
    scenario_index: int = -1,
    prior_type: Optional[str] = None, prior_sigma: Optional[float] = None,
    snr_offset_db: float = 0.0,
) -> Dict:
    """从难度参数+场景RNG生成完整场景（固定测试/验证/OOD场景生成用，含PD_ep）。"""
    N = int(params["n_sectors"])
    target = int(rng.integers(N))
    R = float(rng.uniform(env_cfg["range_min"], env_cfg["range_max"]))
    sector_width = 360.0 / N
    theta_off = float(rng.uniform(-sector_width / 2, sector_width / 2))
    pt = prior_type or params["prior_type"]
    ps = prior_sigma if prior_sigma is not None else params["prior_sigma"]
    b0 = bl.make_initial_belief(N, pt, target, rng=rng, sigma_sectors=ps)
    sc = {
        "n_sectors": N, "target": target, "R": R, "theta_off": theta_off,
        "pfa": params["pfa"], "snr_base_db": params["snr_base_db"] + snr_offset_db,
        "sigma_b_deg": params["sigma_b_deg"],
        "reverb_rnr_db": params["reverb_rnr_db"], "b0": b0.tolist(),
        "rng_seed": int(rng_seed), "difficulty": params["difficulty"],
        "scenario_index": scenario_index, "prior_type": pt, "prior_sigma": ps,
    }
    sc["pd"] = episode_pd(sc, env_cfg)
    sc["nominal_pd"] = _nominal_pd(params, env_cfg)
    return sc
