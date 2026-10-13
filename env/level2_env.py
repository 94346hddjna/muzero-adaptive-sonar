# -*- coding: utf-8 -*-
"""Level 2：信号级声呐验证环境（开题报告第6章、式(6)-(8)）。

在Level 1之上叠加信号级链路（B15决策，复基带实现）：
  式(6) LFM脉冲 s(t)（复基带上扫频，单位能量副本）
  式(7) 接收 y(t)=α·s(t−τ)+c(t)+n(t)（α为Swerling I瑞利起伏×传播×波束增益，
        c(t)混响~复高斯，n(t)噪声~复高斯）
  式(8) 双程时延 τ=2R/c
  匹配滤波（发射副本相关）→ 包络 |·| → 门限V_T（式(12)按MF噪声统计标定）
  检测输出 o_t

统计一致性（可验证）：
  MF输出（单位能量副本）在真实时延处 = z + n_c，z~CN(0,E[a²])，n_c~CN(0,N0_eff)；
  包络门限V_T=√(−N0_eff·ln PFA) ⇒ PFA=exp(−V_T²/N0_eff)（式(12)），
  PD=exp(−V_T²/(E[a²]+N0_eff))=PFA^(1/(1+SNR))（式(13)，SNR=E[a²]/N0_eff）。
  E[a²]由式(11)含混响罚项的SNR链路确定 ⇒ Level 2实测PD/PFA与Level 1一致
  （一致性检验容差见configs/environment.yaml level2.consistency_tolerance）。
"""
from typing import Dict, List, Optional

import numpy as np

from env import acoustic_model as am
from env.level1_env import SonarLevel1VecEnv


class SonarLevel2VecEnv(SonarLevel1VecEnv):
    """信号级环境：继承Level 1全部回合逻辑/信念/奖励，仅检测改为波形级实现。"""

    def __init__(self, *args, level2_cfg: Optional[Dict] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.l2 = level2_cfg or {}
        self.fs = float(self.l2.get("fs", 48000.0))
        self.f0 = float(self.l2.get("f0", 12000.0))
        self.B_bw = float(self.l2.get("bandwidth", 8000.0))
        self.Tp = float(self.l2.get("pulse_duration", 0.02))
        self.noise_sigma = float(self.l2.get("noise_sigma", 0.5))
        self.default_rnr_db = float(self.l2.get("reverberation_rnr_db", 3.0))
        self.pad = float(self.l2.get("window_padding", 0.005))
        # 式(6)复基带LFM副本（单位能量）
        n_samp = max(int(round(self.Tp * self.fs)), 16)
        t = (np.arange(n_samp) + 0.5) / self.fs  # 中心化采样
        k = self.B_bw / self.Tp                   # 调频率 k=B/Tp
        self.replica = np.exp(1j * np.pi * k * t ** 2)
        self.replica = self.replica / np.sqrt(np.sum(np.abs(self.replica) ** 2))  # E_s=1
        self.n_pulse = n_samp
        self._buf: List[Optional[Dict]] = [None] * self.B  # 每slot缓存τ/窗长

    # -- 每slot波形缓存 ------------------------------------------------------
    def _waveform_buf(self, slot: int) -> Dict:
        if self._buf[slot] is None:
            R = float(self.R[slot])
            tau = 2.0 * R / self.env_cfg.get("sound_speed", 1500.0)  # 式(8)
            tau_s = int(round(tau * self.fs))
            L = tau_s + self.n_pulse + int(round(self.pad * self.fs)) + 1
            # 门限V_T：式(12)按MF噪声统计（N0_eff含混响）
            rnr_db = float(self.rnr[slot]) if self.rnr[slot] > 0 else 0.0
            N0 = 2.0 * self.noise_sigma ** 2                     # 每样本复噪声功率
            N0_eff = N0 * (1.0 + 10.0 ** (rnr_db / 10.0))        # 并入混响
            pfa = float(self.pfa[slot])
            V_T = np.sqrt(-N0_eff * np.log(pfa))
            # 回波平均功率：式(11)链路（含波束增益与混响罚）→ E[a²]=SNR_lin·N0_eff
            snr_db = am.snr_db_from_equation(
                self.snr_base[slot], self.R[slot], self.theta_off[slot],
                self.sigma_b[slot], self.env_cfg["absorption_coeff"],
                self.env_cfg["reference_range"], am.reverberation_penalty(rnr_db))
            echo_power = (10.0 ** (snr_db / 10.0)) * N0_eff
            self._buf[slot] = {"tau_s": tau_s, "L": L, "N0": N0, "N0_eff": N0_eff,
                               "V_T": float(V_T), "echo_power": float(echo_power),
                               "rnr_db": rnr_db}
        return self._buf[slot]

    def _apply_scenario(self, slot: int, sc: Dict):
        self._buf[slot] = None  # 新场景→重建波形缓存
        super()._apply_scenario(slot, sc)

    # -- 波形级检测（式(6)-(8)+匹配滤波+包络门限） --------------------------
    def _detect(self, slot: int, a: int, rng: np.random.Generator) -> int:
        buf = self._waveform_buf(slot)
        L, tau_s = buf["L"], buf["tau_s"]
        n_c_per = np.sqrt(buf["N0"] / 2.0)          # 每I/Q分量σ
        r_c_per = np.sqrt(buf["N0"] * (10.0 ** (buf["rnr_db"] / 10.0)) / 2.0)
        # 式(7)：y(t)=α·s(t−τ)+c(t)+n(t)
        y = rng.normal(0.0, n_c_per, L) + 1j * rng.normal(0.0, n_c_per, L)
        if r_c_per > 0:
            y = y + rng.normal(0.0, r_c_per, L) + 1j * rng.normal(0.0, r_c_per, L)
        if a == int(self.target[slot]):
            # Swerling I：α瑞利起伏+均匀相位（回合间独立，z~CN(0,E[a²])）
            zr = rng.normal(0.0, np.sqrt(buf["echo_power"] / 2.0))
            zi = rng.normal(0.0, np.sqrt(buf["echo_power"] / 2.0))
            z = zr + 1j * zi
            y[tau_s:tau_s + self.n_pulse] += z * self.replica
        # 匹配滤波：与副本相关，取真实时延处输出
        seg = y[tau_s:tau_s + self.n_pulse]
        mf_out = np.sum(seg * np.conj(self.replica))
        # 包络检测（式(12)门限）
        return 1 if abs(mf_out) > buf["V_T"] else 0
