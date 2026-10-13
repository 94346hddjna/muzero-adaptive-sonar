# 需求追溯矩阵：开题报告要求 → 代码模块 → 实验 → 输出结果

依据：《基于MuZero的自适应水下目标声呐探测策略研究》开题报告（谭锐，2025）。
本矩阵将开题报告的每一条技术要求映射到具体代码模块、实验脚本与结果文件，保证可追溯、可审计。

## 1. 公式级映射

| 开题报告条目 | 公式/定义 | 代码模块 | 验证方式 | 输出 |
|---|---|---|---|---|
| POMDP七元组 M=⟨S,A,T,O,Z,R,γ⟩ | 式(3) | env/level1_env.py | test_environment.py | results/raw/*.jsonl |
| 信念 b_t(i)=P(x=i\|H_t), Σb=1 | 式(4) | env/belief.py | test_belief_update.py | belief校验日志 |
| 贝叶斯更新 | 式(5) | env/belief.py::bayes_update | test_belief_update.py | — |
| LFM信号 s(t)=A rect(t/Tp)cos(2π(f0t+kt²/2)) | 式(6) | env/level2_env.py::lfm_pulse | test_detection_model.py | Level2一致性报告 |
| 接收信号 y(t)=αs(t−τ)+c(t)+n(t) | 式(7) | env/level2_env.py | test_detection_model.py | — |
| 双程时延测距 R=cτ/2 | 式(8) | env/level2_env.py | test_detection_model.py | — |
| 传播损失 TL=20log10(R)+αf·R | 式(9) | env/acoustic_model.py::transmission_loss | test_detection_model.py | — |
| 波束方向图 G(Δθ)=exp(−Δθ²/2σb²) | 式(10) | env/acoustic_model.py::beam_gain | test_detection_model.py | — |
| 主动声呐方程 SNR | 式(11) | env/acoustic_model.py::compute_snr_db | test_detection_model.py | PD分布记录 |
| 虚警概率 PFA=exp(−VT²/2σn²) | 式(12) | env/acoustic_model.py | test_detection_model.py | — |
| Swerling I 检测概率 PD=PFA^(1/(1+SNR)) | 式(13) | env/acoustic_model.py::pd_swerling1 | test_detection_model.py | PD分布CSV |
| 观测概率模型 P(o\|x,a) | 式(14) | env/level1_env.py::_observe | test_environment.py | — |
| 观测向量 o_t=(b_t,m_t,a_{t−1},d_{t−1},t/Tmax) | 式(15) | env/level1_env.py::_build_obs | test_environment.py | — |
| 奖励 r_t=R_hit·1{o=1}−λs−λf·1{o=1,x≠a}−λe·C(a) | 式(16) | env/level1_env.py::_reward | test_reward.py | — |
| 表示网络+GRU递归编码 z_t=GRU(z_{t−1},o_t) | 式(17) | agents/networks.py::RepresentationNet | test_muzero_networks.py | — |
| 动力学网络 (r,h')=g_θ(h,a) | 式(18) | agents/networks.py::DynamicsNet | test_muzero_networks.py | — |
| 预测网络 (p,v)=f_θ(h) | 式(19) | agents/networks.py::PredictionNet | test_muzero_networks.py | — |
| n步自举目标 z_t | 式(20) | agents/muzero.py::_compute_targets | test_replay_buffer.py | — |
| 等效即时奖励 u_t=z_t−γz_{t+1} | 式(21) | agents/muzero.py | test_replay_buffer.py | — |
| MuZero总损失 L(θ)=Σ_k(l_r+l_v+l_p)+c‖θ‖² | 式(22)(23) | agents/muzero.py::update | test_muzero_networks.py | training_log CSV |
| PUCT选择 | 式(24) | agents/mcts.py | test_mcts.py | — |
| n步折扣回传 G^k, Q=W/N | 式(25) | agents/mcts.py::_backup | test_mcts.py | — |
| 温度规整策略 π(a\|s) | 式(26) | agents/mcts.py::root_policy | test_mcts.py | — |
| 信念贪心 a_t=argmax b_t(i) | 式(27) | baselines/belief_greedy.py | sanity check | — |
| DQN损失 | 式(28) | agents/dqn.py | — | training_log |
| PPO比率/CLIP | 式(29)(30) | agents/ppo.py | — | training_log |
| SR | 式(31) | utils/metrics.py | test_metrics.py | main_results.csv |
| ADS(仅成功回合) | 式(32) | utils/metrics.py | test_metrics.py | main_results.csv |
| 删失期望探测代价 EDC | 式(33) | utils/metrics.py | test_metrics.py | main_results.csv |
| 虚警率 FAR=N_FA/N_neg | 式(34) | utils/metrics.py | test_metrics.py | main_results.csv |
| 相对性能退化 Drop(d) | 式(35) | utils/metrics.py | test_metrics.py | ood/robustness结果 |
| 先验偏差 KL散度 | 式(36) | utils/metrics.py::kl_divergence | test_metrics.py | prior_bias结果 |
| 计算复杂度 C_MuZero≈C_h+M·D·(C_g+C_f) | 式(37) | experiments/evaluate.py计时 | — | runtime_results.csv |

## 2. 表格级映射

| 开题报告表格 | 内容 | 实现 | 输出 |
|---|---|---|---|
| 表2 | 状态/观测/动作/奖励变量定义 | env/level1_env.py + agents | — |
| 表3 | 三网络结构(输入2N+3, GRU256, latent128, N+1) | agents/networks.py | manifest.json |
| 表4 | 难度Easy/Medium/Hard(N,PD,PFA,先验) | configs/environment.yaml + 校准 | sanity_check.json |
| 表5 | 五基线特性 | baselines/ + agents/dqn,ppo | main_results.csv |
| 表6 | 七组消融A-G | experiments/ablation.py | ablation_results.csv |
| 表7 | 超参数初始值 | configs/*.yaml | manifest.json |
| 表1 | 文献方法对比 | (综述内容,不产生实验) | — |

## 3. 实验级映射

| 实验 | 脚本 | 原始数据 | 汇总 |
|---|---|---|---|
| Sanity Check | experiments/sanity.py | results/raw/sanity_checks.json | experiment_status.md |
| 基线测试 | experiments/evaluate.py | results/raw/main/<algo>_<diff>_<seed>.jsonl | main_results.csv |
| DQN/PPO/MuZero训练 | experiments/train_rl.py | results/raw/training_logs/*.csv, checkpoints/ | training curves |
| 验证集选模 | experiments/train_rl.py(定期验证) | results/raw/validation/*.csv | best_val checkpoint |
| 主实验(E/M/H) | experiments/evaluate.py | results/raw/main/ | main_results.csv |
| 消融A-G | experiments/ablation.py | results/raw/ablation/ | ablation_results.csv |
| 难度sweep | experiments/robustness.py | results/raw/robustness/ | robustness_results.csv |
| OOD 5类 | experiments/robustness.py | results/raw/ood/ | ood_results.csv |
| 先验偏差4类 | experiments/prior_bias.py | results/raw/prior_bias/ | prior_bias_results.csv |
| Level2验证 | experiments/level2_validation.py | results/raw/level2/ | level2_results.csv |
| 统计检验 | experiments/statistics.py | → | statistics/statistical_tests.csv, effect_sizes.csv |
| 论文图表22张 | experiments/figures.py | 来自raw | figures/(300dpi PNG+PDF) |
| 论文表格7张 | experiments/tables.py | 来自raw | tables/(CSV/XLSX/MD/LaTeX) |
| 最终报告 | experiments/report.py | 来自raw | reports/final_experiment_report.md |
| 质量检查 | experiments/quality_check.py | 扫raw | 质量检查报告 |

## 4. 种子体系（绝不重叠）

| 用途 | 种子命名空间 | 示例 |
|---|---|---|
| 训练种子(5个,网络初始化+探索) | train_seed ∈ {101,102,103,104,105} | 每个RL算法独立训练 |
| 训练环境采样 | TR-<train_seed> 派生流 | 与场景种子隔离 |
| 验证场景 | VS-20250901..20250920 (固定) | checkpoint/超参选择 |
| 测试场景 | TS-30301..(固定,锁定) | 仅最终测试 |
| OOD场景 | OS-40401..(固定) | 分布外测试 |

test_seed_isolation.py 验证四个命名空间两两不相交且固定测试场景文件存在。
