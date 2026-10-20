# 最终实验报告：基于MuZero的自适应水下目标声呐探测策略研究

> 本报告由程序从results/raw/原始数据自动生成；所有数值来自真实运行的原始日志，可逐条追溯。未运行/失败的实验如实标注。

## 1. Experimental Setup

两级声呐仿真环境（开题报告第6章）：Level 1检测概率抽象环境（式(9)-(14)统计映射）用于训练；Level 2信号级环境（式(6)-(8) LFM/时延/传播+匹配滤波）用于验证一致性。POMDP式(3)；贝叶斯信念更新式(4)(5)；奖励式(16)。

## 2. Hardware

- CPU核数: 24；RAM: 33.97 GB

- GPU: NVIDIA GeForce RTX 5060 Laptop GPU (8.55 GB), CUDA 12.8, PyTorch CUDA可用: True

## 3. Software

- Python 3.12.10, PyTorch 2.11.0+cu128, NumPy 2.4.6, SciPy 1.18.0

## 4. Environment

各难度物理校准（平均PD目标 vs 实测，B2决策）：

| 难度 | N | P_FA | PD目标 | 实测平均PD | PD_std | SNR基准(dB) |
|---|---|---|---|---|---|---|
|easy|8|0.01|0.95|0.9501|0.0507|26.48|
|medium|16|0.05|0.85|0.8504|0.1287|20.69|
|hard|32|0.1|0.7|0.7011|0.1964|17.02|

## 5. Dataset/Scenario Generation

全部数据由声呐仿真环境生成（无外部数据集）。固定场景文件由互不相交的主种子生成并锁定（results/raw/scenarios/）：Train种子[101..105]、Validation 20250901、Test 30303（锁定）、OOD 40404。配对评测：所有算法在完全相同场景（目标位置/环境参数/先验/每回合RNG）下测试。

## 6. Train/Validation/Test Protocol

- 训练：域随机化（Easy25%/Medium50%/Hard25%+SNR±3dB+PFA×[0.8,1.2]+先验类型随机，B7决策）；1e6环境步预算（开题报告公平性）。
- 验证：固定Medium验证场景（200回合/次），仅以Validation SR选择best_val checkpoint（绝不用Test调参）。
- 测试：参数锁定后运行；Test结果不用于任何调参。

## 7. Algorithms

Sequential（顺序扫描）、Random（随机扫描）、Belief-Greedy（式(27)）、DQN（式(28)）、PPO（式(29)(30)）、MuZero+MCTS（式(17)-(26)，主方法）。
所有学习算法：相同观测向量（式15）、相同奖励、相同训练预算、相同测试场景。

## 8. Hyperparameters

见configs/*.yaml（表7初始值：γ=0.99, lr=3e-4 AdamW, batch=128, K=5, n=5, latent=128, GRU=256, M=50, D_p=5, c1=1.25, τ退火1.0→0.25, Dirichlet α=0.3）。每次实验的实际config与manifest存于results/raw/manifests/。

## 9. Main Results

| 方法 | SR(Easy) | SR(Medium) | SR(Hard) | ADS(Medium) | EDC(Medium) | FAR(Medium) |
|---|---|---|---|---|---|---|
|Sequential|0.999|0.899|0.432|8.838|10.975|0.052|
|Random|0.914|0.663|0.364|8.686|15.864|0.051|
|Belief-Greedy|0.999|0.960|0.000|2.143|3.257|0.058|
|POMCP (known model)|0.998|0.911|0.173|5.793|7.958|0.052|
|DQN|0.478|0.335|0.132|6.675|22.221|0.050|
|PPO|0.471|0.303|0.108|11.340|24.326|0.049|
|DQN-GRU (memory)|0.820|0.448|0.099|9.028|20.591|0.050|
|PPO-Stack (memory)|0.448|0.339|0.131|6.640|22.141|0.049|
|MuZero (ours)|0.725|0.679|0.278|8.937|15.677|0.050|

样本效率（达到SR≥0.90所需环境步，未达到则标'未达到'）：

- DQN: 未达到阈值

- PPO: 未达到阈值

- MuZero (ours): 未达到阈值

## 10. Statistical Tests

共284项配对检验（Holm校正，α=0.05），其中257项显著。详见results/statistics/statistical_tests.csv与effect_sizes.csv。

- easy: muzero(full) vs muzero(main|woMCTS): ΔSR=0.082, p=0.0000, cohens_dz=0.223

- easy: muzero(full) vs muzero(wo_domain_rand|woDomRand): ΔSR=0.122, p=0.0000, cohens_dz=0.596

- easy: muzero(full) vs muzero(wo_memory|woMemory): ΔSR=-0.052, p=0.0000, cohens_dz=-0.179

- hard: muzero(full) vs muzero(main|woMCTS): ΔSR=0.136, p=0.0000, cohens_dz=0.621

- hard: muzero(full) vs muzero(wo_domain_rand|woDomRand): ΔSR=0.039, p=0.0008, cohens_dz=0.112

- hard: muzero(full) vs muzero(wo_memory|woMemory): ΔSR=0.053, p=0.0000, cohens_dz=0.181

- medium: muzero(full) vs muzero(main|D1): ΔSR=0.129, p=0.0000, cohens_dz=0.366

- medium: muzero(full) vs muzero(main|M200): ΔSR=0.070, p=0.0000, cohens_dz=0.296

- medium: muzero(full) vs muzero(main|M25): ΔSR=0.452, p=0.0000, cohens_dz=1.079

- medium: muzero(full) vs muzero(main|M50): ΔSR=0.142, p=0.0000, cohens_dz=0.550

- medium: muzero(full) vs muzero(main|woMCTS): ΔSR=0.180, p=0.0000, cohens_dz=0.595

- medium: muzero(full) vs muzero(wo_domain_rand|woDomRand): ΔSR=0.037, p=0.0007, cohens_dz=0.123

- medium: muzero(full) vs muzero(wo_memory|woMemory): ΔSR=0.071, p=0.0000, cohens_dz=0.224

- easy: muzero vs sequential: ΔSR=-0.274, p=0.0000, cohens_dz=-1.109

- easy: muzero vs random: ΔSR=-0.189, p=0.0000, cohens_dz=-0.681

- easy: muzero vs belief_greedy: ΔSR=-0.274, p=0.0000, cohens_dz=-1.095

- easy: muzero vs dqn: ΔSR=0.247, p=0.0000, cohens_dz=0.857

- easy: muzero vs ppo: ΔSR=0.254, p=0.0000, cohens_dz=1.079

- easy: muzero vs dqn_gru: ΔSR=-0.095, p=0.0000, cohens_dz=-0.404

- easy: muzero vs ppo_stack: ΔSR=0.277, p=0.0000, cohens_dz=0.802

- easy: muzero vs pomcp: ΔSR=-0.273, p=0.0000, cohens_dz=-1.087

- hard: muzero vs sequential: ΔSR=-0.154, p=0.0000, cohens_dz=-0.315

- hard: muzero vs random: ΔSR=-0.086, p=0.0000, cohens_dz=-0.301

- hard: muzero vs belief_greedy: ΔSR=0.278, p=0.0000, cohens_dz=1.398

- hard: muzero vs dqn: ΔSR=0.146, p=0.0000, cohens_dz=0.594

- hard: muzero vs ppo: ΔSR=0.170, p=0.0000, cohens_dz=0.592

- hard: muzero vs dqn_gru: ΔSR=0.179, p=0.0000, cohens_dz=0.745

- hard: muzero vs ppo_stack: ΔSR=0.146, p=0.0000, cohens_dz=0.582

- hard: muzero vs pomcp: ΔSR=0.104, p=0.0000, cohens_dz=0.393

- medium: muzero vs sequential: ΔSR=-0.220, p=0.0000, cohens_dz=-0.602

- medium: muzero vs belief_greedy: ΔSR=-0.281, p=0.0000, cohens_dz=-0.882

- medium: muzero vs dqn: ΔSR=0.344, p=0.0000, cohens_dz=1.045

- medium: muzero vs ppo: ΔSR=0.376, p=0.0000, cohens_dz=1.074

- medium: muzero vs dqn_gru: ΔSR=0.230, p=0.0000, cohens_dz=0.587

- medium: muzero vs ppo_stack: ΔSR=0.340, p=0.0000, cohens_dz=0.929

- medium: muzero vs pomcp: ΔSR=-0.232, p=0.0000, cohens_dz=-0.803

- id: muzero vs sequential: ΔSR=-0.207, p=0.0000, cohens_dz=-0.547

- id: muzero vs belief_greedy: ΔSR=-0.295, p=0.0000, cohens_dz=-0.984

- id: muzero vs dqn: ΔSR=0.344, p=0.0000, cohens_dz=1.045

- id: muzero vs ppo: ΔSR=0.342, p=0.0000, cohens_dz=0.923

- id: muzero vs dqn_gru: ΔSR=0.206, p=0.0000, cohens_dz=0.521

- id: muzero vs ppo_stack: ΔSR=0.317, p=0.0000, cohens_dz=0.831

- id: muzero vs pomcp: ΔSR=-0.246, p=0.0000, cohens_dz=-0.868

- ood_n: muzero vs sequential: ΔSR=-0.157, p=0.0000, cohens_dz=-0.350

- ood_n: muzero vs random: ΔSR=-0.107, p=0.0000, cohens_dz=-0.313

- ood_n: muzero vs belief_greedy: ΔSR=-0.787, p=0.0000, cohens_dz=-2.402

- ood_n: muzero vs dqn: ΔSR=0.118, p=0.0000, cohens_dz=0.368

- ood_n: muzero vs ppo: ΔSR=0.141, p=0.0000, cohens_dz=0.478

- ood_n: muzero vs dqn_gru: ΔSR=0.102, p=0.0000, cohens_dz=0.351

- ood_n: muzero vs ppo_stack: ΔSR=0.126, p=0.0000, cohens_dz=0.416

- ood_n: muzero vs pomcp: ΔSR=-0.686, p=0.0000, cohens_dz=-2.007

- ood_pfa: muzero vs sequential: ΔSR=-0.226, p=0.0000, cohens_dz=-0.602

- ood_pfa: muzero vs belief_greedy: ΔSR=-0.300, p=0.0000, cohens_dz=-0.900

- ood_pfa: muzero vs dqn: ΔSR=0.313, p=0.0000, cohens_dz=0.958

- ood_pfa: muzero vs ppo: ΔSR=0.383, p=0.0000, cohens_dz=1.139

- ood_pfa: muzero vs dqn_gru: ΔSR=0.191, p=0.0000, cohens_dz=0.503

- ood_pfa: muzero vs ppo_stack: ΔSR=0.303, p=0.0000, cohens_dz=0.868

- ood_pfa: muzero vs pomcp: ΔSR=-0.234, p=0.0000, cohens_dz=-0.741

- ood_prior: muzero vs sequential: ΔSR=-0.188, p=0.0000, cohens_dz=-0.472

- ood_prior: muzero vs belief_greedy: ΔSR=0.686, p=0.0000, cohens_dz=2.690

- ood_prior: muzero vs dqn: ΔSR=0.346, p=0.0000, cohens_dz=1.039

- ood_prior: muzero vs ppo: ΔSR=0.384, p=0.0000, cohens_dz=1.026

- ood_prior: muzero vs dqn_gru: ΔSR=0.234, p=0.0000, cohens_dz=0.606

- ood_prior: muzero vs ppo_stack: ΔSR=0.344, p=0.0000, cohens_dz=0.928

- ood_prior: muzero vs pomcp: ΔSR=0.371, p=0.0000, cohens_dz=1.167

- ood_reverb: muzero vs sequential: ΔSR=-0.215, p=0.0000, cohens_dz=-0.557

- ood_reverb: muzero vs belief_greedy: ΔSR=-0.287, p=0.0000, cohens_dz=-0.876

- ood_reverb: muzero vs dqn: ΔSR=0.331, p=0.0000, cohens_dz=1.006

- ood_reverb: muzero vs ppo: ΔSR=0.367, p=0.0000, cohens_dz=1.016

- ood_reverb: muzero vs dqn_gru: ΔSR=0.196, p=0.0000, cohens_dz=0.511

- ood_reverb: muzero vs ppo_stack: ΔSR=0.310, p=0.0000, cohens_dz=0.847

- ood_reverb: muzero vs pomcp: ΔSR=-0.239, p=0.0000, cohens_dz=-0.786

- ood_snr: muzero vs sequential: ΔSR=-0.236, p=0.0000, cohens_dz=-0.782

- ood_snr: muzero vs belief_greedy: ΔSR=-0.262, p=0.0000, cohens_dz=-1.011

- ood_snr: muzero vs dqn: ΔSR=0.384, p=0.0000, cohens_dz=1.223

- ood_snr: muzero vs ppo: ΔSR=0.423, p=0.0000, cohens_dz=1.184

- ood_snr: muzero vs dqn_gru: ΔSR=0.259, p=0.0000, cohens_dz=0.669

- ood_snr: muzero vs ppo_stack: ΔSR=0.380, p=0.0000, cohens_dz=1.044

- ood_snr: muzero vs pomcp: ΔSR=-0.242, p=0.0000, cohens_dz=-0.903

- adversarial: muzero vs sequential: ΔSR=-0.195, p=0.0000, cohens_dz=-0.498

- adversarial: muzero vs random: ΔSR=0.029, p=0.0060, cohens_dz=0.087

- adversarial: muzero vs belief_greedy: ΔSR=0.683, p=0.0000, cohens_dz=2.516

- adversarial: muzero vs dqn: ΔSR=0.347, p=0.0000, cohens_dz=1.074

- adversarial: muzero vs ppo: ΔSR=0.388, p=0.0000, cohens_dz=1.119

- adversarial: muzero vs dqn_gru: ΔSR=0.234, p=0.0000, cohens_dz=0.602

- adversarial: muzero vs ppo_stack: ΔSR=0.349, p=0.0000, cohens_dz=0.971

- adversarial: muzero vs pomcp: ΔSR=0.366, p=0.0000, cohens_dz=1.125

- biased: muzero vs sequential: ΔSR=-0.195, p=0.0000, cohens_dz=-0.498

- biased: muzero vs random: ΔSR=0.029, p=0.0060, cohens_dz=0.087

- biased: muzero vs belief_greedy: ΔSR=0.261, p=0.0000, cohens_dz=0.480

- biased: muzero vs dqn: ΔSR=0.347, p=0.0000, cohens_dz=1.074

- biased: muzero vs ppo: ΔSR=0.388, p=0.0000, cohens_dz=1.119

- biased: muzero vs dqn_gru: ΔSR=0.234, p=0.0000, cohens_dz=0.602

- biased: muzero vs ppo_stack: ΔSR=0.349, p=0.0000, cohens_dz=0.971

- biased: muzero vs pomcp: ΔSR=0.136, p=0.0000, cohens_dz=0.361

- correct: muzero vs sequential: ΔSR=-0.195, p=0.0000, cohens_dz=-0.498

- correct: muzero vs random: ΔSR=0.029, p=0.0060, cohens_dz=0.087

- correct: muzero vs belief_greedy: ΔSR=-0.279, p=0.0000, cohens_dz=-0.893

- correct: muzero vs dqn: ΔSR=0.347, p=0.0000, cohens_dz=1.074

- correct: muzero vs ppo: ΔSR=0.388, p=0.0000, cohens_dz=1.119

- correct: muzero vs dqn_gru: ΔSR=0.234, p=0.0000, cohens_dz=0.602

- correct: muzero vs ppo_stack: ΔSR=0.349, p=0.0000, cohens_dz=0.971

- correct: muzero vs pomcp: ΔSR=-0.226, p=0.0000, cohens_dz=-0.763

- uniform: muzero vs sequential: ΔSR=-0.195, p=0.0000, cohens_dz=-0.498

- uniform: muzero vs random: ΔSR=0.029, p=0.0060, cohens_dz=0.087

- uniform: muzero vs belief_greedy: ΔSR=-0.164, p=0.0000, cohens_dz=-0.394

- uniform: muzero vs dqn: ΔSR=0.347, p=0.0000, cohens_dz=1.074

- uniform: muzero vs ppo: ΔSR=0.388, p=0.0000, cohens_dz=1.119

- uniform: muzero vs dqn_gru: ΔSR=0.234, p=0.0000, cohens_dz=0.602

- uniform: muzero vs ppo_stack: ΔSR=0.349, p=0.0000, cohens_dz=0.971

- uniform: muzero vs pomcp: ΔSR=-0.088, p=0.0000, cohens_dz=-0.274

- N16: muzero vs sequential: ΔSR=-0.215, p=0.0000, cohens_dz=-0.549

- N16: muzero vs belief_greedy: ΔSR=-0.297, p=0.0000, cohens_dz=-0.997

- N16: muzero vs dqn: ΔSR=0.357, p=0.0000, cohens_dz=1.051

- N16: muzero vs ppo: ΔSR=0.384, p=0.0000, cohens_dz=1.081

- N16: muzero vs dqn_gru: ΔSR=0.215, p=0.0000, cohens_dz=0.524

- N16: muzero vs ppo_stack: ΔSR=0.329, p=0.0000, cohens_dz=0.925

- N16: muzero vs pomcp: ΔSR=-0.251, p=0.0000, cohens_dz=-0.864

- N24: muzero vs sequential: ΔSR=-0.433, p=0.0000, cohens_dz=-0.790

- N24: muzero vs random: ΔSR=-0.238, p=0.0000, cohens_dz=-0.580

- N24: muzero vs belief_greedy: ΔSR=-0.699, p=0.0000, cohens_dz=-1.802

- N24: muzero vs dqn: ΔSR=0.162, p=0.0000, cohens_dz=0.422

- N24: muzero vs ppo: ΔSR=0.118, p=0.0000, cohens_dz=0.269

- N24: muzero vs dqn_gru: ΔSR=0.040, p=0.0259, cohens_dz=0.100

- N24: muzero vs ppo_stack: ΔSR=0.073, p=0.0000, cohens_dz=0.199

- N24: muzero vs pomcp: ΔSR=-0.636, p=0.0000, cohens_dz=-1.713

- N32: muzero vs sequential: ΔSR=-0.213, p=0.0000, cohens_dz=-0.441

- N32: muzero vs random: ΔSR=-0.102, p=0.0000, cohens_dz=-0.347

- N32: muzero vs belief_greedy: ΔSR=-0.675, p=0.0000, cohens_dz=-2.897

- N32: muzero vs dqn: ΔSR=0.150, p=0.0000, cohens_dz=0.647

- N32: muzero vs ppo: ΔSR=0.200, p=0.0000, cohens_dz=0.793

- N32: muzero vs dqn_gru: ΔSR=0.192, p=0.0000, cohens_dz=0.833

- N32: muzero vs ppo_stack: ΔSR=0.163, p=0.0000, cohens_dz=0.656

- N32: muzero vs pomcp: ΔSR=-0.601, p=0.0000, cohens_dz=-2.387

- N48: muzero vs sequential: ΔSR=-0.178, p=0.0000, cohens_dz=-0.388

- N48: muzero vs random: ΔSR=-0.102, p=0.0000, cohens_dz=-0.294

- N48: muzero vs belief_greedy: ΔSR=-0.784, p=0.0000, cohens_dz=-2.350

- N48: muzero vs dqn: ΔSR=0.145, p=0.0000, cohens_dz=0.437

- N48: muzero vs ppo: ΔSR=0.118, p=0.0000, cohens_dz=0.387

- N48: muzero vs dqn_gru: ΔSR=0.126, p=0.0000, cohens_dz=0.400

- N48: muzero vs ppo_stack: ΔSR=0.126, p=0.0000, cohens_dz=0.376

- N48: muzero vs pomcp: ΔSR=-0.680, p=0.0000, cohens_dz=-1.874

- N64: muzero vs sequential: ΔSR=-0.225, p=0.0000, cohens_dz=-0.521

- N64: muzero vs random: ΔSR=-0.153, p=0.0000, cohens_dz=-0.563

- N64: muzero vs belief_greedy: ΔSR=-0.909, p=0.0000, cohens_dz=-3.918

- N64: muzero vs dqn: ΔSR=0.038, p=0.0003, cohens_dz=0.176

- N64: muzero vs ppo: ΔSR=0.038, p=0.0000, cohens_dz=0.279

- N64: muzero vs dqn_gru: ΔSR=0.033, p=0.0014, cohens_dz=0.153

- N64: muzero vs ppo_stack: ΔSR=0.026, p=0.0058, cohens_dz=0.124

- N64: muzero vs pomcp: ΔSR=-0.772, p=0.0000, cohens_dz=-2.362

- N8: muzero vs sequential: ΔSR=-0.266, p=0.0000, cohens_dz=-0.945

- N8: muzero vs random: ΔSR=-0.189, p=0.0000, cohens_dz=-0.674

- N8: muzero vs belief_greedy: ΔSR=-0.280, p=0.0000, cohens_dz=-1.068

- N8: muzero vs dqn: ΔSR=0.216, p=0.0000, cohens_dz=0.680

- N8: muzero vs ppo: ΔSR=0.270, p=0.0000, cohens_dz=1.096

- N8: muzero vs dqn_gru: ΔSR=-0.081, p=0.0000, cohens_dz=-0.312

- N8: muzero vs ppo_stack: ΔSR=0.281, p=0.0000, cohens_dz=0.836

- N8: muzero vs pomcp: ΔSR=-0.257, p=0.0000, cohens_dz=-0.974

- pfa_0.01: muzero vs sequential: ΔSR=-0.188, p=0.0000, cohens_dz=-0.476

- pfa_0.01: muzero vs belief_greedy: ΔSR=-0.286, p=0.0000, cohens_dz=-0.987

- pfa_0.01: muzero vs dqn: ΔSR=0.338, p=0.0000, cohens_dz=1.027

- pfa_0.01: muzero vs ppo: ΔSR=0.356, p=0.0000, cohens_dz=1.017

- pfa_0.01: muzero vs dqn_gru: ΔSR=0.230, p=0.0000, cohens_dz=0.593

- pfa_0.01: muzero vs ppo_stack: ΔSR=0.360, p=0.0000, cohens_dz=0.991

- pfa_0.01: muzero vs pomcp: ΔSR=-0.252, p=0.0000, cohens_dz=-0.920

- pfa_0.03: muzero vs sequential: ΔSR=-0.207, p=0.0000, cohens_dz=-0.522

- pfa_0.03: muzero vs belief_greedy: ΔSR=-0.271, p=0.0000, cohens_dz=-0.860

- pfa_0.03: muzero vs dqn: ΔSR=0.344, p=0.0000, cohens_dz=1.033

- pfa_0.03: muzero vs ppo: ΔSR=0.368, p=0.0000, cohens_dz=1.028

- pfa_0.03: muzero vs dqn_gru: ΔSR=0.217, p=0.0000, cohens_dz=0.553

- pfa_0.03: muzero vs ppo_stack: ΔSR=0.325, p=0.0000, cohens_dz=0.878

- pfa_0.03: muzero vs pomcp: ΔSR=-0.237, p=0.0000, cohens_dz=-0.846

- pfa_0.05: muzero vs sequential: ΔSR=-0.183, p=0.0000, cohens_dz=-0.444

- pfa_0.05: muzero vs belief_greedy: ΔSR=-0.293, p=0.0000, cohens_dz=-0.971

- pfa_0.05: muzero vs dqn: ΔSR=0.339, p=0.0000, cohens_dz=1.053

- pfa_0.05: muzero vs ppo: ΔSR=0.390, p=0.0000, cohens_dz=1.098

- pfa_0.05: muzero vs dqn_gru: ΔSR=0.211, p=0.0000, cohens_dz=0.548

- pfa_0.05: muzero vs ppo_stack: ΔSR=0.357, p=0.0000, cohens_dz=0.997

- pfa_0.05: muzero vs pomcp: ΔSR=-0.245, p=0.0000, cohens_dz=-0.867

- pfa_0.08: muzero vs sequential: ΔSR=-0.184, p=0.0000, cohens_dz=-0.423

- pfa_0.08: muzero vs belief_greedy: ΔSR=-0.282, p=0.0000, cohens_dz=-0.837

- pfa_0.08: muzero vs dqn: ΔSR=0.327, p=0.0000, cohens_dz=0.950

- pfa_0.08: muzero vs ppo: ΔSR=0.355, p=0.0000, cohens_dz=0.962

- pfa_0.08: muzero vs dqn_gru: ΔSR=0.196, p=0.0000, cohens_dz=0.502

- pfa_0.08: muzero vs ppo_stack: ΔSR=0.294, p=0.0000, cohens_dz=0.801

- pfa_0.08: muzero vs pomcp: ΔSR=-0.234, p=0.0000, cohens_dz=-0.785

- pfa_0.10: muzero vs sequential: ΔSR=-0.231, p=0.0000, cohens_dz=-0.590

- pfa_0.10: muzero vs belief_greedy: ΔSR=-0.293, p=0.0000, cohens_dz=-0.906

- pfa_0.10: muzero vs dqn: ΔSR=0.336, p=0.0000, cohens_dz=0.988

- pfa_0.10: muzero vs ppo: ΔSR=0.356, p=0.0000, cohens_dz=0.977

- pfa_0.10: muzero vs dqn_gru: ΔSR=0.211, p=0.0000, cohens_dz=0.535

- pfa_0.10: muzero vs ppo_stack: ΔSR=0.297, p=0.0000, cohens_dz=0.821

- pfa_0.10: muzero vs pomcp: ΔSR=-0.232, p=0.0000, cohens_dz=-0.786

- pfa_0.12: muzero vs sequential: ΔSR=-0.222, p=0.0000, cohens_dz=-0.571

- pfa_0.12: muzero vs belief_greedy: ΔSR=-0.306, p=0.0000, cohens_dz=-1.002

- pfa_0.12: muzero vs dqn: ΔSR=0.321, p=0.0000, cohens_dz=0.996

- pfa_0.12: muzero vs ppo: ΔSR=0.384, p=0.0000, cohens_dz=1.125

- pfa_0.12: muzero vs dqn_gru: ΔSR=0.207, p=0.0000, cohens_dz=0.541

- pfa_0.12: muzero vs ppo_stack: ΔSR=0.315, p=0.0000, cohens_dz=0.896

- pfa_0.12: muzero vs pomcp: ΔSR=-0.242, p=0.0000, cohens_dz=-0.826

- pfa_0.15: muzero vs sequential: ΔSR=-0.226, p=0.0000, cohens_dz=-0.594

- pfa_0.15: muzero vs belief_greedy: ΔSR=-0.282, p=0.0000, cohens_dz=-0.893

- pfa_0.15: muzero vs dqn: ΔSR=0.344, p=0.0000, cohens_dz=1.118

- pfa_0.15: muzero vs ppo: ΔSR=0.402, p=0.0000, cohens_dz=1.148

- pfa_0.15: muzero vs dqn_gru: ΔSR=0.217, p=0.0000, cohens_dz=0.584

- pfa_0.15: muzero vs ppo_stack: ΔSR=0.313, p=0.0000, cohens_dz=0.889

- pfa_0.15: muzero vs pomcp: ΔSR=-0.211, p=0.0000, cohens_dz=-0.703

- snr_+0dB: muzero vs sequential: ΔSR=-0.202, p=0.0000, cohens_dz=-0.510

- snr_+0dB: muzero vs random: ΔSR=0.038, p=0.0116, cohens_dz=0.113

- snr_+0dB: muzero vs belief_greedy: ΔSR=-0.270, p=0.0000, cohens_dz=-0.839

- snr_+0dB: muzero vs dqn: ΔSR=0.353, p=0.0000, cohens_dz=1.119

- snr_+0dB: muzero vs ppo: ΔSR=0.388, p=0.0000, cohens_dz=1.064

- snr_+0dB: muzero vs dqn_gru: ΔSR=0.242, p=0.0000, cohens_dz=0.628

- snr_+0dB: muzero vs ppo_stack: ΔSR=0.356, p=0.0000, cohens_dz=0.964

- snr_+0dB: muzero vs pomcp: ΔSR=-0.224, p=0.0000, cohens_dz=-0.760

- snr_+10dB: muzero vs sequential: ΔSR=-0.254, p=0.0000, cohens_dz=-0.969

- snr_+10dB: muzero vs belief_greedy: ΔSR=-0.266, p=0.0000, cohens_dz=-1.058

- snr_+10dB: muzero vs dqn: ΔSR=0.377, p=0.0000, cohens_dz=1.202

- snr_+10dB: muzero vs ppo: ΔSR=0.430, p=0.0000, cohens_dz=1.318

- snr_+10dB: muzero vs dqn_gru: ΔSR=0.242, p=0.0000, cohens_dz=0.635

- snr_+10dB: muzero vs ppo_stack: ΔSR=0.396, p=0.0000, cohens_dz=1.168

- snr_+10dB: muzero vs pomcp: ΔSR=-0.256, p=0.0000, cohens_dz=-0.994

- snr_+3dB: muzero vs sequential: ΔSR=-0.232, p=0.0000, cohens_dz=-0.663

- snr_+3dB: muzero vs belief_greedy: ΔSR=-0.284, p=0.0000, cohens_dz=-0.977

- snr_+3dB: muzero vs dqn: ΔSR=0.389, p=0.0000, cohens_dz=1.170

- snr_+3dB: muzero vs ppo: ΔSR=0.424, p=0.0000, cohens_dz=1.217

- snr_+3dB: muzero vs dqn_gru: ΔSR=0.240, p=0.0000, cohens_dz=0.579

- snr_+3dB: muzero vs ppo_stack: ΔSR=0.370, p=0.0000, cohens_dz=1.068

- snr_+3dB: muzero vs pomcp: ΔSR=-0.250, p=0.0000, cohens_dz=-0.865

- snr_+6dB: muzero vs sequential: ΔSR=-0.230, p=0.0000, cohens_dz=-0.675

- snr_+6dB: muzero vs belief_greedy: ΔSR=-0.274, p=0.0000, cohens_dz=-1.061

- snr_+6dB: muzero vs dqn: ΔSR=0.398, p=0.0000, cohens_dz=1.209

- snr_+6dB: muzero vs ppo: ΔSR=0.404, p=0.0000, cohens_dz=1.112

- snr_+6dB: muzero vs dqn_gru: ΔSR=0.262, p=0.0000, cohens_dz=0.632

- snr_+6dB: muzero vs ppo_stack: ΔSR=0.369, p=0.0000, cohens_dz=1.018

- snr_+6dB: muzero vs pomcp: ΔSR=-0.250, p=0.0000, cohens_dz=-0.930

- snr_-10dB: muzero vs sequential: ΔSR=-0.092, p=0.0000, cohens_dz=-0.191

- snr_-10dB: muzero vs belief_greedy: ΔSR=-0.320, p=0.0000, cohens_dz=-0.763

- snr_-10dB: muzero vs dqn: ΔSR=0.187, p=0.0000, cohens_dz=0.601

- snr_-10dB: muzero vs ppo: ΔSR=0.206, p=0.0000, cohens_dz=0.580

- snr_-10dB: muzero vs dqn_gru: ΔSR=0.108, p=0.0000, cohens_dz=0.304

- snr_-10dB: muzero vs ppo_stack: ΔSR=0.168, p=0.0000, cohens_dz=0.467

- snr_-10dB: muzero vs pomcp: ΔSR=-0.164, p=0.0000, cohens_dz=-0.543

- snr_-3dB: muzero vs sequential: ΔSR=-0.226, p=0.0000, cohens_dz=-0.524

- snr_-3dB: muzero vs belief_greedy: ΔSR=-0.292, p=0.0000, cohens_dz=-0.786

- snr_-3dB: muzero vs dqn: ΔSR=0.299, p=0.0000, cohens_dz=0.903

- snr_-3dB: muzero vs ppo: ΔSR=0.344, p=0.0000, cohens_dz=0.944

- snr_-3dB: muzero vs dqn_gru: ΔSR=0.192, p=0.0000, cohens_dz=0.493

- snr_-3dB: muzero vs ppo_stack: ΔSR=0.299, p=0.0000, cohens_dz=0.829

- snr_-3dB: muzero vs pomcp: ΔSR=-0.230, p=0.0000, cohens_dz=-0.733

- snr_-6dB: muzero vs sequential: ΔSR=-0.169, p=0.0000, cohens_dz=-0.376

- snr_-6dB: muzero vs belief_greedy: ΔSR=-0.321, p=0.0000, cohens_dz=-0.805

- snr_-6dB: muzero vs dqn: ΔSR=0.267, p=0.0000, cohens_dz=0.821

- snr_-6dB: muzero vs ppo: ΔSR=0.278, p=0.0000, cohens_dz=0.756

- snr_-6dB: muzero vs dqn_gru: ΔSR=0.163, p=0.0000, cohens_dz=0.433

- snr_-6dB: muzero vs ppo_stack: ΔSR=0.216, p=0.0000, cohens_dz=0.597

- snr_-6dB: muzero vs pomcp: ΔSR=-0.214, p=0.0000, cohens_dz=-0.693

## 11. Ablation Studies

- MuZero完整: SR(Medium)=0.571±0.151

- D w/o belief: SR(Medium)=—±—

- E w/o memory: SR(Medium)=0.608±0.070

- F w/o域随机化: SR(Medium)=0.642±0.059

- G 塑形奖励: SR(Medium)=0.654±0.029

- A w/o MCTS: SR(Medium)=0.499±0.113

- B M=25: SR=0.227

- B M=50: SR=0.536

- B M=100: SR=0.653

- B M=200: SR=0.608

- C D=1: SR=0.614

- C D=3: SR=0.679

- C D=5: SR=0.546

- C D=10: SR=0.679

## 12-13. Robustness & OOD

（OOD相对退化见processed/ood_drop.csv，式(35)）

| 方法 | ID SR | OOD-SNR | OOD-PFA | OOD-N | OOD-Prior | OOD-Reverb |
|---|---|---|---|---|---|---|
|Sequential|0.882|0.974|0.872|0.338|0.874|0.880|
|Random|0.660|0.716|0.644|0.288|0.665|0.656|
|Belief-Greedy|0.970|1.000|0.946|0.968|0.000|0.952|
|POMCP (known model)|0.921|0.980|0.879|0.867|0.315|0.904|
|DQN|0.331|0.354|0.333|0.063|0.340|0.334|
|PPO|0.333|0.316|0.262|0.040|0.302|0.298|
|DQN-GRU (memory)|0.470|0.480|0.454|0.078|0.452|0.469|
|PPO-Stack (memory)|0.358|0.359|0.343|0.055|0.342|0.355|
|MuZero (ours)|0.675|0.738|0.646|0.181|0.686|0.665|

## 14. Prior Bias

- uniform: MuZero SR=0.683, Belief-Greedy SR=0.847

- correct: MuZero SR=0.683, Belief-Greedy SR=0.962

- biased: MuZero SR=0.683, Belief-Greedy SR=0.422

- adversarial: MuZero SR=0.683, Belief-Greedy SR=0.000

## 15. Level 2 Validation

| 方法 | Level1 SR(Medium) | Level2 SR(Medium) |
|---|---|---|
|Sequential|0.899|0.877|
|Belief-Greedy|0.960|0.960|
|PPO|0.303|0.319|
|MuZero (ours)|0.679|0.693|

- easy: 方法排序Spearman ρ=0.949 (p=0.0513)


- medium: 方法排序Spearman ρ=1.000 (p=0.0000)


- hard: 方法排序Spearman ρ=1.000 (p=0.0000)

## 16. Computational Cost

- Sequential: 免训练（规划/启发式），平均决策时延=0.00ms

- Random: 免训练（规划/启发式），平均决策时延=0.00ms

- Belief-Greedy: 免训练（规划/启发式），平均决策时延=0.00ms

- POMCP (known model): 免训练（规划/启发式），平均决策时延=4.50ms

- DQN: 平均每种子训练时长=572s（5种子），平均决策时延=0.03ms

- PPO: 平均每种子训练时长=399s（5种子），平均决策时延=0.04ms

- DQN-GRU (memory): 平均每种子训练时长=357s（5种子），平均决策时延=0.03ms

- PPO-Stack (memory): 平均每种子训练时长=527s（5种子），平均决策时延=0.04ms

- MuZero (ours): 平均每种子训练时长=11114s（5种子），平均决策时延=6.75ms

## 17. Mechanism Analysis

机理数据文件：5个（信念热图/搜索轨迹/MCTS访问分布/原始vs改进策略），对应figures/17-20系列图。

## 18. Unexpected Findings

- MuZero在Medium上未超过Belief-Greedy（信念为该POMDP的充分统计量→单步贪心已很强的结构性原因，见design_notes B6）。

## 19. Limitations

- 二维离散方位空间、单静止目标、离散扇区动作（开题报告研究边界）。
- 声传播为参数化简化模型（式(9)-(11)）；噪声/混响为统计模型。
- Level 2为复基带统计一致实现，非全波形海底混响物理仿真。
- 报告B节所列16项设计决策为开题报告留白处的文档化选择。

## 20. Reproducibility

- 复现命令：`python run_all_experiments.py --mode full`（断点续跑自动skip已完成）。
- 全部配置：configs/*.yaml；实际使用配置与manifest：results/raw/manifests/。
- 原始逐回合数据：results/raw/；质量检查：results/processed/quality_check.json。
