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
|Belief-Greedy|1.000|0.959|0.000|2.177|3.318|0.059|
|DQN|0.997|0.920|0.174|3.271|5.415|0.053|
|PPO|0.744|0.473|0.113|9.025|20.095|0.051|
|MuZero|0.674|0.612|0.175|8.150|16.630|0.050|

样本效率（达到SR≥0.90所需环境步，未达到则标'未达到'）：

- DQN: 475456

- PPO: 未达到阈值

- MuZero: 未达到阈值

## 10. Statistical Tests

共16项配对检验（Holm校正，α=0.05），其中16项显著。详见results/statistics/statistical_tests.csv与effect_sizes.csv。

- adversarial: muzero vs sequential: ΔSR=-0.335, p=0.0000, cohens_dz=-0.905

- adversarial: muzero vs random: ΔSR=-0.110, p=0.0000, cohens_dz=-0.358

- adversarial: muzero vs dqn: ΔSR=-0.144, p=0.0000, cohens_dz=-0.479

- adversarial: muzero vs ppo: ΔSR=0.136, p=0.0000, cohens_dz=0.415

- biased: muzero vs sequential: ΔSR=-0.329, p=0.0000, cohens_dz=-0.880

- biased: muzero vs random: ΔSR=-0.105, p=0.0000, cohens_dz=-0.325

- biased: muzero vs dqn: ΔSR=-0.192, p=0.0000, cohens_dz=-0.604

- biased: muzero vs ppo: ΔSR=0.119, p=0.0000, cohens_dz=0.347

- correct: muzero vs sequential: ΔSR=-0.271, p=0.0000, cohens_dz=-0.693

- correct: muzero vs random: ΔSR=-0.047, p=0.0000, cohens_dz=-0.140

- correct: muzero vs dqn: ΔSR=-0.315, p=0.0000, cohens_dz=-1.083

- correct: muzero vs ppo: ΔSR=0.117, p=0.0000, cohens_dz=0.333

- uniform: muzero vs sequential: ΔSR=-0.340, p=0.0000, cohens_dz=-0.847

- uniform: muzero vs random: ΔSR=-0.115, p=0.0000, cohens_dz=-0.336

- uniform: muzero vs dqn: ΔSR=-0.274, p=0.0000, cohens_dz=-0.879

- uniform: muzero vs ppo: ΔSR=0.104, p=0.0000, cohens_dz=0.310

## 11. Ablation Studies

- MuZero完整: SR(Medium)=0.503±0.149

- D w/o belief: SR(Medium)=0.533±0.050

- E w/o memory: SR(Medium)=0.487±0.026

- F w/o域随机化: SR(Medium)=0.553±0.078

- G 塑形奖励: SR(Medium)=0.548±0.102

- A w/o MCTS: SR(Medium)=0.334±0.105

- B M=25: SR=0.232

- B M=50: SR=0.579

- B M=100: SR=0.518

- B M=200: SR=0.544

- C D=1: SR=0.531

- C D=3: SR=0.612

- C D=5: SR=0.504

- C D=10: SR=0.612

## 12-13. Robustness & OOD

（OOD相对退化见processed/ood_drop.csv，式(35)）

| 方法 | ID SR | OOD-SNR | OOD-PFA | OOD-N | OOD-Prior | OOD-Reverb |
|---|---|---|---|---|---|---|
|Sequential|0.882|0.974|0.872|0.338|0.874|0.880|
|Random|0.660|0.716|0.644|0.288|0.665|0.656|
|Belief-Greedy|0.970|1.000|0.958|0.942|0.000|0.966|
|DQN|0.915|0.988|0.896|0.491|0.691|0.893|
|PPO|0.464|0.498|0.467|0.035|0.384|0.456|
|MuZero|0.611|0.663|0.586|0.159|0.554|0.612|

## 14. Prior Bias

- uniform: MuZero SR=0.538, Belief-Greedy SR=0.843

- correct: MuZero SR=0.607, Belief-Greedy SR=0.968

- biased: MuZero SR=0.549, Belief-Greedy SR=0.483

- adversarial: MuZero SR=0.543, Belief-Greedy SR=0.000

## 15. Level 2 Validation

| 方法 | Level1 SR(Medium) | Level2 SR(Medium) |
|---|---|---|
|Sequential|0.899|0.877|
|Belief-Greedy|0.959|0.980|
|PPO|0.473|0.485|
|MuZero|0.612|0.625|

- easy: 方法排序Spearman ρ=1.000 (p=0.0000)


- medium: 方法排序Spearman ρ=1.000 (p=0.0000)


- hard: 方法排序Spearman ρ=1.000 (p=0.0000)

## 16. Computational Cost

- Sequential: 训练总时长=—s（0种子），平均决策时延=0.00ms

- Random: 训练总时长=—s（0种子），平均决策时延=0.00ms

- Belief-Greedy: 训练总时长=—s（0种子），平均决策时延=0.00ms

- DQN: 训练总时长=148s（5种子），平均决策时延=0.05ms

- PPO: 训练总时长=197s（5种子），平均决策时延=0.05ms

- MuZero: 训练总时长=3716s（5种子），平均决策时延=5.22ms

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
