# 基于MuZero的自适应水下目标声呐探测策略研究 —— 实验工程

硕士课题实验代码库。依据《开题报告》完整实现：POMDP形式化（式3）、贝叶斯信念更新
（式4-5）、两级声呐仿真环境（式6-14）、MuZero三网络+MCTS（式17-26）、五基线、
七组消融（表6）、OOD/先验偏差/Level2验证、统计检验与论文图表。

## 一键运行

```bash
pip install -r requirements.txt

python run_all_experiments.py --mode smoke   # 极小规模端到端验证（约3分钟）
python run_all_experiments.py --mode full    # 正式论文实验（多种子，长时间）
```

分阶段运行（支持断点续跑，已完成实验自动skip）：

```bash
python run_all_experiments.py --mode full --stage selfcheck   # 环境自检+测试+sanity
python run_all_experiments.py --mode full --stage train       # RL训练（DQN/PPO/MuZero+消融变体）
python run_all_experiments.py --mode full --stage main        # 主实验（6算法×3难度）
python run_all_experiments.py --mode full --stage ablation    # 消融A-G
python run_all_experiments.py --mode full --stage robustness  # sweep+OOD
python run_all_experiments.py --mode full --stage prior_bias  # 先验偏差
python run_all_experiments.py --mode full --stage level2      # 信号级验证
python run_all_experiments.py --mode full --stage analysis    # 统计+图表+报告
python run_all_experiments.py --mode full --seeds 101 102     # 自定义种子
```

中断后重新运行同一命令即自动续训（checkpoint+完成标记）。

## 目录结构

```
env/         声呐环境：acoustic_model(式6-13) belief(式4-5) level1_env(抽象环境)
             level2_env(LFM信号级) scenarios(种子隔离+固定场景)
agents/      muzero(式17-23) mcts(式24-26) dqn(式28) ppo(式29-30) networks replay_buffer
baselines/   sequential / random / belief_greedy(式27)
experiments/ train_rl 训练器 | suite评测引擎 | main/ablation/robustness/prior_bias/
             level2_validation 实验 | statistics(配对检验) figures(22图×双语)
             tables(7表×4格式) report quality_check status sanity case_studies
configs/     全部YAML配置（smoke节控制小规模）
tests/       9个单元测试（63项断言）
results/     raw/(原始逐回合数据) processed/ statistics/ figures/ tables/
             checkpoints/ reports/ logs/ + FINAL_SUMMARY.md
```

## 科研诚信机制

- **一切数值来自真实运行**：原始逐回合JSONL存于results/raw/，汇总由raw程序化导出
- **质量闸门**：quality_check.py扫描NaN/Inf/缺种子/种子重叠/常数输出，异常即报错
- **状态如实**：experiment_status.md只显示completed/FAILED/NOT RUN
- **种子隔离**：Train[101-105]/Val 20250901/Test 30303/OOD 40404互不相交并测试验证
- **设计决策透明**：开题报告留白处的16+2项选择全部记录于docs/design_notes.md

## 关键文档

- `docs/requirements_traceability.md`：开题报告要求→代码→实验→输出的追溯矩阵
- `docs/design_notes.md`：设计决策与歧义记录（B1-B18）
- `results/FINAL_SUMMARY.md`：最终交付摘要（实验完成后生成）
