# 评审意见逐条回应（第二轮）

针对评审提出的"论文声称9方法但代码/数据只有6个"等问题，逐条核验并附仓库证据。总体说明：评审快照取自 **Release v1.0-full-data 与旧 main 提交**，本轮整改（commit `5a0d94e` 起）已将三个新基线的**实现、训练、评测、统计数据**全部入库，并发布 **Release v2.0-full-data** 补齐原始数据。

---

## 1. "论文声称9个方法，代码和数据里只有6个，POMCP/DQN-GRU/PPO-Stack的全部数字无支撑"

**评审基于旧快照；当前仓库三者均已真实实现、训练、评测。**

代码（main 分支，`agents/`）：
- `agents/pomcp.py` — POMCP：粒子滤波（K=500）+ PUCT 树搜索（M=200，D=10，γ=0.99），使用真实观测模型的 known-model 规划器（Silver & Veness 2010）
- `agents/dqn_gru.py` — DQN-GRU：GRU(256) 处理逐步 raw 观测 + Q 头
- `agents/ppo_stack.py` — PPO-Stack：最近 K=8 帧拼接（1552 维）+ actor-critic

数据（Release v2.0-full-data `raw_v3.zip`）：
- 主测试：`results_v3/raw/main/{easy,medium,hard}/{pomcp,dqn_gru,ppo_stack}_main_seed{101..105}.jsonl` —— 每方法 5 种子 × 3 难度 × 1000 配对回合
- 扩展实验：OOD 五类、鲁棒性 sweep（SNR/PFA/N）、先验偏差四类均已补齐三新基线（`raw_v3.zip` 内 ood/ robustness/ prior_bias/ 目录）
- 检查点：`checkpoints_v3.zip` 含 dqn_gru/ppo_stack 各种子的 best_val/last/final + 训练日志 + summary

注意 `baselines/` 目录只有 sequential/random/belief_greedy 是**有意为之**：POMCP 是规划器（非静态启发式），按项目结构归 `agents/`。

## 2. "R1(belief-informed) 机制下学习方法的成绩无数据支撑，仓库只有一套运行"

**不存在。** 仓库保留两套完整运行：
- R1（belief-informed）= `results/` 目录（v1 轮）：`results/processed/main_results.csv` 已被 git 跟踪。逐值核验：DQN 0.997/0.920/0.174、PPO 0.744/0.473/0.113、MuZero 0.674/0.612/0.175（Easy/Medium/Hard）——与论文表 3-1 的 R1 块**完全一致**
- R3（blind search）= `results_v3/` 目录：111 行主测试结果，9 方法齐全
- v1 的逐回合原始数据在 Release v1.0-full-data 的 `raw_v1.zip`（266MB）

评审可能只检索了 `results_v3/` 而未注意 `results/`。

## 3. "D=5 消融：仓库扫描值 0.55，论文写 0.679"

**属实，是真 bug，已修复（commit `7b1bae3`）。**
- 根因：消融结果 CSV 中 MuZero 的 main 行混入了 M 扫描与 D 扫描的行（扫描行 variant 也是 main），D=5 聚合时把 M 扫描（D=5 但 M≠100）的行也算了进去 → 0.55 的污染均值
- 修复：M 扫描过滤固定 `planning_depth=5`，D 扫描过滤固定 `num_simulations=100`（双重过滤），并同步修正 `experiments/tables.py`、`experiments/figures.py`、`tools/generate_paper.py` 三处聚合
- 修正后仓库 `table4_mcts_ablation.csv` 的 D=5 行为 0.68±0.04，与主测试一致

## 4. "table7 把训练步数写成 1,000,000"

**属实（旧表格硬编码），已修复。**
- `experiments/tables.py` 的预算列原来硬编码 `"1000000"`，已改为从训练配置读取（`training.total_env_steps`），当前 `table7_computational_cost.csv` 显示 2000000，与 checkpoint 记录一致

## 5. "消融基线口径：仓库 M=50/0.57 vs 论文 0.679/−0.180"

**同第 3 条的连锁问题，已对齐。**
- 仓库表 4 基线行原标签 "M=50, D=5" 是笔误（实际默认配置 M=100, D=5），且数值取了污染均值 0.57
- 修正后基线行 = "MuZero (M=100, D=5)"，数值取主测试干净值 0.725/0.679/0.278（Easy/Medium/Hard），与论文表 3-3 完全同口径
- MCTS 增益 −0.180 = 0.679（full）− 0.499（w/o MCTS），仓库表与论文一致

---

## 复现入口

```bash
python run_all_experiments.py --mode full --config-dir configs_v3 --results-root results_v3
```

全部数字可由 `results_v3/raw/`（Release v2.0）程序化重算；`results_v3/processed/quality_check.json` 质量检查 PASSED。
