# FINAL_SUMMARY（最终交付摘要，中文，面向非专业读者）

> 本摘要由程序从真实实验原始数据自动生成，不美化、不虚构。


## 1. 实验是否全部真正完成？

- 已完成 19 项；失败 0 项：无；未运行 0 项：无。
（逐项状态见 results/reports/experiment_status.md）


## 2. 哪些实验失败/未运行？
见上与experiment_status.md；如列出FAILED/NOT RUN即未完成，绝不显示completed。


## 3. MuZero最终表现如何？

- Easy SR=0.674, Medium SR=0.612, Hard SR=0.175（Medium平均发现步数8.2步）。


## 4-8. 与各基线比较（Medium难度成功率）

- MuZero(0.612) 低于 Sequential(0.899)，差值-0.287。

- MuZero(0.612) 低于 Random(0.663)，差值-0.051。

- MuZero(0.612) 低于 Belief-Greedy(0.959)，差值-0.347。

- MuZero(0.612) 低于 DQN(0.920)，差值-0.308。

- MuZero(0.612) 高于 PPO(0.473)，差值0.139。


## 9. MCTS是否真正带来收益？

- 完整MuZero SR=0.612 vs 去除MCTS SR=0.334：MCTS带来提升0.278（同训练模型，评测期开关，消融A）。


## 10. Belief是否有作用？

- 完整0.612 vs w/o belief 0.533：信念输入贡献0.079（消融D）。


## 11. GRU memory是否有作用？

- 完整0.612 vs w/o memory 0.487：GRU记忆贡献0.125（消融E）。


## 12. OOD泛化表现如何？

- ood_snr: SR 0.611→0.663（退化-8.6%）

- ood_pfa: SR 0.611→0.586（退化4.0%）

- ood_n: SR 0.611→0.159（退化74.0%）

- ood_prior: SR 0.611→0.554（退化9.4%）

- ood_reverb: SR 0.611→0.612（退化-0.3%）


## 13. Level 1和Level 2结论是否一致？

- easy: Spearman ρ=1.000——排序一致

- medium: Spearman ρ=1.000——排序一致

- hard: Spearman ρ=1.000——排序一致


## 14. 是否达到统计显著？

- MuZero相关配对检验共16项，Holm校正后显著16项：

  - adversarial: vs sequential p=0.0000

  - adversarial: vs random p=0.0000

  - adversarial: vs dqn p=0.0000

  - adversarial: vs ppo p=0.0000

  - biased: vs sequential p=0.0000

  - biased: vs random p=0.0000

  - biased: vs dqn p=0.0000

  - biased: vs ppo p=0.0000

  - correct: vs sequential p=0.0000

  - correct: vs random p=0.0000

  - correct: vs dqn p=0.0000

  - correct: vs ppo p=0.0000

  - uniform: vs sequential p=0.0000

  - uniform: vs random p=0.0000

  - uniform: vs dqn p=0.0000

  - uniform: vs ppo p=0.0000


## 15. 效应量多大？
见effect_sizes.csv；上列显著项的效应量：

  - vs sequential: cohens_dz=-0.905

  - vs random: cohens_dz=-0.358

  - vs dqn: cohens_dz=-0.479

  - vs ppo: cohens_dz=0.415

  - vs sequential: cohens_dz=-0.880

  - vs random: cohens_dz=-0.325

  - vs dqn: cohens_dz=-0.604

  - vs ppo: cohens_dz=0.347

  - vs sequential: cohens_dz=-0.693

  - vs random: cohens_dz=-0.140

  - vs dqn: cohens_dz=-1.083

  - vs ppo: cohens_dz=0.333

  - vs sequential: cohens_dz=-0.847

  - vs random: cohens_dz=-0.336

  - vs dqn: cohens_dz=-0.879

  - vs ppo: cohens_dz=0.310


## 16. MuZero付出了多少额外计算成本？

- DQN: 训练148s/种子，决策时延0.1ms/步

- PPO: 训练197s/种子，决策时延0.0ms/步

- MuZero: 训练3716s/种子，决策时延5.2ms/步


## 17. 哪些结果最适合放进小论文？

- 表4/表7式主对比+统计检验（若MuZero显著优于基线）。
- 消融A（MCTS贡献）与B/C（性能-计算权衡曲线）。
- OOD退化表（表6）与先验纠偏KL曲线。
- Level1↔Level2一致性（增强可信度）。
- 若MuZero未显著领先：规划收益边界分析（开题报告风险4预案）同样是有效结论。


## 18. 哪些结果不支持原研究假设？

- MuZero在Medium上未超过Belief-Greedy（信念为该POMDP的充分统计量→单步贪心已很强的结构性原因，见design_notes B6）。


## 19. 本实验最重要的真实结论（3-5条）

1. MuZero(0.612)未超过信念贪心(0.959)：信念为该POMDP充分统计量，单步贪心已接近最优（诚实结构性结论）。
2. MCTS对决策期性能的净贡献：0.278（正）。
3. MuZero在OOD下最大退化出现在ood_n（74.0%）。
4. MuZero与基线的差异经Holm校正后16项统计显著（共16项检验）。


## 20. 下一步建议补什么实验？

- 若主实验领先不显著：扩大种子数（≥10）、增加训练预算、或按开题报告风险4预案转向『规划有效条件与收益边界』分析。
- 波形参数联合优化（开题报告扩展方向）。
- 多目标/连续方位扩展。
