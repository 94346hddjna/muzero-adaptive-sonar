# FINAL_SUMMARY（最终交付摘要，中文，面向非专业读者）

> 本摘要由程序从真实实验原始数据自动生成，不美化、不虚构。


## 1. 实验是否全部真正完成？

- 已完成 19 项；失败 0 项：无；未运行 0 项：无。
（逐项状态见 results/reports/experiment_status.md）


## 2. 哪些实验失败/未运行？
见上与experiment_status.md；如列出FAILED/NOT RUN即未完成，绝不显示completed。


## 3. MuZero最终表现如何？

- Easy SR=0.725, Medium SR=0.679, Hard SR=0.278（Medium平均发现步数8.9步）。


## 4-8. 与各基线比较（Medium难度成功率）

- MuZero(0.679) 低于 Sequential(0.899)，差值-0.220。

- MuZero(0.679) 高于 Random(0.663)，差值0.016。

- MuZero(0.679) 低于 Belief-Greedy(0.960)，差值-0.281。

- MuZero(0.679) 高于 DQN(0.335)，差值0.344。

- MuZero(0.679) 高于 PPO(0.303)，差值0.376。


## 9. MCTS是否真正带来收益？

- 完整MuZero SR=0.679 vs 去除MCTS SR=0.499：MCTS带来提升0.180（同训练模型，评测期开关，消融A）。


## 10. Belief是否有作用？


## 11. GRU memory是否有作用？

- 完整0.679 vs w/o memory 0.608：GRU记忆贡献0.071（消融E）。


## 12. OOD泛化表现如何？

- ood_snr: SR 0.675→0.738（退化-9.4%）

- ood_pfa: SR 0.675→0.646（退化4.4%）

- ood_n: SR 0.675→0.181（退化73.2%）

- ood_prior: SR 0.675→0.686（退化-1.5%）

- ood_reverb: SR 0.675→0.665（退化1.5%）


## 13. Level 1和Level 2结论是否一致？

- easy: Spearman ρ=0.949——排序一致

- medium: Spearman ρ=1.000——排序一致

- hard: Spearman ρ=1.000——排序一致


## 14. 是否达到统计显著？

- MuZero相关配对检验共132项，Holm校正后显著112项：

  - easy: vs sequential p=0.0000

  - easy: vs random p=0.0000

  - easy: vs dqn p=0.0000

  - easy: vs ppo p=0.0000

  - hard: vs sequential p=0.0000

  - hard: vs random p=0.0000

  - hard: vs dqn p=0.0000

  - hard: vs ppo p=0.0000

  - medium: vs sequential p=0.0000

  - medium: vs dqn p=0.0000

  - medium: vs ppo p=0.0000

  - id: vs sequential p=0.0000

  - id: vs dqn p=0.0000

  - id: vs ppo p=0.0000

  - ood_n: vs sequential p=0.0000

  - ood_n: vs random p=0.0000

  - ood_n: vs dqn p=0.0000

  - ood_n: vs ppo p=0.0000

  - ood_pfa: vs sequential p=0.0000

  - ood_pfa: vs dqn p=0.0000

  - ood_pfa: vs ppo p=0.0000

  - ood_prior: vs sequential p=0.0000

  - ood_prior: vs dqn p=0.0000

  - ood_prior: vs ppo p=0.0000

  - ood_reverb: vs sequential p=0.0000

  - ood_reverb: vs dqn p=0.0000

  - ood_reverb: vs ppo p=0.0000

  - ood_snr: vs sequential p=0.0000

  - ood_snr: vs dqn p=0.0000

  - ood_snr: vs ppo p=0.0000

  - adversarial: vs sequential p=0.0000

  - adversarial: vs random p=0.0060

  - adversarial: vs dqn p=0.0000

  - adversarial: vs ppo p=0.0000

  - biased: vs sequential p=0.0000

  - biased: vs random p=0.0060

  - biased: vs dqn p=0.0000

  - biased: vs ppo p=0.0000

  - correct: vs sequential p=0.0000

  - correct: vs random p=0.0060

  - correct: vs dqn p=0.0000

  - correct: vs ppo p=0.0000

  - uniform: vs sequential p=0.0000

  - uniform: vs random p=0.0060

  - uniform: vs dqn p=0.0000

  - uniform: vs ppo p=0.0000

  - N16: vs sequential p=0.0000

  - N16: vs dqn p=0.0000

  - N16: vs ppo p=0.0000

  - N24: vs sequential p=0.0000

  - N24: vs random p=0.0000

  - N24: vs dqn p=0.0000

  - N24: vs ppo p=0.0000

  - N32: vs sequential p=0.0000

  - N32: vs random p=0.0000

  - N32: vs dqn p=0.0000

  - N32: vs ppo p=0.0000

  - N48: vs sequential p=0.0000

  - N48: vs random p=0.0000

  - N48: vs dqn p=0.0000

  - N48: vs ppo p=0.0000

  - N64: vs sequential p=0.0000

  - N64: vs random p=0.0000

  - N64: vs dqn p=0.0001

  - N64: vs ppo p=0.0000

  - N8: vs sequential p=0.0000

  - N8: vs random p=0.0000

  - N8: vs dqn p=0.0000

  - N8: vs ppo p=0.0000

  - pfa_0.01: vs sequential p=0.0000

  - pfa_0.01: vs dqn p=0.0000

  - pfa_0.01: vs ppo p=0.0000

  - pfa_0.03: vs sequential p=0.0000

  - pfa_0.03: vs dqn p=0.0000

  - pfa_0.03: vs ppo p=0.0000

  - pfa_0.05: vs sequential p=0.0000

  - pfa_0.05: vs dqn p=0.0000

  - pfa_0.05: vs ppo p=0.0000

  - pfa_0.08: vs sequential p=0.0000

  - pfa_0.08: vs dqn p=0.0000

  - pfa_0.08: vs ppo p=0.0000

  - pfa_0.10: vs sequential p=0.0000

  - pfa_0.10: vs dqn p=0.0000

  - pfa_0.10: vs ppo p=0.0000

  - pfa_0.12: vs sequential p=0.0000

  - pfa_0.12: vs dqn p=0.0000

  - pfa_0.12: vs ppo p=0.0000

  - pfa_0.15: vs sequential p=0.0000

  - pfa_0.15: vs dqn p=0.0000

  - pfa_0.15: vs ppo p=0.0000

  - snr_+0dB: vs sequential p=0.0000

  - snr_+0dB: vs random p=0.0116

  - snr_+0dB: vs dqn p=0.0000

  - snr_+0dB: vs ppo p=0.0000

  - snr_+10dB: vs sequential p=0.0000

  - snr_+10dB: vs dqn p=0.0000

  - snr_+10dB: vs ppo p=0.0000

  - snr_+3dB: vs sequential p=0.0000

  - snr_+3dB: vs dqn p=0.0000

  - snr_+3dB: vs ppo p=0.0000

  - snr_+6dB: vs sequential p=0.0000

  - snr_+6dB: vs dqn p=0.0000

  - snr_+6dB: vs ppo p=0.0000

  - snr_-10dB: vs sequential p=0.0000

  - snr_-10dB: vs dqn p=0.0000

  - snr_-10dB: vs ppo p=0.0000

  - snr_-3dB: vs sequential p=0.0000

  - snr_-3dB: vs dqn p=0.0000

  - snr_-3dB: vs ppo p=0.0000

  - snr_-6dB: vs sequential p=0.0000

  - snr_-6dB: vs dqn p=0.0000

  - snr_-6dB: vs ppo p=0.0000


## 15. 效应量多大？
见effect_sizes.csv；上列显著项的效应量：

  - vs sequential: cohens_dz=-1.109

  - vs random: cohens_dz=-0.681

  - vs dqn: cohens_dz=0.857

  - vs ppo: cohens_dz=1.079

  - vs sequential: cohens_dz=-0.315

  - vs random: cohens_dz=-0.301

  - vs dqn: cohens_dz=0.594

  - vs ppo: cohens_dz=0.592

  - vs sequential: cohens_dz=-0.602

  - vs dqn: cohens_dz=1.045

  - vs ppo: cohens_dz=1.074

  - vs sequential: cohens_dz=-0.547

  - vs dqn: cohens_dz=1.045

  - vs ppo: cohens_dz=0.923

  - vs sequential: cohens_dz=-0.350

  - vs random: cohens_dz=-0.313

  - vs dqn: cohens_dz=0.368

  - vs ppo: cohens_dz=0.478

  - vs sequential: cohens_dz=-0.602

  - vs dqn: cohens_dz=0.958

  - vs ppo: cohens_dz=1.139

  - vs sequential: cohens_dz=-0.472

  - vs dqn: cohens_dz=1.039

  - vs ppo: cohens_dz=1.026

  - vs sequential: cohens_dz=-0.557

  - vs dqn: cohens_dz=1.006

  - vs ppo: cohens_dz=1.016

  - vs sequential: cohens_dz=-0.782

  - vs dqn: cohens_dz=1.223

  - vs ppo: cohens_dz=1.184

  - vs sequential: cohens_dz=-0.498

  - vs random: cohens_dz=0.087

  - vs dqn: cohens_dz=1.074

  - vs ppo: cohens_dz=1.119

  - vs sequential: cohens_dz=-0.498

  - vs random: cohens_dz=0.087

  - vs dqn: cohens_dz=1.074

  - vs ppo: cohens_dz=1.119

  - vs sequential: cohens_dz=-0.498

  - vs random: cohens_dz=0.087

  - vs dqn: cohens_dz=1.074

  - vs ppo: cohens_dz=1.119

  - vs sequential: cohens_dz=-0.498

  - vs random: cohens_dz=0.087

  - vs dqn: cohens_dz=1.074

  - vs ppo: cohens_dz=1.119

  - vs sequential: cohens_dz=-0.549

  - vs dqn: cohens_dz=1.051

  - vs ppo: cohens_dz=1.081

  - vs sequential: cohens_dz=-0.790

  - vs random: cohens_dz=-0.580

  - vs dqn: cohens_dz=0.422

  - vs ppo: cohens_dz=0.269

  - vs sequential: cohens_dz=-0.441

  - vs random: cohens_dz=-0.347

  - vs dqn: cohens_dz=0.647

  - vs ppo: cohens_dz=0.793

  - vs sequential: cohens_dz=-0.388

  - vs random: cohens_dz=-0.294

  - vs dqn: cohens_dz=0.437

  - vs ppo: cohens_dz=0.387

  - vs sequential: cohens_dz=-0.521

  - vs random: cohens_dz=-0.563

  - vs dqn: cohens_dz=0.176

  - vs ppo: cohens_dz=0.279

  - vs sequential: cohens_dz=-0.945

  - vs random: cohens_dz=-0.674

  - vs dqn: cohens_dz=0.680

  - vs ppo: cohens_dz=1.096

  - vs sequential: cohens_dz=-0.476

  - vs dqn: cohens_dz=1.027

  - vs ppo: cohens_dz=1.017

  - vs sequential: cohens_dz=-0.522

  - vs dqn: cohens_dz=1.033

  - vs ppo: cohens_dz=1.028

  - vs sequential: cohens_dz=-0.444

  - vs dqn: cohens_dz=1.053

  - vs ppo: cohens_dz=1.098

  - vs sequential: cohens_dz=-0.423

  - vs dqn: cohens_dz=0.950

  - vs ppo: cohens_dz=0.962

  - vs sequential: cohens_dz=-0.590

  - vs dqn: cohens_dz=0.988

  - vs ppo: cohens_dz=0.977

  - vs sequential: cohens_dz=-0.571

  - vs dqn: cohens_dz=0.996

  - vs ppo: cohens_dz=1.125

  - vs sequential: cohens_dz=-0.594

  - vs dqn: cohens_dz=1.118

  - vs ppo: cohens_dz=1.148

  - vs sequential: cohens_dz=-0.510

  - vs random: cohens_dz=0.113

  - vs dqn: cohens_dz=1.119

  - vs ppo: cohens_dz=1.064

  - vs sequential: cohens_dz=-0.969

  - vs dqn: cohens_dz=1.202

  - vs ppo: cohens_dz=1.318

  - vs sequential: cohens_dz=-0.663

  - vs dqn: cohens_dz=1.170

  - vs ppo: cohens_dz=1.217

  - vs sequential: cohens_dz=-0.675

  - vs dqn: cohens_dz=1.209

  - vs ppo: cohens_dz=1.112

  - vs sequential: cohens_dz=-0.191

  - vs dqn: cohens_dz=0.601

  - vs ppo: cohens_dz=0.580

  - vs sequential: cohens_dz=-0.524

  - vs dqn: cohens_dz=0.903

  - vs ppo: cohens_dz=0.944

  - vs sequential: cohens_dz=-0.376

  - vs dqn: cohens_dz=0.821

  - vs ppo: cohens_dz=0.756


## 16. MuZero付出了多少额外计算成本？

- DQN: 训练572s/种子，决策时延0.0ms/步

- PPO: 训练399s/种子，决策时延0.0ms/步

- MuZero: 训练11131s/种子，决策时延6.7ms/步


## 17. 哪些结果最适合放进小论文？

- 表4/表7式主对比+统计检验（若MuZero显著优于基线）。
- 消融A（MCTS贡献）与B/C（性能-计算权衡曲线）。
- OOD退化表（表6）与先验纠偏KL曲线。
- Level1↔Level2一致性（增强可信度）。
- 若MuZero未显著领先：规划收益边界分析（开题报告风险4预案）同样是有效结论。


## 18. 哪些结果不支持原研究假设？

- MuZero在Medium上未超过Belief-Greedy（信念为该POMDP的充分统计量→单步贪心已很强的结构性原因，见design_notes B6）。


## 19. 本实验最重要的真实结论（3-5条）

1. MuZero(0.679)未超过信念贪心(0.960)：信念为该POMDP充分统计量，单步贪心已接近最优（诚实结构性结论）。
2. MCTS对决策期性能的净贡献：0.180（正）。
3. MuZero在OOD下最大退化出现在ood_n（73.2%）。
4. MuZero与基线的差异经Holm校正后112项统计显著（共132项检验）。


## 20. 下一步建议补什么实验？

- 若主实验领先不显著：扩大种子数（≥10）、增加训练预算、或按开题报告风险4预案转向『规划有效条件与收益边界』分析。
- 波形参数联合优化（开题报告扩展方向）。
- 多目标/连续方位扩展。
