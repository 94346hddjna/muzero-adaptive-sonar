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

- MuZero(0.679) 低于 POMCP (known model)(0.911)，差值-0.232。

- MuZero(0.679) 高于 DQN(0.335)，差值0.344。

- MuZero(0.679) 高于 PPO(0.303)，差值0.376。

- MuZero(0.679) 高于 DQN-GRU (memory)(0.448)，差值0.230。

- MuZero(0.679) 高于 PPO-Stack (memory)(0.339)，差值0.340。


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

- MuZero相关配对检验共264项，Holm校正后显著244项：

  - easy: vs sequential p=0.0000

  - easy: vs random p=0.0000

  - easy: vs belief_greedy p=0.0000

  - easy: vs dqn p=0.0000

  - easy: vs ppo p=0.0000

  - easy: vs dqn_gru p=0.0000

  - easy: vs ppo_stack p=0.0000

  - easy: vs pomcp p=0.0000

  - hard: vs sequential p=0.0000

  - hard: vs random p=0.0000

  - hard: vs belief_greedy p=0.0000

  - hard: vs dqn p=0.0000

  - hard: vs ppo p=0.0000

  - hard: vs dqn_gru p=0.0000

  - hard: vs ppo_stack p=0.0000

  - hard: vs pomcp p=0.0000

  - medium: vs sequential p=0.0000

  - medium: vs belief_greedy p=0.0000

  - medium: vs dqn p=0.0000

  - medium: vs ppo p=0.0000

  - medium: vs dqn_gru p=0.0000

  - medium: vs ppo_stack p=0.0000

  - medium: vs pomcp p=0.0000

  - id: vs sequential p=0.0000

  - id: vs belief_greedy p=0.0000

  - id: vs dqn p=0.0000

  - id: vs ppo p=0.0000

  - id: vs dqn_gru p=0.0000

  - id: vs ppo_stack p=0.0000

  - id: vs pomcp p=0.0000

  - ood_n: vs sequential p=0.0000

  - ood_n: vs random p=0.0000

  - ood_n: vs belief_greedy p=0.0000

  - ood_n: vs dqn p=0.0000

  - ood_n: vs ppo p=0.0000

  - ood_n: vs dqn_gru p=0.0000

  - ood_n: vs ppo_stack p=0.0000

  - ood_n: vs pomcp p=0.0000

  - ood_pfa: vs sequential p=0.0000

  - ood_pfa: vs belief_greedy p=0.0000

  - ood_pfa: vs dqn p=0.0000

  - ood_pfa: vs ppo p=0.0000

  - ood_pfa: vs dqn_gru p=0.0000

  - ood_pfa: vs ppo_stack p=0.0000

  - ood_pfa: vs pomcp p=0.0000

  - ood_prior: vs sequential p=0.0000

  - ood_prior: vs belief_greedy p=0.0000

  - ood_prior: vs dqn p=0.0000

  - ood_prior: vs ppo p=0.0000

  - ood_prior: vs dqn_gru p=0.0000

  - ood_prior: vs ppo_stack p=0.0000

  - ood_prior: vs pomcp p=0.0000

  - ood_reverb: vs sequential p=0.0000

  - ood_reverb: vs belief_greedy p=0.0000

  - ood_reverb: vs dqn p=0.0000

  - ood_reverb: vs ppo p=0.0000

  - ood_reverb: vs dqn_gru p=0.0000

  - ood_reverb: vs ppo_stack p=0.0000

  - ood_reverb: vs pomcp p=0.0000

  - ood_snr: vs sequential p=0.0000

  - ood_snr: vs belief_greedy p=0.0000

  - ood_snr: vs dqn p=0.0000

  - ood_snr: vs ppo p=0.0000

  - ood_snr: vs dqn_gru p=0.0000

  - ood_snr: vs ppo_stack p=0.0000

  - ood_snr: vs pomcp p=0.0000

  - adversarial: vs sequential p=0.0000

  - adversarial: vs random p=0.0060

  - adversarial: vs belief_greedy p=0.0000

  - adversarial: vs dqn p=0.0000

  - adversarial: vs ppo p=0.0000

  - adversarial: vs dqn_gru p=0.0000

  - adversarial: vs ppo_stack p=0.0000

  - adversarial: vs pomcp p=0.0000

  - biased: vs sequential p=0.0000

  - biased: vs random p=0.0060

  - biased: vs belief_greedy p=0.0000

  - biased: vs dqn p=0.0000

  - biased: vs ppo p=0.0000

  - biased: vs dqn_gru p=0.0000

  - biased: vs ppo_stack p=0.0000

  - biased: vs pomcp p=0.0000

  - correct: vs sequential p=0.0000

  - correct: vs random p=0.0060

  - correct: vs belief_greedy p=0.0000

  - correct: vs dqn p=0.0000

  - correct: vs ppo p=0.0000

  - correct: vs dqn_gru p=0.0000

  - correct: vs ppo_stack p=0.0000

  - correct: vs pomcp p=0.0000

  - uniform: vs sequential p=0.0000

  - uniform: vs random p=0.0060

  - uniform: vs belief_greedy p=0.0000

  - uniform: vs dqn p=0.0000

  - uniform: vs ppo p=0.0000

  - uniform: vs dqn_gru p=0.0000

  - uniform: vs ppo_stack p=0.0000

  - uniform: vs pomcp p=0.0000

  - N16: vs sequential p=0.0000

  - N16: vs belief_greedy p=0.0000

  - N16: vs dqn p=0.0000

  - N16: vs ppo p=0.0000

  - N16: vs dqn_gru p=0.0000

  - N16: vs ppo_stack p=0.0000

  - N16: vs pomcp p=0.0000

  - N24: vs sequential p=0.0000

  - N24: vs random p=0.0000

  - N24: vs belief_greedy p=0.0000

  - N24: vs dqn p=0.0000

  - N24: vs ppo p=0.0000

  - N24: vs dqn_gru p=0.0259

  - N24: vs ppo_stack p=0.0000

  - N24: vs pomcp p=0.0000

  - N32: vs sequential p=0.0000

  - N32: vs random p=0.0000

  - N32: vs belief_greedy p=0.0000

  - N32: vs dqn p=0.0000

  - N32: vs ppo p=0.0000

  - N32: vs dqn_gru p=0.0000

  - N32: vs ppo_stack p=0.0000

  - N32: vs pomcp p=0.0000

  - N48: vs sequential p=0.0000

  - N48: vs random p=0.0000

  - N48: vs belief_greedy p=0.0000

  - N48: vs dqn p=0.0000

  - N48: vs ppo p=0.0000

  - N48: vs dqn_gru p=0.0000

  - N48: vs ppo_stack p=0.0000

  - N48: vs pomcp p=0.0000

  - N64: vs sequential p=0.0000

  - N64: vs random p=0.0000

  - N64: vs belief_greedy p=0.0000

  - N64: vs dqn p=0.0003

  - N64: vs ppo p=0.0000

  - N64: vs dqn_gru p=0.0014

  - N64: vs ppo_stack p=0.0058

  - N64: vs pomcp p=0.0000

  - N8: vs sequential p=0.0000

  - N8: vs random p=0.0000

  - N8: vs belief_greedy p=0.0000

  - N8: vs dqn p=0.0000

  - N8: vs ppo p=0.0000

  - N8: vs dqn_gru p=0.0000

  - N8: vs ppo_stack p=0.0000

  - N8: vs pomcp p=0.0000

  - pfa_0.01: vs sequential p=0.0000

  - pfa_0.01: vs belief_greedy p=0.0000

  - pfa_0.01: vs dqn p=0.0000

  - pfa_0.01: vs ppo p=0.0000

  - pfa_0.01: vs dqn_gru p=0.0000

  - pfa_0.01: vs ppo_stack p=0.0000

  - pfa_0.01: vs pomcp p=0.0000

  - pfa_0.03: vs sequential p=0.0000

  - pfa_0.03: vs belief_greedy p=0.0000

  - pfa_0.03: vs dqn p=0.0000

  - pfa_0.03: vs ppo p=0.0000

  - pfa_0.03: vs dqn_gru p=0.0000

  - pfa_0.03: vs ppo_stack p=0.0000

  - pfa_0.03: vs pomcp p=0.0000

  - pfa_0.05: vs sequential p=0.0000

  - pfa_0.05: vs belief_greedy p=0.0000

  - pfa_0.05: vs dqn p=0.0000

  - pfa_0.05: vs ppo p=0.0000

  - pfa_0.05: vs dqn_gru p=0.0000

  - pfa_0.05: vs ppo_stack p=0.0000

  - pfa_0.05: vs pomcp p=0.0000

  - pfa_0.08: vs sequential p=0.0000

  - pfa_0.08: vs belief_greedy p=0.0000

  - pfa_0.08: vs dqn p=0.0000

  - pfa_0.08: vs ppo p=0.0000

  - pfa_0.08: vs dqn_gru p=0.0000

  - pfa_0.08: vs ppo_stack p=0.0000

  - pfa_0.08: vs pomcp p=0.0000

  - pfa_0.10: vs sequential p=0.0000

  - pfa_0.10: vs belief_greedy p=0.0000

  - pfa_0.10: vs dqn p=0.0000

  - pfa_0.10: vs ppo p=0.0000

  - pfa_0.10: vs dqn_gru p=0.0000

  - pfa_0.10: vs ppo_stack p=0.0000

  - pfa_0.10: vs pomcp p=0.0000

  - pfa_0.12: vs sequential p=0.0000

  - pfa_0.12: vs belief_greedy p=0.0000

  - pfa_0.12: vs dqn p=0.0000

  - pfa_0.12: vs ppo p=0.0000

  - pfa_0.12: vs dqn_gru p=0.0000

  - pfa_0.12: vs ppo_stack p=0.0000

  - pfa_0.12: vs pomcp p=0.0000

  - pfa_0.15: vs sequential p=0.0000

  - pfa_0.15: vs belief_greedy p=0.0000

  - pfa_0.15: vs dqn p=0.0000

  - pfa_0.15: vs ppo p=0.0000

  - pfa_0.15: vs dqn_gru p=0.0000

  - pfa_0.15: vs ppo_stack p=0.0000

  - pfa_0.15: vs pomcp p=0.0000

  - snr_+0dB: vs sequential p=0.0000

  - snr_+0dB: vs random p=0.0116

  - snr_+0dB: vs belief_greedy p=0.0000

  - snr_+0dB: vs dqn p=0.0000

  - snr_+0dB: vs ppo p=0.0000

  - snr_+0dB: vs dqn_gru p=0.0000

  - snr_+0dB: vs ppo_stack p=0.0000

  - snr_+0dB: vs pomcp p=0.0000

  - snr_+10dB: vs sequential p=0.0000

  - snr_+10dB: vs belief_greedy p=0.0000

  - snr_+10dB: vs dqn p=0.0000

  - snr_+10dB: vs ppo p=0.0000

  - snr_+10dB: vs dqn_gru p=0.0000

  - snr_+10dB: vs ppo_stack p=0.0000

  - snr_+10dB: vs pomcp p=0.0000

  - snr_+3dB: vs sequential p=0.0000

  - snr_+3dB: vs belief_greedy p=0.0000

  - snr_+3dB: vs dqn p=0.0000

  - snr_+3dB: vs ppo p=0.0000

  - snr_+3dB: vs dqn_gru p=0.0000

  - snr_+3dB: vs ppo_stack p=0.0000

  - snr_+3dB: vs pomcp p=0.0000

  - snr_+6dB: vs sequential p=0.0000

  - snr_+6dB: vs belief_greedy p=0.0000

  - snr_+6dB: vs dqn p=0.0000

  - snr_+6dB: vs ppo p=0.0000

  - snr_+6dB: vs dqn_gru p=0.0000

  - snr_+6dB: vs ppo_stack p=0.0000

  - snr_+6dB: vs pomcp p=0.0000

  - snr_-10dB: vs sequential p=0.0000

  - snr_-10dB: vs belief_greedy p=0.0000

  - snr_-10dB: vs dqn p=0.0000

  - snr_-10dB: vs ppo p=0.0000

  - snr_-10dB: vs dqn_gru p=0.0000

  - snr_-10dB: vs ppo_stack p=0.0000

  - snr_-10dB: vs pomcp p=0.0000

  - snr_-3dB: vs sequential p=0.0000

  - snr_-3dB: vs belief_greedy p=0.0000

  - snr_-3dB: vs dqn p=0.0000

  - snr_-3dB: vs ppo p=0.0000

  - snr_-3dB: vs dqn_gru p=0.0000

  - snr_-3dB: vs ppo_stack p=0.0000

  - snr_-3dB: vs pomcp p=0.0000

  - snr_-6dB: vs sequential p=0.0000

  - snr_-6dB: vs belief_greedy p=0.0000

  - snr_-6dB: vs dqn p=0.0000

  - snr_-6dB: vs ppo p=0.0000

  - snr_-6dB: vs dqn_gru p=0.0000

  - snr_-6dB: vs ppo_stack p=0.0000

  - snr_-6dB: vs pomcp p=0.0000


## 15. 效应量多大？
见effect_sizes.csv；上列显著项的效应量：

  - vs sequential: cohens_dz=-1.109

  - vs random: cohens_dz=-0.681

  - vs belief_greedy: cohens_dz=-1.095

  - vs dqn: cohens_dz=0.857

  - vs ppo: cohens_dz=1.079

  - vs dqn_gru: cohens_dz=-0.404

  - vs ppo_stack: cohens_dz=0.802

  - vs pomcp: cohens_dz=-1.087

  - vs sequential: cohens_dz=-0.315

  - vs random: cohens_dz=-0.301

  - vs belief_greedy: cohens_dz=1.398

  - vs dqn: cohens_dz=0.594

  - vs ppo: cohens_dz=0.592

  - vs dqn_gru: cohens_dz=0.745

  - vs ppo_stack: cohens_dz=0.582

  - vs pomcp: cohens_dz=0.393

  - vs sequential: cohens_dz=-0.602

  - vs belief_greedy: cohens_dz=-0.882

  - vs dqn: cohens_dz=1.045

  - vs ppo: cohens_dz=1.074

  - vs dqn_gru: cohens_dz=0.587

  - vs ppo_stack: cohens_dz=0.929

  - vs pomcp: cohens_dz=-0.803

  - vs sequential: cohens_dz=-0.547

  - vs belief_greedy: cohens_dz=-0.984

  - vs dqn: cohens_dz=1.045

  - vs ppo: cohens_dz=0.923

  - vs dqn_gru: cohens_dz=0.521

  - vs ppo_stack: cohens_dz=0.831

  - vs pomcp: cohens_dz=-0.868

  - vs sequential: cohens_dz=-0.350

  - vs random: cohens_dz=-0.313

  - vs belief_greedy: cohens_dz=-2.402

  - vs dqn: cohens_dz=0.368

  - vs ppo: cohens_dz=0.478

  - vs dqn_gru: cohens_dz=0.351

  - vs ppo_stack: cohens_dz=0.416

  - vs pomcp: cohens_dz=-2.007

  - vs sequential: cohens_dz=-0.602

  - vs belief_greedy: cohens_dz=-0.900

  - vs dqn: cohens_dz=0.958

  - vs ppo: cohens_dz=1.139

  - vs dqn_gru: cohens_dz=0.503

  - vs ppo_stack: cohens_dz=0.868

  - vs pomcp: cohens_dz=-0.741

  - vs sequential: cohens_dz=-0.472

  - vs belief_greedy: cohens_dz=2.690

  - vs dqn: cohens_dz=1.039

  - vs ppo: cohens_dz=1.026

  - vs dqn_gru: cohens_dz=0.606

  - vs ppo_stack: cohens_dz=0.928

  - vs pomcp: cohens_dz=1.167

  - vs sequential: cohens_dz=-0.557

  - vs belief_greedy: cohens_dz=-0.876

  - vs dqn: cohens_dz=1.006

  - vs ppo: cohens_dz=1.016

  - vs dqn_gru: cohens_dz=0.511

  - vs ppo_stack: cohens_dz=0.847

  - vs pomcp: cohens_dz=-0.786

  - vs sequential: cohens_dz=-0.782

  - vs belief_greedy: cohens_dz=-1.011

  - vs dqn: cohens_dz=1.223

  - vs ppo: cohens_dz=1.184

  - vs dqn_gru: cohens_dz=0.669

  - vs ppo_stack: cohens_dz=1.044

  - vs pomcp: cohens_dz=-0.903

  - vs sequential: cohens_dz=-0.498

  - vs random: cohens_dz=0.087

  - vs belief_greedy: cohens_dz=2.516

  - vs dqn: cohens_dz=1.074

  - vs ppo: cohens_dz=1.119

  - vs dqn_gru: cohens_dz=0.602

  - vs ppo_stack: cohens_dz=0.971

  - vs pomcp: cohens_dz=1.125

  - vs sequential: cohens_dz=-0.498

  - vs random: cohens_dz=0.087

  - vs belief_greedy: cohens_dz=0.480

  - vs dqn: cohens_dz=1.074

  - vs ppo: cohens_dz=1.119

  - vs dqn_gru: cohens_dz=0.602

  - vs ppo_stack: cohens_dz=0.971

  - vs pomcp: cohens_dz=0.361

  - vs sequential: cohens_dz=-0.498

  - vs random: cohens_dz=0.087

  - vs belief_greedy: cohens_dz=-0.893

  - vs dqn: cohens_dz=1.074

  - vs ppo: cohens_dz=1.119

  - vs dqn_gru: cohens_dz=0.602

  - vs ppo_stack: cohens_dz=0.971

  - vs pomcp: cohens_dz=-0.763

  - vs sequential: cohens_dz=-0.498

  - vs random: cohens_dz=0.087

  - vs belief_greedy: cohens_dz=-0.394

  - vs dqn: cohens_dz=1.074

  - vs ppo: cohens_dz=1.119

  - vs dqn_gru: cohens_dz=0.602

  - vs ppo_stack: cohens_dz=0.971

  - vs pomcp: cohens_dz=-0.274

  - vs sequential: cohens_dz=-0.549

  - vs belief_greedy: cohens_dz=-0.997

  - vs dqn: cohens_dz=1.051

  - vs ppo: cohens_dz=1.081

  - vs dqn_gru: cohens_dz=0.524

  - vs ppo_stack: cohens_dz=0.925

  - vs pomcp: cohens_dz=-0.864

  - vs sequential: cohens_dz=-0.790

  - vs random: cohens_dz=-0.580

  - vs belief_greedy: cohens_dz=-1.802

  - vs dqn: cohens_dz=0.422

  - vs ppo: cohens_dz=0.269

  - vs dqn_gru: cohens_dz=0.100

  - vs ppo_stack: cohens_dz=0.199

  - vs pomcp: cohens_dz=-1.713

  - vs sequential: cohens_dz=-0.441

  - vs random: cohens_dz=-0.347

  - vs belief_greedy: cohens_dz=-2.897

  - vs dqn: cohens_dz=0.647

  - vs ppo: cohens_dz=0.793

  - vs dqn_gru: cohens_dz=0.833

  - vs ppo_stack: cohens_dz=0.656

  - vs pomcp: cohens_dz=-2.387

  - vs sequential: cohens_dz=-0.388

  - vs random: cohens_dz=-0.294

  - vs belief_greedy: cohens_dz=-2.350

  - vs dqn: cohens_dz=0.437

  - vs ppo: cohens_dz=0.387

  - vs dqn_gru: cohens_dz=0.400

  - vs ppo_stack: cohens_dz=0.376

  - vs pomcp: cohens_dz=-1.874

  - vs sequential: cohens_dz=-0.521

  - vs random: cohens_dz=-0.563

  - vs belief_greedy: cohens_dz=-3.918

  - vs dqn: cohens_dz=0.176

  - vs ppo: cohens_dz=0.279

  - vs dqn_gru: cohens_dz=0.153

  - vs ppo_stack: cohens_dz=0.124

  - vs pomcp: cohens_dz=-2.362

  - vs sequential: cohens_dz=-0.945

  - vs random: cohens_dz=-0.674

  - vs belief_greedy: cohens_dz=-1.068

  - vs dqn: cohens_dz=0.680

  - vs ppo: cohens_dz=1.096

  - vs dqn_gru: cohens_dz=-0.312

  - vs ppo_stack: cohens_dz=0.836

  - vs pomcp: cohens_dz=-0.974

  - vs sequential: cohens_dz=-0.476

  - vs belief_greedy: cohens_dz=-0.987

  - vs dqn: cohens_dz=1.027

  - vs ppo: cohens_dz=1.017

  - vs dqn_gru: cohens_dz=0.593

  - vs ppo_stack: cohens_dz=0.991

  - vs pomcp: cohens_dz=-0.920

  - vs sequential: cohens_dz=-0.522

  - vs belief_greedy: cohens_dz=-0.860

  - vs dqn: cohens_dz=1.033

  - vs ppo: cohens_dz=1.028

  - vs dqn_gru: cohens_dz=0.553

  - vs ppo_stack: cohens_dz=0.878

  - vs pomcp: cohens_dz=-0.846

  - vs sequential: cohens_dz=-0.444

  - vs belief_greedy: cohens_dz=-0.971

  - vs dqn: cohens_dz=1.053

  - vs ppo: cohens_dz=1.098

  - vs dqn_gru: cohens_dz=0.548

  - vs ppo_stack: cohens_dz=0.997

  - vs pomcp: cohens_dz=-0.867

  - vs sequential: cohens_dz=-0.423

  - vs belief_greedy: cohens_dz=-0.837

  - vs dqn: cohens_dz=0.950

  - vs ppo: cohens_dz=0.962

  - vs dqn_gru: cohens_dz=0.502

  - vs ppo_stack: cohens_dz=0.801

  - vs pomcp: cohens_dz=-0.785

  - vs sequential: cohens_dz=-0.590

  - vs belief_greedy: cohens_dz=-0.906

  - vs dqn: cohens_dz=0.988

  - vs ppo: cohens_dz=0.977

  - vs dqn_gru: cohens_dz=0.535

  - vs ppo_stack: cohens_dz=0.821

  - vs pomcp: cohens_dz=-0.786

  - vs sequential: cohens_dz=-0.571

  - vs belief_greedy: cohens_dz=-1.002

  - vs dqn: cohens_dz=0.996

  - vs ppo: cohens_dz=1.125

  - vs dqn_gru: cohens_dz=0.541

  - vs ppo_stack: cohens_dz=0.896

  - vs pomcp: cohens_dz=-0.826

  - vs sequential: cohens_dz=-0.594

  - vs belief_greedy: cohens_dz=-0.893

  - vs dqn: cohens_dz=1.118

  - vs ppo: cohens_dz=1.148

  - vs dqn_gru: cohens_dz=0.584

  - vs ppo_stack: cohens_dz=0.889

  - vs pomcp: cohens_dz=-0.703

  - vs sequential: cohens_dz=-0.510

  - vs random: cohens_dz=0.113

  - vs belief_greedy: cohens_dz=-0.839

  - vs dqn: cohens_dz=1.119

  - vs ppo: cohens_dz=1.064

  - vs dqn_gru: cohens_dz=0.628

  - vs ppo_stack: cohens_dz=0.964

  - vs pomcp: cohens_dz=-0.760

  - vs sequential: cohens_dz=-0.969

  - vs belief_greedy: cohens_dz=-1.058

  - vs dqn: cohens_dz=1.202

  - vs ppo: cohens_dz=1.318

  - vs dqn_gru: cohens_dz=0.635

  - vs ppo_stack: cohens_dz=1.168

  - vs pomcp: cohens_dz=-0.994

  - vs sequential: cohens_dz=-0.663

  - vs belief_greedy: cohens_dz=-0.977

  - vs dqn: cohens_dz=1.170

  - vs ppo: cohens_dz=1.217

  - vs dqn_gru: cohens_dz=0.579

  - vs ppo_stack: cohens_dz=1.068

  - vs pomcp: cohens_dz=-0.865

  - vs sequential: cohens_dz=-0.675

  - vs belief_greedy: cohens_dz=-1.061

  - vs dqn: cohens_dz=1.209

  - vs ppo: cohens_dz=1.112

  - vs dqn_gru: cohens_dz=0.632

  - vs ppo_stack: cohens_dz=1.018

  - vs pomcp: cohens_dz=-0.930

  - vs sequential: cohens_dz=-0.191

  - vs belief_greedy: cohens_dz=-0.763

  - vs dqn: cohens_dz=0.601

  - vs ppo: cohens_dz=0.580

  - vs dqn_gru: cohens_dz=0.304

  - vs ppo_stack: cohens_dz=0.467

  - vs pomcp: cohens_dz=-0.543

  - vs sequential: cohens_dz=-0.524

  - vs belief_greedy: cohens_dz=-0.786

  - vs dqn: cohens_dz=0.903

  - vs ppo: cohens_dz=0.944

  - vs dqn_gru: cohens_dz=0.493

  - vs ppo_stack: cohens_dz=0.829

  - vs pomcp: cohens_dz=-0.733

  - vs sequential: cohens_dz=-0.376

  - vs belief_greedy: cohens_dz=-0.805

  - vs dqn: cohens_dz=0.821

  - vs ppo: cohens_dz=0.756

  - vs dqn_gru: cohens_dz=0.433

  - vs ppo_stack: cohens_dz=0.597

  - vs pomcp: cohens_dz=-0.693


## 16. MuZero付出了多少额外计算成本？

- DQN: 训练572s/种子，决策时延0.0ms/步

- PPO: 训练399s/种子，决策时延0.0ms/步

- MuZero (ours): 训练11114s/种子，决策时延6.7ms/步


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
4. MuZero与基线的差异经Holm校正后244项统计显著（共264项检验）。


## 20. 下一步建议补什么实验？

- 若主实验领先不显著：扩大种子数（≥10）、增加训练预算、或按开题报告风险4预案转向『规划有效条件与收益边界』分析。
- 波形参数联合优化（开题报告扩展方向）。
- 多目标/连续方位扩展。
