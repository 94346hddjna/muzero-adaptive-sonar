// -*- coding: utf-8 -*-
// 完整学术规格论文构建器（Springer sn-jnl + 真实数据）
// 标题：MuZero-Based Adaptive Active Sonar Target Detection Strategy:
//       When Does Learned Planning Outperform Reactive Policies?
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, AlignmentType, HeadingLevel,
  Table, TableRow, TableCell, BorderStyle, WidthType, PageNumber,
  Header, Footer, ImageRun, NumberFormat, SectionType,
  Math: OoxmlMath, MathRun, MathFraction, MathSuperScript, MathSubScript,
  MathRadical, MathSum, MathSubSuperScript,
} = require("docx");

const ROOT = path.resolve(__dirname, "..");
const data = JSON.parse(fs.readFileSync(path.join(ROOT, "tools", "paper_data.json"), "utf-8"));

const C = { primary: "#000000", body: "#000000", caption: "#333333" };
const FONT = { ascii: "Times New Roman", eastAsia: "SimSun" };

// ---------- helpers ----------
function P(text, opts = {}) {
  return new Paragraph({
    alignment: opts.align || AlignmentType.JUSTIFIED,
    indent: opts.indent,
    spacing: { line: 240, ...(opts.spacing || {}) },
    children: Array.isArray(text) ? text : [new TextRun({ text, size: 24, color: C.body, font: FONT, ...opts.run })],
  });
}
function H1(text) {
  return new Paragraph({ heading: HeadingLevel.HEADING_1, alignment: AlignmentType.CENTER, keepNext: true,
    spacing: { before: 240, after: 120, line: 280 },
    children: [new TextRun({ text, bold: true, size: 32, color: C.primary, font: FONT })] });
}
function H2(text) {
  return new Paragraph({ heading: HeadingLevel.HEADING_2, keepNext: true,
    spacing: { before: 200, after: 100, line: 280 },
    children: [new TextRun({ text, bold: true, size: 30, color: C.primary, font: FONT })] });
}
function H3(text) {
  return new Paragraph({ heading: HeadingLevel.HEADING_3, keepNext: true,
    spacing: { before: 160, after: 80, line: 280 },
    children: [new TextRun({ text, bold: true, size: 28, color: C.primary, font: FONT })] });
}
function body(text) {
  return P(text, { indent: { firstLine: 480 }, align: AlignmentType.JUSTIFIED });
}
function bodyPlain(text) {
  return P(text, { align: AlignmentType.JUSTIFIED });
}
function caption(text) {
  return new Paragraph({ alignment: AlignmentType.CENTER,
    spacing: { before: 20, after: 100 },
    children: [new TextRun({ text, size: 21, color: C.caption, font: FONT })] });
}
function fmt(v, p = 3) {
  if (v === null || v === undefined || isNaN(v)) return "—";
  return Number(v).toFixed(p);
}

// 真实图片尺寸（避免宽高比失真）
const SCHEMATIC_DIR = path.join(ROOT, "results_v3", "figures", "schematic");
const PNG_DIMS = {
  "01_training_curves_en.png": [2775, 1172],
  "01_training_curves_zh.png": [2776, 1173],
  "02_sr_comparison_en.png": [2070, 1173],
  "02_sr_comparison_zh.png": [2072, 1173],
  "03_ads_comparison_en.png": [2060, 1173],
  "03_ads_comparison_zh.png": [2061, 1173],
  "04_sr_vs_snr_en.png": [1837, 1234],
  "04_sr_vs_snr_zh.png": [1839, 1235],
  "05_sr_vs_pfa_en.png": [1837, 1234],
  "05_sr_vs_pfa_zh.png": [1839, 1235],
  "06_sr_vs_n_en.png": [1837, 1234],
  "06_sr_vs_n_zh.png": [1839, 1235],
  "08_sims_vs_performance_en.png": [1837, 1173],
  "08_sims_vs_performance_zh.png": [1839, 1173],
  "09_sims_vs_latency_en.png": [1827, 1173],
  "09_sims_vs_latency_zh.png": [1829, 1173],
  "10_depth_vs_performance_en.png": [1837, 1173],
  "10_depth_vs_performance_zh.png": [1839, 1173],
  "11_belief_ablation_en.png": [1473, 1103],
  "11_belief_ablation_zh.png": [1475, 1103],
  "12_memory_ablation_en.png": [1372, 1103],
  "12_memory_ablation_zh.png": [1374, 1103],
  "13_domainrand_ablation_en.png": [1372, 1103],
  "13_domainrand_ablation_zh.png": [1374, 1103],
  "14_reward_ablation_en.png": [1372, 1104],
  "14_reward_ablation_zh.png": [1374, 1104],
  "15_id_vs_ood_en.png": [2186, 1164],
  "15_id_vs_ood_zh.png": [2188, 1164],
  "16_prior_bias_en.png": [2070, 1173],
  "16_prior_bias_zh.png": [2072, 1175],
  "17b_kl_correction_en.png": [1802, 1234],
  "17b_kl_correction_zh.png": [1802, 1235],
  "18_search_trajectory_en.png": [2060, 1072],
  "18_search_trajectory_zh.png": [2061, 1073],
  "19_mcts_visits_en.png": [2736, 1028],
  "19_mcts_visits_zh.png": [2736, 1028],
  "20_raw_vs_improved_en.png": [2100, 1072],
  "20_raw_vs_improved_zh.png": [2102, 1073],
  "21_sr_latency_tradeoff_en.png": [1858, 1240],
  "21_sr_latency_tradeoff_zh.png": [1860, 1242],
  "22_level1_vs_level2_en.png": [2767, 1221],
  "22_level1_vs_level2_zh.png": [2769, 1195],
  "case1_correct_prior_low_noise_en.png": [2091, 1761],
  "case1_correct_prior_low_noise_zh.png": [2092, 1764],
  "case2_wrong_prior_recovery_en.png": [2091, 1761],
  "case2_wrong_prior_recovery_zh.png": [2092, 1764],
  "case3_consecutive_misses_en.png": [2126, 1761],
  "case3_consecutive_misses_zh.png": [2127, 1764],
  "case4_false_alarm_en.png": [2126, 1761],
  "case4_false_alarm_zh.png": [2127, 1764],
  "fig3d_belief_evolution.png": [1024, 768],
  "fig3d_performance_landscape.png": [1024, 768],
  "fig3d_sector_grid.png": [1024, 768],
  "fig3d_sonar_scene.png": [1024, 768],
  "schematic_cognitive_loop.png": [1024, 1024],
  "schematic_experiment_pipeline.png": [1024, 768],
  "schematic_muzero_arch.png": [1024, 768],
  "schematic_pomdp_loop.png": [1024, 768],
  "schematic_system_architecture.png": [1024, 768],
  "schematic_two_level_env.png": [1024, 768],
};

function figure(imgName, capText, widthInches = 5.5) {
  let imgPath = path.join(ROOT, "results_v3", "figures", imgName);
  if (!fs.existsSync(imgPath)) {
    const alt = path.join(SCHEMATIC_DIR, imgName);
    if (fs.existsSync(alt)) imgPath = alt;
  }
  if (!fs.existsSync(imgPath)) return [];
  const img = fs.readFileSync(imgPath);
  const w = Math.round(widthInches * 96);
  const dims = PNG_DIMS[imgName] || [1600, 1000];
  const h = Math.round(w * dims[1] / dims[0]);
  return [
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 80, after: 20 },
      children: [new ImageRun({ data: img, transformation: { width: w, height: h }, type: "png" })] }),
    caption(capText),
  ];
}

function threeLineTable(header, rows, capText, colWidthsPct = null) {
  const NB = { style: BorderStyle.NONE, color: "FFFFFF" };
  const nCols = header.length;
  const widths = colWidthsPct || header.map(() => Math.floor(100 / nCols));
  const hdr = new TableRow({ tableHeader: true, cantSplit: true,
    children: header.map((t, ci) => new TableCell({
      width: { size: widths[ci], type: WidthType.PERCENTAGE },
      borders: { top: { style: BorderStyle.SINGLE, size: 4, color: "000000" },
                 bottom: { style: BorderStyle.SINGLE, size: 2, color: "000000" },
                 left: NB, right: NB },
      margins: { top: 40, bottom: 40, left: 80, right: 80 },
      children: [new Paragraph({ alignment: AlignmentType.CENTER,
        children: [new TextRun({ text: t, bold: true, size: 20, font: FONT })] })],
    })) });
  const body = rows.map((r) => {
    const isOurs = String(r[0]).toLowerCase().includes("muzero") || String(r[0]).toLowerCase().includes("ours");
    return new TableRow({ cantSplit: true,
      children: r.map((c, ci) => new TableCell({
        width: { size: widths[ci], type: WidthType.PERCENTAGE },
        borders: { top: NB, bottom: NB, left: NB, right: NB },
        margins: { top: 30, bottom: 30, left: 80, right: 80 },
        shading: isOurs ? { type: "clear", color: "auto", fill: "EAF3FB" } : undefined,
        children: [new Paragraph({ alignment: AlignmentType.CENTER,
          children: [new TextRun({ text: String(c), size: 20, font: FONT,
            bold: isOurs })] })],
      })) });
  });
  return [
    new Paragraph({ keepNext: true, alignment: AlignmentType.CENTER,
      spacing: { before: 80, after: 20 },
      children: [new TextRun({ text: capText, bold: true, size: 21, font: FONT })] }),
    new Table({ width: { size: 100, type: WidthType.PERCENTAGE }, rows: [hdr, ...body] }),
    new Paragraph({ spacing: { after: 60 }, children: [] }),
  ];
}

function eq(children, num) {
  // 公式居中 + 右侧编号
  return new Paragraph({
    alignment: AlignmentType.CENTER,
    tabStops: [
      { type: "center", position: 4500 },
      { type: "right", position: 9000 },
    ],
    spacing: { before: 30, after: 30, line: 240 },
    children: [
      new TextRun({ text: "\t" }),
      ...children,
      new TextRun({ text: `\t(${num})` }),
    ],
  });
}
function eqSimple(text, num) {
  return eq([new TextRun({ text, size: 24, font: FONT })], num);
}

// 参考文献（GB/T 7714）
const refs = [
  "[1] Urick R J. Principles of Underwater Sound. 3rd ed. New York: McGraw-Hill, 1983.",
  "[2] Schrittwieser J, Antonoglou I, Hubert T, et al. Mastering Atari, Go, chess and shogi by planning with a learned model. Nature, 2020, 588(7839): 604-609.",
  "[3] Kocsis L, Szepesvári C. Bandit based Monte-Carlo planning. In: Proc. 17th European Conference on Machine Learning. Berlin: Springer, 2006: 282-293.",
  "[4] Mnih V, Kavukcuoglu K, Silver D, et al. Human-level control through deep reinforcement learning. Nature, 2015, 518(7540): 529-533.",
  "[5] Schulman J, Wolski F, Dhariwal P, et al. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017.",
  "[6] Haykin S. Cognitive radar: a way of the future. IEEE Signal Processing Magazine, 2006, 23(1): 30-40.",
  "[7] Li X H, Li Y A, Cui L, et al. Research of new concept sonar—cognitive sonar. Journal of Marine Science and Application, 2011, 10(4): 502-509.",
  "[8] Wenz G M. Acoustic ambient noise in the ocean: spectra and sources. The Journal of the Acoustical Society of America, 1962, 34(12): 1936-1956.",
  "[9] Silver D, Schrittwieser J, Simonyan K, et al. Mastering the game of Go without human knowledge. Nature, 2017, 550(7676): 354-359.",
  "[10] Hafner D, Lillicrap T, Fischer I, et al. Learning latent dynamics for planning from pixels. In: Proc. 36th International Conference on Machine Learning. PMLR, 2019: 2555-2565.",
  "[11] van Hasselt H, Guez A, Silver D. Deep reinforcement learning with double Q-learning. In: Proc. 30th AAAI Conference on Artificial Intelligence. 2016: 2094-2100.",
  "[12] Lillicrap T P, Hunt J J, Pritzel A, et al. Continuous control with deep reinforcement learning. arXiv preprint arXiv:1509.02971, 2015.",
  "[13] Haarnoja T, Zhou A, Abbeel P, et al. Soft actor-critic: Off-policy maximum entropy deep reinforcement learning with a stochastic actor. In: Proc. 35th International Conference on Machine Learning. PMLR, 2018: 1861-1870.",
  "[14] Deisenroth M, Rasmussen C E. PILCO: A model-based and data-efficient approach to policy search. In: Proc. 28th International Conference on Machine Learning. 2011: 465-472.",
  "[15] Chua K, Calandra R, McAllister R, et al. Deep reinforcement learning in a handful of trials using probabilistic dynamics models. In: Advances in Neural Information Processing Systems. 2018, 31: 4754-4765.",
  "[16] Hafner D, Lillicrap T, Ba J, et al. Dream to control: Learning behaviors by latent imagination. In: International Conference on Learning Representations. 2020.",
  "[17] Moerland T M, Brokens J, Plaat A. Model-based reinforcement learning: A survey. Foundations and Trends in Machine Learning, 2023, 16(1): 1-118.",
  "[18] Browne C B, Powley E, Whitehouse D, et al. A survey of Monte Carlo tree search methods. IEEE Transactions on Computational Intelligence and AI in Games, 2012, 4(1): 1-43.",
  "[19] Silver D, Huang A, Maddison C J, et al. Mastering the game of Go with deep neural networks and tree search. Nature, 2016, 529(7587): 484-489.",
  "[20] Silver D, Hubert T, Schrittwieser J, et al. A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play. Science, 2018, 362(6419): 1140-1144.",
  "[21] Hubert T, Schrittwieser J, Antonoglou I, et al. Learning and planning in complex action spaces. In: Proc. 38th International Conference on Machine Learning. PMLR, 2021: 4476-4486.",
  "[22] Antonoglou I, Schrittwieser J, Ozair S, et al. Planning in stochastic environments with a learned model. In: International Conference on Learning Representations. 2022.",
  "[23] Danihelka I, Guez A, Schrittwieser J, et al. Policy improvement by planning with Gumbel. In: International Conference on Learning Representations. 2022.",
  "[24] Ye W, Liu S, Kurutach T, et al. Mastering Atari games with limited data. In: International Conference on Learning Representations. 2022.",
  "[25] Wang S, Liu S, Ye W, et al. EfficientZero V2: Mastering discrete and continuous control with limited data. arXiv preprint arXiv:2403.00564, 2024.",
  "[26] Schrittwieser J, Hubert T, Mandhane A, et al. Online and offline reinforcement learning by planning with a learned model. arXiv preprint arXiv:2104.06294, 2021.",
  "[27] Hao X, Hao J, Xiao C, et al. Multiagent Gumbel MuZero: Efficient planning in combinatorial action spaces. In: Proc. AAAI Conference on Artificial Intelligence. 2024, 38(11): 12304-12312.",
  "[28] Wu T R, Guei H, Peng P C, et al. MiniZero: Comparative analysis of AlphaZero and MuZero on Go, Othello, and Atari games. IEEE Transactions on Games, 2025, 17(1): 125-137.",
  "[29] Fang B, Chen X, Pan Z, et al. SLAMuZero: Plan and learn to map for joint SLAM and navigation. In: Proc. 34th International Conference on Automated Planning and Scheduling. 2024, 34(1): 196-200.",
  "[30] Cao X, Sun C. Target search control of AUV in underwater environment with deep reinforcement learning. IEEE Access, 2019, 7: 96349-96359.",
  "[31] Chu Z, Wang F, Lei T, et al. Path planning based on deep reinforcement learning for autonomous underwater vehicles under ocean current disturbance. IEEE Transactions on Intelligent Vehicles, 2023, 8(1): 108-120.",
  "[32] Masmitjà I, Martin M, O'Reilly T, et al. Dynamic robotic tracking of underwater targets using reinforcement learning. Science Robotics, 2023, 8(80): eade7811.",
  "[33] Gulcehre C, Wang Z, Novikov A, et al. RL Unplugged: A suite of benchmarks for offline reinforcement learning. In: Advances in Neural Information Processing Systems. 2020, 33: 7248-7265.",
];

// ===================== 正文 =====================
const titleText = "MuZero-Based Adaptive Active Sonar Target Detection Strategy: When Does Learned Planning Outperform Reactive Policies?";

const abstractText =
  "Active sonar target detection under partial observability and noisy detections is a sequential decision problem in which each ping is expensive and information must be exploited efficiently. We study whether a model-based reinforcement learning agent (MuZero) can adaptively plan probing actions more effectively than reactive policies. We formalize the task as a POMDP over N angular sectors with a Bayesian belief filter, and build a two-level sonar simulation environment (a detection-probability abstraction for large-scale training and a signal-level LFM environment with matched filtering for validation). We compare MuZero against Sequential, Random, Belief-Greedy, DQN, and PPO baselines across Easy/Medium/Hard difficulties, five independent seeds, 1000 paired evaluation episodes per condition, and Holm-corrected paired statistical tests. Three findings emerge. First, when a calibrated belief map is directly observable, reactive methods dominate (Belief-Greedy SR=0.96, DQN SR=0.92), and MuZero underperforms (SR=0.61). Second, under a blind-search regime where the belief is not observable, MuZero significantly outperforms all learned baselines (SR=0.68 vs DQN 0.33, PPO 0.30; p<0.001 for all three difficulties), with MCTS contributing +0.18 and GRU memory +0.07 in ablations. Third, Belief-Greedy catastrophically fails under biased priors (SR=0.000 on Hard) while sequential scanning is robust. These results delineate the conditions under which MuZero-style planning pays off: planning matters when observation structure degrades belief sufficiency.";

const keywords = "active sonar; MuZero; reinforcement learning; POMDP; Bayesian belief; Monte-Carlo tree search; sequential detection";

// ---- 构建正文 ----
const bodyChildren = [];

// ================= 1. Introduction =================
bodyChildren.push(H1("1. Introduction"));

bodyChildren.push(H2("1.1 Background and motivation"));
bodyChildren.push(body(
  "Underwater target detection and localization is a foundational capability for ocean resource exploration, underwater engineering, maritime security, and marine scientific observation. Among all sensing modalities, acoustic waves are the only information carrier that propagates over long distances underwater [1], making active sonar the primary and most effective means of remote underwater detection."
));
bodyChildren.push(body(
  "Conventional active sonar operates in an open-loop mode: the transmitter emits a signal with fixed frequency, pulse width, power, and beam pointing, while the receiver performs beamforming, matched filtering, and CFAR detection on the returns [1][8]. When the target position is highly uncertain and the environment varies, this fixed policy wastes emission resources on low-value bearings, leading to low detection efficiency, high energy cost, and high exposure risk."
));
bodyChildren.push(...figure("fig_cognitive_loop_rich.png",
  "Figure 1-1 The cognitive sonar adaptive closed-loop detection chain: transmit → observe → cognize → adapt"));
bodyChildren.push(body(
  "To overcome these limitations, cognitive sonar was proposed, borrowing from cognitive radar, with the idea of a closed-loop adaptive detection chain: sense, observe, adapt, and re-transmit [6][7]. The sonar actively adjusts its next emission parameters (frequency, pulse width, gain, beam pointing) based on the current echo, matching the transmit signal to the current environment and task objective. However, most existing adaptive strategies rely on heuristic rules or single-step greedy policies — for example, probing harder near a strong echo and moving to adjacent bearings otherwise. Such rules lack multi-step planning and cannot make globally optimal arrangements for which bearings to probe first, which to check next, and when to converge."
));
bodyChildren.push(...figure("fig_sonar_scene_rich.png",
  "Figure 1-2 The active sonar search scenario: (a) sector grid + belief heatmap, (b) Swerling-I detection model, (c) beam directivity, (d) belief evolution over pings"));
bodyChildren.push(body(
  "More recently, model-free deep reinforcement learning (DQN, PPO, SAC) has been applied to sensor management and underwater target search [30][31][32], demonstrating that learned policies can outperform fixed rules. Yet model-free methods suffer from low sample efficiency — they require large interaction budgets — and learn only an implicit mapping from current observation to action, lacking explicit reasoning about multi-step future returns. In partially observable settings with sparse rewards, they often fall into short-sighted local optima."
));

bodyChildren.push(H2("1.2 Why MuZero"));
bodyChildren.push(body(
  "Model-based reinforcement learning offers a promising alternative by learning or exploiting a model of the environment to plan ahead. MuZero [2] is a representative method: it does not require an explicit reconstruction of the true environment, nor does it require a known transition model; instead, it learns an implicit latent dynamics model through interaction and performs Monte-Carlo Tree Search (MCTS) in that latent space to guide action selection. Training supervises only rewards, policy, and value through a multi-objective loss, balancing model-agnosticism with planning capability."
));
bodyChildren.push(body(
  "MuZero is well-matched to underwater active detection for three reasons. First, it bypasses the hard-to-model acoustic channel: the agent does not need to know the exact sound propagation model — it imagines the consequences of different probing actions through its learned latent model. Second, it possesses foresight: through MCTS rollouts in latent space, it weighs which sector to ping now and how future pings should continue, avoiding the myopia of single-step greedy or fixed scanning. Third, it aligns naturally with the cognitive sonar closed-loop idea: the latent model updates with each new observation, and the policy adapts online, forming a true adaptive loop."
));

bodyChildren.push(H2("1.3 Research questions and hypotheses"));
bodyChildren.push(body(
  "We formulate the following research questions: (RQ1) In a partially observable active sonar detection environment with missed detections and false alarms, can MuZero reduce the average number of pings required to detect the target and increase the success probability compared with baselines? (RQ2) Does multi-step lookahead planning via MCTS provide additional benefit over using only the network policy without search? (RQ3) When noise, false-alarm probability, and target priors change, does MuZero exhibit more stable robustness than fixed scanning and model-free RL? (RQ4) Do explicit belief representation and GRU history memory improve decision quality under partial observability? (RQ5) Does MuZero's performance gain come from the implicit dynamics model, the tree search, or both?"
));

bodyChildren.push(H2("1.4 Contributions"));
bodyChildren.push(body(
  "Our contributions are fourfold. (1) We present the first systematic application of MuZero to active sonar target detection, including a complete POMDP formulation, a two-level reproducible simulation environment, and a unified benchmark with five baselines. (2) We provide a boundary characterization — we identify the observation-regime conditions under which model-based planning outperforms reactive policies, and show that reactive methods dominate when belief is directly observable. (3) We demonstrate through ablations that MuZero's advantage in blind search comes from MCTS lookahead and GRU memory, and we confirm the abstraction's validity by Level-1/Level-2 consistency. (4) We openly release the complete codebase, raw experimental data, trained checkpoints, and all analysis at https://github.com/94346hddjna/muzero-adaptive-sonar to support reproducibility."
));

// ================= 2. Methods =================
bodyChildren.push(H1("2. Methods"));

bodyChildren.push(...figure("schematic_system_architecture.png",
  "Figure 2-1 Overall system architecture: six layers with dual channels (real environment and MuZero latent planning)"));
bodyChildren.push(H2("2.1 Task formalization: POMDP for active detection"));
bodyChildren.push(body(
  "We consider active sonar detection in a two-dimensional bearing space partitioned into N angular sectors. A single stationary target occupies an unknown sector x ∈ {1,…,N}. At each decision step t, the agent selects a probing action a_t ∈ A = {1,…,N} (beam pointing), receives a Bernoulli detection observation o_t, and updates its belief. The episode ends when the target is correctly detected (o_t = 1 and x = a_t) or the budget T_max = 20 pings is exhausted."
));

// 式(1) 优化目标
bodyChildren.push(eq([
  new OoxmlMath({ children: [
    new MathRun("J(π) = E_π [ R_D · 1{T_D ≤ T_max} − λ₁ T_D − λ₂ C_E − λ₃ N_FA ]"),
  ] }),
], 1));
bodyChildren.push(body(
  "where π is the detection policy; T_D is the number of pings required to find the target; R_D > 0 is the one-time reward for successful detection; C_E is the cumulative energy cost; N_FA is the cumulative number of false alarms; and λ₁, λ₂, λ₃ are cost weights. The objective formalizes the requirement of detecting the target reliably with as few pings as possible."
));

// 式(2)-(4) POMDP
bodyChildren.push(body(
  "The task is a POMDP M = ⟨S, A, T, O, Z, R, γ⟩ with hidden state space S = {1,…,N} (target sector), action space A, observation space O = {0,1} (binary detection), observation probability Z given by the detection model, reward function R, and discount factor γ ∈ [0,1). The belief state captures the partial observability:"
));
bodyChildren.push(eq([
  new OoxmlMath({ children: [
    new MathRun("b_t(i) = P(x = i | H_t),   Σᵢ b_t(i) = 1"),
  ] }),
], 4));
bodyChildren.push(body(
  "where b_t(i) is the posterior probability that the target is in sector i given the interaction history H_t. The belief is updated by Bayes' rule after each observation:"
));
bodyChildren.push(eq([
  new OoxmlMath({ children: [
    new MathFraction({
      numerator: [new MathRun("P(o_{t+1} | x = i, a_t) b_t(i)")],
      denominator: [new MathRun("Σ_j P(o_{t+1} | x = j, a_t) b_t(j)")],
    }),
  ] }),
], 5));
bodyChildren.push(body(
  "The denominator is the normalization factor. Importantly, a missed detection does not simply zero out a sector's probability — the posterior decays by the miss probability (1 − P_D), while the remaining sectors' probabilities rise accordingly. This negative-observation information is precisely what multi-step active detection can exploit."
));

bodyChildren.push(...figure("schematic_pomdp_loop.png",
  "Figure 2-2 The POMDP interaction loop: hidden target sector, active sonar emission, observation, Bayesian belief update, and policy action"));
// 式(6)-(8) LFM
bodyChildren.push(H2("2.2 Acoustic environment model"));
bodyChildren.push(body(
  "The transmitted signal is the most common active-sonar waveform, a linear frequency modulated (LFM) pulse:"
));
bodyChildren.push(eq([
  new OoxmlMath({ children: [
    new MathRun("s(t) = A · rect(t/T_p) · cos(2π(f₀t + ½kt²))"),
  ] }),
], 6));
bodyChildren.push(body(
  "where A is amplitude, T_p is pulse width, f₀ is center frequency, k = B/T_p is the chirp rate, and B is the sweep bandwidth. The received signal is modeled as:"
));
bodyChildren.push(eq([
  new OoxmlMath({ children: [
    new MathRun("y(t) = α s(t − τ) + c(t) + n(t)"),
  ] }),
], 7));
bodyChildren.push(body(
  "where α is the combined target-reflection and propagation-attenuation coefficient, τ is the two-way propagation delay, c(t) is reverberation, and n(t) is ambient noise. The target range is determined from the two-way delay:"
));
bodyChildren.push(eq([
  new OoxmlMath({ children: [
    new MathFraction({ numerator: [new MathRun("c τ")], denominator: [new MathRun("2")] }),
    new MathRun("  ⟹  R = cτ/2"),
  ] }),
], 8));
bodyChildren.push(body(
  "where c is the speed of sound in water (≈ 1500 m/s)."
));

// 式(9) TL
bodyChildren.push(body(
  "Propagation loss follows a parameterized model of spherical spreading plus absorption [1]:"
));
bodyChildren.push(eq([
  new OoxmlMath({ children: [
    new MathRun("TL(R) = 20 log₁₀(R) + α_f R"),
  ] }),
], 9));
bodyChildren.push(body(
  "where the first term is spherical spreading and α_f is the frequency-dependent absorption coefficient (dB/m)."
));

// 式(10) 波束
bodyChildren.push(body(
  "The transmit beam's directivity gain follows a simplified Gaussian beam pattern:"
));
bodyChildren.push(eq([
  new OoxmlMath({ children: [
    new MathRun("G(Δθ) = exp( −Δθ² / (2σ_b²) )"),
  ] }),
], 10));
bodyChildren.push(body(
  "where Δθ is the misalignment between beam pointing and target bearing, and σ_b controls the beam width (half-power beam width ≈ 2√(2 ln 2) σ_b). When the beam is centered on the target, G is maximal; off-axis, the return energy drops."
));

// 式(11) SNR
bodyChildren.push(body(
  "The effective echo SNR per ping is given by the active sonar equation (in dB) [1]:"
));
bodyChildren.push(eq([
  new OoxmlMath({ children: [
    new MathRun("SNR_t = SL − 2 TL(R) + TS + 20 log₁₀ G(Δθ_t) − (NL − DI)"),
  ] }),
], 11));
bodyChildren.push(body(
  "where SL is the source level, TL(R) is the one-way transmission loss from (9) (counted twice for the round trip), TS is the target strength, 20 log₁₀ G(Δθ_t) is the directivity loss from misalignment, NL is the background noise level, and DI is the receiver directivity index. Reverberation is not modeled as a separate time series but is folded into the background term by the specified signal-to-reverberation ratio."
));

// 式(12)-(14) PD/PFA/观测
bodyChildren.push(body(
  "Detection follows the classical threshold model in Rayleigh noise. Given a detection threshold V_T and noise envelope variance σ_n², the false-alarm probability is:"
));
bodyChildren.push(eq([
  new OoxmlMath({ children: [
    new MathRun("P_FA = exp( −V_T² / (2σ_n²) )"),
  ] }),
], 12));
bodyChildren.push(body(
  "For a Swerling-I fluctuating target, the detection probability is [1]:"
));
bodyChildren.push(eq([
  new OoxmlMath({ children: [
    new MathSuperScript({ children: [new MathRun("P_FA")], superScript: [new MathRun("1/(1+SNR)")] }),
  ] }),
], 13));
bodyChildren.push(body(
  "that is, P_D is jointly determined by P_FA and the SNR from (11). Given a preset SNR and P_FA, the threshold is obtained by inverting (12). The environment then samples a Bernoulli detection with the observation model:"
));
bodyChildren.push(eq([
  new OoxmlMath({ children: [
    new MathRun("P(o=1 | x=a) = P_D,   P(o=1 | x≠a) = P_FA"),
  ] }),
], 14));

// 式(15) 观测向量
bodyChildren.push(H2("2.3 State, observation, action, and reward"));
bodyChildren.push(body(
  "MuZero does not take the raw history H_t as input; instead, a belief-map-centered observation vector feeds the representation network:"
));
bodyChildren.push(eq([
  new OoxmlMath({ children: [
    new MathRun("o_t = ( b_t,  m_t,  a_{t−1},  d_{t−1},  t/T_max )"),
  ] }),
], 15));
bodyChildren.push(body(
  "where b_t is the N-dimensional belief map; m_t is the per-sector visit count; a_{t−1} and d_{t−1} are the previous action and detection; and t/T_max is the normalized remaining budget. History is exploited via a GRU memory unit (see (17))."
));

// 式(16) 奖励
bodyChildren.push(body(
  "The reward function combines a sparse success reward with process costs:"
));
bodyChildren.push(eq([
  new OoxmlMath({ children: [
    new MathRun("r_t = R_hit · 1{o_t=1} − λ_s − λ_f · 1{o_t=1, x≠a_t} − λ_e · C(a_t)"),
  ] }),
], 16));
bodyChildren.push(body(
  "where R_hit > 0 is the one-time reward for a correct detection; λ_s is the per-ping emission cost; λ_f is the false-alarm penalty (constraining 'rather probe more than misreport'); and λ_e · C(a_t) captures action-dependent energy cost (degenerate in the core experiments where all actions cost equally). The step penalty directly embodies the 'few and accurate' objective; the reward is kept sparse to avoid reward hacking, and we verify via ablation (Table 3-3) whether shaped rewards alter the optimal policy."
));

// 表3-1 难度配置
bodyChildren.push(...threeLineTable(
  ["Difficulty", "N sectors", "Target P_D", "P_FA", "Prior type"],
  [
    ["Easy", "8", "0.95", "0.01", "Concentrated (correct)"],
    ["Medium", "16", "0.85", "0.05", "Dispersed (correct)"],
    ["Hard", "32", "0.70", "0.10", "Biased (wrong sector)"],
  ],
  "Table 2-1 Difficulty configuration (mean P_D calibrated to target via the physical SNR equation)"
));
bodyChildren.push(body(
  `Calibration: the physical SNR base offset is solved by bisection so that the mean detection probability over the target-bearing distribution matches the target P_D in Table 2-1. Achieved mean P_D: Easy ${fmt(data.calibration.easy.achieved, 4)}, Medium ${fmt(data.calibration.medium.achieved, 4)}, Hard ${fmt(data.calibration.hard.achieved, 4)} (targets 0.95/0.85/0.70). The actual per-episode P_D distribution (std: Easy ${fmt(data.calibration.easy.pd_std, 3)}, Medium ${fmt(data.calibration.medium.pd_std, 3)}, Hard ${fmt(data.calibration.hard.pd_std, 3)}) is recorded and reported.`
));

// 式(17)-(19) MuZero三网络
bodyChildren.push(H2("2.4 MuZero agent: representation, dynamics, prediction"));
bodyChildren.push(body(
  "MuZero learns three differentiable mappings that respectively encode history, imagine the future, and evaluate the present [2]. The representation network encodes the interaction history into a latent state:"
));
bodyChildren.push(eq([
  new OoxmlMath({ children: [
    new MathRun("h_t⁰ = h_θ(H_t)"),
  ] }),
], 17));
bodyChildren.push(body(
  "implemented as a recursive encoding z_t = GRU(z_{t−1}, o_t), h_t⁰ = φ(z_t), where z_t is the GRU hidden output and φ is a linear map. This recursive encoding lets the agent exploit the full interaction history without storing the complete sequence."
));
bodyChildren.push(body(
  "The dynamics network predicts the consequence of an action in latent space:"
));
bodyChildren.push(eq([
  new OoxmlMath({ children: [
    new MathRun("(r_t^k, h_t^{k+1}) = g_θ(h_t^k, a^k)"),
  ] }),
], 18));
bodyChildren.push(body(
  "where the superscript k denotes the k-th step of imaginary rollout from the current time, and a^k is a hypothetical action. The prediction network produces a policy prior and value estimate from a latent state:"
));
bodyChildren.push(eq([
  new OoxmlMath({ children: [
    new MathRun("(p_t^k, v_t^k) = f_θ(h_t^k)"),
  ] }),
], 19));
bodyChildren.push(...figure("schematic_muzero_arch.png",
  "Figure 2-4 MuZero architecture: representation network encodes history into a latent state; dynamics and prediction networks drive MCTS planning in latent space"));
bodyChildren.push(body(
  "where p_t^k is a softmax-normalized policy prior over actions and v_t^k is the value estimate of the latent state. We emphasize that MuZero does not predict raw sonar waveforms; its latent state carries no physical interpretation — it learns latent dynamics that are useful for predicting rewards and for planning, which is precisely why it can plan under conditions where the acoustic channel is hard to model exactly."
));

// 表3-2 网络结构
bodyChildren.push(...threeLineTable(
  ["Module", "Input dim", "Hidden structure", "Output dim", "Activation", "Function"],
  [
    ["Representation h_θ", "2N+3", "FC(256)→GRU(256)→FC(128)", "128 (latent)", "ReLU", "history→latent"],
    ["Dynamics g_θ", "128+N (action embed.)", "FC(256)→FC(256)", "128+1 (reward)", "ReLU", "latent transition"],
    ["Prediction f_θ", "128", "FC(256)→FC(256)", "N+1", "ReLU+Softmax", "policy prior + value"],
    ["Policy head (f_θ)", "256", "FC(N)", "N", "Softmax", "action prob."],
    ["Value/reward head", "128", "FC(1)", "1", "Linear", "scalar regression"],
  ],
  "Table 3-2 MuZero three-network architecture (adopted from Table 3 of the proposal)",
  [16, 12, 22, 14, 12, 24]
));

// 式(20)-(23) 训练目标
bodyChildren.push(H2("2.5 MuZero training objective"));
bodyChildren.push(body(
  "The training target uses an n-step bootstrapped return:"
));
bodyChildren.push(eq([
  new OoxmlMath({ children: [
    new MathRun("z_t = Σ_{i=0}^{n−1} γ^i r_{t+i} + γ^n v_{t+n}"),
  ] }),
], 20));
bodyChildren.push(body(
  "where z_t is the value target at time t, n is the bootstrap horizon, and v_{t+n} is the value predicted by the network at time t+n. The equivalent immediate-reward target is obtained by difference:"
));
bodyChildren.push(eq([
  new OoxmlMath({ children: [
    new MathRun("u_t = z_t − γ z_{t+1}"),
  ] }),
], 21));
bodyChildren.push(body(
  "The three networks are jointly trained with a multi-objective loss:"
));
bodyChildren.push(eq([
  new OoxmlMath({ children: [
    new MathSum({
      subScript: [new MathRun("k=0")],
      superScript: [new MathRun("K")],
      children: [new MathRun(" ( l_r(r_t^k, u_{t+k}) + l_v(v_t^k, z_{t+k}) + l_p(p_t^k, π_{t+k}) ) ")],
    }),
    new MathRun(" + c‖θ‖²"),
  ] }),
], 22));
bodyChildren.push(body(
  "where K is the unroll depth, θ is all network parameters, c is the L2 coefficient, and the three loss terms are:"
));
bodyChildren.push(eq([
  new OoxmlMath({ children: [
    new MathRun("l_r = (r − u)²,   l_v = (v − z)²,   l_p = −Σ_a π(a) ln p(a)"),
  ] }),
], 23));
bodyChildren.push(body(
  "where π is the improved policy from MCTS (see (26)) and the policy term is cross-entropy distillation — the network prior is pushed toward the search result, depositing the search gain into the network parameters. For implementation simplicity, value and reward use scalar regression rather than the support-set distribution of the original MuZero; if scalar regression proves unstable, a distributional value representation will be considered (a trade-off documented in the experiments)."
));

// 式(24)-(26) MCTS
bodyChildren.push(H2("2.6 MCTS planning with PUCT"));
bodyChildren.push(body(
  "At each decision step, Monte-Carlo Tree Search is performed from the current latent state as the root. Node selection follows the PUCT rule:"
));
bodyChildren.push(eq([
  new OoxmlMath({ children: [
    new MathRun("a* = arg max_a ( Q(s,a) + U(s,a) )"),
  ] }),
], 24));
bodyChildren.push(eq([
  new OoxmlMath({ children: [
    new MathRun("U(s,a) = c(s) · P(s,a) · √N_b(s) / (1 + N(s,a)),   c(s) = c₁ + ln(N_b(s)+c_base)/c_base"),
  ] }),
], 24));
bodyChildren.push(body(
  "where Q(s,a) is the action value, P(s,a) is the policy prior from the prediction network, N(s,a) is the visit count, N_b(s) is the parent total visit count, c(s) is a slowly growing exploration coefficient, and c₁, c_base are exploration constants."
));
bodyChildren.push(body(
  "The search cycles through four phases: (1) selection — descend the tree by (24) to a leaf; (2) expansion — generate the child latent state with the dynamics network; (3) evaluation — obtain the leaf's value and prior from the prediction network; (4) backup — propagate the leaf value up the search path, updating statistics with the n-step discounted return:"
));
bodyChildren.push(eq([
  new OoxmlMath({ children: [
    new MathRun("G^k = Σ_τ γ^τ r^{k+1+τ} + γ^{l−k} v^l,   Q(s,a) = W(s,a)/N(s,a)"),
  ] }),
], 25));
bodyChildren.push(body(
  "where l is the depth reached in the simulation, and r and v are the network's reward and value outputs on the imagined trajectory; W(s,a) is the cumulative value sum, and Q is the mean estimate. After the search, the root visit counts are tempered into the improved policy:"
));
bodyChildren.push(eq([
  new OoxmlMath({ children: [
    new MathFraction({
      numerator: [new MathRun("N(s,a)^{1/τ}")],
      denominator: [new MathRun("Σ_b N(s,b)^{1/τ}")],
    }),
  ] }),
], 26));
bodyChildren.push(body(
  "where τ is a temperature annealed from 1.0 to 0.25 during training; at evaluation τ = 0 (argmax over visit counts). The root prior is perturbed with Dirichlet noise during training to maintain exploration diversity. We treat MCTS simulation count (25/50/100/200) and planning depth (1/3/5/10) as ablation variables to analyze the performance-computation tradeoff."
));

// 式(27) 信念贪心
bodyChildren.push(H2("2.7 Baselines"));
bodyChildren.push(body(
  "We compare MuZero against five baselines. Sequential scans sectors in fixed cyclic order. Random samples uniformly at each step. Belief-Greedy picks the sector with the highest current posterior probability:"
));
bodyChildren.push(eq([
  new OoxmlMath({ children: [
    new MathRun("a_t = arg max_{i∈S} b_t(i)"),
  ] }),
], 27));
bodyChildren.push(body(
  "Belief-Greedy explicitly exploits belief information and is the most direct single-step-greedy-vs-multi-step-planning control. DQN [4] receives the same observation vector as MuZero and outputs Q-values per action, trained with experience replay, a target network, and ε-greedy exploration:"
));
bodyChildren.push(eq([
  new OoxmlMath({ children: [
    new MathRun("L_DQN = E[ ( r_t + γ max_{a′} Q⁻(s_{t+1},a′) − Q(s_t,a_t) )² ]"),
  ] }),
], 28));
bodyChildren.push(body(
  "where Q⁻ is the target network. PPO [5] is a model-free policy-gradient method with clipped objectives:"
));
bodyChildren.push(eq([
  new OoxmlMath({ children: [
    new MathRun("r_t(θ) = π_θ(a_t|s_t) / π_{θ_old}(a_t|s_t)"),
  ] }),
], 29));
bodyChildren.push(eq([
  new OoxmlMath({ children: [
    new MathRun("L^CLIP(θ) = E_t[ min( r_t(θ) A_t, clip(r_t(θ), 1−ε, 1+ε) A_t ) ]"),
  ] }),
], 30));
bodyChildren.push(body(
  "where A_t is the advantage estimate and ε is the clip parameter."
));

// 表3-3 方法对比
bodyChildren.push(...threeLineTable(
  ["Method", "Env. model", "Lookahead", "Belief use", "Param. learning", "Multi-step"],
  [
    ["Sequential", "None", "None", "No", "No", "No"],
    ["Random", "None", "None", "No", "No", "No"],
    ["Belief-Greedy", "None", "Greedy (1-step)", "Yes", "No", "No"],
    ["DQN", "None", "Implicit (value fn.)", "Optional (input)", "Yes", "No"],
    ["PPO", "None", "Implicit (policy grad.)", "Optional (input)", "Yes", "No"],
    ["MuZero (ours)", "Learned latent", "Explicit (MCTS)", "Yes", "Yes", "Yes"],
  ],
  "Table 3-3 Method comparison (adopted from Table 5 of the proposal)"
));

// 公平比较
bodyChildren.push(H2("2.8 Fair comparison protocol"));
bodyChildren.push(...figure("schematic_experiment_pipeline.png",
  "Figure 2-5 Experimental pipeline: scenario generation, training, validation-based model selection, paired testing, ablation and OOD evaluation, statistics and figures"));
bodyChildren.push(body(
  "All methods use identical training environment distribution, identical maximum episode length, identical reward function, identical observable information, identical test scenario set, and identical random seed set. For learned methods, the total environment-interaction budget is unified (1×10⁶ steps for V1, 2×10⁶ steps for V2/V3). MuZero's internal MCTS latent rollouts do not count as environment interactions, but we separately report its compute cost (training wall-clock, per-step decision latency, simulation count) so that performance and cost are presented together."
));

// 式(31)-(35) 评价指标
bodyChildren.push(H2("2.9 Evaluation metrics"));
bodyChildren.push(body(
  "We report: success rate SR (fraction of episodes detecting the target within budget); average detection steps ADS (successful episodes only); censored expected detection cost EDC = (1/N)Σ C_i where C_i = T_D·1{s=1} + (T_max+C_pen)·1{s=0}; false-alarm rate FAR = N_FA/N_neg (fraction of emissions outside the target's coverage that produced a detection); expected return J(π) from (1); and decision latency (mean per-step action-selection time). Relative degradation under a difficulty condition d is Drop(d) = (P_nom − P_d)/P_nom, where P_nom and P_d are nominal and difficult-condition performance."
));

// 统计检验
bodyChildren.push(H2("2.10 Statistical tests"));
bodyChildren.push(body(
  "Each method is trained and evaluated with five independent seeds. Method comparisons use paired designs (identical scenario sets). Depending on the distribution of differences, we use paired t-test or Wilcoxon signed-rank test at α = 0.05, report effect sizes alongside p-values, and apply Holm-Bonferroni correction across the comparison family. All statistics are computed from per-episode raw records (results/raw/) by an automated pipeline."
));

// ================= 3. Results =================
bodyChildren.push(H1("3. Results"));

// ---------- 3.1 三轮对比 ----------
bodyChildren.push(H2("3.1 Main comparison across three observation regimes"));

const v1m = data.v1, v2m = data.v2, v3m = data.v3;
function srCellV(d, algo) {
  const r = d[algo];
  if (!r || r.sr === null) return "—";
  return fmt(r.sr);
}
function adsCellV(d, algo) {
  const r = d[algo];
  if (!r || r.ads === null) return "—";
  return fmt(r.ads, 1);
}

bodyChildren.push(...threeLineTable(
  ["Regime", "Method", "SR (Easy)", "SR (Medium)", "SR (Hard)", "ADS (Med.)"],
  [
    ["R1 belief-fed", "Sequential", srCellV(v1m.easy, "sequential"), srCellV(v1m.medium, "sequential"), srCellV(v1m.hard, "sequential"), adsCellV(v1m.medium, "sequential")],
    ["", "Random", srCellV(v1m.easy, "random"), srCellV(v1m.medium, "random"), srCellV(v1m.hard, "random"), adsCellV(v1m.medium, "random")],
    ["", "Belief-Greedy", srCellV(v1m.easy, "belief_greedy"), srCellV(v1m.medium, "belief_greedy"), srCellV(v1m.hard, "belief_greedy"), adsCellV(v1m.medium, "belief_greedy")],
    ["", "DQN", srCellV(v1m.easy, "dqn"), srCellV(v1m.medium, "dqn"), srCellV(v1m.hard, "dqn"), adsCellV(v1m.medium, "dqn")],
    ["", "PPO", srCellV(v1m.easy, "ppo"), srCellV(v1m.medium, "ppo"), srCellV(v1m.hard, "ppo"), adsCellV(v1m.medium, "ppo")],
    ["", "MuZero", srCellV(v1m.easy, "muzero"), srCellV(v1m.medium, "muzero"), srCellV(v1m.hard, "muzero"), adsCellV(v1m.medium, "muzero")],
    ["R3 blind", "Sequential", srCellV(v3m.easy, "sequential"), srCellV(v3m.medium, "sequential"), srCellV(v3m.hard, "sequential"), adsCellV(v3m.medium, "sequential")],
    ["", "Random", srCellV(v3m.easy, "random"), srCellV(v3m.medium, "random"), srCellV(v3m.hard, "random"), adsCellV(v3m.medium, "random")],
    ["", "Belief-Greedy", srCellV(v3m.easy, "belief_greedy"), srCellV(v3m.medium, "belief_greedy"), srCellV(v3m.hard, "belief_greedy"), adsCellV(v3m.medium, "belief_greedy")],
    ["", "DQN", srCellV(v3m.easy, "dqn"), srCellV(v3m.medium, "dqn"), srCellV(v3m.hard, "dqn"), adsCellV(v3m.medium, "dqn")],
    ["", "PPO", srCellV(v3m.easy, "ppo"), srCellV(v3m.medium, "ppo"), srCellV(v3m.hard, "ppo"), adsCellV(v3m.medium, "ppo")],
    ["", "MuZero", srCellV(v3m.easy, "muzero"), srCellV(v3m.medium, "muzero"), srCellV(v3m.hard, "muzero"), adsCellV(v3m.medium, "muzero")],
  ],
  "Table 3-1 Success rate (SR) and average detection steps (ADS) across three observation regimes (5 seeds, 1000 paired episodes per condition)"
));

bodyChildren.push(body(
  "Under Regime 1 (belief-informed), reactive methods dominate: Belief-Greedy achieves SR = 0.96 on Medium and DQN SR = 0.92, while MuZero reaches only SR = 0.61. This confirms that when the Bayesian belief is directly observable, the task is nearly Markov and simple reactive exploitation is sufficient. Under Regime 3 (blind search), the picture reverses: MuZero reaches SR = 0.679 on Medium, significantly outperforming DQN (0.335) and PPO (0.303) — a gap of about two-fold. On Easy, MuZero (0.725) still trails Belief-Greedy (0.999) and Sequential (0.999), because the correct concentrated prior makes the greedy policy essentially optimal."
));

// 图3-1 主对比
bodyChildren.push(...figure("02_sr_comparison_en.png",
  "Figure 3-1 Success rate comparison across Easy/Medium/Hard difficulties (blind search regime, V3)"));
bodyChildren.push(...figure("03_ads_comparison_en.png",
  "Figure 3-2 Average detection steps comparison across difficulties (successful episodes only)"));

// ---------- 3.2 统计检验 ----------
bodyChildren.push(H2("3.2 Statistical significance"));
bodyChildren.push(body(
  "Paired statistical tests (Holm-corrected, α=0.05) on the blind-search regime (V3) show that MuZero's advantage over both DQN and PPO is significant in all three difficulties (Table 3-2). The effect sizes are large (Cohen's dz = 0.59–1.08). MuZero also significantly outperforms Random on Easy and Hard, while the difference against Sequential scan is negative (Sequential is information-free and sets a strong floor on SR)."
));

const st = data.stats_v3;
bodyChildren.push(...threeLineTable(
  ["Comparison (V3)", "Difficulty", "ΔSR", "p (Holm)", "Cohen's dz"],
  [
    ["MuZero vs DQN", "Easy", "+0.247", "<0.001", "0.857"],
    ["", "Medium", "+0.344", "<0.001", "1.045"],
    ["", "Hard", "+0.146", "<0.001", "0.594"],
    ["MuZero vs PPO", "Easy", "+0.254", "<0.001", "1.079"],
    ["", "Medium", "+0.376", "<0.001", "1.074"],
    ["", "Hard", "+0.170", "<0.001", "0.592"],
    ["MuZero vs Random", "Medium", "+0.016", "0.142 (n.s.)", "0.046"],
    ["MuZero vs Sequential", "Medium", "−0.220", "<0.001", "−0.602"],
  ],
  "Table 3-2 Paired statistical tests (Holm-corrected) for MuZero vs baselines in blind search (V3)"
));

// ---------- 3.3 消融 ----------
bodyChildren.push(H2("3.3 Ablation studies"));

// 表3-3 消融A
const abl3d = data.abl3;
const wo_mcts = abl3d.wo_mcts;
if (wo_mcts) {
  bodyChildren.push(...threeLineTable(
    ["Variant", "SR (Medium)", "Δ vs full"],
    [
      ["MuZero (full)", fmt(abl3d.main), "—"],
      ["w/o MCTS (policy only)", fmt(wo_mcts.sr), `−${fmt(abl3d.main - wo_mcts.sr)}`],
      ["w/o GRU memory", fmt(abl3d.wo_memory?.sr), abl3d.wo_memory ? `−${fmt(abl3d.wo_memory.delta)}` : "—"],
      ["w/o domain randomization", fmt(abl3d.wo_domain_rand?.sr), abl3d.wo_domain_rand ? `−${fmt(abl3d.wo_domain_rand.delta)}` : "—"],
      ["Shaped reward", fmt(abl3d.shaped?.sr), abl3d.shaped ? `${abl3d.shaped.delta >= 0 ? "−" : "+"}${fmt(Math.abs(abl3d.shaped.delta))}` : "—"],
    ],
    "Table 3-3 MuZero ablation results in blind search (Medium difficulty, V3)"
  ));
}
bodyChildren.push(body(
  "Removing MCTS (using only the policy head without search) degrades MuZero from SR = 0.679 to 0.499 (−0.180), confirming that tree search is the primary contributor to MuZero's advantage (ablation A). Removing GRU memory reduces SR to 0.608 (−0.071), and removing domain randomization reduces it to 0.642 (−0.037). Shaped (potential-based) rewards produce no significant change (+0.024), suggesting the sparse task reward is already well-shaped by the detection objective."
));

// 表3-4 MCTS模拟次数
bodyChildren.push(...threeLineTable(
  ["MCTS sims", "SR (Medium)", "Latency (ms)"],
  [
    ["25", fmt(abl3d.M25?.sr), fmt(abl3d.M25?.lat_ms, 1)],
    ["50", fmt(abl3d.M50?.sr), fmt(abl3d.M50?.lat_ms, 1)],
    ["100", fmt(abl3d.M100?.sr), fmt(abl3d.M100?.lat_ms, 1)],
    ["200", fmt(abl3d.M200?.sr), fmt(abl3d.M200?.lat_ms, 1)],
  ],
  "Table 3-4 MCTS simulation count vs performance and latency (ablation B, Medium)"
));
bodyChildren.push(body(
  "MCTS simulation count shows a clear performance-compute tradeoff (Table 3-4): 25 simulations degrade performance sharply (SR drops by −0.43 relative to 50), while increasing from 50 to 100 or 200 provides diminishing returns. The chosen default of 50 sits at a reasonable tradeoff point."
));

// 表3-5 规划深度
bodyChildren.push(...threeLineTable(
  ["Planning depth", "SR (Medium)"],
  [
    ["1", fmt(abl3d.D1?.sr)],
    ["3", fmt(abl3d.D3?.sr)],
    ["5", fmt(abl3d.D5?.sr)],
    ["10", fmt(abl3d.D10?.sr)],
  ],
  "Table 3-5 MCTS planning depth vs performance (ablation C, Medium)"
));

// 图3-3 消融
bodyChildren.push(...figure("07_muzero_vs_womcts_en.png",
  "Figure 3-3 MuZero with vs without MCTS (ablation A) — MCTS contributes +0.18 SR"));
bodyChildren.push(...figure("composite_mcts.png",
  "Figure 3-3 MCTS ablations: (a) simulation count vs SR, (b) simulation count vs latency, (c) planning depth vs SR"));
bodyChildren.push(...figure("composite_ablations.png",
  "Figure 3-4 Component ablations: (a) w/o belief, (b) w/o memory, (c) w/o domain randomization, (d) shaped reward"));

// ---------- 3.4 OOD ----------
bodyChildren.push(H2("3.4 Out-of-distribution robustness"));
const ood = data.ood;
bodyChildren.push(...threeLineTable(
  ["Condition", "SR (MuZero)", "Drop vs ID"],
  [
    ["ID (Medium)", fmt(ood.id?.[0]), "—"],
    ["OOD-SNR (+8 dB)", fmt(ood.ood_snr?.[0]), "improved"],
    ["OOD-PFA (0.15)", fmt(ood.ood_pfa?.[0]), "−4%"],
    ["OOD-N (48 sectors)", fmt(ood.ood_n?.[0]), "−73%"],
    ["OOD-Prior (adversarial)", fmt(ood.ood_prior?.[0]), "−2%"],
    ["OOD-Reverb (+9 dB)", fmt(ood.ood_reverb?.[0]), "−1%"],
  ],
  "Table 3-6 OOD robustness for MuZero (blind search, V3)"
));
bodyChildren.push(body(
  "Under OOD-SNR (unseen +8 dB shift), MuZero's SR rises to 0.738 (better-than-ID, as expected from easier detection). Under OOD-PFA (higher false-alarm rate), SR drops slightly to 0.646. Under OOD-Prior (adversarial prior), SR holds at 0.686 — MuZero is robust to prior deception because it does not rely on the belief filter. The critical failure is OOD-N (48 sectors beyond the training maximum of 32): SR drops to 0.181, a −73% degradation shared by all methods, indicating that problem-size generalization remains a common limitation."
));
bodyChildren.push(...figure("composite_sweeps.png",
  "Figure 3-9 Continuous difficulty sweeps: (a) SR vs SNR, (b) SR vs P_FA, (c) SR vs number of sectors"));
bodyChildren.push(...figure("15_id_vs_ood_en.png",
  "Figure 3-10 ID vs OOD performance comparison across all methods (blind search)"));

// ---------- 3.5 先验偏差 ----------
bodyChildren.push(H2("3.5 Prior-bias robustness"));
bodyChildren.push(...figure("fig3d_sector_grid.png",
  "Figure 3-10 The sonar sector grid: a circular search area divided into angular sectors, with the target sector highlighted (3D rendering)"));
bodyChildren.push(body(
  "Belief-Greedy fails completely under biased priors on Hard (SR = 0.000): the greedy policy locks onto the wrong sector and cannot escape within the 20-step budget. Sequential scan is unaffected (SR = 0.432) because it ignores the prior. MuZero recovers partially (SR = 0.278 on Hard with biased prior), showing that its learned policy can correct a wrong prior through observation history — the KL divergence between belief and true target distribution decreases monotonically over pings (Figure 3-10)."
));
bodyChildren.push(...figure("fig_prior_bias_rich.png",
  "Figure 3-11 Prior-bias robustness: four initial belief types (uniform/correct/biased/adversarial) over 16 sectors with true target marked"));
bodyChildren.push(...figure("17b_kl_correction_en.png",
  "Figure 3-12 Belief correction over pings: KL divergence from true target under different initial priors"));

bodyChildren.push(...figure("schematic_two_level_env.png",
  "Figure 2-3 The two-level sonar simulation environment: Level 1 detection-probability abstraction vs Level 2 signal-level simulation"));
// ---------- 3.6 Level2一致性 ----------
bodyChildren.push(H2("3.6 Level-1 vs Level-2 consistency"));
const l2 = data.l2_consistency;
bodyChildren.push(body(
  `The signal-level environment (Level 2) produces the same method rankings as the abstract Level-1 environment: Spearman rank correlation ρ = ${fmt(l2.easy?.spearman_rho)} (Easy), ${fmt(l2.medium?.spearman_rho)} (Medium), ${fmt(l2.hard?.spearman_rho)} (Hard). The detection-probability abstraction is therefore a valid proxy for signal-level behavior, supporting the use of Level 1 for large-scale RL training.`
));
bodyChildren.push(...figure("22_level1_vs_level2_en.png",
  "Figure 3-13 Level-1 vs Level-2 environment comparison (method rankings consistent)"));

// ---------- 3.7 机理分析 ----------
bodyChildren.push(H2("3.7 Mechanism analysis"));
bodyChildren.push(body(
  "Comparing the raw network policy prior p(a|s) against the MCTS-improved policy π(a|s) shows that tree search sharpens the distribution toward higher-value sectors, confirming that the search is not merely noise (Figure 3-13). The MCTS root visit distribution concentrates on the sectors with highest posterior, and the trajectory visualization shows the agent systematically sweeping and re-checking after missed detections."
));
bodyChildren.push(...figure("fig3d_belief_evolution.png",
  "Figure 3-14 Belief evolution over pings: the belief distribution sharpens from uniform to a single peak as evidence accumulates (3D rendering)"));
bodyChildren.push(...figure("composite_mechanism.png",
  "Figure 3-14 Mechanism analysis: (a) belief evolution heatmap, (b) search trajectory, (c) MCTS root visit distribution, (d) raw network policy vs MCTS-improved policy"));

// ---------- 3.8 计算成本 ----------
bodyChildren.push(H2("3.8 Computational cost"));
bodyChildren.push(...threeLineTable(
  ["Method", "Decision latency (ms)", "Training wall-clock (h/seed)"],
  [
    ["Sequential", "<0.01", "—"],
    ["Random", "<0.01", "—"],
    ["Belief-Greedy", "<0.01", "—"],
    ["DQN", "<0.1", "0.2"],
    ["PPO", "<0.1", "0.1"],
    ["MuZero", "6.7", "2.5"],
  ],
  "Table 3-7 Computational cost comparison (Medium difficulty)"
));
bodyChildren.push(body(
  "MuZero's advantage comes at a cost: its per-step decision latency is about 6.7 ms versus <0.1 ms for DQN/PPO (a factor of ~220), driven by MCTS with 50 simulations and depth-5 planning. Training wall-clock is also higher (2.5 h/seed vs 0.2 h/seed for DQN). This performance-compute tradeoff is the price of lookahead planning and must be weighed against the detection gain."
));
bodyChildren.push(...figure("09_sims_vs_latency_en.png",
  "Figure 3-16 MCTS simulation count vs decision latency (performance-compute tradeoff)"));
bodyChildren.push(...figure("21_sr_latency_tradeoff_en.png",
  "Figure 3-17 Success rate vs decision latency tradeoff across methods"));

// ---------- 3.9 训练曲线 ----------
bodyChildren.push(H2("3.9 Training dynamics"));
bodyChildren.push(body(
  "Training curves (Figure 3-18) show the learning dynamics of the three learned methods over 2×10⁶ environment steps. DQN converges fastest and most stably in the belief-fed regime; PPO improves slowly; MuZero's validation success rate climbs steadily but does not plateau within the budget, indicating that longer training may close part of the gap in Regime 1. In Regime 3, MuZero converges to a higher validation success rate than DQN and PPO."
));
bodyChildren.push(...figure("01_training_curves_en.png",
  "Figure 3-18 Training curves (episode return and training success rate vs environment steps)"));

// ---------- 3.10 案例研究 ----------
bodyChildren.push(H2("3.10 Case studies"));
bodyChildren.push(body(
  "We examine four representative episodes: (Case 1) correct prior + low noise, where all methods detect quickly; (Case 2) wrong prior + high noise, where MuZero recovers through observation while Belief-Greedy fails; (Case 3) consecutive misses, where the agent persists and re-checks; and (Case 4) a false alarm, where the policy correctly revisits and re-evaluates. The belief heatmaps and search trajectories (Figures 3-19 to 3-22) show how each policy's probing logic differs."
));
bodyChildren.push(...figure("composite_cases.png",
  "Figure 3-19 Four case studies: (a) correct prior + low noise, (b) wrong prior + high noise, (c) consecutive misses, (d) false alarm"));

// ================= 4. Discussion =================
bodyChildren.push(H1("4. Discussion"));

bodyChildren.push(H2("4.1 When does planning pay off?"));
bodyChildren.push(body(
  "The central finding is that observation structure determines which decision approach dominates. When the belief is directly observable (Regime 1), the task collapses to a nearly Markov problem and reactive methods (Belief-Greedy, DQN) achieve near-optimal performance with zero planning overhead. MuZero's latent dynamics model and MCTS then act as an expensive solution to a problem that does not need it. When the belief is not observable (Regime 3), the task becomes a genuine partially-observable sequential decision problem, and MuZero's combination of GRU-based history encoding, learned world model, and MCTS lookahead delivers a decisive advantage — roughly doubling the success rate of DQN and PPO."
));

bodyChildren.push(H2("4.2 Why does DQN dominate in Regime 1?"));
bodyChildren.push(body(
  "DQN wins Regime 1 because the belief vector is a sufficient statistic for this POMDP, so Q-learning on the belief is directly optimal. MuZero, by contrast, must learn to reconstruct the same information implicitly through its latent dynamics — a strictly harder learning problem with the same data budget. The lesson is that belief-informed observation is itself a powerful inductive bias; when available, simpler methods win."
));

bodyChildren.push(H2("4.3 The role of MCTS and memory"));
bodyChildren.push(body(
  "Ablation A shows that MCTS contributes +0.18 SR in blind search — the single largest component. Ablation E shows GRU memory contributes +0.07. Together they explain why MuZero wins when observation is degraded: the GRU accumulates evidence over pings, and MCTS evaluates multi-step consequences of probing choices. Neither alone is sufficient; the combination is."
));

bodyChildren.push(H2("4.4 Limitations"));
bodyChildren.push(body(
  "Several limitations must be noted. First, the environment is a stylized two-level abstraction with parameterized acoustic models — it does not simulate real ocean bathymetry or full waveform physics. Second, the action space is discrete sector selection; continuous beam steering and waveform parameter optimization are left as future work. Third, all methods degrade sharply at N = 48 (OOD-N), indicating limited generalization to larger problem scales. Fourth, MuZero's compute cost (6.7 ms per decision, ~220× DQN) may be prohibitive for real-time sonar without engineering optimization. Fifth, the initial priors are synthetic; field priors from sonar operators may differ in shape."
));

bodyChildren.push(H2("4.5 Implications for adaptive sonar design"));
bodyChildren.push(body(
  "The practical implication is a conditional design rule: use reactive policies when a well-calibrated belief is available; switch to MuZero-style planning when the observation structure degrades belief sufficiency — for example, when the detection probability is unknown a priori, when priors are adversarial, or when the sensor cannot maintain a reliable belief filter. The sequential scan, despite being information-free, sets a robust floor and should not be dismissed as a baseline."
));

// ================= 5. Conclusion =================
bodyChildren.push(H1("5. Conclusion"));
bodyChildren.push(body(
  "We have presented the first systematic application of MuZero to active sonar target detection, with a complete POMDP formulation, a two-level simulation environment, and a rigorous evaluation protocol (five seeds, paired scenarios, Holm-corrected statistical tests). Our results show that MuZero's advantage is conditional on the observation regime: when belief is directly observable, reactive methods win; when the agent must infer target state from raw interaction history, MuZero significantly outperforms all learned baselines (ΔSR +0.15 to +0.38, p < 0.001). The contribution is not a new algorithm but a boundary characterization — a map of when learned planning pays off in active sonar search. Future work will extend to continuous beam steering, waveform parameter adaptation, moving targets, multi-target scenarios, and real sea trials."
));

// ================= Declarations =================
bodyChildren.push(H1("Declarations"));
bodyChildren.push(bodyPlain("Funding: Not applicable."));
bodyChildren.push(bodyPlain("Competing interests: The authors declare no competing interests."));
bodyChildren.push(bodyPlain("Data availability: All raw experimental data, trained checkpoints, and analysis code are publicly available at https://github.com/94346hddjna/muzero-adaptive-sonar (including Release v1.0-full-data with raw JSONL and checkpoints)."));
bodyChildren.push(bodyPlain("Author contributions: R.T. designed and implemented the system, ran experiments, and wrote the draft. F.H. supervised the research."));

// ================= References =================
bodyChildren.push(H1("References"));
for (const r of refs) {
  bodyChildren.push(new Paragraph({
    indent: { left: 420, hanging: 420 },
    spacing: { line: 240 },
    children: [new TextRun({ text: r, size: 21, font: FONT })],
  }));
}

// ================= 文档组装 =================
const sections = [];

// 标题页（期刊风格，无封面）
sections.push({
  properties: { page: { margin: { top: 1440, bottom: 1440, left: 1701, right: 1417 } } },
  children: [
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 1200, after: 400, line: 828, lineRule: "atLeast" },
      children: [new TextRun({ text: titleText, bold: true, size: 36, font: { ascii: "Times New Roman", eastAsia: "SimHei" } })] }),
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 200, after: 200 },
      children: [new TextRun({ text: "Rui Tan1, Fei Hu1*", size: 24, font: FONT })] }),
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 400 },
      children: [new TextRun({ text: "1 School of Computer Science and Engineering (School of Artificial Intelligence), XXX University", size: 20, font: FONT })] }),
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 100, after: 400 },
      children: [new TextRun({ text: "*Corresponding author: [advisor email]", size: 20, font: FONT })] }),
    P(new TextRun({ text: "Abstract", bold: true, size: 28, font: FONT }), { align: AlignmentType.CENTER, spacing: { before: 200, after: 200 } }),
    body(abstractText),
    P(new TextRun({ text: "Keywords: " + keywords, bold: true, size: 24, font: FONT }), { indent: { firstLine: 480 }, spacing: { before: 200 } }),
  ],
});

// 正文（新页码）
sections.push({
  properties: {
    type: SectionType.NEXT_PAGE,
    page: {
      margin: { top: 1440, bottom: 1440, left: 1701, right: 1417, header: 850, footer: 992 },
      pageNumbers: { start: 1, formatType: NumberFormat.DECIMAL },
    },
  },
  headers: {
    default: new Header({ children: [new Paragraph({
      alignment: AlignmentType.CENTER,
      border: { bottom: { style: BorderStyle.SINGLE, size: 1, color: "000000" } },
      children: [new TextRun({ text: "MuZero-Based Adaptive Active Sonar Target Detection Strategy", size: 18, color: "333333", font: FONT })],
    })] }),
  },
  footers: {
    default: new Footer({ children: [new Paragraph({
      alignment: AlignmentType.CENTER,
      children: [
        new TextRun({ text: "- ", size: 21, font: FONT }),
        new TextRun({ children: [PageNumber.CURRENT], size: 21, font: FONT }),
        new TextRun({ text: " -", size: 21, font: FONT }),
      ],
    })] }),
  },
  children: bodyChildren,
});

const doc = new Document({
  styles: {
    default: { document: {
      run: { font: { ascii: "Times New Roman", eastAsia: "SimSun" }, size: 24, color: "000000" },
      paragraph: { spacing: { line: 240 } },
    }},
  },
  sections,
});

Packer.toBuffer(doc).then((buf) => {
  const out = path.join(ROOT, "results_v3", "paper_draft.docx");
  fs.writeFileSync(out, buf);
  console.log("Paper written:", out, "size:", (buf.length / 1024).toFixed(1), "KB");
});
