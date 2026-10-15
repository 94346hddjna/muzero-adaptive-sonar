# -*- coding: utf-8 -*-
"""用 qwen-image 生成论文框架图/原理图（AI生图）。"""
import base64
import json
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results_v3" / "figures" / "schematic"
OUT.mkdir(parents=True, exist_ok=True)

# 凭证池（用户提供的DashScope密钥）
KEYS = [
    ("https://llm-h1hl3i1mrsz01yiz.cn-beijing.maas.aliyuncs.com/compatible-mode/v1",
     "sk-40629d4cccb44bf3b57a88cf69a23fae"),
    ("https://llm-0wh4qxgauf8u61nx.cn-beijing.maas.aliyuncs.com/compatible-mode/v1",
     "sk-8aebe5909b9143eb9b350b0271b71b79"),
]
KEY_IDX = 0
PROXY = {"http": "http://127.0.0.1:7890", "https": "http://127.0.0.1:7890"}
MODEL = "qwen-image-2.1-pro"


def gen(prompt: str, name: str, size: str = "1024x1024") -> Path:
    """生成一张图，返回本地路径。"""
    global KEY_IDX
    base, key = KEYS[KEY_IDX % len(KEYS)]
    KEY_IDX += 1
    payload = {"model": MODEL, "prompt": prompt, "size": size, "n": 1}
    r = requests.post(f"{base}/images/generations",
                      headers={"Authorization": f"Bearer {key}",
                               "Content-Type": "application/json"},
                      json=payload, timeout=180, proxies=PROXY)
    r.raise_for_status()
    url = r.json()["data"][0]["url"]
    img = requests.get(url, timeout=120, proxies=PROXY)
    img.raise_for_status()
    out = OUT / f"{name}.png"
    out.write_bytes(img.content)
    print(f"✅ {name}.png ({len(img.content)/1024:.0f} KB)")
    return out


# ============ 图1: 认知声呐闭环 ============
gen(
    "A clean professional scientific diagram on white background showing the cognitive sonar "
    "closed-loop adaptive detection cycle. A circular flow diagram with 4 stages connected by "
    "curved arrows in a clockwise loop. Stage 1 'Transmit' at top shows a sonar emitting beam; "
    "Stage 2 'Observe' on the right shows echo reception; Stage 3 'Cognize/Learn' at bottom shows "
    "a brain/AI icon updating a belief heatmap; Stage 4 'Re-transmit (adapted)' on the left shows "
    "an adapted beam. In the center, a label 'Adaptive Detection Loop'. Clean minimal flat design, "
    "professional scientific figure style, muted blue/teal palette, crisp vector-like edges, "
    "no photorealistic elements, white background, high contrast text labels in black.",
    "schematic_cognitive_loop",
    "1024x1024",
)

# ============ 图2: MuZero原理示意 ============
gen(
    "A clean professional scientific diagram on white background illustrating the MuZero "
    "reinforcement learning architecture with implicit world model and MCTS planning. "
    "Left side: an 'Interaction History' box feeding into a 'Representation Network h' box "
    "which outputs a 'Latent State' box. Center: a tree diagram labeled 'MCTS in Latent Space' "
    "with branches. Right side: a 'Dynamics Network g' box and 'Prediction Network f' box, "
    "with arrows showing 'reward + next latent' and 'policy + value'. Bottom: an arrow labeled "
    "'improved policy π' going back to the environment. Clean minimal flat design, professional "
    "scientific figure style, muted blue/teal/grey palette, crisp vector-like edges, "
    "white background, black text labels, no photorealistic elements.",
    "schematic_muzero_arch",
    "1024x768",
)

# ============ 图3: 系统总体架构（六层双通道） ============
gen(
    "A clean professional scientific diagram on white background showing a six-layer system "
    "architecture for an adaptive sonar system. Six horizontal stacked rectangular layers from "
    "bottom to top labeled: 1 'Acoustic Environment', 2 'Sonar Simulation', 3 'Belief Update', "
    "4 'MuZero Decision Model', 5 'MCTS Planning', 6 'Action Execution & Feedback'. A prominent "
    "vertical arrow on the left side labeled 'Real Environment Channel' pointing up, and a "
    "vertical arrow on the right side labeled 'MuZero Latent Planning Channel' pointing up. "
    "Clean minimal flat design, professional scientific figure, muted blue/grey palette, "
    "crisp edges, white background, black text labels, no photorealistic elements.",
    "schematic_system_architecture",
    "768x1024",
)

# ============ 图4: POMDP交互示意 ============
gen(
    "A clean professional scientific diagram on white background showing the POMDP interaction "
    "loop for sonar target detection. On the left, a 'Hidden Target Sector' box (hidden state, "
    "dashed border). An arrow labeled 'action a_t (beam pointing)' goes right to an "
    "'Active Sonar' box. An arrow labeled 'observation o_t (detection 0/1)' goes down to a "
    "'Bayesian Belief Filter' box. An arrow labeled 'belief b_t' goes right to a 'Policy π' box, "
    "and from the policy an arrow goes back up to the action. Clean minimal flat design, "
    "professional scientific figure, muted blue/teal palette, crisp vector edges, white background, "
    "black text labels.",
    "schematic_pomdp_loop",
    "1024x768",
)

# ============ 图5: 实验设计与评估流程 ============
gen(
    "A clean professional scientific flowchart on white background showing the experimental "
    "pipeline for sonar RL evaluation. Left to right flow with boxes: 'Scenario Generation' → "
    "'Training (5 seeds, domain randomization)' → 'Validation (model selection)' → 'Test "
    "(paired scenarios)' → 'Ablation & OOD' → 'Statistics & Figures'. Below the main flow, a "
    "second row shows 'Baselines (Sequential/Random/Belief-Greedy/DQN/PPO)' and 'MuZero+MCTS' "
    "feeding into 'Training'. Clean minimal flat design, professional scientific figure, "
    "muted blue/teal palette, crisp vector edges, white background, black text labels.",
    "schematic_experiment_pipeline",
    "1024x768",
)

# ============ 图6: 两级仿真环境 ============
gen(
    "A clean professional scientific diagram on white background showing a two-level sonar "
    "simulation environment. Left panel labeled 'Level 1: Detection-Probability Abstraction' "
    "shows a simple box flow: action → SNR equation → P_D/P_FA → Bernoulli detection. Right "
    "panel labeled 'Level 2: Signal-Level' shows a richer flow: LFM pulse → propagation → "
    "delay → noise+reverberation → matched filter → envelope → threshold → detection. A double "
    "arrow between panels labeled 'consistency check'. Clean minimal flat design, professional "
    "scientific figure, muted blue/teal palette, crisp vector edges, white background, black text.",
    "schematic_two_level_env",
    "1024x768",
)

print("全部框架图生成完毕:", OUT)
