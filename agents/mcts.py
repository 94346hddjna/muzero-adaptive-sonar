# -*- coding: utf-8 -*-
"""MCTS（开题报告式(24)-(26)、第5章）。

PUCT（式(24)）：
  a* = argmax_a (Q(s,a)+U(s,a))
  U(s,a) = c(s)·P(s,a)·√(N_b(s))/(1+N(s,a))，c(s)=c_1+ln((N_b(s)+c_base)/c_base)
  Q(s,a)=W(s,a)/N(s,a)（即子节点自身value_sum/visit_count）
回传（式(25)）：G^k = Σ_τ γ^τ·r^{k+1+τ} + γ^{l−k}·v^l（自叶向根逐级 G←r+γG）
根策略（式(26)）：π(a|s) = N(s,a)^{1/τ}/Σ_b N(s,b)^{1/τ}

四阶段循环：选择/扩展/评估/回传。
- 搜索仅使用MuZero学习的隐式动力学g_θ/预测f_θ（想象轨迹），绝不调用真实环境；
  仅最终选中动作作用于真实声呐环境（双通道，开题报告第(五)节）。
- 根节点先验注入Dirichlet噪声（训练阶段）。
- 批量实现：B个根各自独立顺序搜索M次，每轮把各根的叶扩展请求合并为一次批量
  网络前向（跨环境batching，不改变单根搜索的串行语义）。
- 规划深度上限D_p：到达深度上限的已扩展节点以自身价值估计截断回传（前瞻截断+自举）。
"""
import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from env.level1_env import aggregate_cells_to_sectors, sector_center_cell


class Node:
    """搜索树节点。reward/value均为真实尺度（网络输出已乘尺度系数）。"""
    __slots__ = ("prior", "children", "hidden", "reward", "value_estimate",
                 "visit_count", "value_sum", "depth", "has_children")

    def __init__(self, prior: np.ndarray, depth: int = 0):
        self.prior = prior                       # [N] f_θ策略先验（扇区级，边缘化后）
        self.children: List[Optional["Node"]] = []
        self.hidden: Optional[torch.Tensor] = None
        self.reward: float = 0.0                 # 进入本节点的预测奖励r^k
        self.value_estimate: float = 0.0         # f_θ价值估计（深度截断叶回传用）
        self.visit_count: int = 0
        self.value_sum: float = 0.0
        self.depth = depth
        self.has_children = False                # ⟺已扩展（children数组已建）


class BatchedPUCTMCTS:
    """跨环境批量的PUCT MCTS。每个环境一棵树，顺序M次模拟，网络调用按轮合并。"""

    def __init__(
        self,
        nets,
        num_simulations: int = 50,
        planning_depth: int = 5,
        puct_c1: float = 1.25,
        puct_cbase: float = 19652.0,
        gamma: float = 0.99,
        dirichlet_alpha: float = 0.3,
        dirichlet_weight: float = 0.25,
        canonical_cells: int = 192,
        device: str = "cpu",
        value_scale: float = 1.0,
        reward_scale: float = 1.0,
    ):
        self.nets = nets
        self.M = int(num_simulations)
        self.Dp = max(1, int(planning_depth))
        self.c1 = puct_c1
        self.cbase = puct_cbase
        self.gamma = gamma
        self.dir_alpha = dirichlet_alpha
        self.dir_eps = dirichlet_weight
        self.C = canonical_cells
        self.device = device
        self.value_scale = value_scale
        self.reward_scale = reward_scale
        self.stats = {"net_calls": 0, "simulations": 0}

    # -- 单节点PUCT选择（式24） ---------------------------------------------
    def _select(self, node: Node) -> int:
        children = node.children
        n_b = node.visit_count
        c_exp = self.c1 + math.log((n_b + self.cbase) / self.cbase)
        q = np.zeros(len(children))
        nv = np.zeros(len(children))
        for i, ch in enumerate(children):
            if ch is not None and ch.visit_count > 0:
                q[i] = ch.value_sum / ch.visit_count
                nv[i] = ch.visit_count
        u = c_exp * node.prior * math.sqrt(max(n_b, 1)) / (1.0 + nv)
        return int(np.argmax(q + u))

    # -- 回传（式25） --------------------------------------------------------
    def _backup(self, path: List[Node], leaf_value: float):
        G = leaf_value
        for i in range(len(path) - 1, -1, -1):
            node = path[i]
            node.value_sum += G
            node.visit_count += 1
            G = node.reward + self.gamma * G

    # -- 批量搜索 ------------------------------------------------------------
    @torch.no_grad()
    def search(
        self,
        root_hidden: torch.Tensor,     # [B, hidden]（表示网络输出）
        n_sectors: np.ndarray,         # [B]
        add_dirichlet: bool = False,
        rng: Optional[np.random.Generator] = None,
    ) -> Tuple[List[np.ndarray], np.ndarray, List[Node]]:
        """返回 (visits_list[每env的根访问分布], root_values[B], roots)。

        visits=根各动作访问次数（式26的N(s,a)）；root_values=搜索价值估计
        （真实尺度，用作式(20)n步目标的自举项）。roots供机理可视化分析。
        """
        nets = self.nets
        B = root_hidden.shape[0]
        rng = rng or np.random.default_rng()
        # 根扩展：f_θ(h^0) → 策略先验（+Dirichlet噪声）
        logits, values = nets.prediction(root_hidden)
        prior_cells = torch.softmax(logits, dim=-1).cpu().numpy()
        roots: List[Node] = []
        for e in range(B):
            N = int(n_sectors[e])
            prior = aggregate_cells_to_sectors(prior_cells[e], N)
            prior = prior / max(prior.sum(), 1e-9)
            if add_dirichlet:
                noise = rng.dirichlet(np.full(N, self.dir_alpha))
                prior = (1 - self.dir_eps) * prior + self.dir_eps * noise
            root = Node(prior, depth=0)
            root.hidden = root_hidden[e:e + 1]
            root.children = [None] * N
            root.has_children = True
            roots.append(root)
        self.stats["net_calls"] += 1

        # M次模拟：每轮各env走到叶，合并网络调用
        for _sim in range(self.M):
            expand_reqs = []    # (env, parent_node, action, path(含root..parent))
            truncate_reqs = []  # (path(含root..叶)) 深度截断
            for e in range(B):
                node = roots[e]
                path = [node]
                leaf_found = False
                while node.has_children and node.depth < self.Dp:
                    a = self._select(node)
                    child = node.children[a]
                    if child is None:
                        expand_reqs.append((e, node, a, path))
                        leaf_found = True
                        break
                    path.append(child)
                    node = child
                if not leaf_found:
                    # 到达深度上限的已扩展节点：以自身价值估计截断回传
                    truncate_reqs.append(path)
            # 批量扩展叶：g_θ(父hidden, 动作)→(r,h')，f_θ(h')→(先验,价值)
            if expand_reqs:
                h_par = torch.stack([req[1].hidden[0] for req in expand_reqs], dim=0)
                cells = []
                for (e, node, a, path) in expand_reqs:
                    N = int(n_sectors[e])
                    cells.append(sector_center_cell(a, N, self.C))
                oh = torch.zeros(len(cells), self.C, device=self.device)
                oh[torch.arange(len(cells)), torch.tensor(cells, device=self.device)] = 1.0
                r_hat, h_child = nets.dynamics(h_par, oh)               # 式(18)
                logits_c, v_c = nets.prediction(h_child)                # 式(19)
                priors_c = torch.softmax(logits_c, dim=-1).cpu().numpy()
                r_real = r_hat.squeeze(-1).cpu().numpy() * self.reward_scale
                v_real = v_c.squeeze(-1).cpu().numpy() * self.value_scale
                self.stats["net_calls"] += 2
                for j, (e, node, a, path) in enumerate(expand_reqs):
                    N = int(n_sectors[e])
                    prior = aggregate_cells_to_sectors(priors_c[j], N)
                    prior = prior / max(prior.sum(), 1e-9)
                    child = Node(prior, node.depth + 1)
                    child.hidden = h_child[j:j + 1].detach()
                    child.reward = float(r_real[j])
                    child.value_estimate = float(v_real[j])
                    child.children = [None] * N
                    child.has_children = True
                    node.children[a] = child
                    self._backup(path + [child], float(v_real[j]))       # 式(25)
            for path in truncate_reqs:
                leaf = path[-1]
                self._backup(path, leaf.value_estimate)
            self.stats["simulations"] += B
        # 汇总：根访问分布（式26 τ=1归一化形式由normalized_policy()处理）
        visits_out = []
        root_vals = np.zeros(B)
        for e in range(B):
            root = roots[e]
            visits_out.append(np.array([c.visit_count if c is not None else 0
                                        for c in root.children]))
            root_vals[e] = root.value_sum / max(root.visit_count, 1)
        return visits_out, root_vals, roots


def apply_temperature(visits: np.ndarray, temperature: float, rng: np.random.Generator) -> int:
    """式(26)动作采样：τ>0按N^(1/τ)归一化采样；τ=0访问数argmax（评测）。"""
    if temperature <= 0:
        return int(np.argmax(visits))
    v = np.asarray(visits, dtype=np.float64)
    if v.sum() <= 0:
        return int(rng.integers(len(v)))
    w = v ** (1.0 / temperature)
    w = w / w.sum()
    return int(rng.choice(len(v), p=w))


def normalized_policy(visits: np.ndarray) -> np.ndarray:
    """式(26) τ=1：π(a|s)=N(s,a)/Σ_b N(s,b)（训练策略蒸馏目标）。"""
    v = np.asarray(visits, dtype=np.float64)
    s = v.sum()
    if s <= 0:
        return np.full(len(v), 1.0 / len(v))
    return v / s
