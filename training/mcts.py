"""
蒙特卡洛树搜索 (MCTS)
======================
实现 AlphaZero 风格的 MCTS 搜索。
每次搜索执行 num_simulations 次模拟，逐次选择→扩展→评估→回溯。

UCB公式: Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
"""

import numpy as np
import math
from typing import Dict, Optional, Tuple
from game import XiangqiGame, ACTION_SPACE


class MCTSNode:
    """MCTS 树节点"""

    __slots__ = ['parent', 'children', 'visit_count', 'total_value', 'prior']

    def __init__(self, parent: Optional['MCTSNode'] = None, prior: float = 0.0):
        self.parent = parent
        self.children: Dict[int, 'MCTSNode'] = {}
        self.visit_count = 0
        self.total_value = 0.0
        self.prior = prior

    @property
    def q_value(self) -> float:
        """平均价值"""
        if self.visit_count == 0:
            return 0.0
        return self.total_value / self.visit_count

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def select_child(self, c_puct: float = 1.5) -> Tuple[int, 'MCTSNode']:
        """选择 UCB 值最高的子节点"""
        best_score = -float('inf')
        best_action = -1
        best_child = None

        sqrt_parent = math.sqrt(self.visit_count)

        for action, child in self.children.items():
            ucb = child.q_value + c_puct * child.prior * sqrt_parent / (1 + child.visit_count)
            if ucb > best_score:
                best_score = ucb
                best_action = action
                best_child = child

        return best_action, best_child

    def expand(self, action_priors: Dict[int, float]):
        """扩展节点"""
        for action, prior in action_priors.items():
            if action not in self.children:
                self.children[action] = MCTSNode(parent=self, prior=prior)

    def backup(self, value: float):
        """回溯更新"""
        node = self
        while node is not None:
            node.visit_count += 1
            node.total_value += value
            value = -value
            node = node.parent


class MCTS:
    """
    蒙特卡洛树搜索

    参数:
        model: 神经网络模型
        num_simulations: 每次搜索的模拟次数
        c_puct: 探索常数
        device: 计算设备
    """

    def __init__(self, model, num_simulations: int = 200, c_puct: float = 1.5,
                 device: str = 'cpu'):
        self.model = model
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.device = device

    def search(self, game: XiangqiGame, temperature: float = 1.0,
               add_noise: bool = True) -> np.ndarray:
        """
        MCTS 搜索，返回动作概率分布。

        参数:
            game: 当前游戏状态
            temperature: 温度参数（0=贪心，>0 增加随机性）
            add_noise: 是否在根节点添加 Dirichlet 噪声
        """
        root = MCTSNode()

        # 根节点：获取策略先验
        state = game.get_state_for_nn()
        policy_probs, _ = self.model.predict(state, self.device)

        legal_actions = game.get_legal_actions()
        if len(legal_actions) == 0:
            return np.zeros(ACTION_SPACE)

        action_priors = self._mask_and_normalize(policy_probs, legal_actions)

        # 根节点添加 Dirichlet 噪声增加探索
        if add_noise and len(legal_actions) > 0:
            noise = np.random.dirichlet([0.3] * len(legal_actions))
            actions_list = list(action_priors.keys())
            for i, a in enumerate(actions_list):
                action_priors[a] = 0.75 * action_priors[a] + 0.25 * noise[i]

        root.expand(action_priors)

        # 执行模拟
        for _ in range(self.num_simulations):
            node = root
            sim_game = game.clone()

            # 选择：沿树下行到叶节点
            while not node.is_leaf():
                action, node = node.select_child(self.c_puct)
                sim_game.make_action(action)

            # 评估叶节点
            done, winner = sim_game.is_game_over()
            if done:
                value = 0.0 if winner == 0 else -1.0
            else:
                state = sim_game.get_state_for_nn()
                policy_probs, value = self.model.predict(state, self.device)

                legal_actions = sim_game.get_legal_actions()
                if len(legal_actions) > 0:
                    action_priors = self._mask_and_normalize(policy_probs, legal_actions)
                    node.expand(action_priors)

                value = -value

            # 回溯
            node.backup(value)

        return self._get_action_probs(root, temperature)

    def get_action(self, game: XiangqiGame, temperature: float = 0.0,
                   add_noise: bool = False) -> int:
        """获取最佳动作"""
        action_probs = self.search(game, temperature, add_noise)
        if temperature == 0:
            action = np.argmax(action_probs)
        else:
            action = np.random.choice(len(action_probs), p=action_probs)
        return int(action)

    @staticmethod
    def _mask_and_normalize(policy_probs, legal_actions):
        """对合法动作进行掩码和归一化"""
        action_priors = {}
        prob_sum = sum(policy_probs[a] for a in legal_actions)
        if prob_sum > 0:
            for a in legal_actions:
                action_priors[a] = policy_probs[a] / prob_sum
        else:
            uniform = 1.0 / len(legal_actions)
            for a in legal_actions:
                action_priors[a] = uniform
        return action_priors

    @staticmethod
    def _get_action_probs(root: MCTSNode, temperature: float) -> np.ndarray:
        """从根节点的访问计数计算动作概率"""
        action_probs = np.zeros(ACTION_SPACE)
        for action, child in root.children.items():
            action_probs[action] = child.visit_count

        if temperature == 0:
            best_action = max(root.children, key=lambda a: root.children[a].visit_count)
            action_probs = np.zeros(ACTION_SPACE)
            action_probs[best_action] = 1.0
        else:
            if action_probs.sum() > 0:
                action_probs = action_probs ** (1.0 / temperature)
                action_probs /= action_probs.sum()

        return action_probs
