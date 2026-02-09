"""
蒙特卡洛树搜索 (MCTS)
======================
实现AlphaZero风格的MCTS，支持两种模式：
1. 串行模式：逐次模拟，每次单独调用神经网络（兼容旧接口）
2. 批量模式：多次模拟收集叶节点后批量推理，配合 Virtual Loss 实现对局内并行

对局内并行核心思想（Tree Parallelization + Virtual Loss）：
- 在一次 MCTS search 中，同时从根节点出发执行 N 条路径的选择
- 每条路径选择后对经过的节点施加 Virtual Loss（临时减少其 Q 值）
  使后续路径倾向于探索不同分支，避免所有路径挤到同一条线
- 所有路径到达叶节点后，将叶节点状态打包成一个 batch 送入神经网络
- 批量推理完成后，撤销 Virtual Loss 并用真实值回溯更新

UCB公式: Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
"""

import numpy as np
import math
import torch
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List
from game import XiangqiGame, encode_action, decode_action, ACTION_SPACE


class MCTSNode:
    """MCTS树节点，支持 Virtual Loss"""

    __slots__ = ['parent', 'children', 'visit_count', 'total_value',
                 'prior', 'virtual_loss']

    def __init__(self, parent: Optional['MCTSNode'] = None, prior: float = 0.0):
        self.parent = parent
        self.children: Dict[int, 'MCTSNode'] = {}
        self.visit_count = 0
        self.total_value = 0.0
        self.prior = prior
        self.virtual_loss = 0  # Virtual Loss 计数

    @property
    def q_value(self) -> float:
        """平均价值（含 Virtual Loss 惩罚）"""
        total_visits = self.visit_count + self.virtual_loss
        if total_visits == 0:
            return 0.0
        # Virtual Loss 视为输（-1），降低该节点的吸引力
        return (self.total_value - self.virtual_loss) / total_visits

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def select_child(self, c_puct: float = 1.5) -> Tuple[int, 'MCTSNode']:
        """选择UCB值最高的子节点（考虑 Virtual Loss）"""
        best_score = -float('inf')
        best_action = -1
        best_child = None

        # 父节点的有效访问次数 = 真实访问 + virtual loss
        sqrt_parent = math.sqrt(self.visit_count + self.virtual_loss)

        for action, child in self.children.items():
            effective_visits = child.visit_count + child.virtual_loss
            ucb = child.q_value + c_puct * child.prior * sqrt_parent / (1 + effective_visits)
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
        """回溯更新（真实值）"""
        node = self
        while node is not None:
            node.visit_count += 1
            node.total_value += value
            value = -value
            node = node.parent

    def apply_virtual_loss(self):
        """沿路径施加 Virtual Loss"""
        node = self
        while node is not None:
            node.virtual_loss += 1
            node = node.parent

    def revert_virtual_loss(self):
        """撤销 Virtual Loss"""
        node = self
        while node is not None:
            node.virtual_loss -= 1
            node = node.parent


class MCTS:
    """
    蒙特卡洛树搜索（串行模式，兼容旧接口）

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
        """串行 MCTS 搜索（兼容旧接口）"""
        root = MCTSNode()

        state = game.get_state_for_nn()
        policy_probs, _ = self.model.predict(state, self.device)

        legal_actions = game.get_legal_actions()
        if len(legal_actions) == 0:
            return np.zeros(ACTION_SPACE)

        action_priors = self._mask_and_normalize(policy_probs, legal_actions)

        if add_noise and len(legal_actions) > 0:
            noise = np.random.dirichlet([0.3] * len(legal_actions))
            actions_list = list(action_priors.keys())
            for i, a in enumerate(actions_list):
                action_priors[a] = 0.75 * action_priors[a] + 0.25 * noise[i]

        root.expand(action_priors)

        for _ in range(self.num_simulations):
            node = root
            sim_game = game.clone()

            while not node.is_leaf():
                action, node = node.select_child(self.c_puct)
                sim_game.make_action(action)

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


class BatchMCTS:
    """
    批量推理 MCTS（对局内并行）

    核心优化：
    1. Virtual Loss：多条搜索路径同时从根出发，通过 Virtual Loss 分散到不同分支
    2. 批量推理：收集多个叶节点状态后一次性送入神经网络，利用 GPU/CPU 向量化加速
    3. 可配置批大小：batch_size 控制每次并行搜索的路径数

    参数:
        model: 神经网络模型（nn.Module，需要 forward 方法）
        num_simulations: 总模拟次数
        c_puct: 探索常数
        device: 计算设备
        batch_size: 每次并行搜索的路径数（即 Virtual Loss 并行度）
    """

    def __init__(self, model, num_simulations: int = 200, c_puct: float = 1.5,
                 device: str = 'cpu', batch_size: int = 8):
        self.model = model
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.device = device
        self.batch_size = batch_size

    def search(self, game: XiangqiGame, temperature: float = 1.0,
               add_noise: bool = True) -> np.ndarray:
        """
        批量 MCTS 搜索

        将 num_simulations 次模拟分成多个 batch，每个 batch 内：
        1. 并行执行 batch_size 条路径的选择（带 Virtual Loss）
        2. 收集所有叶节点状态
        3. 批量神经网络推理
        4. 撤销 Virtual Loss，用真实值回溯
        """
        root = MCTSNode()

        # 根节点初始化
        state = game.get_state_for_nn()
        policy_probs, _ = self._predict_single(state)

        legal_actions = game.get_legal_actions()
        if len(legal_actions) == 0:
            return np.zeros(ACTION_SPACE)

        action_priors = MCTS._mask_and_normalize(policy_probs, legal_actions)

        if add_noise and len(legal_actions) > 0:
            noise = np.random.dirichlet([0.3] * len(legal_actions))
            actions_list = list(action_priors.keys())
            for i, a in enumerate(actions_list):
                action_priors[a] = 0.75 * action_priors[a] + 0.25 * noise[i]

        root.expand(action_priors)

        # 分批执行模拟
        sims_done = 0
        while sims_done < self.num_simulations:
            current_batch = min(self.batch_size, self.num_simulations - sims_done)
            self._run_batch(root, game, current_batch)
            sims_done += current_batch

        return MCTS._get_action_probs(root, temperature)

    def _run_batch(self, root: MCTSNode, game: XiangqiGame, batch_size: int):
        """
        执行一个 batch 的并行模拟。

        步骤：
        1. 从根节点出发，沿 UCB 选择路径到叶节点，施加 Virtual Loss
        2. 收集所有需要神经网络评估的叶节点
        3. 批量推理
        4. 撤销 Virtual Loss，扩展叶节点，回溯真实值
        """
        # 收集路径信息
        paths = []  # [(leaf_node, sim_game, is_terminal, terminal_value)]
        nn_eval_indices = []  # 需要 NN 评估的路径索引
        nn_states = []  # 对应的状态

        for i in range(batch_size):
            node = root
            sim_game = game.clone()

            # 选择阶段：沿 UCB 走到叶节点
            while not node.is_leaf():
                action, node = node.select_child(self.c_puct)
                sim_game.make_action(action)

            # 施加 Virtual Loss，使后续路径避开此分支
            node.apply_virtual_loss()

            # 检查是否终局
            done, winner = sim_game.is_game_over()
            if done:
                value = 0.0 if winner == 0 else -1.0
                paths.append((node, sim_game, True, value))
            else:
                paths.append((node, sim_game, False, 0.0))
                nn_eval_indices.append(i)
                nn_states.append(sim_game.get_state_for_nn())

        # 批量神经网络推理
        if nn_states:
            batch_policies, batch_values = self._predict_batch(nn_states)

        # 处理每条路径：撤销 VL，扩展，回溯
        nn_idx = 0
        for i, (node, sim_game, is_terminal, terminal_value) in enumerate(paths):
            # 撤销 Virtual Loss
            node.revert_virtual_loss()

            if is_terminal:
                node.backup(terminal_value)
            else:
                policy_probs = batch_policies[nn_idx]
                value = batch_values[nn_idx]
                nn_idx += 1

                # 扩展叶节点
                legal_actions = sim_game.get_legal_actions()
                if len(legal_actions) > 0:
                    action_priors = MCTS._mask_and_normalize(policy_probs, legal_actions)
                    node.expand(action_priors)

                node.backup(-value)

    def _predict_single(self, state: np.ndarray):
        """单个状态预测"""
        self.model.eval()
        with torch.no_grad():
            x = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            policy_logits, value = self.model(x)
            policy_probs = F.softmax(policy_logits, dim=1).squeeze(0).cpu().numpy()
            value = value.item()
        return policy_probs, value

    def _predict_batch(self, states: List[np.ndarray]):
        """
        批量状态预测 — 对局内并行的核心加速点

        将多个状态打包成一个 batch 送入神经网络，
        利用矩阵运算的并行性（GPU CUDA cores / CPU SIMD）加速推理。
        """
        self.model.eval()
        with torch.no_grad():
            batch = torch.FloatTensor(np.array(states)).to(self.device)
            policy_logits, values = self.model(batch)
            policies = F.softmax(policy_logits, dim=1).cpu().numpy()
            values = values.squeeze(-1).cpu().numpy()
        return policies, values

    def get_action(self, game: XiangqiGame, temperature: float = 0.0,
                   add_noise: bool = False) -> int:
        """获取最佳动作"""
        action_probs = self.search(game, temperature, add_noise)
        if temperature == 0:
            action = np.argmax(action_probs)
        else:
            action = np.random.choice(len(action_probs), p=action_probs)
        return int(action)


class MultiGameBatchMCTS:
    """
    多对局批量 MCTS — 最大化并行度

    同时管理多局对弈，将所有对局的 MCTS 叶节点收集到一起做批量推理。
    这是 128 核机器上最高效的方案：
    - 对局间并行：同时推进 N 局对弈
    - 对局内并行：每局 MCTS 使用 Virtual Loss 并行搜索 M 条路径
    - 批量推理：所有对局的叶节点合并成一个大 batch 推理

    总并行度 = N(对局数) × M(每局VL并行度)

    参数:
        model: 神经网络模型
        num_simulations: 每步 MCTS 模拟次数
        c_puct: 探索常数
        device: 计算设备
        vl_batch_size: 每局 Virtual Loss 并行度
    """

    def __init__(self, model, num_simulations: int = 200, c_puct: float = 1.5,
                 device: str = 'cpu', vl_batch_size: int = 8):
        self.model = model
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.device = device
        self.vl_batch_size = vl_batch_size

    def play_games(self, num_games: int, config) -> List[Tuple]:
        """
        同时执行多局自对弈。

        在同一个进程/线程中交替推进多局对弈，
        每一步将所有对局的 MCTS 叶节点合并做批量推理。

        参数:
            num_games: 同时进行的对局数
            config: TrainingConfig

        返回:
            results: [(training_data, winner, steps), ...]
        """
        import random

        # 初始化所有对局
        games_info = []
        for _ in range(num_games):
            game = XiangqiGame()
            # 随机开局
            if config.random_opening_moves > 0:
                num_random = random.randint(0, config.random_opening_moves)
                for _ in range(num_random):
                    legal = game.get_legal_moves()
                    if not legal:
                        break
                    move = random.choice(legal)
                    game.make_move(*move)
                    done, _ = game.is_game_over()
                    if done:
                        break

            games_info.append({
                'game': game,
                'game_data': [],  # (state, action_probs, current_player)
                'step': 0,
                'done': False,
                'winner': None,
                'resign_counter': 0,
            })

        # 检查是否有已结束的对局（随机开局可能导致）
        for info in games_info:
            done, winner = info['game'].is_game_over()
            if done:
                info['done'] = True
                info['winner'] = winner

        # 主循环：逐步推进所有对局
        while any(not info['done'] for info in games_info):
            active_games = [info for info in games_info if not info['done']]
            if not active_games:
                break

            # 为所有活跃对局执行一步 MCTS 搜索
            self._step_all_games(active_games, config)

        # 收集结果
        results = []
        for info in games_info:
            winner = info['winner'] if info['winner'] is not None else 0
            training_data = []
            for state, action_probs, player in info['game_data']:
                if winner == 0:
                    value = 0.0
                elif winner == player:
                    value = 1.0
                else:
                    value = -1.0
                training_data.append((state, action_probs, value))
            results.append((training_data, winner, info['step']))

        return results

    def _step_all_games(self, active_games: List[dict], config):
        """
        为所有活跃对局执行一步 MCTS 搜索。

        核心流程：
        1. 为每局创建 MCTS 根节点并初始化（批量推理根节点）
        2. 分批执行模拟：每批收集所有对局的叶节点，合并批量推理
        3. 选择动作，推进对局
        """
        num_active = len(active_games)

        # --- 阶段 1：批量初始化根节点 ---
        root_states = []
        root_legal_actions = []
        for info in active_games:
            root_states.append(info['game'].get_state_for_nn())
            root_legal_actions.append(info['game'].get_legal_actions())

        root_policies, _ = self._predict_batch(root_states)

        roots = []
        for i, info in enumerate(active_games):
            root = MCTSNode()
            legal = root_legal_actions[i]
            if len(legal) == 0:
                info['done'] = True
                info['winner'] = -info['game'].current_player
                roots.append(None)
                continue

            action_priors = MCTS._mask_and_normalize(root_policies[i], legal)

            # Dirichlet 噪声
            step = info['step']
            if step < config.temperature_threshold:
                noise = np.random.dirichlet([0.3] * len(legal))
                actions_list = list(action_priors.keys())
                for j, a in enumerate(actions_list):
                    action_priors[a] = 0.75 * action_priors[a] + 0.25 * noise[j]

            root.expand(action_priors)
            roots.append(root)

        # --- 阶段 2：分批执行 MCTS 模拟 ---
        sims_done = 0
        while sims_done < self.num_simulations:
            current_vl_batch = min(self.vl_batch_size, self.num_simulations - sims_done)

            # 收集所有对局所有路径的叶节点
            all_paths = []  # (game_idx, node, sim_game, is_terminal, terminal_value)
            nn_states = []
            nn_path_indices = []

            for g_idx, info in enumerate(active_games):
                if info['done'] or roots[g_idx] is None:
                    continue

                root = roots[g_idx]
                game = info['game']

                for _ in range(current_vl_batch):
                    node = root
                    sim_game = game.clone()

                    while not node.is_leaf():
                        action, node = node.select_child(self.c_puct)
                        sim_game.make_action(action)

                    node.apply_virtual_loss()

                    done, winner = sim_game.is_game_over()
                    if done:
                        value = 0.0 if winner == 0 else -1.0
                        all_paths.append((g_idx, node, sim_game, True, value))
                    else:
                        all_paths.append((g_idx, node, sim_game, False, 0.0))
                        nn_path_indices.append(len(all_paths) - 1)
                        nn_states.append(sim_game.get_state_for_nn())

            # 批量推理
            if nn_states:
                batch_policies, batch_values = self._predict_batch(nn_states)

            # 处理所有路径
            nn_idx = 0
            for path_idx, (g_idx, node, sim_game, is_terminal, terminal_value) in enumerate(all_paths):
                node.revert_virtual_loss()

                if is_terminal:
                    node.backup(terminal_value)
                else:
                    policy_probs = batch_policies[nn_idx]
                    value = batch_values[nn_idx]
                    nn_idx += 1

                    legal_actions = sim_game.get_legal_actions()
                    if len(legal_actions) > 0:
                        action_priors = MCTS._mask_and_normalize(policy_probs, legal_actions)
                        node.expand(action_priors)

                    node.backup(-value)

            sims_done += current_vl_batch

        # --- 阶段 3：选择动作并推进对局 ---
        for g_idx, info in enumerate(active_games):
            if info['done'] or roots[g_idx] is None:
                continue

            root = roots[g_idx]
            game = info['game']
            step = info['step']

            # 温度调度
            if step < config.temperature_threshold:
                temperature = 1.0
            elif step < config.temperature_threshold + 10:
                temperature = 1.0 - 0.9 * (step - config.temperature_threshold) / 10
            else:
                temperature = 0.1

            action_probs = MCTS._get_action_probs(root, temperature)

            # 保存训练数据
            state = game.get_state_for_nn()
            info['game_data'].append((state, action_probs, game.current_player))

            # 选择动作
            if temperature > 0.05:
                action = np.random.choice(len(action_probs), p=action_probs)
            else:
                action = np.argmax(action_probs)

            # 执行动作
            fr, fc, tr, tc = decode_action(action)
            game.make_move(fr, fc, tr, tc)
            info['step'] += 1

            # 检查结束
            done, winner = game.is_game_over()
            if done:
                info['done'] = True
                info['winner'] = winner
                continue

            # 超步数检查
            if info['step'] >= config.max_game_length:
                done, winner = game.is_game_over()
                info['done'] = True
                info['winner'] = winner if winner is not None else 0
                continue

            # 认输机制
            if config.enable_resign and info['step'] > 40:
                eval_state = game.get_state_for_nn()
                _, value = self._predict_single(eval_state)
                if value < config.resign_threshold:
                    info['resign_counter'] += 1
                else:
                    info['resign_counter'] = 0
                if info['resign_counter'] >= config.resign_check_steps:
                    info['done'] = True
                    info['winner'] = -game.current_player

    def _predict_single(self, state: np.ndarray):
        """单个状态预测"""
        self.model.eval()
        with torch.no_grad():
            x = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            policy_logits, value = self.model(x)
            policy_probs = F.softmax(policy_logits, dim=1).squeeze(0).cpu().numpy()
            value = value.item()
        return policy_probs, value

    def _predict_batch(self, states: List[np.ndarray]):
        """批量状态预测"""
        self.model.eval()
        with torch.no_grad():
            batch = torch.FloatTensor(np.array(states)).to(self.device)
            policy_logits, values = self.model(batch)
            policies = F.softmax(policy_logits, dim=1).cpu().numpy()
            values = values.squeeze(-1).cpu().numpy()
        return policies, values


if __name__ == "__main__":
    import time
    from model import XiangqiNet

    model = XiangqiNet(num_channels=64, num_res_blocks=3)
    game = XiangqiGame()

    # 测试串行 MCTS
    print("=" * 60)
    print("测试串行 MCTS (50 次模拟)...")
    mcts = MCTS(model, num_simulations=50, c_puct=1.5)
    t0 = time.time()
    probs = mcts.search(game, temperature=1.0)
    t1 = time.time()
    print(f"  耗时: {t1-t0:.2f}s")

    # 测试批量 MCTS
    print("=" * 60)
    print("测试批量 MCTS (50 次模拟, batch=8)...")
    batch_mcts = BatchMCTS(model, num_simulations=50, c_puct=1.5, batch_size=8)
    t0 = time.time()
    probs2 = batch_mcts.search(game, temperature=1.0)
    t1 = time.time()
    print(f"  耗时: {t1-t0:.2f}s")

    # 测试多对局批量 MCTS
    print("=" * 60)
    print("测试多对局批量 MCTS (4局同时, 50次模拟, VL batch=4)...")

    class MockConfig:
        temperature_threshold = 10
        max_game_length = 50
        random_opening_moves = 2
        enable_resign = False
        resign_threshold = -0.85
        resign_check_steps = 3

    multi_mcts = MultiGameBatchMCTS(model, num_simulations=50, c_puct=1.5, vl_batch_size=4)
    t0 = time.time()
    results = multi_mcts.play_games(4, MockConfig())
    t1 = time.time()
    print(f"  4局总耗时: {t1-t0:.2f}s")
    for i, (data, winner, steps) in enumerate(results):
        w = '红' if winner == 1 else '黑' if winner == -1 else '和'
        print(f"  对局{i+1}: 步数={steps}, 赢家={w}, 样本数={len(data)}")
