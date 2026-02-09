"""
AlphaZero训练框架
==================
实现完整的自对弈强化学习训练流程。

训练流程：
1. 自对弈(Self-Play): 使用当前最优模型 + MCTS 进行自对弈，生成训练数据
2. 训练(Train): 用自对弈数据训练神经网络
3. 评估(Evaluate): 新模型与旧模型对弈，胜率超过阈值则更新最优模型
4. 循环迭代

使用方法：
    python train.py --iterations 100 --games-per-iter 20 --simulations 200
"""

import os
import sys
import time
import json
import random
import argparse
import logging
from collections import deque
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from game import XiangqiGame, ACTION_SPACE, decode_action, PIECE_NAMES
from model import XiangqiNet, count_parameters
from mcts import MCTS

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('training.log')
    ]
)
logger = logging.getLogger(__name__)


class TrainingConfig:
    """训练配置"""

    def __init__(self):
        # 模型参数
        self.num_channels = 128       # 卷积通道数
        self.num_res_blocks = 6       # 残差块数量

        # MCTS参数
        self.num_simulations = 200    # 每步MCTS模拟次数
        self.c_puct = 1.5             # 探索常数
        self.temperature_threshold = 15  # 前N步使用温度=1，之后温度=0

        # 自对弈参数
        self.num_games_per_iter = 20  # 每轮自对弈局数
        self.max_game_length = 200    # 最大游戏步数

        # 训练参数
        self.num_iterations = 100     # 总训练轮数
        self.batch_size = 256         # 批大小
        self.num_epochs = 5           # 每轮训练epoch数
        self.learning_rate = 0.001    # 学习率
        self.weight_decay = 1e-4      # L2正则化
        self.lr_milestones = [50, 80] # 学习率衰减节点
        self.lr_gamma = 0.1           # 学习率衰减因子

        # 数据管理
        self.max_buffer_size = 50000  # 经验回放缓冲区大小
        self.min_buffer_size = 2000   # 开始训练的最小数据量

        # 评估参数
        self.eval_games = 10          # 评估对弈局数
        self.eval_win_rate = 0.55     # 更新模型的最低胜率
        self.eval_simulations = 100   # 评估时MCTS模拟次数

        # 保存
        self.checkpoint_dir = '../models'
        self.save_interval = 5        # 每N轮保存一次

        # 设备
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'


class SelfPlayDataset(Dataset):
    """自对弈数据集"""

    def __init__(self, data: List[Tuple[np.ndarray, np.ndarray, float]]):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        state, policy, value = self.data[idx]
        return (
            torch.FloatTensor(state),
            torch.FloatTensor(policy),
            torch.FloatTensor([value])
        )


def augment_data(state: np.ndarray, policy: np.ndarray, value: float) -> List[Tuple]:
    """
    数据增强：中国象棋棋盘可以左右镜像翻转

    注意：由于中国象棋棋盘不是正方形，只能做水平翻转
    """
    augmented = [(state, policy, value)]

    # 水平翻转
    flipped_state = np.flip(state, axis=2).copy()  # 沿列翻转
    flipped_policy = np.zeros_like(policy)

    for action_idx in range(ACTION_SPACE):
        if policy[action_idx] > 0:
            fr, fc, tr, tc = decode_action(action_idx)
            # 翻转列坐标
            new_fc = 8 - fc
            new_tc = 8 - tc
            from game import encode_action
            new_action = encode_action(fr, new_fc, tr, new_tc)
            flipped_policy[new_action] = policy[action_idx]

    augmented.append((flipped_state, flipped_policy, value))
    return augmented


class AlphaZeroTrainer:
    """AlphaZero训练器"""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = config.device

        # 创建模型
        self.current_model = XiangqiNet(
            num_channels=config.num_channels,
            num_res_blocks=config.num_res_blocks
        ).to(self.device)

        self.best_model = XiangqiNet(
            num_channels=config.num_channels,
            num_res_blocks=config.num_res_blocks
        ).to(self.device)

        # 复制初始权重
        self.best_model.load_state_dict(self.current_model.state_dict())

        # 优化器
        self.optimizer = optim.Adam(
            self.current_model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )

        self.scheduler = optim.lr_scheduler.MultiStepLR(
            self.optimizer,
            milestones=config.lr_milestones,
            gamma=config.lr_gamma
        )

        # 经验回放缓冲区
        self.replay_buffer = deque(maxlen=config.max_buffer_size)

        # 训练统计
        self.iteration = 0
        self.total_games = 0
        self.training_stats = []

        # 创建保存目录
        os.makedirs(config.checkpoint_dir, exist_ok=True)

        logger.info(f"设备: {self.device}")
        logger.info(f"模型参数量: {count_parameters(self.current_model):,}")

    def self_play_game(self) -> List[Tuple[np.ndarray, np.ndarray, int]]:
        """
        执行一局自对弈

        返回: [(state, action_probs, current_player), ...]
        """
        game = XiangqiGame()
        mcts = MCTS(
            self.best_model,
            num_simulations=self.config.num_simulations,
            c_puct=self.config.c_puct,
            device=self.device
        )

        game_data = []
        step = 0

        while step < self.config.max_game_length:
            # 前N步使用温度=1增加探索
            temperature = 1.0 if step < self.config.temperature_threshold else 0.1

            # MCTS搜索
            action_probs = mcts.search(game, temperature=temperature, add_noise=True)

            # 保存训练数据
            state = game.get_state_for_nn()
            game_data.append((state, action_probs, game.current_player))

            # 选择动作
            if temperature > 0:
                action = np.random.choice(len(action_probs), p=action_probs)
            else:
                action = np.argmax(action_probs)

            # 执行动作
            fr, fc, tr, tc = decode_action(action)
            game.make_move(fr, fc, tr, tc)
            step += 1

            # 检查游戏是否结束
            done, winner = game.is_game_over()
            if done:
                break

        # 确定最终结果
        done, winner = game.is_game_over()
        if not done:
            winner = 0  # 超过最大步数判和

        # 将结果分配给每一步
        training_data = []
        for state, action_probs, player in game_data:
            if winner == 0:
                value = 0.0
            elif winner == player:
                value = 1.0
            else:
                value = -1.0
            training_data.append((state, action_probs, value))

        return training_data, winner, step

    def self_play(self) -> dict:
        """执行一轮自对弈"""
        logger.info(f"开始自对弈: {self.config.num_games_per_iter} 局")

        all_data = []
        results = {1: 0, -1: 0, 0: 0}
        total_steps = 0

        for game_idx in range(self.config.num_games_per_iter):
            start_time = time.time()
            game_data, winner, steps = self.self_play_game()
            elapsed = time.time() - start_time

            # 数据增强
            for state, policy, value in game_data:
                augmented = augment_data(state, policy, value)
                all_data.extend(augmented)

            results[winner] = results.get(winner, 0) + 1
            total_steps += steps
            self.total_games += 1

            logger.info(
                f"  对局 {game_idx + 1}/{self.config.num_games_per_iter}: "
                f"步数={steps}, 赢家={'红' if winner == 1 else '黑' if winner == -1 else '和'}, "
                f"耗时={elapsed:.1f}s"
            )

        # 添加到回放缓冲区
        self.replay_buffer.extend(all_data)

        stats = {
            'games': self.config.num_games_per_iter,
            'red_wins': results.get(1, 0),
            'black_wins': results.get(-1, 0),
            'draws': results.get(0, 0),
            'avg_steps': total_steps / self.config.num_games_per_iter,
            'buffer_size': len(self.replay_buffer),
            'new_samples': len(all_data)
        }

        logger.info(
            f"自对弈完成: 红胜={stats['red_wins']}, 黑胜={stats['black_wins']}, "
            f"和={stats['draws']}, 平均步数={stats['avg_steps']:.1f}, "
            f"新样本={stats['new_samples']}, 缓冲区={stats['buffer_size']}"
        )

        return stats

    def train_network(self) -> dict:
        """训练神经网络"""
        if len(self.replay_buffer) < self.config.min_buffer_size:
            logger.info(f"缓冲区数据不足 ({len(self.replay_buffer)}/{self.config.min_buffer_size})，跳过训练")
            return {}

        logger.info(f"开始训练: {self.config.num_epochs} epochs, 数据量={len(self.replay_buffer)}")

        # 创建数据集
        dataset = SelfPlayDataset(list(self.replay_buffer))
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0,
            drop_last=False
        )

        self.current_model.train()
        total_policy_loss = 0
        total_value_loss = 0
        total_loss = 0
        num_batches = 0

        for epoch in range(self.config.num_epochs):
            epoch_policy_loss = 0
            epoch_value_loss = 0
            epoch_batches = 0

            for states, target_policies, target_values in dataloader:
                states = states.to(self.device)
                target_policies = target_policies.to(self.device)
                target_values = target_values.to(self.device)

                # 前向传播
                policy_logits, pred_values = self.current_model(states)

                # 策略损失（交叉熵）
                policy_loss = -torch.mean(
                    torch.sum(target_policies * F.log_softmax(policy_logits, dim=1), dim=1)
                )

                # 价值损失（MSE）
                value_loss = F.mse_loss(pred_values, target_values)

                # 总损失
                loss = policy_loss + value_loss

                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.current_model.parameters(), 1.0)
                self.optimizer.step()

                epoch_policy_loss += policy_loss.item()
                epoch_value_loss += value_loss.item()
                epoch_batches += 1

            avg_p = epoch_policy_loss / max(epoch_batches, 1)
            avg_v = epoch_value_loss / max(epoch_batches, 1)
            logger.info(f"  Epoch {epoch + 1}: policy_loss={avg_p:.4f}, value_loss={avg_v:.4f}")

            total_policy_loss += epoch_policy_loss
            total_value_loss += epoch_value_loss
            num_batches += epoch_batches

        self.scheduler.step()

        stats = {
            'policy_loss': total_policy_loss / max(num_batches, 1),
            'value_loss': total_value_loss / max(num_batches, 1),
            'total_loss': (total_policy_loss + total_value_loss) / max(num_batches, 1),
            'learning_rate': self.optimizer.param_groups[0]['lr']
        }

        logger.info(
            f"训练完成: policy_loss={stats['policy_loss']:.4f}, "
            f"value_loss={stats['value_loss']:.4f}, lr={stats['learning_rate']:.6f}"
        )

        return stats

    def evaluate(self) -> dict:
        """评估新模型 vs 旧模型"""
        logger.info(f"开始评估: {self.config.eval_games} 局")

        new_mcts = MCTS(
            self.current_model,
            num_simulations=self.config.eval_simulations,
            c_puct=self.config.c_puct,
            device=self.device
        )
        old_mcts = MCTS(
            self.best_model,
            num_simulations=self.config.eval_simulations,
            c_puct=self.config.c_puct,
            device=self.device
        )

        new_wins = 0
        old_wins = 0
        draws = 0

        for game_idx in range(self.config.eval_games):
            game = XiangqiGame()

            # 交替先手
            new_is_red = (game_idx % 2 == 0)

            step = 0
            while step < self.config.max_game_length:
                is_red_turn = (game.current_player == 1)
                if (new_is_red and is_red_turn) or (not new_is_red and not is_red_turn):
                    action = new_mcts.get_action(game, temperature=0, add_noise=False)
                else:
                    action = old_mcts.get_action(game, temperature=0, add_noise=False)

                fr, fc, tr, tc = decode_action(action)
                game.make_move(fr, fc, tr, tc)
                step += 1

                done, winner = game.is_game_over()
                if done:
                    break

            done, winner = game.is_game_over()
            if not done:
                winner = 0

            if winner == 0:
                draws += 1
            elif (winner == 1 and new_is_red) or (winner == -1 and not new_is_red):
                new_wins += 1
            else:
                old_wins += 1

            logger.info(
                f"  评估对局 {game_idx + 1}: "
                f"{'新模型执红' if new_is_red else '新模型执黑'}, "
                f"赢家={'红' if winner == 1 else '黑' if winner == -1 else '和'}, "
                f"步数={step}"
            )

        total = self.config.eval_games
        win_rate = (new_wins + 0.5 * draws) / total

        stats = {
            'new_wins': new_wins,
            'old_wins': old_wins,
            'draws': draws,
            'win_rate': win_rate,
            'model_updated': win_rate >= self.config.eval_win_rate
        }

        logger.info(
            f"评估完成: 新模型胜={new_wins}, 旧模型胜={old_wins}, "
            f"和={draws}, 胜率={win_rate:.2%}"
        )

        # 更新最优模型
        if stats['model_updated']:
            self.best_model.load_state_dict(self.current_model.state_dict())
            logger.info(">>> 最优模型已更新！<<<")
        else:
            self.current_model.load_state_dict(self.best_model.state_dict())
            logger.info("新模型未达标，回退到旧模型")

        return stats

    def save_checkpoint(self, iteration: int, is_best: bool = False):
        """保存检查点"""
        checkpoint = {
            'iteration': iteration,
            'model_state_dict': self.current_model.state_dict(),
            'best_model_state_dict': self.best_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': {
                'num_channels': self.config.num_channels,
                'num_res_blocks': self.config.num_res_blocks,
            },
            'total_games': self.total_games,
        }

        path = os.path.join(self.config.checkpoint_dir, f'checkpoint_iter{iteration}.pt')
        torch.save(checkpoint, path)
        logger.info(f"检查点已保存: {path}")

        if is_best:
            best_path = os.path.join(self.config.checkpoint_dir, 'best_model.pt')
            torch.save({
                'model_state_dict': self.best_model.state_dict(),
                'config': {
                    'num_channels': self.config.num_channels,
                    'num_res_blocks': self.config.num_res_blocks,
                },
                'iteration': iteration,
                'total_games': self.total_games,
            }, best_path)
            logger.info(f"最优模型已保存: {best_path}")

    def load_checkpoint(self, path: str):
        """加载检查点"""
        checkpoint = torch.load(path, map_location=self.device)
        self.current_model.load_state_dict(checkpoint['model_state_dict'])
        self.best_model.load_state_dict(checkpoint['best_model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.iteration = checkpoint['iteration']
        self.total_games = checkpoint.get('total_games', 0)
        logger.info(f"检查点已加载: {path}, 迭代={self.iteration}")

    def train(self):
        """主训练循环"""
        logger.info("=" * 60)
        logger.info("中国象棋 AlphaZero 训练开始")
        logger.info(f"配置: channels={self.config.num_channels}, "
                     f"res_blocks={self.config.num_res_blocks}, "
                     f"simulations={self.config.num_simulations}")
        logger.info("=" * 60)

        for iteration in range(self.iteration + 1, self.config.num_iterations + 1):
            self.iteration = iteration
            logger.info(f"\n{'='*60}")
            logger.info(f"迭代 {iteration}/{self.config.num_iterations}")
            logger.info(f"{'='*60}")

            iter_start = time.time()

            # 1. 自对弈
            sp_stats = self.self_play()

            # 2. 训练网络
            train_stats = self.train_network()

            # 3. 评估（每隔几轮评估一次以节省时间）
            eval_stats = {}
            if iteration % 2 == 0 and len(self.replay_buffer) >= self.config.min_buffer_size:
                eval_stats = self.evaluate()

            # 4. 保存
            if iteration % self.config.save_interval == 0:
                is_best = eval_stats.get('model_updated', False)
                self.save_checkpoint(iteration, is_best=True)

            iter_time = time.time() - iter_start

            # 记录统计
            stats = {
                'iteration': iteration,
                'time': iter_time,
                'self_play': sp_stats,
                'training': train_stats,
                'evaluation': eval_stats,
            }
            self.training_stats.append(stats)

            logger.info(f"迭代 {iteration} 完成, 耗时: {iter_time:.1f}s")

            # 保存训练统计
            stats_path = os.path.join(self.config.checkpoint_dir, 'training_stats.json')
            with open(stats_path, 'w') as f:
                json.dump(self.training_stats, f, indent=2, default=str)

        # 保存最终模型
        self.save_checkpoint(self.iteration, is_best=True)
        logger.info("\n训练完成！")


def quick_train():
    """快速训练模式（用于测试和演示）"""
    config = TrainingConfig()
    config.num_channels = 64
    config.num_res_blocks = 3
    config.num_simulations = 50
    config.num_games_per_iter = 5
    config.num_iterations = 10
    config.batch_size = 64
    config.num_epochs = 3
    config.min_buffer_size = 200
    config.eval_games = 4
    config.eval_simulations = 30
    config.save_interval = 2
    config.temperature_threshold = 10
    config.max_game_length = 100
    return config


def main():
    parser = argparse.ArgumentParser(description='中国象棋 AlphaZero 训练')
    parser.add_argument('--mode', type=str, default='quick',
                        choices=['quick', 'standard', 'full'],
                        help='训练模式: quick(快速测试), standard(标准), full(完整)')
    parser.add_argument('--iterations', type=int, default=None, help='训练轮数')
    parser.add_argument('--games-per-iter', type=int, default=None, help='每轮自对弈局数')
    parser.add_argument('--simulations', type=int, default=None, help='MCTS模拟次数')
    parser.add_argument('--channels', type=int, default=None, help='网络通道数')
    parser.add_argument('--res-blocks', type=int, default=None, help='残差块数量')
    parser.add_argument('--resume', type=str, default=None, help='从检查点恢复训练')
    parser.add_argument('--device', type=str, default=None, help='计算设备 (cpu/cuda)')

    args = parser.parse_args()

    # 选择配置
    if args.mode == 'quick':
        config = quick_train()
    elif args.mode == 'standard':
        config = TrainingConfig()
        config.num_channels = 128
        config.num_res_blocks = 6
        config.num_simulations = 200
        config.num_games_per_iter = 20
        config.num_iterations = 50
    else:  # full
        config = TrainingConfig()
        config.num_channels = 256
        config.num_res_blocks = 10
        config.num_simulations = 400
        config.num_games_per_iter = 50
        config.num_iterations = 200

    # 覆盖命令行参数
    if args.iterations:
        config.num_iterations = args.iterations
    if args.games_per_iter:
        config.num_games_per_iter = args.games_per_iter
    if args.simulations:
        config.num_simulations = args.simulations
    if args.channels:
        config.num_channels = args.channels
    if args.res_blocks:
        config.num_res_blocks = args.res_blocks
    if args.device:
        config.device = args.device

    # 创建训练器
    trainer = AlphaZeroTrainer(config)

    # 恢复训练
    if args.resume:
        trainer.load_checkpoint(args.resume)

    # 开始训练
    trainer.train()


if __name__ == "__main__":
    main()
