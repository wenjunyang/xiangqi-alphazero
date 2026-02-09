"""
多进程并行自对弈
================
使用 ProcessPoolExecutor 将多局自对弈分配到多个进程并行执行。
每个 worker 进程独立重建模型，串行完成分配的对局。

架构简洁：
  主进程 → 序列化模型权重 → 分发到 N 个 worker 进程
  每个 worker → 重建模型 → 串行完成 K 局自对弈 → 返回训练数据
  主进程 → 汇总所有数据

并行度 = worker 数 ≈ CPU 核心数
"""

import os
import time
import random
import logging
import math
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import torch
import torch.multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

from game import XiangqiGame, ACTION_SPACE, decode_action, encode_action
from model import XiangqiNet
from mcts import MCTS

logger = logging.getLogger(__name__)


def _play_one_game(model, config, device='cpu') -> Tuple[List, int, int]:
    """
    执行一局自对弈，返回 (training_data, winner, steps)。

    training_data: [(state, action_probs, value), ...]
    winner: 1=红胜, -1=黑胜, 0=和棋
    steps: 总步数
    """
    mcts = MCTS(
        model,
        num_simulations=config.num_simulations,
        c_puct=config.c_puct,
        device=device,
    )

    game = XiangqiGame()
    training_data = []
    resign_value_history = []

    # 随机开局
    random_moves = random.randint(0, config.random_opening_moves)
    for _ in range(random_moves):
        moves = game.get_legal_moves()
        if not moves:
            break
        game.make_move(*random.choice(moves))
        done, _ = game.is_game_over()
        if done:
            game = XiangqiGame()
            break

    while True:
        done, winner = game.is_game_over()
        if done:
            break

        if game.move_count >= config.max_game_length:
            # 超步数：材料判定
            red_score = game.get_material_score(1)
            black_score = game.get_material_score(-1)
            diff = red_score - black_score
            if diff > 30:
                winner = 1
            elif diff < -30:
                winner = -1
            else:
                winner = 0
            break

        # 温度控制
        temperature = 1.0 if game.move_count < config.temperature_threshold else 0.3

        # MCTS 搜索
        action_probs = mcts.search(game, temperature=temperature, add_noise=True)

        # 保存训练数据
        state = game.get_state_for_nn()
        training_data.append((state, action_probs, game.current_player))

        # 选择动作
        if temperature == 0:
            action = int(np.argmax(action_probs))
        else:
            action = int(np.random.choice(len(action_probs), p=action_probs))

        game.make_action(action)

        # 认输检测
        if config.enable_resign and len(training_data) > 10:
            _, value = model.predict(game.get_state_for_nn(), device)
            resign_value_history.append(value)
            if len(resign_value_history) >= config.resign_check_steps:
                recent = resign_value_history[-config.resign_check_steps:]
                if all(v < config.resign_threshold for v in recent):
                    winner = -game.current_player
                    break

    # 填充胜负标签
    final_data = []
    for state, action_probs, player in training_data:
        if winner == 0:
            value = 0.0
        elif winner == player:
            value = 1.0
        else:
            value = -1.0
        final_data.append((state, action_probs, value))

    return final_data, winner, game.move_count


def _worker_entry(args: dict) -> List[Tuple[List, int, int]]:
    """
    Worker 进程入口：串行完成分配的多局对弈。
    """
    # 限制每个进程的线程数，避免 N 进程 × M 线程导致 CPU 过载
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    try:
        torch.set_num_threads(1)
    except Exception:
        pass

    # 随机种子
    seed = int.from_bytes(os.urandom(4), 'big')
    random.seed(seed)
    np.random.seed(seed % (2**31))
    torch.manual_seed(seed)

    try:
        # 重建模型
        model = XiangqiNet(
            num_channels=args['num_channels'],
            num_res_blocks=args['num_res_blocks'],
        )
        model.load_state_dict(args['model_state_dict'])
        model.eval()

        # 构建 config 对象
        class WorkerConfig:
            pass

        cfg = WorkerConfig()
        for key in ['num_simulations', 'c_puct', 'temperature_threshold',
                     'max_game_length', 'random_opening_moves',
                     'enable_resign', 'resign_threshold', 'resign_check_steps']:
            setattr(cfg, key, args[key])

        device = 'cpu'
        num_games = args['num_games']
        results = []

        for _ in range(num_games):
            data, winner, steps = _play_one_game(model, cfg, device)

            # 数据增强：水平翻转
            aug_data = []
            for state, action_probs, value in data:
                aug_data.append((state, action_probs, value))
                # 翻转
                flipped_state = np.flip(state, axis=2).copy()
                flipped_policy = np.zeros_like(action_probs)
                for action_idx in range(ACTION_SPACE):
                    if action_probs[action_idx] > 0:
                        fr, fc, tr, tc = decode_action(action_idx)
                        new_action = encode_action(fr, 8 - fc, tr, 8 - tc)
                        flipped_policy[new_action] = action_probs[action_idx]
                aug_data.append((flipped_state, flipped_policy, value))

            results.append((aug_data, winner, steps))

        return results

    except Exception as e:
        import traceback
        traceback.print_exc()
        return []


def parallel_self_play(
    model: XiangqiNet,
    config,
    num_workers: Optional[int] = None,
) -> Tuple[List[Tuple[np.ndarray, np.ndarray, float]], Dict[str, Any]]:
    """
    多进程并行自对弈。

    参数:
        model: 当前最优模型
        config: TrainingConfig
        num_workers: 进程数（None=自动，使用 CPU 核心数 - 1）

    返回:
        (all_training_data, stats)
    """
    total_cores = os.cpu_count() or 4
    num_games = config.num_games_per_iter

    # 自动计算 worker 数
    if num_workers is None:
        num_workers = max(1, total_cores - 1)

    # worker 数不超过对局数
    num_workers = min(num_workers, num_games)

    # 均匀分配对局
    games_per_worker = num_games // num_workers
    remainder = num_games % num_workers

    logger.info(
        f"开始并行自对弈: {num_games} 局, "
        f"{num_workers} workers, "
        f"~{games_per_worker} 局/worker"
    )

    # 序列化模型权重
    model_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # 构建每个 worker 的参数
    worker_args_list = []
    for w in range(num_workers):
        this_games = games_per_worker + (1 if w < remainder else 0)
        if this_games <= 0:
            continue
        worker_args_list.append({
            'model_state_dict': model_state_dict,
            'num_channels': model.num_channels,
            'num_res_blocks': model.num_res_blocks,
            'num_games': this_games,
            'num_simulations': config.num_simulations,
            'c_puct': config.c_puct,
            'temperature_threshold': config.temperature_threshold,
            'max_game_length': config.max_game_length,
            'random_opening_moves': config.random_opening_moves,
            'enable_resign': config.enable_resign,
            'resign_threshold': config.resign_threshold,
            'resign_check_steps': config.resign_check_steps,
        })

    start_time = time.time()

    # 启动进程池
    ctx = mp.get_context('spawn')
    all_data = []
    win_counts = {1: 0, -1: 0, 0: 0}
    total_steps = 0
    valid_games = 0

    with ProcessPoolExecutor(max_workers=len(worker_args_list), mp_context=ctx) as executor:
        futures = {
            executor.submit(_worker_entry, args): i
            for i, args in enumerate(worker_args_list)
        }

        for future in as_completed(futures):
            worker_idx = futures[future]
            try:
                worker_results = future.result()
                for aug_data, winner, steps in worker_results:
                    if aug_data:
                        valid_games += 1
                        win_counts[winner] = win_counts.get(winner, 0) + 1
                        total_steps += steps
                        all_data.extend(aug_data)

                        winner_str = '红' if winner == 1 else '黑' if winner == -1 else '和'
                        logger.info(f"  对局 {valid_games}/{num_games}: 步数={steps}, 赢家={winner_str}")
            except Exception as e:
                logger.error(f"  Worker {worker_idx} 异常: {e}")

    elapsed = time.time() - start_time
    avg_steps = total_steps / max(valid_games, 1)

    stats = {
        'games': valid_games,
        'red_wins': win_counts.get(1, 0),
        'black_wins': win_counts.get(-1, 0),
        'draws': win_counts.get(0, 0),
        'avg_steps': avg_steps,
        'new_samples': len(all_data),
        'total_time': elapsed,
        'num_workers': len(worker_args_list),
    }

    logger.info(
        f"自对弈完成: {valid_games} 局, {elapsed:.1f}s, "
        f"红{stats['red_wins']}黑{stats['black_wins']}和{stats['draws']}, "
        f"平均{avg_steps:.0f}步, {len(all_data)} 样本"
    )

    return all_data, stats
