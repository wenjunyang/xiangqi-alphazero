"""
多进程并行自对弈
================
使用 Python multiprocessing 实现多进程并行自对弈，充分利用多核 CPU。

设计思路：
- 主进程将模型权重序列化后通过共享内存传递给子进程
- 每个子进程独立创建模型实例并加载权重，独立执行自对弈
- 使用 ProcessPoolExecutor 管理进程池
- 每个子进程限制 PyTorch 为单线程，避免线程竞争

关键优化：
1. 使用 'spawn' 上下文创建进程，避免 fork 导致的锁问题
2. 每个 worker 限制 PyTorch 单线程 intra-op，避免 CPU 过载
3. 模型权重通过 state_dict 传递，避免序列化整个模型对象
4. 支持自动检测 CPU 核心数，动态调整 worker 数量

版本 v3: 修复 set_num_interop_threads 在 spawn 后报错的问题
"""

import os
import sys
import time
import random
import logging
from typing import List, Tuple, Dict, Any, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import torch
import torch.multiprocessing as mp

# 必须在 import 之前设置，确保子进程继承
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

from game import XiangqiGame, ACTION_SPACE, decode_action, encode_action
from model import XiangqiNet
from mcts import MCTS

logger = logging.getLogger(__name__)


def _worker_init():
    """Worker 进程初始化：设置随机种子"""
    # 通过环境变量控制线程数（已在模块级设置）
    # 只在这里安全地设置 intra-op 线程
    try:
        torch.set_num_threads(1)
    except Exception:
        pass
    # 每个 worker 使用独立的随机种子
    seed = int.from_bytes(os.urandom(4), 'big')
    random.seed(seed)
    np.random.seed(seed % (2**31))
    torch.manual_seed(seed)


def _make_random_opening(game: XiangqiGame, num_moves: int) -> XiangqiGame:
    """执行随机开局走法"""
    for _ in range(num_moves):
        legal_moves = game.get_legal_moves()
        if not legal_moves:
            break
        move = random.choice(legal_moves)
        game.make_move(*move)
        done, _ = game.is_game_over()
        if done:
            break
    return game


def _play_one_game(args: dict) -> Tuple[List[Tuple[np.ndarray, np.ndarray, float]], int, int]:
    """
    在子进程中执行一局自对弈。

    这是 worker 的核心函数，每次调用完成一局完整对弈。
    模型在函数内部创建和加载，避免跨进程传递模型对象。

    返回:
        (training_data, winner, steps)
    """
    _worker_init()

    model_state_dict = args['model_state_dict']
    num_channels = args['num_channels']
    num_res_blocks = args['num_res_blocks']
    num_simulations = args['num_simulations']
    c_puct = args['c_puct']
    temperature_threshold = args['temperature_threshold']
    max_game_length = args['max_game_length']
    random_opening_moves = args['random_opening_moves']
    enable_resign = args['enable_resign']
    resign_threshold = args['resign_threshold']
    resign_check_steps = args['resign_check_steps']

    try:
        # 在子进程中重建模型并加载权重
        device = 'cpu'
        model = XiangqiNet(num_channels=num_channels, num_res_blocks=num_res_blocks)
        model.load_state_dict(model_state_dict)
        model.to(device)
        model.eval()

        # 创建游戏和 MCTS
        game = XiangqiGame()
        if random_opening_moves > 0:
            num_random = random.randint(0, random_opening_moves)
            game = _make_random_opening(game, num_random)

        mcts = MCTS(model, num_simulations=num_simulations, c_puct=c_puct, device=device)

        game_data = []
        step = 0
        resign_counter = 0
        done = False
        winner = None

        while step < max_game_length:
            # 温度调度
            if step < temperature_threshold:
                temperature = 1.0
            elif step < temperature_threshold + 10:
                temperature = 1.0 - 0.9 * (step - temperature_threshold) / 10
            else:
                temperature = 0.1

            # MCTS 搜索
            action_probs = mcts.search(game, temperature=temperature, add_noise=True)

            # 保存训练数据
            state = game.get_state_for_nn()
            game_data.append((state, action_probs, game.current_player))

            # 选择动作
            if temperature > 0.05:
                action = np.random.choice(len(action_probs), p=action_probs)
            else:
                action = np.argmax(action_probs)

            # 执行动作
            fr, fc, tr, tc = decode_action(action)
            game.make_move(fr, fc, tr, tc)
            step += 1

            # 检查游戏结束
            done, winner = game.is_game_over()
            if done:
                break

            # 认输机制
            if enable_resign and step > 40:
                eval_state = game.get_state_for_nn()
                _, value = model.predict(eval_state, device)
                if value < resign_threshold:
                    resign_counter += 1
                else:
                    resign_counter = 0
                if resign_counter >= resign_check_steps:
                    winner = -game.current_player
                    done = True
                    break

        # 确定最终结果
        if not done:
            done, winner = game.is_game_over()
        if winner is None:
            winner = 0

        # 分配奖励并做数据增强
        training_data = []
        for state, action_probs, player in game_data:
            if winner == 0:
                value = 0.0
            elif winner == player:
                value = 1.0
            else:
                value = -1.0

            # 原始数据
            training_data.append((state, action_probs, value))

            # 水平翻转增强
            flipped_state = np.flip(state, axis=2).copy()
            flipped_policy = np.zeros_like(action_probs)
            for action_idx in range(ACTION_SPACE):
                if action_probs[action_idx] > 0:
                    afr, afc, atr, atc = decode_action(action_idx)
                    new_action = encode_action(afr, 8 - afc, atr, 8 - atc)
                    flipped_policy[new_action] = action_probs[action_idx]
            training_data.append((flipped_state, flipped_policy, value))

        return training_data, winner, step

    except Exception as e:
        import traceback
        traceback.print_exc()
        return [], 0, 0


def parallel_self_play(
    model: XiangqiNet,
    config,
    num_workers: Optional[int] = None,
) -> Tuple[List[Tuple[np.ndarray, np.ndarray, float]], Dict[str, Any]]:
    """
    多进程并行自对弈。

    参数:
        model: 当前最优模型（主进程中的模型实例）
        config: TrainingConfig 配置对象
        num_workers: 并行 worker 数量，None 则自动检测

    返回:
        (all_training_data, stats)
    """
    if num_workers is None:
        cpu_count = os.cpu_count() or 4
        num_workers = min(cpu_count, config.num_games_per_iter)
        if cpu_count > 2:
            num_workers = min(num_workers, cpu_count - 1)

    num_games = config.num_games_per_iter
    logger.info(f"开始并行自对弈: {num_games} 局, {num_workers} 个 worker 进程")

    # 序列化模型权重（只做一次）
    model_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # 构建每局对弈的参数
    game_args_list = []
    for i in range(num_games):
        game_args_list.append({
            'model_state_dict': model_state_dict,
            'num_channels': model.num_channels,
            'num_res_blocks': model.num_res_blocks,
            'num_simulations': config.num_simulations,
            'c_puct': config.c_puct,
            'temperature_threshold': config.temperature_threshold,
            'max_game_length': config.max_game_length,
            'random_opening_moves': config.random_opening_moves,
            'enable_resign': config.enable_resign,
            'resign_threshold': config.resign_threshold,
            'resign_check_steps': config.resign_check_steps,
            'game_index': i,
        })

    start_time = time.time()

    # 使用 spawn 上下文的 ProcessPoolExecutor
    ctx = mp.get_context('spawn')
    all_data = []
    win_counts = {1: 0, -1: 0, 0: 0}
    total_steps = 0
    valid_games = 0

    with ProcessPoolExecutor(max_workers=num_workers, mp_context=ctx) as executor:
        futures = {
            executor.submit(_play_one_game, args): i
            for i, args in enumerate(game_args_list)
        }

        for future in as_completed(futures):
            game_idx = futures[future]
            try:
                game_data, winner, steps = future.result()
                if game_data:
                    valid_games += 1
                    win_counts[winner] = win_counts.get(winner, 0) + 1
                    total_steps += steps
                    all_data.extend(game_data)

                    winner_str = '红' if winner == 1 else '黑' if winner == -1 else '和'
                    logger.info(f"  对局 {valid_games}/{num_games} 完成: 步数={steps}, 赢家={winner_str}")
                else:
                    logger.warning(f"  对局 {game_idx} 失败，跳过")
            except Exception as e:
                logger.error(f"  对局 {game_idx} 异常: {e}")

    elapsed = time.time() - start_time
    avg_steps = total_steps / max(valid_games, 1)

    # 计算加速比：假设串行时每局耗时 = 总时间/并行局数 * 并行度
    avg_time_per_game_parallel = elapsed / max(valid_games, 1)

    stats = {
        'games': valid_games,
        'red_wins': win_counts.get(1, 0),
        'black_wins': win_counts.get(-1, 0),
        'draws': win_counts.get(0, 0),
        'avg_steps': avg_steps,
        'new_samples': len(all_data),
        'total_time': elapsed,
        'num_workers': num_workers,
    }

    logger.info(
        f"并行自对弈完成 ({elapsed:.1f}s): "
        f"红胜={stats['red_wins']}, 黑胜={stats['black_wins']}, 和={stats['draws']}, "
        f"平均步数={avg_steps:.1f}, 新样本={len(all_data)}, "
        f"{num_workers} workers"
    )

    return all_data, stats


def _eval_one_game(args: dict) -> Tuple[str, int, bool]:
    """在子进程中执行一局评估对弈"""
    _worker_init()

    try:
        device = 'cpu'

        new_model = XiangqiNet(
            num_channels=args['num_channels'],
            num_res_blocks=args['num_res_blocks']
        )
        new_model.load_state_dict(args['new_state_dict'])
        new_model.to(device)
        new_model.eval()

        old_model = XiangqiNet(
            num_channels=args['num_channels'],
            num_res_blocks=args['num_res_blocks']
        )
        old_model.load_state_dict(args['old_state_dict'])
        old_model.to(device)
        old_model.eval()

        new_mcts = MCTS(new_model, num_simulations=args['num_simulations'],
                        c_puct=args['c_puct'], device=device)
        old_mcts = MCTS(old_model, num_simulations=args['num_simulations'],
                        c_puct=args['c_puct'], device=device)

        game = XiangqiGame()
        new_is_red = args['new_is_red']
        step = 0

        while step < args['max_game_length']:
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
            return 'draw', step, new_is_red
        elif (winner == 1 and new_is_red) or (winner == -1 and not new_is_red):
            return 'new', step, new_is_red
        else:
            return 'old', step, new_is_red

    except Exception as e:
        import traceback
        traceback.print_exc()
        return 'draw', 0, args.get('new_is_red', True)


def parallel_evaluate(
    new_model: XiangqiNet,
    old_model: XiangqiNet,
    config,
    num_workers: Optional[int] = None,
) -> Dict[str, Any]:
    """
    多进程并行评估：新模型 vs 旧模型。
    """
    if num_workers is None:
        cpu_count = os.cpu_count() or 4
        num_workers = min(cpu_count - 1, config.eval_games)
        num_workers = max(num_workers, 1)

    logger.info(f"开始并行评估: {config.eval_games} 局, {num_workers} 个 worker 进程")

    new_state_dict = {k: v.cpu().clone() for k, v in new_model.state_dict().items()}
    old_state_dict = {k: v.cpu().clone() for k, v in old_model.state_dict().items()}

    eval_args_list = []
    for i in range(config.eval_games):
        eval_args_list.append({
            'new_state_dict': new_state_dict,
            'old_state_dict': old_state_dict,
            'num_channels': new_model.num_channels,
            'num_res_blocks': new_model.num_res_blocks,
            'num_simulations': config.eval_simulations,
            'c_puct': config.c_puct,
            'max_game_length': config.max_game_length,
            'new_is_red': (i % 2 == 0),
            'game_index': i,
        })

    start_time = time.time()

    ctx = mp.get_context('spawn')
    new_wins = 0
    old_wins = 0
    draws = 0

    with ProcessPoolExecutor(max_workers=num_workers, mp_context=ctx) as executor:
        futures = {
            executor.submit(_eval_one_game, args): i
            for i, args in enumerate(eval_args_list)
        }

        for future in as_completed(futures):
            try:
                winner_side, steps, new_is_red = future.result()
                if winner_side == 'new':
                    new_wins += 1
                elif winner_side == 'old':
                    old_wins += 1
                else:
                    draws += 1
            except Exception as e:
                logger.error(f"评估对局异常: {e}")
                draws += 1

    elapsed = time.time() - start_time

    total = config.eval_games
    win_rate = (new_wins + 0.5 * draws) / max(total, 1)

    stats = {
        'new_wins': new_wins,
        'old_wins': old_wins,
        'draws': draws,
        'win_rate': win_rate,
        'model_updated': win_rate >= config.eval_win_rate,
        'eval_time': elapsed,
    }

    logger.info(
        f"并行评估完成 ({elapsed:.1f}s): "
        f"新模型胜={new_wins}, 旧模型胜={old_wins}, 和={draws}, 胜率={win_rate:.2%}"
    )

    return stats


if __name__ == "__main__":
    """测试并行自对弈性能"""
    import sys
    sys.path.insert(0, os.path.dirname(__file__))

    from train import TrainingConfig

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
    )

    config = TrainingConfig()
    config.num_channels = 64
    config.num_res_blocks = 3
    config.num_simulations = 30
    config.max_game_length = 100
    config.random_opening_moves = 4
    config.enable_resign = False
    config.temperature_threshold = 10

    model = XiangqiNet(num_channels=64, num_res_blocks=3)

    # 测试串行 (2局)
    print("=" * 60)
    print("测试串行自对弈 (2局)...")
    serial_start = time.time()
    for i in range(2):
        r = _play_one_game({
            'model_state_dict': {k: v.cpu().clone() for k, v in model.state_dict().items()},
            'num_channels': 64, 'num_res_blocks': 3,
            'num_simulations': 30, 'c_puct': 1.5,
            'temperature_threshold': 10, 'max_game_length': 100,
            'random_opening_moves': 4, 'enable_resign': False,
            'resign_threshold': -0.85, 'resign_check_steps': 3,
            'game_index': i,
        })
    serial_time = time.time() - serial_start
    print(f"串行 2 局耗时: {serial_time:.1f}s, 平均 {serial_time/2:.1f}s/局")

    # 测试并行 (6局)
    print("=" * 60)
    print("测试并行自对弈 (6局, 5 workers)...")
    config.num_games_per_iter = 6
    all_data, stats = parallel_self_play(model, config, num_workers=5)
    print(f"并行 6 局耗时: {stats['total_time']:.1f}s")
    print(f"新样本数: {stats['new_samples']}")
    print(f"结果: 红胜={stats['red_wins']}, 黑胜={stats['black_wins']}, 和={stats['draws']}")

    # 计算加速比
    serial_per_game = serial_time / 2
    expected_serial = serial_per_game * 6
    actual_parallel = stats['total_time']
    print(f"\n预估串行 6 局耗时: {expected_serial:.1f}s")
    print(f"实际并行 6 局耗时: {actual_parallel:.1f}s")
    print(f"加速比: {expected_serial / actual_parallel:.2f}x")
