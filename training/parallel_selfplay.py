"""
多级并行自对弈
================
三级并行架构，专为高核数机器（如 128 核）设计：

  Level 1 - 进程级并行（多进程）
  │  使用 ProcessPoolExecutor 创建 P 个 worker 进程
  │
  Level 2 - 对局级并行（多对局批量 MCTS）
  │  每个 worker 内同时推进 G 局对弈
  │  所有对局的 MCTS 叶节点合并做批量推理
  │
  Level 3 - 搜索级并行（Virtual Loss）
     每局 MCTS 内使用 Virtual Loss 并行搜索 V 条路径
     不同路径被分散到不同分支，增加搜索多样性

  总并行度 = P × G × V

  例如 128 核机器，推荐配置：
  - P = 16 个进程（每进程允许 8 线程做 PyTorch 矩阵运算）
  - G = 4 局/进程
  - V = 8 条 Virtual Loss 路径
  - 每次批量推理 batch = G × V = 32 个状态
  - 总对局并行度 = 16 × 4 = 64 局同时进行

版本 v4: 三级并行架构
"""

import os
import sys
import time
import random
import logging
import math
from typing import List, Tuple, Dict, Any, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import torch
import torch.multiprocessing as mp

from game import XiangqiGame, ACTION_SPACE, decode_action, encode_action
from model import XiangqiNet
from mcts import MCTS, BatchMCTS, MultiGameBatchMCTS

logger = logging.getLogger(__name__)


def _worker_play_games(args: dict) -> List[Tuple[List, int, int]]:
    """
    Worker 进程入口：在一个进程中同时执行多局对弈。

    使用 MultiGameBatchMCTS 实现对局间 + 对局内的双重并行。
    """
    # 设置线程数：每个进程允许多线程做矩阵运算
    threads_per_worker = args.get('threads_per_worker', 1)
    os.environ['OMP_NUM_THREADS'] = str(threads_per_worker)
    os.environ['MKL_NUM_THREADS'] = str(threads_per_worker)
    try:
        torch.set_num_threads(threads_per_worker)
    except Exception:
        pass

    # 随机种子
    seed = int.from_bytes(os.urandom(4), 'big')
    random.seed(seed)
    np.random.seed(seed % (2**31))
    torch.manual_seed(seed)

    try:
        model_state_dict = args['model_state_dict']
        num_channels = args['num_channels']
        num_res_blocks = args['num_res_blocks']
        num_games = args['games_per_worker']
        num_simulations = args['num_simulations']
        c_puct = args['c_puct']
        vl_batch_size = args['vl_batch_size']
        device = 'cpu'

        # 重建模型
        model = XiangqiNet(num_channels=num_channels, num_res_blocks=num_res_blocks)
        model.load_state_dict(model_state_dict)
        model.to(device)
        model.eval()

        # 构建 config-like 对象
        class WorkerConfig:
            pass

        cfg = WorkerConfig()
        cfg.temperature_threshold = args['temperature_threshold']
        cfg.max_game_length = args['max_game_length']
        cfg.random_opening_moves = args['random_opening_moves']
        cfg.enable_resign = args['enable_resign']
        cfg.resign_threshold = args['resign_threshold']
        cfg.resign_check_steps = args['resign_check_steps']

        # 使用多对局批量 MCTS
        multi_mcts = MultiGameBatchMCTS(
            model,
            num_simulations=num_simulations,
            c_puct=c_puct,
            device=device,
            vl_batch_size=vl_batch_size,
        )

        results = multi_mcts.play_games(num_games, cfg)

        # 数据增强
        augmented_results = []
        for training_data, winner, steps in results:
            aug_data = []
            for state, action_probs, value in training_data:
                aug_data.append((state, action_probs, value))
                # 水平翻转
                flipped_state = np.flip(state, axis=2).copy()
                flipped_policy = np.zeros_like(action_probs)
                for action_idx in range(ACTION_SPACE):
                    if action_probs[action_idx] > 0:
                        fr, fc, tr, tc = decode_action(action_idx)
                        new_action = encode_action(fr, 8 - fc, tr, 8 - tc)
                        flipped_policy[new_action] = action_probs[action_idx]
                aug_data.append((flipped_state, flipped_policy, value))
            augmented_results.append((aug_data, winner, steps))

        return augmented_results

    except Exception as e:
        import traceback
        traceback.print_exc()
        return []


def compute_parallel_config(
    total_games: int,
    total_cores: int,
    num_simulations: int,
) -> dict:
    """
    根据机器核心数和对局数自动计算最优并行配置。

    策略：
    - 每个进程分配 threads_per_worker 个线程用于 PyTorch 矩阵运算
    - 进程数 = total_cores / threads_per_worker
    - 每个进程内同时推进 games_per_worker 局
    - Virtual Loss 批大小根据模拟次数调整

    参数:
        total_games: 总对局数
        total_cores: CPU 核心数
        num_simulations: MCTS 模拟次数

    返回:
        配置字典
    """
    # 每个进程的线程数：核数多时给每个进程更多线程做矩阵运算
    if total_cores >= 64:
        threads_per_worker = 4
    elif total_cores >= 16:
        threads_per_worker = 2
    else:
        threads_per_worker = 1

    # 进程数
    num_workers = max(1, total_cores // threads_per_worker)
    # 留 1-2 个核心给系统
    if total_cores > 4:
        num_workers = max(1, num_workers - 1)

    # 每个进程的对局数：尽量均匀分配
    if total_games <= num_workers:
        # 对局数少于进程数，减少进程数
        num_workers = total_games
        games_per_worker = 1
    else:
        games_per_worker = math.ceil(total_games / num_workers)
        # 调整进程数使总对局数匹配
        num_workers = math.ceil(total_games / games_per_worker)

    # Virtual Loss 批大小
    if num_simulations >= 400:
        vl_batch_size = 16
    elif num_simulations >= 200:
        vl_batch_size = 8
    else:
        vl_batch_size = 4

    return {
        'num_workers': num_workers,
        'games_per_worker': games_per_worker,
        'threads_per_worker': threads_per_worker,
        'vl_batch_size': vl_batch_size,
        'total_parallelism': num_workers * games_per_worker * vl_batch_size,
    }


def parallel_self_play(
    model: XiangqiNet,
    config,
    num_workers: Optional[int] = None,
) -> Tuple[List[Tuple[np.ndarray, np.ndarray, float]], Dict[str, Any]]:
    """
    三级并行自对弈。

    参数:
        model: 当前最优模型
        config: TrainingConfig
        num_workers: 进程数覆盖（None=自动计算）

    返回:
        (all_training_data, stats)
    """
    total_cores = os.cpu_count() or 4
    num_games = config.num_games_per_iter

    # 计算并行配置
    par_config = compute_parallel_config(
        total_games=num_games,
        total_cores=total_cores,
        num_simulations=config.num_simulations,
    )

    if num_workers is not None:
        par_config['num_workers'] = num_workers
        par_config['games_per_worker'] = math.ceil(num_games / num_workers)

    # 支持从 config 覆盖并行参数
    if getattr(config, 'vl_batch_size', None) is not None:
        par_config['vl_batch_size'] = config.vl_batch_size
    if getattr(config, 'games_per_worker', None) is not None:
        par_config['games_per_worker'] = config.games_per_worker
        par_config['num_workers'] = math.ceil(num_games / config.games_per_worker)

    actual_workers = par_config['num_workers']
    games_per_worker = par_config['games_per_worker']
    threads_per_worker = par_config['threads_per_worker']
    vl_batch_size = par_config['vl_batch_size']

    logger.info(
        f"开始三级并行自对弈: {num_games} 局\n"
        f"  Level 1 - 进程: {actual_workers} workers × {threads_per_worker} threads\n"
        f"  Level 2 - 对局: {games_per_worker} 局/worker\n"
        f"  Level 3 - VL搜索: batch={vl_batch_size}\n"
        f"  总并行度: ~{par_config['total_parallelism']}"
    )

    # 序列化模型权重
    model_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # 分配对局到各 worker
    worker_args_list = []
    games_assigned = 0
    for w in range(actual_workers):
        remaining = num_games - games_assigned
        if remaining <= 0:
            break
        this_worker_games = min(games_per_worker, remaining)
        games_assigned += this_worker_games

        worker_args_list.append({
            'model_state_dict': model_state_dict,
            'num_channels': model.num_channels,
            'num_res_blocks': model.num_res_blocks,
            'games_per_worker': this_worker_games,
            'num_simulations': config.num_simulations,
            'c_puct': config.c_puct,
            'vl_batch_size': vl_batch_size,
            'threads_per_worker': threads_per_worker,
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
            executor.submit(_worker_play_games, args): i
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
        'num_workers': actual_workers,
        'games_per_worker': games_per_worker,
        'vl_batch_size': vl_batch_size,
        'threads_per_worker': threads_per_worker,
    }

    logger.info(
        f"并行自对弈完成 ({elapsed:.1f}s): "
        f"红胜={stats['red_wins']}, 黑胜={stats['black_wins']}, 和={stats['draws']}, "
        f"平均步数={avg_steps:.1f}, 新样本={len(all_data)}"
    )

    return all_data, stats


def parallel_evaluate(
    new_model: XiangqiNet,
    old_model: XiangqiNet,
    config,
    num_workers: Optional[int] = None,
) -> Dict[str, Any]:
    """
    多进程并行评估：新模型 vs 旧模型。
    评估对弈使用串行 MCTS（保证确定性），但对局间并行。
    """
    total_cores = os.cpu_count() or 4
    if num_workers is None:
        num_workers = min(total_cores - 1, config.eval_games)
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


def _eval_one_game(args: dict) -> Tuple[str, int, bool]:
    """评估 worker"""
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    try:
        torch.set_num_threads(1)
    except Exception:
        pass

    seed = int.from_bytes(os.urandom(4), 'big')
    random.seed(seed)
    np.random.seed(seed % (2**31))

    try:
        device = 'cpu'
        new_model = XiangqiNet(num_channels=args['num_channels'], num_res_blocks=args['num_res_blocks'])
        new_model.load_state_dict(args['new_state_dict'])
        new_model.to(device)
        new_model.eval()

        old_model = XiangqiNet(num_channels=args['num_channels'], num_res_blocks=args['num_res_blocks'])
        old_model.load_state_dict(args['old_state_dict'])
        old_model.to(device)
        old_model.eval()

        new_mcts = MCTS(new_model, num_simulations=args['num_simulations'], c_puct=args['c_puct'], device=device)
        old_mcts = MCTS(old_model, num_simulations=args['num_simulations'], c_puct=args['c_puct'], device=device)

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


if __name__ == "__main__":
    """测试三级并行性能"""
    import sys
    sys.path.insert(0, os.path.dirname(__file__))

    from train import TrainingConfig

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
    )

    total_cores = os.cpu_count() or 1
    print(f"CPU 核心数: {total_cores}")

    config = TrainingConfig()
    config.num_channels = 64
    config.num_res_blocks = 3
    config.num_simulations = 30
    config.num_games_per_iter = 6
    config.max_game_length = 100
    config.random_opening_moves = 4
    config.enable_resign = False
    config.temperature_threshold = 10

    model = XiangqiNet(num_channels=64, num_res_blocks=3)

    # 显示自动计算的并行配置
    par_config = compute_parallel_config(
        total_games=config.num_games_per_iter,
        total_cores=total_cores,
        num_simulations=config.num_simulations,
    )
    print(f"\n自动并行配置:")
    for k, v in par_config.items():
        print(f"  {k}: {v}")

    # 测试并行
    print(f"\n{'='*60}")
    print(f"测试三级并行自对弈 ({config.num_games_per_iter} 局)...")
    all_data, stats = parallel_self_play(model, config)
    print(f"\n结果:")
    print(f"  耗时: {stats['total_time']:.1f}s")
    print(f"  新样本: {stats['new_samples']}")
    print(f"  红胜={stats['red_wins']}, 黑胜={stats['black_wins']}, 和={stats['draws']}")
