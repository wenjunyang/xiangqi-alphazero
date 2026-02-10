"""
多进程并行自对弈
================
支持两种推理模式：
1. CPU 本地推理（默认）：每个 worker 进程独立重建模型，串行完成对局
2. GPU 集中推理：所有 worker 通过 InferenceClient 将推理请求发送到 GPU 推理服务

架构:
  模式1 (CPU):
    主进程 → 序列化模型权重 → 分发到 N 个 worker
    每个 worker → 重建模型(CPU) → 串行完成 K 局 → 返回数据

  模式2 (GPU):
    主进程 → 启动 InferenceServer(GPU)
    主进程 → 启动 N 个 worker 进程
    每个 worker → InferenceClient → 发送推理请求 → 等待结果
    InferenceServer → 收集请求 → 组 batch → GPU 推理 → 分发结果
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


def _play_one_game(model_or_client, config, device='cpu') -> Tuple[List, int, int]:
    """
    执行一局自对弈，返回 (training_data, winner, steps)。

    参数:
        model_or_client: XiangqiNet 实例或 InferenceClient 实例
        config: 训练配置
        device: 计算设备（仅 XiangqiNet 模式使用）
    """
    mcts = MCTS(
        model_or_client,
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
            state_nn = game.get_state_for_nn()
            try:
                _, value = model_or_client.predict(state_nn, device)
            except TypeError:
                _, value = model_or_client.predict(state_nn)
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


def _augment_data(data):
    """数据增强：水平翻转"""
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
    return aug_data


# ============================================================
# 模式1: CPU 本地推理 Worker
# ============================================================

def _cpu_worker_entry(args: dict) -> List[Tuple[List, int, int]]:
    """CPU 模式 Worker 进程入口"""
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    try:
        torch.set_num_threads(1)
    except Exception:
        pass

    seed = int.from_bytes(os.urandom(4), 'big')
    random.seed(seed)
    np.random.seed(seed % (2**31))
    torch.manual_seed(seed)

    try:
        model = XiangqiNet(
            num_channels=args['num_channels'],
            num_res_blocks=args['num_res_blocks'],
        )
        model.load_state_dict(args['model_state_dict'])
        model.eval()

        class WorkerConfig:
            pass

        cfg = WorkerConfig()
        for key in ['num_simulations', 'c_puct', 'temperature_threshold',
                     'max_game_length', 'random_opening_moves',
                     'enable_resign', 'resign_threshold', 'resign_check_steps']:
            setattr(cfg, key, args[key])

        results = []
        for _ in range(args['num_games']):
            data, winner, steps = _play_one_game(model, cfg, 'cpu')
            aug_data = _augment_data(data)
            results.append((aug_data, winner, steps))

        return results

    except Exception as e:
        import traceback
        traceback.print_exc()
        return []


# ============================================================
# 模式2: GPU 集中推理 Worker
# ============================================================

def _gpu_worker_entry(args: dict) -> List[Tuple[List, int, int]]:
    """GPU 模式 Worker 进程入口：使用 InferenceClient 发送推理请求"""
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
        from inference_server import InferenceClient

        client = InferenceClient(
            worker_id=args['worker_id'],
            request_queue=args['request_queue'],
            response_queue=args['response_queue'],
        )

        class WorkerConfig:
            pass

        cfg = WorkerConfig()
        for key in ['num_simulations', 'c_puct', 'temperature_threshold',
                     'max_game_length', 'random_opening_moves',
                     'enable_resign', 'resign_threshold', 'resign_check_steps']:
            setattr(cfg, key, args[key])

        results = []
        for _ in range(args['num_games']):
            data, winner, steps = _play_one_game(client, cfg, 'cpu')
            aug_data = _augment_data(data)
            results.append((aug_data, winner, steps))

        return results

    except Exception as e:
        import traceback
        traceback.print_exc()
        return []


# ============================================================
# 统一入口
# ============================================================

def parallel_self_play(
    model: XiangqiNet,
    config,
    num_workers: Optional[int] = None,
    use_gpu_server: bool = False,
    gpu_device: str = 'cuda',
) -> Tuple[List[Tuple[np.ndarray, np.ndarray, float]], Dict[str, Any]]:
    """
    多进程并行自对弈。

    参数:
        model: 当前最优模型
        config: TrainingConfig
        num_workers: 进程数（None=自动）
        use_gpu_server: 是否使用 GPU 集中推理服务
        gpu_device: GPU 设备（仅 use_gpu_server=True 时使用）

    返回:
        (all_training_data, stats)
    """
    total_cores = os.cpu_count() or 4
    num_games = config.num_games_per_iter

    if num_workers is None:
        num_workers = max(1, total_cores - 1)
    num_workers = min(num_workers, num_games)

    # 均匀分配对局
    games_per_worker = num_games // num_workers
    remainder = num_games % num_workers

    mode_str = f"GPU集中推理({gpu_device})" if use_gpu_server else "CPU本地推理"
    logger.info(
        f"开始并行自对弈: {num_games} 局, "
        f"{num_workers} workers, "
        f"模式={mode_str}"
    )

    start_time = time.time()

    if use_gpu_server:
        all_data, win_counts, total_steps, valid_games = _run_gpu_mode(
            model, config, num_workers, games_per_worker, remainder, num_games, gpu_device
        )
    else:
        all_data, win_counts, total_steps, valid_games = _run_cpu_mode(
            model, config, num_workers, games_per_worker, remainder, num_games
        )

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
        'num_workers': num_workers,
        'mode': 'gpu' if use_gpu_server else 'cpu',
    }

    logger.info(
        f"自对弈完成: {valid_games} 局, {elapsed:.1f}s, "
        f"红{stats['red_wins']}黑{stats['black_wins']}和{stats['draws']}, "
        f"平均{avg_steps:.0f}步, {len(all_data)} 样本"
    )

    return all_data, stats


def _run_cpu_mode(model, config, num_workers, games_per_worker, remainder, num_games):
    """CPU 本地推理模式"""
    model_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}

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

    all_data = []
    win_counts = {1: 0, -1: 0, 0: 0}
    total_steps = 0
    valid_games = 0

    ctx = mp.get_context('spawn')
    with ProcessPoolExecutor(max_workers=len(worker_args_list), mp_context=ctx) as executor:
        futures = {
            executor.submit(_cpu_worker_entry, args): i
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

    return all_data, win_counts, total_steps, valid_games


def _run_gpu_mode(model, config, num_workers, games_per_worker, remainder, num_games, gpu_device):
    """GPU 集中推理模式"""
    from inference_server import InferenceServer

    # 启动推理服务
    server = InferenceServer(
        model_class=XiangqiNet,
        model_kwargs={
            'num_channels': model.num_channels,
            'num_res_blocks': model.num_res_blocks,
        },
        state_dict={k: v.cpu().clone() for k, v in model.state_dict().items()},
        device=gpu_device,
        max_batch_size=num_workers * 2,
        batch_timeout_ms=5.0,
    )

    # 为每个 worker 创建响应队列
    response_queues = {}
    for w in range(num_workers):
        response_queues[w] = server.create_worker_queue(w)

    server.start()

    try:
        # 构建 worker 参数
        worker_args_list = []
        for w in range(num_workers):
            this_games = games_per_worker + (1 if w < remainder else 0)
            if this_games <= 0:
                continue
            worker_args_list.append({
                'worker_id': w,
                'request_queue': server.request_queue,
                'response_queue': response_queues[w],
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

        all_data = []
        win_counts = {1: 0, -1: 0, 0: 0}
        total_steps = 0
        valid_games = 0

        ctx = mp.get_context('spawn')
        with ProcessPoolExecutor(max_workers=len(worker_args_list), mp_context=ctx) as executor:
            futures = {
                executor.submit(_gpu_worker_entry, args): i
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

    finally:
        server.stop()

    return all_data, win_counts, total_steps, valid_games
