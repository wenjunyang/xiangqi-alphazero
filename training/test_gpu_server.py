"""
GPU 集中推理服务测试
===================
在 CPU 上模拟 GPU 推理服务的完整流程，验证：
1. InferenceServer 启动/停止
2. InferenceClient 发送请求和接收结果
3. 多 worker 并发推理
4. 模型权重更新
5. 与 MCTS 的集成
"""

import os
import sys
import time
import logging
import numpy as np
import torch
import torch.multiprocessing as mp

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

from game import XiangqiGame, _USE_CYTHON
from model import XiangqiNet
from mcts import MCTS
from inference_server import InferenceServer, InferenceClient

print(f"Cython 加速: {_USE_CYTHON}")


def test_basic_inference():
    """测试1: 基本推理请求/响应"""
    print("\n=== 测试1: 基本推理 ===")

    model = XiangqiNet(num_channels=32, num_res_blocks=2)
    model.eval()

    server = InferenceServer(
        model_class=XiangqiNet,
        model_kwargs={'num_channels': 32, 'num_res_blocks': 2},
        state_dict={k: v.cpu().clone() for k, v in model.state_dict().items()},
        device='cpu',  # 用 CPU 模拟
        max_batch_size=16,
        batch_timeout_ms=10.0,
    )

    resp_q = server.create_worker_queue(0)
    server.start()

    try:
        client = InferenceClient(
            worker_id=0,
            request_queue=server.request_queue,
            response_queue=resp_q,
        )

        # 单次推理
        game = XiangqiGame()
        state = game.get_state_for_nn()
        policy, value = client.predict(state)

        from game import ACTION_SPACE
        assert policy.shape == (ACTION_SPACE,), f"Policy shape: {policy.shape}"
        assert isinstance(value, float), f"Value type: {type(value)}"
        assert abs(policy.sum() - 1.0) < 0.01, f"Policy sum: {policy.sum()}"

        print(f"  ✓ 单次推理: policy shape={policy.shape}, value={value:.4f}")

        # 批量推理
        states = [game.get_state_for_nn() for _ in range(5)]
        results = client.batch_predict(states)
        assert len(results) == 5
        print(f"  ✓ 批量推理: {len(results)} 个结果")

    finally:
        server.stop()

    print("  ✓ 测试1 通过")


def test_mcts_with_client():
    """测试2: MCTS 使用 InferenceClient"""
    print("\n=== 测试2: MCTS + InferenceClient ===")

    model = XiangqiNet(num_channels=32, num_res_blocks=2)
    model.eval()

    server = InferenceServer(
        model_class=XiangqiNet,
        model_kwargs={'num_channels': 32, 'num_res_blocks': 2},
        state_dict={k: v.cpu().clone() for k, v in model.state_dict().items()},
        device='cpu',
        max_batch_size=16,
        batch_timeout_ms=10.0,
    )

    resp_q = server.create_worker_queue(0)
    server.start()

    try:
        client = InferenceClient(
            worker_id=0,
            request_queue=server.request_queue,
            response_queue=resp_q,
        )

        # 使用 InferenceClient 作为 MCTS 的 model
        mcts = MCTS(client, num_simulations=10, c_puct=1.5, device='cpu')

        game = XiangqiGame()
        action_probs = mcts.search(game, temperature=1.0, add_noise=True)

        from game import ACTION_SPACE as AS
        assert action_probs.shape == (AS,), f"Action probs shape: {action_probs.shape}"
        assert action_probs.sum() > 0.99, f"Action probs sum: {action_probs.sum()}"

        action = mcts.get_action(game, temperature=0, add_noise=False)
        assert 0 <= action < AS, f"Action: {action}"

        print(f"  ✓ MCTS 搜索完成: action={action}")

    finally:
        server.stop()

    print("  ✓ 测试2 通过")


def _worker_func(args):
    """多 worker 测试的 worker 函数"""
    from inference_server import InferenceClient
    from game import XiangqiGame
    from mcts import MCTS

    client = InferenceClient(
        worker_id=args['worker_id'],
        request_queue=args['request_queue'],
        response_queue=args['response_queue'],
    )

    mcts = MCTS(client, num_simulations=args['num_sims'], c_puct=1.5, device='cpu')
    game = XiangqiGame()

    # 走几步
    results = []
    for step in range(3):
        action = mcts.get_action(game, temperature=1.0, add_noise=True)
        game.make_action(action)
        done, _ = game.is_game_over()
        if done:
            break
        results.append(action)

    return {'worker_id': args['worker_id'], 'actions': results}


def test_multi_worker():
    """测试3: 多 worker 并发推理"""
    print("\n=== 测试3: 多 Worker 并发 ===")

    model = XiangqiNet(num_channels=32, num_res_blocks=2)
    model.eval()

    num_workers = 3

    server = InferenceServer(
        model_class=XiangqiNet,
        model_kwargs={'num_channels': 32, 'num_res_blocks': 2},
        state_dict={k: v.cpu().clone() for k, v in model.state_dict().items()},
        device='cpu',
        max_batch_size=32,
        batch_timeout_ms=10.0,
    )

    # 创建 worker 队列
    worker_args = []
    for w in range(num_workers):
        resp_q = server.create_worker_queue(w)
        worker_args.append({
            'worker_id': w,
            'request_queue': server.request_queue,
            'response_queue': resp_q,
            'num_sims': 5,
        })

    server.start()

    try:
        from concurrent.futures import ProcessPoolExecutor, as_completed

        ctx = mp.get_context('spawn')
        with ProcessPoolExecutor(max_workers=num_workers, mp_context=ctx) as executor:
            futures = {executor.submit(_worker_func, args): args['worker_id']
                       for args in worker_args}

            results = []
            for future in as_completed(futures):
                wid = futures[future]
                try:
                    result = future.result(timeout=120)
                    results.append(result)
                    print(f"  ✓ Worker {wid}: {len(result['actions'])} 步")
                except Exception as e:
                    print(f"  ✗ Worker {wid} 异常: {e}")

        assert len(results) == num_workers, f"只有 {len(results)}/{num_workers} workers 完成"

    finally:
        server.stop()

    print("  ✓ 测试3 通过")


def test_model_update():
    """测试4: 运行中更新模型权重"""
    print("\n=== 测试4: 模型权重更新 ===")

    model = XiangqiNet(num_channels=32, num_res_blocks=2)
    model.eval()

    server = InferenceServer(
        model_class=XiangqiNet,
        model_kwargs={'num_channels': 32, 'num_res_blocks': 2},
        state_dict={k: v.cpu().clone() for k, v in model.state_dict().items()},
        device='cpu',
        max_batch_size=16,
        batch_timeout_ms=10.0,
    )

    resp_q = server.create_worker_queue(0)
    server.start()

    try:
        client = InferenceClient(
            worker_id=0,
            request_queue=server.request_queue,
            response_queue=resp_q,
        )

        game = XiangqiGame()
        state = game.get_state_for_nn()

        # 推理1
        policy1, value1 = client.predict(state)

        # 更新模型（随机新权重）
        new_model = XiangqiNet(num_channels=32, num_res_blocks=2)
        new_state_dict = {k: v.cpu().clone() for k, v in new_model.state_dict().items()}
        server.update_model(new_state_dict)
        time.sleep(0.5)  # 等待更新生效

        # 推理2
        policy2, value2 = client.predict(state)

        # 权重不同，结果应该不同
        diff = np.abs(policy1 - policy2).sum()
        print(f"  ✓ 更新前后策略差异: {diff:.4f}")
        assert diff > 0.01, "更新后策略应该有变化"

    finally:
        server.stop()

    print("  ✓ 测试4 通过")


def test_performance():
    """测试5: 性能对比（本地推理 vs 推理服务）"""
    print("\n=== 测试5: 性能对比 ===")

    model = XiangqiNet(num_channels=32, num_res_blocks=2)
    model.eval()

    game = XiangqiGame()
    state = game.get_state_for_nn()
    num_requests = 50

    # 本地推理
    start = time.time()
    for _ in range(num_requests):
        model.predict(state, 'cpu')
    local_time = time.time() - start

    # 推理服务
    server = InferenceServer(
        model_class=XiangqiNet,
        model_kwargs={'num_channels': 32, 'num_res_blocks': 2},
        state_dict={k: v.cpu().clone() for k, v in model.state_dict().items()},
        device='cpu',
        max_batch_size=32,
        batch_timeout_ms=2.0,
    )

    resp_q = server.create_worker_queue(0)
    server.start()

    try:
        client = InferenceClient(
            worker_id=0,
            request_queue=server.request_queue,
            response_queue=resp_q,
        )

        start = time.time()
        for _ in range(num_requests):
            client.predict(state)
        server_time = time.time() - start

    finally:
        server.stop()

    print(f"  本地推理: {local_time:.3f}s ({local_time/num_requests*1000:.1f}ms/次)")
    print(f"  推理服务: {server_time:.3f}s ({server_time/num_requests*1000:.1f}ms/次)")
    print(f"  开销比: {server_time/local_time:.2f}x")
    print(f"  (注: 单 worker CPU 模式下推理服务有 IPC 开销，GPU + 多 worker 时优势显现)")
    print("  ✓ 测试5 通过")


if __name__ == '__main__':
    test_basic_inference()
    test_mcts_with_client()
    test_model_update()
    test_multi_worker()
    test_performance()
    print("\n✓ 全部测试通过！")
