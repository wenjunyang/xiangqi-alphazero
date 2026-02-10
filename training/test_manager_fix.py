"""
测试 Manager 死锁修复
"""
import sys
import time
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import torch.multiprocessing as mp

# 确保使用 spawn 模式
mp.set_start_method('spawn', force=True)

from model import XiangqiNet
from inference_server import InferenceServer, InferenceClient
from game import XiangqiGame


def worker_task(worker_id, request_queue, response_queue, num_requests=5):
    """Worker 任务：发送推理请求"""
    print(f"[Worker {worker_id}] 启动")
    
    client = InferenceClient(worker_id, request_queue, response_queue)
    game = XiangqiGame()
    
    for i in range(num_requests):
        state = game.get_state_for_nn()
        policy, value = client.predict(state)
        print(f"[Worker {worker_id}] 请求 {i}: policy.shape={policy.shape}, value={value:.3f}")
    
    print(f"[Worker {worker_id}] 完成")
    return worker_id


def test_manager_fix():
    """测试修复后的 InferenceServer"""
    print("=" * 60)
    print("测试 Manager 死锁修复")
    print("=" * 60)
    
    # 创建模型
    model = XiangqiNet(num_channels=64, num_res_blocks=2)
    
    # 创建推理服务
    num_workers = 4
    server = InferenceServer(
        model_class=XiangqiNet,
        model_kwargs={'num_channels': 64, 'num_res_blocks': 2},
        state_dict=model.state_dict(),
        device='cpu',  # 沙箱中使用 CPU
        num_workers=num_workers,
        max_batch_size=16,
        batch_timeout_ms=10.0,
    )
    
    print(f"\n1. 启动推理服务（{num_workers} workers）")
    response_queues = server.start()
    print(f"   ✓ 推理服务已启动，获得 {len(response_queues)} 个响应队列")
    
    time.sleep(0.5)  # 等待服务完全启动
    
    print(f"\n2. 启动 {num_workers} 个 worker 进程")
    start_time = time.time()
    
    with ProcessPoolExecutor(max_workers=num_workers, mp_context=mp.get_context('spawn')) as executor:
        futures = []
        for wid in range(num_workers):
            future = executor.submit(
                worker_task,
                wid,
                server.request_queue,
                response_queues[wid],
                num_requests=5
            )
            futures.append(future)
        
        # 等待所有 worker 完成
        results = [f.result(timeout=60) for f in futures]
    
    elapsed = time.time() - start_time
    print(f"\n3. 所有 worker 完成，耗时 {elapsed:.2f}s")
    print(f"   ✓ 完成的 workers: {results}")
    
    print("\n4. 停止推理服务")
    server.stop()
    print("   ✓ 推理服务已停止")
    
    print("\n" + "=" * 60)
    print("✓ 测试通过！Manager 死锁已修复")
    print("=" * 60)


if __name__ == '__main__':
    test_manager_fix()
