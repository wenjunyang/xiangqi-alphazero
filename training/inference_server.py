"""
GPU 集中推理服务
================
所有 worker 进程将推理请求发送到一个独立的 GPU 推理进程，
该进程收集请求、组成大 batch、在 GPU 上一次性推理，然后返回结果。

架构:
    Worker 1 ──┐
    Worker 2 ──┤  通过 Manager().Queue() 发送状态
    Worker 3 ──┼──→ InferenceServer（单进程，GPU 批量推理）
    Worker N ──┘    收集请求 → 组 batch → GPU 推理 → 分发结果

关键修复:
    - Manager 在 start() 中延迟初始化，避免 spawn 模式下的死锁
    - start() 返回 response_queues 字典，供 worker 使用
"""

import os
import time
import logging
import numpy as np
from typing import Optional, Dict

import torch
import torch.multiprocessing as mp

logger = logging.getLogger(__name__)


def _server_loop(
    model_class,
    model_kwargs,
    state_dict,
    device,
    max_batch_size,
    batch_timeout_ms,
    request_queue,
    response_queues_list,
    stop_event,
):
    """
    推理服务主循环（在子进程中运行）。

    参数:
        response_queues_list: [(worker_id, Queue), ...] 列表形式传递
    """
    torch.set_num_threads(2)
    os.environ['OMP_NUM_THREADS'] = '2'

    # 重建响应队列字典
    response_queues: Dict[int, object] = {}
    for wid, q in response_queues_list:
        response_queues[wid] = q

    # 初始化模型
    model = model_class(**model_kwargs)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    print(f"[InferenceServer] 模型已加载到 {device}, 服务 {len(response_queues)} 个 worker")

    batch_timeout_s = batch_timeout_ms / 1000.0
    total_requests = 0
    total_batches = 0

    while not stop_event.is_set():
        batch_requests = []

        # 等待第一个请求
        try:
            item = request_queue.get(timeout=0.5)
        except Exception:
            continue

        # 处理控制消息
        if isinstance(item, tuple) and len(item) >= 2 and item[0] == 'UPDATE_MODEL':
            new_state_dict = item[1]
            model.load_state_dict(new_state_dict)
            model.to(device)
            model.eval()
            print("[InferenceServer] 模型权重已更新")
            continue

        batch_requests.append(item)

        # 继续收集直到 batch 满或超时
        deadline = time.monotonic() + batch_timeout_s
        while len(batch_requests) < max_batch_size:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                item = request_queue.get(timeout=min(remaining, 0.001))
                if isinstance(item, tuple) and len(item) >= 2 and item[0] == 'UPDATE_MODEL':
                    new_state_dict = item[1]
                    model.load_state_dict(new_state_dict)
                    model.to(device)
                    model.eval()
                    continue
                batch_requests.append(item)
            except Exception:
                break

        if not batch_requests:
            continue

        # 组 batch 推理
        request_ids = []
        worker_ids = []
        states = []

        for req_id, worker_id, state in batch_requests:
            request_ids.append(req_id)
            worker_ids.append(worker_id)
            states.append(state)

        try:
            batch_tensor = torch.from_numpy(np.array(states)).float().to(device)
            with torch.no_grad():
                policy_logits, values = model(batch_tensor)
                policies = torch.softmax(policy_logits, dim=1).cpu().numpy()
                values_np = values.cpu().numpy().flatten()

            for i in range(len(batch_requests)):
                wid = worker_ids[i]
                rid = request_ids[i]
                if wid in response_queues:
                    response_queues[wid].put((rid, policies[i], float(values_np[i])))
        except Exception as e:
            print(f"[InferenceServer] 推理错误: {e}")
            import traceback
            traceback.print_exc()
            for i in range(len(batch_requests)):
                wid = worker_ids[i]
                rid = request_ids[i]
                if wid in response_queues:
                    action_space = 8100
                    dummy_policy = np.ones(action_space, dtype=np.float32) / action_space
                    response_queues[wid].put((rid, dummy_policy, 0.0))

        total_requests += len(batch_requests)
        total_batches += 1

    print(f"[InferenceServer] 结束: {total_requests} 请求, {total_batches} 批次, "
          f"平均 batch={total_requests / max(total_batches, 1):.1f}")


class InferenceServer:
    """
    GPU 集中推理服务。

    使用方法:
        server = InferenceServer(model_class, model_kwargs, state_dict, device='cuda', num_workers=14)
        response_queues = server.start()  # 返回 {worker_id: Queue}
        # worker 中: client = InferenceClient(wid, server.request_queue, response_queues[wid])
        server.stop()
    """

    def __init__(
        self,
        model_class,
        model_kwargs: dict,
        state_dict: dict,
        device: str = 'cuda',
        num_workers: int = 1,
        max_batch_size: int = 256,
        batch_timeout_ms: float = 5.0,
    ):
        self.model_class = model_class
        self.model_kwargs = model_kwargs
        self.state_dict = state_dict
        self.device = device
        self.num_workers = num_workers
        self.max_batch_size = max_batch_size
        self.batch_timeout_ms = batch_timeout_ms

        # 延迟初始化，避免 spawn 模式下的死锁
        self._manager = None
        self.request_queue = None
        self._response_queues: Dict[int, object] = {}
        self._process = None
        self._stop_event = None
        self._started = False

    def start(self) -> Dict[int, object]:
        """
        启动推理服务进程。
        
        返回:
            response_queues: {worker_id: Queue} 字典，供 worker 使用
        """
        if self._started:
            raise RuntimeError("推理服务已经启动")
        
        # 在 start() 中创建 Manager，避免 spawn 模式下的死锁
        self._manager = mp.Manager()
        self.request_queue = self._manager.Queue()
        self._stop_event = self._manager.Event()
        
        # 为每个 worker 创建响应队列
        for wid in range(self.num_workers):
            self._response_queues[wid] = self._manager.Queue()
        
        response_queues_list = [(wid, q) for wid, q in self._response_queues.items()]

        ctx = mp.get_context('spawn')
        self._process = ctx.Process(
            target=_server_loop,
            args=(
                self.model_class,
                self.model_kwargs,
                self.state_dict,
                self.device,
                self.max_batch_size,
                self.batch_timeout_ms,
                self.request_queue,
                response_queues_list,
                self._stop_event,
            ),
            daemon=True,
        )
        self._process.start()
        self._started = True
        logger.info(f"GPU 推理服务已启动 (PID={self._process.pid}, device={self.device})")
        
        # 返回响应队列供 worker 使用
        return self._response_queues

    def stop(self):
        """停止推理服务"""
        if self._stop_event is not None:
            self._stop_event.set()
        if self._process is not None and self._process.is_alive():
            self._process.join(timeout=5)
            if self._process.is_alive():
                self._process.terminate()
            logger.info("GPU 推理服务已停止")
        # 关闭 Manager
        if self._manager is not None:
            try:
                self._manager.shutdown()
            except Exception:
                pass

    def update_model(self, state_dict: dict):
        """更新模型权重"""
        if self.request_queue is not None:
            self.request_queue.put(('UPDATE_MODEL', state_dict, None))


class InferenceClient:
    """
    Worker 端的推理客户端。
    替代 model.predict()，将请求发送到 GPU 推理服务。

    兼容 MCTS 中 model.predict(state, device) 的调用方式。
    """

    def __init__(self, worker_id: int, request_queue, response_queue):
        self.worker_id = worker_id
        self.request_queue = request_queue
        self.response_queue = response_queue
        self._request_counter = 0

    def predict(self, state: np.ndarray, device: str = 'cpu') -> tuple:
        """
        发送推理请求并等待结果。
        
        参数:
            state: (15, 10, 9) 的 numpy 数组
            device: 忽略（由推理服务决定）
        
        返回:
            (policy, value): policy 是 (8100,) 的概率分布，value 是标量
        """
        req_id = self._request_counter
        self._request_counter += 1

        # 发送请求
        self.request_queue.put((req_id, self.worker_id, state))

        # 等待响应
        while True:
            try:
                resp_id, policy, value = self.response_queue.get(timeout=30)
                if resp_id == req_id:
                    return policy, value
            except Exception as e:
                print(f"[InferenceClient] Worker {self.worker_id} 等待响应超时: {e}")
                # 返回 dummy 结果
                action_space = 8100
                dummy_policy = np.ones(action_space, dtype=np.float32) / action_space
                return dummy_policy, 0.0
