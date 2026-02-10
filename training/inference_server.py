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

关键设计:
    - 使用 multiprocessing.Manager().Queue() 创建可序列化的代理队列
    - 这些代理队列可以安全地通过 pickle 传递给 spawn 模式的子进程
    - request_queue: 所有 worker 共享的请求队列
    - response_queues: dict[worker_id -> Queue]，每个 worker 一个响应队列
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

    使用 multiprocessing.Manager() 创建代理队列，
    这些队列可以安全地通过 pickle 传递给 spawn 模式的子进程。

    使用方法:
        server = InferenceServer(model_class, model_kwargs, state_dict, device='cuda')
        q0 = server.create_worker_queue(0)
        q1 = server.create_worker_queue(1)
        server.start()
        # worker 中: client = InferenceClient(wid, server.request_queue, q_i)
        server.stop()
    """

    def __init__(
        self,
        model_class,
        model_kwargs: dict,
        state_dict: dict,
        device: str = 'cuda',
        max_batch_size: int = 256,
        batch_timeout_ms: float = 5.0,
    ):
        self.model_class = model_class
        self.model_kwargs = model_kwargs
        self.state_dict = state_dict
        self.device = device
        self.max_batch_size = max_batch_size
        self.batch_timeout_ms = batch_timeout_ms

        # 使用 Manager 创建可序列化的代理队列
        self._manager = mp.Manager()
        self.request_queue = self._manager.Queue()
        self._response_queues: Dict[int, object] = {}
        self._process = None
        self._stop_event = self._manager.Event()

    def create_worker_queue(self, worker_id: int):
        """为 worker 创建响应队列（Manager 代理队列，可跨 spawn 进程传递）"""
        q = self._manager.Queue()
        self._response_queues[worker_id] = q
        return q

    def start(self):
        """启动推理服务进程"""
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
        logger.info(f"GPU 推理服务已启动 (PID={self._process.pid}, device={self.device})")

    def stop(self):
        """停止推理服务"""
        self._stop_event.set()
        if self._process is not None and self._process.is_alive():
            self._process.join(timeout=5)
            if self._process.is_alive():
                self._process.terminate()
            logger.info("GPU 推理服务已停止")
        # 关闭 Manager
        try:
            self._manager.shutdown()
        except Exception:
            pass

    def update_model(self, state_dict: dict):
        """更新模型权重"""
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

    def predict(self, state: np.ndarray, device: str = None):
        """
        发送推理请求并等待结果。
        兼容 XiangqiNet.predict(state, device) 接口。

        参数:
            state: numpy 数组 (15, 10, 9)
            device: 忽略（由 InferenceServer 决定设备）
        返回:
            (policy, value) - policy 是 numpy 数组, value 是 float
        """
        self._request_counter += 1
        req_id = self._request_counter

        self.request_queue.put((req_id, self.worker_id, state))
        resp_id, policy, value = self.response_queue.get(timeout=60)
        return policy, float(value)

    def batch_predict(self, states: list):
        """批量推理"""
        req_ids = []
        for state in states:
            self._request_counter += 1
            req_id = self._request_counter
            req_ids.append(req_id)
            self.request_queue.put((req_id, self.worker_id, state))

        results = {}
        while len(results) < len(req_ids):
            resp_id, policy, value = self.response_queue.get(timeout=60)
            results[resp_id] = (policy, float(value))

        return [results[rid] for rid in req_ids]
