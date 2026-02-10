"""
GPU 集中推理服务 - 使用 spawn 模式 + SharedMemory/Socket 通信

架构:
    主进程 → spawn 推理服务进程（独立初始化 CUDA + 模型）
    Worker 进程 → 通过 Unix socket 发送推理请求
    推理服务 → 收集请求、组 batch、GPU 推理、返回结果

关键设计:
    1. 推理服务使用 spawn 启动，避免 fork + PyTorch 死锁
    2. 使用 Unix socket (multiprocessing.connection) 通信，避免 Manager 开销
    3. 推理服务内部使用 select/poll 非阻塞收集请求，组大 batch
    4. Worker 使用 InferenceClient，接口与 model.predict() 兼容
"""

import os
import sys
import time
import struct
import pickle
import socket
import tempfile
import logging
import numpy as np
from typing import Dict, Optional, Tuple
from multiprocessing.connection import Listener, Client

import torch
import torch.multiprocessing as mp

logger = logging.getLogger(__name__)

# ============================================================
# 推理服务进程（spawn 模式启动）
# ============================================================

def _server_main(
    model_class_name: str,
    model_module: str,
    model_kwargs: dict,
    state_dict_path: str,
    device_str: str,
    num_workers: int,
    max_batch_size: int,
    batch_timeout_ms: float,
    socket_path: str,
    ready_event,
    stop_event,
):
    """
    推理服务主函数（在 spawn 的子进程中运行）。
    
    通过 Unix socket 接收推理请求，组 batch 后在 GPU 上推理，返回结果。
    """
    # 限制推理服务的线程数
    torch.set_num_threads(4)
    os.environ['OMP_NUM_THREADS'] = '4'
    
    # ========== CUDA 诊断 ==========
    print(f"[InferenceServer] PID={os.getpid()}", flush=True)
    print(f"[InferenceServer] 请求的设备: {device_str}", flush=True)
    print(f"[InferenceServer] PyTorch 版本: {torch.__version__}", flush=True)
    print(f"[InferenceServer] CUDA 可用: {torch.cuda.is_available()}", flush=True)
    
    if torch.cuda.is_available():
        print(f"[InferenceServer] CUDA 设备数: {torch.cuda.device_count()}", flush=True)
        for i in range(torch.cuda.device_count()):
            print(f"[InferenceServer]   GPU {i}: {torch.cuda.get_device_name(i)}", flush=True)
            props = torch.cuda.get_device_properties(i)
            print(f"[InferenceServer]   显存: {props.total_mem / 1024**3:.1f} GB", flush=True)
    
    # 确定实际使用的设备
    actual_device = device_str
    if 'cuda' in device_str and not torch.cuda.is_available():
        print(f"[InferenceServer] 警告: CUDA 不可用，回退到 CPU", flush=True)
        actual_device = 'cpu'
    
    # ========== 动态导入模型类并加载 ==========
    try:
        import importlib
        mod = importlib.import_module(model_module)
        model_class = getattr(mod, model_class_name)
        
        model = model_class(**model_kwargs)
        state_dict = torch.load(state_dict_path, map_location='cpu', weights_only=True)
        model.load_state_dict(state_dict)
        model.to(actual_device)
        model.eval()
        print(f"[InferenceServer] 模型已加载到 {actual_device}", flush=True)
        
        # 验证 GPU 是否真正在使用
        if 'cuda' in actual_device:
            mem_alloc = torch.cuda.memory_allocated() / 1024**2
            mem_reserved = torch.cuda.memory_reserved() / 1024**2
            print(f"[InferenceServer] GPU 显存占用: {mem_alloc:.1f}MB (已分配), "
                  f"{mem_reserved:.1f}MB (已预留)", flush=True)
            
            # Warmup 推理
            dummy = torch.randn(max_batch_size, 15, 10, 9, device=actual_device)
            with torch.no_grad():
                model(dummy)
            torch.cuda.synchronize()
            mem_alloc = torch.cuda.memory_allocated() / 1024**2
            print(f"[InferenceServer] Warmup 完成 (batch={max_batch_size}), "
                  f"GPU 显存: {mem_alloc:.1f}MB", flush=True)
        
    except Exception as e:
        print(f"[InferenceServer] 模型加载失败: {e}", flush=True)
        import traceback
        traceback.print_exc()
        ready_event.set()
        return
    
    # ========== 启动 Unix socket 监听 ==========
    try:
        if os.path.exists(socket_path):
            os.unlink(socket_path)
        listener = Listener(socket_path, family='AF_UNIX')
        print(f"[InferenceServer] Socket 监听: {socket_path}", flush=True)
    except Exception as e:
        print(f"[InferenceServer] Socket 创建失败: {e}", flush=True)
        ready_event.set()
        return
    
    print(f"[InferenceServer] 服务 {num_workers} 个 worker, "
          f"max_batch={max_batch_size}, timeout={batch_timeout_ms}ms", flush=True)
    
    # 通知主进程：服务已就绪
    ready_event.set()
    
    # ========== 主循环：接受连接 + 处理请求 ==========
    connections = {}  # worker_id -> connection
    total_requests = 0
    total_batches = 0
    start_time = time.time()
    batch_timeout_s = batch_timeout_ms / 1000.0
    
    # 先接受所有 worker 连接
    listener._listener._socket.settimeout(1.0)
    
    import select
    
    while not stop_event.is_set():
        # 阶段 1：接受新连接（非阻塞）
        try:
            listener._listener._socket.settimeout(0.001)
            conn = listener.accept()
            # 读取 worker_id
            wid = conn.recv()
            connections[wid] = conn
            print(f"[InferenceServer] Worker {wid} 已连接 (共 {len(connections)} 个)", flush=True)
        except socket.timeout:
            pass
        except OSError:
            pass
        
        if not connections:
            time.sleep(0.001)
            continue
        
        # 阶段 2：收集推理请求（组 batch）
        pending = []  # [(worker_id, state_array), ...]
        
        # 使用 select 检查哪些连接有数据
        readable_conns = []
        conn_to_wid = {id(c): w for w, c in connections.items()}
        filenos = {}
        for wid, conn in list(connections.items()):
            try:
                fd = conn.fileno()
                filenos[fd] = wid
            except Exception:
                continue
        
        if not filenos:
            time.sleep(0.001)
            continue
        
        try:
            readable, _, _ = select.select(list(filenos.keys()), [], [], batch_timeout_s)
        except (ValueError, OSError):
            time.sleep(0.001)
            continue
        
        for fd in readable:
            wid = filenos[fd]
            conn = connections[wid]
            try:
                msg = conn.recv()
                if msg is None:
                    # Worker 断开
                    conn.close()
                    del connections[wid]
                    continue
                # msg 是 numpy array (15, 10, 9)
                pending.append((wid, msg))
            except (EOFError, ConnectionResetError, BrokenPipeError):
                try:
                    conn.close()
                except:
                    pass
                if wid in connections:
                    del connections[wid]
                continue
        
        if not pending:
            continue
        
        # 等待更多请求以组成更大的 batch（短暂等待）
        if len(pending) < max_batch_size and len(pending) < len(connections):
            deadline = time.time() + batch_timeout_s
            while len(pending) < max_batch_size and time.time() < deadline:
                remaining_fds = [
                    fd for fd, w in filenos.items() 
                    if w in connections and w not in {p[0] for p in pending}
                ]
                if not remaining_fds:
                    break
                try:
                    readable2, _, _ = select.select(remaining_fds, [], [], 
                                                     max(0, deadline - time.time()))
                except (ValueError, OSError):
                    break
                for fd in readable2:
                    wid = filenos.get(fd)
                    if wid is None or wid not in connections:
                        continue
                    conn = connections[wid]
                    try:
                        msg = conn.recv()
                        if msg is None:
                            conn.close()
                            del connections[wid]
                            continue
                        pending.append((wid, msg))
                    except (EOFError, ConnectionResetError, BrokenPipeError):
                        try:
                            conn.close()
                        except:
                            pass
                        if wid in connections:
                            del connections[wid]
                if not readable2:
                    break
        
        if not pending:
            continue
        
        # 阶段 3：批量推理
        batch_size = len(pending)
        worker_ids = [p[0] for p in pending]
        states = np.stack([p[1] for p in pending], axis=0)  # (B, 15, 10, 9)
        
        batch_tensor = torch.from_numpy(states).float().to(actual_device)
        
        with torch.no_grad():
            policy_logits, values = model(batch_tensor)
            
            # softmax
            policies = torch.softmax(policy_logits, dim=1).cpu().numpy()
            values = values.squeeze(-1).cpu().numpy()
        
        # 阶段 4：发送结果
        for i, wid in enumerate(worker_ids):
            if wid in connections:
                try:
                    connections[wid].send((policies[i], float(values[i])))
                except (BrokenPipeError, ConnectionResetError, OSError):
                    try:
                        connections[wid].close()
                    except:
                        pass
                    if wid in connections:
                        del connections[wid]
        
        total_requests += batch_size
        total_batches += 1
    
    # 清理
    elapsed = max(1, time.time() - start_time)
    avg_batch = total_requests / max(1, total_batches)
    print(f"[InferenceServer] 结束: {total_requests} 请求, {total_batches} 批次, "
          f"平均batch={avg_batch:.1f}, {total_requests/elapsed:.0f} req/s, "
          f"运行 {elapsed:.0f}s", flush=True)
    
    for conn in connections.values():
        try:
            conn.close()
        except:
            pass
    listener.close()
    try:
        os.unlink(socket_path)
    except:
        pass


# ============================================================
# InferenceClient（在 worker 进程中使用）
# ============================================================

class InferenceClient:
    """
    推理客户端，通过 Unix socket 与推理服务通信。
    接口与 model.predict() 兼容，可直接替换。
    """
    
    def __init__(self, worker_id: int, socket_path: str):
        self.worker_id = worker_id
        self.socket_path = socket_path
        self._conn = None
        self._connected = False
    
    def _ensure_connected(self):
        """延迟连接：首次调用 predict 时才建立连接"""
        if not self._connected:
            max_retries = 30
            for attempt in range(max_retries):
                try:
                    self._conn = Client(self.socket_path, family='AF_UNIX')
                    # 发送 worker_id 作为握手
                    self._conn.send(self.worker_id)
                    self._connected = True
                    return
                except (ConnectionRefusedError, FileNotFoundError):
                    time.sleep(0.1 * (attempt + 1))
            raise RuntimeError(
                f"Worker {self.worker_id}: 无法连接到推理服务 ({self.socket_path})"
            )
    
    def predict(self, state: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        发送状态到推理服务，返回 (policy, value)。
        
        Args:
            state: numpy array, shape (15, 10, 9)
        Returns:
            policy: numpy array, shape (ACTION_SPACE,)
            value: float
        """
        self._ensure_connected()
        try:
            self._conn.send(state)
            result = self._conn.recv()
            return result  # (policy_array, value_float)
        except (EOFError, ConnectionResetError, BrokenPipeError) as e:
            raise RuntimeError(f"Worker {self.worker_id}: 推理服务连接断开: {e}")
    
    def close(self):
        """关闭连接"""
        if self._conn is not None:
            try:
                self._conn.send(None)  # 通知服务端断开
            except:
                pass
            try:
                self._conn.close()
            except:
                pass
            self._connected = False


# ============================================================
# InferenceServer（在主进程中管理）
# ============================================================

class InferenceServer:
    """
    GPU 集中推理服务管理器。
    
    使用 spawn 模式启动独立进程，通过 Unix socket 与 worker 通信。
    彻底避免 fork + PyTorch 死锁问题。
    """
    
    def __init__(
        self,
        model_class,
        model_kwargs: dict,
        state_dict: dict,
        device: str = 'cuda',
        num_workers: int = 4,
        max_batch_size: int = 32,
        batch_timeout_ms: float = 5.0,
    ):
        self.model_class = model_class
        self.model_class_name = model_class.__name__
        self.model_module = model_class.__module__
        self.model_kwargs = model_kwargs
        self.state_dict = state_dict
        self.device = device
        self.num_workers = num_workers
        self.max_batch_size = max_batch_size
        self.batch_timeout_ms = batch_timeout_ms
        
        # 创建临时文件路径
        self._tmpdir = tempfile.mkdtemp(prefix='xiangqi_inference_')
        self.socket_path = os.path.join(self._tmpdir, 'inference.sock')
        self._state_dict_path = os.path.join(self._tmpdir, 'model_state.pt')
        
        self._process = None
        self._started = False
        
        # 使用 spawn 上下文
        self._ctx = mp.get_context('spawn')
        self._ready_event = self._ctx.Event()
        self._stop_event = self._ctx.Event()
    
    def start(self):
        """
        启动推理服务进程。
        
        返回:
            socket_path: str, worker 用于连接的 Unix socket 路径
        """
        if self._started:
            raise RuntimeError("推理服务已经启动")
        
        # 将 state_dict 保存到临时文件（避免 pickle 大对象通过 spawn 传递）
        torch.save(self.state_dict, self._state_dict_path)
        logger.info(f"模型权重已保存到临时文件: {self._state_dict_path}")
        
        # 使用 spawn 启动推理服务进程
        self._process = self._ctx.Process(
            target=_server_main,
            args=(
                self.model_class_name,
                self.model_module,
                self.model_kwargs,
                self._state_dict_path,
                self.device,
                self.num_workers,
                self.max_batch_size,
                self.batch_timeout_ms,
                self.socket_path,
                self._ready_event,
                self._stop_event,
            ),
            daemon=True,
        )
        self._process.start()
        
        logger.info(f"等待 GPU 推理服务就绪 (PID={self._process.pid})...")
        ready = self._ready_event.wait(timeout=120)
        if not ready:
            raise RuntimeError("GPU 推理服务启动超时（120秒）")
        
        if not self._process.is_alive():
            raise RuntimeError("GPU 推理服务进程异常退出")
        
        self._started = True
        logger.info(f"GPU 推理服务已就绪 (PID={self._process.pid}, "
                     f"device={self.device}, socket={self.socket_path})")
        
        return self.socket_path
    
    def stop(self):
        """停止推理服务"""
        self._stop_event.set()
        if self._process is not None and self._process.is_alive():
            self._process.join(timeout=10)
            if self._process.is_alive():
                self._process.terminate()
                self._process.join(timeout=5)
        
        # 清理临时文件
        try:
            if os.path.exists(self.socket_path):
                os.unlink(self.socket_path)
            if os.path.exists(self._state_dict_path):
                os.unlink(self._state_dict_path)
            os.rmdir(self._tmpdir)
        except:
            pass
        
        logger.info("GPU 推理服务已停止")
    
    def update_model(self, state_dict: dict):
        """
        热更新模型权重（用于训练循环中每轮更新后）。
        
        实现方式：停止旧服务 → 保存新权重 → 启动新服务
        """
        was_started = self._started
        if was_started:
            self.stop()
        
        self.state_dict = state_dict
        self._started = False
        self._ready_event = self._ctx.Event()
        self._stop_event = self._ctx.Event()
        
        if was_started:
            return self.start()
        return self.socket_path
