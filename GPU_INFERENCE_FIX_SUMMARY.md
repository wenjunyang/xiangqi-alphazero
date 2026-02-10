# GPU 推理服务修复总结

## 问题描述

训练启动后存在两个关键问题：
1. **Worker 进程 CPU 利用率极低**（6-40%）：Worker 大部分时间阻塞在 `Queue.get()` 上等待推理结果
2. **GPU 进程不出现在 nvidia-smi 中**：推理服务进程在初始化 CUDA/模型时死锁

## 根因分析

### 问题1：fork + PyTorch 死锁

**根本原因**：`train.py` 在启动 InferenceServer 之前已经使用了 PyTorch（模型初始化、权重复制等），PyTorch 内部的线程池和锁已经被初始化。当使用 `fork` 模式创建子进程时，子进程继承了这些锁的状态，导致在子进程中重新初始化 PyTorch（创建模型、加载权重）时触发死锁。

**表现**：
- `_server_loop` 进程启动后，在 `XiangqiNet(**model_kwargs)` 处永久挂起
- `ready_event` 永远不会被设置
- 所有 Worker 进程阻塞等待推理服务就绪
- GPU 进程因为死锁无法完成 CUDA 初始化，所以不出现在 nvidia-smi 中

### 问题2：Manager().Queue() 高开销

**根本原因**：`multiprocessing.Manager().Queue()` 通过 TCP socket + 代理对象实现跨进程通信，每次 `put()/get()` 都涉及序列化、网络传输、反序列化，开销约为原生 Queue 的 10-50 倍。

**表现**：
- 即使推理服务正常工作，Worker 也会在 Queue 通信上花费大量时间
- CPU 利用率低，因为大部分时间在等待 IPC

## 解决方案

### 核心改动：spawn 模式 + Unix socket 通信

| 组件 | 修复前 | 修复后 |
|------|--------|--------|
| InferenceServer 启动模式 | fork (继承主进程状态) | **spawn** (全新进程) |
| Worker 启动模式 | fork (mp.Process) | **spawn** (ProcessPoolExecutor) |
| IPC 通信方式 | Manager().Queue() | **Unix socket** (multiprocessing.connection) |
| 模型传递方式 | pickle 通过 Queue | **临时文件** (torch.save → torch.load) |
| 模型类传递 | 直接传递类对象 | **字符串** (module + class name, importlib 动态导入) |

### 架构变更

```
修复前:
  主进程 (fork) → InferenceServer._server_loop (继承 PyTorch 状态 → 死锁)
  主进程 (fork) → Worker 进程 (通过 Manager().Queue() 通信 → 高开销)

修复后:
  主进程 (spawn) → InferenceServer (全新进程, 独立初始化 CUDA → 无死锁)
  主进程 (spawn) → Worker 进程 (通过 Unix socket 通信 → 低开销)
```

### API 变更

```python
# 修复前
server = InferenceServer(model_class, model_kwargs, state_dict, device, ...)
response_queues = server.start()  # 返回 Queue 列表
client = InferenceClient(worker_id, request_queue, response_queues[i])

# 修复后
server = InferenceServer(model_class, model_kwargs, state_dict, device, ...)
socket_path = server.start()  # 返回 Unix socket 路径 (str)
client = InferenceClient(worker_id, socket_path)
```

## 性能测试结果

| 测试项 | 结果 |
|--------|------|
| 直接模型推理 | 5.58ms/次 |
| Server 推理（单 worker） | 5.75ms/次 (**开销仅 1.03x**) |
| 多 worker 并发（4 workers） | 吞吐量 **255 req/s**，平均 batch=3.5 |
| 主进程已使用 PyTorch 后启动 | **无死锁** |
| 完整训练流程（自对弈→训练→评估） | **全部通过** |

## 在 128 核 GPU 机器上的预期效果

1. **GPU 进程正常出现在 nvidia-smi 中**：spawn 模式确保 CUDA 在全新进程中初始化
2. **Worker CPU 利用率大幅提升**：Unix socket 通信开销极低，Worker 不再阻塞在 IPC 上
3. **GPU 利用率提升**：多 Worker 并发请求自然组成大 batch，GPU 计算效率更高
4. **可扩展到 128 个 Worker**：Unix socket 支持大量并发连接

## 使用方式

```bash
# GPU 集中推理模式（推荐用于有 GPU 的机器）
python train.py --mode standard --gpu-server --gpu-device cuda:0

# CPU 本地推理模式（无 GPU 时使用）
python train.py --mode standard

# 指定 worker 数量
python train.py --mode standard --gpu-server --workers 64
```
