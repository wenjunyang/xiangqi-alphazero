# 中国象棋强化学习AI

基于 **AlphaZero** 算法的中国象棋AI项目，使用 PyTorch 从零开始进行强化学习训练，并提供可视化人机对战界面。

---

## 项目概述

本项目实现了完整的 AlphaZero 训练流程，包含以下核心模块：

| 模块 | 文件 | 说明 |
|------|------|------|
| 游戏引擎 | `training/game.py` | 中国象棋规则引擎（支持 Cython 加速） |
| Cython 加速 | `training/cython_engine/` | C 扩展棋盘引擎，走法生成提速 ~110x |
| 神经网络 | `training/model.py` | ResNet 策略-价值双头网络 |
| 蒙特卡洛树搜索 | `training/mcts.py` | AlphaZero 风格的 MCTS 实现 |
| 训练框架 | `training/train.py` | 自对弈 + 网络训练 + 模型评估的完整训练循环 |
| 并行自对弈 | `training/parallel_selfplay.py` | 多进程并行自对弈，支持 CPU/GPU 两种推理模式 |
| GPU 推理服务 | `training/inference_server.py` | 集中式 GPU 批量推理服务 |
| 模型导出 | `training/export_model.py` | 将模型导出为 ONNX/TorchScript 格式 |
| 对战界面 | `web/` | React + TypeScript 实现的人机对战界面 |

---

## 一、训练代码详解

### 1.1 游戏引擎 (`training/game.py`)

**棋盘表示**采用 10×9 的二维 NumPy 数组，使用整数编码棋子：

| 编码 | 红方 | 编码 | 黑方 |
|------|------|------|------|
| 1 | 帅 | -1 | 将 |
| 2 | 仕 | -2 | 士 |
| 3 | 相 | -3 | 象 |
| 4 | 马 | -4 | 马 |
| 5 | 车 | -5 | 车 |
| 6 | 炮 | -6 | 炮 |
| 7 | 兵 | -7 | 卒 |

**动作空间编码**：每个走法 `(from_row, from_col, to_row, to_col)` 编码为 `from_pos × 90 + to_pos`，其中 `pos = row × 9 + col`。总动作空间大小为 **8100**。

**神经网络输入特征**：15 个特征平面（每个 10×9）：
- 平面 0-6：当前玩家的 7 种棋子（二值化）
- 平面 7-13：对手的 7 种棋子（二值化）
- 平面 14：当前玩家标识（全 1 = 红方，全 0 = 黑方）

**核心方法**：
- `get_legal_moves()` — 生成所有合法走法（含将军/飞将检测）
- `get_state_for_nn()` — 将棋盘转换为神经网络输入
- `is_game_over()` — 判断游戏是否结束（将死/和棋/超步数）

### 1.2 Cython 加速引擎 (`training/cython_engine/`)

将走法生成、将军检测等热点函数用 Cython 重写为 C 扩展，**自动检测并加载**（编译失败时自动回退到纯 Python）。

**性能对比**：

| 函数 | 纯 Python | Cython | 加速倍数 |
|------|-----------|--------|----------|
| `get_legal_moves` | 0.612 ms | 0.006 ms | **110x** |
| `is_attacked` | 0.190 ms | 0.007 ms | **27x** |
| MCTS 场景 (1000次调用) | 4733 ms | 46 ms | **103x** |

**编译方法**：

```bash
cd training/cython_engine
python setup.py build_ext --inplace
```

需要安装 `cython` 和 C 编译器（`gcc`/`python3-dev`）。

### 1.3 神经网络模型 (`training/model.py`)

采用 **ResNet 策略-价值双头网络**，参考 AlphaZero 论文架构：

```
输入 (15 × 10 × 9)
    │
    ▼
┌─────────────────┐
│  Conv2d 3×3      │  15 → num_channels
│  BatchNorm + ReLU│
└─────────────────┘
    │
    ▼
┌─────────────────┐
│  Residual Block  │  × num_res_blocks
│  (2× Conv + BN)  │
└─────────────────┘
    │
    ├──────────────────────┐
    ▼                      ▼
┌──────────┐        ┌──────────┐
│ 策略头    │        │ 价值头    │
│ Conv 1×1  │        │ Conv 1×1  │
│ FC → 8100 │        │ FC → 128  │
│           │        │ FC → 1    │
│ (logits)  │        │ Tanh      │
└──────────┘        └──────────┘
```

**可配置参数**：
- `num_channels`：残差块通道数（默认 128，推荐 128-256）
- `num_res_blocks`：残差块数量（默认 6，推荐 6-20）

### 1.4 蒙特卡洛树搜索 (`training/mcts.py`)

实现 AlphaZero 风格的 MCTS，核心流程：

1. **选择 (Select)**：从根节点沿 UCB 值最高的路径向下选择
2. **扩展 (Expand)**：到达叶节点时，用神经网络评估并扩展子节点
3. **回溯 (Backup)**：将评估值沿路径回传更新

**UCB 公式**：

```
UCB(s, a) = Q(s, a) + c_puct × P(s, a) × √N(s) / (1 + N(s, a))
```

其中：
- `Q(s, a)` — 动作平均价值
- `P(s, a)` — 神经网络给出的先验概率
- `N(s)` — 父节点访问次数
- `N(s, a)` — 子节点访问次数
- `c_puct` — 探索常数（默认 1.5）

**根节点 Dirichlet 噪声**：`P'(s, a) = 0.75 × P(s, a) + 0.25 × η`，其中 `η ~ Dir(0.3)`

### 1.5 训练框架 (`training/train.py`)

完整的 AlphaZero 训练循环，包含以下改进机制：

- **多进程并行自对弈**：充分利用多核 CPU 加速数据收集
- **GPU 集中推理服务**：可选的 GPU 批量推理模式，所有 worker 共享一个 GPU 推理进程
- **材料差判定**：超步数时根据双方棋子材料分差判定胜负，避免训练早期全部和棋
- **认输机制**：当价值网络连续评估极低时提前结束对局，节省训练时间
- **随机开局**：前 0-N 步随机走法，增加训练数据多样性
- **温度退火**：温度从 1.0 线性降至 0.3，平衡探索与利用

### 1.6 并行架构 (`training/parallel_selfplay.py`)

支持两种推理模式：

**模式 1：CPU 本地推理（默认）**

```
主进程 → 序列化模型权重 → 分发到 N 个 worker
每个 worker → 重建模型(CPU) → 串行完成 K 局 → 返回数据
主进程 → 汇总数据 → 训练网络
```

每个 worker 独立持有模型副本，在 CPU 上本地推理。适合没有 GPU 或 GPU 显存不足的场景。

**模式 2：GPU 集中推理**

```
主进程 → 启动 InferenceServer(GPU) + N 个 worker
Worker 1 ──┐
Worker 2 ──┤  通过 Manager().Queue() 发送状态
Worker 3 ──┼──→ InferenceServer（单进程，GPU 批量推理）
Worker N ──┘    收集请求 → 组 batch → GPU 推理 → 分发结果
```

所有 worker 的推理请求汇聚到一个 GPU 推理进程，组成大 batch 一次性推理。适合有 GPU 的场景，batch 越大 GPU 利用率越高。

**GPU 推理服务关键设计**（`training/inference_server.py`）：
- 使用 `multiprocessing.Manager().Queue()` 创建可序列化的代理队列，兼容 `spawn` 模式
- 自动收集请求组 batch，可配置 `max_batch_size` 和 `batch_timeout_ms`
- 支持运行中更新模型权重（`server.update_model(state_dict)`）
- `InferenceClient` 兼容 `model.predict()` 接口，MCTS 无需修改

训练循环结构：

```
┌─────────────────────────────────────────────────┐
│                 训练主循环                         │
│                                                   │
│  for iteration in 1..N:                          │
│    ┌─────────────────────────────────┐           │
│    │ 1. 自对弈 (Self-Play)            │           │
│    │    - 使用当前最优模型 + MCTS      │           │
│    │    - 生成 (state, policy, value)  │           │
│    │    - 数据增强（水平翻转）          │           │
│    └─────────────────────────────────┘           │
│                    │                              │
│                    ▼                              │
│    ┌─────────────────────────────────┐           │
│    │ 2. 网络训练 (Train)              │           │
│    │    - 策略损失: 交叉熵             │           │
│    │    - 价值损失: MSE               │           │
│    │    - Adam优化器 + 梯度裁剪        │           │
│    └─────────────────────────────────┘           │
│                    │                              │
│                    ▼                              │
│    ┌─────────────────────────────────┐           │
│    │ 3. 模型评估 (Evaluate)           │           │
│    │    - 新模型 vs 旧模型对弈         │           │
│    │    - 胜率 > 55% 则更新最优模型    │           │
│    └─────────────────────────────────┘           │
│                    │                              │
│                    ▼                              │
│    ┌─────────────────────────────────┐           │
│    │ 4. 保存检查点                     │           │
│    └─────────────────────────────────┘           │
└─────────────────────────────────────────────────┘
```

---

## 二、快速开始

### 2.1 环境安装

```bash
pip install torch numpy

# 可选：编译 Cython 加速引擎（推荐，提速 ~100x）
pip install cython
sudo apt-get install python3-dev gcc  # Ubuntu
cd training/cython_engine
python setup.py build_ext --inplace
```

### 2.2 训练模型

提供三种训练模式：

```bash
# 快速测试模式（约10分钟，验证流程是否正常）
python training/train.py --mode quick

# 标准训练模式（约数小时）
python training/train.py --mode standard

# 完整训练模式（约数天，需要强力硬件）
python training/train.py --mode full
```

**使用 GPU 集中推理服务**（推荐有 GPU 时使用）：

```bash
# 启用 GPU 推理服务
python training/train.py --mode standard --gpu-server

# 指定 GPU 设备
python training/train.py --mode full --gpu-server --gpu-device cuda:0

# 128 核 + GPU 推荐配置
python training/train.py \
    --mode full \
    --games-per-iter 128 \
    --workers 120 \
    --gpu-server \
    --gpu-device cuda:0
```

**自定义参数**：

```bash
python training/train.py \
    --mode standard \
    --iterations 100 \
    --games-per-iter 30 \
    --simulations 400 \
    --channels 256 \
    --res-blocks 10 \
    --workers 8

# 禁用并行（调试用）
python training/train.py --mode quick --no-parallel
```

**从检查点恢复训练**：

```bash
python training/train.py --resume models/checkpoint_iter50.pt
```

### 2.3 训练参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--mode` | quick | 训练模式：quick/standard/full |
| `--iterations` | 视模式 | 总训练轮数 |
| `--games-per-iter` | 视模式 | 每轮自对弈局数 |
| `--simulations` | 视模式 | MCTS 每步模拟次数 |
| `--channels` | 视模式 | 网络通道数 |
| `--res-blocks` | 视模式 | 残差块数量 |
| `--device` | auto | 训练设备 (cpu/cuda) |
| `--workers` | auto | 并行 worker 进程数 |
| `--gpu-server` | false | 启用 GPU 集中推理服务 |
| `--gpu-device` | cuda | GPU 推理设备 (如 cuda:0) |
| `--no-parallel` | false | 禁用并行自对弈 |
| `--resume` | None | 恢复训练的检查点路径 |

### 2.4 训练建议

| 训练阶段 | 建议配置 | 预期效果 |
|----------|----------|----------|
| 初期验证 | channels=64, blocks=3, sims=50 | 验证代码正确性 |
| 基础训练 | channels=128, blocks=6, sims=200 | 学会基本规则和简单战术 |
| 深度训练 | channels=256, blocks=10, sims=400 | 掌握中级战术和开局 |
| 高级训练 | channels=256, blocks=20, sims=800 | 接近业余高手水平 |

**硬件推荐**：

| 场景 | CPU | GPU | 推理模式 | 预计时间 |
|------|-----|-----|----------|----------|
| 快速测试 | 4核+ | 无需 | CPU 本地 | 30-60 分钟 |
| 标准训练 | 8核+ | RTX 3060+ | GPU 集中推理 | 6-24 小时 |
| 完整训练 | 32核+ | RTX 3090+ | GPU 集中推理 | 3-7 天 |
| 高性能 | 128核 | A100 | GPU 集中推理 | 1-3 天 |

---

## 三、模型导出

训练完成后，可以将模型导出为 ONNX 格式：

```bash
# 导出为 ONNX（推荐，可在Web端使用）
python training/export_model.py --model models/best_model.pt --output models/model.onnx

# 导出为 TorchScript
python training/export_model.py --model models/best_model.pt --output models/model.ts --format torchscript
```

---

## 四、人机对战界面

Web 演示界面基于 React + TypeScript 构建，提供完整的人机对战功能。

### 功能特性

- 精美的木质棋馆风格界面
- 完整的中国象棋规则实现
- 支持选择执红/执黑
- 4 级 AI 难度（随机/入门/业余/进阶）
- 走法高亮和合法走法提示
- 悔棋功能
- 走法历史记录

### 技术架构

界面内置了基于 **Minimax + Alpha-Beta 剪枝** 的搜索引擎作为默认 AI。如需使用训练好的神经网络模型，可将模型导出为 ONNX 格式后，通过 ONNX Runtime Web 在浏览器端加载推理。

---

## 五、项目文件结构

```
xiangqi-alphazero/
├── training/                      # 训练代码
│   ├── game.py                   # 游戏引擎（自动加载 Cython 加速）
│   ├── model.py                  # 神经网络模型（ResNet双头）
│   ├── mcts.py                   # 蒙特卡洛树搜索
│   ├── train.py                  # AlphaZero训练框架
│   ├── parallel_selfplay.py      # 多进程并行自对弈
│   ├── inference_server.py       # GPU 集中推理服务
│   ├── export_model.py           # 模型导出工具
│   ├── benchmark.py              # 性能剖析工具
│   └── cython_engine/            # Cython C 扩展
│       ├── game_core.pyx         # Cython 源码
│       ├── setup.py              # 编译脚本
│       └── __init__.py
├── web/                           # Web 对战界面
│   ├── client/src/
│   │   ├── lib/xiangqi-engine.ts # TS版象棋引擎
│   │   ├── hooks/useXiangqi.ts   # 游戏状态管理
│   │   ├── components/
│   │   │   ├── XiangqiBoard.tsx  # 棋盘渲染
│   │   │   ├── GamePanel.tsx     # 控制面板
│   │   │   └── MoveHistory.tsx   # 走法记录
│   │   └── pages/Home.tsx        # 主页面
│   ├── package.json
│   └── README.md
├── models/                        # 模型存储目录
├── requirements.txt               # Python依赖
└── README.md                      # 本文档
```

---

## 六、算法原理

### AlphaZero 核心思想

AlphaZero 是 DeepMind 提出的通用棋类 AI 算法，其核心创新在于：

1. **从零开始学习**：不需要任何人类棋谱数据，完全通过自对弈学习
2. **神经网络 + MCTS**：用神经网络指导蒙特卡洛树搜索，实现高效的局面评估和走法选择
3. **策略-价值双头网络**：一个网络同时输出走法概率分布和局面评估值

### 训练数据生成

每局自对弈生成的训练样本格式为 `(s, π, z)`：
- `s`：棋盘状态特征（15 × 10 × 9）
- `π`：MCTS 搜索得到的走法概率分布（8100 维）
- `z`：最终对局结果（+1 赢，-1 输，0 和）

### 损失函数

```
L = (z - v)² - π^T · log(p) + c · ||θ||²
```

其中 `v` 是价值头输出，`p` 是策略头输出，`c` 是 L2 正则化系数。

---

## 七、性能优化总结

本项目经过多轮性能优化，以下是各优化措施的效果：

| 优化项 | 优化前 | 优化后 | 提升 |
|--------|--------|--------|------|
| Cython 走法生成 | 0.612 ms | 0.006 ms | 110x |
| 反向将军检测 | 6.43 ms | 0.007 ms | 860x |
| 走法缓存 | 6.25 ms | 0.003 ms | 2083x |
| NumPy 向量化特征提取 | 0.064 ms | 0.022 ms | 2.9x |
| 多进程并行自对弈 | 串行 | N 核并行 | ~Nx |
| GPU 集中推理 | CPU 分散推理 | GPU 批量推理 | 取决于 batch 大小 |

---

## 八、参考资料

- Silver, D., et al. "Mastering the game of Go without human knowledge." Nature, 2017.
- Silver, D., et al. "A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play." Science, 2018.
