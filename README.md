# 中国象棋强化学习AI

基于 **AlphaZero** 算法的中国象棋AI项目，使用 PyTorch 从零开始进行强化学习训练，并提供可视化人机对战界面。

---

## 项目概述

本项目实现了完整的 AlphaZero 训练流程，包含以下核心模块：

| 模块 | 文件 | 说明 |
|------|------|------|
| 游戏引擎 | `training/game.py` | 中国象棋规则引擎，包含棋盘表示、走法生成、合法性验证 |
| 神经网络 | `training/model.py` | ResNet 策略-价值双头网络 |
| 蒙特卡洛树搜索 | `training/mcts.py` | AlphaZero 风格的 MCTS 实现 |
| 训练框架 | `training/train.py` | 自对弈 + 网络训练 + 模型评估的完整训练循环 |
| 模型导出 | `training/export_model.py` | 将模型导出为 ONNX/TorchScript 格式 |
| 对战界面 | Web 前端 | React + TypeScript 实现的人机对战界面 |

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

### 1.2 神经网络模型 (`training/model.py`)

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

### 1.3 蒙特卡洛树搜索 (`training/mcts.py`)

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

### 1.4 训练框架 (`training/train.py`)

完整的 AlphaZero 训练循环：

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
```

### 2.2 训练模型

提供三种训练模式：

```bash
# 快速测试模式（约10分钟，验证流程是否正常）
python training/train.py --mode quick

# 标准训练模式（约数小时，适合单GPU）
python training/train.py --mode standard

# 完整训练模式（约数天，需要强力GPU）
python training/train.py --mode full
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
    --device cuda
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
| `--device` | auto | 计算设备 (cpu/cuda) |
| `--resume` | None | 恢复训练的检查点路径 |

### 2.4 训练建议

| 训练阶段 | 建议配置 | 预期效果 |
|----------|----------|----------|
| 初期验证 | channels=64, blocks=3, sims=50 | 验证代码正确性 |
| 基础训练 | channels=128, blocks=6, sims=200 | 学会基本规则和简单战术 |
| 深度训练 | channels=256, blocks=10, sims=400 | 掌握中级战术和开局 |
| 高级训练 | channels=256, blocks=20, sims=800 | 接近业余高手水平 |

**硬件需求**：
- 快速测试：CPU 即可，约 10-30 分钟
- 标准训练：建议 GPU（RTX 3060 以上），约 6-24 小时
- 完整训练：建议多 GPU 或 TPU，约 3-7 天

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
xiangqi-ai/
├── training/                  # 训练代码
│   ├── game.py               # 游戏引擎（棋盘、走法、规则）
│   ├── model.py              # 神经网络模型（ResNet双头）
│   ├── mcts.py               # 蒙特卡洛树搜索
│   ├── train.py              # AlphaZero训练框架
│   └── export_model.py       # 模型导出工具
├── models/                    # 模型存储目录
│   ├── best_model.pt         # 最优模型
│   ├── checkpoint_iter*.pt   # 训练检查点
│   └── training_stats.json   # 训练统计数据
├── requirements.txt           # Python依赖
└── README.md                  # 本文档
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

## 七、参考资料

- Silver, D., et al. "Mastering the game of Go without human knowledge." Nature, 2017.
- Silver, D., et al. "A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play." Science, 2018.
