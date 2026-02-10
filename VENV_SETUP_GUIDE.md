# Python 虚拟环境迁移完整指南

## 概述

本指南提供了三种虚拟环境迁移方案，适应不同的场景和需求。

---

## 方案 1：完整虚拟环境迁移（推荐用于相同架构）

### 适用场景
- 源机器和目标机器 CPU 架构相同（如都是 x86_64 或都是 ARM64）
- 操作系统相同（如都是 Linux）
- 需要完全复现环境

### 步骤

#### 源机器：导出虚拟环境

```bash
cd xiangqi-ai
source venv/bin/activate
bash migrate_venv.sh
```

生成文件：
- `xiangqi-ai-venv-YYYYMMDD_HHMMSS.tar.gz` — 完整迁移包（~1-2GB）
- `venv_config.json` — 配置信息
- `requirements_full.txt` — 依赖列表

#### 目标机器：导入虚拟环境

```bash
# 解压迁移包
tar -xzf xiangqi-ai-venv-*.tar.gz
cd xiangqi-ai

# 一键导入
bash setup_venv.sh

# 验证
source venv/bin/activate
python training/train.py --mode quick
```

### 优点
- 完全复现源环境
- 包括所有编译的 C 扩展（.so 文件）
- 迁移速度快（无需重新编译）

### 缺点
- 文件体积大（~1-2GB）
- 必须架构相同
- 传输可能较慢

---

## 方案 2：轻量级迁移（推荐用于不同架构）

### 适用场景
- 源机器和目标机器架构不同（x86_64 vs ARM64）
- 需要减小迁移包体积
- 目标机器有网络连接

### 步骤

#### 源机器：导出依赖列表

```bash
cd xiangqi-ai
source venv/bin/activate

# 导出依赖
pip freeze > requirements_full.txt

# 导出核心依赖（可选，体积更小）
pip freeze | grep -E 'torch|numpy|cython' > requirements_core.txt
```

#### 目标机器：创建虚拟环境

```bash
# 进入项目目录
cd xiangqi-ai

# 创建虚拟环境
python3 -m venv venv

# 激活虚拟环境
source venv/bin/activate

# 安装依赖（选择一个）
# 方案 A：完整依赖
pip install -r requirements_full.txt

# 方案 B：核心依赖（更快）
pip install -r requirements_core.txt

# 方案 C：手动安装（最灵活）
pip install torch numpy cython

# 编译 Cython 扩展（可选，加速 ~100x）
cd training/cython_engine
python setup.py build_ext --inplace
cd ../..

# 验证
python training/train.py --mode quick
```

### 优点
- 文件体积小（仅 ~100KB）
- 支持不同架构
- 自动适配目标机器

### 缺点
- 需要网络连接下载包
- 安装时间较长（取决于网速）
- 需要编译 C 扩展

---

## 方案 3：Docker 容器迁移（推荐用于完全隔离）

### 适用场景
- 需要完全隔离的环境
- 目标机器有 Docker
- 需要在多台机器上部署

### 步骤

#### 创建 Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# 安装依赖
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# 复制项目
COPY . .

# 创建虚拟环境并安装依赖
RUN python -m venv venv && \
    . venv/bin/activate && \
    pip install --upgrade pip && \
    pip install -r requirements_full.txt

# 编译 Cython 扩展
RUN . venv/bin/activate && \
    cd training/cython_engine && \
    python setup.py build_ext --inplace

# 设置入口点
ENV PATH="/app/venv/bin:$PATH"
CMD ["python", "training/train.py", "--mode", "quick"]
```

#### 构建和运行

```bash
# 构建镜像
docker build -t xiangqi-ai:latest .

# 运行容器
docker run --gpus all -it xiangqi-ai:latest

# 或在容器中启动训练
docker run --gpus all \
    -v $(pwd)/models:/app/models \
    xiangqi-ai:latest \
    python training/train.py --mode standard --gpu-server
```

### 优点
- 完全隔离的环境
- 易于在多台机器部署
- 支持 GPU 加速
- 无需担心系统依赖

### 缺点
- 需要 Docker
- 镜像体积较大（~2-3GB）
- 学习曲线陡

---

## 方案 4：Conda 环境迁移（推荐用于复杂依赖）

### 适用场景
- 已安装 Conda/Miniconda
- 需要管理复杂的依赖关系
- 需要多个 Python 版本

### 步骤

#### 源机器：导出 Conda 环境

```bash
# 激活虚拟环境
conda activate xiangqi

# 导出环境
conda env export > environment.yml

# 或导出为 pip 格式（更轻量）
pip freeze > requirements.txt
```

#### 目标机器：导入 Conda 环境

```bash
# 方案 A：使用 environment.yml（完整）
conda env create -f environment.yml

# 方案 B：创建新环境并安装（推荐）
conda create -n xiangqi python=3.11
conda activate xiangqi
pip install -r requirements.txt

# 编译 Cython 扩展
cd training/cython_engine
python setup.py build_ext --inplace
```

### 优点
- 自动管理依赖关系
- 支持多个 Python 版本
- 易于切换环境

### 缺点
- 需要 Conda
- 环境文件可能包含系统特定信息
- 导出文件较大

---

## 快速参考

| 方案 | 体积 | 速度 | 架构兼容 | 复杂度 |
|------|------|------|----------|--------|
| 完整迁移 | ~1-2GB | ⚡⚡⚡ | ❌ | 低 |
| 轻量级迁移 | ~100KB | ⚡ | ✅ | 中 |
| Docker | ~2-3GB | ⚡⚡ | ✅ | 高 |
| Conda | ~500MB | ⚡⚡ | ✅ | 中 |

---

## 常见问题

### Q: 哪种方案最推荐？

A: 取决于你的场景：
- **同架构、快速迁移**：方案 1（完整迁移）
- **不同架构、有网络**：方案 2（轻量级迁移）
- **需要隔离、有 Docker**：方案 3（Docker）
- **已有 Conda**：方案 4（Conda）

### Q: 迁移后出现 ImportError，怎么办？

A: 通常是 C 扩展不兼容。解决方案：

```bash
# 重新编译 Cython 扩展
cd training/cython_engine
python setup.py build_ext --inplace

# 或重新安装 torch/numpy
pip install --force-reinstall torch numpy
```

### Q: 能在 HPC 集群上使用吗？

A: 可以，建议：

```bash
# 加载 Python 模块
module load python/3.11

# 创建虚拟环境
python -m venv venv

# 激活并安装
source venv/bin/activate
pip install -r requirements_full.txt

# 在 SLURM 脚本中使用
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=128
#SBATCH --gpus-per-task=1

source venv/bin/activate
python training/train.py --mode full --gpu-server
```

### Q: 虚拟环境能共享吗？

A: 不推荐。虽然技术上可行，但会导致权限问题。建议每个用户都有自己的虚拟环境。

### Q: 如何在没有网络的机器上安装？

A: 使用完整迁移方案（方案 1），或预先下载所有 wheel 文件。

---

## 脚本文件说明

| 脚本 | 用途 | 使用场景 |
|------|------|----------|
| `migrate_venv.sh` | 导出完整虚拟环境 | 方案 1 |
| `setup_venv.sh` | 导入虚拟环境 | 方案 1 |
| `export_venv.sh` | 手动导出虚拟环境 | 方案 1（手动） |

---

## 性能优化

### 加快安装速度

```bash
# 使用清华镜像（中国用户）
pip install -i https://pypi.tsinghua.edu.cn/simple -r requirements_full.txt

# 使用阿里镜像
pip install -i http://mirrors.aliyun.com/pypi/simple/ -r requirements_full.txt

# 优先使用预编译包
pip install --prefer-binary -r requirements_full.txt

# 并行安装（使用 pip-tools）
pip install pip-tools
pip-compile requirements.txt
pip-sync
```

### 减小虚拟环境体积

```bash
# 清理 pip 缓存
pip cache purge

# 删除不必要的文件
find venv -name '*.pyc' -delete
find venv -name '__pycache__' -type d -exec rm -rf {} +

# 删除测试文件
find venv -name 'tests' -type d -exec rm -rf {} +
find venv -name '*.dist-info' -type d -exec rm -rf {} +
```

---

## 支持的平台

| 平台 | 状态 | 备注 |
|------|------|------|
| Linux (x86_64) | ✅ | 完全支持 |
| Linux (ARM64) | ✅ | 需要重编译 Cython |
| macOS (Intel) | ✅ | 完全支持 |
| macOS (Apple Silicon) | ✅ | 需要重编译 Cython |
| Windows (WSL2) | ✅ | 在 WSL2 中运行 |
| Windows (Native) | ⚠️ | 需要修改脚本为 .bat |
| HPC 集群 | ✅ | 使用 module load |

---

## 许可证

MIT License
