# Python 虚拟环境迁移指南

## 快速开始（3 步）

### 1️⃣ 源机器：导出虚拟环境

```bash
# 进入项目目录
cd xiangqi-ai

# 激活虚拟环境
source venv/bin/activate

# 导出迁移包
bash migrate_venv.sh
```

✅ 生成文件：`xiangqi-ai-venv-YYYYMMDD_HHMMSS.tar.gz`

### 2️⃣ 传输到目标机器

```bash
# 使用 scp
scp xiangqi-ai-venv-*.tar.gz user@target-host:/path/to/

# 或其他方式（FTP、U盘、云盘等）
```

### 3️⃣ 目标机器：导入虚拟环境

```bash
# 解压迁移包
tar -xzf xiangqi-ai-venv-*.tar.gz
cd xiangqi-ai

# 一键导入
bash setup_venv.sh

# 激活虚拟环境
source venv/bin/activate

# 验证
python training/train.py --mode quick
```

✅ 完成！无需 root 权限。

---

## 详细说明

### 什么是虚拟环境？

Python 虚拟环境是一个独立的 Python 环境，包含：
- Python 解释器副本
- pip 包管理器
- 所有已安装的第三方包（torch、numpy 等）

**优点**：
- 不影响系统 Python
- 可以在没有 root 权限的机器上使用
- 便于迁移和复现环境

### 迁移包包含什么？

```
xiangqi-ai-venv-20240209_120000.tar.gz
├── xiangqi-ai/
│   ├── training/              # 训练代码
│   ├── web/                   # Web 界面
│   ├── venv_backup.tar.gz     # 压缩的虚拟环境
│   ├── venv_config.json       # 配置信息
│   ├── requirements_full.txt  # 依赖列表
│   ├── setup_venv.sh          # 导入脚本
│   └── ...
```

### 脚本工作原理

#### `migrate_venv.sh`（导出）

1. 检查虚拟环境激活状态
2. 复制项目文件
3. 导出依赖列表（`requirements_full.txt`）
4. 生成配置文件（`venv_config.json`）
5. 压缩虚拟环境（`venv_backup.tar.gz`）
6. 打包所有文件

#### `setup_venv.sh`（导入）

1. 检查必要文件
2. 解压虚拟环境
3. **修复路径**（关键步骤）
   - 更新 `pyvenv.cfg`
   - 更新 `bin/activate` 脚本
   - 重建 pip 和 setuptools
4. 重新安装依赖
5. 验证环境完整性

### 为什么需要修复路径？

虚拟环境中的某些文件包含绝对路径（如 `/home/user/xiangqi-ai/venv`）。当虚拟环境被迁移到不同路径时，这些路径会失效。

导入脚本会自动将所有路径更新为新的位置。

---

## 常见场景

### 场景 1：从笔记本迁移到服务器

```bash
# 笔记本上
cd ~/xiangqi-ai
source venv/bin/activate
bash migrate_venv.sh

# 上传到服务器
scp xiangqi-ai-venv-*.tar.gz user@server.com:/home/user/

# 服务器上
cd /home/user
tar -xzf xiangqi-ai-venv-*.tar.gz
cd xiangqi-ai
bash setup_venv.sh

# 开始训练
source venv/bin/activate
python training/train.py --mode standard --gpu-server
```

### 场景 2：在 HPC 集群上使用

```bash
# 登录集群
ssh user@hpc.cluster

# 解压迁移包
tar -xzf xiangqi-ai-venv-*.tar.gz
cd xiangqi-ai
bash setup_venv.sh

# 创建 SLURM 脚本
cat > train.slurm << 'EOF'
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=128
#SBATCH --gpus-per-task=1
#SBATCH --time=24:00:00

cd /home/user/xiangqi-ai
source venv/bin/activate
python training/train.py --mode full --gpu-server --workers 120
EOF

# 提交任务
sbatch train.slurm
```

### 场景 3：多机器部署

```bash
# 源机器
bash migrate_venv.sh

# 分发到多台机器
for host in machine1 machine2 machine3; do
  scp xiangqi-ai-venv-*.tar.gz user@$host:/tmp/
  ssh user@$host "cd /tmp && tar -xzf xiangqi-ai-venv-*.tar.gz && cd xiangqi-ai && bash setup_venv.sh"
done
```

---

## 故障排除

### 问题 1：解压后虚拟环境无法激活

```bash
# 症状
source venv/bin/activate
# 报错：command not found

# 解决方案
bash setup_venv.sh  # 重新运行导入脚本
```

### 问题 2：导入后出现 ImportError

```bash
# 症状
python -c "import torch"
# 报错：ImportError: cannot open shared object file

# 原因：C 扩展架构不匹配（如 x86_64 vs ARM64）

# 解决方案
cd training/cython_engine
python setup.py build_ext --inplace
```

### 问题 3：磁盘空间不足

```bash
# 迁移包通常 1-3GB，取决于已安装的包

# 减小体积的方法
# 1. 清理 pip 缓存
pip cache purge

# 2. 删除不必要的包
pip uninstall -y <package-name>

# 3. 重新导出
bash migrate_venv.sh
```

### 问题 4：目标机器 Python 版本不同

```bash
# 检查版本
python --version

# 如果版本差异大（如 3.8 vs 3.11）
# 建议在目标机器上重新创建虚拟环境

# 方案 A：使用目标机器的 Python
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements_full.txt

# 方案 B：使用 conda（如果可用）
conda create -n xiangqi python=3.11
conda activate xiangqi
pip install -r requirements_full.txt
```

---

## 性能提示

### 加快导入速度

```bash
# 使用 --no-deps 跳过依赖检查（如果已知依赖完整）
pip install -r requirements_full.txt --no-deps

# 使用 --prefer-binary 优先使用预编译包
pip install -r requirements_full.txt --prefer-binary
```

### 减小迁移包体积

```bash
# 删除不必要的文件
rm -rf venv/lib/python*/site-packages/__pycache__
rm -rf venv/lib/python*/site-packages/*.dist-info

# 重新压缩
tar -czf xiangqi-ai-venv-slim.tar.gz venv/
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
| Windows (Native) | ⚠️ | 需要修改脚本为 .bat 格式 |

---

## 文件清单

| 文件 | 用途 |
|------|------|
| `migrate_venv.sh` | 导出虚拟环境（源机器） |
| `setup_venv.sh` | 导入虚拟环境（目标机器） |
| `export_venv.sh` | 手动导出虚拟环境 |
| `venv_config.json` | 虚拟环境配置信息 |
| `requirements_full.txt` | 完整依赖列表 |
| `VENV_MIGRATION.md` | 本文档 |
| `MIGRATION_GUIDE.md` | 详细迁移指南 |

---

## 常见问题

**Q: 迁移包能跨操作系统使用吗？**

A: 不能。虚拟环境包含编译的 C 扩展，必须与目标操作系统和架构匹配。

**Q: 能在 Docker 中使用吗？**

A: 可以，但通常不推荐。Docker 镜像本身就提供了完整的环境隔离。

**Q: 如何更新虚拟环境中的包？**

A: 激活虚拟环境后直接使用 pip：
```bash
source venv/bin/activate
pip install --upgrade torch
```

**Q: 虚拟环境能共享吗？**

A: 不推荐。虽然技术上可行，但会导致权限问题。建议每个用户都有自己的虚拟环境。

---

## 许可证

MIT License
