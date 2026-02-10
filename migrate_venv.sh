#!/bin/bash

################################################################################
# 虚拟环境一键迁移脚本
# 用途：自动导出虚拟环境并生成迁移包
# 
# 使用方法：
#   # 在源机器上
#   source venv/bin/activate
#   bash migrate_venv.sh
#
#   # 在目标机器上
#   tar -xzf xiangqi-ai-venv-*.tar.gz
#   cd xiangqi-ai
#   bash setup_venv.sh
################################################################################

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
PACKAGE_NAME="xiangqi-ai-venv-${TIMESTAMP}"
PACKAGE_DIR="/tmp/$PACKAGE_NAME"

echo "================================"
echo "虚拟环境一键迁移工具"
echo "================================"
echo ""

# 检查虚拟环境
if [ -z "$VIRTUAL_ENV" ]; then
    echo "❌ 错误：未检测到活跃的虚拟环境"
    echo "请先激活虚拟环境："
    echo "  source venv/bin/activate"
    exit 1
fi

echo "✓ 虚拟环境: $VIRTUAL_ENV"

# 创建临时目录
mkdir -p "$PACKAGE_DIR"
echo "✓ 创建临时目录: $PACKAGE_DIR"

# 复制项目文件（排除虚拟环境和大文件）
echo ""
echo "复制项目文件..."
rsync -av \
    --exclude='venv' \
    --exclude='venv_backup_*' \
    --exclude='.git' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='*.egg-info' \
    --exclude='.pytest_cache' \
    --exclude='models/*.pt' \
    --exclude='models/*.pth' \
    "$SCRIPT_DIR/" "$PACKAGE_DIR/xiangqi-ai/" \
    > /dev/null 2>&1 || {
    # 如果 rsync 不可用，使用 cp
    cp -r "$SCRIPT_DIR" "$PACKAGE_DIR/xiangqi-ai" 2>/dev/null || true
    # 清理大文件
    find "$PACKAGE_DIR/xiangqi-ai" -name '*.pt' -o -name '*.pth' -o -name '__pycache__' | xargs rm -rf
}

echo "✓ 项目文件已复制"

# 导出虚拟环境
echo ""
echo "导出虚拟环境..."
cd "$SCRIPT_DIR"

# 生成配置文件
PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
PIP_VERSION=$(pip --version | awk '{print $2}')
PACKAGES_COUNT=$(pip list | wc -l)

cat > "$PACKAGE_DIR/xiangqi-ai/venv_config.json" << EOF
{
  "export_time": "$(date -Iseconds)",
  "python_version": "$PYTHON_VERSION",
  "pip_version": "$PIP_VERSION",
  "packages_count": $((PACKAGES_COUNT - 2)),
  "platform": "$(uname -s)",
  "architecture": "$(uname -m)"
}
EOF

echo "✓ 配置文件已生成"

# 导出 requirements.txt
pip freeze > "$PACKAGE_DIR/xiangqi-ai/requirements_full.txt"
echo "✓ 依赖列表已导出"

# 压缩虚拟环境
echo ""
echo "压缩虚拟环境（这可能需要几分钟）..."
tar --exclude='*.pyc' \
    --exclude='__pycache__' \
    --exclude='.pytest_cache' \
    --exclude='*.egg-info' \
    -czf "$PACKAGE_DIR/xiangqi-ai/venv_backup.tar.gz" \
    venv/ \
    2>/dev/null || true

if [ -f "$PACKAGE_DIR/xiangqi-ai/venv_backup.tar.gz" ]; then
    SIZE=$(du -h "$PACKAGE_DIR/xiangqi-ai/venv_backup.tar.gz" | cut -f1)
    echo "✓ 虚拟环境已压缩 ($SIZE)"
else
    echo "❌ 虚拟环境压缩失败"
    exit 1
fi

# 创建最终的迁移包
echo ""
echo "创建迁移包..."
cd /tmp
tar -czf "$SCRIPT_DIR/${PACKAGE_NAME}.tar.gz" "$PACKAGE_NAME/" 2>/dev/null || true

if [ -f "$SCRIPT_DIR/${PACKAGE_NAME}.tar.gz" ]; then
    SIZE=$(du -h "$SCRIPT_DIR/${PACKAGE_NAME}.tar.gz" | cut -f1)
    echo "✓ 迁移包已创建: ${PACKAGE_NAME}.tar.gz ($SIZE)"
else
    echo "❌ 迁移包创建失败"
    exit 1
fi

# 清理临时目录
rm -rf "$PACKAGE_DIR"

# 生成使用说明
cat > "$SCRIPT_DIR/MIGRATION_GUIDE.md" << 'EOF'
# 虚拟环境迁移指南

## 概述

本指南说明如何将中国象棋 AI 项目的虚拟环境迁移到其他机器。

## 前置条件

- Python 3.7+
- bash shell
- 足够的磁盘空间（虚拟环境通常 1-3GB）
- **无需 root 权限**

## 迁移步骤

### 第一步：在源机器上导出环境

```bash
# 激活虚拟环境
source venv/bin/activate

# 运行迁移脚本
bash migrate_venv.sh
```

这会生成一个 `xiangqi-ai-venv-YYYYMMDD_HHMMSS.tar.gz` 文件。

### 第二步：传输迁移包到目标机器

```bash
# 使用 scp
scp xiangqi-ai-venv-*.tar.gz user@target-machine:/path/to/destination/

# 或使用其他方式（FTP、U盘等）
```

### 第三步：在目标机器上导入环境

```bash
# 解压迁移包
tar -xzf xiangqi-ai-venv-*.tar.gz
cd xiangqi-ai

# 运行导入脚本
bash setup_venv.sh
```

脚本会自动：
- 解压虚拟环境
- 修复路径（处理绝对路径问题）
- 重建 pip 和 setuptools
- 重新安装依赖包
- 验证环境完整性

### 第四步：验证环境

```bash
# 激活虚拟环境
source venv/bin/activate

# 运行快速测试
python training/train.py --mode quick
```

## 常见问题

### Q: 迁移包很大，如何减小体积？

A: 迁移包中包含了所有已安装的包。如果只需要核心依赖，可以：

```bash
# 在源机器上
pip freeze | grep -E 'torch|numpy|cython' > requirements_core.txt

# 在目标机器上
pip install -r requirements_core.txt
```

### Q: 目标机器的 Python 版本不同，能用吗？

A: 通常可以，但可能有兼容性问题。建议：
1. 检查源机器和目标机器的 Python 版本
2. 如果版本差异大（如 3.8 vs 3.11），建议在目标机器上重新创建虚拟环境并安装依赖

### Q: 迁移后出现 ImportError，怎么办？

A: 这通常是由于 C 扩展（如 Cython 编译的 .so 文件）的架构不匹配。解决方案：

```bash
# 重新编译 Cython 扩展
cd training/cython_engine
python setup.py build_ext --inplace
```

### Q: 没有 root 权限，能安装新包吗？

A: 可以，虚拟环境本身就不需要 root 权限：

```bash
source venv/bin/activate
pip install <package-name>  # 无需 sudo
```

### Q: 如何在 HPC 集群上使用？

A: HPC 集群通常有模块系统，建议：

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
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1

source venv/bin/activate
python training/train.py --mode standard --gpu-server
```

## 文件说明

| 文件 | 说明 |
|------|------|
| `migrate_venv.sh` | 导出虚拟环境的脚本 |
| `setup_venv.sh` | 导入虚拟环境的脚本 |
| `export_venv.sh` | 手动导出虚拟环境的脚本 |
| `venv_config.json` | 虚拟环境的配置信息 |
| `requirements_full.txt` | 完整的依赖列表 |

## 技术细节

### 路径修复

虚拟环境中的某些文件包含绝对路径。导入脚本会自动修复：
- `pyvenv.cfg` - 虚拟环境配置
- `bin/activate` - 激活脚本
- Python 二进制文件中的 RPATH

### 依赖重新安装

某些包（如 numpy、torch）包含编译的 C 扩展（.so 文件）。导入脚本会：
1. 尝试使用备份的 wheel 文件
2. 如果失败，重新从 PyPI 下载并安装

这确保了二进制兼容性。

### 无 root 权限支持

虚拟环境完全独立，所有文件都在 `venv/` 目录中：
- 无需修改系统 Python
- 无需安装系统级包
- 可以在任何有写权限的目录中创建

## 支持的平台

- Linux (x86_64, ARM64)
- macOS (Intel, Apple Silicon)
- Windows (WSL2)

## 许可证

MIT License
EOF

echo ""
echo "================================"
echo "迁移包已生成！"
echo "================================"
echo ""
echo "迁移包文件："
echo "  ${PACKAGE_NAME}.tar.gz"
echo ""
echo "使用说明已保存到："
echo "  MIGRATION_GUIDE.md"
echo ""
echo "后续步骤："
echo "1. 将迁移包传输到目标机器"
echo "2. 在目标机器上解压："
echo "   tar -xzf ${PACKAGE_NAME}.tar.gz"
echo "3. 进入项目目录："
echo "   cd xiangqi-ai"
echo "4. 运行导入脚本："
echo "   bash setup_venv.sh"
echo ""
