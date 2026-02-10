#!/bin/bash

################################################################################
# 虚拟环境导出脚本
# 用途：将当前虚拟环境导出为可迁移的压缩包
# 
# 使用方法：
#   bash export_venv.sh
#
# 输出：
#   venv_backup_<timestamp>.tar.gz  - 虚拟环境压缩包
#   venv_config.json                 - 环境配置文件（用于验证）
################################################################################

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_NAME="venv_backup_${TIMESTAMP}.tar.gz"
CONFIG_FILE="venv_config.json"

echo "================================"
echo "虚拟环境导出工具"
echo "================================"

# 检查是否在虚拟环境中
if [ -z "$VIRTUAL_ENV" ]; then
    echo "❌ 错误：未检测到活跃的虚拟环境"
    echo "请先激活虚拟环境："
    echo "  source venv/bin/activate"
    exit 1
fi

echo "✓ 虚拟环境路径: $VIRTUAL_ENV"

# 收集环境信息
PYTHON_VERSION=$(python --version 2>&1)
PIP_VERSION=$(pip --version)
PACKAGES_COUNT=$(pip list | wc -l)

echo "✓ Python 版本: $PYTHON_VERSION"
echo "✓ Pip 版本: $PIP_VERSION"
echo "✓ 已安装包数: $((PACKAGES_COUNT - 2))"

# 生成配置文件
cat > "$SCRIPT_DIR/$CONFIG_FILE" << EOF
{
  "export_time": "$(date -Iseconds)",
  "python_version": "$(python --version 2>&1 | awk '{print $2}')",
  "pip_version": "$(pip --version | awk '{print $2}')",
  "packages_count": $((PACKAGES_COUNT - 2)),
  "platform": "$(uname -s)",
  "architecture": "$(uname -m)",
  "backup_file": "$BACKUP_NAME"
}
EOF

echo "✓ 配置文件已生成: $CONFIG_FILE"

# 导出 requirements.txt
echo ""
echo "导出依赖列表..."
pip freeze > "$SCRIPT_DIR/requirements_full.txt"
echo "✓ 完整依赖列表: requirements_full.txt ($(wc -l < "$SCRIPT_DIR/requirements_full.txt") 个包)"

# 导出虚拟环境
echo ""
echo "压缩虚拟环境（这可能需要几分钟）..."
cd "$SCRIPT_DIR"
tar --exclude='*.pyc' \
    --exclude='__pycache__' \
    --exclude='.pytest_cache' \
    --exclude='*.egg-info' \
    -czf "$BACKUP_NAME" \
    venv/ \
    2>/dev/null || true

if [ -f "$BACKUP_NAME" ]; then
    SIZE=$(du -h "$BACKUP_NAME" | cut -f1)
    echo "✓ 虚拟环境已压缩: $BACKUP_NAME ($SIZE)"
else
    echo "❌ 压缩失败"
    exit 1
fi

echo ""
echo "================================"
echo "导出完成！"
echo "================================"
echo ""
echo "迁移步骤："
echo "1. 将以下文件复制到目标机器："
echo "   - $BACKUP_NAME"
echo "   - $CONFIG_FILE"
echo "   - requirements_full.txt"
echo "   - setup_venv.sh"
echo ""
echo "2. 在目标机器上运行："
echo "   bash setup_venv.sh"
echo ""
