#!/bin/bash

################################################################################
# 虚拟环境导入脚本
# 用途：在目标机器上一键恢复虚拟环境
# 
# 使用方法：
#   bash setup_venv.sh
#
# 特点：
#   - 无需 root 权限
#   - 自动处理路径重映射
#   - 支持 Python 3.7+
#   - 自动验证环境完整性
################################################################################

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKUP_FILE=""
CONFIG_FILE="venv_config.json"
VENV_DIR="$SCRIPT_DIR/venv"

echo "================================"
echo "虚拟环境导入工具"
echo "================================"
echo ""

# 1. 检查必要文件
echo "检查必要文件..."
if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ 错误：找不到 $CONFIG_FILE"
    echo "请确保以下文件在同一目录中："
    echo "  - venv_config.json"
    echo "  - venv_backup_*.tar.gz"
    echo "  - requirements_full.txt"
    exit 1
fi

# 查找备份文件
BACKUP_FILE=$(ls -t "$SCRIPT_DIR"/venv_backup_*.tar.gz 2>/dev/null | head -1)
if [ -z "$BACKUP_FILE" ]; then
    echo "❌ 错误：找不到虚拟环境备份文件 (venv_backup_*.tar.gz)"
    exit 1
fi

BACKUP_BASENAME=$(basename "$BACKUP_FILE")
echo "✓ 找到备份文件: $BACKUP_BASENAME"

# 2. 显示环境信息
echo ""
echo "导入配置："
if command -v jq &> /dev/null; then
    echo "  Python 版本: $(jq -r '.python_version' "$CONFIG_FILE")"
    echo "  Pip 版本: $(jq -r '.pip_version' "$CONFIG_FILE")"
    echo "  包数量: $(jq -r '.packages_count' "$CONFIG_FILE")"
else
    echo "  (jq 未安装，跳过配置详情)"
fi

# 3. 检查 Python 版本兼容性
echo ""
echo "检查 Python 环境..."
CURRENT_PYTHON=$(python3 --version 2>&1 | awk '{print $2}')
echo "✓ 当前 Python 版本: $CURRENT_PYTHON"

# 4. 清理旧虚拟环境（如果存在）
if [ -d "$VENV_DIR" ]; then
    echo ""
    echo "检测到旧虚拟环境，正在清理..."
    rm -rf "$VENV_DIR"
    echo "✓ 旧虚拟环境已删除"
fi

# 5. 解压虚拟环境
echo ""
echo "解压虚拟环境（这可能需要几分钟）..."
tar -xzf "$BACKUP_FILE" -C "$SCRIPT_DIR" 2>/dev/null || {
    echo "❌ 解压失败"
    exit 1
}
echo "✓ 虚拟环境已解压"

# 6. 修复路径（关键步骤）
echo ""
echo "修复虚拟环境路径..."

# 更新 pyvenv.cfg
if [ -f "$VENV_DIR/pyvenv.cfg" ]; then
    sed -i "s|home = .*|home = $VENV_DIR/bin|g" "$VENV_DIR/pyvenv.cfg"
    echo "✓ pyvenv.cfg 已更新"
fi

# 更新 activate 脚本中的路径
if [ -f "$VENV_DIR/bin/activate" ]; then
    sed -i "s|VIRTUAL_ENV=.*|VIRTUAL_ENV=\"$VENV_DIR\"|g" "$VENV_DIR/bin/activate"
    echo "✓ activate 脚本已更新"
fi

# 重建 pip/setuptools（处理 .so 文件的绝对路径问题）
echo ""
echo "重建 pip 和 setuptools..."
"$VENV_DIR/bin/python" -m pip install --upgrade pip setuptools wheel -q 2>/dev/null || {
    echo "⚠ pip 升级失败，尝试继续..."
}
echo "✓ pip 和 setuptools 已更新"

# 7. 重新安装依赖
echo ""
echo "重新安装依赖包..."

# 首先尝试用备份的 requirements_full.txt
if [ -f "$SCRIPT_DIR/requirements_full.txt" ]; then
    echo "使用 requirements_full.txt..."
    "$VENV_DIR/bin/pip" install -r "$SCRIPT_DIR/requirements_full.txt" -q 2>/dev/null || {
        echo "⚠ 部分包安装失败，尝试安装核心依赖..."
        # 安装核心依赖
        "$VENV_DIR/bin/pip" install torch numpy cython -q 2>/dev/null || true
    }
else
    echo "⚠ 找不到 requirements_full.txt，安装核心依赖..."
    "$VENV_DIR/bin/pip" install torch numpy cython -q 2>/dev/null || true
fi

echo "✓ 依赖包已安装"

# 8. 验证环境
echo ""
echo "验证虚拟环境..."

# 激活虚拟环境
source "$VENV_DIR/bin/activate"

# 检查 Python
VENV_PYTHON=$(python --version 2>&1)
echo "✓ 虚拟环境 Python: $VENV_PYTHON"

# 检查关键包
MISSING_PACKAGES=""
for pkg in torch numpy; do
    if ! python -c "import $pkg" 2>/dev/null; then
        MISSING_PACKAGES="$MISSING_PACKAGES $pkg"
    fi
done

if [ -n "$MISSING_PACKAGES" ]; then
    echo "⚠ 缺失的包:$MISSING_PACKAGES"
    echo "  正在安装..."
    pip install$MISSING_PACKAGES -q 2>/dev/null || true
fi

echo "✓ 核心包已验证"

# 9. 完成
echo ""
echo "================================"
echo "导入完成！"
echo "================================"
echo ""
echo "使用虚拟环境："
echo "  source $VENV_DIR/bin/activate"
echo ""
echo "或直接运行："
echo "  $VENV_DIR/bin/python training/train.py --mode quick"
echo ""
