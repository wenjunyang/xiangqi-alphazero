"""
中国象棋 AlphaZero AI 演示界面
使用 Streamlit 构建交互式演示
"""

import os
import sys
import glob
from pathlib import Path
from typing import Optional, Tuple, List

import streamlit as st
import numpy as np
import torch

# 添加 training 目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'training'))

from game import XiangqiGame, decode_action, encode_action, PIECE_NAMES
from model import XiangqiNet
from mcts import MCTS


# ============================================================
# 配置
# ============================================================

MODELS_DIR = Path(__file__).parent / 'models'
DEFAULT_MODEL_PATH = MODELS_DIR / 'best_model.pt'

# 棋盘样式
BOARD_STYLE = """
<style>
.chess-board {
    font-family: 'KaiTi', 'STKaiti', serif;
    margin: 20px auto;
}
.piece {
    width: 50px;
    height: 50px;
    border-radius: 50%;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-size: 24px;
    font-weight: bold;
    cursor: pointer;
    margin: 2px;
}
.piece-red {
    background: linear-gradient(135deg, #ff6b6b 0%, #ee5a6f 100%);
    color: white;
    border: 2px solid #c92a2a;
    box-shadow: 0 2px 4px rgba(0,0,0,0.2);
}
.piece-black {
    background: linear-gradient(135deg, #495057 0%, #343a40 100%);
    color: white;
    border: 2px solid #212529;
    box-shadow: 0 2px 4px rgba(0,0,0,0.2);
}
.piece-selected {
    border: 3px solid #ffd43b !important;
    box-shadow: 0 0 15px rgba(255, 212, 59, 0.8) !important;
}
.empty-cell {
    width: 50px;
    height: 50px;
    display: inline-block;
    margin: 2px;
}
.move-hint {
    background: rgba(76, 175, 80, 0.3);
    border: 2px dashed #4caf50;
}
</style>
"""


# ============================================================
# 工具函数
# ============================================================

def discover_models(models_dir: Path) -> List[Path]:
    """自动发现模型文件"""
    if not models_dir.exists():
        return []
    return sorted(models_dir.glob('*.pt'), key=lambda p: p.stat().st_mtime, reverse=True)


def load_model(model_path: Path, device: str = 'cpu') -> Tuple[XiangqiNet, dict]:
    """加载模型"""
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # 兼容不同的检查点格式
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        config = checkpoint.get('config', {})
    else:
        state_dict = checkpoint
        config = {}
    
    # 默认配置
    num_channels = config.get('num_channels', 128)
    num_res_blocks = config.get('num_res_blocks', 6)
    
    model = XiangqiNet(num_channels=num_channels, num_res_blocks=num_res_blocks)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    return model, config


def format_move(action_idx: int) -> str:
    """格式化走法为中文"""
    fr, fc, tr, tc = decode_action(action_idx)
    
    # 转换为中国象棋坐标（红方视角）
    file_names = ['九', '八', '七', '六', '五', '四', '三', '二', '一']
    
    return f"({fr},{fc})→({tr},{tc})"


def get_top_moves(policy: np.ndarray, game: XiangqiGame, top_k: int = 10) -> List[Tuple[int, float, bool]]:
    """
    获取 top-k 走法及其概率
    
    Returns:
        List of (action_idx, probability, is_legal)
    """
    legal_actions = set()
    for fr, fc, tr, tc in game.get_legal_moves():
        legal_actions.add(encode_action(fr, fc, tr, tc))
    
    # 获取 top-k
    top_indices = np.argsort(policy)[::-1][:top_k]
    
    results = []
    for idx in top_indices:
        prob = float(policy[idx])
        is_legal = int(idx) in legal_actions
        results.append((int(idx), prob, is_legal))
    
    return results


# ============================================================
# Streamlit 界面
# ============================================================

def init_session_state():
    """初始化 session state"""
    if 'game' not in st.session_state:
        st.session_state.game = XiangqiGame()
    if 'selected_piece' not in st.session_state:
        st.session_state.selected_piece = None
    if 'ai_side' not in st.session_state:
        st.session_state.ai_side = -1  # -1: 黑方, 1: 红方
    if 'model' not in st.session_state:
        st.session_state.model = None
    if 'mcts' not in st.session_state:
        st.session_state.mcts = None
    if 'last_policy' not in st.session_state:
        st.session_state.last_policy = None
    if 'last_value' not in st.session_state:
        st.session_state.last_value = None
    if 'last_action' not in st.session_state:
        st.session_state.last_action = None
    if 'game_over' not in st.session_state:
        st.session_state.game_over = False
    if 'winner' not in st.session_state:
        st.session_state.winner = None


def render_board(game: XiangqiGame):
    """渲染棋盘（简化版本，使用文本表示）"""
    board = game.board
    
    # 使用 DataFrame 渲染棋盘
    import pandas as pd
    
    board_display = []
    for r in range(10):
        row = []
        for c in range(9):
            piece = board[r, c]
            if piece != 0:
                piece_name = PIECE_NAMES.get(abs(piece), '?')
                if piece > 0:
                    row.append(f"🔴{piece_name}")
                else:
                    row.append(f"⚫{piece_name}")
            else:
                row.append("·")
        board_display.append(row)
    
    df = pd.DataFrame(board_display, 
                     columns=[str(i) for i in range(9)],
                     index=[str(i) for i in range(10)])
    
    st.dataframe(df, use_container_width=True, height=400)
    



def render_model_output(policy: np.ndarray, value: float, game: XiangqiGame, 
                       last_action: Optional[int] = None):
    """渲染模型输出"""
    st.subheader("🤖 模型输出")
    
    # Value score
    st.metric("局面评分 (Value)", f"{value:.3f}", 
             help="范围 [-1, 1]，正值表示当前玩家优势")
    
    # Top 走法
    st.markdown("### 📊 Top 走法概率")
    
    top_moves = get_top_moves(policy, game, top_k=15)
    
    for i, (action_idx, prob, is_legal) in enumerate(top_moves, 1):
        move_str = format_move(action_idx)
        
        # 标记当前选择的动作
        is_chosen = (last_action is not None and action_idx == last_action)
        
        # 颜色标记
        if is_chosen:
            icon = "✅"
            color = "#4caf50"
        elif not is_legal:
            icon = "❌"
            color = "#f44336"
        else:
            icon = "⭕"
            color = "#2196f3"
        
        # 进度条
        col1, col2, col3 = st.columns([0.5, 2, 1])
        with col1:
            st.markdown(f"**{i}**")
        with col2:
            st.markdown(f"{icon} {move_str}")
        with col3:
            st.progress(prob, text=f"{prob*100:.2f}%")
        
        if is_chosen:
            st.success("← AI 选择了这步")
        elif not is_legal:
            st.caption("非法走法")


def make_ai_move():
    """AI 走棋"""
    if st.session_state.game_over:
        return
    
    game = st.session_state.game
    mcts = st.session_state.mcts
    
    if game.current_player != st.session_state.ai_side:
        return
    
    # MCTS 搜索
    with st.spinner('AI 思考中...'):
        action_probs = mcts.search(game, temperature=0.1, add_noise=False)
        action = np.argmax(action_probs)
    
    # 保存模型输出
    st.session_state.last_policy = action_probs
    state = game.get_state_for_nn()
    _, value = st.session_state.model.predict(state, 'cpu')
    st.session_state.last_value = float(value)
    st.session_state.last_action = int(action)
    
    # 执行走法
    fr, fc, tr, tc = decode_action(action)
    game.make_move(fr, fc, tr, tc)
    
    # 检查游戏是否结束
    done, winner = game.is_game_over()
    if done:
        st.session_state.game_over = True
        st.session_state.winner = winner


def main():
    st.set_page_config(
        page_title="中国象棋 AlphaZero AI",
        page_icon="♟️",
        layout="wide"
    )
    
    st.title("♟️ 中国象棋 AlphaZero AI 演示")
    
    init_session_state()
    
    # 侧边栏：配置
    with st.sidebar:
        st.header("⚙️ 配置")
        
        # 模型选择
        models = discover_models(MODELS_DIR)
        if not models:
            st.error(f"未找到模型文件！请将 .pt 文件放入 {MODELS_DIR}")
            return
        
        model_names = [m.name for m in models]
        selected_model_name = st.selectbox("选择模型", model_names)
        selected_model_path = MODELS_DIR / selected_model_name
        
        # 加载模型
        if st.session_state.model is None or \
           st.session_state.get('current_model_path') != selected_model_path:
            with st.spinner('加载模型...'):
                model, config = load_model(selected_model_path)
                st.session_state.model = model
                st.session_state.mcts = MCTS(
                    model, 
                    num_simulations=100,
                    c_puct=1.5,
                    device='cpu'
                )
                st.session_state.current_model_path = selected_model_path
                st.success("模型加载成功！")
                
                # 显示模型信息
                st.info(f"通道数: {config.get('num_channels', '未知')}\n\n"
                       f"残差块: {config.get('num_res_blocks', '未知')}")
        
        st.divider()
        
        # AI 执棋选择
        ai_side_option = st.radio(
            "AI 执棋",
            ["红方（先手）", "黑方（后手）"],
            index=1
        )
        st.session_state.ai_side = 1 if ai_side_option == "红方（先手）" else -1
        
        st.divider()
        
        # 控制按钮
        if st.button("🔄 重新开始", use_container_width=True):
            st.session_state.game = XiangqiGame()
            st.session_state.selected_piece = None
            st.session_state.last_policy = None
            st.session_state.last_value = None
            st.session_state.last_action = None
            st.session_state.game_over = False
            st.session_state.winner = None
            st.rerun()
        
        if st.button("🤖 AI 走棋", use_container_width=True, 
                    disabled=st.session_state.game_over):
            make_ai_move()
            st.rerun()
    
    # 主界面：左侧棋盘，右侧模型输出
    col_board, col_output = st.columns([2, 1])
    
    with col_board:
        st.subheader("🎯 棋盘")
        
        # 显示当前玩家
        game = st.session_state.game
        current_player_str = "红方" if game.current_player == 1 else "黑方"
        
        if st.session_state.game_over:
            if st.session_state.winner == 0:
                st.warning("游戏结束：和棋")
            else:
                winner_str = "红方" if st.session_state.winner == 1 else "黑方"
                st.success(f"游戏结束：{winner_str} 获胜！")
        else:
            st.info(f"当前回合：{current_player_str}")
        
        # 渲染棋盘
        render_board(game)
        
        # 棋盘交互（使用列选择）
        st.markdown("---")
        st.markdown("### 🖱️ 走棋")
        
        if not st.session_state.game_over:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                from_row = st.number_input("起点行", 0, 9, 0, key='from_row')
            with col2:
                from_col = st.number_input("起点列", 0, 8, 0, key='from_col')
            with col3:
                to_row = st.number_input("终点行", 0, 9, 0, key='to_row')
            with col4:
                to_col = st.number_input("终点列", 0, 8, 0, key='to_col')
            
            if st.button("执行走法", use_container_width=True):
                try:
                    game.make_move(from_row, from_col, to_row, to_col)
                    
                    # 检查游戏是否结束
                    done, winner = game.is_game_over()
                    if done:
                        st.session_state.game_over = True
                        st.session_state.winner = winner
                    else:
                        # 如果轮到 AI，自动走棋
                        if game.current_player == st.session_state.ai_side:
                            make_ai_move()
                    
                    st.rerun()
                except ValueError as e:
                    st.error(f"非法走法：{e}")
    
    with col_output:
        if st.session_state.last_policy is not None:
            render_model_output(
                st.session_state.last_policy,
                st.session_state.last_value,
                st.session_state.game,
                st.session_state.last_action
            )
        else:
            st.info("等待 AI 走棋以查看模型输出...")
    
    # 如果游戏开始时 AI 执红，自动走第一步
    if not st.session_state.game_over and \
       st.session_state.game.current_player == st.session_state.ai_side and \
       len(st.session_state.game.history) == 0:
        make_ai_move()
        st.rerun()


if __name__ == '__main__':
    main()
