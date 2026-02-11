"""
中国象棋AI演示 - Flask后端
===========================
提供模型加载、MCTS推理、走法验证等API。
"""

import os
import sys
import json
import glob
import traceback
import numpy as np

from flask import Flask, jsonify, request, send_from_directory

# 添加training目录到路径
TRAINING_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'training')
sys.path.insert(0, TRAINING_DIR)

from game import XiangqiGame, PIECE_NAMES, decode_action, encode_action, ACTION_SPACE
from mcts import MCTS

# 延迟导入torch（可能不可用）
torch = None
XiangqiNet = None

def _import_torch():
    global torch, XiangqiNet
    if torch is None:
        import torch as _torch
        torch = _torch
        from model import XiangqiNet as _Net
        XiangqiNet = _Net

app = Flask(__name__, static_folder='static', static_url_path='')

# ============================================================
# 全局状态
# ============================================================
_state = {
    'model': None,
    'model_name': None,
    'device': 'cpu',
    'game': None,
    'mcts': None,
    'num_simulations': 500,
    'human_side': 1,  # 1=红方, -1=黑方
}

# 模型目录：优先使用 ../models，其次 ../training/checkpoints
MODEL_DIRS = [
    os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models'),
    os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'training', 'checkpoints'),
]


def _find_models():
    """扫描所有模型目录，返回可用模型列表"""
    models = []
    seen = set()
    for d in MODEL_DIRS:
        d = os.path.abspath(d)
        if os.path.isdir(d):
            for f in sorted(glob.glob(os.path.join(d, '*.pt'))):
                name = os.path.basename(f)
                if name not in seen:
                    seen.add(name)
                    models.append({
                        'name': name,
                        'path': f,
                        'size_mb': round(os.path.getsize(f) / 1024 / 1024, 1),
                        'dir': d,
                    })
    return models


def _load_model(model_path, device='cpu'):
    """加载PyTorch模型"""
    _import_torch()
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    # 从checkpoint中获取模型配置
    if 'model_config' in checkpoint:
        cfg = checkpoint['model_config']
        model = XiangqiNet(
            num_channels=cfg.get('num_channels', 128),
            num_res_blocks=cfg.get('num_res_blocks', 6)
        )
    else:
        model = XiangqiNet(num_channels=128, num_res_blocks=6)

    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()
    return model


def _board_to_list(board):
    """将numpy棋盘转为JSON可序列化的列表"""
    return board.tolist()


def _get_piece_info(piece_code):
    """获取棋子信息"""
    if piece_code == 0:
        return None
    return {
        'code': int(piece_code),
        'name': PIECE_NAMES.get(piece_code, '?'),
        'side': 'red' if piece_code > 0 else 'black',
    }


def _format_move(action, board):
    """格式化走法为人类可读的字符串"""
    fr, fc, tr, tc = decode_action(action)
    piece = board[fr][fc]
    captured = board[tr][tc]
    name = PIECE_NAMES.get(int(piece), '?')
    cap_name = PIECE_NAMES.get(int(captured), '')
    move_str = f"{name}({fr},{fc})→({tr},{tc})"
    if captured != 0:
        move_str += f" 吃{cap_name}"
    return move_str


# ============================================================
# API 路由
# ============================================================

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')


@app.route('/api/models', methods=['GET'])
def list_models():
    """列出可用模型"""
    models = _find_models()
    return jsonify({
        'models': models,
        'current': _state['model_name'],
        'device': _state['device'],
    })


@app.route('/api/load_model', methods=['POST'])
def load_model():
    """加载指定模型"""
    data = request.json
    model_name = data.get('model_name')
    device = data.get('device', 'cpu')

    models = _find_models()
    model_info = next((m for m in models if m['name'] == model_name), None)
    if not model_info:
        return jsonify({'error': f'模型 {model_name} 未找到'}), 404

    try:
        _import_torch()
        # 检查CUDA是否可用
        if device.startswith('cuda') and not torch.cuda.is_available():
            device = 'cpu'

        model = _load_model(model_info['path'], device)
        _state['model'] = model
        _state['model_name'] = model_name
        _state['device'] = device

        # 重建MCTS
        _state['mcts'] = MCTS(
            model=model,
            num_simulations=_state['num_simulations'],
            c_puct=1.5,
            device=device,
        )

        return jsonify({
            'success': True,
            'model_name': model_name,
            'device': device,
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/new_game', methods=['POST'])
def new_game():
    """开始新游戏"""
    data = request.json or {}
    human_side = data.get('human_side', 'red')
    num_simulations = data.get('num_simulations', 500)

    _state['human_side'] = 1 if human_side == 'red' else -1
    _state['num_simulations'] = max(10, min(10000, int(num_simulations)))
    _state['game'] = XiangqiGame()

    # 更新MCTS模拟次数
    if _state['mcts']:
        _state['mcts'].num_simulations = _state['num_simulations']

    result = {
        'board': _board_to_list(_state['game'].board),
        'current_player': int(_state['game'].current_player),
        'human_side': int(_state['human_side']),
        'game_over': False,
        'winner': None,
        'ai_analysis': None,
    }

    # 如果AI先手（人类执黑），AI自动走第一步
    if _state['human_side'] == -1 and _state['mcts']:
        ai_result = _do_ai_move()
        if ai_result:
            result.update(ai_result)

    return jsonify(result)


@app.route('/api/human_move', methods=['POST'])
def human_move():
    """人类走棋"""
    data = request.json
    fr, fc = data['from_row'], data['from_col']
    tr, tc = data['to_row'], data['to_col']

    game = _state['game']
    if game is None:
        return jsonify({'error': '游戏未开始'}), 400

    # 检查是否轮到人类
    if game.current_player != _state['human_side']:
        return jsonify({'error': '不是你的回合'}), 400

    # 检查走法是否合法
    legal_moves = game.get_legal_moves()
    if (fr, fc, tr, tc) not in legal_moves:
        return jsonify({'error': '非法走法'}), 400

    # 执行走法
    game.make_move(fr, fc, tr, tc)

    # 检查游戏是否结束
    done, winner = game.is_game_over()

    result = {
        'board': _board_to_list(game.board),
        'current_player': int(game.current_player),
        'game_over': done,
        'winner': int(winner) if winner else None,
        'human_move': {'from': [fr, fc], 'to': [tr, tc]},
        'ai_analysis': None,
        'ai_move': None,
    }

    # 如果游戏未结束，AI走棋
    if not done and _state['mcts']:
        ai_result = _do_ai_move()
        if ai_result:
            result.update(ai_result)

    return jsonify(result)


@app.route('/api/get_legal_moves', methods=['POST'])
def get_legal_moves():
    """获取指定棋子的合法走法"""
    data = request.json
    row, col = data['row'], data['col']

    game = _state['game']
    if game is None:
        return jsonify({'error': '游戏未开始'}), 400

    piece = game.board[row][col]
    if piece == 0:
        return jsonify({'moves': []})

    # 只允许操作自己的棋子
    if (piece > 0 and _state['human_side'] != 1) or \
       (piece < 0 and _state['human_side'] != -1):
        return jsonify({'moves': []})

    # 只在自己的回合才能选择棋子
    if game.current_player != _state['human_side']:
        return jsonify({'moves': []})

    legal_moves = game.get_legal_moves()
    valid_targets = []
    for fr, fc, tr, tc in legal_moves:
        if fr == row and fc == col:
            valid_targets.append([tr, tc])

    return jsonify({'moves': valid_targets})


@app.route('/api/game_state', methods=['GET'])
def game_state():
    """获取当前游戏状态"""
    game = _state['game']
    if game is None:
        return jsonify({'error': '游戏未开始'}), 400

    done, winner = game.is_game_over()
    return jsonify({
        'board': _board_to_list(game.board),
        'current_player': int(game.current_player),
        'human_side': int(_state['human_side']),
        'game_over': done,
        'winner': int(winner) if winner else None,
        'move_count': game.move_count,
        'model_name': _state['model_name'],
        'num_simulations': _state['num_simulations'],
    })


def _do_ai_move():
    """执行AI走棋，返回走法信息和分析数据"""
    game = _state['game']
    mcts = _state['mcts']

    if game is None or mcts is None:
        return None

    # MCTS搜索
    action_probs = mcts.search(game, temperature=0.0, add_noise=False)

    # 获取模型直接输出（用于分析面板）
    state = game.get_state_for_nn()
    raw_policy, value_score = _state['model'].predict(state, _state['device'])

    # 获取合法动作
    legal_actions = game.get_legal_actions()
    legal_set = set(legal_actions)

    # Top走法分析（从MCTS搜索结果）
    top_indices = np.argsort(action_probs)[::-1][:15]
    top_moves = []
    selected_action = int(np.argmax(action_probs))

    for idx in top_indices:
        prob = float(action_probs[idx])
        if prob < 0.001:
            continue
        fr, fc, tr, tc = decode_action(int(idx))
        is_legal = int(idx) in legal_set
        top_moves.append({
            'action': int(idx),
            'from': [int(fr), int(fc)],
            'to': [int(tr), int(tc)],
            'prob': round(prob, 4),
            'raw_prob': round(float(raw_policy[int(idx)]), 6),
            'legal': is_legal,
            'selected': int(idx) == selected_action,
            'label': _format_move(int(idx), game.board),
        })

    # 格式化AI走法label（在走棋前，此时board还未变）
    ai_fr, ai_fc, ai_tr, ai_tc = decode_action(selected_action)
    ai_label = _format_move(selected_action, game.board)

    # 执行AI走法
    game.make_action(selected_action)
    done, winner = game.is_game_over()

    return {
        'board': _board_to_list(game.board),
        'current_player': int(game.current_player),
        'game_over': done,
        'winner': int(winner) if winner else None,
        'ai_move': {
            'from': [int(ai_fr), int(ai_fc)],
            'to': [int(ai_tr), int(ai_tc)],
            'action': selected_action,
            'label': ai_label,
        },
        'ai_analysis': {
            'value_score': round(float(value_score), 4),
            'top_moves': top_moves,
            'num_simulations': _state['num_simulations'],
        },
    }


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='中国象棋AI演示')
    parser.add_argument('--host', default='0.0.0.0', help='监听地址')
    parser.add_argument('--port', type=int, default=5000, help='端口')
    parser.add_argument('--device', default='cpu', help='计算设备 (cpu/cuda:0)')
    parser.add_argument('--debug', action='store_true', help='调试模式')
    args = parser.parse_args()

    print(f"\n{'='*50}")
    print(f"  中国象棋AI演示")
    print(f"  地址: http://{args.host}:{args.port}")
    print(f"  设备: {args.device}")
    print(f"{'='*50}\n")

    # 列出可用模型
    models = _find_models()
    if models:
        print(f"发现 {len(models)} 个模型:")
        for m in models:
            print(f"  - {m['name']} ({m['size_mb']} MB) [{m['dir']}]")
    else:
        print("未发现模型文件，请将 .pt 文件放入 models/ 目录")

    app.run(host=args.host, port=args.port, debug=args.debug)
