"""中国象棋游戏引擎 (v4 Cython 加速版)
================================
实现棋盘表示、走法生成、规则判定等核心逻辑。

v4 优化内容：
1. Cython C 扩展加速走法生成（~100x）、将军检测（~27x）
2. 自动检测 Cython 引擎，不可用时回退到 Python 实现
3. _is_in_check: 从将的位置反向检测攻击
4. is_game_over: 缓存走法结果，避免重复计算
5. get_state_for_nn / get_material_score: NumPy 向量化环

棋盘表示：
- 10行 x 9列的二维数组
- 红方(正数): 帅1 仕2 相3 马4 车5 炮6 兵7
- 黑方(负数): 将-1 士-2 象-3 马-4 车-5 炮-6 卒-7
- 空位: 0

坐标系：
- (row, col)，row从0(红方底线)到9(黑方底线)，col从0到8
"""

import numpy as np
from typing import List, Tuple, Optional
import os
import sys
import logging

logger = logging.getLogger(__name__)

# 尝试导入 Cython 加速引擎
_USE_CYTHON = False
try:
    _cython_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cython_engine')
    if _cython_dir not in sys.path:
        sys.path.insert(0, _cython_dir)
    from game_core import (
        cy_generate_legal_moves,
        cy_is_in_check,
        cy_find_king,
        cy_is_attacked,
        cy_has_legal_moves,
    )
    _USE_CYTHON = True
    logger.info("Cython 加速引擎已加载 (~100x 加速)")
except ImportError:
    logger.warning("Cython 引擎未编译，使用 Python 回退版本。"
                   "编译方法: cd training/cython_engine && python setup.py build_ext --inplace")

# 棋子编码
EMPTY = 0
R_KING = 1    # 帅
R_ADVISOR = 2 # 仕
R_BISHOP = 3  # 相
R_KNIGHT = 4  # 马
R_ROOK = 5    # 车
R_CANNON = 6  # 炮
R_PAWN = 7    # 兵

B_KING = -1    # 将
B_ADVISOR = -2 # 士
B_BISHOP = -3  # 象
B_KNIGHT = -4  # 马
B_ROOK = -5    # 车
B_CANNON = -6  # 炮
B_PAWN = -7    # 卒

# 棋子名称映射
PIECE_NAMES = {
    0: '．', 1: '帅', 2: '仕', 3: '相', 4: '马', 5: '车', 6: '炮', 7: '兵',
    -1: '将', -2: '士', -3: '象', -4: '马', -5: '车', -6: '炮', -7: '卒'
}

# 棋子价值（用于材料评估）
PIECE_VALUES = np.array([0, 0, 20, 20, 40, 90, 45, 10], dtype=np.int32)  # index=abs(piece)

ROWS = 10
COLS = 9

# 动作空间编码
ACTION_SPACE = 90 * 90  # 8100

# ============================================================
# 预计算查找表（模块加载时一次性计算）
# ============================================================

# 宫格位置集合（用于帅/将/仕/士走法检查）
_RED_PALACE = frozenset([(0,3),(0,4),(0,5),(1,3),(1,4),(1,5),(2,3),(2,4),(2,5)])
_BLACK_PALACE = frozenset([(7,3),(7,4),(7,5),(8,3),(8,4),(8,5),(9,3),(9,4),(9,5)])

# 仕/士可站位置
_RED_ADVISOR_POS = frozenset([(0,3),(0,5),(1,4),(2,3),(2,5)])
_BLACK_ADVISOR_POS = frozenset([(7,3),(7,5),(8,4),(9,3),(9,5)])

# 马的走法偏移: (dr, dc, 蹩脚dr, 蹩脚dc)
_KNIGHT_MOVES = (
    (-2, -1, -1, 0), (-2, 1, -1, 0),
    (2, -1, 1, 0), (2, 1, 1, 0),
    (-1, -2, 0, -1), (-1, 2, 0, 1),
    (1, -2, 0, -1), (1, 2, 0, 1),
)

# 反向马攻击偏移: 从被攻击位置看，哪些位置的马可以攻击到这里
# (马的偏移dr, 马的偏移dc, 蹩脚检查偏移br, bc)
_KNIGHT_ATTACK_OFFSETS = (
    (-2, -1, -1, 0), (-2, 1, -1, 0),
    (2, -1, 1, 0), (2, 1, 1, 0),
    (-1, -2, 0, -1), (-1, 2, 0, 1),
    (1, -2, 0, -1), (1, 2, 0, 1),
)


def encode_action(from_row: int, from_col: int, to_row: int, to_col: int) -> int:
    """将走法编码为动作索引"""
    return (from_row * COLS + from_col) * 90 + (to_row * COLS + to_col)


def decode_action(action: int) -> Tuple[int, int, int, int]:
    """将动作索引解码为走法"""
    from_pos = action // 90
    to_pos = action % 90
    return from_pos // COLS, from_pos % COLS, to_pos // COLS, to_pos % COLS


class XiangqiGame:
    """中国象棋游戏类 (v3 性能优化版)"""

    __slots__ = ['board', 'current_player', 'move_count', 'history',
                 'no_capture_count', '_legal_moves_cache']

    def __init__(self):
        self.board = np.zeros((ROWS, COLS), dtype=np.int8)
        self.current_player = 1  # 1=红方, -1=黑方
        self.move_count = 0
        self.history = []
        self.no_capture_count = 0
        self._legal_moves_cache = None  # 走法缓存
        self._init_board()

    def _init_board(self):
        """初始化棋盘"""
        self.board[0] = [R_ROOK, R_KNIGHT, R_BISHOP, R_ADVISOR, R_KING,
                         R_ADVISOR, R_BISHOP, R_KNIGHT, R_ROOK]
        self.board[2][1] = R_CANNON
        self.board[2][7] = R_CANNON
        self.board[3][0] = R_PAWN
        self.board[3][2] = R_PAWN
        self.board[3][4] = R_PAWN
        self.board[3][6] = R_PAWN
        self.board[3][8] = R_PAWN

        self.board[9] = [B_ROOK, B_KNIGHT, B_BISHOP, B_ADVISOR, B_KING,
                         B_ADVISOR, B_BISHOP, B_KNIGHT, B_ROOK]
        self.board[7][1] = B_CANNON
        self.board[7][7] = B_CANNON
        self.board[6][0] = B_PAWN
        self.board[6][2] = B_PAWN
        self.board[6][4] = B_PAWN
        self.board[6][6] = B_PAWN
        self.board[6][8] = B_PAWN

    def clone(self) -> 'XiangqiGame':
        """深拷贝当前游戏状态"""
        game = XiangqiGame.__new__(XiangqiGame)
        game.board = self.board.copy()
        game.current_player = self.current_player
        game.move_count = self.move_count
        game.history = self.history.copy()
        game.no_capture_count = self.no_capture_count
        game._legal_moves_cache = None
        return game

    # ============================================================
    # 快速将军检测（从将的位置反向检查）
    # ============================================================

    @staticmethod
    def _is_attacked(board: np.ndarray, kr: int, kc: int, by_player: int) -> bool:
        """
        快速检测位置 (kr, kc) 是否被 by_player 方攻击。
        从目标位置反向检查各方向，比遍历所有敌方棋子快得多。

        参数:
            board: 棋盘数组
            kr, kc: 被检测的位置
            by_player: 攻击方 (1=红方, -1=黑方)
        """
        # 敌方棋子的符号
        enemy_rook = R_ROOK * by_player
        enemy_cannon = R_CANNON * by_player
        enemy_knight = R_KNIGHT * by_player
        enemy_pawn = R_PAWN * by_player
        enemy_king = R_KING * by_player

        # --- 1. 检查车和将的直线攻击（四个方向）---
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            r, c = kr + dr, kc + dc
            while 0 <= r < ROWS and 0 <= c < COLS:
                p = board[r, c]
                if p != EMPTY:
                    if p == enemy_rook or p == enemy_king:
                        return True
                    break  # 被阻挡
                r += dr
                c += dc

        # --- 2. 检查炮的攻击（四个方向，需要一个炮架）---
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            r, c = kr + dr, kc + dc
            found_platform = False
            while 0 <= r < ROWS and 0 <= c < COLS:
                p = board[r, c]
                if not found_platform:
                    if p != EMPTY:
                        found_platform = True
                else:
                    if p != EMPTY:
                        if p == enemy_cannon:
                            return True
                        break
                r += dr
                c += dc

        # --- 3. 检查马的攻击（8个方向，含蹩脚检查）---
        for dr, dc, br, bc in _KNIGHT_ATTACK_OFFSETS:
            nr, nc = kr + dr, kc + dc
            if 0 <= nr < ROWS and 0 <= nc < COLS:
                if board[nr, nc] == enemy_knight:
                    # 蹩脚检查：从马的位置看蹩脚点
                    block_r, block_c = nr + br, nc + bc
                    # 注意：蹩脚点是从马的位置算的，不是从被攻击位置算的
                    # 马在 (nr,nc)，要到 (kr,kc)
                    # 蹩脚点 = 马位置 + (目标方向的一半)
                    # 重新计算：马从(nr,nc)到(kr,kc)的蹩脚点
                    move_dr = kr - nr
                    move_dc = kc - nc
                    if abs(move_dr) == 2:
                        blk_r, blk_c = nr + move_dr // 2, nc
                    else:
                        blk_r, blk_c = nr, nc + move_dc // 2
                    if board[blk_r, blk_c] == EMPTY:
                        return True

        # --- 4. 检查兵/卒的攻击 ---
        if by_player == 1:
            # 红兵攻击：红兵在 (kr-1, kc) 向上攻击到 (kr, kc)
            if kr - 1 >= 0 and board[kr - 1, kc] == enemy_pawn:
                return True
            # 过河红兵左右攻击
            if kr >= 5:  # 在黑方区域
                if kc - 1 >= 0 and board[kr, kc - 1] == enemy_pawn:
                    return True
                if kc + 1 < COLS and board[kr, kc + 1] == enemy_pawn:
                    return True
        else:
            # 黑卒攻击：黑卒在 (kr+1, kc) 向下攻击到 (kr, kc)
            if kr + 1 < ROWS and board[kr + 1, kc] == enemy_pawn:
                return True
            # 过河黑卒左右攻击
            if kr <= 4:  # 在红方区域
                if kc - 1 >= 0 and board[kr, kc - 1] == enemy_pawn:
                    return True
                if kc + 1 < COLS and board[kr, kc + 1] == enemy_pawn:
                    return True

        return False

    @staticmethod
    def _kings_facing_fast(board: np.ndarray) -> bool:
        """
        快速检查将帅是否面对面（飞将）。
        使用 NumPy 操作替代 Python 循环。
        """
        # 找帅和将
        r_pos = np.argwhere(board == R_KING)
        b_pos = np.argwhere(board == B_KING)
        if len(r_pos) == 0 or len(b_pos) == 0:
            return False

        r_row, r_col = int(r_pos[0, 0]), int(r_pos[0, 1])
        b_row, b_col = int(b_pos[0, 0]), int(b_pos[0, 1])

        if r_col != b_col:
            return False

        # 检查中间是否有棋子（NumPy 切片）
        col = r_col
        min_row = min(r_row, b_row) + 1
        max_row = max(r_row, b_row)
        if min_row >= max_row:
            return True
        return np.all(board[min_row:max_row, col] == EMPTY)

    # ============================================================
    # 走法生成（保持原有逻辑，内部辅助函数不变）
    # ============================================================

    def _generate_piece_moves(self, r: int, c: int, player: int) -> List[Tuple[int, int]]:
        """生成指定位置棋子的所有伪合法走法（不检查将军）"""
        piece = abs(self.board[r, c])
        board = self.board

        if piece == 1:  # 帅/将
            moves = []
            palace = _RED_PALACE if player == 1 else _BLACK_PALACE
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = r + dr, c + dc
                if (nr, nc) in palace:
                    p = board[nr, nc]
                    if p == EMPTY or (player == 1 and p < 0) or (player == -1 and p > 0):
                        moves.append((nr, nc))
            return moves

        elif piece == 2:  # 仕/士
            moves = []
            palace = _RED_ADVISOR_POS if player == 1 else _BLACK_ADVISOR_POS
            for dr, dc in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
                nr, nc = r + dr, c + dc
                if (nr, nc) in palace:
                    p = board[nr, nc]
                    if p == EMPTY or (player == 1 and p < 0) or (player == -1 and p > 0):
                        moves.append((nr, nc))
            return moves

        elif piece == 3:  # 相/象
            moves = []
            for dr, dc in ((-2, -2), (-2, 2), (2, -2), (2, 2)):
                nr, nc = r + dr, c + dc
                if not (0 <= nr < ROWS and 0 <= nc < COLS):
                    continue
                if player == 1 and nr > 4:
                    continue
                if player == -1 and nr < 5:
                    continue
                if board[r + dr // 2, c + dc // 2] != EMPTY:
                    continue
                p = board[nr, nc]
                if p == EMPTY or (player == 1 and p < 0) or (player == -1 and p > 0):
                    moves.append((nr, nc))
            return moves

        elif piece == 4:  # 马
            moves = []
            for dr, dc, br, bc in _KNIGHT_MOVES:
                nr, nc = r + dr, c + dc
                if not (0 <= nr < ROWS and 0 <= nc < COLS):
                    continue
                if board[r + br, c + bc] != EMPTY:
                    continue
                p = board[nr, nc]
                if p == EMPTY or (player == 1 and p < 0) or (player == -1 and p > 0):
                    moves.append((nr, nc))
            return moves

        elif piece == 5:  # 车
            moves = []
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = r + dr, c + dc
                while 0 <= nr < ROWS and 0 <= nc < COLS:
                    p = board[nr, nc]
                    if p == EMPTY:
                        moves.append((nr, nc))
                    elif (player == 1 and p < 0) or (player == -1 and p > 0):
                        moves.append((nr, nc))
                        break
                    else:
                        break
                    nr += dr
                    nc += dc
            return moves

        elif piece == 6:  # 炮
            moves = []
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = r + dr, c + dc
                while 0 <= nr < ROWS and 0 <= nc < COLS and board[nr, nc] == EMPTY:
                    moves.append((nr, nc))
                    nr += dr
                    nc += dc
                if 0 <= nr < ROWS and 0 <= nc < COLS:
                    nr += dr
                    nc += dc
                    while 0 <= nr < ROWS and 0 <= nc < COLS:
                        p = board[nr, nc]
                        if p != EMPTY:
                            if (player == 1 and p < 0) or (player == -1 and p > 0):
                                moves.append((nr, nc))
                            break
                        nr += dr
                        nc += dc
            return moves

        elif piece == 7:  # 兵/卒
            moves = []
            if player == 1:
                if r + 1 < ROWS:
                    p = board[r + 1, c]
                    if p == EMPTY or p < 0:
                        moves.append((r + 1, c))
                if r >= 5:
                    if c - 1 >= 0:
                        p = board[r, c - 1]
                        if p == EMPTY or p < 0:
                            moves.append((r, c - 1))
                    if c + 1 < COLS:
                        p = board[r, c + 1]
                        if p == EMPTY or p < 0:
                            moves.append((r, c + 1))
            else:
                if r - 1 >= 0:
                    p = board[r - 1, c]
                    if p == EMPTY or p > 0:
                        moves.append((r - 1, c))
                if r <= 4:
                    if c - 1 >= 0:
                        p = board[r, c - 1]
                        if p == EMPTY or p > 0:
                            moves.append((r, c - 1))
                    if c + 1 < COLS:
                        p = board[r, c + 1]
                        if p == EMPTY or p > 0:
                            moves.append((r, c + 1))
            return moves

        return []

    def _find_king_pos(self, player: int, board: np.ndarray) -> Optional[Tuple[int, int]]:
        """快速找将/帅位置（只搜索宫格范围）"""
        target = R_KING if player == 1 else B_KING
        if player == 1:
            for r in range(3):
                for c in range(3, 6):
                    if board[r, c] == target:
                        return (r, c)
        else:
            for r in range(7, 10):
                for c in range(3, 6):
                    if board[r, c] == target:
                        return (r, c)
        return None

    def _is_move_legal(self, fr: int, fc: int, tr: int, tc: int, player: int) -> bool:
        """
        快速检查一步走法是否合法（走后不被将军、不飞将）。
        使用就地修改 + 恢复，避免 board.copy()。
        """
        board = self.board
        moving_piece = board[fr, fc]
        captured = board[tr, tc]

        # 就地走棋
        board[tr, tc] = moving_piece
        board[fr, fc] = EMPTY

        # 找己方将/帅
        king_pos = self._find_king_pos(player, board)
        if king_pos is None:
            # 将被吃了（不应该发生）
            board[fr, fc] = moving_piece
            board[tr, tc] = captured
            return False

        kr, kc = king_pos

        # 检查飞将（只有在将/帅所在列有变化时才检查）
        legal = True
        enemy_king_piece = B_KING if player == 1 else R_KING
        enemy_king_pos = self._find_king_pos(-player, board)
        if enemy_king_pos is not None:
            ekr, ekc = enemy_king_pos
            if kc == ekc:
                # 检查中间是否全空
                min_r = min(kr, ekr) + 1
                max_r = max(kr, ekr)
                all_empty = True
                for check_r in range(min_r, max_r):
                    if board[check_r, kc] != EMPTY:
                        all_empty = False
                        break
                if all_empty:
                    legal = False

        # 检查是否被将军
        if legal:
            legal = not self._is_attacked(board, kr, kc, -player)

        # 恢复棋盘
        board[fr, fc] = moving_piece
        board[tr, tc] = captured

        return legal

    def get_legal_moves(self) -> List[Tuple[int, int, int, int]]:
        """
        获取当前玩家的所有合法走法。
        使用缓存避免重复计算。
        自动使用 Cython 加速版本（如果可用）。
        """
        if self._legal_moves_cache is not None:
            return self._legal_moves_cache

        if _USE_CYTHON:
            # Cython 加速版本 (~100x)
            moves = cy_generate_legal_moves(self.board, self.current_player)
        else:
            # Python 回退版本
            moves = []
            player = self.current_player
            board = self.board
            for r in range(ROWS):
                for c in range(COLS):
                    p = board[r, c]
                    if p == EMPTY:
                        continue
                    if (player == 1 and p > 0) or (player == -1 and p < 0):
                        targets = self._generate_piece_moves(r, c, player)
                        for tr, tc in targets:
                            if self._is_move_legal(r, c, tr, tc, player):
                                moves.append((r, c, tr, tc))

        self._legal_moves_cache = moves
        return moves

    def get_legal_actions(self) -> List[int]:
        """获取合法动作的编码列表"""
        moves = self.get_legal_moves()
        return [encode_action(fr, fc, tr, tc) for fr, fc, tr, tc in moves]

    def make_move(self, from_row: int, from_col: int, to_row: int, to_col: int) -> bool:
        """执行走法"""
        captured = self.board[to_row, to_col]

        self.history.append(self.board.tobytes())

        self.board[to_row, to_col] = self.board[from_row, from_col]
        self.board[from_row, from_col] = EMPTY

        if captured != EMPTY:
            self.no_capture_count = 0
        else:
            self.no_capture_count += 1

        self.current_player = -self.current_player
        self.move_count += 1
        self._legal_moves_cache = None  # 走法后清除缓存
        return True

    def make_action(self, action: int) -> bool:
        """通过动作编码执行走法"""
        fr, fc, tr, tc = decode_action(action)
        return self.make_move(fr, fc, tr, tc)

    def get_material_score(self, player: int) -> int:
        """
        计算指定方的材料总分（NumPy 向量化）。
        """
        board = self.board
        if player == 1:
            mask = board > 0
            pieces = board[mask]
        else:
            mask = board < 0
            pieces = -board[mask]  # 取绝对值
        return int(np.sum(PIECE_VALUES[pieces]))

    def is_game_over(self) -> Tuple[bool, Optional[int]]:
        """
        检查游戏是否结束。
        使用缓存的走法结果，避免重复计算 get_legal_moves。
        返回: (是否结束, 赢家) 赢家: 1=红方, -1=黑方, 0=和棋, None=未结束
        """
        # 检查将/帅是否存在
        if _USE_CYTHON:
            r_king = cy_find_king(self.board, 1)
            b_king = cy_find_king(self.board, -1)
        else:
            r_king = self._find_king_pos(1, self.board)
            b_king = self._find_king_pos(-1, self.board)

        if r_king is None:
            return True, -1
        if b_king is None:
            return True, 1

        # 检查当前玩家是否无合法走法（被将死）
        # 这里会触发缓存，后续 MCTS 调用 get_legal_moves 直接复用
        legal_moves = self.get_legal_moves()
        if len(legal_moves) == 0:
            return True, -self.current_player

        # 和棋判定
        if self.no_capture_count >= 120:
            return True, 0

        # 超步数材料判定
        if self.move_count >= 200:
            red_score = self.get_material_score(1)
            black_score = self.get_material_score(-1)
            diff = red_score - black_score
            if diff > 30:
                return True, 1
            elif diff < -30:
                return True, -1
            else:
                return True, 0

        # 重复局面判和
        if len(self.history) >= 6:
            current = self.board.tobytes()
            repeat_count = 0
            for h in self.history[-12:]:
                if h == current:
                    repeat_count += 1
                    if repeat_count >= 3:
                        return True, 0

        return False, None

    def get_state_for_nn(self) -> np.ndarray:
        """
        将棋盘状态转换为神经网络输入特征（NumPy 向量化）。

        特征平面 (共15个平面，每个10x9):
        - 平面 0-6: 当前玩家的7种棋子
        - 平面 7-13: 对手的7种棋子
        - 平面 14: 当前玩家标识
        """
        features = np.zeros((15, ROWS, COLS), dtype=np.float32)
        board = self.board

        if self.current_player == 1:
            for i in range(1, 8):
                features[i - 1] = (board == i)
                features[7 + i - 1] = (board == -i)
            features[14] = 1.0
        else:
            for i in range(1, 8):
                features[i - 1] = (board == -i)
                features[7 + i - 1] = (board == i)

        return features

    def _find_king(self, player: int) -> Optional[Tuple[int, int]]:
        """找到指定方的将/帅位置"""
        if _USE_CYTHON:
            return cy_find_king(self.board, player)
        return self._find_king_pos(player, self.board)

    def _kings_facing(self, board: np.ndarray) -> bool:
        """检查将帅是否面对面（兼容旧接口）"""
        return self._kings_facing_fast(board)

    def _is_in_check(self, player: int, board: Optional[np.ndarray] = None) -> bool:
        """检查指定方是否被将军"""
        if board is None:
            board = self.board
        if _USE_CYTHON:
            return cy_is_in_check(board, player)
        king_pos = self._find_king_pos(player, board)
        if king_pos is None:
            return True
        return self._is_attacked(board, king_pos[0], king_pos[1], -player)

    def get_canonical_board(self) -> np.ndarray:
        """获取标准化棋盘"""
        if self.current_player == 1:
            return self.board.copy()
        else:
            flipped = np.flip(self.board, axis=0).copy()
            return -flipped

    def display(self):
        """打印棋盘"""
        print("\n  ０ １ ２ ３ ４ ５ ６ ７ ８")
        print("  ＋－＋－＋－＋－＋－＋－＋－＋－＋")
        for r in range(ROWS - 1, -1, -1):
            row_str = f"{r} "
            for c in range(COLS):
                piece = self.board[r, c]
                row_str += PIECE_NAMES[piece] + " "
            print(row_str)
            if r == 5:
                print("  ＝＝＝＝＝楚河汉界＝＝＝＝＝")
        print("  ＋－＋－＋－＋－＋－＋－＋－＋－＋")
        player_name = "红方" if self.current_player == 1 else "黑方"
        print(f"  当前: {player_name}  步数: {self.move_count}")


if __name__ == "__main__":
    game = XiangqiGame()
    game.display()
    print(f"\n合法走法数: {len(game.get_legal_moves())}")
    print(f"动作空间大小: {ACTION_SPACE}")

    features = game.get_state_for_nn()
    print(f"神经网络输入形状: {features.shape}")

    moves = game.get_legal_moves()
    print(f"\n前5个合法走法:")
    for m in moves[:5]:
        fr, fc, tr, tc = m
        piece = PIECE_NAMES[game.board[fr, fc]]
        print(f"  {piece} ({fr},{fc}) -> ({tr},{tc})")

    print(f"\n红方材料分: {game.get_material_score(1)}")
    print(f"黑方材料分: {game.get_material_score(-1)}")
