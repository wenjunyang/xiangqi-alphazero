"""
中国象棋游戏引擎
================
实现棋盘表示、走法生成、规则判定等核心逻辑。

棋盘表示：
- 10行 x 9列的二维数组
- 红方(正数): 帅1 仕2 相3 马4 车5 炮6 兵7
- 黑方(负数): 将-1 士-2 象-3 马-4 车-5 炮-6 卒-7
- 空位: 0

坐标系：
- (row, col)，row从0(红方底线)到9(黑方底线)，col从0到8
"""

import numpy as np
import copy
from typing import List, Tuple, Optional

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

ROWS = 10
COLS = 9

# 动作空间编码：
# 每个动作 = (from_row, from_col, to_row, to_col)
# 编码为 from_pos * 90 + to_pos，其中 pos = row * 9 + col
# 总动作空间 = 90 * 90 = 8100（包含非法动作，实际合法动作远少于此）
ACTION_SPACE = 90 * 90  # 8100


def encode_action(from_row: int, from_col: int, to_row: int, to_col: int) -> int:
    """将走法编码为动作索引"""
    from_pos = from_row * COLS + from_col
    to_pos = to_row * COLS + to_col
    return from_pos * 90 + to_pos


def decode_action(action: int) -> Tuple[int, int, int, int]:
    """将动作索引解码为走法"""
    from_pos = action // 90
    to_pos = action % 90
    from_row, from_col = divmod(from_pos, COLS)
    to_row, to_col = divmod(to_pos, COLS)
    return from_row, from_col, to_row, to_col


class XiangqiGame:
    """中国象棋游戏类"""

    def __init__(self):
        self.board = np.zeros((ROWS, COLS), dtype=np.int8)
        self.current_player = 1  # 1=红方, -1=黑方
        self.move_count = 0
        self.history = []  # 历史局面用于判断重复
        self.no_capture_count = 0  # 无吃子步数（用于和棋判定）
        self._init_board()

    def _init_board(self):
        """初始化棋盘"""
        # 红方（底部，row 0-4）
        self.board[0] = [R_ROOK, R_KNIGHT, R_BISHOP, R_ADVISOR, R_KING,
                         R_ADVISOR, R_BISHOP, R_KNIGHT, R_ROOK]
        self.board[2][1] = R_CANNON
        self.board[2][7] = R_CANNON
        self.board[3][0] = R_PAWN
        self.board[3][2] = R_PAWN
        self.board[3][4] = R_PAWN
        self.board[3][6] = R_PAWN
        self.board[3][8] = R_PAWN

        # 黑方（顶部，row 5-9）
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
        return game

    def _is_in_board(self, r: int, c: int) -> bool:
        return 0 <= r < ROWS and 0 <= c < COLS

    def _is_own_piece(self, r: int, c: int, player: int) -> bool:
        if not self._is_in_board(r, c):
            return False
        piece = self.board[r][c]
        return (player == 1 and piece > 0) or (player == -1 and piece < 0)

    def _is_enemy_piece(self, r: int, c: int, player: int) -> bool:
        if not self._is_in_board(r, c):
            return False
        piece = self.board[r][c]
        return (player == 1 and piece < 0) or (player == -1 and piece > 0)

    def _is_empty_or_enemy(self, r: int, c: int, player: int) -> bool:
        if not self._is_in_board(r, c):
            return False
        return not self._is_own_piece(r, c, player)

    def _find_king(self, player: int) -> Optional[Tuple[int, int]]:
        """找到指定方的将/帅位置"""
        target = R_KING if player == 1 else B_KING
        positions = np.argwhere(self.board == target)
        if len(positions) == 0:
            return None
        return int(positions[0][0]), int(positions[0][1])

    def _kings_facing(self, board: np.ndarray) -> bool:
        """检查将帅是否面对面（飞将）"""
        r_king_pos = None
        b_king_pos = None
        for r in range(ROWS):
            for c in range(COLS):
                if board[r][c] == R_KING:
                    r_king_pos = (r, c)
                elif board[r][c] == B_KING:
                    b_king_pos = (r, c)

        if r_king_pos is None or b_king_pos is None:
            return False

        if r_king_pos[1] != b_king_pos[1]:
            return False

        col = r_king_pos[1]
        min_row = min(r_king_pos[0], b_king_pos[0])
        max_row = max(r_king_pos[0], b_king_pos[0])

        for r in range(min_row + 1, max_row):
            if board[r][col] != EMPTY:
                return False
        return True

    def _generate_king_moves(self, r: int, c: int, player: int) -> List[Tuple[int, int]]:
        """生成帅/将的走法"""
        moves = []
        if player == 1:
            palace = [(0, 3), (0, 4), (0, 5), (1, 3), (1, 4), (1, 5), (2, 3), (2, 4), (2, 5)]
        else:
            palace = [(7, 3), (7, 4), (7, 5), (8, 3), (8, 4), (8, 5), (9, 3), (9, 4), (9, 5)]

        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if (nr, nc) in palace and self._is_empty_or_enemy(nr, nc, player):
                moves.append((nr, nc))
        return moves

    def _generate_advisor_moves(self, r: int, c: int, player: int) -> List[Tuple[int, int]]:
        """生成仕/士的走法"""
        moves = []
        if player == 1:
            palace = [(0, 3), (0, 5), (1, 4), (2, 3), (2, 5)]
        else:
            palace = [(7, 3), (7, 5), (8, 4), (9, 3), (9, 5)]

        for dr, dc in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
            nr, nc = r + dr, c + dc
            if (nr, nc) in palace and self._is_empty_or_enemy(nr, nc, player):
                moves.append((nr, nc))
        return moves

    def _generate_bishop_moves(self, r: int, c: int, player: int) -> List[Tuple[int, int]]:
        """生成相/象的走法（田字，不能过河，不能蹩脚）"""
        moves = []
        for dr, dc in [(-2, -2), (-2, 2), (2, -2), (2, 2)]:
            nr, nc = r + dr, c + dc
            # 蹩脚检查
            br, bc = r + dr // 2, c + dc // 2
            if not self._is_in_board(nr, nc):
                continue
            # 不能过河
            if player == 1 and nr > 4:
                continue
            if player == -1 and nr < 5:
                continue
            if self.board[br][bc] != EMPTY:
                continue
            if self._is_empty_or_enemy(nr, nc, player):
                moves.append((nr, nc))
        return moves

    def _generate_knight_moves(self, r: int, c: int, player: int) -> List[Tuple[int, int]]:
        """生成马的走法（日字，蹩马腿检查）"""
        moves = []
        # (dr, dc, 蹩脚位置)
        knight_moves = [
            (-2, -1, -1, 0), (-2, 1, -1, 0),
            (2, -1, 1, 0), (2, 1, 1, 0),
            (-1, -2, 0, -1), (-1, 2, 0, 1),
            (1, -2, 0, -1), (1, 2, 0, 1),
        ]
        for dr, dc, br, bc in knight_moves:
            nr, nc = r + dr, c + dc
            block_r, block_c = r + br, c + bc
            if not self._is_in_board(nr, nc):
                continue
            if self.board[block_r][block_c] != EMPTY:
                continue
            if self._is_empty_or_enemy(nr, nc, player):
                moves.append((nr, nc))
        return moves

    def _generate_rook_moves(self, r: int, c: int, player: int) -> List[Tuple[int, int]]:
        """生成车的走法（直线）"""
        moves = []
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            while self._is_in_board(nr, nc):
                if self.board[nr][nc] == EMPTY:
                    moves.append((nr, nc))
                elif self._is_enemy_piece(nr, nc, player):
                    moves.append((nr, nc))
                    break
                else:
                    break
                nr += dr
                nc += dc
        return moves

    def _generate_cannon_moves(self, r: int, c: int, player: int) -> List[Tuple[int, int]]:
        """生成炮的走法（直线移动，隔子吃子）"""
        moves = []
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            # 不吃子移动
            while self._is_in_board(nr, nc) and self.board[nr][nc] == EMPTY:
                moves.append((nr, nc))
                nr += dr
                nc += dc
            # 跳过一个棋子（炮架）
            if self._is_in_board(nr, nc):
                nr += dr
                nc += dc
                while self._is_in_board(nr, nc):
                    if self.board[nr][nc] != EMPTY:
                        if self._is_enemy_piece(nr, nc, player):
                            moves.append((nr, nc))
                        break
                    nr += dr
                    nc += dc
        return moves

    def _generate_pawn_moves(self, r: int, c: int, player: int) -> List[Tuple[int, int]]:
        """生成兵/卒的走法"""
        moves = []
        if player == 1:
            # 红兵向上（row增大）
            if r + 1 < ROWS and self._is_empty_or_enemy(r + 1, c, player):
                moves.append((r + 1, c))
            # 过河后可以左右移动
            if r >= 5:
                if c - 1 >= 0 and self._is_empty_or_enemy(r, c - 1, player):
                    moves.append((r, c - 1))
                if c + 1 < COLS and self._is_empty_or_enemy(r, c + 1, player):
                    moves.append((r, c + 1))
        else:
            # 黑卒向下（row减小）
            if r - 1 >= 0 and self._is_empty_or_enemy(r - 1, c, player):
                moves.append((r - 1, c))
            # 过河后可以左右移动
            if r <= 4:
                if c - 1 >= 0 and self._is_empty_or_enemy(r, c - 1, player):
                    moves.append((r, c - 1))
                if c + 1 < COLS and self._is_empty_or_enemy(r, c + 1, player):
                    moves.append((r, c + 1))
        return moves

    def _generate_piece_moves(self, r: int, c: int, player: int) -> List[Tuple[int, int]]:
        """生成指定位置棋子的所有走法"""
        piece = abs(self.board[r][c])
        if piece == 1:
            return self._generate_king_moves(r, c, player)
        elif piece == 2:
            return self._generate_advisor_moves(r, c, player)
        elif piece == 3:
            return self._generate_bishop_moves(r, c, player)
        elif piece == 4:
            return self._generate_knight_moves(r, c, player)
        elif piece == 5:
            return self._generate_rook_moves(r, c, player)
        elif piece == 6:
            return self._generate_cannon_moves(r, c, player)
        elif piece == 7:
            return self._generate_pawn_moves(r, c, player)
        return []

    def _is_in_check(self, player: int, board: Optional[np.ndarray] = None) -> bool:
        """检查指定方是否被将军"""
        if board is None:
            board = self.board
        king_pos = None
        target = R_KING if player == 1 else B_KING
        positions = np.argwhere(board == target)
        if len(positions) == 0:
            return True  # 将/帅不存在，视为被将
        king_pos = (int(positions[0][0]), int(positions[0][1]))

        enemy = -player
        for r in range(ROWS):
            for c in range(COLS):
                piece = board[r][c]
                if (enemy == 1 and piece > 0) or (enemy == -1 and piece < 0):
                    # 临时使用当前board来检查
                    old_board = self.board
                    self.board = board
                    targets = self._generate_piece_moves(r, c, enemy)
                    self.board = old_board
                    if king_pos in targets:
                        return True
        return False

    def get_legal_moves(self) -> List[Tuple[int, int, int, int]]:
        """获取当前玩家的所有合法走法"""
        moves = []
        player = self.current_player

        for r in range(ROWS):
            for c in range(COLS):
                if (player == 1 and self.board[r][c] > 0) or \
                   (player == -1 and self.board[r][c] < 0):
                    targets = self._generate_piece_moves(r, c, player)
                    for tr, tc in targets:
                        # 模拟走棋，检查是否会导致自己被将
                        new_board = self.board.copy()
                        new_board[tr][tc] = new_board[r][c]
                        new_board[r][c] = EMPTY

                        # 检查飞将
                        if self._kings_facing(new_board):
                            continue

                        # 检查走后是否被将
                        if not self._is_in_check(player, new_board):
                            moves.append((r, c, tr, tc))
        return moves

    def get_legal_actions(self) -> List[int]:
        """获取合法动作的编码列表"""
        moves = self.get_legal_moves()
        return [encode_action(fr, fc, tr, tc) for fr, fc, tr, tc in moves]

    def make_move(self, from_row: int, from_col: int, to_row: int, to_col: int) -> bool:
        """执行走法"""
        captured = self.board[to_row][to_col]

        # 记录历史
        self.history.append(self.board.tobytes())

        self.board[to_row][to_col] = self.board[from_row][from_col]
        self.board[from_row][from_col] = EMPTY

        if captured != EMPTY:
            self.no_capture_count = 0
        else:
            self.no_capture_count += 1

        self.current_player = -self.current_player
        self.move_count += 1
        return True

    def make_action(self, action: int) -> bool:
        """通过动作编码执行走法"""
        fr, fc, tr, tc = decode_action(action)
        return self.make_move(fr, fc, tr, tc)

    def is_game_over(self) -> Tuple[bool, Optional[int]]:
        """
        检查游戏是否结束
        返回: (是否结束, 赢家) 赢家: 1=红方, -1=黑方, 0=和棋, None=未结束
        """
        # 检查将/帅是否存在
        r_king = self._find_king(1)
        b_king = self._find_king(-1)

        if r_king is None:
            return True, -1
        if b_king is None:
            return True, 1

        # 检查当前玩家是否无合法走法（被将死）
        legal_moves = self.get_legal_moves()
        if len(legal_moves) == 0:
            return True, -self.current_player

        # 和棋判定：无吃子步数超过120步（60回合）
        if self.no_capture_count >= 120:
            return True, 0

        # 总步数超过200步判和
        if self.move_count >= 200:
            return True, 0

        # 重复局面判和
        if len(self.history) >= 6:
            current = self.board.tobytes()
            repeat_count = sum(1 for h in self.history[-12:] if h == current)
            if repeat_count >= 3:
                return True, 0

        return False, None

    def get_state_for_nn(self) -> np.ndarray:
        """
        将棋盘状态转换为神经网络输入特征

        特征平面 (共15个平面，每个10x9):
        - 平面 0-6: 当前玩家的7种棋子（帅/将、仕/士、相/象、马、车、炮、兵/卒）
        - 平面 7-13: 对手的7种棋子
        - 平面 14: 当前玩家标识（全1=红方，全0=黑方）
        """
        features = np.zeros((15, ROWS, COLS), dtype=np.float32)

        for r in range(ROWS):
            for c in range(COLS):
                piece = self.board[r][c]
                if piece == 0:
                    continue
                if self.current_player == 1:
                    if piece > 0:
                        features[piece - 1][r][c] = 1.0
                    else:
                        features[7 + (-piece) - 1][r][c] = 1.0
                else:
                    if piece < 0:
                        features[(-piece) - 1][r][c] = 1.0
                    else:
                        features[7 + piece - 1][r][c] = 1.0

        if self.current_player == 1:
            features[14] = 1.0

        return features

    def get_canonical_board(self) -> np.ndarray:
        """
        获取标准化棋盘（始终从当前玩家视角看）
        如果当前是黑方，则翻转棋盘并交换颜色
        """
        if self.current_player == 1:
            return self.board.copy()
        else:
            # 翻转棋盘并交换颜色
            flipped = np.flip(self.board, axis=0).copy()
            return -flipped

    def display(self):
        """打印棋盘"""
        print("\n  ０ １ ２ ３ ４ ５ ６ ７ ８")
        print("  ＋－＋－＋－＋－＋－＋－＋－＋－＋")
        for r in range(ROWS - 1, -1, -1):
            row_str = f"{r} "
            for c in range(COLS):
                piece = self.board[r][c]
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

    # 测试特征提取
    features = game.get_state_for_nn()
    print(f"神经网络输入形状: {features.shape}")

    # 测试走几步
    moves = game.get_legal_moves()
    print(f"\n前5个合法走法:")
    for m in moves[:5]:
        fr, fc, tr, tc = m
        piece = PIECE_NAMES[game.board[fr][fc]]
        print(f"  {piece} ({fr},{fc}) -> ({tr},{tc})")
