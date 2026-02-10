# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True
# cython: language_level=3
"""
中国象棋引擎 Cython 加速核心
============================
将走法生成、将军检测、终局判定等热点函数用 C 类型重写。
所有函数操作 int8 二维数组（10x9），与 NumPy int8 棋盘兼容。
"""

from libc.stdlib cimport malloc, free
from libc.string cimport memcpy

import numpy as np
cimport numpy as cnp

cnp.import_array()

# 常量
DEF ROWS = 10
DEF COLS = 9
DEF EMPTY = 0
DEF R_KING = 1
DEF R_ADVISOR = 2
DEF R_BISHOP = 3
DEF R_KNIGHT = 4
DEF R_ROOK = 5
DEF R_CANNON = 6
DEF R_PAWN = 7

# 马的走法偏移: (dr, dc, block_dr, block_dc)
cdef int KNIGHT_MOVES[8][4]
KNIGHT_MOVES[0][:] = [-2, -1, -1, 0]
KNIGHT_MOVES[1][:] = [-2,  1, -1, 0]
KNIGHT_MOVES[2][:] = [ 2, -1,  1, 0]
KNIGHT_MOVES[3][:] = [ 2,  1,  1, 0]
KNIGHT_MOVES[4][:] = [-1, -2,  0, -1]
KNIGHT_MOVES[5][:] = [-1,  2,  0,  1]
KNIGHT_MOVES[6][:] = [ 1, -2,  0, -1]
KNIGHT_MOVES[7][:] = [ 1,  2,  0,  1]

# 四个直线方向
cdef int DIRECTIONS[4][2]
DIRECTIONS[0][:] = [-1, 0]
DIRECTIONS[1][:] = [ 1, 0]
DIRECTIONS[2][:] = [ 0, -1]
DIRECTIONS[3][:] = [ 0,  1]

# 走法存储结构: (from_row, from_col, to_row, to_col)
# 最大可能走法数（保守估计）
DEF MAX_MOVES = 200

# ============================================================
# 内部 C 函数
# ============================================================

cdef inline bint _in_board(int r, int c) noexcept nogil:
    return 0 <= r < ROWS and 0 <= c < COLS

cdef inline bint _is_enemy(signed char piece, int player) noexcept nogil:
    if player == 1:
        return piece < 0
    else:
        return piece > 0

cdef inline bint _is_own(signed char piece, int player) noexcept nogil:
    if player == 1:
        return piece > 0
    else:
        return piece < 0

cdef inline bint _can_move_to(signed char target, int player) noexcept nogil:
    """目标格可以走（空或敌方）"""
    if target == EMPTY:
        return True
    return _is_enemy(target, player)


cdef void _find_king(signed char board[ROWS][COLS], int player,
                     int *kr, int *kc) noexcept nogil:
    """找将/帅位置，只搜索宫格范围"""
    cdef int r, c, target
    cdef int r_start, r_end

    if player == 1:
        target = R_KING
        r_start = 0
        r_end = 3
    else:
        target = -R_KING
        r_start = 7
        r_end = 10

    for r in range(r_start, r_end):
        for c in range(3, 6):
            if board[r][c] == target:
                kr[0] = r
                kc[0] = c
                return

    kr[0] = -1
    kc[0] = -1


cdef bint _is_attacked(signed char board[ROWS][COLS], int kr, int kc,
                        int by_player) noexcept nogil:
    """
    快速检测位置 (kr, kc) 是否被 by_player 方攻击。
    从目标位置反向检查各方向。
    """
    cdef int dr, dc, r, c, i
    cdef signed char p
    cdef signed char enemy_rook = R_ROOK * by_player
    cdef signed char enemy_cannon = R_CANNON * by_player
    cdef signed char enemy_knight = R_KNIGHT * by_player
    cdef signed char enemy_pawn = R_PAWN * by_player
    cdef signed char enemy_king = R_KING * by_player
    cdef bint found_platform
    cdef int nr, nc, move_dr, move_dc, blk_r, blk_c

    # 1. 车和将的直线攻击
    for i in range(4):
        dr = DIRECTIONS[i][0]
        dc = DIRECTIONS[i][1]
        r = kr + dr
        c = kc + dc
        while _in_board(r, c):
            p = board[r][c]
            if p != EMPTY:
                if p == enemy_rook or p == enemy_king:
                    return True
                break
            r += dr
            c += dc

    # 2. 炮的攻击
    for i in range(4):
        dr = DIRECTIONS[i][0]
        dc = DIRECTIONS[i][1]
        r = kr + dr
        c = kc + dc
        found_platform = False
        while _in_board(r, c):
            p = board[r][c]
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

    # 3. 马的攻击
    for i in range(8):
        nr = kr + KNIGHT_MOVES[i][0]
        nc = kc + KNIGHT_MOVES[i][1]
        if _in_board(nr, nc) and board[nr][nc] == enemy_knight:
            move_dr = kr - nr
            move_dc = kc - nc
            if move_dr == 2 or move_dr == -2:
                blk_r = nr + move_dr / 2
                blk_c = nc
            else:
                blk_r = nr
                blk_c = nc + move_dc / 2
            if board[blk_r][blk_c] == EMPTY:
                return True

    # 4. 兵/卒的攻击
    if by_player == 1:
        if kr - 1 >= 0 and board[kr - 1][kc] == enemy_pawn:
            return True
        if kr >= 5:
            if kc - 1 >= 0 and board[kr][kc - 1] == enemy_pawn:
                return True
            if kc + 1 < COLS and board[kr][kc + 1] == enemy_pawn:
                return True
    else:
        if kr + 1 < ROWS and board[kr + 1][kc] == enemy_pawn:
            return True
        if kr <= 4:
            if kc - 1 >= 0 and board[kr][kc - 1] == enemy_pawn:
                return True
            if kc + 1 < COLS and board[kr][kc + 1] == enemy_pawn:
                return True

    return False


cdef bint _kings_facing(signed char board[ROWS][COLS]) noexcept nogil:
    """检查将帅是否面对面"""
    cdef int rkr, rkc, bkr, bkc, r
    _find_king(board, 1, &rkr, &rkc)
    _find_king(board, -1, &bkr, &bkc)

    if rkr == -1 or bkr == -1:
        return False
    if rkc != bkc:
        return False

    for r in range(rkr + 1, bkr):
        if board[r][rkc] != EMPTY:
            return False
    return True


cdef bint _is_move_legal(signed char board[ROWS][COLS],
                          int fr, int fc, int tr, int tc,
                          int player) noexcept nogil:
    """
    检查走法是否合法（走后不被将军、不飞将）。
    就地修改 + 恢复。
    """
    cdef signed char moving_piece = board[fr][fc]
    cdef signed char captured = board[tr][tc]
    cdef int kr, kc, ekr, ekc, check_r
    cdef bint legal = True

    # 就地走棋
    board[tr][tc] = moving_piece
    board[fr][fc] = EMPTY

    # 找己方将
    _find_king(board, player, &kr, &kc)
    if kr == -1:
        board[fr][fc] = moving_piece
        board[tr][tc] = captured
        return False

    # 检查飞将
    _find_king(board, -player, &ekr, &ekc)
    if ekr != -1 and kc == ekc:
        legal = True
        for check_r in range(min(kr, ekr) + 1, max(kr, ekr)):
            if board[check_r][kc] != EMPTY:
                legal = True
                break
        else:
            # 循环正常结束 = 中间全空 = 飞将
            legal = False

    # 检查被将军
    if legal:
        legal = not _is_attacked(board, kr, kc, -player)

    # 恢复
    board[fr][fc] = moving_piece
    board[tr][tc] = captured

    return legal


cdef inline int min_i(int a, int b) noexcept nogil:
    return a if a < b else b

cdef inline int max_i(int a, int b) noexcept nogil:
    return a if a > b else b


cdef int _generate_moves(signed char board[ROWS][COLS], int player,
                          int moves[MAX_MOVES][4]) noexcept nogil:
    """
    生成所有合法走法，返回走法数量。
    moves[i] = [from_row, from_col, to_row, to_col]
    """
    cdef int count = 0
    cdef int r, c, nr, nc, i
    cdef signed char piece, p, abs_piece
    cdef int dr, dc, br, bc

    # 红方宫格范围
    cdef int palace_r_start, palace_r_end

    for r in range(ROWS):
        for c in range(COLS):
            piece = board[r][c]
            if piece == EMPTY:
                continue
            if not _is_own(piece, player):
                continue

            abs_piece = piece if piece > 0 else -piece

            # === 帅/将 ===
            if abs_piece == R_KING:
                if player == 1:
                    palace_r_start = 0
                    palace_r_end = 2
                else:
                    palace_r_start = 7
                    palace_r_end = 9
                for i in range(4):
                    nr = r + DIRECTIONS[i][0]
                    nc = c + DIRECTIONS[i][1]
                    if palace_r_start <= nr <= palace_r_end and 3 <= nc <= 5:
                        if _can_move_to(board[nr][nc], player):
                            if _is_move_legal(board, r, c, nr, nc, player):
                                moves[count][0] = r
                                moves[count][1] = c
                                moves[count][2] = nr
                                moves[count][3] = nc
                                count += 1

            # === 仕/士 ===
            elif abs_piece == R_ADVISOR:
                for dr in range(-1, 2, 2):
                    for dc in range(-1, 2, 2):
                        nr = r + dr
                        nc = c + dc
                        if not _in_board(nr, nc):
                            continue
                        if not (3 <= nc <= 5):
                            continue
                        if player == 1 and not (0 <= nr <= 2):
                            continue
                        if player == -1 and not (7 <= nr <= 9):
                            continue
                        if _can_move_to(board[nr][nc], player):
                            if _is_move_legal(board, r, c, nr, nc, player):
                                moves[count][0] = r
                                moves[count][1] = c
                                moves[count][2] = nr
                                moves[count][3] = nc
                                count += 1

            # === 相/象 ===
            elif abs_piece == R_BISHOP:
                for dr in range(-2, 3, 4):
                    for dc in range(-2, 3, 4):
                        nr = r + dr
                        nc = c + dc
                        if not _in_board(nr, nc):
                            continue
                        if player == 1 and nr > 4:
                            continue
                        if player == -1 and nr < 5:
                            continue
                        # 塞象眼
                        if board[r + dr / 2][c + dc / 2] != EMPTY:
                            continue
                        if _can_move_to(board[nr][nc], player):
                            if _is_move_legal(board, r, c, nr, nc, player):
                                moves[count][0] = r
                                moves[count][1] = c
                                moves[count][2] = nr
                                moves[count][3] = nc
                                count += 1

            # === 马 ===
            elif abs_piece == R_KNIGHT:
                for i in range(8):
                    nr = r + KNIGHT_MOVES[i][0]
                    nc = c + KNIGHT_MOVES[i][1]
                    if not _in_board(nr, nc):
                        continue
                    # 蹩马腿
                    if board[r + KNIGHT_MOVES[i][2]][c + KNIGHT_MOVES[i][3]] != EMPTY:
                        continue
                    if _can_move_to(board[nr][nc], player):
                        if _is_move_legal(board, r, c, nr, nc, player):
                            moves[count][0] = r
                            moves[count][1] = c
                            moves[count][2] = nr
                            moves[count][3] = nc
                            count += 1

            # === 车 ===
            elif abs_piece == R_ROOK:
                for i in range(4):
                    dr = DIRECTIONS[i][0]
                    dc = DIRECTIONS[i][1]
                    nr = r + dr
                    nc = c + dc
                    while _in_board(nr, nc):
                        p = board[nr][nc]
                        if p == EMPTY:
                            if _is_move_legal(board, r, c, nr, nc, player):
                                moves[count][0] = r
                                moves[count][1] = c
                                moves[count][2] = nr
                                moves[count][3] = nc
                                count += 1
                        elif _is_enemy(p, player):
                            if _is_move_legal(board, r, c, nr, nc, player):
                                moves[count][0] = r
                                moves[count][1] = c
                                moves[count][2] = nr
                                moves[count][3] = nc
                                count += 1
                            break
                        else:
                            break
                        nr += dr
                        nc += dc

            # === 炮 ===
            elif abs_piece == R_CANNON:
                for i in range(4):
                    dr = DIRECTIONS[i][0]
                    dc = DIRECTIONS[i][1]
                    nr = r + dr
                    nc = c + dc
                    # 不吃子移动
                    while _in_board(nr, nc) and board[nr][nc] == EMPTY:
                        if _is_move_legal(board, r, c, nr, nc, player):
                            moves[count][0] = r
                            moves[count][1] = c
                            moves[count][2] = nr
                            moves[count][3] = nc
                            count += 1
                        nr += dr
                        nc += dc
                    # 跳过炮架
                    if _in_board(nr, nc):
                        nr += dr
                        nc += dc
                        while _in_board(nr, nc):
                            p = board[nr][nc]
                            if p != EMPTY:
                                if _is_enemy(p, player):
                                    if _is_move_legal(board, r, c, nr, nc, player):
                                        moves[count][0] = r
                                        moves[count][1] = c
                                        moves[count][2] = nr
                                        moves[count][3] = nc
                                        count += 1
                                break
                            nr += dr
                            nc += dc

            # === 兵/卒 ===
            elif abs_piece == R_PAWN:
                if player == 1:
                    # 向前
                    nr = r + 1
                    if nr < ROWS and _can_move_to(board[nr][c], player):
                        if _is_move_legal(board, r, c, nr, c, player):
                            moves[count][0] = r
                            moves[count][1] = c
                            moves[count][2] = nr
                            moves[count][3] = c
                            count += 1
                    # 过河后左右
                    if r >= 5:
                        if c - 1 >= 0 and _can_move_to(board[r][c - 1], player):
                            if _is_move_legal(board, r, c, r, c - 1, player):
                                moves[count][0] = r
                                moves[count][1] = c
                                moves[count][2] = r
                                moves[count][3] = c - 1
                                count += 1
                        if c + 1 < COLS and _can_move_to(board[r][c + 1], player):
                            if _is_move_legal(board, r, c, r, c + 1, player):
                                moves[count][0] = r
                                moves[count][1] = c
                                moves[count][2] = r
                                moves[count][3] = c + 1
                                count += 1
                else:
                    nr = r - 1
                    if nr >= 0 and _can_move_to(board[nr][c], player):
                        if _is_move_legal(board, r, c, nr, c, player):
                            moves[count][0] = r
                            moves[count][1] = c
                            moves[count][2] = nr
                            moves[count][3] = c
                            count += 1
                    if r <= 4:
                        if c - 1 >= 0 and _can_move_to(board[r][c - 1], player):
                            if _is_move_legal(board, r, c, r, c - 1, player):
                                moves[count][0] = r
                                moves[count][1] = c
                                moves[count][2] = r
                                moves[count][3] = c - 1
                                count += 1
                        if c + 1 < COLS and _can_move_to(board[r][c + 1], player):
                            if _is_move_legal(board, r, c, r, c + 1, player):
                                moves[count][0] = r
                                moves[count][1] = c
                                moves[count][2] = r
                                moves[count][3] = c + 1
                                count += 1

    return count


# ============================================================
# Python 接口
# ============================================================

def cy_find_king(cnp.ndarray[signed char, ndim=2] board, int player):
    """找将/帅位置"""
    cdef signed char c_board[ROWS][COLS]
    cdef int kr, kc, r, c

    for r in range(ROWS):
        for c in range(COLS):
            c_board[r][c] = board[r, c]

    _find_king(c_board, player, &kr, &kc)
    if kr == -1:
        return None
    return (kr, kc)


def cy_is_attacked(cnp.ndarray[signed char, ndim=2] board,
                    int kr, int kc, int by_player):
    """检测位置是否被攻击"""
    cdef signed char c_board[ROWS][COLS]
    cdef int r, c

    for r in range(ROWS):
        for c in range(COLS):
            c_board[r][c] = board[r, c]

    return _is_attacked(c_board, kr, kc, by_player)


def cy_generate_legal_moves(cnp.ndarray[signed char, ndim=2] board, int player):
    """
    生成所有合法走法。
    返回: list of (from_row, from_col, to_row, to_col)
    """
    cdef signed char c_board[ROWS][COLS]
    cdef int c_moves[MAX_MOVES][4]
    cdef int r, c, count, i

    for r in range(ROWS):
        for c in range(COLS):
            c_board[r][c] = board[r, c]

    count = _generate_moves(c_board, player, c_moves)

    result = []
    for i in range(count):
        result.append((c_moves[i][0], c_moves[i][1],
                        c_moves[i][2], c_moves[i][3]))
    return result


def cy_is_in_check(cnp.ndarray[signed char, ndim=2] board, int player):
    """检查指定方是否被将军"""
    cdef signed char c_board[ROWS][COLS]
    cdef int kr, kc, r, c

    for r in range(ROWS):
        for c in range(COLS):
            c_board[r][c] = board[r, c]

    _find_king(c_board, player, &kr, &kc)
    if kr == -1:
        return True
    return _is_attacked(c_board, kr, kc, -player)


def cy_has_legal_moves(cnp.ndarray[signed char, ndim=2] board, int player):
    """检查是否有合法走法（用于终局检测，找到一个即返回）"""
    cdef signed char c_board[ROWS][COLS]
    cdef int c_moves[MAX_MOVES][4]
    cdef int r, c, count

    for r in range(ROWS):
        for c in range(COLS):
            c_board[r][c] = board[r, c]

    count = _generate_moves(c_board, player, c_moves)
    return count > 0
