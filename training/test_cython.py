"""
Cython 引擎正确性验证 + 性能对比测试
=====================================
1. 正确性：对比 Python 版和 Cython 版在多个局面下的走法生成结果
2. 性能：对比两者在走法生成、将军检测上的耗时
"""

import sys
import os
import time
import numpy as np

# 添加路径
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'cython_engine'))

from game import XiangqiGame, encode_action

# 尝试导入 Cython 引擎
try:
    from cython_engine.game_core import (
        cy_generate_legal_moves,
        cy_is_attacked,
        cy_find_king,
        cy_is_in_check,
        cy_has_legal_moves,
    )
    print("✓ Cython 引擎导入成功")
except ImportError as e:
    print(f"✗ Cython 引擎导入失败: {e}")
    print("请先编译: cd cython_engine && python setup.py build_ext --inplace")
    sys.exit(1)


def test_correctness():
    """正确性验证：对比 Python 和 Cython 的走法生成结果"""
    print("\n" + "=" * 60)
    print("正确性验证")
    print("=" * 60)

    errors = 0
    total_tests = 0

    # 测试1: 初始局面
    print("\n--- 测试1: 初始局面 ---")
    game = XiangqiGame()
    py_moves = set(game.get_legal_moves())
    cy_moves = set(cy_generate_legal_moves(game.board, game.current_player))
    total_tests += 1

    if py_moves == cy_moves:
        print(f"  ✓ 初始局面: Python={len(py_moves)}步, Cython={len(cy_moves)}步 一致")
    else:
        print(f"  ✗ 初始局面不一致!")
        print(f"    Python 独有: {py_moves - cy_moves}")
        print(f"    Cython 独有: {cy_moves - py_moves}")
        errors += 1

    # 测试2: 走几步后的局面
    print("\n--- 测试2: 对弈中的局面 ---")
    game = XiangqiGame()
    test_moves = [
        (2, 1, 4, 2),  # 红炮
        (7, 1, 5, 2),  # 黑炮
        (0, 1, 2, 2),  # 红马
        (9, 1, 7, 2),  # 黑马
        (3, 0, 4, 0),  # 红兵
        (6, 0, 5, 0),  # 黑卒
    ]
    for fr, fc, tr, tc in test_moves:
        game.make_move(fr, fc, tr, tc)

    py_moves = set(game.get_legal_moves())
    game._legal_moves_cache = None  # 清缓存
    cy_moves = set(cy_generate_legal_moves(game.board, game.current_player))
    total_tests += 1

    if py_moves == cy_moves:
        print(f"  ✓ 中局局面: Python={len(py_moves)}步, Cython={len(cy_moves)}步 一致")
    else:
        print(f"  ✗ 中局局面不一致!")
        print(f"    Python 独有: {py_moves - cy_moves}")
        print(f"    Cython 独有: {cy_moves - py_moves}")
        errors += 1

    # 测试3: 随机对弈多局，每步对比
    print("\n--- 测试3: 随机对弈 50 局，每步对比 ---")
    import random
    mismatch_count = 0
    total_steps = 0

    for game_idx in range(50):
        game = XiangqiGame()
        for step in range(100):
            py_moves = set(game.get_legal_moves())
            game._legal_moves_cache = None
            cy_moves = set(cy_generate_legal_moves(game.board, game.current_player))
            total_steps += 1

            if py_moves != cy_moves:
                mismatch_count += 1
                if mismatch_count <= 3:
                    print(f"  ✗ 局{game_idx} 步{step}: Python={len(py_moves)}, Cython={len(cy_moves)}")
                    print(f"    Python 独有: {py_moves - cy_moves}")
                    print(f"    Cython 独有: {cy_moves - py_moves}")
                    game.display()

            if len(py_moves) == 0:
                break

            move = random.choice(list(py_moves))
            game.make_move(*move)

            over, winner = game.is_game_over()
            if over:
                break

    total_tests += 1
    if mismatch_count == 0:
        print(f"  ✓ {total_steps} 步全部一致")
    else:
        print(f"  ✗ {mismatch_count}/{total_steps} 步不一致")
        errors += 1

    # 测试4: 将军检测
    print("\n--- 测试4: 将军检测 ---")
    game = XiangqiGame()
    for player in [1, -1]:
        king_pos = cy_find_king(game.board, player)
        py_check = game._is_in_check(player)
        cy_check = cy_is_in_check(game.board, player)
        total_tests += 1
        name = "红方" if player == 1 else "黑方"
        if py_check == cy_check:
            print(f"  ✓ {name}将军检测一致: {py_check}, 将位置={king_pos}")
        else:
            print(f"  ✗ {name}将军检测不一致: Python={py_check}, Cython={cy_check}")
            errors += 1

    print(f"\n正确性总结: {total_tests - errors}/{total_tests} 通过")
    return errors == 0


def test_performance():
    """性能对比测试"""
    print("\n" + "=" * 60)
    print("性能对比测试")
    print("=" * 60)

    game = XiangqiGame()

    # 预热
    for _ in range(10):
        game._legal_moves_cache = None
        game.get_legal_moves()
        cy_generate_legal_moves(game.board, game.current_player)

    N = 5000

    # --- 走法生成性能 ---
    print(f"\n--- 走法生成 (初始局面, {N} 次) ---")

    # Python
    start = time.perf_counter()
    for _ in range(N):
        game._legal_moves_cache = None
        game.get_legal_moves()
    py_time = time.perf_counter() - start

    # Cython
    start = time.perf_counter()
    for _ in range(N):
        cy_generate_legal_moves(game.board, game.current_player)
    cy_time = time.perf_counter() - start

    print(f"  Python:  {py_time * 1000:.1f} ms ({py_time / N * 1000:.3f} ms/次)")
    print(f"  Cython:  {cy_time * 1000:.1f} ms ({cy_time / N * 1000:.3f} ms/次)")
    print(f"  加速比:  {py_time / cy_time:.1f}x")

    # --- 中局走法生成 ---
    print(f"\n--- 走法生成 (中局局面, {N} 次) ---")
    game2 = XiangqiGame()
    moves_seq = [
        (2, 1, 4, 2), (7, 1, 5, 2), (0, 1, 2, 2), (9, 1, 7, 2),
        (3, 0, 4, 0), (6, 0, 5, 0), (3, 4, 4, 4), (6, 4, 5, 4),
    ]
    for m in moves_seq:
        game2.make_move(*m)

    start = time.perf_counter()
    for _ in range(N):
        game2._legal_moves_cache = None
        game2.get_legal_moves()
    py_time2 = time.perf_counter() - start

    start = time.perf_counter()
    for _ in range(N):
        cy_generate_legal_moves(game2.board, game2.current_player)
    cy_time2 = time.perf_counter() - start

    print(f"  Python:  {py_time2 * 1000:.1f} ms ({py_time2 / N * 1000:.3f} ms/次)")
    print(f"  Cython:  {cy_time2 * 1000:.1f} ms ({cy_time2 / N * 1000:.3f} ms/次)")
    print(f"  加速比:  {py_time2 / cy_time2:.1f}x")

    # --- 将军检测性能 ---
    print(f"\n--- 将军检测 ({N} 次) ---")

    start = time.perf_counter()
    for _ in range(N):
        game._is_in_check(1)
    py_check_time = time.perf_counter() - start

    start = time.perf_counter()
    for _ in range(N):
        cy_is_in_check(game.board, 1)
    cy_check_time = time.perf_counter() - start

    print(f"  Python:  {py_check_time * 1000:.1f} ms ({py_check_time / N * 1000:.4f} ms/次)")
    print(f"  Cython:  {cy_check_time * 1000:.1f} ms ({cy_check_time / N * 1000:.4f} ms/次)")
    print(f"  加速比:  {py_check_time / cy_check_time:.1f}x")

    # --- 模拟 MCTS 场景 ---
    print(f"\n--- 模拟 MCTS 场景 (80 次模拟 × 走法生成+将军检测) ---")
    import random
    game3 = XiangqiGame()

    # Python 版
    start = time.perf_counter()
    for _ in range(80):
        g = game3.clone()
        for step in range(50):
            g._legal_moves_cache = None
            moves = g.get_legal_moves()
            if not moves:
                break
            m = random.choice(moves)
            g.make_move(*m)
            over, _ = g.is_game_over()
            if over:
                break
    py_mcts_time = time.perf_counter() - start

    # Cython 版
    start = time.perf_counter()
    for _ in range(80):
        g = game3.clone()
        for step in range(50):
            moves = cy_generate_legal_moves(g.board, g.current_player)
            if not moves:
                break
            m = random.choice(moves)
            g.make_move(*m)
            # 简单终局检测
            if not cy_has_legal_moves(g.board, g.current_player):
                break
    cy_mcts_time = time.perf_counter() - start

    print(f"  Python:  {py_mcts_time * 1000:.1f} ms")
    print(f"  Cython:  {cy_mcts_time * 1000:.1f} ms")
    print(f"  加速比:  {py_mcts_time / cy_mcts_time:.1f}x")


if __name__ == "__main__":
    ok = test_correctness()
    if ok:
        test_performance()
    else:
        print("\n正确性测试未通过，跳过性能测试")
        sys.exit(1)
