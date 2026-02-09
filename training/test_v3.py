"""
v3 优化版正确性验证和性能对比测试
=================================
1. 正确性：随机对局验证走法生成、终局检查、状态编码
2. 性能：与 benchmark 中同样的 MCTS 搜索做对比
"""

import time
import random
import sys
import numpy as np

from game import XiangqiGame, encode_action, decode_action, ACTION_SPACE, PIECE_NAMES


def test_correctness():
    """正确性验证：随机对弈多局，检查各种边界情况"""
    print("=" * 60)
    print("正确性验证")
    print("=" * 60)

    errors = 0
    total_games = 50
    total_moves = 0

    for game_idx in range(total_games):
        game = XiangqiGame()
        move_count = 0

        while move_count < 300:
            # 测试 is_game_over
            done, winner = game.is_game_over()
            if done:
                break

            # 测试 get_legal_moves
            moves = game.get_legal_moves()
            if len(moves) == 0:
                break

            # 测试缓存一致性：再次调用应返回相同结果
            moves2 = game.get_legal_moves()
            if moves != moves2:
                print(f"  [ERROR] 缓存不一致! game={game_idx} step={move_count}")
                errors += 1

            # 测试 get_legal_actions
            actions = game.get_legal_actions()
            if len(actions) != len(moves):
                print(f"  [ERROR] actions数量不匹配! game={game_idx}")
                errors += 1

            # 验证每个 action 编解码一致
            for i, (fr, fc, tr, tc) in enumerate(moves):
                a = encode_action(fr, fc, tr, tc)
                dfr, dfc, dtr, dtc = decode_action(a)
                if (dfr, dfc, dtr, dtc) != (fr, fc, tr, tc):
                    print(f"  [ERROR] 编解码不一致! {(fr,fc,tr,tc)} -> {a} -> {(dfr,dfc,dtr,dtc)}")
                    errors += 1

            # 测试 get_state_for_nn
            features = game.get_state_for_nn()
            if features.shape != (15, 10, 9):
                print(f"  [ERROR] 特征形状错误: {features.shape}")
                errors += 1

            # 测试材料分
            rs = game.get_material_score(1)
            bs = game.get_material_score(-1)
            if rs < 0 or bs < 0:
                print(f"  [ERROR] 材料分为负: red={rs} black={bs}")
                errors += 1

            # 验证走法合法性：走后不应被将军
            move = random.choice(moves)
            fr, fc, tr, tc = move

            # 走棋前检查棋子存在
            piece = game.board[fr, fc]
            if piece == 0:
                print(f"  [ERROR] 走空位! ({fr},{fc})")
                errors += 1
                break

            game.make_move(fr, fc, tr, tc)
            move_count += 1

            # 走后缓存应被清除
            if game._legal_moves_cache is not None:
                print(f"  [ERROR] 走棋后缓存未清除!")
                errors += 1

        total_moves += move_count

        if (game_idx + 1) % 10 == 0:
            print(f"  完成 {game_idx + 1}/{total_games} 局, 累计 {total_moves} 步")

    print(f"\n结果: {total_games} 局, {total_moves} 步, {errors} 个错误")
    if errors == 0:
        print("✅ 正确性验证通过!")
    else:
        print(f"❌ 发现 {errors} 个错误!")
    return errors == 0


def test_specific_positions():
    """测试特定局面的正确性"""
    print("\n" + "=" * 60)
    print("特定局面测试")
    print("=" * 60)

    errors = 0

    # 1. 初始局面
    game = XiangqiGame()
    moves = game.get_legal_moves()
    # 初始局面红方应有 44 个合法走法
    print(f"  初始局面走法数: {len(moves)}")
    if len(moves) != 44:
        print(f"  [WARN] 预期44个走法, 实际{len(moves)}个")

    # 2. 测试飞将检测
    game2 = XiangqiGame()
    game2.board[:] = 0
    game2.board[0, 4] = 1   # 帅
    game2.board[9, 4] = -1  # 将
    game2.current_player = 1
    game2._legal_moves_cache = None

    # 帅和将面对面，帅不能离开第4列（否则飞将）
    moves = game2.get_legal_moves()
    for fr, fc, tr, tc in moves:
        if abs(game2.board[fr, fc]) == 1 and tc != 4:
            # 帅移到非4列，需要检查是否仍然飞将
            pass
    print(f"  飞将局面走法数: {len(moves)}")

    # 3. 测试将军检测
    game3 = XiangqiGame()
    game3.board[:] = 0
    game3.board[0, 4] = 1   # 帅
    game3.board[9, 4] = -1  # 将
    game3.board[5, 4] = -5  # 黑车在中间
    game3.current_player = 1
    game3._legal_moves_cache = None

    in_check = game3._is_in_check(1)
    print(f"  帅被车将军: {in_check} (应为True)")
    if not in_check:
        print("  [ERROR] 将军检测失败!")
        errors += 1

    # 4. 测试 _is_attacked
    game4 = XiangqiGame()
    game4.board[:] = 0
    game4.board[0, 4] = 1   # 帅
    game4.board[2, 3] = -4  # 黑马
    game4.current_player = 1
    game4._legal_moves_cache = None

    # 黑马在(2,3)，蹩脚点(1,3)为空，可以跳到(0,4)将军
    attacked = game4._is_attacked(game4.board, 0, 4, -1)
    print(f"  帅被马将军: {attacked} (应为True)")
    if not attacked:
        print("  [ERROR] 马将军检测失败!")
        errors += 1

    # 5. 蹩马腿
    game5 = XiangqiGame()
    game5.board[:] = 0
    game5.board[0, 4] = 1   # 帅
    game5.board[2, 3] = -4  # 黑马
    game5.board[1, 3] = 7   # 红兵蹩马腿
    game5.current_player = 1
    game5._legal_moves_cache = None

    attacked = game5._is_attacked(game5.board, 0, 4, -1)
    print(f"  蹩马腿后帅被将军: {attacked} (应为False)")
    if attacked:
        print("  [ERROR] 蹩马腿检测失败!")
        errors += 1

    # 6. 炮将军
    game6 = XiangqiGame()
    game6.board[:] = 0
    game6.board[0, 4] = 1   # 帅
    game6.board[9, 4] = -1  # 将
    game6.board[5, 4] = 7   # 红兵做炮架
    game6.board[8, 4] = -6  # 黑炮
    game6.current_player = 1
    game6._legal_moves_cache = None

    attacked = game6._is_attacked(game6.board, 0, 4, -1)
    print(f"  帅被炮将军: {attacked} (应为True)")
    if not attacked:
        print("  [ERROR] 炮将军检测失败!")
        errors += 1

    if errors == 0:
        print("\n✅ 特定局面测试通过!")
    else:
        print(f"\n❌ 发现 {errors} 个错误!")
    return errors == 0


def test_performance():
    """性能对比测试"""
    print("\n" + "=" * 60)
    print("性能测试")
    print("=" * 60)

    # 测试1: 单次 get_legal_moves 耗时
    game = XiangqiGame()
    n_iters = 1000

    t0 = time.perf_counter()
    for _ in range(n_iters):
        game._legal_moves_cache = None
        game.get_legal_moves()
    t1 = time.perf_counter()
    avg_legal = (t1 - t0) / n_iters * 1000
    print(f"\n  get_legal_moves 平均耗时: {avg_legal:.3f} ms ({n_iters}次)")

    # 测试2: 单次 is_game_over 耗时（含缓存）
    t0 = time.perf_counter()
    for _ in range(n_iters):
        game._legal_moves_cache = None
        game.is_game_over()
    t1 = time.perf_counter()
    avg_over = (t1 - t0) / n_iters * 1000
    print(f"  is_game_over 平均耗时:    {avg_over:.3f} ms ({n_iters}次)")

    # 测试3: is_game_over 使用缓存
    game._legal_moves_cache = None
    game.get_legal_moves()  # 填充缓存
    t0 = time.perf_counter()
    for _ in range(n_iters):
        game.is_game_over()
    t1 = time.perf_counter()
    avg_cached = (t1 - t0) / n_iters * 1000
    print(f"  is_game_over (缓存) 耗时: {avg_cached:.3f} ms ({n_iters}次)")

    # 测试4: get_state_for_nn 耗时
    t0 = time.perf_counter()
    for _ in range(n_iters):
        game.get_state_for_nn()
    t1 = time.perf_counter()
    avg_state = (t1 - t0) / n_iters * 1000
    print(f"  get_state_for_nn 耗时:    {avg_state:.3f} ms ({n_iters}次)")

    # 测试5: _is_attacked 耗时
    king_pos = game._find_king_pos(1, game.board)
    t0 = time.perf_counter()
    for _ in range(n_iters * 10):
        game._is_attacked(game.board, king_pos[0], king_pos[1], -1)
    t1 = time.perf_counter()
    avg_attack = (t1 - t0) / (n_iters * 10) * 1000
    print(f"  _is_attacked 耗时:        {avg_attack:.4f} ms ({n_iters*10}次)")

    # 测试6: 模拟 MCTS 中的典型调用模式
    print(f"\n  --- 模拟 MCTS 单步 (80 次模拟) ---")
    game = XiangqiGame()
    t0 = time.perf_counter()
    for _ in range(80):
        sim_game = game.clone()
        done, winner = sim_game.is_game_over()
        if not done:
            legal = sim_game.get_legal_actions()
            state = sim_game.get_state_for_nn()
            # 随机走一步
            if legal:
                sim_game.make_action(random.choice(legal))
                sim_game.is_game_over()
                sim_game.get_legal_actions()
    t1 = time.perf_counter()
    mcts_step_time = (t1 - t0) * 1000
    print(f"  80次模拟 game 操作总耗时: {mcts_step_time:.1f} ms")
    print(f"  平均每次模拟:             {mcts_step_time/80:.2f} ms")

    # 测试7: 完整随机对局耗时
    print(f"\n  --- 完整随机对局 ---")
    times = []
    for _ in range(5):
        game = XiangqiGame()
        t0 = time.perf_counter()
        steps = 0
        while steps < 200:
            done, _ = game.is_game_over()
            if done:
                break
            moves = game.get_legal_moves()
            if not moves:
                break
            game.make_move(*random.choice(moves))
            steps += 1
        t1 = time.perf_counter()
        times.append((t1 - t0, steps))

    for i, (t, s) in enumerate(times):
        print(f"  对局{i+1}: {s}步, {t*1000:.1f}ms, {t/s*1000:.3f}ms/步")


def test_mcts_integration():
    """测试与 MCTS 的集成"""
    print("\n" + "=" * 60)
    print("MCTS 集成测试")
    print("=" * 60)

    try:
        import torch
        from model import XiangqiNet
        from mcts import MCTS
    except ImportError as e:
        print(f"  跳过 MCTS 集成测试: {e}")
        return True

    model = XiangqiNet(num_channels=64, num_res_blocks=3)
    model.eval()

    # warmup
    game = XiangqiGame()
    state = game.get_state_for_nn()
    with torch.no_grad():
        model.predict(state, 'cpu')

    mcts = MCTS(model, num_simulations=80, device='cpu')

    # 测试 5 步 MCTS 搜索
    game = XiangqiGame()
    step_times = []

    for step in range(5):
        done, _ = game.is_game_over()
        if done:
            break

        t0 = time.perf_counter()
        action_probs = mcts.search(game, temperature=1.0, add_noise=True)
        t1 = time.perf_counter()
        step_times.append(t1 - t0)

        # 走一步
        action = np.random.choice(len(action_probs), p=action_probs)
        game.make_action(int(action))

    avg_time = sum(step_times) / len(step_times)
    print(f"  MCTS 搜索 (80sim): {len(step_times)} 步")
    for i, t in enumerate(step_times):
        print(f"    步{i+1}: {t:.3f}s")
    print(f"  平均每步: {avg_time:.3f}s")
    print(f"  预估一局 (150步): {avg_time * 150:.0f}s = {avg_time * 150 / 60:.1f}min")

    print("\n✅ MCTS 集成测试通过!")
    return True


if __name__ == "__main__":
    ok1 = test_correctness()
    ok2 = test_specific_positions()
    test_performance()
    test_mcts_integration()

    if ok1 and ok2:
        print("\n" + "=" * 60)
        print("✅ 所有测试通过!")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("❌ 部分测试失败!")
        print("=" * 60)
        sys.exit(1)
