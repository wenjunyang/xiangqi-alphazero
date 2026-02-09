"""
自对弈性能剖析
===============
精确测量一局自对弈中各环节的耗时占比。
直接在 MCTS.search 的关键路径上插桩计时。
"""

import time
import random
import numpy as np
import torch
from collections import defaultdict

from game import XiangqiGame, ACTION_SPACE, decode_action, encode_action
from model import XiangqiNet


def profile_single_search(model, game, num_simulations, device='cpu'):
    """
    对一次 MCTS search 进行精细计时，
    手动展开 MCTS.search 的逻辑以插入计时点。
    """
    from mcts import MCTSNode

    timers = defaultdict(float)
    counters = defaultdict(int)

    t_total = time.perf_counter()

    # ---- 根节点初始化 ----
    root = MCTSNode()

    t0 = time.perf_counter()
    state = game.get_state_for_nn()
    timers['get_state'] += time.perf_counter() - t0

    t0 = time.perf_counter()
    policy_probs, _ = model.predict(state, device)
    timers['nn_predict'] += time.perf_counter() - t0
    counters['nn_predict'] += 1

    t0 = time.perf_counter()
    legal_actions = game.get_legal_actions()
    timers['get_legal_actions'] += time.perf_counter() - t0
    counters['get_legal_actions'] += 1

    if len(legal_actions) == 0:
        timers['total'] = time.perf_counter() - t_total
        return timers, counters

    # 掩码 + 归一化
    t0 = time.perf_counter()
    action_priors = {}
    prob_sum = sum(policy_probs[a] for a in legal_actions)
    if prob_sum > 0:
        for a in legal_actions:
            action_priors[a] = policy_probs[a] / prob_sum
    else:
        uniform = 1.0 / len(legal_actions)
        for a in legal_actions:
            action_priors[a] = uniform
    timers['mask_normalize'] += time.perf_counter() - t0

    # Dirichlet 噪声
    t0 = time.perf_counter()
    noise = np.random.dirichlet([0.3] * len(legal_actions))
    actions_list = list(action_priors.keys())
    for i, a in enumerate(actions_list):
        action_priors[a] = 0.75 * action_priors[a] + 0.25 * noise[i]
    timers['dirichlet'] += time.perf_counter() - t0

    root.expand(action_priors)

    # ---- 模拟循环 ----
    for sim_idx in range(num_simulations):
        node = root

        # 1. clone 游戏状态
        t0 = time.perf_counter()
        sim_game = game.clone()
        timers['game_clone'] += time.perf_counter() - t0
        counters['game_clone'] += 1

        # 2. 选择（沿树下行）
        t0 = time.perf_counter()
        depth = 0
        while not node.is_leaf():
            action, node = node.select_child(1.5)
            sim_game.make_action(action)
            depth += 1
        timers['select'] += time.perf_counter() - t0
        counters['select'] += 1
        counters['select_depth_total'] += depth

        # 3. 检查终局
        t0 = time.perf_counter()
        done, winner = sim_game.is_game_over()
        timers['is_game_over'] += time.perf_counter() - t0
        counters['is_game_over'] += 1

        if done:
            value = 0.0 if winner == 0 else -1.0
        else:
            # 4. 状态编码
            t0 = time.perf_counter()
            state = sim_game.get_state_for_nn()
            timers['get_state'] += time.perf_counter() - t0
            counters['get_state'] += 1

            # 5. 神经网络推理
            t0 = time.perf_counter()
            policy_probs, value = model.predict(state, device)
            timers['nn_predict'] += time.perf_counter() - t0
            counters['nn_predict'] += 1

            # 6. 获取合法走法
            t0 = time.perf_counter()
            leaf_legal = sim_game.get_legal_actions()
            timers['get_legal_actions'] += time.perf_counter() - t0
            counters['get_legal_actions'] += 1

            # 7. 扩展节点
            t0 = time.perf_counter()
            if len(leaf_legal) > 0:
                lp_sum = sum(policy_probs[a] for a in leaf_legal)
                if lp_sum > 0:
                    ap = {a: policy_probs[a] / lp_sum for a in leaf_legal}
                else:
                    u = 1.0 / len(leaf_legal)
                    ap = {a: u for a in leaf_legal}
                node.expand(ap)
            timers['expand'] += time.perf_counter() - t0

            value = -value

        # 8. 回溯
        t0 = time.perf_counter()
        node.backup(value)
        timers['backup'] += time.perf_counter() - t0
        counters['backup'] += 1

    # 提取策略
    t0 = time.perf_counter()
    action_probs = np.zeros(ACTION_SPACE)
    for action, child in root.children.items():
        action_probs[action] = child.visit_count
    total_visits = action_probs.sum()
    if total_visits > 0:
        action_probs /= total_visits
    timers['get_policy'] += time.perf_counter() - t0

    timers['total'] = time.perf_counter() - t_total
    return timers, counters


def run_profile(num_channels, num_res_blocks, num_simulations, num_steps, desc):
    """对指定配置运行性能剖析"""
    print(f"\n{'='*70}")
    print(f"配置: {desc}")
    print(f"{'='*70}")

    model = XiangqiNet(num_channels=num_channels, num_res_blocks=num_res_blocks)
    model.eval()

    # warmup
    game = XiangqiGame()
    state = game.get_state_for_nn()
    with torch.no_grad():
        model.predict(state, 'cpu')

    # 随机开局
    for _ in range(random.randint(0, 4)):
        moves = game.get_legal_moves()
        if not moves:
            break
        game.make_move(*random.choice(moves))
        done, _ = game.is_game_over()
        if done:
            game = XiangqiGame()
            break

    # 逐步剖析
    agg_timers = defaultdict(float)
    agg_counters = defaultdict(int)
    step_durations = []

    for step in range(num_steps):
        done, _ = game.is_game_over()
        if done:
            break

        timers, counters = profile_single_search(model, game, num_simulations)
        step_durations.append(timers['total'])

        for k, v in timers.items():
            agg_timers[k] += v
        for k, v in counters.items():
            agg_counters[k] += v

        # 随机走一步
        legal = game.get_legal_moves()
        if not legal:
            break
        game.make_move(*random.choice(legal))

    actual_steps = len(step_durations)
    total = agg_timers['total']
    avg_step = total / actual_steps if actual_steps > 0 else 0

    print(f"\n剖析步数: {actual_steps}, 总耗时: {total:.2f}s, 平均每步: {avg_step:.3f}s")
    print(f"预估一局 (150步): {avg_step * 150:.0f}s = {avg_step * 150 / 60:.1f}min")

    # 详细分解
    print(f"\n{'环节':<28} {'耗时(s)':>10} {'占比':>8} {'次数':>8} {'平均(ms)':>10}")
    print("-" * 70)

    items = [
        ('nn_predict',        '神经网络推理'),
        ('get_legal_actions',  '走法生成(get_legal_actions)'),
        ('game_clone',         '游戏状态克隆(clone)'),
        ('select',             '树搜索(select_child)'),
        ('is_game_over',       '终局检查(is_game_over)'),
        ('get_state',          '状态编码(get_state_for_nn)'),
        ('expand',             '节点扩展(expand)'),
        ('backup',             '回溯更新(backup)'),
        ('mask_normalize',     '策略掩码归一化'),
        ('dirichlet',          'Dirichlet噪声'),
        ('get_policy',         '策略提取'),
    ]

    for key, label in items:
        t = agg_timers.get(key, 0)
        c = agg_counters.get(key, 1)
        pct = t / total * 100 if total > 0 else 0
        avg_ms = t / c * 1000 if c > 0 else 0
        print(f"  {label:<26} {t:>10.3f} {pct:>7.1f}% {c:>8} {avg_ms:>10.3f}")

    # 关键指标
    nn_time = agg_timers.get('nn_predict', 0)
    nn_count = agg_counters.get('nn_predict', 1)
    legal_time = agg_timers.get('get_legal_actions', 0)
    legal_count = agg_counters.get('get_legal_actions', 1)
    clone_time = agg_timers.get('game_clone', 0)
    clone_count = agg_counters.get('game_clone', 1)
    select_time = agg_timers.get('select', 0)
    game_over_time = agg_timers.get('is_game_over', 0)
    state_time = agg_timers.get('get_state', 0)

    print(f"\n--- 关键指标 ---")
    print(f"单次 NN 推理:          {nn_time/nn_count*1000:.2f} ms")
    print(f"单次走法生成:          {legal_time/legal_count*1000:.2f} ms")
    print(f"单次游戏克隆:          {clone_time/clone_count*1000:.3f} ms")
    print(f"每步 NN 推理次数:      {nn_count/actual_steps:.0f} (≈simulations+1)")
    print(f"每步走法生成次数:      {legal_count/actual_steps:.0f} (≈simulations+1)")
    print(f"每步游戏克隆次数:      {clone_count/actual_steps:.0f} (=simulations)")
    avg_depth = agg_counters.get('select_depth_total', 0) / max(agg_counters.get('select', 1), 1)
    print(f"平均选择深度:          {avg_depth:.1f}")

    # 耗时占比可视化
    print(f"\n--- 耗时占比可视化 ---")
    categories = [
        ('神经网络推理', nn_time),
        ('走法生成', legal_time),
        ('游戏克隆', clone_time),
        ('树搜索(选择)', select_time),
        ('终局检查', game_over_time),
        ('状态编码', state_time),
        ('其他', total - nn_time - legal_time - clone_time - select_time - game_over_time - state_time),
    ]
    for label, t in categories:
        pct = t / total * 100 if total > 0 else 0
        bar = '█' * int(pct / 2)
        print(f"  {label:<16} {bar:<50} {pct:.1f}%")


def main():
    print("=" * 70)
    print("中国象棋自对弈性能剖析")
    print("=" * 70)

    # 小模型 - 20 步
    run_profile(64, 3, 80, 20, "小模型 (64ch/3res/80sim)")

    # 标准模型 - 10 步 (更慢，少测几步)
    run_profile(128, 6, 200, 10, "标准模型 (128ch/6res/200sim)")


if __name__ == "__main__":
    main()
