"""最小化调试: 测试 Queue 在 Process 中的通信"""
import time
import numpy as np
import torch
from multiprocessing import Process, Queue

def server_loop(request_q, response_q, stop_flag_q):
    """简单的推理服务循环"""
    from model import XiangqiNet
    model = XiangqiNet(num_channels=32, num_res_blocks=2)
    model.eval()
    print("[Server] 模型已加载")

    while True:
        try:
            item = request_q.get(timeout=1.0)
        except Exception:
            # 检查是否需要停止
            try:
                stop_flag_q.get_nowait()
                print("[Server] 收到停止信号")
                break
            except:
                continue

        req_id, worker_id, state = item
        print(f"[Server] 收到请求 req_id={req_id}, worker_id={worker_id}, state shape={state.shape}")

        # 推理
        state_t = torch.from_numpy(state).float().unsqueeze(0)
        with torch.no_grad():
            policy_logits, value = model(state_t)
            policy = torch.softmax(policy_logits, dim=1).cpu().numpy()[0]
            val = value.cpu().item()

        response_q.put((req_id, policy, val))
        print(f"[Server] 已发送响应 req_id={req_id}")


if __name__ == '__main__':
    from game import XiangqiGame

    request_q = Queue()
    response_q = Queue()
    stop_q = Queue()

    p = Process(target=server_loop, args=(request_q, response_q, stop_q), daemon=True)
    p.start()
    print(f"[Main] Server PID={p.pid}")

    time.sleep(2)  # 等待模型加载

    # 发送请求
    game = XiangqiGame()
    state = game.get_state_for_nn()
    print(f"[Main] 发送请求, state shape={state.shape}")
    request_q.put((1, 0, state))

    # 等待响应
    print("[Main] 等待响应...")
    try:
        resp = response_q.get(timeout=10)
        req_id, policy, value = resp
        print(f"[Main] 收到响应: req_id={req_id}, policy shape={policy.shape}, value={value:.4f}")
        print("[Main] ✓ 通信成功!")
    except Exception as e:
        print(f"[Main] ✗ 超时或错误: {e}")

    stop_q.put(True)
    p.join(timeout=5)
    print("[Main] 完成")
