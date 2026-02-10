"""
GPU 模式完整训练流程测试
========================
在 CPU 上模拟 GPU 推理服务模式的完整训练循环。
"""

if __name__ == '__main__':
    import logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

    from game import XiangqiGame, _USE_CYTHON
    print(f'Cython: {_USE_CYTHON}')

    from train import quick_train, AlphaZeroTrainer

    config = quick_train()
    config.num_iterations = 1
    config.num_games_per_iter = 4
    config.num_simulations = 20
    config.eval_games = 0  # 跳过评估
    config.min_buffer_size = 10
    config.max_game_length = 60
    config.parallel = True
    config.use_gpu_server = True
    config.gpu_device = 'cpu'  # 用 CPU 模拟 GPU

    trainer = AlphaZeroTrainer(config)
    trainer.train()
    print('\n✓ GPU 模式训练流程测试通过！')
