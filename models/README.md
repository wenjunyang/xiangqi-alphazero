# 模型存储目录
训练产生的模型文件（.pt）会保存在此目录。
由于文件体积较大，不纳入版本控制。

运行训练后会自动生成：
- best_model.pt — 最优模型
- checkpoint_iter*.pt — 训练检查点
- training_stats.json — 训练统计数据

