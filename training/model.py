"""
神经网络模型
============
基于ResNet的策略-价值双头网络，参考AlphaZero架构。

网络结构：
1. 输入层：15个特征平面 (15 x 10 x 9)
2. 残差塔：多个残差块
3. 策略头：输出每个动作的概率分布 (8100维)
4. 价值头：输出当前局面的评估值 (-1到1)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from game import ACTION_SPACE, ROWS, COLS


class ResBlock(nn.Module):
    """残差块"""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + residual
        out = F.relu(out)
        return out


class XiangqiNet(nn.Module):
    """
    中国象棋策略-价值网络

    参数:
        num_channels: 残差块中的卷积通道数（默认128）
        num_res_blocks: 残差块数量（默认6）
    """

    def __init__(self, num_channels: int = 128, num_res_blocks: int = 6):
        super().__init__()

        self.num_channels = num_channels
        self.num_res_blocks = num_res_blocks

        # 输入卷积层：15个特征平面 -> num_channels
        self.input_conv = nn.Sequential(
            nn.Conv2d(15, num_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(num_channels),
            nn.ReLU()
        )

        # 残差塔
        self.res_blocks = nn.ModuleList([
            ResBlock(num_channels) for _ in range(num_res_blocks)
        ])

        # 策略头
        self.policy_head = nn.Sequential(
            nn.Conv2d(num_channels, 32, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * ROWS * COLS, ACTION_SPACE)
        )

        # 价值头
        self.value_head = nn.Sequential(
            nn.Conv2d(num_channels, 4, 1, bias=False),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(4 * ROWS * COLS, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Tanh()
        )

    def forward(self, x):
        """
        前向传播

        输入: x - 形状 (batch, 15, 10, 9) 的特征张量
        输出: (policy_logits, value)
            - policy_logits: 形状 (batch, 8100) 的策略logits
            - value: 形状 (batch, 1) 的价值评估
        """
        # 输入卷积
        out = self.input_conv(x)

        # 残差塔
        for block in self.res_blocks:
            out = block(out)

        # 策略头和价值头
        policy_logits = self.policy_head(out)
        value = self.value_head(out)

        return policy_logits, value

    def predict(self, state: np.ndarray, device: str = 'cpu') -> tuple:
        """
        预测单个状态

        输入: state - 形状 (15, 10, 9) 的numpy数组
        输出: (policy_probs, value)
            - policy_probs: 形状 (8100,) 的概率分布
            - value: 标量值
        """
        self.eval()
        with torch.no_grad():
            x = torch.FloatTensor(state).unsqueeze(0).to(device)
            policy_logits, value = self(x)
            policy_probs = F.softmax(policy_logits, dim=1).squeeze(0).cpu().numpy()
            value = value.item()
        return policy_probs, value


def count_parameters(model):
    """统计模型参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # 测试模型
    model = XiangqiNet(num_channels=128, num_res_blocks=6)
    print(f"模型参数量: {count_parameters(model):,}")

    # 测试前向传播
    dummy_input = torch.randn(4, 15, 10, 9)
    policy, value = model(dummy_input)
    print(f"策略输出形状: {policy.shape}")
    print(f"价值输出形状: {value.shape}")

    # 测试单个预测
    single_state = np.random.randn(15, 10, 9).astype(np.float32)
    probs, val = model.predict(single_state)
    print(f"预测概率分布形状: {probs.shape}")
    print(f"预测价值: {val:.4f}")
    print(f"概率之和: {probs.sum():.4f}")
