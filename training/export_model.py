"""
模型导出工具
============
将训练好的PyTorch模型导出为ONNX格式，以便在Web端使用ONNX Runtime推理。

使用方法：
    python export_model.py --model best_model.pt --output model.onnx
"""

import argparse
import torch
import numpy as np
from model import XiangqiNet
from game import ROWS, COLS, ACTION_SPACE


def export_to_onnx(model_path: str, output_path: str):
    """将PyTorch模型导出为ONNX格式"""

    # 加载模型
    checkpoint = torch.load(model_path, map_location='cpu')

    config = checkpoint.get('config', {})
    num_channels = config.get('num_channels', 128)
    num_res_blocks = config.get('num_res_blocks', 6)

    model = XiangqiNet(num_channels=num_channels, num_res_blocks=num_res_blocks)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # 创建示例输入
    dummy_input = torch.randn(1, 15, ROWS, COLS)

    # 导出ONNX
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=13,
        do_constant_folding=True,
        input_names=['state'],
        output_names=['policy', 'value'],
        dynamic_axes={
            'state': {0: 'batch_size'},
            'policy': {0: 'batch_size'},
            'value': {0: 'batch_size'}
        }
    )

    print(f"模型已导出: {output_path}")
    print(f"  网络配置: channels={num_channels}, res_blocks={num_res_blocks}")
    print(f"  输入形状: (batch, 15, {ROWS}, {COLS})")
    print(f"  输出: policy (batch, {ACTION_SPACE}), value (batch, 1)")

    # 验证
    try:
        import onnxruntime as ort
        session = ort.InferenceSession(output_path)
        test_input = np.random.randn(1, 15, ROWS, COLS).astype(np.float32)
        outputs = session.run(None, {'state': test_input})
        print(f"\n验证通过:")
        print(f"  Policy shape: {outputs[0].shape}")
        print(f"  Value shape: {outputs[1].shape}")
        print(f"  Value: {outputs[1][0][0]:.4f}")
    except ImportError:
        print("\n提示: 安装 onnxruntime 可验证导出的模型")
        print("  pip install onnxruntime")


def export_to_torchscript(model_path: str, output_path: str):
    """将PyTorch模型导出为TorchScript格式"""

    checkpoint = torch.load(model_path, map_location='cpu')
    config = checkpoint.get('config', {})
    num_channels = config.get('num_channels', 128)
    num_res_blocks = config.get('num_res_blocks', 6)

    model = XiangqiNet(num_channels=num_channels, num_res_blocks=num_res_blocks)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    dummy_input = torch.randn(1, 15, ROWS, COLS)
    traced = torch.jit.trace(model, dummy_input)
    traced.save(output_path)

    print(f"TorchScript模型已导出: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='模型导出工具')
    parser.add_argument('--model', type=str, required=True, help='PyTorch模型路径')
    parser.add_argument('--output', type=str, default='model.onnx', help='输出路径')
    parser.add_argument('--format', type=str, default='onnx',
                        choices=['onnx', 'torchscript'], help='导出格式')
    args = parser.parse_args()

    if args.format == 'onnx':
        export_to_onnx(args.model, args.output)
    else:
        export_to_torchscript(args.model, args.output)
