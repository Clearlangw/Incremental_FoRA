#!/usr/bin/env python3
"""
测试Detect到ReDetect的转换功能
"""

import torch
import yaml
from pathlib import Path

# 模拟一个简单的YAML配置
test_yaml = {
    "nc": 80,  # 总类别数
    "base_nc": 60,  # 基础类别数
    "ch": 3,
    "backbone": [
        [-1, 1, "Conv", [64, 3, 2]],
        [-1, 1, "Conv", [128, 3, 2]],
        [-1, 1, "Conv", [256, 3, 2]],
    ],
    "head": [
        [-1, 1, "Conv", [512, 3, 1]],
        [-1, 1, "Conv", [256, 3, 1]],
        [[-1, -2], 1, "Concat", [1]],
        [[-1], 1, "Detect", [80]],  # 原始Detect层
    ]
}

def create_mock_model():
    """创建一个模拟的模型状态字典"""
    state_dict = {}
    
    # 模拟backbone层
    state_dict['model.0.conv.weight'] = torch.randn(64, 3, 3, 3)
    state_dict['model.0.conv.bias'] = torch.randn(64)
    state_dict['model.1.conv.weight'] = torch.randn(128, 64, 3, 3)
    state_dict['model.1.conv.bias'] = torch.randn(128)
    state_dict['model.2.conv.weight'] = torch.randn(256, 128, 3, 3)
    state_dict['model.2.conv.bias'] = torch.randn(256)
    
    # 模拟head层
    state_dict['model.3.conv.weight'] = torch.randn(512, 256, 3, 3)
    state_dict['model.3.conv.bias'] = torch.randn(512)
    state_dict['model.4.conv.weight'] = torch.randn(256, 512, 3, 3)
    state_dict['model.4.conv.bias'] = torch.randn(256)
    
    # 模拟Detect层的cv2和cv3
    state_dict['model.6.cv2.0.0.conv.weight'] = torch.randn(64, 256, 3, 3)
    state_dict['model.6.cv2.0.0.conv.bias'] = torch.randn(64)
    state_dict['model.6.cv2.1.0.conv.weight'] = torch.randn(64, 64, 3, 3)
    state_dict['model.6.cv2.1.0.conv.bias'] = torch.randn(64)
    state_dict['model.6.cv2.2.weight'] = torch.randn(64, 64, 1, 1)  # 4 * reg_max
    state_dict['model.6.cv2.2.bias'] = torch.randn(64)
    
    state_dict['model.6.cv3.0.0.conv.weight'] = torch.randn(256, 256, 3, 3)
    state_dict['model.6.cv3.0.0.conv.bias'] = torch.randn(256)
    state_dict['model.6.cv3.1.0.conv.weight'] = torch.randn(256, 256, 3, 3)
    state_dict['model.6.cv3.1.0.conv.bias'] = torch.randn(256)
    state_dict['model.6.cv3.2.weight'] = torch.randn(80, 256, 1, 1)  # nc
    state_dict['model.6.cv3.2.bias'] = torch.randn(80)
    
    # 模拟dfl层
    state_dict['model.6.dfl.conv.weight'] = torch.randn(16, 4, 1, 1)
    state_dict['model.6.dfl.conv.bias'] = torch.randn(16)
    
    return state_dict

def test_conversion():
    """测试转换功能"""
    print("开始测试Detect到ReDetect的转换...")
    
    # 创建模拟的检查点
    ckpt = {
        'model': type('MockModel', (), {
            'yaml': test_yaml,
            'state_dict': lambda: create_mock_model()
        })(),
        'train_args': {}
    }
    
    # 导入转换函数
    import sys
    sys.path.append('.')
    from ultralytics.nn.tasks import rebuild_model_with_redetect
    
    # 执行转换
    try:
        converted_ckpt = rebuild_model_with_redetect(ckpt)
        print("✅ 转换成功！")
        
        # 检查转换后的状态字典
        state_dict = converted_ckpt['model'].state_dict()
        
        # 检查是否存在base_cv3和novel_cv3
        base_cv3_keys = [k for k in state_dict.keys() if 'base_cv3' in k]
        novel_cv3_keys = [k for k in state_dict.keys() if 'novel_cv3' in k]
        
        print(f"找到 {len(base_cv3_keys)} 个base_cv3键")
        print(f"找到 {len(novel_cv3_keys)} 个novel_cv3键")
        
        # 检查权重形状
        for key in base_cv3_keys:
            if 'weight' in key and '2.weight' in key:  # 最后一层
                print(f"base_cv3输出层权重形状: {state_dict[key].shape}")
                assert state_dict[key].shape[0] == 60, f"base_cv3输出应该是60，实际是{state_dict[key].shape[0]}"
        
        for key in novel_cv3_keys:
            if 'weight' in key and '2.weight' in key:  # 最后一层
                print(f"novel_cv3输出层权重形状: {state_dict[key].shape}")
                assert state_dict[key].shape[0] == 80, f"novel_cv3输出应该是80，实际是{state_dict[key].shape[0]}"
        
        print("✅ 权重形状检查通过！")
        
    except Exception as e:
        print(f"❌ 转换失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_conversion() 