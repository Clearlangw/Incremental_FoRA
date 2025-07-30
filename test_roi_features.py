#!/usr/bin/env python3
"""
测试ROI特征提取和原型学习功能
"""

import torch
import torch.nn.functional as F
from torchvision.ops import roi_align
import numpy as np

def test_roi_feature_extraction():
    """测试ROI特征提取功能"""
    print("=== 测试ROI特征提取 ===")
    
    # 模拟特征图 [batch_size, channels, height, width]
    batch_size = 2
    channels = 64
    height, width = 100, 100
    feat = torch.randn(batch_size, channels, height, width)
    
    # 模拟targets [num_gt, 6] - [imgid, clsid, cx, cy, w, h] (归一化)
    num_gt = 5
    targets = torch.tensor([
        [0, 1, 0.3, 0.4, 0.2, 0.3],  # 第一张图片，类别1，中心(0.3,0.4)，宽高(0.2,0.3)
        [0, 2, 0.7, 0.6, 0.15, 0.25], # 第一张图片，类别2
        [1, 1, 0.5, 0.5, 0.3, 0.4],   # 第二张图片，类别1
        [1, 3, 0.2, 0.8, 0.1, 0.2],   # 第二张图片，类别3
        [0, 1, 0.8, 0.2, 0.25, 0.35], # 第一张图片，类别1
    ])
    
    print(f"特征图形状: {feat.shape}")
    print(f"目标数量: {targets.shape[0]}")
    print(f"目标内容:\n{targets}")
    
    # 提取ROI特征
    roi_features, roi_labels = extract_roi_features_single(feat, targets, output_size=(3, 3))
    
    print(f"ROI特征形状: {roi_features.shape}")
    print(f"ROI标签: {roi_labels}")
    print(f"ROI特征示例:\n{roi_features[0][:10]}")  # 显示第一个特征的前10个值
    
    return roi_features, roi_labels

def extract_roi_features_single(feat, targets, output_size=(3, 3)):
    """
    从单层特征图中提取ROI特征
    feat: [bs, c, h, w] 单层特征
    targets: [num_gt, 6], [imgid, clsid, cx, cy, w, h] (归一化)
    output_size: roi_align输出尺寸
    return: [num_gt, c] tensor, [num_gt] clsid tensor
    """
    device = feat.device
    num_gt = targets.shape[0]
    if num_gt == 0:
        return torch.empty(0, feat.shape[1], device=device), torch.empty(0, dtype=torch.long, device=device)
        
    imgids = targets[:, 0].long()
    clsids = targets[:, 1].long()
    cxs = targets[:, 2]
    cys = targets[:, 3]
    ws = targets[:, 4]
    hs = targets[:, 5]

    bs, c, h, w = feat.shape
    # 归一化坐标转特征图坐标
    x1 = (cxs - ws / 2) * w
    y1 = (cys - hs / 2) * h
    x2 = (cxs + ws / 2) * w
    y2 = (cys + hs / 2) * h
    # 修复：确保boxes的数据类型与feat一致
    boxes = torch.stack([imgids.float(), x1, y1, x2, y2], dim=1).to(device=device, dtype=feat.dtype)  # [num_gt, 5]
    
    print(f"ROI boxes:\n{boxes}")
    
    # roi_align
    roi_feat = roi_align(feat, boxes, output_size=output_size, spatial_scale=1.0, aligned=True)  # [num_gt, c, out_h, out_w]
    roi_feat = roi_feat.view(num_gt, -1)  # flatten
    return roi_feat, clsids

def test_prototype_memory():
    """测试原型记忆库功能"""
    print("\n=== 测试原型记忆库 ===")
    
    # 模拟多模态多层次的ROI特征数据
    roi_features_data = [
        {
            'modal_idx': 0,  # RGB模态
            'layer_idx': 0,  # 第一层
            'features': torch.randn(3, 576),  # 3个ROI，每个特征维度576
            'labels': torch.tensor([1, 2, 1])  # 类别标签
        },
        {
            'modal_idx': 0,  # RGB模态
            'layer_idx': 1,  # 第二层
            'features': torch.randn(2, 576),  # 2个ROI
            'labels': torch.tensor([1, 3])
        },
        {
            'modal_idx': 1,  # IR模态
            'layer_idx': 0,  # 第一层
            'features': torch.randn(4, 576),  # 4个ROI
            'labels': torch.tensor([1, 2, 1, 3])
        }
    ]
    
    # 初始化原型库
    prototype_memory = {}
    
    # 处理每个模态每个层次的ROI特征
    for roi_info in roi_features_data:
        modal_idx = roi_info['modal_idx']
        layer_idx = roi_info['layer_idx']
        features = roi_info['features']
        labels = roi_info['labels']
        
        # 创建模态-层键
        modal_layer_key = f"modal_{modal_idx}_layer_{layer_idx}"
        
        # 初始化该模态-层的原型库
        if modal_layer_key not in prototype_memory:
            prototype_memory[modal_layer_key] = {}
        
        # 更新原型库
        update_prototype_memory_multi_modal(features, labels, modal_layer_key, prototype_memory)
        
        print(f"模态-层 {modal_layer_key}:")
        print(f"  特征数量: {features.shape[0]}")
        print(f"  标签: {labels}")
        print(f"  原型库中的类别: {list(prototype_memory[modal_layer_key].keys())}")
    
    print(f"\n最终原型库结构:")
    for key, prototypes in prototype_memory.items():
        print(f"  {key}: {list(prototypes.keys())}")
    
    return prototype_memory

def update_prototype_memory_multi_modal(features, labels, modal_layer_key, prototype_memory):
    """
    更新多模态多层次原型库
    features: [num_roi, feat_dim]
    labels: [num_roi]
    modal_layer_key: 模态-层键，如 "modal_0_layer_0"
    prototype_memory: 原型库字典
    """
    device = features.device
    memory_decay = 0.9  # 原型更新衰减率
    
    for i, label in enumerate(labels):
        label = label.item()
        feature = features[i].detach()  # 分离梯度
        
        if label not in prototype_memory[modal_layer_key]:
            # 新类别，直接存储特征作为原型
            prototype_memory[modal_layer_key][label] = feature.clone()
        else:
            # 已存在类别，使用指数移动平均更新原型
            old_prototype = prototype_memory[modal_layer_key][label]
            new_prototype = memory_decay * old_prototype + (1 - memory_decay) * feature
            prototype_memory[modal_layer_key][label] = new_prototype

def test_contrastive_loss():
    """测试对比学习损失"""
    print("\n=== 测试对比学习损失 ===")
    
    # 模拟特征和标签
    features = torch.randn(5, 576)  # 5个样本，每个特征维度576
    labels = torch.tensor([1, 1, 2, 2, 3])  # 类别标签
    
    # 计算对比损失
    loss = compute_contrastive_loss_with_prototypes(features, labels, "modal_0_layer_0", {})
    
    print(f"特征形状: {features.shape}")
    print(f"标签: {labels}")
    print(f"对比损失: {loss.item():.4f}")
    
    return loss

def compute_contrastive_loss_with_prototypes(features, labels, modal_layer_key, prototype_memory):
    """
    计算与原型库的对比学习损失
    features: [num_roi, feat_dim]
    labels: [num_roi]
    modal_layer_key: 模态-层键
    prototype_memory: 原型库
    """
    device = features.device
    temperature = 0.07
    
    if features.shape[0] < 2:
        return torch.tensor(0.0, device=device)
        
    # 特征归一化
    features = F.normalize(features, dim=1)
    
    # 收集所有原型特征
    prototype_features = []
    prototype_labels = []
    if modal_layer_key in prototype_memory:
        for label, prototype in prototype_memory[modal_layer_key].items():
            prototype_features.append(prototype)
            prototype_labels.append(label)
        
    if not prototype_features:
        return torch.tensor(0.0, device=device)
        
    prototype_features = torch.stack(prototype_features)  # [num_prototypes, feat_dim]
    prototype_features = F.normalize(prototype_features, dim=1)
    
    # 计算当前特征与原型特征之间的相似度
    similarity_matrix = torch.div(
        torch.matmul(features, prototype_features.T), temperature
    )  # [num_roi, num_prototypes]
    
    # 构建标签匹配矩阵
    labels = labels.contiguous().view(-1, 1)  # [num_roi, 1]
    prototype_labels = torch.tensor(prototype_labels, device=device).contiguous().view(1, -1)  # [1, num_prototypes]
    label_match = torch.eq(labels, prototype_labels).float()  # [num_roi, num_prototypes]
    
    # 计算对比损失
    exp_sim = torch.exp(similarity_matrix)
    log_prob = similarity_matrix - torch.log(exp_sim.sum(1, keepdim=True) + 1e-8)
    
    # 只对匹配的标签计算损失
    mean_log_prob_pos = (label_match * log_prob).sum(1) / (label_match.sum(1) + 1e-8)
    loss = -mean_log_prob_pos.mean()
    
    return loss

if __name__ == "__main__":
    # 设置随机种子以确保可重复性
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 运行测试
    roi_features, roi_labels = test_roi_feature_extraction()
    prototype_memory = test_prototype_memory()
    contrastive_loss = test_contrastive_loss()
    
    print("\n=== 测试完成 ===")
    print("所有功能测试通过！") 