import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.modules.utils import _reverse_repeat_tuple, _pair
import torch.nn.functional as F

import math
from typing import Optional, List, Tuple, Union

__all__ = (
    "Conv_adalora_sym_m",
    "C2f_adalora_sym_m",
    "SPPF_adalora_sym_m",
    "ABlock_adalora_sym_m",
    "AAttn_adalora_sym_m",
    "A2C2f_adalora_sym_m",
    "C3k_adalora_sym_m",
    "CrossAAttn_adalora_sym_m",
    "CrossABlock_adalora_sym_m",
    "CrossA2C2f_adalora_sym_m",
    "HybridAtt_adalora_sym_m",
)

import csv
import numpy as np

# pearson = False
# if pearson:
#     f_cnt = open('count.csv', 'w', encoding='utf-8')
#     fcnt = csv.writer(f_cnt)
#     fcnt.writerow(['0'])
#     f_cnt.close()

# def calculate_pearson(x0, x1):
#     x0_ = x0 - np.mean(x0)
#     x1_ = x1 - np.mean(x1)
#     p = np.dot(x0_, x1_) / (np.linalg.norm(x0_) * np.linalg.norm(x1_))
#     return p

class Conv2d_adalora_sym_m(nn.Module):
    def __init__(self, c1, c2, k, r, lora_alpha, lora_dropout, \
                 stride, padding, groups, dilation, bias):  # merge_weights, 
        super().__init__()
        self.r = r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        # self.merge_weights = merge_weights

        self.conv = nn.Conv2d(c1, c2, k, stride, padding, dilation, groups, bias)
        if r > 0:
            self.lora_A_rgb = nn.Parameter(self.conv.weight.new_zeros((r, c1 * k)))
            self.lora_A_ir = nn.Parameter(self.conv.weight.new_zeros((r, c1 * k)))
            self.lora_E = nn.Parameter(self.conv.weight.new_zeros((r, 1)))
            self.lora_B_rgb = nn.Parameter(self.conv.weight.new_zeros((c2//self.conv.groups*k, r)))
            self.lora_B_ir = nn.Parameter(self.conv.weight.new_zeros((c2//self.conv.groups*k, r)))
            self.ranknum = nn.Parameter(self.conv.weight.new_zeros(1), requires_grad=False)
            self.ranknum.data.fill_(float(self.r))
            self.scaling = self.lora_alpha / self.r
            self.ranknum.requires_grad = False
        self.reset_parameters()

    def reset_parameters(self):
        self.conv.reset_parameters()
        if hasattr(self, 'lora_A_rgb'):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A_rgb, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.lora_A_ir, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.lora_B_rgb, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.lora_B_ir, a=math.sqrt(5))
            nn.init.zeros_(self.lora_E)

    def forward(self, x):
        x_rgb = self.conv._conv_forward(
                x[0],
                self.conv.weight + (self.lora_B_rgb @ (self.lora_A_rgb * self.lora_E)).view(self.conv.weight.shape) * self.scaling,
                self.conv.bias
            )
        x_ir = self.conv._conv_forward(
                x[1],
                self.conv.weight + (self.lora_B_ir @ (self.lora_A_ir * self.lora_E)).view(self.conv.weight.shape) * self.scaling,
                self.conv.bias
            )
        
        # x_rgb1 = self.conv._conv_forward(
        #         x[0],
        #         self.conv.weight,
        #         self.conv.bias
        #     )
        # x_rgb2 = self.conv._conv_forward(
        #         x[0],
        #         (self.lora_B_rgb @ (self.lora_A_rgb * self.lora_E)).view(self.conv.weight.shape) * self.scaling,
        #         self.conv.bias
        #     )
        # x_ir1 = self.conv._conv_forward(
        #         x[1],
        #         self.conv.weight,
        #         self.conv.bias
        #     )
        # x_ir2 = self.conv._conv_forward(
        #         x[1],
        #         (self.lora_B_ir @ (self.lora_A_ir * self.lora_E)).view(self.conv.weight.shape) * self.scaling,
        #         self.conv.bias
        #     )
        
        # if pearson and (x_rgb1.size()[0] != 1):
        #     f_cnt = open('count.csv', 'r', encoding='utf-8')
        #     fcnt = csv.reader(f_cnt)
        #     save_count = int(next(fcnt)[0])
        #     f_cnt.close()

        #     if save_count % 5 == 0:
        #         prefix = 'adalora_r9-6_sym_every'
        #         f_inter = open(f'{prefix}_inter.csv', 'a', encoding='utf-8')
        #         f_rgb_ir = open(f'{prefix}_rgb_ir.csv', 'a', encoding='utf-8')
        #         f_rgb_intra = open(f'{prefix}_rgb_intra.csv', 'a', encoding='utf-8')
        #         f_ir_intra = open(f'{prefix}_ir_intra.csv', 'a', encoding='utf-8')
        #         finter = csv.writer(f_inter)
        #         frgb_ir = csv.writer(f_rgb_ir)
        #         frgb_intra = csv.writer(f_rgb_intra)
        #         fir_intra = csv.writer(f_ir_intra)
                
                
        #         for i in range(x_rgb1.size()[0]):
        #             xrgb1 = x_rgb1[i].view(-1).cpu().numpy()
        #             xrgb2 = x_rgb2[i].view(-1).cpu().numpy()
        #             xir1 = x_ir1[i].view(-1).cpu().numpy()
        #             xir2 = x_ir2[i].view(-1).cpu().numpy()
        #             p_inter = calculate_pearson(xrgb1, xir1)
        #             p_rgb_ir = calculate_pearson(xrgb2, xir2)
        #             p_rgb_intra = calculate_pearson(xrgb1, xrgb2)
        #             p_ir_intra = calculate_pearson(xir1, xir2)
        #             finter.writerow([str(p_inter)])
        #             frgb_ir.writerow([str(p_rgb_ir)])
        #             frgb_intra.writerow([str(p_rgb_intra)])
        #             fir_intra.writerow([str(p_ir_intra)])
        #         f_inter.close()
        #         f_rgb_ir.close()
        #         f_rgb_intra.close()
        #         f_ir_intra.close()
        
        #     save_count += 1
        #     f_cnt = open('count.csv', 'w', encoding='utf-8')
        #     fcnt = csv.writer(f_cnt)
        #     fcnt.writerow([str(save_count)])
        #     f_cnt.close()
        # return (x_rgb1+x_rgb2, x_ir1+x_ir2)

        return (x_rgb, x_ir)

# 添加必要的常量
NEG_INF = -1e10
EPSILON = 1e-10

class FlashAttention(nn.Module):
    def __init__(self, q_block_size, kv_block_size, p_drop=0.0):
        super().__init__()
        self.q_block_size = q_block_size
        self.kv_block_size = kv_block_size
        self.p_drop = p_drop
        
    def forward(self, Q, K, V):
        # Q, K, V: [B, num_heads, Q_LEN, D] / [B, num_heads, K_LEN, D]
        B, H, Q_LEN, D = Q.shape
        _, _, K_LEN, _ = K.shape
        Q_BLOCK_SIZE = self.q_block_size
        KV_BLOCK_SIZE = self.kv_block_size
        Tr = Q_LEN // Q_BLOCK_SIZE
        Tc = K_LEN // KV_BLOCK_SIZE
        
        # 确保块大小能整除序列长度
        if Q_LEN % Q_BLOCK_SIZE != 0:
            pad_len = Q_BLOCK_SIZE - (Q_LEN % Q_BLOCK_SIZE)
            Q = torch.cat([Q, torch.zeros((B, H, pad_len, D), device=Q.device)], dim=2)
            Q_LEN = Q.shape[2]
            Tr = Q_LEN // Q_BLOCK_SIZE
            
        if K_LEN % KV_BLOCK_SIZE != 0:
            pad_len = KV_BLOCK_SIZE - (K_LEN % KV_BLOCK_SIZE)
            K = torch.cat([K, torch.zeros((B, H, pad_len, D), device=K.device)], dim=2)
            V = torch.cat([V, torch.zeros((B, H, pad_len, D), device=V.device)], dim=2)
            K_LEN = K.shape[2]
            Tc = K_LEN // KV_BLOCK_SIZE
        
        # 设备一致性
        device = Q.device
        
        # 初始化输出和中间状态变量
        O = torch.zeros_like(Q)
        l = torch.zeros(Q.shape[:-1], device=device)[..., None]
        m = torch.ones(Q.shape[:-1], device=device)[..., None] * NEG_INF

        Q_BLOCKS = torch.split(Q, Q_BLOCK_SIZE, dim=2)
        K_BLOCKS = torch.split(K, KV_BLOCK_SIZE, dim=2)
        V_BLOCKS = torch.split(V, KV_BLOCK_SIZE, dim=2)

        O_BLOCKS = list(torch.split(O, Q_BLOCK_SIZE, dim=2))
        l_BLOCKS = list(torch.split(l, Q_BLOCK_SIZE, dim=2))
        m_BLOCKS = list(torch.split(m, Q_BLOCK_SIZE, dim=2))

        for j in range(Tc):
            Kj = K_BLOCKS[j]
            Vj = V_BLOCKS[j]
            for i in range(Tr):
                Qi = Q_BLOCKS[i]  # [B, H, Q_BLOCK_SIZE, D]
                Oi = O_BLOCKS[i]  # [B, H, Q_BLOCK_SIZE, D]
                li = l_BLOCKS[i]  # [B, H, Q_BLOCK_SIZE, 1]
                mi = m_BLOCKS[i]  # [B, H, Q_BLOCK_SIZE, 1]
                
                # 使用batch矩阵乘法计算得分
                # [B, H, Q_BLOCK_SIZE, D] @ [B, H, D, KV_BLOCK_SIZE] -> [B, H, Q_BLOCK_SIZE, KV_BLOCK_SIZE]
                S_ij = torch.matmul(Qi, Kj.transpose(-1, -2)) / (D ** 0.5)
                
                # 找到每行最大值用于数值稳定性
                m_block_ij, _ = torch.max(S_ij, dim=-1, keepdim=True)  # [B, H, Q_BLOCK_SIZE, 1]
                
                # 计算注意力权重和归一化因子
                P_ij = torch.exp(S_ij - m_block_ij)  # [B, H, Q_BLOCK_SIZE, KV_BLOCK_SIZE]
                l_block_ij = torch.sum(P_ij, dim=-1, keepdim=True) + EPSILON  # [B, H, Q_BLOCK_SIZE, 1]
                
                # 更新最大值和归一化因子
                mi_new = torch.maximum(mi, m_block_ij)  # [B, H, Q_BLOCK_SIZE, 1]
                li_new = torch.exp(mi - mi_new) * li + torch.exp(m_block_ij - mi_new) * l_block_ij  # [B, H, Q_BLOCK_SIZE, 1]
                
                # 计算加权平均的值向量
                P_ij_Vj = torch.matmul(P_ij, Vj)  # [B, H, Q_BLOCK_SIZE, D]
                
                # 应用dropout
                if self.p_drop > 0:
                    P_ij_Vj = F.dropout(P_ij_Vj, p=self.p_drop, training=self.training)
                
                # 更新输出
                scale1 = (li / li_new) * torch.exp(mi - mi_new)
                scale2 = (torch.exp(m_block_ij - mi_new) / li_new)
                O_BLOCKS[i] = scale1 * Oi + scale2 * P_ij_Vj
                
                # 更新状态
                l_BLOCKS[i] = li_new
                m_BLOCKS[i] = mi_new
        
        # 合并输出块
        O = torch.cat(O_BLOCKS, dim=2)
        
        # 如果有padding，去掉padding部分
        if O.shape[2] > Q_LEN:
            O = O[:, :, :Q_LEN, :]
            
        return O

def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class Conv_adalora_sym_m(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, r=0, lora_alpha=1, lora_dropout=0., p=None, g=1, d=1, act=True):
        # merge_weights=True, 
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = Conv2d_adalora_sym_m(c1, c2, k, r, lora_alpha, lora_dropout, \
                           stride=s, padding=autopad(k, p, d), groups=g, dilation=d, bias=False)
        # merge_weights, 
        self.bn_rgb = nn.BatchNorm2d(c2)
        self.bn_ir = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        x = self.conv(x)
        x_rgb = self.act(self.bn_rgb(x[0]))
        x_ir = self.act(self.bn_ir(x[1]))
        return (x_rgb, x_ir)

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        x = self.conv(x)
        x_rgb = self.act(x[0])
        x_ir = self.act(x[1])
        return (x_rgb, x_ir)


class C2f_adalora_sym_m(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, r=0, lora_alpha=1, lora_dropout=0., g=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv_adalora_sym_m(c1, 2 * self.c, 1, 1, r, lora_alpha, lora_dropout)
        self.cv2 = Conv_adalora_sym_m((2 + n) * self.c, c2, 1, 1, r, lora_alpha, lora_dropout)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck_adalora_sym_m(self.c, self.c, shortcut, g, k=(3, 3), e=1.0, \
                                          r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)\
                                              for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        # y = list(self.cv1(x).chunk(2, 1))
        # y.extend(m(y[-1]) for m in self.m)
        # return self.cv2(torch.cat(y, 1))

        x = self.cv1(x)
        y_rgb = list(x[0].chunk(2, 1))
        y_ir = list(x[1].chunk(2, 1))
        for m in self.m:
            y1 = m((y_rgb[-1], y_ir[-1]))
            y_rgb.append(y1[0])
            y_ir.append(y1[1])
        y_rgb = torch.cat(y_rgb, 1)
        y_ir = torch.cat(y_ir, 1)
        return self.cv2((y_rgb, y_ir))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        # y = list(self.cv1(x).split((self.c, self.c), 1))
        # y.extend(m(y[-1]) for m in self.m)
        # return self.cv2(torch.cat(y, 1))

        x = self.cv1(x)
        y_rgb = list(x[0].split((self.c, self.c), 1))
        y_ir = list(x[1].split((self.c, self.c), 1))
        # y = list(self.cv1(x).split((self.c, self.c), 1))
        # y.extend(m(y[-1]) for m in self.m)
        for m in self.m:
            y1 = m((y_rgb[-1], y_ir[-1]))
            y_rgb.append(y1[0])
            y_ir.append(y1[1])
        y_rgb = torch.cat(y_rgb, 1)
        y_ir = torch.cat(y_ir, 1)
        return self.cv2((y_rgb, y_ir))

class Bottleneck_adalora_sym_m(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5, r=0, lora_alpha=1, lora_dropout=0.):
        """Initializes a bottleneck module with given input/output channels, shortcut option, group, kernels, and
        expansion.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv_adalora_sym_m(c1, c_, k[0], 1, r, lora_alpha, lora_dropout)
        self.cv2 = Conv_adalora_sym_m(c_, c2, k[1], 1, r, lora_alpha, lora_dropout, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """'forward()' applies the YOLO FPN to input data."""
        # return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))
        if self.add:
            x_1 = self.cv2(self.cv1(x))
            return (x[0] + x_1[0], x[1] + x_1[1])
        else:
            return self.cv2(self.cv1(x))


class SPPF_adalora_sym_m(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher."""

    def __init__(self, c1, c2, k=5, r=0, lora_alpha=1, lora_dropout=0.):
        """
        Initializes the SPPF layer with given input/output channels and kernel size.

        This module is equivalent to SPP(k=(5, 9, 13)).
        """
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv_adalora_sym_m(c1, c_, 1, 1, r, lora_alpha, lora_dropout)
        self.cv2 = Conv_adalora_sym_m(c_ * 4, c2, 1, 1, r, lora_alpha, lora_dropout)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        """Forward pass through Ghost Convolution block."""
        # x = self.cv1(x)
        # y1 = self.m(x)
        # y2 = self.m(y1)
        # return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))

        x = self.cv1(x)
        y1_rgb = self.m(x[0])
        y1_ir = self.m(x[1])
        y2_rgb = self.m(y1_rgb)
        y2_ir = self.m(y1_ir)
        return self.cv2((torch.cat((x[0], y1_rgb, y2_rgb, self.m(y2_rgb)), 1), \
                         torch.cat((x[1], y1_ir, y2_ir, self.m(y2_ir)), 1)))

class ABlock_adalora_sym_m(nn.Module):
    """
    多模态Adalora版Area-attention block。
    """
    def __init__(self, dim, num_heads, mlp_ratio=1.2, area=1, r=0, lora_alpha=1, lora_dropout=0.):
        super().__init__()
        # 注意：这里的AAttn和MLP都要用Adalora多模态Conv
        self.attn = AAttn_adalora_sym_m(dim, num_heads=num_heads, area=area, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            Conv_adalora_sym_m(dim, mlp_hidden_dim, 1, 1, r, lora_alpha, lora_dropout),
            Conv_adalora_sym_m(mlp_hidden_dim, dim, 1, 1, r, lora_alpha, lora_dropout, act=False)
        )

    def forward(self, x):
        x_rgb, x_ir = x
        attn_rgb, attn_ir = self.attn((x_rgb, x_ir))
        x_rgb = x_rgb + attn_rgb
        x_ir = x_ir + attn_ir
        # 释放注意力中间变量
        del attn_rgb, attn_ir
        
        mlp_rgb, mlp_ir = self.mlp((x_rgb, x_ir))
        result = (x_rgb + mlp_rgb, x_ir + mlp_ir)
        
        # 释放MLP中间变量
        del mlp_rgb, mlp_ir, x_rgb, x_ir
        
        return result

class AAttn_adalora_sym_m(nn.Module):
    """
    多模态Adalora版Area-attention，仅softmax注意力替换为FlashAttention。
    """
    def __init__(self, dim, num_heads, area=1, r=0, lora_alpha=1, lora_dropout=0., block_rows=4, block_cols=4):
        super().__init__()
        self.area = area
        self.num_heads = num_heads
        self.head_dim = head_dim = dim // num_heads
        all_head_dim = head_dim * num_heads
        self.qkv = Conv_adalora_sym_m(dim, all_head_dim * 3, 1, 1, r, lora_alpha, lora_dropout, act=False)
        self.proj = Conv_adalora_sym_m(all_head_dim, dim, 1, 1, r, lora_alpha, lora_dropout, act=False)
        self.pe = Conv_adalora_sym_m(all_head_dim, dim, 7, 1, r, lora_alpha, lora_dropout, p=3, g=dim, act=False)
        # self.flash_attn = FlashAttention(block_rows, block_cols,0.2)

    def forward(self, x):
        x_rgb, x_ir = x
        # print(f'x_rgb.shape:{x_rgb.shape}')
        B, C, H, W = x_rgb.shape
        N = H * W
        # qkv: [B, 3*all_head_dim, H, W] -> [B, 3, num_heads, head_dim, N]
        qkv = self.qkv((x_rgb, x_ir))
        qkv_rgb = qkv[0].flatten(2).transpose(1, 2)
        qkv_ir = qkv[1].flatten(2).transpose(1, 2)
        # print(f'qkv_rgb.shape:{qkv_rgb.shape}')
        # 释放qkv中间结果
        del qkv
        
        if self.area > 1:
            qkv_rgb = qkv_rgb.reshape(B * self.area, N // self.area, C * 3)
            qkv_ir = qkv_ir.reshape(B * self.area, N // self.area, C * 3)
            B, N, _ = qkv_rgb.shape
        
        q_rgb, k_rgb, v_rgb = qkv_rgb.view(B, N, self.num_heads, self.head_dim * 3).permute(0, 2, 3, 1).split([self.head_dim, self.head_dim, self.head_dim], dim=2)
        q_ir, k_ir, v_ir = qkv_ir.view(B, N, self.num_heads, self.head_dim * 3).permute(0, 2, 3, 1).split([self.head_dim, self.head_dim, self.head_dim], dim=2)
        # print(f'q_rgb.shape:{q_rgb.shape}') 
        # 释放qkv分解后的中间变量
        del qkv_rgb, qkv_ir
        # [B, num_heads, head_dim, N] -> [B, num_heads, N, head_dim]
        q_rgb = q_rgb.permute(0, 1, 3, 2).contiguous()
        k_rgb = k_rgb.permute(0, 1, 3, 2).contiguous()
        v_rgb = v_rgb.permute(0, 1, 3, 2).contiguous()
        q_ir = q_ir.permute(0, 1, 3, 2).contiguous()
        k_ir = k_ir.permute(0, 1, 3, 2).contiguous()
        v_ir = v_ir.permute(0, 1, 3, 2).contiguous()
        # 用FlashAttention替换softmax部分
        # x_rgb = self.flash_attn(q_rgb, k_rgb, v_rgb)
        # x_ir = self.flash_attn(q_ir, k_ir, v_ir)
        x_rgb = F.scaled_dot_product_attention(q_rgb, k_rgb, v_rgb, is_causal=False, dropout_p=0.2) 
        x_ir = F.scaled_dot_product_attention(q_ir, k_ir, v_ir, is_causal=False, dropout_p=0.2) 
        x_rgb = x_rgb.permute(0, 2, 1, 3).contiguous()
        x_ir = x_ir.permute(0, 2, 1, 3).contiguous()
        v_rgb = v_rgb.permute(0, 2, 1, 3).contiguous()
        v_ir = v_ir.permute(0, 2, 1, 3).contiguous()
        
        if self.area > 1:
            x_rgb = x_rgb.reshape(B // self.area, N * self.area, C)
            v_rgb = v_rgb.reshape(B // self.area, N * self.area, C)
            x_ir = x_ir.reshape(B // self.area, N * self.area, C)
            v_ir = v_ir.reshape(B // self.area, N * self.area, C)
            B, N, _ = x_rgb.shape
            
        x_rgb = x_rgb.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        v_rgb = v_rgb.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        x_ir = x_ir.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        v_ir = v_ir.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        
        pe_v = self.pe((v_rgb, v_ir))
        
        # 释放v变量
        del v_rgb, v_ir
        
        x_rgb = x_rgb + pe_v[0]
        x_ir = x_ir + pe_v[1]
        
        # 释放pe_v
        del pe_v
        
        proj_x = self.proj((x_rgb, x_ir))
        result = (proj_x[0], proj_x[1])
        
        # 释放投影中间变量
        del proj_x, x_rgb, x_ir, q_rgb, q_ir, k_rgb, k_ir
        
        return result

class A2C2f_adalora_sym_m(nn.Module):
    """
    多模态Adalora版Area-Attention C2f模块。
    """
    def __init__(self, c1, c2, n=1, a2=True, area=1, residual=False, mlp_ratio=2.0, r=0, lora_alpha=1, lora_dropout=0., g=1, e=0.5, shortcut=True):
        super().__init__()
        
        c_ = int(c2 * e)
        # c_ = 128
        # print(f'c_:{c_}')
        c_ = min(96, c_)
        assert c_ % 32 == 0, "Dimension of ABlock must be a multiple of 32."
        self.cv1 = Conv_adalora_sym_m(c1, c_, 1, 1, r, lora_alpha, lora_dropout)
        self.cv2 = Conv_adalora_sym_m((1 + n) * c_, c2, 1, 1, r, lora_alpha, lora_dropout)
        self.gamma = nn.Parameter(0.01 * torch.ones(c2), requires_grad=True) if a2 and residual else None
        self.m = nn.ModuleList(
            nn.Sequential(*(ABlock_adalora_sym_m(c_, c_ // 32, mlp_ratio, area, r, lora_alpha, lora_dropout) for _ in range(2))) if a2
            else C3k_adalora_sym_m(c_, c_, 2, shortcut, g, r, lora_alpha, lora_dropout)
            for _ in range(n)
        )

    def forward(self, x):
        y = self.cv1(x)
        y_rgb = [y[0]]
        y_ir = [y[1]]
        
        # 释放cv1输出变量
        del y
        
        for m in self.m:
            y1 = m((y_rgb[-1], y_ir[-1]))
            y_rgb.append(y1[0])
            y_ir.append(y1[1])
            
            # 释放每个模块的输出
            del y1
            
        y_rgb_cat = torch.cat(y_rgb, 1)
        y_ir_cat = torch.cat(y_ir, 1)
        
        # 释放列表
        del y_rgb, y_ir
        
        out = self.cv2((y_rgb_cat, y_ir_cat))
        
        # 释放拼接结果
        del y_rgb_cat, y_ir_cat
        
        if self.gamma is not None:
            result = (x[0] + self.gamma.view(-1, len(self.gamma), 1, 1) * out[0], 
                      x[1] + self.gamma.view(-1, len(self.gamma), 1, 1) * out[1])
            # 释放中间结果
            del out
            return result
        return out

class C3k_adalora_sym_m(nn.Module):
    """
    多模态Adalora版C3k（CSP bottleneck with kxk convs）。
    """
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, r=0, lora_alpha=1, lora_dropout=0., e=0.5, k=3):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv_adalora_sym_m(c1, c_, 1, 1, r, lora_alpha, lora_dropout)
        self.cv2 = Conv_adalora_sym_m(c1, c_, 1, 1, r, lora_alpha, lora_dropout)
        self.cv3 = Conv_adalora_sym_m(2 * c_, c2, 1, 1, r, lora_alpha, lora_dropout)
        # 主干bottleneck序列，全部用Bottleneck_adalora_sym_m，kernel大小为(k, k)
        self.m = nn.Sequential(*[Bottleneck_adalora_sym_m(c_, c_, shortcut, g, k=(k, k), e=1.0, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout) for _ in range(n)])

    def forward(self, x):
        # x: (rgb, ir)
        x1 = self.cv1(x)
        x2 = self.cv2(x)
        y1 = self.m(x1)
        out = self.cv3((torch.cat((y1[0], x2[0]), 1), torch.cat((y1[1], x2[1]), 1)))
        return out

class CrossAAttn_adalora_sym_m(nn.Module):
    """
    多模态Adalora版Cross-Area-attention，仅softmax注意力替换为FlashAttention。
    """
    def __init__(self, dim, num_heads, area=1, r=0, lora_alpha=1, lora_dropout=0., block_rows=4, block_cols=4):
        super().__init__()
        self.area = area
        self.num_heads = num_heads
        self.head_dim = head_dim = dim // num_heads
        all_head_dim = head_dim * num_heads
        self.q_proj = Conv_adalora_sym_m(dim, all_head_dim, 1, 1, r, lora_alpha, lora_dropout, act=False)
        self.kv_proj = Conv_adalora_sym_m(dim, all_head_dim * 2, 1, 1, r, lora_alpha, lora_dropout, act=False)
        self.proj = Conv_adalora_sym_m(all_head_dim, dim, 1, 1, r, lora_alpha, lora_dropout, act=False)
        self.pe = Conv_adalora_sym_m(all_head_dim, dim, 7, 1, r, lora_alpha, lora_dropout, p=3, g=dim, act=False)
        #self.flash_attn = FlashAttention(block_rows, block_cols,0.2)

    def forward(self, q_x, kv_x):
        # q_x, kv_x: (rgb, ir)
        q_rgb, q_ir = q_x
        kv_rgb, kv_ir = kv_x
        B, C, H, W = q_rgb.shape
        N = H * W
        # print(f'q_rgb.shape:{q_rgb.shape}')
        # q, k, v 投影
        q_pj = self.q_proj(q_x)
        k_pj = self.kv_proj(kv_x)
        q_rgb_flat = q_pj[0].flatten(2).transpose(1, 2)
        q_ir_flat = q_pj[1].flatten(2).transpose(1, 2)
        kv_rgb_flat = k_pj[0].flatten(2).transpose(1, 2)
        kv_ir_flat = k_pj[1].flatten(2).transpose(1, 2)
        
        # 释放投影结果
        del q_pj, k_pj
        
        if self.area > 1:
            q_rgb_flat = q_rgb_flat.reshape(B * self.area, N // self.area, C)
            q_ir_flat = q_ir_flat.reshape(B * self.area, N // self.area, C)
            kv_rgb_flat = kv_rgb_flat.reshape(B * self.area, N // self.area, C * 2)
            kv_ir_flat = kv_ir_flat.reshape(B * self.area, N // self.area, C * 2)
            B, N, _ = q_rgb_flat.shape
            
        k_rgb, v_rgb = kv_rgb_flat.view(B, N, self.num_heads, self.head_dim * 2).permute(0, 2, 3, 1).split([self.head_dim, self.head_dim], dim=2)
        k_ir, v_ir = kv_ir_flat.view(B, N, self.num_heads, self.head_dim * 2).permute(0, 2, 3, 1).split([self.head_dim, self.head_dim], dim=2)
        q_rgb = q_rgb_flat.view(B, N, self.num_heads, self.head_dim).permute(0, 2, 3, 1).contiguous()
        q_ir = q_ir_flat.view(B, N, self.num_heads, self.head_dim).permute(0, 2, 3, 1).contiguous()
        
        # 释放展平的中间变量
        del q_rgb_flat, q_ir_flat, kv_rgb_flat, kv_ir_flat
        # 只替换softmax部分
        # [B, num_heads, head_dim, N] -> [B, num_heads, N, head_dim]
        q_rgb = q_rgb.permute(0, 1, 3, 2).contiguous()
        k_rgb = k_rgb.permute(0, 1, 3, 2).contiguous()
        v_rgb = v_rgb.permute(0, 1, 3, 2).contiguous()
        q_ir = q_ir.permute(0, 1, 3, 2).contiguous()
        k_ir = k_ir.permute(0, 1, 3, 2).contiguous()
        v_ir = v_ir.permute(0, 1, 3, 2).contiguous()
        # 用FlashAttention替换softmax部分
        # x_rgb = self.flash_attn(q_rgb, k_rgb, v_rgb)
        # x_ir = self.flash_attn(q_ir, k_ir, v_ir)
        x_rgb = F.scaled_dot_product_attention(q_rgb, k_rgb, v_rgb, is_causal=False, dropout_p=0.2) 
        x_ir = F.scaled_dot_product_attention(q_ir, k_ir, v_ir, is_causal=False, dropout_p=0.2) 
        x_rgb = x_rgb.permute(0, 2, 1, 3).contiguous()
        x_ir = x_ir.permute(0, 2, 1, 3).contiguous()
        v_rgb = v_rgb.permute(0, 2, 1, 3).contiguous()
        v_ir = v_ir.permute(0, 2, 1, 3).contiguous()
        
        if self.area > 1:
            x_rgb = x_rgb.reshape(B // self.area, N * self.area, C)
            v_rgb = v_rgb.reshape(B // self.area, N * self.area, C)
            x_ir = x_ir.reshape(B // self.area, N * self.area, C)
            v_ir = v_ir.reshape(B // self.area, N * self.area, C)
            B, N, _ = x_rgb.shape
            
        x_rgb = x_rgb.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        v_rgb = v_rgb.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        x_ir = x_ir.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        v_ir = v_ir.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        
        pe_v = self.pe((v_rgb, v_ir))
        
        # 释放v中间变量
        del v_rgb, v_ir
        
        x_rgb = x_rgb + pe_v[0]
        x_ir = x_ir + pe_v[1]
        
        # 释放pe结果
        del pe_v
        
        proj_x = self.proj((x_rgb, x_ir))
        result = (proj_x[0], proj_x[1])
        
        # 释放最后的中间变量
        del proj_x, x_rgb, x_ir
        
        return result

class CrossABlock_adalora_sym_m(nn.Module):
    """
    多模态Adalora版Cross-Area-attention block。
    """
    def __init__(self, dim, num_heads, mlp_ratio=1.2, area=1, r=0, lora_alpha=1, lora_dropout=0.):
        super().__init__()
        self.attn = CrossAAttn_adalora_sym_m(dim, num_heads=num_heads, area=area, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            Conv_adalora_sym_m(dim, mlp_hidden_dim, 1, 1, r, lora_alpha, lora_dropout),
            Conv_adalora_sym_m(mlp_hidden_dim, dim, 1, 1, r, lora_alpha, lora_dropout, act=False)
        )

    def forward(self, q_x, kv_x):
        attn_rgb, attn_ir = self.attn(q_x, kv_x)
        q_rgb, q_ir = q_x
        x_rgb = q_rgb + attn_rgb
        x_ir = q_ir + attn_ir
        
        # 释放注意力中间结果
        del attn_rgb, attn_ir
        
        mlp_result = self.mlp((x_rgb, x_ir))
        mlp_rgb, mlp_ir = mlp_result
        result = (x_rgb + mlp_rgb, x_ir + mlp_ir)
        
        # 释放MLP中间结果
        del mlp_rgb, mlp_ir, x_rgb, x_ir
        
        return result

class CrossA2C2f_adalora_sym_m(nn.Module):
    """
    多模态Adalora版Cross-Area-Attention C2f模块。
    """
    def __init__(self, c1, c2, n=1, area=1, residual=False, mlp_ratio=2.0, r=0, lora_alpha=1, lora_dropout=0., g=1, e=0.5, shortcut=True):
        super().__init__()
        c_ = int(c2 * e)
        c_ = min(96, c_)
        assert c_ % 32 == 0, "Dimension of CrossABlock must be a multiple of 32."
        self.cv1_q = Conv_adalora_sym_m(c1, c_, 1, 1, r, lora_alpha, lora_dropout)
        self.cv1_kv = Conv_adalora_sym_m(c1*2, c_, 1, 1, r, lora_alpha, lora_dropout)
        self.cv2 = Conv_adalora_sym_m((1 + n) * c_, c2, 1, 1, r, lora_alpha, lora_dropout)
        self.gamma = nn.Parameter(0.01 * torch.ones(c2), requires_grad=True) if residual else None
        self.m = nn.ModuleList(
            CrossABlock_adalora_sym_m(c_, c_ // 32, mlp_ratio, area, r, lora_alpha, lora_dropout)
            for _ in range(n)
        )

    def forward(self, q_x, kv_x):
        q = self.cv1_q(q_x)
        kv = self.cv1_kv(kv_x)
        y_rgb = [q[0]]
        y_ir = [q[1]]
        
        for m in self.m:
            y1 = m((y_rgb[-1], y_ir[-1]), kv)
            y_rgb.append(y1[0])
            y_ir.append(y1[1])
            
            # 释放模块输出
            del y1
            
        y_rgb_cat = torch.cat(y_rgb, 1)
        y_ir_cat = torch.cat(y_ir, 1)
        
        # 释放列表
        del y_rgb, y_ir
        
        out = self.cv2((y_rgb_cat, y_ir_cat))
        
        # 释放拼接结果
        del y_rgb_cat, y_ir_cat, q, kv
        
        if self.gamma is not None:
            result = (q_x[0] + self.gamma.view(-1, len(self.gamma), 1, 1) * out[0], 
                     q_x[1] + self.gamma.view(-1, len(self.gamma), 1, 1) * out[1])
            # 释放输出
            del out
            return result
        return out

class HybridAtt_adalora_sym_m(nn.Module):
    """
    多模态Adalora版Hybrid-fusion模块。输入有包括3个元素的元组（每个元组里面有rgb和ir两部分），每一个都过一个TransformerBlock，之后将
    所有的作为kv,不同q和kv交互后，最终拼接过最终模块
    """
    def __init__(self, c_in_1=1, c_in_2=1, c_in_3=1, c_out=1, n=1, a2=True, area=1, residual=False, mlp_ratio=2.0, r=0, lora_alpha=1, lora_dropout=0., g=1, e=0.5, shortcut=True):
        super().__init__()
        #self.in_att_1 = A2C2f_adalora_sym_m(c1=c_in_1, c2=c_in_1, n=n, a2=a2, area=area*4, residual=residual, mlp_ratio=mlp_ratio, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, g=g, e=e, shortcut=shortcut)
        #self.in_att_2 = A2C2f_adalora_sym_m(c1=c_in_2, c2=c_in_2, n=n, a2=a2, area=area*2, residual=residual, mlp_ratio=mlp_ratio, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, g=g, e=e, shortcut=shortcut)
        self.in_att_3 = A2C2f_adalora_sym_m(c1=c_in_3, c2=c_in_3, n=n, a2=a2, area=area, residual=residual, mlp_ratio=mlp_ratio, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, g=g, e=e, shortcut=shortcut)
        self.down_att1_1 = Conv_adalora_sym_m(c1=c_in_1, c2=c_in_2, k=3, s=2, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, p=None, g=g, d=1, act=True)
        self.down_att1_2 = Conv_adalora_sym_m(c1=c_in_2, c2=c_in_3, k=3, s=2, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, p=None, g=g, d=1, act=True)
        self.down_att2 = Conv_adalora_sym_m(c1=c_in_2, c2=c_in_3, k=3, s=2, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, p=None, g=g, d=1, act=True)
        self.fuse_attn = CrossA2C2f_adalora_sym_m(c1=c_in_3, c2=c_out, n=n, area=area, residual=residual, mlp_ratio=mlp_ratio, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, g=g, e=e, shortcut=shortcut)
        self.fuse_conv = Conv_adalora_sym_m(c1=3*c_in_3, c2=c_out, k=3, s=1, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
        
    def forward(self, x):
        x_64 = x[0]
        x_32 = x[1]
        x_16 = x[2]
        # print(f'in------------x_64.shape:{x_64[0].shape}')
        # print(f'in------------x_32.shape:{x_32[0].shape}')
        # print(f'in------------x_16.shape:{x_16[0].shape}')
        # x_64 = self.in_att_1(x_64)
        # x_32 = self.in_att_2(x_32)
        x_16 = self.in_att_3(x_16)
        # print(f'out------------x_64.shape:{x_64[0].shape}')
        # print(f'out------------x_32.shape:{x_32[0].shape}')
        # print(f'out------------x_16.shape:{x_16[0].shape}')
        x_64 = self.down_att1_2(self.down_att1_1(x_64))
        x_32 = self.down_att2(x_32)
        kv_16 = (torch.cat((x_64[0], x_32[0]), dim=1), torch.cat((x_64[1], x_32[1]), dim=1))
        out_16 = self.fuse_attn(x_16, kv_16)
        kv_32 = (torch.cat((x_64[0], x_16[0]), dim=1), torch.cat((x_64[1], x_16[1]), dim=1))
        out_32 = self.fuse_attn(x_32, kv_32)
        kv_64 = (torch.cat((x_32[0], x_16[0]), dim=1), torch.cat((x_32[1], x_16[1]), dim=1))
        out_64 = self.fuse_attn(x_64, kv_64)
        out = (torch.cat((out_64[0], out_32[0],out_16[0]), dim=1), torch.cat((out_64[1], out_32[1],out_16[1]), dim=1))
        del x_64, x_32, x_16, out_64, out_32, out_16
        out = self.fuse_conv(out)
        return out 

