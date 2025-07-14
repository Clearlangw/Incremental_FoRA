# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
YOLO-specific modules

Usage:
    $ python path/to/models/yolo.py --cfg yolov5s.yaml
"""

import argparse
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
# ROOT = ROOT.relative_to(Path.cwd())  # relative

from utils.general import LOGGER, check_version

try:
    import thop  # for FLOPs computation
except ImportError:
    thop = None

def clone_tensor_structure(x):
    if torch.is_tensor(x):
        return x.clone()
    elif isinstance(x, list):
        return [clone_tensor_structure(i) for i in x]
    elif isinstance(x, tuple):
        return tuple(clone_tensor_structure(i) for i in x)
    elif isinstance(x, dict):
        return {k: clone_tensor_structure(v) for k, v in x.items()}
    else:
        return x  # 其他类型直接返回

class RedetectHeads(nn.Module):
    stride = None  # strides computed during build
    onnx_dynamic = False  # ONNX export parameter

    def __init__(self, nc=80, anchors=(), cfg=None, ch=(), inplace=True):  # detection layer
        super().__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5 + 180  # number of outputs per anchor 包含类别数nc+置信度1+xywh4，故nc+5 + 旋转分类
        self.base_nc = cfg['base_nc']
        self.base_no = self.base_nc + 5 + 180
        # self.nl = len(anchors)  # number of detection layers
        self.nl = len(anchors) # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        self.anchor_grid = [torch.zeros(1)] * self.nl  # init anchor grid
        self.register_buffer('anchors', torch.tensor(anchors).float().view(self.nl, -1, 2))  # shape(nl,na,2)
        self.base = nn.ModuleList(nn.Conv2d(x, self.base_no * self.na, 1) for x in ch[:3])  # output conv
        self.inplace = inplace  # use in-place ops (e.g. slice assignment)

        
        self.cfg = cfg
        self.return_x_novel = cfg.get('return_x_novel', False)
        self.cosine = cfg['cosine']
        self.scale = cfg['COSINE_SCALE']
        self.conf_thres = cfg['conf_thres']
        self.base_bonus = cfg['base_bonus']
        self.novel_bonus = cfg['novel_bonus']
        self.consistency_coeff = cfg['consistency_coeff']
        self.base_head = False
        if self.cosine:
            self.novel = nn.ModuleList(nn.Linear(
                x, self.no * self.na, bias=not self.cosine
            ) for x in ch[3:])  # output conv
        else:
            self.novel = nn.ModuleList(nn.Conv2d(
                x, self.no * self.na, 1) for x in ch[3:])  # output conv
        self.m = nn.ModuleList(self.base).extend(self.novel)

    def _nforward(self, x, cls_only=False):
        z = []  # inference output
        for i in range(self.nl):
            if self.cosine:
                x[i] = torch.tensor(x[i])       # x[i] (b, c, x, y)
                bs = x[i].shape[0]
                dim = x[i].shape[1]
                ny = x[i].shape[2]
                nx = x[i].shape[3]
                x[i] = x[i].permute(0, 2, 3, 1).contiguous().view(-1, dim)
                x[i] = self.novel[i](x[i])  # full connect / cosine
                x[i] = x[i].view(bs, self.no * self.na,ny, nx)
            else:
                x[i] = self.novel[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  # x[i](bs,self.no * self.na,20,20) to x[i](bs,self.na,20,20,self.no)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
            # follow https://openaccess.thecvf.com/content/CVPR2021/papers/Fan_Generalized_Few-Shot_Object_Detection_Without_Forgetting_CVPR_2021_paper.pdf
            if self.cosine:
                x[..., 5:5+self.nc] *= self.scale 

            if not self.training:  # inference
                if self.onnx_dynamic or self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)

                y = x[i].sigmoid() # (tensor): (b, self.na, h, w, self.no)

                # 增强对base model的依赖 base class conf += 0.1
                bonus = y.new_zeros(y.size())
                bonus[y > self.conf_thres / 2] = self.novel_bonus
                # TODO
                # ?仅对新类别还是所有类别
                bonus[..., :4] = 0
                bonus[..., 5:5+self.base_nc] = 0
                bonus[..., 5+self.nc:] = 0
                y += bonus

                if self.inplace:
                    y[..., 0:2] = (y[..., 0:2] * 2 - 0.5 + self.grid[i]) * self.stride[i]  # xy
                    y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                else:  # for YOLOv5 on AWS Inferentia https://github.com/ultralytics/yolov5/pull/2953
                    xy = (y[..., 0:2] * 2 - 0.5 + self.grid[i]) * self.stride[i]  # xy
                    wh = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                    y = torch.cat((xy, wh, y[..., 4:]), -1) 
                z.append(y.view(bs, -1, self.no)) # z (list[P3_pred]): Torch.Size(b, n_anchors, self.no)

        return x if self.training else (torch.cat(z, 1), x)

    def _bforward(self, x):
        """
        Args:
            x (list[P3_in,...]): torch.Size(b, c_i, h_i, w_i)

        Return：
            if train:
                x (list[P3_out,...]): torch.Size(b, self.na, h_i, w_i, self.no), self.na means the number of anchors scales
            else:
                inference (tensor): (b, n_all_anchors, self.no)
                x (list[P3_in,...]): torch.Size(b, c_i, h_i, w_i)
        """
        z = []  # inference output
        for i in range(self.nl):
            x[i] = self.base[i](x[i])  # conv 
            bs, _, ny, nx = x[i].shape  # x[i](bs,self.no * self.na,20,20) to x[i](bs,self.na,20,20,self.no)  self.no: (xywh, conf, nc, 180)
            x[i] = x[i].view(bs, self.na, self.base_no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
            # follow 3.3 inference part in https://openaccess.thecvf.com/content/CVPR2021/papers/Fan_Generalized_Few-Shot_Object_Detection_Without_Forgetting_CVPR_2021_paper.pdf
            #? TODO
            # 把base_nc扩展到nc?
            # import pdb; pdb.set_trace()
            shape = [x[i].shape[0], x[i].shape[1], x[i].shape[2], x[i].shape[3], self.no]
            temp_logits = torch.zeros(shape, device=x[i].device)
            temp_logits[..., 0:5+self.base_nc] = x[i][..., 0:5+self.base_nc]
            temp_logits[..., 5+self.nc:] = x[i][..., 5+self.base_nc:]
            x[i] = temp_logits


            if not self.training:  # inference
                if self.onnx_dynamic or self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)

                y = x[i].sigmoid() # (tensor): (b, self.na, h, w, self.no)
                # follow 3.3 inference part in https://openaccess.thecvf.com/content/CVPR2021/papers/Fan_Generalized_Few-Shot_Object_Detection_Without_Forgetting_CVPR_2021_paper.pdf
                # 增强对base model的依赖 base class conf += 0.1
                bonus = y.new_zeros(y.size())
                bonus[y > self.conf_thres] = self.base_bonus
                bonus[..., :4] = 0
                bonus[..., 5+self.base_nc:] = 0
                y += bonus

                if self.inplace:
                    y[..., 0:2] = (y[..., 0:2] * 2 - 0.5 + self.grid[i]) * self.stride[i]  # xy
                    y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                else:  # for YOLOv5 on AWS Inferentia https://github.com/ultralytics/yolov5/pull/2953
                    xy = (y[..., 0:2] * 2 - 0.5 + self.grid[i]) * self.stride[i]  # xy
                    wh = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                    y = torch.cat((xy, wh, y[..., 4:]), -1) 
                z.append(y.view(bs, -1, self.no)) # z (list[P3_pred]): Torch.Size(b, n_anchors, self.no)

        return x if self.training else (torch.cat(z, 1), x)

    def forward(self, x):
        # import pdb; pdb.set_trace()
        x_base = x[:3]
        x_novel = x[3:]
        init_x_novel = clone_tensor_structure(x_novel)
        # init_x_novel[0].shape is torch.Size([8, 512, 72, 84])
        # init_x_novel[1].shape is torch.Size([8, 512, 36, 42])
        # init_x_novel[2].shape is torch.Size([8, 512, 18, 21])
        # import sys
        # sys.exit()
        if self.training:
            # import pdb; pdb.set_trace()
            # x_clone = deepcopy(x)
            base_logits = self._bforward(x_base)
            novel_logits = self._nforward(x_novel)
            self._cls_consistency_loss = self.calculate_cls_consistency_loss(base_logits, novel_logits)
            # 计算_cls_consistency_loss
            if self.return_x_novel:
                return novel_logits, self._cls_consistency_loss, init_x_novel
            else:
                return novel_logits, self._cls_consistency_loss
        else:
        # 推理测试，使用base + novel
        # import pdb; pdb.set_trace()
        # x_clone = deepcopy(x)
            base_pred, base_logits = self._bforward(x_base)
            novel_pred, novel_logits = self._nforward(x_novel)
            final_pred = torch.cat([base_pred, novel_pred], dim=1)          # Torch.Size(b, n_anchors, self.no)
            final_logits = [torch.cat([novel_logits[i], base_logits[i]], dim=1) for i in range(self.nl)] # 感觉只能沿着na维度拼接(bs,self.na,20,20,self.no)   
            # return final_pred, final_logits
            # if self.base_head:
            if True:
                return final_pred, final_logits
            else:
                return novel_pred, novel_logits

    def calculate_cls_consistency_loss(self, base_logits, novel_logits):
        # TODO
        base_mask = self.base_no_mask()
        KLloss = nn.KLDivLoss(reduction="batchmean")
        bs = novel_logits[0].shape[0]
        loss = torch.zeros(1, device=novel_logits[0].device)
        #! TODO
        # 使用全部的anchor计算 相似度吗？ 
        for i in range(self.nl):
            base_logits_baseclass = base_logits[i][..., base_mask]
            novel_logits_baseclass = novel_logits[i][..., base_mask]
            # base_log_probs = nn.LogSoftmax(dim=-1)(base_logits_baseclass)     # 源码实现有问题，base和novel写反了 https://github.com/Megvii-BaseDetection/GFSD/blob/master/playground/fsdet/coco/retentive_rcnn/30shot/seed0/modeling/roi_heads.py
            # novel_probs = torch.softmax(novel_logits_baseclass, dim=-1)
            novel_log_probs = F.log_softmax(novel_logits_baseclass, dim=-1)
            base_probs = F.softmax(base_logits_baseclass, dim=-1)
            # 对anchors 进行归一化
            bs, na, ny, nx, _ = base_logits[i].shape           # bs, na, ny, nx, no
            distill_loss = KLloss(novel_log_probs, base_probs) / (ny * nx * na) 
            loss += distill_loss
            # assert not torch.any(torch.isnan(distill_loss)), f'Error: distill_loss_i: {distill_loss}, (ny * nx * na): {(ny * nx * na)}'
            # assert not torch.any(torch.isnan(loss)), f'Error: distill_loss: {loss}, distill_loss_i: {distill_loss}, (ny * nx * na): {(ny * nx * na)}'
        return loss * bs * self.consistency_coeff

    @property
    def cls_consistency_loss(self):
        return self._cls_consistency_loss

    def base_no_mask(self):
        mask = torch.zeros(self.no, dtype=torch.bool)
        # mask[:5+self.base_nc] = True
        # mask[5+self.nc:] = True
        mask[4: 5+self.base_nc] = True
        return mask

    def _make_grid(self, nx=20, ny=20, i=0):
        d = self.anchors[i].device
        if check_version(torch.__version__, '1.10.0'):  # torch>=1.10.0 meshgrid workaround for torch>=0.7 compatibility
            yv, xv = torch.meshgrid([torch.arange(ny, device=d), torch.arange(nx, device=d)], indexing='ij')
        else:
            yv, xv = torch.meshgrid([torch.arange(ny, device=d), torch.arange(nx, device=d)])
        grid = torch.stack((xv, yv), 2).expand((1, self.na, ny, nx, 2)).float()
        anchor_grid = (self.anchors[i].clone() * self.stride[i]) \
            .view((1, self.na, 1, 1, 2)).expand((1, self.na, ny, nx, 2)).float()
        return grid, anchor_grid
