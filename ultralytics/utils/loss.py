# Ultralytics YOLO 🚀, AGPL-3.0 license
import ultralytics.global_mode as gb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import roi_align

from ultralytics.utils.metrics import OKS_SIGMA
from ultralytics.utils.ops import crop_mask, xywh2xyxy, xyxy2xywh
from ultralytics.utils.tal import RotatedTaskAlignedAssigner, TaskAlignedAssigner, dist2bbox, dist2rbox, make_anchors
from .metrics import bbox_iou, probiou
from .tal import bbox2dist


class VarifocalLoss(nn.Module):
    """
    Varifocal loss by Zhang et al.

    https://arxiv.org/abs/2008.13367.
    """

    def __init__(self):
        """Initialize the VarifocalLoss class."""
        super().__init__()

    @staticmethod
    def forward(pred_score, gt_score, label, alpha=0.75, gamma=2.0):
        """Computes varfocal loss."""
        weight = alpha * pred_score.sigmoid().pow(gamma) * (1 - label) + gt_score * label
        with torch.cuda.amp.autocast(enabled=False):
            loss = (
                (F.binary_cross_entropy_with_logits(pred_score.float(), gt_score.float(), reduction="none") * weight)
                    .mean(1)
                    .sum()
            )
        return loss


class FocalLoss(nn.Module):
    """Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)."""

    def __init__(self):
        """Initializer for FocalLoss class with no parameters."""
        super().__init__()

    @staticmethod
    def forward(pred, label, gamma=1.5, alpha=0.25):
        """Calculates and updates confusion matrix for object detection/classification tasks."""
        loss = F.binary_cross_entropy_with_logits(pred, label, reduction="none")
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = pred.sigmoid()  # prob from logits
        p_t = label * pred_prob + (1 - label) * (1 - pred_prob)
        modulating_factor = (1.0 - p_t) ** gamma
        loss *= modulating_factor
        if alpha > 0:
            alpha_factor = label * alpha + (1 - label) * (1 - alpha)
            loss *= alpha_factor
        return loss.mean(1).sum()



class FocalLossforDetection(nn.Module):
    """
    Focal Loss, a wrapper around BCEWithLogitsLoss.
    This class is designed to be a drop-in replacement for nn.BCEWithLogitsLoss(reduction="none").
    """
    def __init__(self, gamma=1.5, alpha=0.25):
        """
        Initializer for the Focal Loss class.
        Args:
            gamma (float): The focusing parameter. Defaults to 1.5.
            alpha (float): The balancing parameter. Defaults to 0.25.
        """
        super(FocalLossforDetection, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, pred, label):
        """
        Calculates the focal loss for each element.
        Args:
            pred (torch.Tensor): The model's raw output (logits).
            label (torch.Tensor): The ground truth labels.
        Returns:
            torch.Tensor: A tensor of the same shape as input, containing the per-element focal loss.
        """
        # 1. 计算基础的 BCE Loss，不进行任何缩减
        bce_loss = F.binary_cross_entropy_with_logits(pred, label, reduction="none")

        # 2. 计算 p_t，即模型预测正确的概率
        pred_prob = torch.sigmoid(pred)  # 将 logits 转换为概率
        p_t = label * pred_prob + (1 - label) * (1 - pred_prob)

        # 3. 计算调制因子 (modulating factor)
        modulating_factor = (1.0 - p_t) ** self.gamma

        # 4. 计算 alpha 权重因子
        # alpha_factor = label * self.alpha + (1 - label) * (1 - self.alpha)
        # 写法优化：
        alpha_factor = torch.where(label.bool(), self.alpha, 1 - self.alpha)


        # 5. 计算最终的 Focal Loss
        focal_loss = alpha_factor * modulating_factor * bce_loss

        return focal_loss


class BboxLoss(nn.Module):
    """Criterion class for computing training losses during training."""

    def __init__(self, reg_max, use_dfl=False):
        """Initialize the BboxLoss module with regularization maximum and DFL settings."""
        super().__init__()
        self.reg_max = reg_max
        self.use_dfl = use_dfl

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask):
        """IoU loss."""
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

        # DFL loss
        if self.use_dfl:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.reg_max)
            loss_dfl = self._df_loss(pred_dist[fg_mask].view(-1, self.reg_max + 1), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0).to(pred_dist.device)

        return loss_iou, loss_dfl

    @staticmethod
    def _df_loss(pred_dist, target):
        """
        Return sum of left and right DFL losses.

        Distribution Focal Loss (DFL) proposed in Generalized Focal Loss
        https://ieeexplore.ieee.org/document/9792391
        """
        tl = target.long()  # target left
        tr = tl + 1  # target right
        wl = tr - target  # weight left
        wr = 1 - wl  # weight right
        return (
                F.cross_entropy(pred_dist, tl.view(-1), reduction="none").view(tl.shape) * wl
                + F.cross_entropy(pred_dist, tr.view(-1), reduction="none").view(tl.shape) * wr
        ).mean(-1, keepdim=True)


class RotatedBboxLoss(BboxLoss):
    """Criterion class for computing training losses during training."""

    def __init__(self, reg_max, use_dfl=False):
        """Initialize the BboxLoss module with regularization maximum and DFL settings."""
        super().__init__(reg_max, use_dfl)

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask):
        """IoU loss."""
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        iou = probiou(pred_bboxes[fg_mask], target_bboxes[fg_mask])
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

        # DFL loss
        if self.use_dfl:
            target_ltrb = bbox2dist(anchor_points, xywh2xyxy(target_bboxes[..., :4]), self.reg_max)
            loss_dfl = self._df_loss(pred_dist[fg_mask].view(-1, self.reg_max + 1), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0).to(pred_dist.device)

        return loss_iou, loss_dfl


class KeypointLoss(nn.Module):
    """Criterion class for computing training losses."""

    def __init__(self, sigmas) -> None:
        """Initialize the KeypointLoss class."""
        super().__init__()
        self.sigmas = sigmas

    def forward(self, pred_kpts, gt_kpts, kpt_mask, area):
        """Calculates keypoint loss factor and Euclidean distance loss for predicted and actual keypoints."""
        d = (pred_kpts[..., 0] - gt_kpts[..., 0]).pow(2) + (pred_kpts[..., 1] - gt_kpts[..., 1]).pow(2)
        kpt_loss_factor = kpt_mask.shape[1] / (torch.sum(kpt_mask != 0, dim=1) + 1e-9)
        # e = d / (2 * (area * self.sigmas) ** 2 + 1e-9)  # from formula
        e = d / (2 * self.sigmas).pow(2) / (area + 1e-9) / 2  # from cocoeval
        return (kpt_loss_factor.view(-1, 1) * ((1 - torch.exp(-e)) * kpt_mask)).mean()


class v8DetectionLoss:
    """Criterion class for computing training losses."""

    def __init__(self, model,is_incremental=False,is_contrastive_or_prototype=False,base_nc=0):  # model must be de-paralleled
        """Initializes v8DetectionLoss with the model, defining model-related properties and BCE loss function."""
        device = next(model.parameters()).device  # get model device
        h = model.args  # hyperparameters
        #self.modal = model.modal if hasattr(model, 'modal') else 'rgb' # TODO:这里改了
        self.modal = gb.read_global_mode()
        #print(self.modal)
        m = model.model[-1]  # Detect() module
        # m_list = [model.model[-3],model.model[-2],model.model[-1]]
        self.bce_mode = True
        if self.bce_mode:
            self.bce = nn.BCEWithLogitsLoss(reduction="none")
        else:
            self.bce = FocalLossforDetection(gamma=1.5, alpha=0.25)  # 使用FocalLoss替换BCE
        self.hyp = h
        self.stride = m.stride  # model strides
        self.nc = m.nc  # number of classes
        self.no = m.no
        self.reg_max = m.reg_max
        self.device = device
        self.use_gfsd = True
        self.use_dfl = m.reg_max > 1

        self.assigner = TaskAlignedAssigner(topk=10, num_classes=self.nc, alpha=0.5, beta=6.0)
        #print(f"self.assigner.nc = {self.nc}")
        self.bbox_loss = BboxLoss(m.reg_max - 1, use_dfl=self.use_dfl).to(device)
        self.proj = torch.arange(m.reg_max, dtype=torch.float, device=device)
        self.is_incremental = is_incremental
        self.is_contrastive_or_prototype = is_contrastive_or_prototype
        self.base_nc = base_nc
        

    def extract_roi_features_single(self, feat, targets, output_size=(3, 3)):
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
        # roi_align
        roi_feat = roi_align(feat, boxes, output_size=output_size, spatial_scale=1.0, aligned=True)  # [num_gt, c, out_h, out_w]
        roi_feat = roi_feat.view(num_gt, -1)  # flatten
        return roi_feat, clsids

    # def supervised_contrastive_loss(self, features, labels, temperature=0.07):
    #     """
    #     计算监督对比损失
    #     features: [num_gt, feat_dim]
    #     labels: [num_gt]
    #     """
    #     if features.shape[0] < 2:  # 至少需要2个样本才能计算对比损失
    #         return torch.tensor(0.0, device=features.device)
            
    #     device = features.device
    #     features = F.normalize(features, dim=1)  # 特征归一化
    #     similarity_matrix = torch.div(torch.matmul(features, features.T), temperature)  # [N, N]
    #     # 去除自身
    #     logits_mask = torch.ones_like(similarity_matrix) - torch.eye(features.shape[0], device=device)
    #     similarity_matrix = similarity_matrix * logits_mask

    #     # 数值稳定性处理：每行减去最大值
    #     logits_max, _ = torch.max(similarity_matrix, dim=1, keepdim=True)
    #     similarity_matrix = similarity_matrix - logits_max.detach()
    #     similarity_matrix = similarity_matrix * logits_mask  # 再次mask，防止数值误差

    #     # 构建正样本掩码
    #     labels = labels.contiguous().view(-1, 1)
    #     mask = torch.eq(labels, labels.T).float().to(device)
    #     mask = mask * logits_mask  # 去掉自身

    #     # log-softmax
    #     exp_sim = torch.exp(similarity_matrix) * logits_mask
    #     log_prob = similarity_matrix - torch.log(exp_sim.sum(1, keepdim=True) + 1e-8)

    #     # 只对正样本求平均
    #     mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-8)
    #     loss = -mean_log_prob_pos.mean()
    #     return loss



    def preprocess(self, targets, batch_size, scale_tensor):
        """Preprocesses the target counts and matches with the input batch size to output a tensor."""
        if targets.shape[0] == 0:
            out = torch.zeros(batch_size, 0, 5, device=self.device)
        else:
            i = targets[:, 0]  # image index
            _, counts = i.unique(return_counts=True)
            counts = counts.to(dtype=torch.int32)
            out = torch.zeros(batch_size, counts.max(), 5, device=self.device)
            for j in range(batch_size):
                matches = i == j
                n = matches.sum()
                if n:
                    out[j, :n] = targets[matches, 1:]
            out[..., 1:5] = xywh2xyxy(out[..., 1:5].mul_(scale_tensor))
        return out

    def normalize_bboxes(self, targets,batch_size):
        """
        将GT相对坐标归纳在一起，和preprocess不一样
        batch_size: 批量大小
        """
        if targets.shape[0] == 0:
            out = torch.zeros(batch_size, 0, 5, device=self.device)
        else:
            i = targets[:, 0]  # image index
            _, counts = i.unique(return_counts=True)
            counts = counts.to(dtype=torch.int32)
            out = torch.zeros(batch_size, counts.max(), 5, device=self.device)
            for j in range(batch_size):
                matches = i == j
                n = matches.sum()
                if n:
                    out[j, :n] = targets[matches, 1:]
            # out[..., 1:5] = xywh2xyxy(out[..., 1:5].mul_(scale_tensor))
        return out[...,1:5]
        

    def bbox_decode(self, anchor_points, pred_dist):
        """Decode predicted object bounding box coordinates from anchor points and distribution."""
        if self.use_dfl:
            b, a, c = pred_dist.shape  # batch, anchors, channels
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = pred_dist.view(b, a, c // 4, 4).transpose(2,3).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = (pred_dist.view(b, a, c // 4, 4).softmax(2) * self.proj.type(pred_dist.dtype).view(1, 1, -1, 1)).sum(2)
        return dist2bbox(pred_dist, anchor_points, xywh=False)

    def __call__(self, preds, batch):
        """Calculate the sum of the loss for box, cls and dfl multiplied by batch size."""
        self.modal = gb.read_global_mode()
        loss = torch.zeros(3, device=self.device)
        # if self.is_incremental and self.base_nc!=0 and self.use_gfsd:
        #     loss = torch.zeros(4, device=self.device)
        # else:
        #     loss = torch.zeros(3, device=self.device)  # box, cls, dfl
        batch_size = 0
        all_roi_features = []
        # all_roi_labels = []
        # print(len(preds))
        # print(len(preds[0]))
        # print(len(preds[1]))
        # print(len(preds[2]))
        # print(preds[0][0].shape)
        # print(preds[0][1].shape)
        # print(preds[0][2].shape)
        # print(self.modal)
        # print('|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||')
        # import sys
        # sys.exit()
        # print(self.hyp.model)
        if self.modal == 'both' and 'yolov8x.yaml' not in self.hyp.model:
            hyper_weight = 0.5
            for i in range(3): #遍历RGB，IR，Fusion三个模态
                # det_index = i + 1
                if i == 2:
                    hyper_weight = 1
                if self.is_incremental:
                    tmp_loss = torch.zeros(4, device=self.device)
                else:
                    tmp_loss = torch.zeros(3, device=self.device)
                # print(f'i is {i}')
                # print(f'preds is {preds}')
                if isinstance(preds[i], tuple):
                    feats = preds[i][1]  # 多尺度特征 [feat1, feat2, feat3]
                    if len(preds[i])>2:
                         base_nc_feats = preds[i][2]
                elif isinstance(preds[i], dict) and self.is_incremental:
                    feats = preds[i]['x']
                    base_nc_feats = preds[i]['base_x']
                else:
                    feats = preds[i]

                # print(feats[0].shape) #torch.Size([2, 77, 100, 100])
                if self.is_incremental:
                    # 处理feats，每个特征层跳过base_scores部分
                    tmp_pred_feat = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats],2)
                    pred_distri=tmp_pred_feat[:, :self.reg_max*4, :]
                    pred_scores=tmp_pred_feat[:, self.reg_max*4:, :]
                    base_scores=torch.cat([xi.view(base_nc_feats[0].shape[0], self.base_nc, -1) for xi in base_nc_feats],2)
                    base_scores=base_scores.permute(0, 2, 1).contiguous()

                else:
                    pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats],
                                                        2).split(
                        (self.reg_max * 4, self.nc), 1
                    )

                pred_scores = pred_scores.permute(0, 2, 1).contiguous()
                pred_distri = pred_distri.permute(0, 2, 1).contiguous()
                dtype = pred_scores.dtype
                batch_size = pred_scores.shape[0]
                imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[
                    0]  # image size (h,w)
                anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

                # Targets
                targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
                if self.is_contrastive_or_prototype:
                    norm_bboxes = self.normalize_bboxes(targets.to(self.device),batch_size)

                targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
                # 示例tensor([[  7.0000, 169.8720, 337.9572, 185.6079, 354.6466],
                #     [  7.0000, 658.1540, 337.9572, 673.8897, 354.6466]], device='cuda:0')
                gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
                mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)
                
                # Pboxes
                pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)

                tg_labels, target_bboxes, target_scores, fg_mask, _ = self.assigner(
                    pred_scores.detach().sigmoid(),
                    (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
                    anchor_points * stride_tensor,
                    gt_labels,
                    gt_bboxes,
                    mask_gt,
                )

                target_scores_sum = max(target_scores.sum(), 1)

                tmp_loss[1] += self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE
                # Bbox loss
                if fg_mask.sum():
                    target_bboxes /= stride_tensor
                    tmp_loss[0], tmp_loss[2] = self.bbox_loss(
                        pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum,
                        fg_mask
                    )
                    
                    # 提取多尺度ROI特征用于原型学习和对比学习
                    # 为每个模态的每个特征层提取ROI特征
                    if self.is_contrastive_or_prototype:
                        for layer_idx, feat in enumerate(feats):
                            # 构建targets用于ROI特征提取: [imgid, clsid, cx, cy, w, h] (归一化格式)
                            roi_targets_list = []
                            for b in range(batch_size):
                                # 获取当前batch的有效GT
                                valid_mask = mask_gt[b].squeeze(-1)  # (max_objects,)
                                valid_indices = torch.where(valid_mask)[0]  # 有效GT的索引
                                
                                if len(valid_indices) == 0:
                                    continue
                                    
                                # 获取有效GT的标签和边界框
                                tmp_labels = gt_labels[b, valid_indices, 0]  # (num_valid_gt,)
                                tmp_bboxes = norm_bboxes[b, valid_indices]     # (num_valid_gt, 4) (cxcywh格式)
                                
                                # 提取bbox的各个坐标分量
                                cx = tmp_bboxes[:, 0]  # 中心x坐标
                                cy = tmp_bboxes[:, 1]  # 中心y坐标
                                w = tmp_bboxes[:, 2]   # 宽度
                                h = tmp_bboxes[:, 3]   # 高度
                                
                                # 构建targets: [imgid, clsid, cx, cy, w, h]
                                batch_targets = torch.stack([
                                    torch.full_like(tmp_labels, b),  # imgid
                                    tmp_labels,  # clsid
                                    cx, cy, w, h  # bbox坐标分量
                                ], dim=1)
                                
                                roi_targets_list.append(batch_targets)
                                # import pdb
                                # pdb.set_trace()
                            if roi_targets_list:
                                roi_targets = torch.cat(roi_targets_list, dim=0)  # (total_valid_gt, 6)
                                roi_features, roi_labels = self.extract_roi_features_single(
                                    feat, roi_targets, output_size=(3, 3)
                                )
                                
                                if roi_features.shape[0] > 0:
                                    # 存储特征信息：(模态索引, 层索引, 特征, 标签)
                                    all_roi_features.append({
                                        'modal_idx': i,
                                        'layer_idx': layer_idx,
                                        'features': roi_features,
                                        'labels': roi_labels
                                    })
                                    # all_roi_labels.extend(roi_labels.cpu().numpy())
                                    #有多个gt所以len(all_roi_labels)和len(all_roi_features)不一样

                if self.is_incremental and self.base_nc!=0 and self.use_gfsd:
                    # KL损失：保证pred_scores和base_scores的前base_nc个类别的预测值尽可能相似
                    # 对base_scores和pred_scores的前base_nc个类别进行sigmoid
                    base_fg_mask = (tg_labels<self.base_nc).bool()&fg_mask.bool()
                    base_fg_mask_dim1 = base_fg_mask.reshape(-1)
                    # 获取所有前景目标（positive anchors）的数量
                    num_fg = base_fg_mask_dim1.sum()
                    # 只有当存在前景目标时，才计算KL损失,另外KL散度综合一定得为1，所以只能softmax
                    if num_fg > 0:
                        # 2. 从展平的 scores 中，根据行掩码 `base_fg_mask_dim1` 取出前景样本的 logits
                        #    得到的张量形状为 (num_fg, total_num_classes)
                        pred_logits_fg = pred_scores.reshape(-1, pred_scores.shape[-1])[base_fg_mask_dim1]
                        base_logits_fg = base_scores.reshape(-1, base_scores.shape[-1])[base_fg_mask_dim1]

                        # 3. 【列筛选】使用切片操作，直接从类别维度取出前 self.base_nc 个基类
                        #    形状从 (num_fg, total_num_classes) -> (num_fg, self.base_nc)
                        pred_logits_base = pred_logits_fg[..., :self.base_nc]
                        base_logits_base = base_logits_fg[..., :self.base_nc]

                        # 4. 【计算分布】在筛选出的基类集合上计算概率分布
                        #    使用 log_softmax 以获得更好的数值稳定性
                        log_pred_dist = F.log_softmax(pred_logits_base, dim=-1)
                        base_dist = F.softmax(base_logits_base, dim=-1)

                        # 5. 【计算损失】计算KL散度，'mean' reduction 会对所有元素取平均
                        kl_loss = F.kl_div(
                            log_pred_dist,
                            base_dist,
                            reduction='mean'
                        )
                    else:
                        # 如果没有前景目标，则损失为0
                        kl_loss = torch.tensor(0.0, device=pred_scores.device)
                    
                    
                tmp_loss[0] *= (self.hyp.box * hyper_weight)  # box gain
                if self.bce_mode:
                    tmp_loss[1] *= (self.hyp.cls * hyper_weight)  # cls gain
                else:
                    tmp_loss[1] *= (self.hyp.cls * hyper_weight * 10)  # cls gain
                tmp_loss[2] *= (self.hyp.dfl * hyper_weight)  # dfl gain
                if self.is_incremental and self.base_nc!=0 and self.use_gfsd:
                    tmp_loss[2] += kl_loss*0.5

                loss[0] += tmp_loss[0]
                loss[1] += tmp_loss[1]
                loss[2] += tmp_loss[2]

            # 如果有ROI特征，则作为额外返回值输出
            if self.is_contrastive_or_prototype and all_roi_features:
                return loss.sum() * batch_size, loss.detach(), all_roi_features
            else:
                return loss.sum() * batch_size, loss.detach()
        else:
            if isinstance(preds[i], tuple):
                feats = preds[i][1]
            elif isinstance(preds[i], dict) and self.is_incremental:
                feats = preds[i]['x']
                base_nc_feats = preds[i]['base_x']
            else:
                feats = preds[i]

            pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
                (self.reg_max * 4, self.nc), 1
            )

            pred_scores = pred_scores.permute(0, 2, 1).contiguous()
            pred_distri = pred_distri.permute(0, 2, 1).contiguous()

            dtype = pred_scores.dtype
            batch_size = pred_scores.shape[0]
            imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[
                0]  # image size (h,w)
            anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

            # Targets
            targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
            targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
            gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
            mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)

            # Pboxes
            pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)

            _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
                pred_scores.detach().sigmoid(),
                (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
                anchor_points * stride_tensor,
                gt_labels,
                gt_bboxes,
                mask_gt,
            )

            target_scores_sum = max(target_scores.sum(), 1)

            # Cls loss
            # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
            loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

            # Bbox loss
            if fg_mask.sum():
                target_bboxes /= stride_tensor
                loss[0], loss[2] = self.bbox_loss(
                    pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask
                )

            loss[0] *= self.hyp.box  # box gain
            if self.bce_mode:
                loss[1] *= self.hyp.cls  # cls gain
            else:
                loss[1] *= (self.hyp.cls * 10)  # cls gain
            loss[2] *= self.hyp.dfl  # dfl gain

            return loss.sum() * batch_size, loss.detach()  # loss(box, cls, dfl)


class v8SegmentationLoss(v8DetectionLoss):
    """Criterion class for computing training losses."""

    def __init__(self, model):  # model must be de-paralleled
        """Initializes the v8SegmentationLoss class, taking a de-paralleled model as argument."""
        super().__init__(model)
        self.overlap = model.args.overlap_mask

    def __call__(self, preds, batch):
        """Calculate and return the loss for the YOLO model."""
        loss = torch.zeros(4, device=self.device)  # box, cls, dfl
        feats, pred_masks, proto = preds if len(preds) == 3 else preds[1]
        batch_size, _, mask_h, mask_w = proto.shape  # batch size, number of masks, mask height, mask width
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1
        )

        # B, grids, ..
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        pred_masks = pred_masks.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # Targets
        try:
            batch_idx = batch["batch_idx"].view(-1, 1)
            targets = torch.cat((batch_idx, batch["cls"].view(-1, 1), batch["bboxes"]), 1)
            targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
            gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
            mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)
        except RuntimeError as e:
            raise TypeError(
                "ERROR ❌ segment dataset incorrectly formatted or not a segment dataset.\n"
                "This error can occur when incorrectly training a 'segment' model on a 'detect' dataset, "
                "i.e. 'yolo train model=yolov8n-seg.pt data=coco8.yaml'.\nVerify your dataset is a "
                "correctly formatted 'segment' dataset using 'data=coco8-seg.yaml' "
                "as an example.\nSee https://docs.ultralytics.com/datasets/segment/ for help."
            ) from e

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)

        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        # Cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[2] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        if fg_mask.sum():
            # Bbox loss
            loss[0], loss[3] = self.bbox_loss(
                pred_distri,
                pred_bboxes,
                anchor_points,
                target_bboxes / stride_tensor,
                target_scores,
                target_scores_sum,
                fg_mask,
            )
            # Masks loss
            masks = batch["masks"].to(self.device).float()
            if tuple(masks.shape[-2:]) != (mask_h, mask_w):  # downsample
                masks = F.interpolate(masks[None], (mask_h, mask_w), mode="nearest")[0]

            loss[1] = self.calculate_segmentation_loss(
                fg_mask, masks, target_gt_idx, target_bboxes, batch_idx, proto, pred_masks, imgsz, self.overlap
            )

        # WARNING: lines below prevent Multi-GPU DDP 'unused gradient' PyTorch errors, do not remove
        else:
            loss[1] += (proto * 0).sum() + (pred_masks * 0).sum()  # inf sums may lead to nan loss

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.box  # seg gain
        loss[2] *= self.hyp.cls  # cls gain
        loss[3] *= self.hyp.dfl  # dfl gain

        return loss.sum() * batch_size, loss.detach()  # loss(box, cls, dfl)

    @staticmethod
    def single_mask_loss(
            gt_mask: torch.Tensor, pred: torch.Tensor, proto: torch.Tensor, xyxy: torch.Tensor, area: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the instance segmentation loss for a single image.

        Args:
            gt_mask (torch.Tensor): Ground truth mask of shape (n, H, W), where n is the number of objects.
            pred (torch.Tensor): Predicted mask coefficients of shape (n, 32).
            proto (torch.Tensor): Prototype masks of shape (32, H, W).
            xyxy (torch.Tensor): Ground truth bounding boxes in xyxy format, normalized to [0, 1], of shape (n, 4).
            area (torch.Tensor): Area of each ground truth bounding box of shape (n,).

        Returns:
            (torch.Tensor): The calculated mask loss for a single image.

        Notes:
            The function uses the equation pred_mask = torch.einsum('in,nhw->ihw', pred, proto) to produce the
            predicted masks from the prototype masks and predicted mask coefficients.
        """
        pred_mask = torch.einsum("in,nhw->ihw", pred, proto)  # (n, 32) @ (32, 80, 80) -> (n, 80, 80)
        loss = F.binary_cross_entropy_with_logits(pred_mask, gt_mask, reduction="none")
        return (crop_mask(loss, xyxy).mean(dim=(1, 2)) / area).sum()

    def calculate_segmentation_loss(
            self,
            fg_mask: torch.Tensor,
            masks: torch.Tensor,
            target_gt_idx: torch.Tensor,
            target_bboxes: torch.Tensor,
            batch_idx: torch.Tensor,
            proto: torch.Tensor,
            pred_masks: torch.Tensor,
            imgsz: torch.Tensor,
            overlap: bool,
    ) -> torch.Tensor:
        """
        Calculate the loss for instance segmentation.

        Args:
            fg_mask (torch.Tensor): A binary tensor of shape (BS, N_anchors) indicating which anchors are positive.
            masks (torch.Tensor): Ground truth masks of shape (BS, H, W) if `overlap` is False, otherwise (BS, ?, H, W).
            target_gt_idx (torch.Tensor): Indexes of ground truth objects for each anchor of shape (BS, N_anchors).
            target_bboxes (torch.Tensor): Ground truth bounding boxes for each anchor of shape (BS, N_anchors, 4).
            batch_idx (torch.Tensor): Batch indices of shape (N_labels_in_batch, 1).
            proto (torch.Tensor): Prototype masks of shape (BS, 32, H, W).
            pred_masks (torch.Tensor): Predicted masks for each anchor of shape (BS, N_anchors, 32).
            imgsz (torch.Tensor): Size of the input image as a tensor of shape (2), i.e., (H, W).
            overlap (bool): Whether the masks in `masks` tensor overlap.

        Returns:
            (torch.Tensor): The calculated loss for instance segmentation.

        Notes:
            The batch loss can be computed for improved speed at higher memory usage.
            For example, pred_mask can be computed as follows:
                pred_mask = torch.einsum('in,nhw->ihw', pred, proto)  # (i, 32) @ (32, 160, 160) -> (i, 160, 160)
        """
        _, _, mask_h, mask_w = proto.shape
        loss = 0

        # Normalize to 0-1
        target_bboxes_normalized = target_bboxes / imgsz[[1, 0, 1, 0]]

        # Areas of target bboxes
        marea = xyxy2xywh(target_bboxes_normalized)[..., 2:].prod(2)

        # Normalize to mask size
        mxyxy = target_bboxes_normalized * torch.tensor([mask_w, mask_h, mask_w, mask_h], device=proto.device)

        for i, single_i in enumerate(zip(fg_mask, target_gt_idx, pred_masks, proto, mxyxy, marea, masks)):
            fg_mask_i, target_gt_idx_i, pred_masks_i, proto_i, mxyxy_i, marea_i, masks_i = single_i
            if fg_mask_i.any():
                mask_idx = target_gt_idx_i[fg_mask_i]
                if overlap:
                    gt_mask = masks_i == (mask_idx + 1).view(-1, 1, 1)
                    gt_mask = gt_mask.float()
                else:
                    gt_mask = masks[batch_idx.view(-1) == i][mask_idx]

                loss += self.single_mask_loss(
                    gt_mask, pred_masks_i[fg_mask_i], proto_i, mxyxy_i[fg_mask_i], marea_i[fg_mask_i]
                )

            # WARNING: lines below prevents Multi-GPU DDP 'unused gradient' PyTorch errors, do not remove
            else:
                loss += (proto * 0).sum() + (pred_masks * 0).sum()  # inf sums may lead to nan loss

        return loss / fg_mask.sum()


class v8PoseLoss(v8DetectionLoss):
    """Criterion class for computing training losses."""

    def __init__(self, model):  # model must be de-paralleled
        """Initializes v8PoseLoss with model, sets keypoint variables and declares a keypoint loss instance."""
        super().__init__(model)
        self.kpt_shape = model.model[-1].kpt_shape
        self.bce_pose = nn.BCEWithLogitsLoss()
        is_pose = self.kpt_shape == [17, 3]
        nkpt = self.kpt_shape[0]  # number of keypoints
        sigmas = torch.from_numpy(OKS_SIGMA).to(self.device) if is_pose else torch.ones(nkpt, device=self.device) / nkpt
        self.keypoint_loss = KeypointLoss(sigmas=sigmas)

    def __call__(self, preds, batch):
        """Calculate the total loss and detach it."""
        loss = torch.zeros(5, device=self.device)  # box, cls, dfl, kpt_location, kpt_visibility
        feats, pred_kpts = preds if isinstance(preds[0], list) else preds[1]
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1
        )

        # B, grids, ..
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        pred_kpts = pred_kpts.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # Targets
        batch_size = pred_scores.shape[0]
        batch_idx = batch["batch_idx"].view(-1, 1)
        targets = torch.cat((batch_idx, batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)
        pred_kpts = self.kpts_decode(anchor_points, pred_kpts.view(batch_size, -1, *self.kpt_shape))  # (b, h*w, 17, 3)

        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        # Cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[3] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        # Bbox loss
        if fg_mask.sum():
            target_bboxes /= stride_tensor
            loss[0], loss[4] = self.bbox_loss(
                pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask
            )
            keypoints = batch["keypoints"].to(self.device).float().clone()
            keypoints[..., 0] *= imgsz[1]
            keypoints[..., 1] *= imgsz[0]

            loss[1], loss[2] = self.calculate_keypoints_loss(
                fg_mask, target_gt_idx, keypoints, batch_idx, stride_tensor, target_bboxes, pred_kpts
            )

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.pose  # pose gain
        loss[2] *= self.hyp.kobj  # kobj gain
        loss[3] *= self.hyp.cls  # cls gain
        loss[4] *= self.hyp.dfl  # dfl gain

        return loss.sum() * batch_size, loss.detach()  # loss(box, cls, dfl)

    @staticmethod
    def kpts_decode(anchor_points, pred_kpts):
        """Decodes predicted keypoints to image coordinates."""
        y = pred_kpts.clone()
        y[..., :2] *= 2.0
        y[..., 0] += anchor_points[:, [0]] - 0.5
        y[..., 1] += anchor_points[:, [1]] - 0.5
        return y

    def calculate_keypoints_loss(
            self, masks, target_gt_idx, keypoints, batch_idx, stride_tensor, target_bboxes, pred_kpts
    ):
        """
        Calculate the keypoints loss for the model.

        This function calculates the keypoints loss and keypoints object loss for a given batch. The keypoints loss is
        based on the difference between the predicted keypoints and ground truth keypoints. The keypoints object loss is
        a binary classification loss that classifies whether a keypoint is present or not.

        Args:
            masks (torch.Tensor): Binary mask tensor indicating object presence, shape (BS, N_anchors).
            target_gt_idx (torch.Tensor): Index tensor mapping anchors to ground truth objects, shape (BS, N_anchors).
            keypoints (torch.Tensor): Ground truth keypoints, shape (N_kpts_in_batch, N_kpts_per_object, kpts_dim).
            batch_idx (torch.Tensor): Batch index tensor for keypoints, shape (N_kpts_in_batch, 1).
            stride_tensor (torch.Tensor): Stride tensor for anchors, shape (N_anchors, 1).
            target_bboxes (torch.Tensor): Ground truth boxes in (x1, y1, x2, y2) format, shape (BS, N_anchors, 4).
            pred_kpts (torch.Tensor): Predicted keypoints, shape (BS, N_anchors, N_kpts_per_object, kpts_dim).

        Returns:
            (tuple): Returns a tuple containing:
                - kpts_loss (torch.Tensor): The keypoints loss.
                - kpts_obj_loss (torch.Tensor): The keypoints object loss.
        """
        batch_idx = batch_idx.flatten()
        batch_size = len(masks)

        # Find the maximum number of keypoints in a single image
        max_kpts = torch.unique(batch_idx, return_counts=True)[1].max()

        # Create a tensor to hold batched keypoints
        batched_keypoints = torch.zeros(
            (batch_size, max_kpts, keypoints.shape[1], keypoints.shape[2]), device=keypoints.device
        )

        # TODO: any idea how to vectorize this?
        # Fill batched_keypoints with keypoints based on batch_idx
        for i in range(batch_size):
            keypoints_i = keypoints[batch_idx == i]
            batched_keypoints[i, : keypoints_i.shape[0]] = keypoints_i

        # Expand dimensions of target_gt_idx to match the shape of batched_keypoints
        target_gt_idx_expanded = target_gt_idx.unsqueeze(-1).unsqueeze(-1)

        # Use target_gt_idx_expanded to select keypoints from batched_keypoints
        selected_keypoints = batched_keypoints.gather(
            1, target_gt_idx_expanded.expand(-1, -1, keypoints.shape[1], keypoints.shape[2])
        )

        # Divide coordinates by stride
        selected_keypoints /= stride_tensor.view(1, -1, 1, 1)

        kpts_loss = 0
        kpts_obj_loss = 0

        if masks.any():
            gt_kpt = selected_keypoints[masks]
            area = xyxy2xywh(target_bboxes[masks])[:, 2:].prod(1, keepdim=True)
            pred_kpt = pred_kpts[masks]
            kpt_mask = gt_kpt[..., 2] != 0 if gt_kpt.shape[-1] == 3 else torch.full_like(gt_kpt[..., 0], True)
            kpts_loss = self.keypoint_loss(pred_kpt, gt_kpt, kpt_mask, area)  # pose loss

            if pred_kpt.shape[-1] == 3:
                kpts_obj_loss = self.bce_pose(pred_kpt[..., 2], kpt_mask.float())  # keypoint obj loss

        return kpts_loss, kpts_obj_loss


class v8ClassificationLoss:
    """Criterion class for computing training losses."""

    def __call__(self, preds, batch):
        """Compute the classification loss between predictions and true labels."""
        loss = torch.nn.functional.cross_entropy(preds, batch["cls"], reduction="mean")
        loss_items = loss.detach()
        return loss, loss_items


class v8OBBLoss(v8DetectionLoss):
    def __init__(self, model):
        """
        Initializes v8OBBLoss with model, assigner, and rotated bbox loss.

        Note model must be de-paralleled.
        """
        super().__init__(model)
        self.assigner = RotatedTaskAlignedAssigner(topk=10, num_classes=self.nc, alpha=0.5, beta=6.0)
        self.bbox_loss = RotatedBboxLoss(self.reg_max - 1, use_dfl=self.use_dfl).to(self.device)

    def preprocess(self, targets, batch_size, scale_tensor):
        """Preprocesses the target counts and matches with the input batch size to output a tensor."""
        if targets.shape[0] == 0:
            out = torch.zeros(batch_size, 0, 6, device=self.device)
        else:
            i = targets[:, 0]  # image index
            _, counts = i.unique(return_counts=True)
            counts = counts.to(dtype=torch.int32)
            out = torch.zeros(batch_size, counts.max(), 6, device=self.device)
            for j in range(batch_size):
                matches = i == j
                n = matches.sum()
                if n:
                    bboxes = targets[matches, 2:]
                    bboxes[..., :4].mul_(scale_tensor)
                    out[j, :n] = torch.cat([targets[matches, 1:2], bboxes], dim=-1)
        return out

    def __call__(self, preds, batch):
        """Calculate and return the loss for the YOLO model."""
        loss = torch.zeros(3, device=self.device)  # box, cls, dfl
        feats, pred_angle = preds if isinstance(preds[0], list) else preds[1]
        batch_size = pred_angle.shape[0]  # batch size, number of masks, mask height, mask width
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1
        )

        # b, grids, ..
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        pred_angle = pred_angle.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # targets
        try:
            batch_idx = batch["batch_idx"].view(-1, 1)
            targets = torch.cat((batch_idx, batch["cls"].view(-1, 1), batch["bboxes"].view(-1, 5)), 1)
            rw, rh = targets[:, 4] * imgsz[0].item(), targets[:, 5] * imgsz[1].item()
            targets = targets[(rw >= 2) & (rh >= 2)]  # filter rboxes of tiny size to stabilize training
            targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
            gt_labels, gt_bboxes = targets.split((1, 5), 2)  # cls, xywhr
            mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)
        except RuntimeError as e:
            raise TypeError(
                "ERROR ❌ OBB dataset incorrectly formatted or not a OBB dataset.\n"
                "This error can occur when incorrectly training a 'OBB' model on a 'detect' dataset, "
                "i.e. 'yolo train model=yolov8n-obb.pt data=dota8.yaml'.\nVerify your dataset is a "
                "correctly formatted 'OBB' dataset using 'data=dota8.yaml' "
                "as an example.\nSee https://docs.ultralytics.com/datasets/obb/ for help."
            ) from e

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri, pred_angle)  # xyxy, (b, h*w, 4)

        bboxes_for_assigner = pred_bboxes.clone().detach()
        # Only the first four elements need to be scaled
        bboxes_for_assigner[..., :4] *= stride_tensor
        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(),
            bboxes_for_assigner.type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        # Cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        # Bbox loss
        if fg_mask.sum():
            target_bboxes[..., :4] /= stride_tensor
            loss[0], loss[2] = self.bbox_loss(
                pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask
            )
        else:
            loss[0] += (pred_angle * 0).sum()

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.cls  # cls gain
        loss[2] *= self.hyp.dfl  # dfl gain

        return loss.sum() * batch_size, loss.detach()  # loss(box, cls, dfl)

    def bbox_decode(self, anchor_points, pred_dist, pred_angle):
        """
        Decode predicted object bounding box coordinates from anchor points and distribution.

        Args:
            anchor_points (torch.Tensor): Anchor points, (h*w, 2).
            pred_dist (torch.Tensor): Predicted rotated distance, (bs, h*w, 4).
            pred_angle (torch.Tensor): Predicted angle, (bs, h*w, 1).

        Returns:
            (torch.Tensor): Predicted rotated bounding boxes with angles, (bs, h*w, 5).
        """
        if self.use_dfl:
            b, a, c = pred_dist.shape  # batch, anchors, channels
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
        return torch.cat((dist2rbox(pred_dist, pred_angle, anchor_points), pred_angle), dim=-1)
