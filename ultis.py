import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2


class ObjectDetectorLoss(nn.Module):
    def __init__(self, weight_box=5, weight_cls=0.5):
        super(ObjectDetectorLoss, self).__init__()
        self.weight_box = weight_box
        self.weight_cls = weight_cls
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')
        self.iou_loss = IoULoss()
        self.assigner = MatchingAssigner()
    
    def remove_empty_boxes_vectorized(self, targets):
        valid_mask = (targets.sum(dim=-1) > 0)  # [B, N]
        return targets, valid_mask
    
    def decode_boxes_vectorized(self, preds, anchors, strides):
        box_xy = (preds[..., :2] + anchors.unsqueeze(0)) * strides  # [B, H*W, 2]
        box_wh = preds[..., 2:] * strides  # [B, H*W, 2]
        return torch.cat([box_xy, box_wh], dim=-1)  # [B, H*W, 4]
    
    def compute_iou_matrix_vectorized(self, pred_boxes, gt_boxes):
        B, H_W, _ = pred_boxes.shape
        _, N, _ = gt_boxes.shape
        
        # Reshape for vectorized computation
        pred_boxes_flat = pred_boxes.view(-1, 4)  # [B*H*W, 4]
        gt_boxes_flat = gt_boxes.view(-1, 4)  # [B*N, 4]
        
        # Compute IoU matrix
        iou_matrix = box_iou(cxcywh_to_xyxy(pred_boxes_flat), cxcywh_to_xyxy(gt_boxes_flat))
        
        # Reshape back to [B, H*W, N]
        iou_matrix = iou_matrix.view(B, H_W, B, N)
        
        # Extract diagonal elements (same batch)
        batch_indices = torch.arange(B, device=pred_boxes.device)
        iou_matrix = iou_matrix[batch_indices, :, batch_indices, :]  # [B, H*W, N]
        
        return iou_matrix
    
    def get_topk_matches_vectorized(self, iou_matrix, topk=9):
        B, H_W, N = iou_matrix.shape
        
        # Get max IoU for each anchor across all GTs
        max_ious, _ = torch.max(iou_matrix, dim=-1)  # [B, H*W]
        
        # Get topk anchors for each batch
        _, topk_indices = torch.topk(max_ious, min(topk, H_W), dim=-1)  # [B, topk]
        
        # Create positive mask
        positive_mask = torch.zeros_like(max_ious, dtype=torch.bool)
        batch_indices = torch.arange(B, device=iou_matrix.device).unsqueeze(1)
        positive_mask[batch_indices, topk_indices] = True
        
        return positive_mask
    
    def forward(self, preds, targets):
        device = preds.device
        anchors = make_anchors(preds)  # [H*W, 2]
        
        # Reshape predictions
        preds = preds.view(preds.size(0), -1, preds.size(-1))  # [B, H*W, 5]
        B, H_W, _ = preds.shape
        _, N, _ = targets.shape
        
        # Split predictions
        cls_pred = preds[:, :, 0]  # [B, H*W] - confidence
        box_pred = preds[:, :, 1:]  # [B, H*W, 4] - box coordinates
        
        # Clamp and decode box predictions
        box_pred = torch.clamp(box_pred, -1, 1)
        box_pred[:, :, :2] = box_pred[:, :, :2] + anchors.unsqueeze(0)  # Add anchor offsets
        
        # Remove empty boxes vectorized
        targets_clean, valid_mask = self.remove_empty_boxes_vectorized(targets)  # [B, N, 4], [B, N]
        
        # Compute IoU matrix for all batches
        iou_matrix = self.compute_iou_matrix_vectorized(box_pred, targets_clean)  # [B, H*W, N]
        
        # Apply valid mask to IoU matrix
        valid_mask_expanded = valid_mask.unsqueeze(1).expand(-1, H_W, -1)  # [B, H*W, N]
        iou_matrix = iou_matrix * valid_mask_expanded.float()
        
        # Get positive samples vectorized
        positive_mask = self.get_topk_matches_vectorized(iou_matrix, self.assigner.topk)  # [B, H*W]
        
        # Get target assignments
        target_assignments = torch.argmax(iou_matrix, dim=-1)  # [B, H*W]
        
        # Create target boxes for positive samples
        batch_indices = torch.arange(B, device=device).unsqueeze(1)  # [B, 1]
        pos_target_boxes = targets_clean[batch_indices, target_assignments]  # [B, H*W, 4]
        cls_target = torch.zeros_like(cls_pred)
        cls_target[positive_mask] = 1.0
        
        # Balanced confidence weights
        num_positives = positive_mask.sum(dim=-1, keepdim=True)  # [B, 1]
        balanced_conf = torch.ones_like(cls_pred)
        balanced_conf[positive_mask] = num_positives.expand(-1, H_W)[positive_mask] / (num_positives.expand(-1, H_W)[positive_mask] * 2)
        
        cls_loss = (self.bce_loss(cls_pred.sigmoid(), cls_target) * balanced_conf).sum(dim=-1).mean()
        
        # Box regression loss
        if positive_mask.any():
            pos_box_pred = box_pred[positive_mask]  # [total_positives, 4]
            pos_target_boxes_flat = pos_target_boxes[positive_mask]  # [total_positives, 4]
            box_loss = self.iou_loss(pos_box_pred, pos_target_boxes_flat)
        else:
            box_loss = torch.tensor(0.0, device=device)
        
        return (self.weight_box * box_loss), (self.weight_cls * cls_loss)

class MatchingAssigner:
    def __init__(self, topk=9):
        self.topk = topk
    
    def assign(self, anchors, preds, targets):

        if len(targets) == 0:
            return torch.zeros(anchors.shape[0], dtype=torch.bool, device=anchors.device), \
                   torch.zeros(anchors.shape[0], device=anchors.device), \
                   torch.zeros_like(preds, device=anchors.device)
        
        # Tính IoU giữa anchors và targets
        lambda_ious = torch.zeros((anchors.shape[0], targets.shape[0]), device=anchors.device)
        ious = torch.zeros((anchors.shape[0], targets.shape[0]), device=anchors.device)
        
        for i, target in enumerate(targets):
            # Tạo lambda_preds từ anchors và target size
            lambda_preds = torch.cat([anchors, target[2:].unsqueeze(0).repeat(anchors.shape[0], 1)], dim=-1)
            
            for j, (lambda_pred, pred) in enumerate(zip(lambda_preds, preds)):
                lambda_ious[j, i] = self.bbox_iou(lambda_pred, target)
                ious[j, i] = self.bbox_iou(pred, target)
        
        positive_mask = self._get_topk_positives(lambda_ious)
        
        target_conf = torch.max(ious, dim=1).values
        target_identities = torch.argmax(lambda_ious, dim=1)
        
        # Tạo new_box_label
        new_box_label = torch.zeros_like(preds, device=anchors.device)
        new_box_label[:] = targets[target_identities]
        
        return positive_mask, target_conf, new_box_label
    
    def _get_topk_positives(self, iou_matrix):
        max_ious, _ = torch.max(iou_matrix, dim=1)

        _, topk_indices = torch.topk(max_ious, min(self.topk, len(max_ious)))
        
        positive_mask = torch.zeros(len(max_ious), dtype=torch.bool, device=max_ious.device)
        positive_mask[topk_indices] = True
        
        return positive_mask
    
    def bbox_iou(self, box1, box2):
        """
        Tính IoU giữa 2 boxes (cx, cy, w, h)
        """
        # Convert to xyxy
        box1_xyxy = self.cxcywh_to_xyxy(box1.unsqueeze(0))
        box2_xyxy = self.cxcywh_to_xyxy(box2.unsqueeze(0))
        
        # Tính IoU
        return box_iou(box1_xyxy, box2_xyxy).squeeze()
    
    def cxcywh_to_xyxy(self, boxes):
        """
        Convert (cx, cy, w, h) to (x1, y1, x2, y2)
        """
        cx, cy, w, h = boxes.unbind(-1)
        x1 = cx - (1/2 * w)
        x2 = cx + (1/2 * w)
        y1 = cy - (1/2 * h)
        y2 = cy + (1/2 * h)
        return torch.stack((x1, y1, x2, y2), dim=-1)

class IoULoss(nn.Module):
    def __init__(self, eps=1e-6):
        super(IoULoss, self).__init__()
        self.eps = eps
    
    def forward(self, pred, target):
        pred_xyxy = cxcywh_to_xyxy(pred)
        target_xyxy = cxcywh_to_xyxy(target)
        iou = box_iou(pred_xyxy, target_xyxy)
        iou = torch.diag(iou)
        loss = 1 - iou
        return loss.mean()

def cxcywh_to_xyxy(boxes):
    cx, cy, w, h = boxes.unbind(-1)
    x1 = cx - (1/2 * w)
    x2 = cx + (1/2 * w)
    y1 = cy - (1/2 * h)
    y2 = cy + (1/2 * h)
    return torch.stack((x1,y1, x2,y2), dim = -1)


def box_iou(box1, box2):
    lt = torch.max(box1[:, None, :2], box2[:, :2])  # [N,M,2]
    rb = torch.min(box1[:, None, 2:], box2[:, 2:])  # [N,M,2]
    
    wh = (rb - lt).clamp(min=0)  # width & height của vùng giao
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N, M]

    # tính diện tích
    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])  # [N]
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])  # [M]

    # IoU
    union = area1[:, None] + area2 - inter
    iou = inter / (union + 1e-6)
    return iou

def make_anchors(feat, grid_cell_offset=0.5):
    assert feat is not None
    dtype, device = feat.dtype, feat.device 
    _, h, w, _ = feat.shape 
    sx = torch.arange(end=w, device=device, dtype=dtype) + grid_cell_offset # shift x 
    sy = torch.arange(end=h, device=device, dtype=dtype) + grid_cell_offset # shift y 
    sy, sx = torch.meshgrid(sy, sx) 
    anchor_points = torch.stack((sx, sy), -1).view(-1, 2) 
    anchor_points[..., 0] /= w # normalize 0~1 
    anchor_points[..., 1] /= h # normalize 0~1 return anchor_points
    return anchor_points