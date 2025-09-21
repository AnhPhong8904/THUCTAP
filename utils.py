import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
import os
from scipy.optimize import linear_sum_assignment
import math


class ObjectDetectorLoss(nn.Module):
    def __init__(self, 
                 weight_box=5,
                 weight_cls=0.5):
        super(ObjectDetectorLoss, self).__init__()
        self.weight_box = weight_box
        self.weight_cls = weight_cls
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')
        self.iou_loss = IoULoss()
        self.assigner = Assigner()
    
    def remove_empty_boxes(self, targets):
        # targets: [B, N, 4] (x1, y1, x2, y2) normalized
        mask = (targets.sum(dim=-1) > 0)  # [B, N]
        num_boxes = torch.sum(mask, dim=-1)  # [B]
        return targets[:num_boxes]
    
    @staticmethod
    def box_decode(anchors, boxes):
        a, b = boxes.chunk(2, -1)
        a = anchors - a
        b = anchors + b
        return torch.cat((a, b), -1)
    
    def forward(self, preds:torch.Tensor, targets:torch.Tensor):
        # preds: [B, H*W, 5] (conf, x1, y1, x2, y2) normalized
        # targets: [B, N, 4] (x1, y1, x2, y2) normalized
        anchors = make_anchors(preds)  # [W*H, 2], [W*H, 1]
        preds = preds.view(preds.size(0), -1, preds.size(-1))
        B, N, _ = targets.shape
        cls_loss = 0.0
        box_loss = 0.0
        for b in range(B):
            org_target = self.remove_empty_boxes(targets[b])
            pred = preds[b]  # [H*W, 5]
            cls_pred = pred[:, 0]  # [H*W]
            box_pred = pred[:, 1:]  # [H*W, 4]
            box_pred = self.box_decode(anchors, box_pred)  # [H*W, 4]
            cls_target = torch.zeros_like(cls_pred, device=preds.device)
            balanced_conf = torch.ones_like(cls_target, device=preds.device)
            if org_target.shape[0] != 0:
                # không có bbox GT trong ảnh này
                # tính loss toàn bộ là loss của class = 0
                positive_mask, target_conf, new_target_box = self.assigner.assign(anchors, box_pred, org_target)  # [topk]
                pos_box_pred = box_pred[positive_mask]  # [topk, 4]
                # pos_box_pred = box_pred[assigned_indices]  # [topk, 4]
                # # tính loss box
                box_loss += self.iou_loss(pos_box_pred, new_target_box[positive_mask]) / positive_mask.shape[0]
                # # tính loss class
                cls_target[positive_mask] = 1.0
                balanced_conf[positive_mask] = positive_mask.shape[0] / torch.sum(positive_mask) / 2
            else:
                # không có bbox GT trong ảnh này
                # tính loss toàn bộ là loss của class = 0
                pass
            cls_loss += (self.bce_loss(cls_pred, cls_target) * balanced_conf / cls_pred.shape[0]).sum()
            
        
        return (self.weight_box * box_loss / B), (self.weight_cls * cls_loss / B)
    
    

class Assigner:
    def __init__(self):
        self.topk = 3
    
    def assign(self, anchors, preds, targets):
        # anchors: [H*W, 2] (x1, y1) normalized
        # targets: [N, 4] (x1, y1, x2, y2) normalized
        ious = torch.zeros((anchors.shape[0], targets.shape[0]), device=anchors.device)
        for i, target in enumerate(targets):
            for j, pred in enumerate(preds):
                ious[j, i] = bbox_iou(pred, target)
        # get index of positives values 
        topk_indices = torch.topk(ious, self.topk, dim=0, largest=True).indices
        topk_masks = torch.zeros((anchors.shape[0], ), dtype=torch.bool, device=anchors.device)
        topk_masks[topk_indices]  =True
        target_conf = torch.max(ious, dim=1).values
        target_identities = torch.argmax(ious, dim=1)
        new_box_label = torch.zeros_like(preds, device=anchors.device)
        new_box_label[:] = targets[target_identities]  
        # visulize matching result   
        img = np.zeros((224, 224, 3), dtype=np.uint8)
        for i, (anchor, positive) in enumerate(zip(anchors, topk_masks)):
            x, y = int(anchor[0] * 224), int(anchor[1] * 224)
            color = (0, 255, 0) if positive else (255, 0, 0)
            cv2.circle(img, (x, y), 3, color, -1)
        os.makedirs("visualize/assigner", exist_ok=True)
        cv2.imwrite("visualize/assigner/assigner.jpg", img)
        return topk_masks, target_conf, new_box_label
        

class IoULoss(nn.Module):
    def __init__(self, eps=1e-6):
        super(IoULoss, self).__init__()
        self.eps = eps
        self.mse = nn.MSELoss(reduction='mean')

    def forward(self, preds, targets):
        """
        preds: [M, 4] (x1, y1, x2, y2) normalized
        targets: [N, 4] (x1, y1, x2, y2) normalized
        """
        # loss = self.mse(preds, targets)  # [M] IoU of matched boxes
        # # compute regression loss with MSE
        # return loss.sum()
        preds = preds   # scale to input image size
        targets = targets 
        # targets *= 224  # scale to input image size
        # preds *= 224  # scale to input image size
        ious = []
        for pred, target in zip(preds, targets):
            ious.append(bbox_iou(pred, target).unsqueeze(0))
        ious = torch.cat(ious, dim=0)  # [M*N]
        loss = 1 - ious  # [M*N]
        return loss.sum()
    
    
    
def bbox_iou(box1, box2):
    # box1, box2: (x1, y1, x2, y2)

    # Calculate intersection
    x1 = torch.max(box1[0], box2[0])
    y1 = torch.max(box1[1], box2[1])
    x2 = torch.min(box1[2], box2[2])
    y2 = torch.min(box1[3], box2[3])

    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)

    # Calculate union
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = box1_area + box2_area - intersection + 1e-6

    # Calculate IoU
    iou = intersection / union
    return iou

def make_anchors(feat, grid_cell_offset=0.5):
    """Generate anchors from features."""
    assert feat is not None
    dtype, device = feat.dtype, feat.device
    _, h, w, _ = feat.shape
    sx = torch.arange(end=w, device=device, dtype=dtype) + grid_cell_offset  # shift x
    sy = torch.arange(end=h, device=device, dtype=dtype) + grid_cell_offset  # shift y
    sy, sx = torch.meshgrid(sy, sx)
    anchor_points = torch.stack((sx, sy), -1).view(-1, 2)
    anchor_points[..., 0] /= w  # normalize 0~1
    anchor_points[..., 1] /= h  # normalize 0~1
    return anchor_points

def visualize_training_data(dataloader, save_dir="train_vis", num_batches=10):
    os.makedirs(save_dir, exist_ok=True)
    batch_count = 0
    for imgs, targets in dataloader:
        grid_cols = max(1, int(np.sqrt(len(imgs))))
        # imgs: [B, 3, 224, 224], targets: [B, 4]
        B, _, H, W = imgs.shape
        grid_rows = int(np.ceil(B / grid_cols))

        # tạo canvas chứa cả batch
        grid_img = np.zeros((grid_rows * H, grid_cols * W, 3), dtype=np.uint8)

        for i in range(B):
            img = imgs[i].permute(1, 2, 0).numpy() * 255  # CHW->HWC
            img = img.astype(np.uint8)
            img = np.ascontiguousarray(img)

            # bbox (cx,cy,bw,bh) normalized
            for box in targets[i]:
                box = box.numpy()
                if np.all(box == 0):
                    continue
                xmin, ymin, xmax, ymax = box
                xmin = int(xmin * W)
                ymin = int(ymin * W)
                xmax = int(xmax * H)
                ymax = int(ymax * H)
                cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                cv2.putText(img, "GT", (xmin, max(ymin - 5, 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # vị trí trong grid
            row, col = divmod(i, grid_cols)
            y0, y1 = row * H, (row + 1) * H
            x0, x1 = col * W, (col + 1) * W
            grid_img[y0:y1, x0:x1] = img

        save_path = os.path.join(save_dir, f"batch{batch_count}.jpg")
        cv2.imwrite(save_path, cv2.cvtColor(grid_img, cv2.COLOR_RGB2BGR))
        print(f"✅ Saved {save_path}")

        batch_count += 1
        if batch_count >= num_batches:
            break

    print(f"Saved {batch_count} batches of training samples to '{save_dir}'")
    
    