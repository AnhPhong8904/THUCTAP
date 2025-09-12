import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import os
import numpy as np

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
    
    def forward():
        pass

class Assigner:
    def __init__(self, topk=3):
        self.topk = topk
    def assign(self, anchors, preds, targets):
        ious = torch.zeros(anchors.shape[0], targets.shape[0], device=anchors.device)
        for i, target in enumerate(targets):
            




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
        for pred in preds:
            for target in targets:
                ious.append(bbox_iou(pred, target).unsqueeze(0))
        ious = torch.cat(ious, dim=0)  # [M*N]
        loss = 1 - ious  # [M*N]
        return loss.sum()



def bbox_iou(box1, box2, eps=1e-7):
    inter_x1 = torch.max(box1[:, None, 0], box2[:, 0])
    inter_y1 = torch.max(box1[:, None, 1], box2[:, 1])
    inter_x2 = torch.min(box1[:, None, 2], box2[:, 2])
    inter_y2 = torch.min(box1[:, None, 3], box2[:, 3])

    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    inter_area = inter_w * inter_h

    # Diện tích box
    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])

    # Union
    union = area1[:, None] + area2 - inter_area + eps

    return inter_area / union

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


