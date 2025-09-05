import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
import os


class ObjectDetectorLoss(nn.Module):
    def __init__(self, 
                 weight_box=5,
                 weight_cls=0.5,
                 topk=3):
        super(ObjectDetectorLoss, self).__init__()
        self.weight_box = weight_box
        self.weight_cls = weight_cls
        self.mse_loss = nn.MSELoss(reduction='sum')
        self.ce_loss = nn.CrossEntropyLoss(reduction='sum')
        self.iou_loss = IoULoss()
        self.assigner = Assigner(topk=topk)
    
    def remove_empty_boxes(self, targets):
        # targets: [B, N, 4] (cx, cy, w, h) normalized
        mask = (targets.sum(dim=-1) > 0)  # [B, N]
        num_boxes = torch.sum(mask, dim=-1)  # [B]
        return targets[:num_boxes]
    
    def forward(self, preds:torch.Tensor, targets:torch.Tensor):
        # preds: [B, 196, 5] (conf, cx, cy, w, h) normalized
        # targets: [B, N, 4] (cx, cy, w, h) normalized
        B, N, _ = targets.shape
        cls_loss = 0.0
        box_loss = 0.0
        for b in range(B):
            org_target = self.remove_empty_boxes(targets[b])
            pred = preds[b]  # [196, 5]
            cls_pred = pred[:, 0]  # [196]
            box_pred = pred[:, 1:]  # [196, 4]
            cls_target = torch.zeros_like(cls_pred, device=preds.device)
            if org_target.shape[0] != 0:
                # không có bbox GT trong ảnh này
                # tính loss toàn bộ là loss của class = 0
                assigned_indices = self.assigner.assign(box_pred, org_target)  # [topk]
                pos_box_pred = box_pred[assigned_indices]  # [topk, 4]
                # tính loss box
                box_loss += self.iou_loss(pos_box_pred, org_target)
                # tính loss class
                cls_target[assigned_indices] = 1.0  # có object
            else:
                # không có bbox GT trong ảnh này
                # tính loss toàn bộ là loss của class = 0
                pass
            cls_loss += self.ce_loss(cls_pred.unsqueeze(-1), cls_target.unsqueeze(-1))
        
        print(box_loss, cls_loss)
        return (self.weight_box * box_loss + self.weight_cls * cls_loss) / B
    
    

class Assigner:
    def __init__(self, topk=3):
        self.topk = topk
    
    def assign(self, preds, targets):
        # preds: [196, 5] (conf, cx, cy, w, h) normalized
        # targets: [N, 4] (cx, cy, w, h) normalized
        ious = torch.zeros((preds.shape[0], targets.shape[0]), device=preds.device)
        for i, target in enumerate(targets):
            for j, pred in enumerate(preds):
                ious[j, i] = bbox_iou(pred, target)
        assigned_indices = torch.topk(ious, self.topk, dim=0).indices  # [topk, N]
        assigned_indices = assigned_indices.view(-1).unique()  # loại bỏ trùng lặp
        return assigned_indices  # [M] M <= topk * N
    

class IoULoss(nn.Module):
    def __init__(self, eps=1e-6):
        super(IoULoss, self).__init__()
        self.eps = eps

    def forward(self, preds, targets):
        # preds: [M, 4] (cx, cy, w, h) normalized
        # targets: [N, 4] (cx, cy, w, h) normalized
        # preds and targets are in (cx, cy, w, h) format
        preds = self.cxcywh_to_xyxy(preds)
        targets = self.cxcywh_to_xyxy(targets)
        # use mse distance to calculate loss
        for target in targets:
            # TODO

    @staticmethod
    def cxcywh_to_xyxy(boxes):
        cx, cy, w, h = boxes.unbind(-1)
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        return torch.stack((x1, y1, x2, y2), dim=-1)
    
    
    
def bbox_iou(box1, box2):
    # box1, box2: (cx, cy, w, h)
    box1 = cxcywh_to_xyxy(box1)
    box2 = cxcywh_to_xyxy(box2)

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

def cxcywh_to_xyxy(box):
    cx, cy, w, h = box
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    return torch.tensor([x1, y1, x2, y2], device=box.device)


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
                cx, cy, bw, bh = box
                xmin = int((cx - bw / 2) * W)
                xmax = int((cx + bw / 2) * W)
                ymin = int((cy - bh / 2) * H)
                ymax = int((cy + bh / 2) * H)

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

    print(f"Saved {num_batches} batches of training samples to '{save_dir}'")