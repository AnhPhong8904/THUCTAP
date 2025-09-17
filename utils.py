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
        # targets: [N, 4] (cho 1 ảnh)
        mask = (targets.sum(dim=-1) > 0)  # [N]
        return targets[mask]

    
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
        org_target = self.remove_empty_boxes(targets)
        pred = preds  # [H*W, 5]
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
    def __init__(self, topk=3):
        self.topk = topk
    
    def assign(self, anchors, preds, targets):
        # anchors: [M, 2] (x1, y1)
        # preds:   [M, 4] (x1, y1, x2, y2)  z
        # targets: [N, 4] (x1, y1, x2, y2)

        # 1. Tính IoU giữa toàn bộ preds và targets → [M, N]
        ious = bbox_iou(preds, targets)  

        # 2. Chọn top-k anchors cho từng target (theo IoU lớn nhất)
        topk_indices = torch.topk(ious, self.topk, dim=0, largest=True).indices  # [topk, N]

        # 3. Tạo mask đánh dấu positive anchors
        topk_masks = torch.zeros((anchors.shape[0],), dtype=torch.bool, device=anchors.device)
        topk_masks[topk_indices.reshape(-1)] = True   # flatten trước khi set True

        # 4. Confidence (theo IoU max với bất kỳ target nào)
        target_conf = ious.max(dim=1).values  # [M]

        # 5. Identity của target tương ứng (theo IoU lớn nhất)
        target_identities = ious.argmax(dim=1)  # [M]

        # 6. Box label mới = box của target tương ứng
        new_box_label = targets[target_identities]  # [M, 4]

        # 7. Visualization (optional)
        img = np.zeros((224, 224, 3), dtype=np.uint8)
        for anchor, positive in zip(anchors, topk_masks):
            x, y = int(anchor[0] * 224), int(anchor[1] * 224)
            color = (0, 255, 0) if positive else (255, 0, 0)
            cv2.circle(img, (x, y), 3, color, -1)
        os.makedirs("visualize/assigner", exist_ok=True)
        cv2.imwrite("visualize/assigner/assigner.jpg", img)

        return topk_masks, None, new_box_label

class IoULoss(nn.Module):
    def __init__(self, eps=1e-6):
        super(IoULoss, self).__init__()
        self.eps = eps

    def forward(self, preds, targets):
        """
        preds:   [M, 4] (x1, y1, x2, y2)
        targets: [M, 4] (x1, y1, x2, y2)  # matched one-to-one
        """
        # Tính IoU từng cặp (pred_i vs target_i)
        ious = bbox_iou(preds, targets, self.eps)  # [M]
        loss = 1.0 - ious
        return loss.sum() #




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

    iou = inter_area / union

    return iou # [M,N]

def make_anchors(feat, grid_cell_offset=0.5): 
    """Generate anchors from features."""
    assert feat is not None
    dtype, device = feat.dtype, feat.device
    _, h, w, _ = feat.shape  #feat = [B, H, W, C]
    sx = torch.arange(end=w, device=device, dtype=dtype) + grid_cell_offset  # shift x
    sy = torch.arange(end=h, device=device, dtype=dtype) + grid_cell_offset  # shift y
    sy, sx = torch.meshgrid(sy, sx)
    anchor_points = torch.stack((sx, sy), -1).view(-1, 2)
    anchor_points[..., 0] /= w  # normalize 0~1
    anchor_points[..., 1] /= h  # normalize 0~1
    return anchor_points # [H*W, 2]


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


