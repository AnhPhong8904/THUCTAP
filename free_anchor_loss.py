import torch
import torch.nn as nn
import torch.nn.functional as F


def cxcywh_to_xyxy(boxes):
    cx, cy, w, h = boxes.unbind(-1)
    x1 = cx - 1/2 * w
    y1 = cy - 1/2 * h
    x2 = cx + 1/2 * w
    y2 = cy + 1/2 * h
    return torch.stack([x1,y1,x2,y2], dim=-1)

def box_iou(boxes1, boxes2):
    N, M = boxes1.size(0), boxes2.size(0)
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # (N,M,2)
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # (N,M,2)
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    iou = inter / (area1[:, None] + area2 - inter + 1e-6)
    return iou

class FreeAnchorLoss(nn.Module):
    def __init__(self, topk=3, lamda=1.0):
        super(FreeAnchorLoss, self).__init__()
        self.topk =topk
        self.lamda = lamda
        
