"""
 @file: dataset.py
 @brief Custom PyTorch Dataset for bounding box regression with augmentations 
        (aspect-ratio preserving resize to 224x224 with cv2.copyMakeBorder, 
        plus random flip, scale, translate).
"""

import os
import csv
import torch
import cv2
import random
import numpy as np
from torch.utils.data import Dataset


class BBoxDataset(Dataset):
    def __init__(self, img_dir, csv_file, augment=True):
        self.img_dir = img_dir
        self.augment = augment

        # read csv
        self.data = []
        with open(csv_file, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.data.append(row)

    def __len__(self):
        return len(self.data)

    @staticmethod
    def resize_and_pad(image, target_size=224, random_pad=False):
        """Resize image with aspect ratio preserved, then pad with black borders"""
        h, w = image.shape[:2]
        scale = min(target_size / w, target_size / h)

        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(image, (new_w, new_h))

        pad_w = target_size - new_w
        pad_h = target_size - new_h

        top = random.randint(0, pad_h) if random_pad else pad_h // 2
        bottom = pad_h - top
        left = random.randint(0, pad_w) if random_pad else pad_w // 2
        right = pad_w - left

        padded = cv2.copyMakeBorder(resized, top, bottom, left, right,
                                    borderType=cv2.BORDER_REPLICATE, value=(0, 0, 0))
        return padded, scale, left, top

    @staticmethod
    def preprocess_image(image, random_pad=False):
        """Convert BGR->RGB, resize+pad, normalize, convert to tensor (CHW)"""
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image, scale, left, top = BBoxDataset.resize_and_pad(image, target_size=224, random_pad=random_pad)
        image = image / 255.0
        tensor = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)  # HWC -> CHW
        return tensor, scale, left, top

    @staticmethod
    def normalize_bbox(xmin, ymin, xmax, ymax, w, h, scale, left, top):
        """Convert pixel bbox to normalized (cx, cy, w, h)"""
        xmin = (xmin * scale) + left
        ymin = (ymin * scale) + top
        xmax = (xmax * scale) + left
        ymax = (ymax * scale) + top
        cx = (xmin + xmax) / 2 / w
        cy = (ymin + ymax) / 2 / h
        bw = (xmax - xmin) / w
        bh = (ymax - ymin) / h
        return torch.tensor([cx, cy, bw, bh], dtype=torch.float32)

    @staticmethod
    def augment_image(image, target):
        """Apply random flip, scale, translate to image & bbox"""
        _, H, W = image.shape
        cx, cy, bw, bh = target.tolist()

        # Convert normalized -> pixel
        xmin = (cx - bw / 2) * W
        ymin = (cy - bh / 2) * H
        xmax = (cx + bw / 2) * W
        ymax = (cy + bh / 2) * H

        img = (image.permute(1, 2, 0).numpy() * 255).astype(np.uint8)

        # Random horizontal flip
        if random.random() < 0.5:
            img = cv2.flip(img, 1)
            xmin, xmax = W - xmax, W - xmin  # flip bbox

        # Random scale (zoom in/out)
        if random.random() < 0.5:
            scale = random.uniform(0.5, 1.2)
            new_w, new_h = int(W * scale), int(H * scale)
            img = cv2.resize(img, (new_w, new_h))
            # Pad or crop to original size
            if scale < 1:  # pad
                pad_w = (W - new_w) // 2
                pad_h = (H - new_h) // 2
                img = cv2.copyMakeBorder(img, pad_h, H - new_h - pad_h, pad_w, W - new_w - pad_w,
                                         borderType=cv2.BORDER_REPLICATE)
                xmin = xmin * scale + pad_w
                xmax = xmax * scale + pad_w
                ymin = ymin * scale + pad_h
                ymax = ymax * scale + pad_h
            else:  # crop
                start_x = (new_w - W) // 2
                start_y = (new_h - H) // 2
                img = img[start_y:start_y + H, start_x:start_x + W]
                xmin = (xmin * scale) - start_x
                xmax = (xmax * scale) - start_x
                ymin = (ymin * scale) - start_y
                ymax = (ymax * scale) - start_y

        # Random translate
        if random.random() < 0.5:
            max_tx, max_ty = int(W * 0.2), int(H * 0.2)
            tx, ty = random.randint(-max_tx, max_tx), random.randint(-max_ty, max_ty)
            M = np.float32([[1, 0, tx], [0, 1, ty]])
            img = cv2.warpAffine(img, M, (W, H), borderValue=(0, 0, 0))
            xmin += tx
            xmax += tx
            ymin += ty
            ymax += ty

        # Clip bbox
        xmin, ymin = max(0, xmin), max(0, ymin)
        xmax, ymax = min(W, xmax), min(H, ymax)

        # Convert back to normalized (cx, cy, w, h)
        cx = (xmin + xmax) / 2 / W
        cy = (ymin + ymax) / 2 / H
        bw = (xmax - xmin) / W
        bh = (ymax - ymin) / H

        img = img.astype(np.float32) / 255.0
        tensor = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1)
        target = torch.tensor([cx, cy, bw, bh], dtype=torch.float32)
        return tensor, target

    def __getitem__(self, idx):
        row = self.data[idx]
        img_path = os.path.join(self.img_dir, row["filename"])
        image = cv2.imread(img_path)

        xmin = float(row["xmin"])
        ymin = float(row["ymin"])
        xmax = float(row["xmax"])
        ymax = float(row["ymax"])

        image, scale, left, top = BBoxDataset.preprocess_image(image, random_pad=self.augment)
        h, w = image.shape[1:3]
        target = BBoxDataset.normalize_bbox(xmin, ymin, xmax, ymax, w, h, scale, left, top)

        # apply augmentation
        if self.augment:
            image, target = BBoxDataset.augment_image(image, target)

        return image, target
