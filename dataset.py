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
    def __init__(self, img_dir, csv_file, augment=True, target_size=224, max_objects=4):
        self.img_dir = img_dir
        self.augment = augment
        self.target_size = target_size
        self.max_objects = max_objects

        # group annotations by filename
        self.data = {}
        with open(csv_file, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                fname = row["filename"]
                if fname not in self.data:
                    self.data[fname] = []
                self.data[fname].append(row)
        self.files = list(self.data.keys())

    def __len__(self):
        return len(self.files)

    @staticmethod
    def resize_and_pad(image, target_size=224, random_pad=False):
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

        padded = cv2.copyMakeBorder(
            resized, top, bottom, left, right,
            borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0)
        )
        return padded, scale, left, top

    @staticmethod
    def preprocess_image(image, target_size=224, random_pad=False):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image, scale, left, top = BBoxDataset.resize_and_pad(
            image, target_size=target_size, random_pad=random_pad
        )
        image = image / 255.0
        tensor = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)
        return tensor, scale, left, top

    @staticmethod
    def normalize_bbox(xmin, ymin, xmax, ymax, w, h, scale, left, top):
        xmin = (xmin * scale) + left
        ymin = (ymin * scale) + top
        xmax = (xmax * scale) + left
        ymax = (ymax * scale) + top
        cx = (xmin + xmax) / 2 / w
        cy = (ymin + ymax) / 2 / h
        bw = (xmax - xmin) / w
        bh = (ymax - ymin) / h
        return [cx, cy, bw, bh]

    def __getitem__(self, idx):
        fname = self.files[idx]
        rows = self.data[fname]
        img_path = os.path.join(self.img_dir, fname)
        image = cv2.imread(img_path)

        image, scale, left, top = BBoxDataset.preprocess_image(
            image, target_size=self.target_size, random_pad=self.augment
        )
        h, w = image.shape[1:3]

        bboxes = []
        for row in rows:
            xmin = float(row["xmin"])
            ymin = float(row["ymin"])
            xmax = float(row["xmax"])
            ymax = float(row["ymax"])
            bbox = BBoxDataset.normalize_bbox(
                xmin, ymin, xmax, ymax, w, h, scale, left, top
            )
            bboxes.append(bbox)

        # pad hoặc cắt về đúng số max_objects
        if len(bboxes) < self.max_objects:
            bboxes += [[0, 0, 0, 0]] * (self.max_objects - len(bboxes))
        else:
            bboxes = bboxes[:self.max_objects]

        bboxes = torch.tensor(bboxes, dtype=torch.float32)  # (max_objects, 4)

        return image, bboxes
