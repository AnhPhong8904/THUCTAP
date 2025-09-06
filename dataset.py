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
import base64
import json
import tqdm


class BBoxDataset(Dataset):
    def __init__(self, json_dir, augment=True):
        self.augment = augment
        self.json_dir = json_dir
        # read csv
        self.images = []
        self.bboxes = []
        self.max_boxes = 0
        self.load_data()
        
    def load_data(self):
        json_files = [f for f in os.listdir(self.json_dir) if f.endswith('.json')]
        for json_file in tqdm.tqdm(json_files, desc="Loading JSON files..."):
            if json_file != "1.json":  # skip corrupted file
                continue
            json_path = os.path.join(self.json_dir, json_file)
            with open(json_path, 'r') as f:
                data = json.load(f)
                image = base64.b64decode(data["imageData"])
                image = np.frombuffer(image, np.uint8)
                image = cv2.imdecode(image, cv2.IMREAD_COLOR)
                self.images.append(image)
                bboxes = []
                for item in data["shapes"]:
                    box = np.array(item["points"]).astype(np.int32).flatten().tolist()
                    # make sure box is [xmin, ymin, xmax, ymax]
                    box = [min(box[0], box[2]), min(box[1], box[3]), max(box[0], box[2]), max(box[1], box[3])]
                    bboxes.append(box)
                self.bboxes.append(bboxes)
                self.max_boxes = max(self.max_boxes, len(bboxes))
                cv2.imwrite(json_file.replace(".json", ".jpg"), image)
                
    def __len__(self):
        return len(self.images)

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


    def __getitem__(self, idx):
        image = self.images[idx]
        bboxes = self.bboxes[idx]

        image, scale, left, top = BBoxDataset.preprocess_image(image, random_pad=self.augment)
        h, w = image.shape[1:3]
        bboxes = [BBoxDataset.normalize_bbox(box[0], box[1], box[2], box[3], w, h, scale, left, top) for box in bboxes]
        if len(bboxes) < self.max_boxes:
            bboxes += [torch.tensor([0, 0, 0, 0], dtype=torch.float32)] * (self.max_boxes - len(bboxes))
        bboxes = torch.stack(bboxes)
        return image, bboxes


if __name__ == "__main__":
    dataset = BBoxDataset(json_dir="dataset/LabelSua", augment=True)
    print(f"Dataset size: {len(dataset)}")
    for i in range(5):
        img, tgt = dataset[i]
        print(f"Image {i} shape: {img.shape}, Target: {tgt}")