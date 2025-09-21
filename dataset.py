"""
 @file: dataset.py
 @brief Custom PyTorch Dataset for bounding box regression with augmentations 
        (aspect-ratio preserving resize to 224x224 with cv2.copyMakeBorder, 
        plus random flip, scale, translate).
"""

import os
import tqdm
import torch
import cv2
import numpy as np
import json
import glob
from torch.utils.data import Dataset


class HumanPoseDataset(Dataset):
    def __init__(self, dataset_dir, augment=True):
        self.img_dir = dataset_dir
        self.augment = augment
        self.images = []
        self.labels = []
        self._load_data()
        self.max_objs = max(len(lbl) for lbl in self.labels)
        
    def _load_data(self):
        image_paths = glob.glob(self.img_dir + "/*.jpg")
        for imp in tqdm.tqdm(image_paths):
            ann_path = imp.replace(".jpg", ".json")
            image = cv2.imread(imp)
            if os.path.exists(ann_path):
                js = json.loads(open(ann_path).read())
                anns = js["annotations"]
                labels = []
                for ann in anns:
                    labels.append(np.array(ann["keypoints"]).reshape(-1, 3))
                self.labels.append(labels)
            else:
                self.labels.append([])
            self.images.append(image)
                

    def __len__(self):
        return len(self.images)

    @staticmethod
    def resize_and_pad(image, target_size=224, random=False):
        """Resize image with aspect ratio preserved, then pad with black borders"""
        h, w = image.shape[:2]
        scale = min(target_size / w, target_size / h)
        if random:
            scale *= np.random.uniform(0.5, 1., size=1)[0]

        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(image, (new_w, new_h))

        pad_w = target_size - new_w
        pad_h = target_size - new_h

        top = np.random.randint(0, pad_h) if random else pad_h // 2
        bottom = pad_h - top
        left = np.random.randint(0, pad_w) if random else pad_w // 2
        right = pad_w - left

        padded = cv2.copyMakeBorder(resized, top, bottom, left, right,
                                    borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0))
        return padded, scale, left, top

    @staticmethod
    def preprocess_image(image, random=False):
        """Convert BGR->RGB, resize+pad, normalize, convert to tensor (CHW)"""
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image, scale, left, top = HumanPoseDataset.resize_and_pad(image, target_size=224, random=random)
        image = image / 255.0
        tensor = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)  # HWC -> CHW
        return tensor, scale, left, top

    def __getitem__(self, idx):
        image = self.images[idx]
        labels = self.labels[idx]
        image, scale, left, top = self.preprocess_image(image, random=self.augment)
        # process keypoint labels
        keypoints = np.zeros((self.max_objs, 17, 3), dtype=np.float32)
        for i, kp in enumerate(labels):
            kp = kp.copy()
            kp[:, 0] = kp[:, 0] * scale + left
            kp[:, 1] = kp[:, 1] * scale + top
            keypoints[i] = kp
        keypoints = torch.tensor(keypoints, dtype=torch.float32)
        return image, keypoints
