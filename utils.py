import os
import cv2
import numpy as np
import torch

COCO_SKELETON = [
    (15, 13), (13, 11), (16, 14), (14, 12), (11, 12),
    (5, 11), (6, 12), (5, 6), (5, 7),
    (6, 8), (7, 9), (8, 10), (1, 2),
    (0, 1), (0, 2), (1, 3), (2, 4),
    (3, 5), (4, 6)
]

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

            # vẽ keypoints và skeleton
            keypoints = targets[i].numpy().reshape(-1, 3)  # (17, 3)
            for x, y, v in keypoints:
                if v > 0:  # v=0: not labeled, v=1: labeled but not visible, v=2: labeled and visible
                    cv2.circle(img, (int(x), int(y)), 3, (0, 255, 0), -1)
            for j, (start, end) in enumerate(COCO_SKELETON):
                if keypoints[start][2] > 0 and keypoints[end][2] > 0:
                    cv2.line(img, (int(keypoints[start][0]), int(keypoints[start][1])),
                             (int(keypoints[end][0]), int(keypoints[end][1])), (255, 0, 0), 2)

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