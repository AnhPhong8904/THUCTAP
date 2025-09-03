import os
import cv2
import numpy as np
import torch
from torch.nn import MSELoss, L1Loss
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

from model import SimpleCNN  # Assuming SimpleCNN is defined in model.py
from dataset import BBoxDataset  # Assuming BBoxDataset is defined in dataset.py

def visualize_training_data(dataloader, save_dir="train_vis", num_batches=10):
    os.makedirs(save_dir, exist_ok=True)
    batch_count = 0
    for imgs, targets in dataloader:
        # imgs: [B, 3, H, W]
        # targets: [B, num_boxes, 4] (cx, cy, bw, bh)
        B, _, H, W = imgs.shape
        num_boxes = targets.shape[1]

        grid_cols = max(1, int(np.sqrt(B)))
        grid_rows = int(np.ceil(B / grid_cols))

        grid_img = np.zeros((grid_rows * H, grid_cols * W, 3), dtype=np.uint8)

        for i in range(B):
            img = imgs[i].permute(1, 2, 0).numpy() * 255
            img = img.astype(np.uint8)
            img = np.ascontiguousarray(img)

            # lặp qua từng bbox trong ảnh
            for j in range(num_boxes):
                cx, cy, bw, bh = targets[i, j].numpy()

                # bỏ qua box rỗng
                if bw == 0 and bh == 0:
                    continue

                xmin = int((cx - bw / 2) * W)
                xmax = int((cx + bw / 2) * W)
                ymin = int((cy - bh / 2) * H)
                ymax = int((cy + bh / 2) * H)

                cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                cv2.putText(img, f"GT{j+1}", (xmin, max(ymin - 5, 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # gắn vào grid
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

    

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset = BBoxDataset(
        img_dir=r"datasua\train",
        csv_file=r"datasua\train.csv",
    )
    
    test_dataset = BBoxDataset(
        img_dir=r"datasua\test",
        csv_file=r"datasua\test.csv",
        augment=False
    )

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    visualize_training_data(train_loader, save_dir="visualize/train", num_batches=10)
    visualize_training_data(test_loader, save_dir="visualize/test", num_batches=3)
    
    model = SimpleCNN().to(device)
    criterion = MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 100
    min_loss = float('inf')
    for epoch in range(epochs):
        running_loss = 0.0
        model.train()
        for imgs, targets in train_loader:
            imgs, targets = imgs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * imgs.size(0)

        epoch_loss = running_loss / len(train_loader)
        
        
        # test phase
        model.eval()
        test_loss = 0
        for imgs, targets in test_loader:
            imgs, targets = imgs.to(device), targets.to(device)
            with torch.no_grad():
                outputs = model(imgs)
                test_loss += criterion(outputs, targets).item()
        test_loss /= len(test_loader)
        print(f"Epoch [{epoch + 1}/{epochs}], Train Loss: {epoch_loss:.4f}, Test Loss: {test_loss:.4f}")
        if test_loss < min_loss:
            min_loss = test_loss
            torch.save(model.state_dict(), "best1.pt")
            print(f"✅ Model saved with loss: {min_loss:.4f}")


if __name__ == "__main__":
    train()
