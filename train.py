import os
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

<<<<<<< HEAD
from model import SimpleCNN  # Assuming SimpleCNN is defined in model.py
from dataset import BBoxDataset  # Assuming BBoxDataset is defined in dataset.py
from ultis import ObjectDetectorLoss  # Import new loss function

def visualize_training_data(dataloader, save_dir="train_vis", num_batches=10):
    os.makedirs(save_dir, exist_ok=True)
    batch_count = 0
    for imgs, targets in dataloader:
        # imgs: [B, 3, H, W]
        # targets: [B, num_boxes, 4] (cx, cy, bw, bh) normalized
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
=======
from model import SimpleCNN
from dataset import BBoxDataset
from utils import visualize_training_data 
from utils import ObjectDetectorLoss
from inference import infer

def train(epochs=1000,
          batch_size=128,
          learning_rate=0.1):
>>>>>>> 06598f4e01cb973dad671c62be1a5184304610d3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset = BBoxDataset(json_dir="LabelSua", augment=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    visualize_training_data(train_loader, 
                            save_dir="visualize/train", 
                            num_batches=3)
    
    model = SimpleCNN().to(device)
<<<<<<< HEAD
    criterion = ObjectDetectorLoss(weight_box=5.0, weight_cls=0.5)  # New loss function
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 100
=======
    criterion = ObjectDetectorLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.99)
>>>>>>> 06598f4e01cb973dad671c62be1a5184304610d3
    min_loss = float('inf')
    for epoch in range(epochs):
        running_loss, running_loss_box, running_loss_cls = 0.0, 0.0, 0.0
        model.train()
        for imgs, targets in train_loader:
            imgs, targets = imgs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
<<<<<<< HEAD
            
            # ObjectDetectorLoss returns (box_loss, cls_loss)
            box_loss, cls_loss = criterion(outputs, targets)
            total_loss = box_loss + cls_loss
            
            total_loss.backward()
            optimizer.step()

            running_loss += total_loss.item() * imgs.size(0)

=======
            loss_box, loss_cls = criterion(outputs, targets)
            loss = loss_cls + loss_box
            loss.backward()
            optimizer.step()

            running_loss_box += loss_box.item()
            running_loss_cls += loss_cls.item()
            running_loss += loss.item()
        scheduler.step()
        epoch_loss_box = running_loss_box / len(train_loader)
        epoch_loss_cls = running_loss_cls / len(train_loader)
>>>>>>> 06598f4e01cb973dad671c62be1a5184304610d3
        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1: 3d}/{epochs: 3d}], Loss Box: {epoch_loss_box:.4f}, Loss Cls: {epoch_loss_cls:.4f}", end="")
        os.makedirs("checkpoints", exist_ok=True)
        torch.save(model.state_dict(), "checkpoints/last.pt")
        if epoch_loss < min_loss:
            min_loss = epoch_loss
            torch.save(model.state_dict(), "checkpoints/best.pt")
            print(f" ==> ✅ Model saved with loss: {min_loss:.4f}")
        else:
            print("")
        infer(train_dataset.images[0], model, save_path="test.jpg", confident_score_threshold=0.3)
        
<<<<<<< HEAD
        
        # test phase
        model.eval()
        test_loss = 0
        test_box_loss = 0
        test_cls_loss = 0
        for imgs, targets in test_loader:
            imgs, targets = imgs.to(device), targets.to(device)
            with torch.no_grad():
                outputs = model(imgs)
                box_loss, cls_loss = criterion(outputs, targets)
                total_loss = box_loss + cls_loss
                test_loss += total_loss.item()
                test_box_loss += box_loss.item()
                test_cls_loss += cls_loss.item()
        
        test_loss /= len(test_loader)
        test_box_loss /= len(test_loader)
        test_cls_loss /= len(test_loader)
        
        print(f"Epoch [{epoch + 1}/{epochs}], Train Loss: {epoch_loss:.4f}")
        print(f"  Test - Total: {test_loss:.4f}, Box: {test_box_loss:.4f}, Cls: {test_cls_loss:.4f}")
        if test_loss < min_loss:
            min_loss = test_loss
            torch.save(model.state_dict(), "best1.pt")
            print(f"✅ Model saved with loss: {min_loss:.4f}")

=======
>>>>>>> 06598f4e01cb973dad671c62be1a5184304610d3

if __name__ == "__main__":
    train()