import os
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

from model import SimpleCNN
from dataset import BBoxDataset
from utils import visualize_training_data 
from utils import ObjectDetectorLoss
from inference import infer

def train(epochs=1000,
          batch_size=128,
          learning_rate=0.1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset = BBoxDataset(json_dir="dataset/LabelSua", augment=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    visualize_training_data(train_loader, 
                            save_dir="visualize/train", 
                            num_batches=3)
    
    model = SimpleCNN().to(device)
    criterion = ObjectDetectorLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.99)
    min_loss = float('inf')
    for epoch in range(epochs):
        running_loss, running_loss_box, running_loss_cls = 0.0, 0.0, 0.0
        model.train()
        for imgs, targets in train_loader:
            imgs, targets = imgs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
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
        

if __name__ == "__main__":
    train()
