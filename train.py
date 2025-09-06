import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from model import SimpleCNN
from dataset import BBoxDataset
from utils import visualize_training_data 
from utils import ObjectDetectorLoss


def train(epochs=1000,
          batch_size=128,
          learning_rate=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset = BBoxDataset(json_dir="dataset/LabelSua", augment=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    visualize_training_data(train_loader, 
                            save_dir="visualize/train", 
                            num_batches=3)
    
    model = SimpleCNN().to(device)
    criterion = ObjectDetectorLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    min_loss = float('inf')
    for epoch in range(epochs):
        running_loss_box, running_loss_cls = 0.0, 0.0
        model.train()
        for imgs, targets in train_loader:
            imgs, targets = imgs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss_box, loss_cls = criterion(outputs, targets)
            loss = loss_box + loss_cls
            loss.backward()
            optimizer.step()

            running_loss_box += loss_box.item()
            running_loss_cls += loss_cls.item()

        epoch_loss_box = running_loss_box / len(train_loader)
        epoch_loss_cls = running_loss_cls / len(train_loader)
        epoch_loss = epoch_loss_box + epoch_loss_cls
        print(f"Epoch [{epoch+1: 3d}/{epochs: 3d}], Loss Box: {epoch_loss_box:.4f}, Loss Cls: {epoch_loss_cls:.4f}", end="")
        os.makedirs("checkpoints", exist_ok=True)
        torch.save(model.state_dict(), "checkpoints/last.pt")
        if epoch_loss < min_loss:
            min_loss = epoch_loss
            torch.save(model.state_dict(), "checkpoints/best.pt")
            print(f" ==> ✅ Model saved with loss: {min_loss:.4f}")
        else:
            print("")
        

if __name__ == "__main__":
    train()
