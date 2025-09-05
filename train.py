import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from model import SimpleCNN
from dataset import BBoxDataset
from utils import visualize_training_data 
from utils import ObjectDetectorLoss


def train(epochs=1000,
          batch_size=16,
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
        print(f"Epoch [{epoch+1: 3d}/{epochs: 3d}], Loss: {epoch_loss:.4f}", end="")
    
        if epoch_loss < min_loss:
            min_loss = epoch_loss
            torch.save(model.state_dict(), "best.pt")
            print(f" ==> ✅ Model saved with loss: {min_loss:.4f}")
        else:
            print("")
        torch.save(model.state_dict(), "last.pt")

if __name__ == "__main__":
    train()
