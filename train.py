import os
import cv2
import numpy as np
from tqdm import tqdm
import torch
from torch.nn import MSELoss, L1Loss, BCEWithLogitsLoss as Bit
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

from model import MyModel
from dataset import HumanPoseDataset 

from utils import visualize_training_data

    

def train(train_dir, model_save_path, epochs=100, batch_size=32, learning_rate=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset = HumanPoseDataset(train_dir, augment=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    visualize_training_data(train_loader, save_dir="visualize/train", num_batches=10)
    
    model = MyModel(train_dataset.max_objs).to(device)
    kp_criterion = MSELoss()
    obj_criterion = Bit()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    min_loss = float('inf')
    bpar = tqdm(range(epochs), desc="Training")
    for epoch in bpar:
        running_loss = 0.0
        model.train()
        for imgs, targets in train_loader:
            imgs, targets = imgs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            obj_target = torch.sum(torch.sum(targets[:, :, 2] > 0, dim=1) > 0, dim=1) > 0
            obj_target = obj_target.unsqueeze(1).to(device)
            obj_pred = outputs[..., 0]
            kp_pred = outputs[..., 1:].reshape(outputs.size(0), -1)
            kp_target = targets.reshape(targets.size(0), -1)
            obj_loss = obj_criterion(obj_pred, obj_target.float())
            kp_loss = kp_criterion(kp_pred, kp_target)
            optimizer.step()

            running_loss += (obj_loss + kp_loss).item() * imgs.size(0)

        epoch_loss = running_loss / len(train_loader)

        if epoch_loss < min_loss:
            min_loss = epoch_loss
            torch.save(model.state_dict(), "best.pt")
            print(f"✅ Model saved with loss: {min_loss:.4f}")
        bpar.set_postfix({"loss": epoch_loss, "best_loss": min_loss})


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", type=str, required=True, help="Path to training dataset")
    parser.add_argument("--model_save_path", type=str, default="best.pt", help="Path to save the trained model")
    parser.add_argument("--epochs", type=int, default=10000, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate for optimizer")
    args = parser.parse_args()
    train(args.train_dir, args.model_save_path, args.epochs, args.batch_size, args.learning_rate)