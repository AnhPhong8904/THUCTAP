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

from utils import visualize_training_data, COCO_SKELETON

    

def train(train_dir, model_save_path, epochs=100, batch_size=32, learning_rate=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset = HumanPoseDataset(train_dir, augment=False)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
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
            loss = obj_loss + kp_loss
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)

        if epoch_loss < min_loss:
            min_loss = epoch_loss
            torch.save(model.state_dict(), model_save_path)
            # try to predict and visualize
            model.eval()
            with torch.no_grad():
                sample_imgs, sample_targets = next(iter(train_loader))
                sample_imgs, sample_targets = sample_imgs.to(device), sample_targets.to(device)
                sample_outputs = model(sample_imgs)
                # visualize
                B, _, H, W = sample_imgs.shape
                images = []
                for i in range(B):
                    img = sample_imgs[i].permute(1, 2, 0).cpu().numpy() * 255  # CHW->HWC
                    img = img.astype(np.uint8)
                    img = np.ascontiguousarray(img)
                    conf = torch.sigmoid(sample_outputs[i][0][0]).item()
                    keypoints = sample_outputs[i][0][1:].cpu().numpy().reshape(-1, 3)  # (17, 3)
                    if conf > 0.5:
                        for x, y, v in keypoints:
                            if v > 0:
                                cv2.circle(img, (int(x), int(y)), 3, (0, 255, 0), -1)
                        for j, (start, end) in enumerate(COCO_SKELETON):
                            if keypoints[start][2] > 0 and keypoints[end][2] > 0:
                                cv2.line(img, (int(keypoints[start][0]), int(keypoints[start][1])),
                                        (int(keypoints[end][0]), int(keypoints[end][1])), (255, 0, 0), 2)
                    cv2.putText(img, f"Conf: {conf:.2f}", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                    images.append(img)
                img = np.hstack(images)
                save_path = os.path.join("visualize", f"training_pred.jpg")
                os.makedirs("visualize", exist_ok=True)
                cv2.imwrite(save_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        bpar.set_postfix({"loss": epoch_loss, "best_loss": min_loss})


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", type=str, required=True, help="Path to training dataset")
    parser.add_argument("--model_save_path", type=str, default="best.pt", help="Path to save the trained model")
    parser.add_argument("--epochs", type=int, default=200, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate for optimizer")
    args = parser.parse_args()
    train(args.train_dir, args.model_save_path, args.epochs, args.batch_size, args.learning_rate)