"""
 @file: inference.py
 @brief Inference script for bounding box regression model (cx, cy, w, h normalized 0~1) with correct resize+pad handling
"""

import cv2
import torch
from model import SimpleCNN
from dataset import BBoxDataset


def infer(image_path, model_path="simple_cnn_bbox.pth", save_path="result.jpg"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load model
    model = SimpleCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # read image
    image = cv2.imread(image_path)
    # preprocess to tensor (resize+pad -> 224x224)
    input_tensor, scale, left, top = BBoxDataset.preprocess_image(image)
    input_tensor = input_tensor.unsqueeze(0).to(device)  # add batch dimension

    # forward pass
    with torch.no_grad():
        output = model(input_tensor).cpu().numpy()[0]  # [cx, cy, w, h] normalized

    h, w = input_tensor.shape[2:4]  # should be 224x224
    cx, cy, bw, bh = output
    
    # convert to pixel coordinates
    xmin = (cx - bw / 2) * w 
    xmax = (cx + bw / 2) * w
    ymin = (cy - bh / 2) * h
    ymax = (cy + bh / 2) * h
    
    xmin = (xmin - left) / scale
    ymin = (ymin - top) / scale
    xmax = (xmax - left) / scale
    ymax = (ymax - top) / scale
    
    xmin = max(0, xmin)
    ymin = max(0, ymin)
    xmax = min(image.shape[1], xmax)
    ymax = min(image.shape[0], ymax)
    # draw bbox
    img_cv = image.copy()
    cv2.rectangle(img_cv, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 2)
    cv2.putText(img_cv, "Pred", (int(xmin), max(0, int(ymin) - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # save result
    cv2.imwrite(save_path, img_cv)
    print(f"Predicted bbox: ({xmin}, {ymin}), ({xmax}, {ymax})")
    print(f"Result saved to {save_path}")
    return xmin, ymin, xmax, ymax


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Inference script for bounding box regression model")
    parser.add_argument("--image", "-i", type=str, default="dataset/test/airplane_001.jpg", help="Path to input image")
    parser.add_argument("--output", "-o", type=str, default="result.jpg", help="Path to input image")
    args = parser.parse_args()
    test_img = args.image  # ảnh test
    infer(test_img, "best.pt", args.output)  # model path, output path
