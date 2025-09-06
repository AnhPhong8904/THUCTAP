"""
 @file: inference.py
 @brief Inference script for bounding box regression model (cx, cy, w, h normalized 0~1) with correct resize+pad handling
"""

import cv2
import numpy as np
import torch
from model import SimpleCNN
from dataset import BBoxDataset
import random
import time
from utils import make_anchors

COLORS = np.random.randint(0, 255, size=(196, 3), dtype=np.int32)

def infer(image_path, model, save_path="result.jpg", confident_score_threshold=0.1):
    device = next(model.parameters()).device
    # read image
    image = cv2.imread(image_path)
    # preprocess to tensor (resize+pad -> 224x224)
    input_tensor, scale, left, top = BBoxDataset.preprocess_image(image)
    input_tensor = input_tensor.unsqueeze(0).to(device)  # add batch dimension

    # forward pass
    with torch.no_grad():
        output = model(input_tensor).cpu()[0]  # [conf, cx, cy, w, h] normalized
    # postprocess
    anchors = make_anchors(output.unsqueeze_(0))  # [H*W, 2], [H*W, 1]
    output = output.view(-1, 5) # H*W, 5
    conf = output[:, 0].sigmoid().numpy()
    bboxes = output[:, 1:] # (cx, cy, w, h) normalized
    bboxes[:, :2] = (bboxes[:, :2] + anchors[:, :2])  # cx, cy
    conf_threshold = confident_score_threshold
    keep = conf >= conf_threshold
    h, w = input_tensor.shape[2:4] 
    # draw bbox
    img_cv = image.copy()
    for c, (cx, cy, bw, bh), (ax, ay), color in zip(conf[keep], bboxes[keep], anchors[keep], COLORS[keep]):
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
        ax, ay = int((ax * w - left) / scale), int((ay * h - top) / scale)
        cv2.rectangle(img_cv, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color.tolist(), 2)
        cv2.putText(img_cv, f"{c:.2f}", (int(ax) - 40,  int(ay) - 35),
                        cv2.FONT_HERSHEY_COMPLEX_SMALL, max(0.5, img_cv.shape[0] / 1000), color.tolist(),  2)
        cv2.circle(img_cv, (ax, ay), 30, color.tolist(), -1)

    # save result
    cv2.imwrite(save_path, img_cv)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Inference script for bounding box regression model")
    parser.add_argument("--image", "-i", type=str, required=True, help="Path to input image")
    parser.add_argument("--output", "-o", type=str, default="result.jpg", help="Path to input image")
    parser.add_argument("--model", "-m", type=str, default="checkpoints/best.pt", help="Path to model weights")
    parser.add_argument("--conf", "-c", type=float, default=0.1, help="Confidence threshold")
    args = parser.parse_args()
    
    # load model
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN().to(device)
    while True:
        try:
            model.load_state_dict(torch.load(args.model, map_location=device))
            break
        except:
            time.sleep(0.2)
            pass
    model.eval() 
    test_img = args.image  # ảnh test
    infer(test_img, model, args.output, args.conf)  # model path, output path
