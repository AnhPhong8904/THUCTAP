"""
 @file: inference.py
 @brief Inference script for bounding box regression model (cx, cy, w, h normalized 0~1) with correct resize+pad handling
"""

import cv2
import torch
from model import SimpleCNN
from dataset import BBoxDataset


def infer(image_path, model_path, save_path="result.jpg", confident_score_threshold=0.1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load model
    model = SimpleCNN().to(device)
    while True:
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
            break
        except:
            pass
    model.eval()

    # read image
    image = cv2.imread(image_path)
    # preprocess to tensor (resize+pad -> 224x224)
    input_tensor, scale, left, top = BBoxDataset.preprocess_image(image)
    input_tensor = input_tensor.unsqueeze(0).to(device)  # add batch dimension

    # forward pass
    with torch.no_grad():
        output = model(input_tensor).cpu()[0]  # [conf, cx, cy, w, h] normalized
    # postprocess
    conf = output[:, 0].sigmoid().numpy()
    bboxes = output[:, 1:].numpy()  # (cx, cy, w, h) normalized
    conf_threshold = confident_score_threshold
    keep = conf >= conf_threshold
    print(f"Detected {keep.sum()} boxes with conf >= {conf_threshold}")
    h, w = input_tensor.shape[2:4] 
    # draw bbox
    img_cv = image.copy()
    for c, (cx, cy, bw, bh) in zip(conf[keep], bboxes[keep]):
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
        cv2.rectangle(img_cv, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 2)
        cv2.putText(img_cv, f"Pred {c:.2f}", (int(xmin),  int(ymin) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

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
    test_img = args.image  # ảnh test
    infer(test_img, args.model, args.output, args.conf)  # model path, output path
