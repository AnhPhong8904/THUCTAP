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
    if image is None:
        raise FileNotFoundError(f"❌ Cannot read image: {image_path}")

    # preprocess to tensor (resize+pad -> 224x224)
    input_tensor, scale, left, top = BBoxDataset.preprocess_image(image)
    input_tensor = input_tensor.unsqueeze(0).to(device)  # add batch dimension

    # forward pass
    with torch.no_grad():
        output = model(input_tensor).cpu().numpy()[0]  # shape (num_boxes, 4)

    h, w = input_tensor.shape[2:4]  # 224x224
    results = []

    for i, (cx, cy, bw, bh) in enumerate(output):
        # bỏ qua box rỗng
        if bw == 0 and bh == 0:
            continue

        # convert từ normalized -> pixel trên ảnh resized
        xmin = (cx - bw / 2) * w
        xmax = (cx + bw / 2) * w
        ymin = (cy - bh / 2) * h
        ymax = (cy + bh / 2) * h

        # chuyển ngược về tọa độ ảnh gốc (khử pad + scale)
        xmin = (xmin - left) / scale
        ymin = (ymin - top) / scale
        xmax = (xmax - left) / scale
        ymax = (ymax - top) / scale

        # clamp trong biên ảnh
        xmin = max(0, xmin)
        ymin = max(0, ymin)
        xmax = min(image.shape[1] - 1, xmax)
        ymax = min(image.shape[0] - 1, ymax)

        # vẽ box
        cv2.rectangle(image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 2)
        cv2.putText(image, f"Pred{i+1}", (int(xmin), max(0, int(ymin) - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        results.append((xmin, ymin, xmax, ymax))

    # save result
    cv2.imwrite(save_path, image)
    print(f"✅ Predicted {len(results)} boxes")
    print(f"Result saved to {save_path}")
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Inference script for multi-box regression model")
    parser.add_argument("--image", "-i", type=str, default="dataset/test/airplane_001.jpg", help="Path to input image")
    parser.add_argument("--output", "-o", type=str, default="result.jpg", help="Path to save result image")
    parser.add_argument("--model", "-m", type=str, default="best1.pt", help="Path to trained model")
    args = parser.parse_args()

    infer(args.image, args.model, args.output)
