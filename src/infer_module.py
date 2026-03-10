# infer_module.py
import os
import numpy as np
import cv2
import requests
import torch
from ultralytics import YOLO
from pathlib import Path

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = os.getenv("MODEL_PATH", "/models/best.pt")
model = YOLO(str(MODEL_PATH))
print(f"Using device: {DEVICE}")
print(f"Running model {MODEL_PATH}")


def _load_image(image_url: str = None, image_path: str = None) -> np.ndarray:
    """
    Load image from URL or local path and return as numpy array.
    Exactly one of image_url or image_path must be provided.
    """

    if (image_url is None and image_path is None) or (image_url and image_path):
        raise ValueError("Provide exactly one of image_url or image_path")

    if image_url:
        try:
            response = requests.get(image_url, timeout=10)
            response.raise_for_status()
        except requests.RequestException as e:
            raise RuntimeError(f"Failed to download image from URL: {e}")

        image_array = np.frombuffer(response.content, dtype=np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        if image is None:
            raise RuntimeError("Downloaded data is not a valid image")

        return image

    if image_path:
        image = cv2.imread(image_path)

        if image is None:
            raise RuntimeError(f"Failed to read image from path: {image_path}")

        return image


def infer_image(image_url: str = None, image_path: str = None):
    """
    Run YOLO inference on an image loaded from URL or disk.
    """
    image = _load_image(image_url=image_url, image_path=image_path)

    results = model(image, imgsz=1024, device=DEVICE)

    output_data = []
    for result in results:
        boxes = result.boxes
        if boxes is None:
            continue

        for box, conf, cls in zip(
            boxes.xyxy.cpu().numpy(),
            boxes.conf.cpu().numpy(),
            boxes.cls.cpu().numpy(),
        ):
            if conf < 0.3:
                continue

            x1, y1, x2, y2 = box
            output_data.append({
                "class": int(cls),
                "confidence": float(conf),
                "box": (float(x1), float(y1), float(x2), float(y2))
            })

    return output_data
