import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import torch
from PIL import Image
import numpy as np
from ultralytics import YOLO
from models.segmentation_model import SegmentationModel

def test_segmentation():
    # Initialize the segmentation model
    model = SegmentationModel()

    # Specify the path to your test image
    image_path = 'E:\downloads2\waser 5\waser\data\input_images\dog-park-petting-dog.jpg'

    # Run segmentation
    masks, boxes, labels, class_names = model.segment_image(image_path)

    # Output the results
    print(f"Detected Classes: {class_names}")
    print(f"Bounding Boxes: {boxes}")
    print(f"Labels: {labels}")
    print(f"Masks Shape: {masks.shape if masks.size else 'No masks detected'}")

    # Optionally, visualize the results
    if boxes.size > 0:
        image = cv2.imread(image_path)
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, class_names[i], (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
        cv2.imshow("Segmented Image", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    test_segmentation()