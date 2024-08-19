import matplotlib.pyplot as plt
import pandas as pd
import cv2
from ultralytics import YOLO
from PIL import Image
def visualize_detections(image_path, output_path):
    
    model = YOLO('yolov8s.pt')  # You can change this to other YOLOv8 models as needed
    # Read the image
    image = cv2.imread(image_path)

    # Run YOLOv8 inference on the image
    results = model(image)

    # Process the results and draw bounding boxes
    for result in results:
        boxes = result.boxes.cpu().numpy()
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            class_name = model.names[class_id]

            # Draw bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Prepare label
            label = f"{class_name}"
            
            # Get label size
            (label_width, label_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

            # Draw filled rectangle for label background
            cv2.rectangle(image, (x1, y1 - label_height - baseline), (x1 + label_width, y1), (0, 255, 0), cv2.FILLED)

            # Put label text
            cv2.putText(image, label, (x1, y1 - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    # Save the output image
    cv2.imwrite(output_path, image)

def visualize_segmentation(image, masks, output_file):
    #plt.imshow(image)
    for mask in masks:
        plt.imshow(mask, alpha=0.5)
    plt.axis('off')
    plt.savefig(output_file,bbox_inches='tight', pad_inches=0)
    plt.close()


def create_summary_table(mapped_data, output_file):
    df = pd.DataFrame.from_dict(mapped_data, orient='index')
    df.to_csv(output_file)
