from ultralytics import YOLO
import numpy as np
import torchvision.transforms as transforms

class SegmentationModel:
    def __init__(self):
        self.model = YOLO('yolov8m-seg.pt')
        self.transform = transforms.Compose([
            transforms.Resize((640, 640)),  # Resize to YOLOv8 input size
            transforms.Lambda(lambda x: x.mul(255).byte()),  # Scale to 0-255 and convert to uint8
            transforms.Lambda(lambda x: x.permute(1, 2, 0).numpy())  # Change from BCHW to HWC
        ])

    def segment_image(self, image_path):
        
        results = self.model(image_path, conf=0.25)
        class_name=[]
        if results[0].masks is not None:
            for counter, detection in enumerate(results[0].masks.data):
                cls_id = int(results[0].boxes[counter].cls.item())
                class_name.append(self.model.names[cls_id])  
        print(class_name)


            
        # Extract masks, boxes, and labels
        result = results[0]
        masks = result.masks.data.cpu().numpy() if result.masks is not None else np.array([])
        boxes = result.boxes.xyxy.cpu().numpy() if result.boxes is not None else np.array([])
        labels = result.boxes.cls.cpu().numpy() if result.boxes is not None else np.array([])
        
        return masks, boxes, labels, class_name