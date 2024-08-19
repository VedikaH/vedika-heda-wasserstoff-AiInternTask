import os
from PIL import Image

def save_segmented_objects(image, masks, boxes, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    objects = []
    for i, (mask, box) in enumerate(zip(masks, boxes)):
        obj_image = image.crop(box)
        file_path = os.path.join(output_dir, f"object_{i}.png")
        obj_image.save(file_path)
        objects.append((f"object_{i}", file_path, box.tolist()))
    return objects