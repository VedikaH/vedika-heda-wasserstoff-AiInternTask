from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch

class IdentificationModel:
    def __init__(self):
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    def identify_objects(self, image_path, text_descriptions):
        # Load image
        image = Image.open(image_path)

        # Prepare inputs
        inputs = self.processor(text=text_descriptions, images=image, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Run inference
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Get logits and compute probabilities
        logits_per_image = outputs.logits_per_image # this is the image-text similarity score
        probs = logits_per_image.softmax(dim=1) # convert logits to probabilities

        # Find the detection with the maximum probability
        max_prob, max_idx = torch.max(probs[0], dim=0)

        # Prepare the result for the highest probability detection
        detection=[]
        detection.append({
            'description': text_descriptions[max_idx],
            'probability': float(max_prob)
        })

        return detection


