import pytesseract
from PIL import Image

class TextExtractionModel:
    def __init__(self):
        # No initialization needed for pytesseract
        pass

    def extract_text(self, image_path):
        # Open the image file
        img = Image.open(image_path)
        # Use pytesseract to extract text
        result = pytesseract.image_to_string(img)
        return result
