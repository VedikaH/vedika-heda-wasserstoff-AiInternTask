import pytesseract
from PIL import Image
import shutil

def find_tesseract_binary() -> str:
    return shutil.which("tesseract")

# Set the tesseract binary path
pytesseract.pytesseract.tesseract_cmd = find_tesseract_binary()
st.write("Tesseract binary path:", pytesseract.pytesseract.tesseract_cmd)

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
