import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.text_extraction_model import TextExtractionModel

def test_text_extraction_model():
    # Initialize the text extraction model
    text_extraction_model = TextExtractionModel()

    # Define the path to the test image
    # Make sure to replace this with an actual image file path in your input_images directory
    test_image_path = 'E:\downloads2\waser 5\waser\data\input_images\dog-park-petting-dog.jpg'

    # Check if the image exists
    if not os.path.exists(test_image_path):
        print(f"Test image not found: {test_image_path}")
        return

    # Perform text extraction
    extracted_text = text_extraction_model.extract_text(test_image_path)

    # Print the extracted text for inspection
    print("Extracted Text:", extracted_text)

    # Add assertions for testing (optional)
    assert extracted_text is not None, "No text was extracted."
    assert isinstance(extracted_text, str), "Extracted text is not a string."

if __name__ == "__main__":
    test_text_extraction_model()