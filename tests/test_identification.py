import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.identification_model import IdentificationModel

def test_identification_model():
    # Initialize the identification model
    identification_model = IdentificationModel()

    # Define the path to the test image
    # Make sure to replace this with an actual image file path in your input_images directory
    test_image_path = 'E:\downloads2\waser 5\waser\data\input_images\dog-park-petting-dog.jpg'

    # Check if the image exists
    if not os.path.exists(test_image_path):
        print(f"Test image not found: {test_image_path}")
        return

    # Define text descriptions for the objects you expect to find in the image
    text_descriptions = ["a stop sign", "a car", "a road", "a tree"]

    # Perform identification
    detections = identification_model.identify_objects(test_image_path, text_descriptions)

    # Print the results for inspection
    print("Detections:", detections)

    # Add assertions for testing (optional)
    assert len(detections) > 0, "No objects were identified."
    assert detections[0]['description'] in text_descriptions, "Identified object description is not in the provided text descriptions."
    assert 0 <= detections[0]['probability'] <= 1, "Probability is not within the expected range (0 to 1)."

if __name__ == "__main__":
    test_identification_model()
