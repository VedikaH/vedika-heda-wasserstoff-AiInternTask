import shutil
import streamlit as st
import os
import sys
import pandas as pd
import json
from PIL import Image
import logging


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.segmentation_model import SegmentationModel
from models.identification_model import IdentificationModel
from models.text_extraction_model import TextExtractionModel
from models.summarization_model import SummarizationModel
from utils.postprocessing import save_segmented_objects
from utils.data_mapping import map_data, save_mapped_data
from utils.visualization import visualize_detections, visualize_segmentation, create_summary_table

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

@st.cache_resource
def load_segmentation_model():
    return SegmentationModel()

@st.cache_resource
def load_identification_model():
    return IdentificationModel()

@st.cache_resource
def load_text_extraction_model():
    return TextExtractionModel()

@st.cache_resource
def load_summarization_model():
    return SummarizationModel()

def main():
    st.set_page_config(layout="wide")
    st.markdown("""
    <style>
    .stImage > div {
        margin-left: auto;
        margin-right: auto;
    }
    .stTable > div {
        margin-left: auto;
        margin-right: auto;
    }
    h1{ /* Title style */
            text-align: center;
        }
    </style>
    """, unsafe_allow_html=True)

    def clear_segmented_objects_folder(folder_path):
        # Remove all files in the segmented_objects folder
        if os.path.exists(folder_path) and os.path.isdir(folder_path):
            for filename in os.listdir(folder_path):
                file_path = os.path.join(folder_path, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)  # Remove the file
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)  # Remove the directory
                except Exception as e:
                    st.error(f'Failed to delete {file_path}. Reason: {e}')
        else:
            print(f"Folder '{folder_path}' does not exist, skipping the clearing step.")
        
    clear_segmented_objects_folder("data/segmented_objects")
    
    st.title("Image Processing Pipeline 🤖")

    # File upload
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    logging.debug(f"Uploaded file: {uploaded_file}")

    if uploaded_file is not None:
        # Save uploaded file
        input_path = os.path.join("data", "input_images", uploaded_file.name)
        with open(input_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        logging.debug(f"File saved to: {input_path}")

        image = Image.open(input_path)

        # Segmentation
        segmentation_model = load_segmentation_model()
        masks, boxes, labels, class_name = segmentation_model.segment_image(input_path)
        logging.debug(f"Segmentation results: {len(masks)} masks, {len(boxes)} boxes, {len(labels)} labels")
        
        # Save segmented objects
        objects = save_segmented_objects(image, masks, boxes, "data/segmented_objects")
        logging.debug(f"Saved {len(objects)} segmented objects")

        # Object identification
        identification_model = load_identification_model()
        detections = []
        for file in sorted(os.listdir("data/segmented_objects")):
            f = os.path.join("data/segmented_objects", file)
            obj_detections = identification_model.identify_objects(f, class_name)
            if obj_detections:  # Only append if the object was identified
                class_name.remove(obj_detections[0]['description'])
                detections.extend(obj_detections)
        logging.debug(f"Detections: {len(detections)} objects identified")

        # Match detections to segmented objects
        object_descriptions = []
        for obj, det in zip(objects, detections):
            if det:
                object_descriptions.append(f"This is a {det['description']} with confidence {det['probability']:.2f}")
            else:
                object_descriptions.append("Unidentified object")
        logging.debug(f"Object description: {detections}")

        output_dir = "data/output"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        # Save detections
        with open("data/output/detections.json", "w") as f:
            json.dump(detections, f)
        logging.debug("Detections saved to data/output/detections.json")

        # Text extraction
        text_extraction_model = load_text_extraction_model()
        extracted_texts = [text_extraction_model.extract_text(obj[1]) for obj in objects]
        logging.debug(f"Extracted texts: {extracted_texts}")

        # Summarization
        summarization_model = load_summarization_model()
        summaries = [summarization_model.summarize(f"{desc} {text}") for desc, text in zip(object_descriptions, extracted_texts)]
        logging.debug(f"Summaries: {summaries}")

        # Data mapping
        mapped_data = map_data(objects, detections, object_descriptions, extracted_texts, summaries)
        save_mapped_data(mapped_data, "data/output/mapped_data.json")

        # Visualization
        visualize_segmentation(image, masks, "data/output/segmented_image.png")
        visualize_detections(input_path, "data/output/detected_objects.png")
        create_summary_table(mapped_data, "data/output/summary_table.csv")
    
        # Load the images and table

        # Initialize session state if not already done
        if 'show_original_image' not in st.session_state:
            st.session_state.show_original_image = False
        if 'show_segmented_image' not in st.session_state:
            st.session_state.show_segmented_image = False
        if 'show_detected_objects' not in st.session_state:
            st.session_state.show_detected_objects = False
        if 'show_summary_table' not in st.session_state:
            st.session_state.show_summary_table = False

        button_col1, button_col2, button_col3, button_col4 = st.columns(4)

        with button_col1:
            if st.button("Show Original Image"):
                st.session_state.show_original_image = not st.session_state.show_original_image

        with button_col2:
            if st.button("Show Segmented Image"):
                st.session_state.show_segmented_image = not st.session_state.show_segmented_image

        with button_col3:
            if st.button("Show Detected Objects"):
                st.session_state.show_detected_objects = not st.session_state.show_detected_objects

        with button_col4:
            if st.button("Show Summary Table"):
                st.session_state.show_summary_table = not st.session_state.show_summary_table

        # Display components based on session state
        def resize_image(image_path, target_width, target_height):
            image = Image.open(image_path)
            resized_image = image.resize((target_width, target_height))
            return resized_image

        # Set desired width and height
        IMAGE_WIDTH = 600
        IMAGE_HEIGHT = 400
        
        if st.session_state.show_original_image:
            col1, col2, col3 = st.columns([0.3, 0.4, 0.3])
            with col2:
                resized_image = resize_image(input_path, IMAGE_WIDTH, IMAGE_HEIGHT)
                st.image(resized_image, caption="Original Image", use_column_width=True)

        if st.session_state.show_segmented_image:
            col1, col2, col3 = st.columns([0.3, 0.4, 0.3])
            with col2:
                resized_image = resize_image("data/output/segmented_image.png", IMAGE_WIDTH, IMAGE_HEIGHT)
                st.image(resized_image, caption="Segmented Image", use_column_width=True)

        if st.session_state.show_detected_objects:
            col1, col2, col3 = st.columns([0.3, 0.4, 0.3])
            with col2:
                resized_image = resize_image("data/output/detected_objects.png", IMAGE_WIDTH, IMAGE_HEIGHT)
                st.image(resized_image, caption="Detected Objects", use_column_width=True)

        if st.session_state.show_summary_table:
            col1, col2, col3 = st.columns([1, 3, 1])
            with col2:
                summary_table = pd.read_csv("data/output/summary_table.csv")
                st.table(summary_table)

if __name__ == "__main__":
    main()
