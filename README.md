# Image Processing Pipeline 🤖

## Overview

This repository contains an Image Processing Pipeline built using Streamlit. The pipeline allows users to upload an image, process it through various machine learning models for segmentation, object identification, text extraction, and summarization, and visualize the results interactively.


## Features

- Image Upload: Upload an image in .jpg, .jpeg, or .png format.
- Image Segmentation: Segment the uploaded image into distinct objects using a pre-trained segmentation model.
- Object Identification: Identify objects in the segmented images using an identification model.
- Text Extraction: Extract text from the segmented objects using an OCR model.
- Summarization: Generate a summarized description of each object based on identified objects and extracted text.
- Data Mapping: Map the segmented objects, their descriptions, and extracted texts into a structured format.
- Visualization: Visualize the original image, segmented image, detected objects, and a summary table.

## Prerequisites

Ensure you have the following installed:

- Python 3.7+
- pip

## Installation(For Local Development)

1. Clone the repository:
```
git clone https://github.com/your-username/image-processing-pipeline.git
cd image-processing-pipeline
```

2. Create a virtual environment:
```
python -m venv venv
```
3. Activate the virtual environment:

  On Windows:
  ```
  venv\Scripts\activate
  ```  
  On macOS/Linux:
  ```
  source venv/bin/activate
  ```
4. Install the required packages:
```
pip install -r requirements.txt
```

## Project Structure
``` bash
.
├── data
│   ├── input_images            # Uploaded images
│   ├── output
│   │   ├── detections.json     # JSON file storing detection results
│   │   ├── mapped_data.json    # JSON file storing mapped data
│   │   ├── summary_table.csv   # CSV file storing the summary table
│   │   ├── detected_objects.png  # Visualization of detected objects
│   │   └── segmented_image.png  # Visualization of segmented image
│   └── segmented_objects       # Directory storing segmented objects
├── models
│   ├── segmentation_model.py   # Segmentation model implementation
│   ├── identification_model.py # Object identification model implementation
│   ├── text_extraction_model.py  # Text extraction model implementation
│   └── summarization_model.py  # Summarization model implementation
├── utils
│   ├── postprocessing.py       # Utility functions for postprocessing
│   ├── data_mapping.py         # Utility functions for data mapping
│   └── visualization.py        # Utility functions for visualization
├── app.py                      # Main Streamlit app script
└── requirements.txt            # Required Python packages
```

## Usage

### Accessing the Deployed App
You can access the Image Processing Pipeline directly via the deployed Streamlit application 

### Running Locally (Optional)

1. Run the Streamlit app:
```
streamlit run main.py
```
2. Open your web browser and navigate to the provided local URL (usually `http://localhost:8501`).

3. Use the file uploader to select an image for processing.

4. The app will process the image and display various outputs:
- Original image
- Segmented image
- Detected objects
- Summary table of identified objects, extracted text, and summaries

5. Use the buttons to toggle the visibility of different outputs.

## Workflow

1. Upload Image: Upload an image from your local machine.

2. Segmentation: The app segments the image into distinct objects using a segmentation model.

3. Object Identification: Identified objects are matched to segmented parts using a pre-trained identification model.

4. Text Extraction: The app extracts text from each segmented object.

5. Summarization: Summaries are generated based on the identified objects and extracted text.

6. Visualization:

   - View the original image.
   - View the segmented image.
   - View the detected objects.
   - View the summary table.


## Logging

The application uses Python's built-in `logging` module for debugging. Log messages are displayed in the console.
