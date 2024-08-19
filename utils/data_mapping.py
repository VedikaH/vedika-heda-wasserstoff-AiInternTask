import json

def map_data(objects,detections, descriptions, extracted_texts, summaries):
    mapped_data = {}
    for (obj_id, file_path, box),det, description, text, summary in zip(objects,detections, descriptions, extracted_texts, summaries):
            mapped_data[obj_id] = {
            "file_path": file_path,
            "box": box,
            "description": description,
            "extracted_text": text,
            "summary": summary
        }

    return mapped_data

def save_mapped_data(mapped_data, output_file):
    with open(output_file, "w") as f:
        json.dump(mapped_data, f, indent=2)