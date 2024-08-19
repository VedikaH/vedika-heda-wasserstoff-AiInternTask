import easyocr

class TextExtractionModel:
    def __init__(self):
        self.reader = easyocr.Reader(['en'])

    def extract_text(self, image_path):
        result = self.reader.readtext(image_path)
        return ' '.join([detection[1] for detection in result])
