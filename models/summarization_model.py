from transformers import BartForConditionalGeneration, BartTokenizer

class SummarizationModel:
    def __init__(self):
        self.model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
        self.tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
    
    def summarize(self, text):
    # Split the text into lines and remove empty lines
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        
        # If there's only one line, return it as is
        if len(lines) <= 1:
            return text.strip()
        
        # Otherwise, proceed with summarization
        inputs = self.tokenizer([text], max_length=1024, return_tensors="pt", truncation=True)
        summary_ids = self.model.generate(inputs["input_ids"], num_beams=4, max_length=100, early_stopping=True)
        return self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
