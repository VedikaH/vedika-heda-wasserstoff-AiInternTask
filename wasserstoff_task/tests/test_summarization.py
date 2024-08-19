import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.summarization_model import SummarizationModel

def test_summarization_model():
    # Initialize the summarization model
    summarization_model = SummarizationModel()

    # Define a sample text to summarize
    sample_text = """
    The recent advancements in artificial intelligence have been remarkable. 
    With the advent of new models and techniques, AI has become increasingly proficient in tasks such as natural language processing, image recognition, and decision-making. 
    However, challenges still remain, particularly in areas like ethical considerations, data privacy, and the interpretability of AI models. 
    Moving forward, it is crucial to address these challenges to ensure that AI technology is developed responsibly and benefits society as a whole.
    """

    # Perform summarization
    summarized_text = summarization_model.summarize(sample_text)

    # Print the summarized text for inspection
    print("Summarized Text:", summarized_text)

    # Add assertions for testing (optional)
    assert summarized_text is not None, "Summarization returned None."
    assert isinstance(summarized_text, str), "Summarized text is not a string."
    assert len(summarized_text) < len(sample_text), "Summarized text is not shorter than the original text."

if __name__ == "__main__":
    test_summarization_model()
