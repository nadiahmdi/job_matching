
# Import Libraries 
from typing import Any
from sentence_transformers import SentenceTransformer

# Load a pre-trained Sentence-BERT model
model: SentenceTransformer = SentenceTransformer('all-MiniLM-L6-v2')

# UDF function for generating sentence embeddings 
def generate_embedding(text: str) -> Any:
    """
    Generate a sentence embedding for a given text using Sentence-BERT.
    Inout: text (str): Input text.
    Output: numpy.ndarray: The generated sentence embedding.
    """
    return model.encode(text)
