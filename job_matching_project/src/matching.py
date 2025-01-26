# Import Libraries
import numpy as np
from typing import List
from sklearn.metrics.pairwise import cosine_similarity

# UDF function for similarity calculation
def calculate_similarity(job_embedding: np.ndarray, resume_embeddings: List[np.ndarray]) -> List[float]:
    """
    Calculate cosine similarity between a job embedding and multiple resume embeddings.

    Inputs: job_embedding (np.ndarray): Embedding of the job description.
            resume_embeddings (List[np.ndarray]): List of resume embeddings.
    Outputs: List[float]: List of similarity scores.
    """
    return cosine_similarity([job_embedding], resume_embeddings).flatten().tolist()
