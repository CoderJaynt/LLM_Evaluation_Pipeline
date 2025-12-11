import numpy as np

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors.
    Returns a float between -1 and 1.
    """
    if vec1.ndim > 1:
        vec1 = vec1.flatten()
    if vec2.ndim > 1:
        vec2 = vec2.flatten()

    dot = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return float(dot / (norm1 * norm2))


def relevance_score(query_emb: np.ndarray, answer_emb: np.ndarray) -> float:
    """
    Measure how relevant the LLM answer is to the user query.
    High = good.
    """
    return cosine_similarity(query_emb, answer_emb)


def completeness_score(answer_emb: np.ndarray, context_emb: np.ndarray) -> float:
    """
    Measure how well the answer matches the full context.
    High = good (answer covers the main ideas).
    """
    return cosine_similarity(answer_emb, context_emb)


def hallucination_score(answer_emb: np.ndarray, context_chunk_embs: np.ndarray) -> float:
    """
    Measures hallucination.
    Idea:
        - Compare answer with each context chunk.
        - Find the highest similarity.
        - If the highest similarity is low → answer is not grounded → hallucination.

    Returns:
        A score between 0 and 1.
        High score = high hallucination (bad).
        Low score  = low hallucination (good).
    """
    similarities = []

    for chunk_emb in context_chunk_embs:
        sim = cosine_similarity(answer_emb, chunk_emb)
        similarities.append(sim)

    max_sim = max(similarities) if similarities else 0
    
    hallucination = 1 - max_sim

    return float(max(0, min(1, hallucination)))
