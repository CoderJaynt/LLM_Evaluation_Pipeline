

from typing import List, Optional
import numpy as np
from functools import lru_cache

@lru_cache(maxsize=1)
def _load_sentence_transformer(model_name: str = "all-MiniLM-L6-v2"):
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as e:
        raise ImportError(
            "sentence-transformers is required. Install with: pip install sentence-transformers"
        ) from e

    model = SentenceTransformer(model_name)
    return model


def load_model(model_name: str = "all-MiniLM-L6-v2"):
    """
    Load (or fetch cached) sentence-transformers model.

    Args:
        model_name: HF / sentence-transformers model name string.

    Returns:
        SentenceTransformer model instance
    """
    return _load_sentence_transformer(model_name)


def embed_texts(texts: List[str], model=None) -> np.ndarray:
    """
    Embed a list of texts.

    Args:
        texts: List of strings to embed.
        model: optional preloaded model (returned by load_model).

    Returns:
        2D numpy array of shape (len(texts), embedding_dim)
    """
    if model is None:
        model = load_model()

    if not isinstance(texts, list):
        raise ValueError("texts must be a list of strings")

    embeddings = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    return np.asarray(embeddings)


def embed_text(text: str, model=None) -> np.ndarray:
    """
    Embed a single string and return a 1D numpy array.

    Args:
        text: input string
        model: optional preloaded model

    Returns:
        1D numpy array
    """
    if model is None:
        model = load_model()

    emb = model.encode([text], show_progress_bar=False, convert_to_numpy=True)
    return np.asarray(emb[0])
