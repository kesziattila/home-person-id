"""Embedding math shared by face recognition, Re-ID and unidentified-face dedup."""

import numpy as np

_EPS = 1e-8


def l2_normalize(x: np.ndarray) -> np.ndarray:
    """L2-normalize a vector, or each row of a 2-D array."""
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + _EPS)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    return float(np.dot(l2_normalize(a), l2_normalize(b)))


def cosine_similarity_batch(query: np.ndarray, gallery: np.ndarray) -> np.ndarray:
    """Cosine similarity of one query vector against each row of `gallery` (N, D)."""
    return l2_normalize(np.asarray(gallery)) @ l2_normalize(query)
