"""
Vector Store — lightweight in-memory cosine-similarity store using numpy.

Replaces ChromaDB to eliminate the ~400 MB native dependency footprint
(onnxruntime, sqlite3 extensions) that exceeds Vercel's 500 MB limit.

Each PDF upload creates its own isolated in-memory collection identified by
the session UUID, so multiple documents never pollute each other's search results.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from langchain_core.documents import Document

from app.rag.embeddings import create_and_fit_embeddings, delete_embeddings, get_embeddings
from app.utils import get_logger

logger = get_logger(__name__)

# Module-level store: { collection_name: { "docs": [...], "matrix": np.ndarray } }
_store: Dict[str, Dict] = {}


def _cosine_similarity(query_vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Return cosine similarity of query_vec against each row of matrix."""
    # Vectors are already L2-normalised by TFIDFEmbeddings
    return matrix @ query_vec


def get_vector_store(collection_name: str) -> Dict:
    """Return the raw store dict for a collection (used only for compatibility)."""
    return _store.get(collection_name, {})


def add_documents(chunks: List[Document], collection_name: str) -> int:
    """
    Embed *chunks* and store them in an in-memory collection.

    Args:
        chunks:          Pre-split :class:`Document` objects.
        collection_name: Target collection (session UUID).

    Returns:
        Number of chunks stored.
    """
    texts = [c.page_content for c in chunks]

    # Fit a fresh TF-IDF model on this session's corpus, then embed
    embeddings = create_and_fit_embeddings(texts, collection_name)
    vectors = embeddings.embed_documents(texts)
    matrix = np.array(vectors, dtype=np.float32)

    _store[collection_name] = {
        "docs": list(chunks),
        "matrix": matrix,
    }
    logger.info("Stored %d chunks → collection '%s'", len(chunks), collection_name)
    return len(chunks)


def similarity_search_with_score(
    query: str,
    collection_name: str,
    k: int = 10,
) -> List[Tuple[Document, float]]:
    """
    Dense cosine-similarity search against *collection_name*.

    Args:
        query:           Raw query string (will be embedded internally).
        collection_name: Target collection.
        k:               Max results to return.

    Returns:
        List of ``(Document, similarity_score)`` pairs, best first.
    """
    collection = _store.get(collection_name)
    if not collection:
        logger.warning("Collection '%s' not found in store", collection_name)
        return []

    embeddings = get_embeddings(collection_name)
    query_vec = np.array(embeddings.embed_query(query), dtype=np.float32)
    scores = _cosine_similarity(query_vec, collection["matrix"])

    top_k = min(k, len(scores))
    top_indices = np.argpartition(scores, -top_k)[-top_k:]
    top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

    results = [
        (collection["docs"][i], float(scores[i]))
        for i in top_indices
    ]
    logger.debug("Dense search returned %d results for query: %r", len(results), query[:60])
    return results


def delete_collection(collection_name: str) -> None:
    """Remove a collection from memory."""
    if collection_name in _store:
        del _store[collection_name]
        logger.info("Deleted collection '%s'", collection_name)
    else:
        logger.warning("Could not delete collection '%s': not found", collection_name)
    delete_embeddings(collection_name)
