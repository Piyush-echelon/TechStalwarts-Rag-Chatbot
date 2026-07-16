"""
Embeddings — lightweight TF-IDF vectorizer implemented with pure numpy.

Replaces Groq embeddings (model not publicly available) and FastEmbed (too
large for Vercel). This vectorizer:

- Builds a vocabulary from all ingested chunks at index time
- Computes TF-IDF weighted term vectors per document
- Produces L2-normalised float32 vectors suitable for cosine similarity

No external model downloads, no API keys, no native extensions required.
TF-IDF retrieval is highly effective for single-document RAG where queries
tend to share vocabulary with the source text.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

import numpy as np
from langchain_core.embeddings import Embeddings

from app.utils import get_logger

logger = get_logger(__name__)


def _tokenise(text: str) -> List[str]:
    """Lowercase, strip punctuation, split on whitespace."""
    return re.findall(r"[a-z0-9]+", text.lower())


class TFIDFEmbeddings(Embeddings):
    """
    In-process TF-IDF embedding model.

    Must be *fitted* on a corpus before use.  Call :meth:`fit` with all
    document texts once at ingestion time; the resulting instance can then
    embed both documents and queries against the same vocabulary.
    """

    def __init__(self) -> None:
        self._vocab: Dict[str, int] = {}      # term → column index
        self._idf: Optional[np.ndarray] = None  # shape (vocab_size,)
        self._fitted = False

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self, texts: List[str]) -> "TFIDFEmbeddings":
        """
        Build vocabulary and IDF weights from *texts*.

        Args:
            texts: All document chunk texts for a session.

        Returns:
            self (for chaining)
        """
        doc_freq: Counter = Counter()
        tokenised = [_tokenise(t) for t in texts]
        n_docs = len(tokenised)

        for tokens in tokenised:
            for term in set(tokens):
                doc_freq[term] += 1

        # Keep the top 8 192 terms by document frequency (caps memory)
        top_terms = [t for t, _ in doc_freq.most_common(8192)]
        self._vocab = {term: i for i, term in enumerate(top_terms)}

        idf_arr = np.zeros(len(self._vocab), dtype=np.float32)
        for term, idx in self._vocab.items():
            idf_arr[idx] = math.log((n_docs + 1) / (doc_freq[term] + 1)) + 1.0
        self._idf = idf_arr
        self._fitted = True
        logger.info(
            "TFIDFEmbeddings fitted: vocab=%d, n_docs=%d", len(self._vocab), n_docs
        )
        return self

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _vectorise(self, text: str) -> np.ndarray:
        """Return an L2-normalised TF-IDF vector for *text*."""
        if not self._fitted or self._idf is None:
            raise RuntimeError("TFIDFEmbeddings.fit() must be called before embed_*.")

        tokens = _tokenise(text)
        if not tokens:
            return np.zeros(len(self._vocab), dtype=np.float32)

        tf_counter = Counter(tokens)
        max_tf = max(tf_counter.values())
        vec = np.zeros(len(self._vocab), dtype=np.float32)
        for term, count in tf_counter.items():
            if term in self._vocab:
                tf = count / max_tf  # augmented TF
                vec[self._vocab[term]] = tf * self._idf[self._vocab[term]]

        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec

    # ------------------------------------------------------------------
    # LangChain Embeddings interface
    # ------------------------------------------------------------------

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._vectorise(t).tolist() for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._vectorise(text).tolist()


# Session-keyed cache: { collection_name: TFIDFEmbeddings }
_embeddings_cache: Dict[str, TFIDFEmbeddings] = {}


def get_embeddings(collection_name: str = "_default") -> TFIDFEmbeddings:
    """Return the fitted TFIDFEmbeddings for a given collection (session)."""
    return _embeddings_cache.get(collection_name, TFIDFEmbeddings())


def create_and_fit_embeddings(texts: List[str], collection_name: str) -> TFIDFEmbeddings:
    """
    Create, fit, cache, and return a :class:`TFIDFEmbeddings` for *collection_name*.

    Args:
        texts:           All document texts for this session.
        collection_name: Session UUID (used as cache key).

    Returns:
        The fitted :class:`TFIDFEmbeddings` instance.
    """
    emb = TFIDFEmbeddings()
    emb.fit(texts)
    _embeddings_cache[collection_name] = emb
    return emb


def delete_embeddings(collection_name: str) -> None:
    """Remove cached embeddings for a session."""
    _embeddings_cache.pop(collection_name, None)
