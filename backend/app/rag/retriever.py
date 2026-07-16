"""
Hybrid Retriever — combines dense (ChromaDB) and sparse (BM25) search via
Reciprocal Rank Fusion (RRF).

Strategy
--------
1. **Dense retrieval**: cosine-similarity search against the ChromaDB
   collection for the session.
2. **Sparse retrieval**: BM25 keyword search over the same chunk corpus
   (built in-memory at ingestion time).
3. **RRF fusion**: scores from both ranked lists are combined with the
   standard RRF formula  ``score = 1 / (rank + k)`` where ``k=60``.
4. The top-*k* documents after fusion are returned.

This approach improves recall for both semantic and keyword-heavy queries.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from app.config import settings
from app.rag.vector_store import similarity_search_with_score
from app.utils import get_logger

logger = get_logger(__name__)

_RRF_K = 60  # Standard RRF constant — higher values dampen high-rank rewards


class HybridRetriever:
    """
    Hybrid retriever with dense + BM25 fusion.

    Args:
        collection_name: ChromaDB collection for this session.
        documents:       All document chunks ingested in this session
                         (used to build the BM25 index).
        top_k:           Number of final results to return.
    """

    def __init__(
        self,
        collection_name: str,
        documents: List[Document],
        top_k: int | None = None,
    ) -> None:
        self.collection_name = collection_name
        self.documents = documents
        self.top_k = top_k or settings.retrieval_top_k

        # Build BM25 index over tokenised chunk content
        tokenised = [doc.page_content.lower().split() for doc in documents]
        self._bm25 = BM25Okapi(tokenised) if tokenised else None
        logger.info(
            "HybridRetriever ready — %d chunks, collection=%s",
            len(documents),
            collection_name,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(self, query: str) -> List[Document]:
        """
        Retrieve the top-*k* most relevant chunks for *query*.

        Args:
            query: Raw user question.

        Returns:
            Ordered list of :class:`Document` objects (most relevant first).
        """
        candidate_k = self.top_k * 3  # Retrieve extra candidates for fusion

        # ── Dense retrieval ──────────────────────────────────────────────────
        dense_results: List[Tuple[Document, float]] = similarity_search_with_score(
            query=query,
            collection_name=self.collection_name,
            k=candidate_k,
        )
        # Map chunk_hash → (rank, document)
        dense_ranking: Dict[str, Tuple[int, Document]] = {
            doc.metadata.get("chunk_hash", str(i)): (i, doc)
            for i, (doc, _score) in enumerate(dense_results)
        }

        # ── Sparse BM25 retrieval ────────────────────────────────────────────
        sparse_ranking: Dict[str, int] = {}
        if self._bm25 and self.documents:
            bm25_scores = self._bm25.get_scores(query.lower().split())
            sorted_indices = sorted(
                range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True
            )
            for rank, idx in enumerate(sorted_indices[:candidate_k]):
                h = self.documents[idx].metadata.get("chunk_hash", str(idx))
                sparse_ranking[h] = rank

        # ── Reciprocal Rank Fusion ───────────────────────────────────────────
        rrf_scores: Dict[str, float] = {}

        for chunk_hash, (rank, _) in dense_ranking.items():
            rrf_scores[chunk_hash] = rrf_scores.get(chunk_hash, 0.0) + 1.0 / (
                rank + _RRF_K
            )

        for chunk_hash, rank in sparse_ranking.items():
            rrf_scores[chunk_hash] = rrf_scores.get(chunk_hash, 0.0) + 1.0 / (
                rank + _RRF_K
            )

        # Sort by fused score descending
        sorted_hashes = sorted(rrf_scores, key=lambda h: rrf_scores[h], reverse=True)

        # Resolve hashes back to documents (prefer dense results; fall back to BM25)
        hash_to_doc: Dict[str, Document] = {}
        for doc, _ in dense_results:
            h = doc.metadata.get("chunk_hash", "")
            hash_to_doc[h] = doc
        for i, doc in enumerate(self.documents):
            h = doc.metadata.get("chunk_hash", str(i))
            hash_to_doc.setdefault(h, doc)

        top_docs: List[Document] = []
        for h in sorted_hashes[: self.top_k]:
            if h in hash_to_doc:
                top_docs.append(hash_to_doc[h])

        logger.debug(
            "Hybrid retrieval: dense=%d, bm25=%d, fused top-%d",
            len(dense_ranking),
            len(sparse_ranking),
            len(top_docs),
        )
        return top_docs
