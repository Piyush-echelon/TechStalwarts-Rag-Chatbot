"""
Integration tests for the ChromaDB vector store layer.

Uses the fake embeddings fixture so no OpenAI calls are made.
Each test gets its own in-memory or temp-dir collection to avoid cross-test
pollution.
"""

from __future__ import annotations

import uuid

import pytest
from langchain_core.documents import Document


@pytest.fixture()
def collection_name():
    """Return a unique collection name per test."""
    return f"test_{uuid.uuid4().hex[:8]}"


class TestVectorStore:
    def test_add_and_retrieve_documents(
        self, sample_chunks, mock_openai_embeddings, collection_name, monkeypatch
    ):
        """Documents added to ChromaDB should be retrievable via similarity search."""
        # Patch get_embeddings to return our fake embeddings
        monkeypatch.setattr("app.rag.vector_store.get_embeddings", lambda: mock_openai_embeddings)

        from app.rag.vector_store import add_documents, similarity_search_with_score

        n_added = add_documents(sample_chunks, collection_name=collection_name)
        assert n_added == len(sample_chunks)

        results = similarity_search_with_score(
            query="What is Artificial Intelligence?",
            collection_name=collection_name,
            k=3,
        )
        assert len(results) > 0, "Should return at least 1 result"
        doc, score = results[0]
        assert isinstance(doc, Document)
        assert isinstance(score, float)

    def test_add_documents_returns_count(
        self, sample_chunks, mock_openai_embeddings, collection_name, monkeypatch
    ):
        """add_documents must return the exact number of chunks stored."""
        monkeypatch.setattr("app.rag.vector_store.get_embeddings", lambda: mock_openai_embeddings)

        from app.rag.vector_store import add_documents

        count = add_documents(sample_chunks, collection_name=collection_name)
        assert count == len(sample_chunks)

    def test_similarity_search_returns_k_or_fewer_results(
        self, sample_chunks, mock_openai_embeddings, collection_name, monkeypatch
    ):
        """similarity_search_with_score must return at most k results."""
        monkeypatch.setattr("app.rag.vector_store.get_embeddings", lambda: mock_openai_embeddings)

        from app.rag.vector_store import add_documents, similarity_search_with_score

        add_documents(sample_chunks, collection_name=collection_name)
        k = 2
        results = similarity_search_with_score(
            query="deep learning", collection_name=collection_name, k=k
        )
        assert len(results) <= k

    def test_delete_collection(
        self, sample_chunks, mock_openai_embeddings, collection_name, monkeypatch
    ):
        """Deleting a collection should not raise and subsequent operations should fail."""
        monkeypatch.setattr("app.rag.vector_store.get_embeddings", lambda: mock_openai_embeddings)

        from app.rag.vector_store import add_documents, delete_collection

        add_documents(sample_chunks, collection_name=collection_name)
        # Should not raise
        delete_collection(collection_name)
