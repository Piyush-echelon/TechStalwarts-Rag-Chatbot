"""
Integration tests for the full RAG pipeline.

Strategy
--------
- The ChromaDB layer uses fake embeddings (no OpenAI calls).
- The LLM ``ChatOpenAI`` is mocked with ``unittest.mock.AsyncMock`` so no
  actual GPT calls are made.
- The tests validate:
  1. In-scope queries return an answer containing source citations.
  2. Out-of-scope queries trigger the graceful refusal phrase.
  3. The FastAPI endpoint responds correctly (status, streaming format).
"""

from __future__ import annotations

import json
import uuid
from typing import AsyncIterator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from langchain_core.messages import AIMessageChunk


# ── Helpers ───────────────────────────────────────────────────────────────────

REFUSAL_PHRASE = (
    "I don't have enough information in the provided document to answer this question."
)


async def _fake_astream(messages) -> AsyncIterator[AIMessageChunk]:
    """Yield a single AIMessageChunk that looks like a real streamed response."""
    answer = (
        "AI stands for Artificial Intelligence, which is the simulation of "
        "human intelligence by computer systems. [Sources: Page 1]"
    )
    for word in answer.split():
        yield AIMessageChunk(content=word + " ")


async def _fake_astream_empty(messages) -> AsyncIterator[AIMessageChunk]:
    """Simulate a model that says it has no context (empty doc scenario)."""
    yield AIMessageChunk(content=REFUSAL_PHRASE)


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture()
def unique_session_id():
    return f"test-{uuid.uuid4().hex[:8]}"


@pytest.fixture()
def patched_llm_astream():
    """Patch ChatGroq.astream with the fake streamer."""
    with patch(
        "app.rag.graph.ChatGroq",
        return_value=MagicMock(astream=MagicMock(side_effect=_fake_astream)),
    ) as mock_cls:
        yield mock_cls


# ── HybridRetriever unit test ─────────────────────────────────────────────────


class TestHybridRetriever:
    def test_retriever_returns_documents(
        self, sample_chunks, mock_openai_embeddings, monkeypatch
    ):
        """Retriever must return a non-empty list for a relevant query."""
        monkeypatch.setattr("app.rag.vector_store.get_embeddings", lambda: mock_openai_embeddings)

        col = f"test_{uuid.uuid4().hex[:8]}"
        from app.rag.vector_store import add_documents
        from app.rag.retriever import HybridRetriever

        add_documents(sample_chunks, collection_name=col)
        retriever = HybridRetriever(
            collection_name=col, documents=sample_chunks, top_k=3
        )
        results = retriever.retrieve("Artificial Intelligence")
        assert len(results) > 0
        assert all(hasattr(r, "page_content") for r in results)

    def test_retriever_returns_at_most_top_k(
        self, sample_chunks, mock_openai_embeddings, monkeypatch
    ):
        """Retriever must not return more than top_k results."""
        monkeypatch.setattr("app.rag.vector_store.get_embeddings", lambda: mock_openai_embeddings)

        col = f"test_{uuid.uuid4().hex[:8]}"
        from app.rag.vector_store import add_documents
        from app.rag.retriever import HybridRetriever

        add_documents(sample_chunks, collection_name=col)
        top_k = 2
        retriever = HybridRetriever(
            collection_name=col, documents=sample_chunks, top_k=top_k
        )
        results = retriever.retrieve("machine learning")
        assert len(results) <= top_k


# ── End-to-end pipeline with mocked LLM ──────────────────────────────────────


@pytest.mark.asyncio
class TestRAGPipeline:
    async def test_full_pipeline_returns_answer_with_sources(
        self, sample_chunks, mock_openai_embeddings, monkeypatch
    ):
        """Full pipeline: retrieve → generate → answer contains source info."""
        monkeypatch.setattr("app.rag.vector_store.get_embeddings", lambda: mock_openai_embeddings)
        monkeypatch.setattr("app.rag.graph.settings.groq_api_key", "gsk_fake")

        col = f"test_{uuid.uuid4().hex[:8]}"
        from app.rag.vector_store import add_documents
        from app.rag.retriever import HybridRetriever
        from app.rag.graph import build_rag_graph

        add_documents(sample_chunks, collection_name=col)
        retriever = HybridRetriever(collection_name=col, documents=sample_chunks, top_k=3)

        with patch(
            "app.rag.graph.ChatGroq",
            return_value=MagicMock(astream=MagicMock(side_effect=_fake_astream)),
        ):
            graph = build_rag_graph(retriever)
            result = await graph.ainvoke({
                "query": "What is Artificial Intelligence?",
                "session_id": "test-session",
                "retrieved_chunks": [],
                "answer": "",
                "sources": [],
                "error": None,
            })

        assert result["answer"], "Answer must not be empty"
        assert result["sources"], "Sources list must not be empty"
        # Verify at least one source has page_number
        assert any("page_number" in s for s in result["sources"])

    async def test_out_of_scope_query_returns_refusal(
        self, sample_chunks, mock_openai_embeddings, monkeypatch
    ):
        """
        When retrieved chunks are empty (simulated by returning empty list
        from retriever), the generate node should emit the refusal phrase.
        """
        monkeypatch.setattr("app.rag.vector_store.get_embeddings", lambda: mock_openai_embeddings)
        monkeypatch.setattr("app.rag.graph.settings.groq_api_key", "gsk_fake")

        col = f"test_{uuid.uuid4().hex[:8]}"
        from app.rag.vector_store import add_documents
        from app.rag.retriever import HybridRetriever
        from app.rag.graph import build_rag_graph

        add_documents(sample_chunks, collection_name=col)

        # Patch retriever.retrieve to return empty list
        retriever = HybridRetriever(collection_name=col, documents=sample_chunks, top_k=3)
        retriever.retrieve = lambda q: []  # Force no results

        with patch(
            "app.rag.graph.ChatGroq",
            return_value=MagicMock(astream=MagicMock(side_effect=_fake_astream_empty)),
        ):
            graph = build_rag_graph(retriever)
            result = await graph.ainvoke({
                "query": "What is the weather on Mars?",
                "session_id": "test-out-of-scope",
                "retrieved_chunks": [],
                "answer": "",
                "sources": [],
                "error": None,
            })

        assert REFUSAL_PHRASE in result["answer"], (
            f"Expected refusal phrase in answer, got: {result['answer'][:200]}"
        )
        assert result["sources"] == []


# ── FastAPI endpoint smoke tests ──────────────────────────────────────────────


@pytest.mark.asyncio
class TestAPIEndpoints:
    async def test_health_check(self):
        """GET /api/health should return 200 with status=healthy."""
        from app.main import app

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/api/health")

        assert response.status_code == 200
        assert response.json()["status"] == "healthy"

    async def test_ingest_rejects_non_pdf(self):
        """POST /api/ingest with a non-PDF file should return 400."""
        from app.main import app

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post(
                "/api/ingest",
                files={"file": ("test.txt", b"hello world", "text/plain")},
            )

        assert response.status_code == 400

    async def test_chat_stream_404_on_unknown_session(self):
        """POST /api/chat/stream with unknown session_id should return 404."""
        from app.main import app

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post(
                "/api/chat/stream",
                json={"query": "hello", "session_id": "nonexistent-session"},
            )

        assert response.status_code == 404
