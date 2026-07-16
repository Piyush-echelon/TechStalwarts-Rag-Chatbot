"""
Test fixtures and shared utilities.

Includes:
- A sample PDF created in-memory with fpdf2 (no network calls).
- A mock Groq API key so pydantic-settings initialises Settings.
- A pre-built list of LangChain Documents for unit tests.
- A fake embeddings fixture that bypasses the TF-IDF vectorizer.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import List

import pytest
from fpdf import FPDF
from langchain_core.documents import Document

# ── Patch env so Settings initialises without a real API key ─────────────────
os.environ.setdefault("GROQ_API_KEY", "gsk_test_0000000000000000000000000000000000000000000000000000")


# ── Sample PDF fixture ────────────────────────────────────────────────────────

SAMPLE_TEXT_PAGE_1 = (
    "Artificial Intelligence (AI) is the simulation of human intelligence processes "
    "by computer systems. These processes include learning, reasoning, and self-correction. "
    "AI applications include expert systems, natural language processing, and computer vision."
)

SAMPLE_TEXT_PAGE_2 = (
    "Machine Learning is a subset of AI that gives systems the ability to automatically "
    "learn and improve from experience without being explicitly programmed. "
    "Deep learning uses neural networks with many layers to analyse various factors of data."
)

SAMPLE_TEXT_PAGE_3 = (
    "Large Language Models (LLMs) are neural networks trained on massive text datasets. "
    "They can generate coherent text, answer questions, and perform translation. "
    "Examples include GPT-4, Claude, and Gemini."
)


@pytest.fixture(scope="session")
def sample_pdf_path(tmp_path_factory) -> Path:
    """
    Create a minimal 3-page PDF and return its path.

    The PDF is created once per test session to save time.
    """
    pdf = FPDF()
    for page_text in [SAMPLE_TEXT_PAGE_1, SAMPLE_TEXT_PAGE_2, SAMPLE_TEXT_PAGE_3]:
        pdf.add_page()
        pdf.set_font("Helvetica", size=12)
        pdf.multi_cell(0, 10, page_text)

    tmp_dir = tmp_path_factory.mktemp("pdfs")
    pdf_path = tmp_dir / "sample.pdf"
    pdf.output(str(pdf_path))
    return pdf_path


@pytest.fixture(scope="session")
def sample_documents(sample_pdf_path) -> List[Document]:
    """Load documents from the sample PDF fixture."""
    from app.ingestion.pdf_loader import load_pdf
    return load_pdf(sample_pdf_path)


@pytest.fixture(scope="session")
def sample_chunks(sample_documents) -> List[Document]:
    """Chunk the sample documents."""
    from app.ingestion.chunker import chunk_documents
    return chunk_documents(sample_documents, chunk_size=300, chunk_overlap=50)


@pytest.fixture()
def mock_embeddings(monkeypatch):
    """
    Replace TFIDFEmbeddings with a deterministic fake that returns
    stable 128-dim vectors without fitting on any corpus.
    """
    import hashlib

    class _FakeEmbeddings:
        def fit(self, texts):
            return self

        def embed_documents(self, texts: List[str]) -> List[List[float]]:
            return [self._text_to_vec(t) for t in texts]

        def embed_query(self, text: str) -> List[float]:
            return self._text_to_vec(text)

        @staticmethod
        def _text_to_vec(text: str) -> List[float]:
            """Stable 128-dim float vector derived from text hash."""
            digest = hashlib.sha256(text.encode()).digest()
            raw = list(digest) * (128 // len(digest) + 1)
            return [b / 255.0 for b in raw[:128]]

    fake = _FakeEmbeddings()

    monkeypatch.setattr("app.rag.embeddings.TFIDFEmbeddings", lambda: fake)
    monkeypatch.setattr("app.rag.embeddings.get_embeddings", lambda collection_name="_default": fake)
    monkeypatch.setattr("app.rag.embeddings.create_and_fit_embeddings", lambda texts, collection_name: fake)
    return fake


# Backwards-compatible alias
@pytest.fixture()
def mock_openai_embeddings(mock_embeddings):
    return mock_embeddings
