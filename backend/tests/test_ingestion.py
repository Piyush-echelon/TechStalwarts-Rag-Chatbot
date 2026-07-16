"""
Tests for the PDF ingestion pipeline (loader + chunker).

All tests are pure unit tests — no network, no LLM, no vector store.
"""

from __future__ import annotations

import pytest
from langchain_core.documents import Document


# ═══════════════════════════════════════════════════════════════════════════════
# PDF Loader
# ═══════════════════════════════════════════════════════════════════════════════


class TestPDFLoader:
    def test_load_returns_documents(self, sample_documents):
        """Loader should return at least one Document per non-empty page."""
        assert len(sample_documents) >= 1

    def test_load_page_count(self, sample_documents):
        """Sample PDF has 3 pages — loader must emit 3 documents."""
        assert len(sample_documents) == 3

    def test_documents_have_content(self, sample_documents):
        """Every document must have non-empty page_content."""
        for doc in sample_documents:
            assert doc.page_content.strip(), "page_content must not be empty"

    def test_documents_have_page_number_metadata(self, sample_documents):
        """page_number metadata must be present and 1-based."""
        for i, doc in enumerate(sample_documents):
            assert "page_number" in doc.metadata
            assert doc.metadata["page_number"] == i + 1

    def test_documents_have_filename_metadata(self, sample_documents):
        """filename metadata must be present and end with .pdf."""
        for doc in sample_documents:
            assert "filename" in doc.metadata
            assert doc.metadata["filename"].endswith(".pdf")

    def test_documents_have_total_pages_metadata(self, sample_documents):
        """total_pages metadata must equal 3 for the sample PDF."""
        for doc in sample_documents:
            assert doc.metadata.get("total_pages") == 3

    def test_load_raises_on_missing_file(self, tmp_path):
        """Loader must raise FileNotFoundError for non-existent paths."""
        from app.ingestion.pdf_loader import load_pdf

        with pytest.raises(FileNotFoundError):
            load_pdf(tmp_path / "does_not_exist.pdf")

    def test_content_contains_expected_text(self, sample_documents):
        """Page 1 content should reference 'Artificial Intelligence'."""
        page1 = sample_documents[0]
        assert "Artificial Intelligence" in page1.page_content


# ═══════════════════════════════════════════════════════════════════════════════
# Chunker
# ═══════════════════════════════════════════════════════════════════════════════


class TestChunker:
    def test_chunk_produces_output(self, sample_chunks):
        """Chunking 3 pages must produce at least 1 chunk."""
        assert len(sample_chunks) > 0

    def test_chunks_are_within_size_limit(self, sample_chunks):
        """No chunk should exceed 2× the configured chunk_size (300 chars)."""
        max_allowed = 300 * 2  # generous upper bound accounting for overlap
        for chunk in sample_chunks:
            assert len(chunk.page_content) <= max_allowed, (
                f"Chunk too large: {len(chunk.page_content)} chars"
            )

    def test_chunks_inherit_page_number(self, sample_chunks):
        """Each chunk must carry the page_number from its parent document."""
        for chunk in sample_chunks:
            assert "page_number" in chunk.metadata
            assert isinstance(chunk.metadata["page_number"], int)

    def test_chunks_have_chunk_index(self, sample_chunks):
        """Each chunk must have a chunk_index field."""
        for chunk in sample_chunks:
            assert "chunk_index" in chunk.metadata
            assert chunk.metadata["chunk_index"] >= 0

    def test_chunks_have_chunk_hash(self, sample_chunks):
        """Each chunk must have a chunk_hash for deduplication."""
        for chunk in sample_chunks:
            assert "chunk_hash" in chunk.metadata
            assert len(chunk.metadata["chunk_hash"]) == 16  # MD5 short hash

    def test_chunk_hashes_are_unique(self, sample_chunks):
        """All chunk_hash values must be unique (no duplicates slipped through)."""
        hashes = [c.metadata["chunk_hash"] for c in sample_chunks]
        assert len(hashes) == len(set(hashes)), "Duplicate chunk hashes detected"

    def test_chunker_respects_chunk_size(self, sample_documents):
        """Chunker with explicit size=100 should produce more chunks than size=800."""
        from app.ingestion.chunker import chunk_documents

        small_chunks = chunk_documents(sample_documents, chunk_size=100, chunk_overlap=10)
        large_chunks = chunk_documents(sample_documents, chunk_size=800, chunk_overlap=50)
        assert len(small_chunks) >= len(large_chunks), (
            "Smaller chunk_size should produce >= chunks vs larger chunk_size"
        )

    def test_empty_documents_returns_empty_list(self):
        """Chunking an empty list must return an empty list without errors."""
        from app.ingestion.chunker import chunk_documents

        result = chunk_documents([])
        assert result == []
