"""
Document Chunker — splits LangChain Documents into overlapping text chunks.

Uses :class:`langchain.text_splitter.RecursiveCharacterTextSplitter` with
semantically aware separators.  Each chunk retains the page-level metadata
from its parent document and gains additional chunk-level fields.
"""

from __future__ import annotations

import hashlib
from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from app.config import settings
from app.utils import get_logger

logger = get_logger(__name__)

# Preferred split boundaries — ordered from coarsest to finest grain.
_SEPARATORS = ["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""]


def _content_hash(text: str) -> str:
    """Return a short MD5 hex digest of *text* for deduplication."""
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:16]


def chunk_documents(
    documents: List[Document],
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> List[Document]:
    """
    Split a list of :class:`Document` objects into overlapping chunks.

    Each output chunk preserves the parent document's metadata and adds:

    - ``chunk_index``  – 0-based index within the **same parent document**
    - ``chunk_hash``   – short MD5 digest of the chunk content (for dedup)

    Args:
        documents:     Source documents (typically one per PDF page).
        chunk_size:    Target chunk size in characters.  Defaults to the
                       value from :mod:`app.config`.
        chunk_overlap: Overlap between consecutive chunks.  Defaults to the
                       value from :mod:`app.config`.

    Returns:
        A flat list of chunked :class:`Document` objects.
    """
    size = chunk_size or settings.chunk_size
    overlap = chunk_overlap or settings.chunk_overlap

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=overlap,
        separators=_SEPARATORS,
        length_function=len,
        is_separator_regex=False,
    )

    all_chunks: List[Document] = []
    seen_hashes: set[str] = set()

    for doc in documents:
        raw_chunks = splitter.split_documents([doc])
        chunk_idx = 0

        for chunk in raw_chunks:
            content_hash = _content_hash(chunk.page_content)

            # Skip exact duplicates (can appear near chunk boundaries)
            if content_hash in seen_hashes:
                logger.debug("Duplicate chunk skipped (hash=%s)", content_hash)
                continue
            seen_hashes.add(content_hash)

            # Enrich metadata
            chunk.metadata = {
                **doc.metadata,        # inherit parent metadata (page, filename…)
                "chunk_index": chunk_idx,
                "chunk_hash": content_hash,
            }
            all_chunks.append(chunk)
            chunk_idx += 1

    logger.info(
        "Chunked %d documents → %d chunks  (size=%d, overlap=%d)",
        len(documents),
        len(all_chunks),
        size,
        overlap,
    )
    return all_chunks
