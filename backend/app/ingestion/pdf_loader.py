"""
PDF Loader — extracts text and metadata from PDF files using PyMuPDF.

Each page is returned as a :class:`langchain_core.documents.Document` so the
output integrates directly with the LangChain / LangGraph ecosystem.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import fitz  # PyMuPDF
from langchain_core.documents import Document

from app.utils import get_logger

logger = get_logger(__name__)


def load_pdf(file_path: str | Path) -> List[Document]:
    """
    Load a PDF file and return one :class:`Document` per page.

    Metadata included per document:
    - ``page_number``   – 1-based page index
    - ``total_pages``   – total page count of the PDF
    - ``source``        – absolute path to the file
    - ``filename``      – base name of the file

    Args:
        file_path: Absolute or relative path to the PDF.

    Returns:
        A list of :class:`Document` objects, one per non-empty page.

    Raises:
        FileNotFoundError: If *file_path* does not exist.
        ValueError: If the PDF contains no extractable text.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    logger.info("Loading PDF: %s", path.name)

    documents: List[Document] = []

    with fitz.open(str(path)) as pdf:
        total_pages = len(pdf)
        logger.debug("Total pages: %d", total_pages)

        for page_num, page in enumerate(pdf, start=1):
            text = page.get_text("text")  # plain text extraction
            cleaned = text.strip()

            if not cleaned:
                logger.debug("Page %d is empty — skipping.", page_num)
                continue

            doc = Document(
                page_content=cleaned,
                metadata={
                    "page_number": page_num,
                    "total_pages": total_pages,
                    "source": str(path.resolve()),
                    "filename": path.name,
                },
            )
            documents.append(doc)

    if not documents:
        raise ValueError(
            f"No extractable text found in '{path.name}'. "
            "The PDF may be scanned or image-only."
        )

    logger.info("Extracted %d pages from '%s'", len(documents), path.name)
    return documents
