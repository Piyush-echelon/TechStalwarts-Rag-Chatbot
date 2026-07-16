"""
LLM Prompt Templates — system prompt and context formatter.

Centralising prompts here keeps them version-controlled, reviewable,
and easy to iterate without touching orchestration code.
"""

from __future__ import annotations

from typing import List

from langchain_core.documents import Document

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """\
You are an expert document assistant with access to specific context extracted \
from a user-uploaded PDF. Your only knowledge source is the provided context.

STRICT RULES:
1. Answer ONLY from the provided context — never use outside knowledge.
2. If the context does not contain enough information to answer the question, \
respond EXACTLY with the refusal phrase below and nothing else.
3. Always cite the exact page number(s) you drew the answer from, at the end \
of your response in the format: [Sources: Page X, Page Y].
4. Be concise, accurate, and factual.
5. If the user asks something that is clearly outside the document's scope \
(e.g., a general knowledge question), politely note that you can only answer \
questions about the uploaded document.

REFUSAL PHRASE (use verbatim when context is insufficient):
"I don't have enough information in the provided document to answer this \
question. Please try rephrasing or ask about a topic covered in the document."
"""

# ---------------------------------------------------------------------------
# Context formatter
# ---------------------------------------------------------------------------

def format_context(chunks: List[Document]) -> str:
    """
    Build a structured context block from retrieved *chunks*.

    Each chunk is prefixed with its page number so the LLM can accurately
    cite sources in its response.

    Args:
        chunks: Retrieved :class:`Document` objects from the vector store.

    Returns:
        A multi-line string ready to be injected into the user prompt.
    """
    if not chunks:
        return "(No relevant context found.)"

    parts: List[str] = []
    for i, chunk in enumerate(chunks, start=1):
        page = chunk.metadata.get("page_number", "?")
        filename = chunk.metadata.get("filename", "document")
        parts.append(
            f"[Chunk {i} | {filename} | Page {page}]\n{chunk.page_content.strip()}"
        )

    return "\n\n---\n\n".join(parts)
