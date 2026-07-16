"""
LangGraph RAG Graph — defines the stateful RAG pipeline.

Graph topology
--------------

    START → retrieve → generate → END

- ``retrieve``:  Uses the session's :class:`HybridRetriever` to fetch the
  top-*k* relevant chunks.
- ``generate``:  Composes the prompt, calls Groq's LLM with ``streaming=True``
  so that ``astream_events`` propagates token-level events to the SSE endpoint.

The graph is compiled once per session and stored in the session registry.
"""

from __future__ import annotations

from typing import List, Optional, TypedDict

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langgraph.graph import END, START, StateGraph

from app.config import settings
from app.rag.prompts import SYSTEM_PROMPT, format_context
from app.rag.retriever import HybridRetriever
from app.utils import get_logger

logger = get_logger(__name__)

# ── State definition ──────────────────────────────────────────────────────────


class RAGState(TypedDict):
    """Typed state flowing through the LangGraph nodes."""

    query: str
    session_id: str
    retrieved_chunks: List[Document]
    answer: str
    sources: List[dict]
    error: Optional[str]


# ── Node factories ────────────────────────────────────────────────────────────


def _make_retrieve_node(retriever: HybridRetriever):
    """Return a retrieval node bound to *retriever*."""

    def retrieve_node(state: RAGState) -> dict:
        logger.info(
            "[%s] Retrieving for query: %r", state["session_id"], state["query"][:80]
        )
        try:
            chunks = retriever.retrieve(state["query"])
            return {"retrieved_chunks": chunks, "error": None}
        except Exception as exc:
            logger.exception("Retrieval failed: %s", exc)
            return {"retrieved_chunks": [], "error": str(exc)}

    return retrieve_node


def _make_generate_node():
    """Return a generation node that streams tokens via Groq."""

    llm = ChatGroq(
        model=settings.llm_model,
        api_key=settings.groq_api_key,
        temperature=0,
        streaming=True,  # Required for astream_events to emit token events
    )

    async def generate_node(state: RAGState) -> dict:
        chunks = state.get("retrieved_chunks", [])

        # Graceful refusal when no relevant context was retrieved
        if not chunks or state.get("error"):
            refusal = (
                "I don't have enough information in the provided document to "
                "answer this question. Please try rephrasing or ask about a "
                "topic covered in the document."
            )
            logger.info(
                "[%s] No relevant chunks — returning refusal.", state["session_id"]
            )
            return {"answer": refusal, "sources": []}

        context = format_context(chunks)
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(
                content=f"Context:\n\n{context}\n\nQuestion: {state['query']}"
            ),
        ]

        logger.info(
            "[%s] Generating answer with %d chunks via Groq (%s).",
            state["session_id"],
            len(chunks),
            settings.llm_model,
        )

        # Consume async stream — tokens are emitted via astream_events
        full_content = ""
        async for token in llm.astream(messages):
            full_content += token.content

        # Build source citations (deduplicate by page)
        sources = []
        seen_pages: set = set()
        for chunk in chunks:
            page = chunk.metadata.get("page_number", "?")
            fname = chunk.metadata.get("filename", "document")
            key = (fname, page)
            if key not in seen_pages:
                seen_pages.add(key)
                sources.append(
                    {
                        "page_number": page,
                        "filename": fname,
                        "content_preview": chunk.page_content[:300],
                    }
                )

        return {"answer": full_content, "sources": sources}

    return generate_node


# ── Graph builder ─────────────────────────────────────────────────────────────


def build_rag_graph(retriever: HybridRetriever):
    """
    Compile and return the LangGraph RAG graph for a session.

    Args:
        retriever: A :class:`HybridRetriever` pre-loaded with the session's
                   document chunks.

    Returns:
        A compiled :class:`CompiledStateGraph` ready for ``ainvoke`` /
        ``astream_events``.
    """
    graph: StateGraph = StateGraph(RAGState)

    graph.add_node("retrieve", _make_retrieve_node(retriever))
    graph.add_node("generate", _make_generate_node())

    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)

    compiled = graph.compile()
    logger.info("RAG graph compiled (LLM: %s).", settings.llm_model)
    return compiled
