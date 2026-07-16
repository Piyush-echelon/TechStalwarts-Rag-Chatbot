"""
FastAPI Application — the HTTP + SSE interface for the RAG chatbot.

Endpoints
---------
POST  /api/ingest              Upload a PDF → ingest → return session_id
POST  /api/chat/stream         Stream a RAG answer via Server-Sent Events
GET   /api/sessions/{id}       Retrieve session metadata + history
DELETE /api/sessions/{id}      Tear down a session and its vector collection
GET   /api/health              Liveness probe
"""

from __future__ import annotations

import json
import os
import tempfile
import uuid
from typing import Any, Dict

import sys
# Ensure backend directory is in sys.path for Vercel/Docker import compatibility
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel

from app.config import settings
from app.ingestion.chunker import chunk_documents
from app.ingestion.pdf_loader import load_pdf
from app.rag.graph import build_rag_graph
from app.rag.retriever import HybridRetriever
from app.rag.vector_store import add_documents, delete_collection
from app.utils import get_logger

logger = get_logger(__name__)

# ── FastAPI app ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="Advanced RAG Chatbot API",
    description="LangGraph-powered Retrieval-Augmented Generation backend.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Global Error Handling ──────────────────────────────────────────────────────

from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    if isinstance(exc, StarletteHTTPException):
        logger.warning("HTTPException: %d - %s", exc.status_code, exc.detail)
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.detail}
        )
    if isinstance(exc, RequestValidationError):
        logger.warning("Validation Error: %s", exc.errors())
        return JSONResponse(
            status_code=422,
            content={"detail": exc.errors()}
        )
    
    logger.exception("Unhandled error occurred: %s", exc)
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal Server Error: {str(exc)}"}
    )

def validate_groq_key():
    key = settings.groq_api_key
    if not key or "xxxx" in key or key.strip() == "":
        raise HTTPException(
            status_code=400,
            detail="GROQ_API_KEY is not configured or is using the placeholder. Please set a valid Groq API key in your backend/.env file."
        )

# ── In-memory session registry ────────────────────────────────────────────────
# Structure: { session_id: { "graph": ..., "retriever": ..., "metadata": ..., "history": [...] } }
_sessions: Dict[str, Dict[str, Any]] = {}


# ── Request / Response models ─────────────────────────────────────────────────


class ChatRequest(BaseModel):
    query: str
    session_id: str


# ── Endpoints ─────────────────────────────────────────────────────────────────


@app.get("/api/health", tags=["System"])
async def health_check():
    """Liveness probe."""
    return {"status": "healthy", "version": "1.0.0"}


@app.post("/api/ingest", tags=["Ingestion"])
async def ingest_pdf(file: UploadFile = File(...)):
    """
    Accept a PDF upload, extract text, chunk it, embed and store in ChromaDB.

    Returns:
        JSON with ``session_id``, ``filename``, ``num_chunks``, ``num_pages``.
    """
    validate_groq_key()
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files (.pdf) are accepted.")

    session_id = str(uuid.uuid4())
    logger.info("Ingesting PDF '%s' → session %s", file.filename, session_id)

    # Save upload to a temp file
    suffix = ".pdf"
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=suffix)
    try:
        content = await file.read()
        with os.fdopen(tmp_fd, "wb") as f:
            f.write(content)

        # ── Pipeline: load → chunk → embed → store ─────────────────────────
        pages = load_pdf(tmp_path)
        chunks = chunk_documents(pages)

        if not chunks:
            raise HTTPException(status_code=422, detail="No text could be extracted from the PDF.")

        add_documents(chunks, collection_name=session_id)

        # Build retriever + LangGraph for this session
        retriever = HybridRetriever(
            collection_name=session_id,
            documents=chunks,
            top_k=settings.retrieval_top_k,
        )
        rag_graph = build_rag_graph(retriever)

        num_pages = max(
            (c.metadata.get("page_number", 0) for c in chunks), default=0
        )

        _sessions[session_id] = {
            "graph": rag_graph,
            "retriever": retriever,
            "filename": file.filename,
            "num_chunks": len(chunks),
            "num_pages": num_pages,
            "history": [],
        }

        logger.info(
            "Session %s ready: %d chunks, %d pages", session_id, len(chunks), num_pages
        )
        return {
            "session_id": session_id,
            "filename": file.filename,
            "num_chunks": len(chunks),
            "num_pages": num_pages,
        }

    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


@app.post("/api/chat/stream", tags=["Chat"])
async def chat_stream(request: ChatRequest):
    """
    Stream a RAG answer via Server-Sent Events (SSE).

    Event types emitted:
    - ``{"type": "token",   "content": "..."}``   — incremental LLM tokens
    - ``{"type": "sources", "sources": [...]}``   — source citations (at end)
    - ``{"type": "done"}``                        — stream complete
    - ``{"type": "error",   "message": "..."}``   — error occurred
    """
    validate_groq_key()
    session = _sessions.get(request.session_id)
    if not session:
        raise HTTPException(
            status_code=404,
            detail="Session not found. Please upload a PDF first.",
        )

    graph = session["graph"]

    initial_state = {
        "query": request.query,
        "session_id": request.session_id,
        "retrieved_chunks": [],
        "answer": "",
        "sources": [],
        "error": None,
    }

    async def event_generator():
        collected_answer = ""
        collected_sources: list = []

        try:
            async for event in graph.astream_events(initial_state, version="v2"):
                event_name = event.get("event", "")

                # Token-level streaming from the LLM
                if event_name == "on_chat_model_stream":
                    chunk = event["data"].get("chunk")
                    if chunk and chunk.content:
                        collected_answer += chunk.content
                        payload = json.dumps({"type": "token", "content": chunk.content})
                        yield f"data: {payload}\n\n"

                # Capture sources when the generate node completes
                elif event_name == "on_chain_end" and event.get("name") == "generate":
                    output = event["data"].get("output") or {}
                    collected_sources = output.get("sources", [])

            # Emit sources after streaming completes
            yield f"data: {json.dumps({'type': 'sources', 'sources': collected_sources})}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"

            # Persist to session history
            session["history"].append(
                {
                    "query": request.query,
                    "answer": collected_answer,
                    "sources": collected_sources,
                }
            )

        except Exception as exc:
            logger.exception("Streaming error for session %s: %s", request.session_id, exc)
            payload = json.dumps({"type": "error", "message": str(exc)})
            yield f"data: {payload}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@app.get("/api/sessions/{session_id}", tags=["Sessions"])
async def get_session(session_id: str):
    """Return metadata and conversation history for a session."""
    session = _sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")
    return {
        "session_id": session_id,
        "filename": session["filename"],
        "num_chunks": session["num_chunks"],
        "num_pages": session["num_pages"],
        "history": session["history"],
    }


@app.delete("/api/sessions/{session_id}", tags=["Sessions"])
async def delete_session(session_id: str):
    """Delete a session and its ChromaDB collection."""
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail="Session not found.")
    delete_collection(session_id)
    del _sessions[session_id]
    logger.info("Session %s deleted.", session_id)
    return {"message": f"Session {session_id} deleted successfully."}
