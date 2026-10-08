from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field

from src.cli import (
    _collect_source_texts,
    _load_retriever_and_store,
    _search_single,
)
from src.generation import AnswerGenerator
from src.indexer import (
    DEFAULT_BM25_INDEX_PATH,
    DEFAULT_CHUNKS_PATH,
    normalize_retriever_name,
    resolve_index_path,
)
from src.models import MinimalAnswer, MinimalSearchResults
from typing import Optional

_generator: Optional[AnswerGenerator] = None


def _get_generator() -> AnswerGenerator:
    """Return the shared AnswerGenerator, creating it on first use."""
    global _generator
    if _generator is None:
        _generator = AnswerGenerator()
    return _generator


class SearchRequest(BaseModel):
    """POST body for /search."""

    query: str
    k: int = Field(default=5, ge=1)
    retriever: str = "bm25"
    chunks_path: str = DEFAULT_CHUNKS_PATH
    index_path: str = ""
    bm25_index_path: str = DEFAULT_BM25_INDEX_PATH
    cache: bool = True


class AnswerRequest(BaseModel):
    """POST body for /answer."""

    query: str
    k: int = Field(default=5, ge=1)
    retriever: str = "bm25"
    chunks_path: str = DEFAULT_CHUNKS_PATH
    index_path: str = ""
    bm25_index_path: str = DEFAULT_BM25_INDEX_PATH
    cache: bool = True


app = FastAPI(
    title="RAG Against the Machine",
    description="Local HTTP API — search and answer over an indexed codebase.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    """Redirect root to interactive API docs."""
    return RedirectResponse(url="/docs")


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness check."""
    return {"status": "ok"}


@app.post("/search", response_model=MinimalSearchResults)
def search(req: SearchRequest) -> MinimalSearchResults:
    """Retrieve sources for a single query.

    Mirrors ``CLI.search`` — same retriever loading, same scoring.
    """
    if not req.query or not req.query.strip():
        raise HTTPException(status_code=400, detail="Empty query.")

    try:
        ret_name = normalize_retriever_name(req.retriever)
        resolved = str(
            resolve_index_path(
                req.retriever, req.index_path or None,
                req.bm25_index_path,
            )
        )
        retriever_impl, store = _load_retriever_and_store(
            req.chunks_path, resolved, req.retriever,
            use_cache=req.cache,
        )
        sources = _search_single(
            req.query, req.k, retriever_impl, store,
            retriever_name=ret_name,
            index_path=resolved,
            use_cache=req.cache,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    return MinimalSearchResults(
        question=req.query,
        retrieved_sources=sources,
    )


@app.post("/answer", response_model=MinimalAnswer)
def answer(req: AnswerRequest) -> MinimalAnswer:
    """Search then generate an answer for a single query.

    Mirrors ``CLI.answer`` — same retrieval, same generation.
    """
    if not req.query or not req.query.strip():
        raise HTTPException(status_code=400, detail="Empty query.")

    try:
        ret_name = normalize_retriever_name(req.retriever)
        resolved = str(
            resolve_index_path(
                req.retriever, req.index_path or None,
                req.bm25_index_path,
            )
        )
        retriever_impl, store = _load_retriever_and_store(
            req.chunks_path, resolved, req.retriever,
            use_cache=req.cache,
        )
        sources = _search_single(
            req.query, req.k, retriever_impl, store,
            retriever_name=ret_name,
            index_path=resolved,
            use_cache=req.cache,
        )
        source_texts = _collect_source_texts(sources, store)
        generator = _get_generator()
        answer_text = generator.generate(
            question=req.query,
            source_texts=source_texts,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    return MinimalAnswer(
        question=req.query,
        retrieved_sources=sources,
        answer=answer_text,
    )
