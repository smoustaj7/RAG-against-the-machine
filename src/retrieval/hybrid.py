"""Hybrid retriever combining lexical and semantic search."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from src.chunk_store import Chunk
from src.retrieval.base import Retriever
from src.retrieval.embedding import EmbeddingRetriever
from src.retrieval.lexical import BM25Retriever


DEFAULT_RRF_K = 60
DEFAULT_CANDIDATE_POOL = 50

LEXICAL_FILENAME = "lexical.pkl"
EMBEDDING_FILENAME = "embedding.npz"
META_FILENAME = "hybrid_meta.json"


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[Tuple[str, float]]],
    weights: Optional[Sequence[float]] = None,
    rrf_k: int = DEFAULT_RRF_K,
) -> List[Tuple[str, float]]:
    """Fuse several ranked lists with Reciprocal Rank Fusion.

    Each list contributes ``weight / (rrf_k + rank)`` to every
    chunk it ranks, with ``rank`` starting at 1. Scores are summed
    across lists, so a chunk ranked decently by both retrievers
    beats a chunk ranked first by only one of them.

    RRF deliberately ignores the raw scores: BM25 scores and cosine
    similarities live on incomparable scales, and rank position is
    the only signal the two share.

    Args:
        ranked_lists: the (chunk_id, score) lists to fuse, each
            already sorted by descending relevance.
        weights: per-list weights. Defaults to 1.0 for every list.
        rrf_k: the RRF damping constant; larger values flatten the
            contribution of the top ranks.

    Returns:
        A fused list of (chunk_id, rrf_score) tuples sorted by
        descending score. Ties are broken by first appearance, so
        the output is deterministic.

    Raises:
        ValueError: if ``weights`` length differs from
            ``ranked_lists`` length, or if ``rrf_k`` is negative.
    """
    if rrf_k < 0:
        raise ValueError(f"rrf_k must be >= 0, got {rrf_k}")

    if weights is None:
        weights = [1.0] * len(ranked_lists)
    if len(weights) != len(ranked_lists):
        raise ValueError(
            f"Got {len(weights)} weights for "
            f"{len(ranked_lists)} ranked lists."
        )

    scores: Dict[str, float] = {}
    first_seen: Dict[str, int] = {}
    counter = 0

    for ranked, weight in zip(ranked_lists, weights):
        for rank, (chunk_id, _score) in enumerate(ranked, start=1):
            scores[chunk_id] = (
                scores.get(chunk_id, 0.0) + weight / (rrf_k + rank)
            )
            if chunk_id not in first_seen:
                first_seen[chunk_id] = counter
                counter += 1

    return sorted(
        scores.items(),
        key=lambda item: (-item[1], first_seen[item[0]]),
    )


def _drop_non_positive(
    ranked: List[Tuple[str, float]],
) -> List[Tuple[str, float]]:
    """Drop results whose score is <= 0.

    A BM25 score of 0 means the chunk shares no query term at all,
    so its position in the ranking is arbitrary. Feeding those
    arbitrary positions into RRF would let noise outrank genuine
    hits from the other retriever.
    """
    return [(cid, score) for cid, score in ranked if score > 0]


class HybridRetriever(Retriever):
    """Fuse a lexical and a semantic retriever via RRF.

    Both component retrievers index the *same* chunk registry and
    return ``(chunk_id, score)`` lists, so fusing them needs no
    knowledge of how either one works. Swapping a component out
    requires no change here or at any call site.

    Unlike the other retrievers, ``save()`` and ``load()`` take a
    *directory*: the two component indexes are separate artifacts
    and are persisted side by side inside it.
    """

    def __init__(
        self,
        lexical: Optional[Retriever] = None,
        embedding: Optional[Retriever] = None,
        rrf_k: int = DEFAULT_RRF_K,
        candidate_pool: int = DEFAULT_CANDIDATE_POOL,
        lexical_weight: float = 1.0,
        embedding_weight: float = 1.0,
        drop_non_positive: bool = True,
    ) -> None:
        """Initialize a hybrid retriever.
        Args:
            lexical: the lexical component. Defaults to a fresh
                ``BM25Retriever``.
            embedding: the semantic component. Defaults to a fresh
                ``EmbeddingRetriever``.
            rrf_k: RRF damping constant.
            candidate_pool: how many results to pull from each
                component before fusing. Always at least ``k``.
            lexical_weight: RRF weight for the lexical component.
            embedding_weight: RRF weight for the semantic component.
            drop_non_positive: discard component results scoring
                <= 0 before fusing.
        """
        self.lexical: Retriever = (
            lexical if lexical is not None else BM25Retriever()
        )
        self.embedding: Retriever = (
            embedding
            if embedding is not None
            else EmbeddingRetriever()
        )
        self.rrf_k = rrf_k
        self.candidate_pool = max(1, candidate_pool)
        self.lexical_weight = lexical_weight
        self.embedding_weight = embedding_weight
        self.drop_non_positive = drop_non_positive

    @property
    def is_fitted(self) -> bool:
        """Return True if both components report being fitted."""
        return bool(
            getattr(self.lexical, "is_fitted", True)
        ) and bool(getattr(self.embedding, "is_fitted", True))

    def index(self, chunks: Dict[str, Chunk]) -> None:
        """Fit both component retrievers on the same chunks.

        Args:
            chunks: mapping from chunk_id to Chunk objects.
        """
        self.lexical.index(chunks)
        self.embedding.index(chunks)

    def search(
        self, query: str, k: int
    ) -> List[Tuple[str, float]]:
        """Search both components and fuse their rankings.

        Args:
            query: the search query string.
            k: number of results to return.

        Returns:
            List of (chunk_id, rrf_score) tuples sorted by
            descending fused score. Note that the returned scores
            are RRF scores, not BM25 scores or cosine similarities.
        """
        if k <= 0:
            return []
        if not query or not query.strip():
            return []

        pool = max(k, self.candidate_pool)
        lexical_hits = self.lexical.search(query, pool)
        embedding_hits = self.embedding.search(query, pool)

        if self.drop_non_positive:
            lexical_hits = _drop_non_positive(lexical_hits)
            embedding_hits = _drop_non_positive(embedding_hits)

        fused = reciprocal_rank_fusion(
            [lexical_hits, embedding_hits],
            weights=[self.lexical_weight, self.embedding_weight],
            rrf_k=self.rrf_k,
        )
        return fused[:k]

    def save(self, path: Union[str, Path]) -> None:
        """Persist both components into a directory.

        Args:
            path: directory to write the component indexes into.
                It is created if missing.
        """
        dir_path = Path(path)
        dir_path.mkdir(parents=True, exist_ok=True)

        self.lexical.save(dir_path / LEXICAL_FILENAME)
        self.embedding.save(dir_path / EMBEDDING_FILENAME)

        meta = {
            "rrf_k": self.rrf_k,
            "candidate_pool": self.candidate_pool,
            "lexical_weight": self.lexical_weight,
            "embedding_weight": self.embedding_weight,
            "drop_non_positive": self.drop_non_positive,
        }
        (dir_path / META_FILENAME).write_text(
            json.dumps(meta, indent=2), encoding="utf-8"
        )

    @classmethod
    def load(cls, path: Union[str, Path]) -> "HybridRetriever":
        """Load a hybrid retriever from a directory.

        Args:
            path: directory previously written by ``save()``.

        Returns:
            A fitted HybridRetriever instance.

        Raises:
            FileNotFoundError: if the directory or either component
                index is missing.
        """
        dir_path = Path(path)
        if not dir_path.is_dir():
            raise FileNotFoundError(
                f"Hybrid index directory not found at: {dir_path}"
            )

        lexical = BM25Retriever.load(dir_path / LEXICAL_FILENAME)
        embedding = EmbeddingRetriever.load(
            dir_path / EMBEDDING_FILENAME
        )

        meta_path = dir_path / META_FILENAME
        meta: Dict[str, Any] = {}
        if meta_path.exists():
            try:
                meta = json.loads(
                    meta_path.read_text(encoding="utf-8")
                )
            except (json.JSONDecodeError, UnicodeDecodeError):
                meta = {}

        return cls(
            lexical=lexical,
            embedding=embedding,
            rrf_k=int(meta.get("rrf_k", DEFAULT_RRF_K)),
            candidate_pool=int(
                meta.get("candidate_pool", DEFAULT_CANDIDATE_POOL)
            ),
            lexical_weight=float(meta.get("lexical_weight", 1.0)),
            embedding_weight=float(
                meta.get("embedding_weight", 1.0)
            ),
            drop_non_positive=bool(
                meta.get("drop_non_positive", True)
            ),
        )
