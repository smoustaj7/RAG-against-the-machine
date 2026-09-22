"""Tests for the embedding and hybrid retrievers.

These tests never download or run a transformer model: the
``EmbeddingRetriever`` is fitted by writing its vectors directly,
and query encoding is stubbed out. That keeps the suite offline
and fast while still exercising ranking, fusion and persistence.
"""

import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pytest

from src.chunk_store import Chunk
from src.retrieval.base import Retriever
from src.retrieval.embedding import EmbeddingRetriever
from src.retrieval.hybrid import (
    DEFAULT_RRF_K,
    HybridRetriever,
    reciprocal_rank_fusion,
)
from src.retrieval.lexical import BM25Retriever


def _make_chunk(chunk_id: str, text: str) -> Chunk:
    """Helper to create a Chunk with minimal required fields."""
    return Chunk(
        chunk_id=chunk_id,
        file_path="test.py",
        first_character_index=0,
        last_character_index=len(text),
        text=text,
        file_hash="abc123",
    )


def _fit_manually(
    retriever: EmbeddingRetriever,
    chunk_ids: Sequence[str],
    vectors: Sequence[Sequence[float]],
) -> None:
    """Fit an EmbeddingRetriever without running a model."""
    matrix = np.array(vectors, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    retriever._chunk_ids = list(chunk_ids)
    retriever._embeddings = (matrix / norms).astype(np.float32)


def _stub_query_vector(
    retriever: EmbeddingRetriever,
    vector: Sequence[float],
) -> None:
    """Make ``_encode`` return a fixed normalised query vector."""
    array = np.array([vector], dtype=np.float32)
    norm = float(np.linalg.norm(array))
    if norm:
        array = array / norm

    def _encode(
        texts: List[str], show_progress: bool = False
    ) -> np.ndarray:
        return array.astype(np.float32)

    retriever._encode = _encode  # type: ignore[method-assign]


class _FakeRetriever(Retriever):
    """Retriever that replays a fixed ranked list."""

    def __init__(
        self, results: List[Tuple[str, float]]
    ) -> None:
        self.results = results
        self.indexed: Optional[Dict[str, Chunk]] = None

    def index(self, chunks: Dict[str, Chunk]) -> None:
        self.indexed = chunks

    def search(
        self, query: str, k: int
    ) -> List[Tuple[str, float]]:
        return self.results[:k]

    def save(self, path: Union[str, Path]) -> None:
        Path(path).write_text("fake", encoding="utf-8")

    @classmethod
    def load(cls, path: Union[str, Path]) -> "_FakeRetriever":
        return cls([])


class TestReciprocalRankFusion:
    """Tests for the pure RRF scoring function."""

    def test_single_list_preserves_order(self) -> None:
        ranked = [("a", 9.0), ("b", 4.0), ("c", 1.0)]
        fused = reciprocal_rank_fusion([ranked])
        assert [cid for cid, _ in fused] == ["a", "b", "c"]

    def test_agreement_beats_single_first_place(self) -> None:
        """A chunk ranked by both lists outranks a lone winner."""
        lexical = [("a", 9.0), ("b", 8.0)]
        semantic = [("b", 0.9), ("c", 0.8)]
        fused = reciprocal_rank_fusion([lexical, semantic])
        assert fused[0][0] == "b"
        assert set(cid for cid, _ in fused) == {"a", "b", "c"}

    def test_scores_are_rrf_scores(self) -> None:
        fused = reciprocal_rank_fusion([[("a", 123.4)]])
        expected = 1.0 / (DEFAULT_RRF_K + 1)
        assert fused[0][1] == pytest.approx(expected)

    def test_zero_weight_silences_a_list(self) -> None:
        lexical = [("a", 1.0)]
        semantic = [("c", 1.0)]
        fused = reciprocal_rank_fusion(
            [lexical, semantic], weights=[1.0, 0.0]
        )
        assert fused[0][0] == "a"
        assert fused[1][1] == pytest.approx(0.0)

    def test_weight_can_flip_the_winner(self) -> None:
        lexical = [("a", 1.0)]
        semantic = [("c", 1.0)]
        fused = reciprocal_rank_fusion(
            [lexical, semantic], weights=[1.0, 5.0]
        )
        assert fused[0][0] == "c"

    def test_ties_broken_by_first_appearance(self) -> None:
        fused = reciprocal_rank_fusion([[("a", 1.0)], [("b", 1.0)]])
        assert [cid for cid, _ in fused] == ["a", "b"]

    def test_empty_inputs(self) -> None:
        assert reciprocal_rank_fusion([]) == []
        assert reciprocal_rank_fusion([[], []]) == []

    def test_weight_count_mismatch_raises(self) -> None:
        with pytest.raises(ValueError):
            reciprocal_rank_fusion([[("a", 1.0)]], weights=[1.0, 1.0])

    def test_negative_rrf_k_raises(self) -> None:
        with pytest.raises(ValueError):
            reciprocal_rank_fusion([[("a", 1.0)]], rrf_k=-1)


class TestEmbeddingRetrieverState:
    """Fit / is_fitted behaviour."""

    def test_not_fitted_initially(self) -> None:
        r = EmbeddingRetriever()
        assert not r.is_fitted
        assert r.embedding_dim is None

    def test_index_empty_dict_stays_unfitted(self) -> None:
        r = EmbeddingRetriever()
        r.index({})
        assert not r.is_fitted

    def test_manual_fit_reports_dimension(self) -> None:
        r = EmbeddingRetriever()
        _fit_manually(r, ["c1", "c2"], [[1.0, 0.0], [0.0, 1.0]])
        assert r.is_fitted
        assert r.embedding_dim == 2


class TestEmbeddingRetrieverSearch:
    """Ranking behaviour with a stubbed encoder."""

    def test_ranks_by_cosine_similarity(self) -> None:
        r = EmbeddingRetriever()
        _fit_manually(
            r,
            ["c1", "c2", "c3"],
            [[1.0, 0.0], [0.0, 1.0], [0.7, 0.7]],
        )
        _stub_query_vector(r, [1.0, 0.0])

        results = r.search("anything", k=3)
        assert [cid for cid, _ in results] == ["c1", "c3", "c2"]
        assert results[0][1] == pytest.approx(1.0, abs=1e-5)

    def test_k_larger_than_corpus(self) -> None:
        r = EmbeddingRetriever()
        _fit_manually(r, ["c1"], [[1.0, 0.0]])
        _stub_query_vector(r, [1.0, 0.0])
        assert len(r.search("anything", k=10)) == 1

    def test_search_k_zero(self) -> None:
        r = EmbeddingRetriever()
        _fit_manually(r, ["c1"], [[1.0, 0.0]])
        _stub_query_vector(r, [1.0, 0.0])
        assert r.search("anything", k=0) == []

    def test_search_empty_query(self) -> None:
        r = EmbeddingRetriever()
        _fit_manually(r, ["c1"], [[1.0, 0.0]])
        _stub_query_vector(r, [1.0, 0.0])
        assert r.search("   ", k=5) == []

    def test_search_unfitted(self) -> None:
        r = EmbeddingRetriever()
        assert r.search("anything", k=5) == []

    def test_dimension_mismatch_raises(self) -> None:
        r = EmbeddingRetriever()
        _fit_manually(r, ["c1"], [[1.0, 0.0]])
        _stub_query_vector(r, [1.0, 0.0, 0.0])
        with pytest.raises(ValueError):
            r.search("anything", k=1)


class TestEmbeddingRetrieverPersistence:
    """Save / load round-trip."""

    def test_save_load_roundtrip(self) -> None:
        r = EmbeddingRetriever(
            model_name="stub/model", max_seq_length=128
        )
        _fit_manually(
            r, ["c1", "c2"], [[1.0, 0.0], [0.0, 1.0]]
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "embeddings.npz"
            r.save(path)
            assert path.exists()

            loaded = EmbeddingRetriever.load(path)

        assert loaded.is_fitted
        assert loaded.model_name == "stub/model"
        assert loaded.max_seq_length == 128
        assert loaded._chunk_ids == ["c1", "c2"]
        assert loaded._embeddings is not None
        assert np.allclose(loaded._embeddings, r._embeddings)

    def test_save_unfitted_then_load(self) -> None:
        r = EmbeddingRetriever()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "embeddings.npz"
            r.save(path)
            loaded = EmbeddingRetriever.load(path)
        assert not loaded.is_fitted
        assert loaded.search("anything", k=5) == []

    def test_load_nonexistent_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            EmbeddingRetriever.load("/nonexistent/embeddings.npz")


class TestHybridRetrieverSearch:
    """Fusion of the two components."""

    def test_fuses_both_components(self) -> None:
        lexical = _FakeRetriever([("a", 9.0), ("b", 8.0)])
        semantic = _FakeRetriever([("b", 0.9), ("c", 0.8)])
        h = HybridRetriever(lexical=lexical, embedding=semantic)

        results = h.search("query", k=3)
        assert results[0][0] == "b"
        assert len(results) == 3

    def test_respects_k(self) -> None:
        lexical = _FakeRetriever([("a", 9.0), ("b", 8.0)])
        semantic = _FakeRetriever([("c", 0.9), ("d", 0.8)])
        h = HybridRetriever(lexical=lexical, embedding=semantic)
        assert len(h.search("query", k=2)) == 2

    def test_k_zero(self) -> None:
        h = HybridRetriever(
            lexical=_FakeRetriever([("a", 1.0)]),
            embedding=_FakeRetriever([("b", 1.0)]),
        )
        assert h.search("query", k=0) == []

    def test_empty_query(self) -> None:
        h = HybridRetriever(
            lexical=_FakeRetriever([("a", 1.0)]),
            embedding=_FakeRetriever([("b", 1.0)]),
        )
        assert h.search("", k=5) == []

    def test_drops_non_positive_scores(self) -> None:
        """BM25 zeros carry no signal and must not be fused."""
        lexical = _FakeRetriever([("a", 0.0), ("b", 0.0)])
        semantic = _FakeRetriever([("c", 0.9)])
        h = HybridRetriever(lexical=lexical, embedding=semantic)

        results = h.search("query", k=5)
        assert [cid for cid, _ in results] == ["c"]

    def test_keeps_non_positive_when_disabled(self) -> None:
        lexical = _FakeRetriever([("a", 0.0)])
        semantic = _FakeRetriever([("c", 0.9)])
        h = HybridRetriever(
            lexical=lexical,
            embedding=semantic,
            drop_non_positive=False,
        )
        assert len(h.search("query", k=5)) == 2

    def test_index_fits_both_components(self) -> None:
        lexical = _FakeRetriever([])
        semantic = _FakeRetriever([])
        h = HybridRetriever(lexical=lexical, embedding=semantic)

        chunks = {"c1": _make_chunk("c1", "hello world")}
        h.index(chunks)

        assert lexical.indexed == chunks
        assert semantic.indexed == chunks


class TestHybridRetrieverPersistence:
    """Save / load round-trip over a component directory."""

    def test_save_load_roundtrip(self) -> None:
        lexical = BM25Retriever()
        lexical.index(
            {
                "c1": _make_chunk("c1", "Python class method"),
                "c2": _make_chunk("c2", "Markdown documentation"),
            }
        )
        semantic = EmbeddingRetriever(model_name="stub/model")
        _fit_manually(
            semantic, ["c1", "c2"], [[1.0, 0.0], [0.0, 1.0]]
        )

        h = HybridRetriever(
            lexical=lexical,
            embedding=semantic,
            rrf_k=17,
            candidate_pool=25,
            embedding_weight=2.0,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "hybrid_index"
            h.save(path)
            assert (path / "lexical.pkl").exists()
            assert (path / "embedding.npz").exists()
            assert (path / "hybrid_meta.json").exists()

            loaded = HybridRetriever.load(path)

        assert loaded.is_fitted
        assert loaded.rrf_k == 17
        assert loaded.candidate_pool == 25
        assert loaded.embedding_weight == pytest.approx(2.0)
        assert loaded.lexical.search("Python class", k=2)

    def test_load_nonexistent_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            HybridRetriever.load("/nonexistent/hybrid_index")

    def test_load_missing_component_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "empty_index"
            path.mkdir()
            with pytest.raises(FileNotFoundError):
                HybridRetriever.load(path)
