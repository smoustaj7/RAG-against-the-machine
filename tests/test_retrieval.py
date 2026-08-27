"""Tests for the BM25 retriever."""

import tempfile
from pathlib import Path

from src.chunk_store import Chunk
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


class TestBM25RetrieverIndex:
    """Tests for BM25Retriever.index()."""

    def test_not_fitted_initially(self) -> None:
        r = BM25Retriever()
        assert not r.is_fitted

    def test_fit_on_chunks(self) -> None:
        r = BM25Retriever()
        chunks = {
            "c1": _make_chunk("c1", "hello world"),
            "c2": _make_chunk("c2", "foo bar baz"),
        }
        r.index(chunks)
        assert r.is_fitted

    def test_fit_on_empty_dict(self) -> None:
        r = BM25Retriever()
        r.index({})
        assert not r.is_fitted


class TestBM25RetrieverSearch:
    """Tests for BM25Retriever.search()."""

    def test_search_returns_relevant_chunk(self) -> None:
        r = BM25Retriever()
        chunks = {
            "c1": _make_chunk(
                "c1", "Python function definition decorator"
            ),
            "c2": _make_chunk(
                "c2", "Markdown header section documentation"
            ),
        }
        r.index(chunks)
        results = r.search("function decorator", k=2)
        assert len(results) == 2
        # The Python chunk should rank first.
        assert results[0][0] == "c1"
        # Scores should be floats.
        assert isinstance(results[0][1], float)

    def test_search_k_larger_than_corpus(self) -> None:
        r = BM25Retriever()
        chunks = {"c1": _make_chunk("c1", "only one document here")}
        r.index(chunks)
        results = r.search("document", k=10)
        assert len(results) == 1

    def test_search_k_zero(self) -> None:
        r = BM25Retriever()
        chunks = {"c1": _make_chunk("c1", "hello world")}
        r.index(chunks)
        results = r.search("hello", k=0)
        assert results == []

    def test_search_empty_query(self) -> None:
        r = BM25Retriever()
        chunks = {"c1": _make_chunk("c1", "hello world")}
        r.index(chunks)
        results = r.search("", k=5)
        assert results == []

    def test_search_unfitted(self) -> None:
        r = BM25Retriever()
        results = r.search("anything", k=5)
        assert results == []


class TestBM25RetrieverPersistence:
    """Tests for save/load round-trip."""

    def test_save_load_roundtrip(self) -> None:
        r = BM25Retriever()
        chunks = {
            "c1": _make_chunk("c1", "Python class method"),
            "c2": _make_chunk("c2", "Markdown documentation"),
        }
        r.index(chunks)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "bm25.pkl"
            r.save(path)
            assert path.exists()

            loaded = BM25Retriever.load(path)
            assert loaded.is_fitted

            # Search results should be identical.
            original_results = r.search("Python class", k=2)
            loaded_results = loaded.search("Python class", k=2)
            assert len(original_results) == len(loaded_results)
            for orig, load in zip(original_results, loaded_results):
                assert orig[0] == load[0]
                assert abs(orig[1] - load[1]) < 1e-6

    def test_load_nonexistent_raises(self) -> None:
        import pytest

        with pytest.raises(FileNotFoundError):
            BM25Retriever.load("/nonexistent/path/bm25.pkl")
