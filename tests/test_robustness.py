"""Phase 6 — Robustness tests.

Sweep all 6 CLI commands with adversarial / degenerate inputs and
confirm every failure prints a clean message (to stderr) without
raw tracebacks.  No test here should raise an unhandled exception.
"""

import json
import sys
from io import StringIO
from pathlib import Path

import pytest

from src.cli import CLI


# ── Fixtures ────────────────────────────────────────────────────


@pytest.fixture
def cli() -> CLI:
    """Return a fresh CLI instance for each test."""
    return CLI()


@pytest.fixture
def tmp_dir(tmp_path: Path) -> Path:
    """Provide a temporary directory scoped per test."""
    return tmp_path


@pytest.fixture
def valid_dataset(tmp_dir: Path) -> Path:
    """Write a minimal valid UnansweredQuestions dataset JSON."""
    dataset = {
        "rag_questions": [
            {
                "question_id": "12345678-1234-1234-1234-123456789abc",
                "question": "What is vLLM?",
            }
        ]
    }
    path = tmp_dir / "valid_dataset.json"
    path.write_text(json.dumps(dataset), encoding="utf-8")
    return path


@pytest.fixture
def answered_dataset(tmp_dir: Path) -> Path:
    """Write a minimal valid AnsweredQuestions dataset JSON."""
    dataset = {
        "rag_questions": [
            {
                "question_id": "12345678-1234-1234-1234-123456789abc",
                "question": "What is vLLM?",
                "answer": "A fast LLM serving engine.",
                "sources": [
                    {
                        "file_path": "a.py",
                        "first_character_index": 0,
                        "last_character_index": 100,
                    }
                ],
            }
        ]
    }
    path = tmp_dir / "answered_dataset.json"
    path.write_text(json.dumps(dataset), encoding="utf-8")
    return path


@pytest.fixture
def valid_student_results(tmp_dir: Path) -> Path:
    """Write a minimal valid StudentSearchResults JSON."""
    data = {
        "search_results": [
            {
                "question_id": "12345678-1234-1234-1234-123456789abc",
                "question": "What is vLLM?",
                "retrieved_sources": [
                    {
                        "file_path": "a.py",
                        "first_character_index": 0,
                        "last_character_index": 100,
                    }
                ],
            }
        ],
        "k": 5,
    }
    path = tmp_dir / "student_results.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


@pytest.fixture
def malformed_json(tmp_dir: Path) -> Path:
    """Write a file containing invalid JSON."""
    path = tmp_dir / "malformed.json"
    path.write_text("{not valid json!!!", encoding="utf-8")
    return path


@pytest.fixture
def wrong_schema_json(tmp_dir: Path) -> Path:
    """Write valid JSON but with the wrong schema."""
    path = tmp_dir / "wrong_schema.json"
    path.write_text(
        json.dumps({"unrelated_key": [1, 2, 3]}),
        encoding="utf-8",
    )
    return path


def _capture_stderr(func: object) -> str:
    """Call a callable and capture stderr output."""
    captured = StringIO()
    old_stderr = sys.stderr
    sys.stderr = captured
    try:
        func()  # type: ignore[operator]
    finally:
        sys.stderr = old_stderr
    return captured.getvalue()


# ── 1. Index Command Robustness ─────────────────────────────────


class TestIndexRobustness:
    """Robustness tests for the ``index`` command."""

    def test_nonexistent_corpus_dir(self, cli: CLI) -> None:
        """Indexing a nonexistent directory prints an error."""
        stderr = _capture_stderr(
            lambda: cli.index(corpus_dir="/no/such/dir")
        )
        assert "Error" in stderr or "error" in stderr.lower()

    def test_corpus_dir_is_a_file(
        self, cli: CLI, tmp_dir: Path
    ) -> None:
        """Indexing a file (not a dir) prints an error."""
        fake = tmp_dir / "notadir.txt"
        fake.write_text("hello", encoding="utf-8")
        stderr = _capture_stderr(
            lambda: cli.index(corpus_dir=str(fake))
        )
        assert "error" in stderr.lower()

    def test_empty_corpus_dir(
        self, cli: CLI, tmp_dir: Path
    ) -> None:
        """Indexing an empty directory prints a warning."""
        empty = tmp_dir / "empty_corpus"
        empty.mkdir()
        _capture_stderr(
            lambda: cli.index(corpus_dir=str(empty))
        )
        # Should not crash; may print "No files to index"

    def test_corpus_with_non_utf8_file(
        self, cli: CLI, tmp_dir: Path
    ) -> None:
        """Non-UTF8 files in corpus are skipped gracefully."""
        corpus = tmp_dir / "corpus"
        corpus.mkdir()
        bad_file = corpus / "bad.py"
        bad_file.write_bytes(b"\x80\x81\x82\x83\x84")
        good_file = corpus / "good.py"
        good_file.write_text(
            "def hello():\n    pass\n", encoding="utf-8"
        )
        # Should not crash.
        cli.index(corpus_dir=str(corpus))

    def test_negative_max_chunk_size(self, cli: CLI) -> None:
        """Negative max_chunk_size prints an error."""
        stderr = _capture_stderr(
            lambda: cli.index(max_chunk_size=-1)
        )
        assert "error" in stderr.lower()

    def test_zero_max_chunk_size(self, cli: CLI) -> None:
        """Zero max_chunk_size prints an error."""
        stderr = _capture_stderr(
            lambda: cli.index(max_chunk_size=0)
        )
        assert "error" in stderr.lower()


# ── 2. Search Command Robustness ────────────────────────────────


class TestSearchRobustness:
    """Robustness tests for the ``search`` command."""

    def test_empty_query(self, cli: CLI) -> None:
        """Empty string query prints a warning, no crash."""
        stderr = _capture_stderr(
            lambda: cli.search(query="")
        )
        assert "warning" in stderr.lower() or "empty" in stderr.lower()

    def test_whitespace_query(self, cli: CLI) -> None:
        """Whitespace-only query prints a warning, no crash."""
        stderr = _capture_stderr(
            lambda: cli.search(query="   ")
        )
        assert "warning" in stderr.lower() or "empty" in stderr.lower()

    def test_k_zero(self, cli: CLI) -> None:
        """k=0 prints a warning, no crash."""
        stderr = _capture_stderr(
            lambda: cli.search(query="test", k=0)
        )
        assert "k" in stderr.lower() or "warning" in stderr.lower()

    def test_k_negative(self, cli: CLI) -> None:
        """k=-5 prints a warning, no crash."""
        stderr = _capture_stderr(
            lambda: cli.search(query="test", k=-5)
        )
        assert "k" in stderr.lower() or "warning" in stderr.lower()

    def test_missing_index(self, cli: CLI) -> None:
        """Searching without a built index prints an error."""
        stderr = _capture_stderr(
            lambda: cli.search(
                query="test",
                k=5,
                chunks_path="/nonexistent/chunks.jsonl",
                bm25_index_path="/nonexistent/bm25.pkl",
            )
        )
        assert "error" in stderr.lower()

    def test_nonexistent_chunks_path(self, cli: CLI) -> None:
        """Nonexistent chunks path prints an error."""
        stderr = _capture_stderr(
            lambda: cli.search(
                query="test",
                chunks_path="/no/such/chunks.jsonl",
            )
        )
        assert "error" in stderr.lower()


# ── 3. Search-Dataset Command Robustness ────────────────────────


class TestSearchDatasetRobustness:
    """Robustness tests for the ``search_dataset`` command."""

    def test_nonexistent_dataset(self, cli: CLI) -> None:
        """Nonexistent dataset path prints an error."""
        stderr = _capture_stderr(
            lambda: cli.search_dataset(
                dataset_path="/no/such/dataset.json"
            )
        )
        assert "error" in stderr.lower()

    def test_malformed_json(
        self, cli: CLI, malformed_json: Path
    ) -> None:
        """Malformed JSON dataset prints an error."""
        stderr = _capture_stderr(
            lambda: cli.search_dataset(
                dataset_path=str(malformed_json)
            )
        )
        assert "error" in stderr.lower()

    def test_wrong_schema(
        self, cli: CLI, wrong_schema_json: Path
    ) -> None:
        """JSON with wrong schema prints an error."""
        stderr = _capture_stderr(
            lambda: cli.search_dataset(
                dataset_path=str(wrong_schema_json)
            )
        )
        assert "error" in stderr.lower()

    def test_k_zero(
        self, cli: CLI, valid_dataset: Path
    ) -> None:
        """k=0 prints a warning, no crash."""
        stderr = _capture_stderr(
            lambda: cli.search_dataset(
                dataset_path=str(valid_dataset), k=0
            )
        )
        assert "k" in stderr.lower() or "warning" in stderr.lower()

    def test_k_negative(
        self, cli: CLI, valid_dataset: Path
    ) -> None:
        """k=-3 prints a warning, no crash."""
        stderr = _capture_stderr(
            lambda: cli.search_dataset(
                dataset_path=str(valid_dataset), k=-3
            )
        )
        assert "k" in stderr.lower() or "warning" in stderr.lower()

    def test_empty_dataset_path(self, cli: CLI) -> None:
        """Empty dataset_path prints an error, no crash."""
        stderr = _capture_stderr(
            lambda: cli.search_dataset(dataset_path="")
        )
        assert "error" in stderr.lower()


# ── 4. Evaluate Command Robustness ──────────────────────────────


class TestEvaluateRobustness:
    """Robustness tests for the ``evaluate`` command."""

    def test_nonexistent_student_results(self, cli: CLI) -> None:
        """Nonexistent student results path prints an error."""
        stderr = _capture_stderr(
            lambda: cli.evaluate(
                student_search_results_path="/no/such/results.json",
                dataset_path="/no/such/dataset.json",
            )
        )
        assert "error" in stderr.lower()

    def test_nonexistent_dataset(
        self, cli: CLI, valid_student_results: Path
    ) -> None:
        """Nonexistent dataset path prints an error."""
        stderr = _capture_stderr(
            lambda: cli.evaluate(
                student_search_results_path=str(
                    valid_student_results
                ),
                dataset_path="/no/such/dataset.json",
            )
        )
        assert "error" in stderr.lower()

    def test_malformed_student_results(
        self, cli: CLI, malformed_json: Path, answered_dataset: Path
    ) -> None:
        """Malformed student results JSON prints an error."""
        stderr = _capture_stderr(
            lambda: cli.evaluate(
                student_search_results_path=str(malformed_json),
                dataset_path=str(answered_dataset),
            )
        )
        assert "error" in stderr.lower()

    def test_malformed_dataset(
        self, cli: CLI, valid_student_results: Path,
        malformed_json: Path,
    ) -> None:
        """Malformed dataset JSON prints an error."""
        stderr = _capture_stderr(
            lambda: cli.evaluate(
                student_search_results_path=str(
                    valid_student_results
                ),
                dataset_path=str(malformed_json),
            )
        )
        assert "error" in stderr.lower()

    def test_wrong_schema_student_results(
        self, cli: CLI, wrong_schema_json: Path,
        answered_dataset: Path,
    ) -> None:
        """Wrong schema in student results prints an error."""
        stderr = _capture_stderr(
            lambda: cli.evaluate(
                student_search_results_path=str(wrong_schema_json),
                dataset_path=str(answered_dataset),
            )
        )
        assert "error" in stderr.lower()

    def test_empty_paths(self, cli: CLI) -> None:
        """Empty paths print an error, no crash."""
        stderr = _capture_stderr(
            lambda: cli.evaluate(
                student_search_results_path="",
                dataset_path="",
            )
        )
        assert "error" in stderr.lower()


# ── 5. Answer Command Robustness ────────────────────────────────


class TestAnswerRobustness:
    """Robustness tests for the ``answer`` command."""

    def test_empty_query(self, cli: CLI) -> None:
        """Empty query prints a warning, no crash."""
        stderr = _capture_stderr(
            lambda: cli.answer(query="")
        )
        assert "warning" in stderr.lower() or "empty" in stderr.lower()

    def test_whitespace_query(self, cli: CLI) -> None:
        """Whitespace-only query prints a warning, no crash."""
        stderr = _capture_stderr(
            lambda: cli.answer(query="   ")
        )
        assert "warning" in stderr.lower() or "empty" in stderr.lower()

    def test_k_zero(self, cli: CLI) -> None:
        """k=0 prints a warning, no crash."""
        stderr = _capture_stderr(
            lambda: cli.answer(query="test", k=0)
        )
        assert "k" in stderr.lower() or "warning" in stderr.lower()

    def test_k_negative(self, cli: CLI) -> None:
        """k=-1 prints a warning, no crash."""
        stderr = _capture_stderr(
            lambda: cli.answer(query="test", k=-1)
        )
        assert "k" in stderr.lower() or "warning" in stderr.lower()

    def test_missing_index(self, cli: CLI) -> None:
        """Answer without a built index prints an error."""
        stderr = _capture_stderr(
            lambda: cli.answer(
                query="test",
                k=5,
                chunks_path="/nonexistent/chunks.jsonl",
                bm25_index_path="/nonexistent/bm25.pkl",
            )
        )
        assert "error" in stderr.lower()


# ── 6. Answer-Dataset Command Robustness ────────────────────────


class TestAnswerDatasetRobustness:
    """Robustness tests for the ``answer_dataset`` command."""

    def test_nonexistent_student_results(self, cli: CLI) -> None:
        """Nonexistent student results path prints an error."""
        stderr = _capture_stderr(
            lambda: cli.answer_dataset(
                student_search_results_path="/no/such/file.json"
            )
        )
        assert "error" in stderr.lower()

    def test_malformed_student_results(
        self, cli: CLI, malformed_json: Path
    ) -> None:
        """Malformed student results JSON prints an error."""
        stderr = _capture_stderr(
            lambda: cli.answer_dataset(
                student_search_results_path=str(malformed_json)
            )
        )
        assert "error" in stderr.lower()

    def test_wrong_schema_student_results(
        self, cli: CLI, wrong_schema_json: Path
    ) -> None:
        """Wrong schema prints an error."""
        stderr = _capture_stderr(
            lambda: cli.answer_dataset(
                student_search_results_path=str(wrong_schema_json)
            )
        )
        assert "error" in stderr.lower()

    def test_empty_path(self, cli: CLI) -> None:
        """Empty path prints an error, no crash."""
        stderr = _capture_stderr(
            lambda: cli.answer_dataset(
                student_search_results_path=""
            )
        )
        assert "error" in stderr.lower()


# ── 7. Cross-cutting Edge Cases ─────────────────────────────────


class TestCrossCuttingEdgeCases:
    """Edge cases that span multiple commands."""

    def test_search_with_unicode_query(self, cli: CLI) -> None:
        """Unicode queries don't crash (missing index is OK)."""
        stderr = _capture_stderr(
            lambda: cli.search(
                query="日本語のクエリ 🎉",
                chunks_path="/nonexistent/chunks.jsonl",
                bm25_index_path="/nonexistent/bm25.pkl",
            )
        )
        # May error due to missing index, but must not crash.
        assert "traceback" not in stderr.lower()

    def test_search_with_very_long_query(self, cli: CLI) -> None:
        """Very long queries don't crash."""
        long_query = "word " * 10000
        stderr = _capture_stderr(
            lambda: cli.search(
                query=long_query,
                chunks_path="/nonexistent/chunks.jsonl",
                bm25_index_path="/nonexistent/bm25.pkl",
            )
        )
        assert "traceback" not in stderr.lower()

    def test_index_with_binary_file_in_corpus(
        self, cli: CLI, tmp_dir: Path
    ) -> None:
        """Binary files in corpus are skipped gracefully."""
        corpus = tmp_dir / "bin_corpus"
        corpus.mkdir()
        # Create a binary file with .py extension
        (corpus / "binary.py").write_bytes(
            bytes(range(256)) * 10
        )
        # Should not crash.
        cli.index(
            corpus_dir=str(corpus),
            chunks_path=str(tmp_dir / "chunks.jsonl"),
            bm25_index_path=str(tmp_dir / "bm25.pkl"),
        )

    def test_index_with_empty_files(
        self, cli: CLI, tmp_dir: Path
    ) -> None:
        """Empty files in corpus are handled gracefully."""
        corpus = tmp_dir / "empty_corpus"
        corpus.mkdir()
        (corpus / "empty.py").write_text("", encoding="utf-8")
        (corpus / "whitespace.py").write_text(
            "   \n\n  \n", encoding="utf-8"
        )
        # Should not crash.
        cli.index(
            corpus_dir=str(corpus),
            chunks_path=str(tmp_dir / "chunks.jsonl"),
            bm25_index_path=str(tmp_dir / "bm25.pkl"),
        )

    def test_evaluate_valid_but_unmatched(
        self, cli: CLI,
        valid_student_results: Path,
        answered_dataset: Path,
    ) -> None:
        """Evaluate with valid files that have no matching IDs.

        The evaluation should produce 0% recall but not crash.
        """
        # The question_id matches but the sources don't overlap
        # so this should print results gracefully.
        cli.evaluate(
            student_search_results_path=str(
                valid_student_results
            ),
            dataset_path=str(answered_dataset),
        )
