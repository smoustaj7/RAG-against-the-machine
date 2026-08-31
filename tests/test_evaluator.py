"""Tests for the evaluation module."""

import pytest

from src.evaluator import (
    evaluate_search_results,
    question_hit_at_k,
    source_overlaps,
)
from src.models import (
    AnsweredQuestion,
    MinimalSearchResults,
    MinimalSource,
    RagDataset,
    StudentSearchResults,
)


# ── source_overlaps ─────────────────────────────────────────────


class TestSourceOverlaps:
    """Tests for IoU-based source overlap detection."""

    def test_different_file_paths_no_overlap(self) -> None:
        """Different file paths never overlap."""
        r = MinimalSource(
            file_path="a.py",
            first_character_index=0,
            last_character_index=100,
        )
        g = MinimalSource(
            file_path="b.py",
            first_character_index=0,
            last_character_index=100,
        )
        assert source_overlaps(r, g) is False

    def test_exact_match(self) -> None:
        """Exact same range should overlap."""
        s = MinimalSource(
            file_path="a.py",
            first_character_index=10,
            last_character_index=110,
        )
        assert source_overlaps(s, s) is True

    def test_partial_overlap_above_threshold(self) -> None:
        """Partial overlap with IoU >= 0.05 should match."""
        r = MinimalSource(
            file_path="a.py",
            first_character_index=0,
            last_character_index=100,
        )
        g = MinimalSource(
            file_path="a.py",
            first_character_index=90,
            last_character_index=200,
        )
        # intersection = 10, union = 100+110-10 = 200
        # IoU = 10/200 = 0.05 => exactly at threshold
        assert source_overlaps(r, g) is True

    def test_partial_overlap_below_threshold(self) -> None:
        """Overlap below IoU 0.05 should not match."""
        r = MinimalSource(
            file_path="a.py",
            first_character_index=0,
            last_character_index=100,
        )
        g = MinimalSource(
            file_path="a.py",
            first_character_index=99,
            last_character_index=200,
        )
        # intersection = 1, union = 100+101-1 = 200
        # IoU = 1/200 = 0.005 => below threshold
        assert source_overlaps(r, g) is False

    def test_no_overlap_at_all(self) -> None:
        """Disjoint ranges on same file should not match."""
        r = MinimalSource(
            file_path="a.py",
            first_character_index=0,
            last_character_index=50,
        )
        g = MinimalSource(
            file_path="a.py",
            first_character_index=100,
            last_character_index=200,
        )
        assert source_overlaps(r, g) is False

    def test_one_contains_the_other(self) -> None:
        """One range fully containing the other should overlap."""
        r = MinimalSource(
            file_path="a.py",
            first_character_index=10,
            last_character_index=50,
        )
        g = MinimalSource(
            file_path="a.py",
            first_character_index=0,
            last_character_index=100,
        )
        # intersection = 40, union = 40 + 100 - 40 = 100
        # IoU = 40/100 = 0.4
        assert source_overlaps(r, g) is True

    def test_zero_length_range(self) -> None:
        """Zero-length range should not cause errors."""
        r = MinimalSource(
            file_path="a.py",
            first_character_index=50,
            last_character_index=50,
        )
        g = MinimalSource(
            file_path="a.py",
            first_character_index=0,
            last_character_index=100,
        )
        assert source_overlaps(r, g) is False


# ── question_hit_at_k ───────────────────────────────────────────


class TestQuestionHitAtK:
    """Tests for recall hit detection at rank k."""

    def _make_source(
        self, path: str, start: int, end: int
    ) -> MinimalSource:
        return MinimalSource(
            file_path=path,
            first_character_index=start,
            last_character_index=end,
        )

    def test_hit_at_k1(self) -> None:
        """First result matches ground truth."""
        retrieved = [self._make_source("a.py", 0, 100)]
        gt = [self._make_source("a.py", 0, 100)]
        assert question_hit_at_k(retrieved, gt, k=1) is True

    def test_miss_at_k1_hit_at_k2(self) -> None:
        """Match is at position 2 — miss at k=1 but hit at k=2."""
        retrieved = [
            self._make_source("b.py", 0, 100),
            self._make_source("a.py", 0, 100),
        ]
        gt = [self._make_source("a.py", 0, 100)]
        assert question_hit_at_k(retrieved, gt, k=1) is False
        assert question_hit_at_k(retrieved, gt, k=2) is True

    def test_no_hit(self) -> None:
        """No retrieved source matches any ground truth."""
        retrieved = [self._make_source("x.py", 0, 100)]
        gt = [self._make_source("a.py", 0, 100)]
        assert question_hit_at_k(retrieved, gt, k=5) is False

    def test_empty_retrieved(self) -> None:
        """Empty retrieval list is a miss."""
        gt = [self._make_source("a.py", 0, 100)]
        assert question_hit_at_k([], gt, k=5) is False

    def test_empty_gt(self) -> None:
        """No ground truth sources is always a miss."""
        retrieved = [self._make_source("a.py", 0, 100)]
        assert question_hit_at_k(retrieved, [], k=5) is False


# ── evaluate_search_results ─────────────────────────────────────


class TestEvaluateSearchResults:
    """Integration tests for full evaluation pipeline."""

    def _make_answered(
        self, qid: str, path: str, start: int, end: int
    ) -> AnsweredQuestion:
        from uuid import UUID

        return AnsweredQuestion(
            question_id=UUID(qid),
            question="test?",
            answer="test answer",
            sources=[
                MinimalSource(
                    file_path=path,
                    first_character_index=start,
                    last_character_index=end,
                )
            ],
        )

    def _make_search_result(
        self,
        qid: str,
        sources: list[MinimalSource],
    ) -> MinimalSearchResults:
        from uuid import UUID

        return MinimalSearchResults(
            question_id=UUID(qid),
            question="test?",
            retrieved_sources=sources,
        )

    def test_perfect_recall(self) -> None:
        """All questions matched → 100% recall at all k."""
        qid = "12345678-1234-1234-1234-123456789abc"
        gt = self._make_answered(qid, "a.py", 0, 100)
        sr = self._make_search_result(
            qid,
            [MinimalSource(
                file_path="a.py",
                first_character_index=0,
                last_character_index=100,
            )],
        )

        dataset = RagDataset(rag_questions=[gt])
        student = StudentSearchResults(search_results=[sr], k=5)
        metrics = evaluate_search_results(student, dataset)

        assert metrics["recall@1"] == 1.0
        assert metrics["recall@5"] == 1.0

    def test_zero_recall(self) -> None:
        """No overlap → 0% recall at all k."""
        qid = "12345678-1234-1234-1234-123456789abc"
        gt = self._make_answered(qid, "a.py", 0, 100)
        sr = self._make_search_result(
            qid,
            [MinimalSource(
                file_path="b.py",
                first_character_index=0,
                last_character_index=100,
            )],
        )

        dataset = RagDataset(rag_questions=[gt])
        student = StudentSearchResults(search_results=[sr], k=5)
        metrics = evaluate_search_results(student, dataset)

        assert metrics["recall@1"] == 0.0
        assert metrics["recall@5"] == 0.0

    def test_empty_dataset(self) -> None:
        """Empty ground truth returns 0% recall."""
        from src.models import UnansweredQuestion

        dataset = RagDataset(
            rag_questions=[
                UnansweredQuestion(question="no answer here")
            ]
        )
        student = StudentSearchResults(search_results=[], k=5)
        metrics = evaluate_search_results(student, dataset)

        assert metrics["recall@1"] == 0.0
