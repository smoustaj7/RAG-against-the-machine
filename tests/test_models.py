"""Unit tests for Phase 1 data models."""

import json
from pathlib import Path
from uuid import UUID
from src.models import (
    AnsweredQuestion,
    MinimalAnswer,
    MinimalSearchResults,
    MinimalSource,
    RagDataset,
    StudentSearchResults,
    StudentSearchResultsAndAnswer,
    UnansweredQuestion,
)


def test_minimal_source() -> None:
    source = MinimalSource(
        file_path="data/raw/vllm-0.10.1/vllm/test.py",
        first_character_index=10,
        last_character_index=100,
    )
    assert source.file_path == "data/raw/vllm-0.10.1/vllm/test.py"
    assert source.first_character_index == 10
    assert source.last_character_index == 100


def test_unanswered_question() -> None:
    q = UnansweredQuestion(question="What is vLLM?")
    assert isinstance(q.question_id, UUID)
    assert q.question == "What is vLLM?"


def test_answered_question() -> None:
    source = MinimalSource(
        file_path="test.py", first_character_index=0, last_character_index=50
    )
    q = AnsweredQuestion(
        question="How does attention work?",
        sources=[source],
        answer="Attention is all you need.",
    )
    assert q.answer == "Attention is all you need."
    assert len(q.sources) == 1


def test_rag_dataset_load_answered() -> None:
    path = Path("datasets_public/public/AnsweredQuestions/dataset_code_public.json")
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        dataset = RagDataset.model_validate(data)
        assert len(dataset.rag_questions) > 0
        assert isinstance(dataset.rag_questions[0], AnsweredQuestion)


def test_rag_dataset_load_unanswered() -> None:
    path = Path("datasets_public/public/UnansweredQuestions/dataset_code_public.json")
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        dataset = RagDataset.model_validate(data)
        assert len(dataset.rag_questions) > 0
        assert isinstance(dataset.rag_questions[0], UnansweredQuestion)


def test_minimal_search_results() -> None:
    source = MinimalSource(
        file_path="test.py", first_character_index=0, last_character_index=50
    )
    res = MinimalSearchResults(
        question="What is this?",
        retrieved_sources=[source],
    )
    assert isinstance(res.question_id, UUID)
    assert len(res.retrieved_sources) == 1


def test_minimal_answer() -> None:
    source = MinimalSource(
        file_path="test.py", first_character_index=0, last_character_index=50
    )
    ans = MinimalAnswer(
        question="What is this?",
        retrieved_sources=[source],
        answer="This is a test.",
    )
    assert ans.answer == "This is a test."


def test_student_search_results() -> None:
    source = MinimalSource(
        file_path="test.py", first_character_index=0, last_character_index=50
    )
    res = MinimalSearchResults(
        question="What is this?",
        retrieved_sources=[source],
    )
    student_res = StudentSearchResults(search_results=[res], k=10)
    assert student_res.k == 10
    assert len(student_res.search_results) == 1

    dumped = json.loads(student_res.model_dump_json())
    assert "search_results" in dumped
    assert dumped["k"] == 10


def test_student_search_results_and_answer() -> None:
    source = MinimalSource(
        file_path="test.py", first_character_index=0, last_character_index=50
    )
    ans = MinimalAnswer(
        question="What is this?",
        retrieved_sources=[source],
        answer="Answer text",
    )
    student_ans = StudentSearchResultsAndAnswer(search_results=[ans])
    assert len(student_ans.results) == 1
    assert len(student_ans.search_results) == 1
