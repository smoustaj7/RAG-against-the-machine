"""Tests for src/generation.py — prompt building and truncation logic.

The actual Qwen model is NOT loaded during these tests.  Instead we
mock the tokenizer and model so tests run instantly without downloading
weights.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.generation import (
    AnswerGenerator,
    _build_sources_block,
    _NO_SOURCES_ANSWER,
)


# -----------------------------------------------------------------------
# Fixtures: fake tokenizer that counts 1 token per whitespace-word
# -----------------------------------------------------------------------

class _FakeTokenizer:
    """Minimal tokenizer mock: each whitespace-separated word = 1 token."""

    def encode(
        self, text: str, add_special_tokens: bool = True, **kw: object,
    ) -> list[int]:
        tokens = text.split()
        return list(range(len(tokens)))

    def decode(
        self, token_ids: list[int], skip_special_tokens: bool = True,
    ) -> str:
        # For tests we just return a placeholder of N words
        return " ".join(f"w{i}" for i in token_ids)


@pytest.fixture
def fake_tokenizer() -> _FakeTokenizer:
    return _FakeTokenizer()


# -----------------------------------------------------------------------
# _build_sources_block
# -----------------------------------------------------------------------

class TestBuildSourcesBlock:
    """Tests for the token-aware source block builder."""

    def test_empty_sources(self, fake_tokenizer: _FakeTokenizer) -> None:
        result = _build_sources_block([], fake_tokenizer, max_tokens=100)
        assert result == "(no sources provided)"

    def test_single_short_source(
        self, fake_tokenizer: _FakeTokenizer,
    ) -> None:
        result = _build_sources_block(
            ["hello world"], fake_tokenizer, max_tokens=100,
        )
        assert "[Source 1]" in result
        assert "hello world" in result

    def test_multiple_sources_all_fit(
        self, fake_tokenizer: _FakeTokenizer,
    ) -> None:
        result = _build_sources_block(
            ["aaa", "bbb", "ccc"],
            fake_tokenizer,
            max_tokens=100,
        )
        assert "[Source 1]" in result
        assert "[Source 2]" in result
        assert "[Source 3]" in result

    def test_truncation_when_over_budget(
        self, fake_tokenizer: _FakeTokenizer,
    ) -> None:
        """With a very tight budget only the first source should appear."""
        long_text = " ".join(f"word{i}" for i in range(50))
        result = _build_sources_block(
            [long_text, "should not appear"],
            fake_tokenizer,
            max_tokens=10,
        )
        assert "[Source 1]" in result
        # Source 2 should not fit
        assert "[Source 2]" not in result

    def test_zero_budget(self, fake_tokenizer: _FakeTokenizer) -> None:
        result = _build_sources_block(
            ["anything"], fake_tokenizer, max_tokens=0,
        )
        # With zero budget nothing should be added
        assert result == ""


# -----------------------------------------------------------------------
# AnswerGenerator.generate  (model is mocked)
# -----------------------------------------------------------------------

class TestAnswerGeneratorGenerate:
    """Test the generate() method without loading a real model."""

    @staticmethod
    def _make_generator() -> AnswerGenerator:
        """Create an AnswerGenerator with mocked internals."""
        with patch.object(
            AnswerGenerator, "__init__", lambda self, **kw: None
        ):
            gen = AnswerGenerator()

        gen.model_name = "test-model"
        gen.max_context_tokens = 500
        gen.max_new_tokens = 64
        gen._device = "cpu"

        # Mock tokenizer
        mock_tok = MagicMock()
        mock_tok.encode.return_value = MagicMock(
            to=MagicMock(return_value=MagicMock(shape=[1, 10])),
        )
        mock_tok.apply_chat_template.return_value = "prompt"
        mock_tok.decode.return_value = "Generated answer text."
        gen._tokenizer = mock_tok

        # Mock model
        mock_model = MagicMock()
        import torch
        mock_model.generate.return_value = torch.tensor([[0] * 20])
        gen._model = mock_model

        return gen

    def test_no_sources_returns_canned(self) -> None:
        gen = self._make_generator()
        result = gen.generate("some question", source_texts=None)
        assert result == _NO_SOURCES_ANSWER

    def test_empty_sources_returns_canned(self) -> None:
        gen = self._make_generator()
        result = gen.generate("some question", source_texts=[])
        assert result == _NO_SOURCES_ANSWER

    def test_empty_question(self) -> None:
        gen = self._make_generator()
        result = gen.generate("", source_texts=["some text"])
        assert "No question" in result

    def test_whitespace_question(self) -> None:
        gen = self._make_generator()
        result = gen.generate("   ", source_texts=["some text"])
        assert "No question" in result

    def test_normal_generation(self) -> None:
        gen = self._make_generator()
        result = gen.generate(
            "What is foo?",
            source_texts=["def foo(): pass"],
        )
        assert isinstance(result, str)
        assert len(result) > 0
