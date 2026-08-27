"""Tests for the code-aware tokenizer."""

from src.tokenizer import tokenize


class TestTokenizeBasic:
    """Basic tokenization behaviour."""

    def test_empty_string(self) -> None:
        assert tokenize("") == []

    def test_single_word(self) -> None:
        result = tokenize("retriever")
        assert result == ["retriever"]

    def test_lowercases(self) -> None:
        result = tokenize("HELLO WORLD")
        assert result == ["hello", "world"]


class TestSnakeCaseSplitting:
    """Snake_case identifiers are split on underscores."""

    def test_simple_snake_case(self) -> None:
        result = tokenize("chunk_store")
        assert result == ["chunk", "store"]

    def test_triple_snake(self) -> None:
        # "all" is a stop-word and is filtered out.
        result = tokenize("get_all_chunks")
        assert result == ["get", "chunks"]

    def test_leading_underscore(self) -> None:
        # Leading underscore produces an empty segment which is skipped.
        result = tokenize("_private_method")
        assert "private" in result
        assert "method" in result


class TestCamelCaseSplitting:
    """camelCase and PascalCase identifiers are split on boundaries."""

    def test_camel_case(self) -> None:
        result = tokenize("camelCase")
        assert result == ["camel", "case"]

    def test_pascal_case(self) -> None:
        result = tokenize("PascalCase")
        assert result == ["pascal", "case"]

    def test_acronym_boundary(self) -> None:
        # XMLParser -> ["xml", "parser"]
        result = tokenize("XMLParser")
        assert "xml" in result or "xmlparser" in result
        assert "parser" in result

    def test_mixed_camel_and_snake(self) -> None:
        result = tokenize("get_camelCase_value")
        assert "get" in result
        assert "camel" in result
        assert "case" in result
        assert "value" in result


class TestPunctuationAndFiltering:
    """Punctuation stripping, short-token removal, stop-words."""

    def test_punctuation_stripped(self) -> None:
        result = tokenize("hello, world! foo.bar")
        assert "hello" in result
        assert "world" in result
        assert "foo" in result
        assert "bar" in result
        # No punctuation tokens
        assert "," not in result
        assert "!" not in result
        assert "." not in result

    def test_single_char_removed(self) -> None:
        result = tokenize("a b c hello")
        assert "hello" in result
        assert "a" not in result
        assert "b" not in result
        assert "c" not in result

    def test_stop_words_removed(self) -> None:
        result = tokenize("the quick brown fox")
        assert "the" not in result
        assert "quick" in result
        assert "brown" in result
        assert "fox" in result

    def test_code_query(self) -> None:
        """Realistic code search query."""
        result = tokenize("How does BM25Okapi handle empty documents?")
        assert "bm25" in result or "bm25okapi" in result
        assert "handle" in result
        assert "empty" in result
        assert "documents" in result
