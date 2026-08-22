"""Unit tests for Phase 2 chunk registry and chunking strategies."""

import json
from pathlib import Path
from src.chunk_store import Chunk, ChunkStore
from src.chunking import (
    MarkdownChunker,
    PythonChunker,
    TextChunker,
    chunk_file,
    compute_file_hash,
    should_index_file,
)
from src.models import MinimalSource


def test_chunk_and_minimal_source_conversion() -> None:
    chunk = Chunk(
        chunk_id="test.py:0-50",
        file_path="test.py",
        first_character_index=0,
        last_character_index=50,
        text="print('hello world')",
        file_hash="abcdef123456",
    )
    source = chunk.to_minimal_source()
    assert isinstance(source, MinimalSource)
    assert source.file_path == "test.py"
    assert source.first_character_index == 0
    assert source.last_character_index == 50


def test_chunk_store_operations(tmp_path: Path) -> None:
    store = ChunkStore()
    assert len(store) == 0

    chunk1 = Chunk(
        chunk_id="file1.py:0-10",
        file_path="file1.py",
        first_character_index=0,
        last_character_index=10,
        text="0123456789",
        file_hash="hash1",
    )
    chunk2 = Chunk(
        chunk_id="file1.py:10-20",
        file_path="file1.py",
        first_character_index=10,
        last_character_index=20,
        text="abcdefghij",
        file_hash="hash1",
    )
    chunk3 = Chunk(
        chunk_id="file2.md:0-15",
        file_path="file2.md",
        first_character_index=0,
        last_character_index=15,
        text="Header text here",
        file_hash="hash2",
    )

    store.add_chunks([chunk1, chunk2, chunk3])
    assert len(store) == 3
    assert store.get_chunk("file1.py:0-10") == chunk1
    assert len(store.get_chunks_by_file("file1.py")) == 2

    # Test JSONL serialization / deserialization
    jsonl_file = tmp_path / "chunks.jsonl"
    store.save_jsonl(jsonl_file)
    assert jsonl_file.exists()

    loaded_store = ChunkStore.load_jsonl(jsonl_file)
    assert len(loaded_store) == 3
    assert loaded_store.get_chunk("file2.md:0-15") == chunk3

    store.clear()
    assert len(store) == 0


def test_should_index_file() -> None:
    assert should_index_file("vllm/model.py") is True
    assert should_index_file("docs/guide.md") is True
    assert should_index_file("README.rst") is True
    assert should_index_file("notes.txt") is True

    # Ignored
    assert should_index_file(".git/config") is False
    assert should_index_file("__pycache__/model.cpython-310.pyc") is False
    assert should_index_file("uv.lock") is False
    assert should_index_file("data/weights.pt") is False
    assert should_index_file("image.png") is False


def test_compute_file_hash() -> None:
    content = "def foo(): pass\n"
    h1 = compute_file_hash(content)
    h2 = compute_file_hash(content)
    assert h1 == h2
    assert len(h1) == 64  # SHA256 hex length


def test_text_chunker_invariants() -> None:
    content = "Line one.\nLine two.\nLine three.\nLine four.\n" * 50
    file_hash = compute_file_hash(content)
    max_chunk_size = 100

    chunks = TextChunker.chunk(
        file_path="sample.txt",
        content=content,
        file_hash=file_hash,
        max_chunk_size=max_chunk_size,
    )

    assert len(chunks) > 0
    for chunk in chunks:
        # Invariant 1: Max chunk size cap
        assert len(chunk.text) <= max_chunk_size
        # Invariant 2: Exact character slice identity
        assert (
            content[
                chunk.first_character_index:chunk.last_character_index
            ]
            == chunk.text
        )
        assert chunk.first_character_index >= 0
        assert chunk.last_character_index <= len(content)


def test_python_chunker_ast() -> None:
    code = (
        '"""Module docstring."""\n\n'
        "import os\n"
        "import sys\n\n"
        "class MyClass:\n"
        '    """Class docstring."""\n'
        "    def __init__(self, val: int):\n"
        "        self.val = val\n\n"
        "    def get_val(self) -> int:\n"
        "        return self.val\n\n"
        "def top_level_func():\n"
        '    return "hello"\n'
    )
    file_hash = compute_file_hash(code)
    max_chunk_size = 200

    chunks = chunk_file(
        file_path="test_module.py",
        content=code,
        max_chunk_size=max_chunk_size,
    )

    assert len(chunks) > 0
    for chunk in chunks:
        assert len(chunk.text) <= max_chunk_size
        assert (
            code[chunk.first_character_index:chunk.last_character_index]
            == chunk.text
        )


def test_python_chunker_syntax_error_fallback() -> None:
    invalid_code = "def broken_func(:\n    pass bad syntax {{{"
    file_hash = compute_file_hash(invalid_code)

    chunks = PythonChunker.chunk(
        file_path="broken.py",
        content=invalid_code,
        file_hash=file_hash,
        max_chunk_size=50,
    )

    assert len(chunks) > 0
    for chunk in chunks:
        assert len(chunk.text) <= 50
        assert (
            invalid_code[
                chunk.first_character_index:chunk.last_character_index
            ]
            == chunk.text
        )


def test_markdown_chunker() -> None:
    md_content = (
        "# Title Header\n\n"
        "This is the intro section under title header.\n\n"
        "## Subhead 1\n\n"
        "Detailed explanation in subhead 1 paragraph.\n\n"
        "## Subhead 2\n\n"
        "Detailed explanation in subhead 2 paragraph.\n"
    )

    chunks = chunk_file(
        file_path="doc.md",
        content=md_content,
        max_chunk_size=100,
    )

    assert len(chunks) > 0
    for chunk in chunks:
        assert len(chunk.text) <= 100
        assert (
            md_content[
                chunk.first_character_index:chunk.last_character_index
            ]
            == chunk.text
        )


def test_python_chunker_includes_decorators() -> None:
    """Decorators must be included in the same chunk as their function."""
    code = (
        "import os\n\n"
        "@staticmethod\n"
        "@some_decorator(arg=1)\n"
        "def decorated_func():\n"
        '    return "hello"\n'
    )
    file_hash = compute_file_hash(code)

    chunks = PythonChunker.chunk(
        file_path="deco.py",
        content=code,
        file_hash=file_hash,
        max_chunk_size=2000,
    )

    assert len(chunks) > 0
    # Find the chunk that contains the function body.
    func_chunks = [c for c in chunks if "decorated_func" in c.text]
    assert len(func_chunks) == 1
    # The decorator text must be in the same chunk.
    assert "@staticmethod" in func_chunks[0].text
    assert "@some_decorator" in func_chunks[0].text


def test_rst_chunker() -> None:
    """MarkdownChunker should split on RST-style underline headers."""
    rst_content = (
        "Title\n"
        "=====\n\n"
        "Introduction paragraph under the title.\n\n"
        "Subtitle\n"
        "--------\n\n"
        "Content under the subtitle.\n"
    )

    chunks = chunk_file(
        file_path="doc.rst",
        content=rst_content,
        max_chunk_size=80,
    )

    assert len(chunks) > 0
    for chunk in chunks:
        assert len(chunk.text) <= 80
        assert (
            rst_content[
                chunk.first_character_index:chunk.last_character_index
            ]
            == chunk.text
        )
    # There should be at least 2 chunks (one per header section).
    assert len(chunks) >= 2


def test_full_coverage_invariant() -> None:
    """The union of all chunk spans must cover the entire file content."""
    content = (
        "# Header\n\n"
        "Some text.\n\n"
        "## Another Header\n\n"
        "More text.\n"
    )
    chunks = chunk_file(
        file_path="coverage.md",
        content=content,
        max_chunk_size=2000,
    )

    # Collect all covered character positions.
    covered = set()
    for chunk in chunks:
        for i in range(chunk.first_character_index, chunk.last_character_index):
            covered.add(i)

    # Every position in the content must be covered.
    for i in range(len(content)):
        assert i in covered, f"Position {i} not covered by any chunk"
