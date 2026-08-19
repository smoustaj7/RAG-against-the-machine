"""Chunking strategies for source code and documentation."""

import ast
import hashlib
import re
from pathlib import Path
from typing import List, Union

from src.chunk_store import Chunk


SUPPORTED_EXTENSIONS = {".py", ".md", ".rst", ".txt"}

IGNORED_DIRS = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".env",
    "node_modules",
    "build",
    "dist",
    ".egg-info",
}

IGNORED_EXTENSIONS = {
    ".lock",
    ".pyc",
    ".so",
    ".dll",
    ".exe",
    ".bin",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".pdf",
    ".zip",
    ".tar",
    ".gz",
    ".pkl",
    ".pt",
    ".safetensors",
    ".onnx",
    ".db",
    ".sqlite",
}


def compute_file_hash(content: str) -> str:
    """Compute SHA256 hex digest of source content."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def should_index_file(file_path: Union[str, Path]) -> bool:
    """Determine whether a file should be indexed."""
    path = Path(file_path)

    if path.suffix.lower() in IGNORED_EXTENSIONS:
        return False

    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        return False

    for part in path.parts:
        if part.startswith(".") and part not in {".", ".."}:
            return False
        if part in IGNORED_DIRS:
            return False

    return True


def create_chunk(
    file_path: str,
    first_idx: int,
    last_idx: int,
    text: str,
    file_hash: str,
) -> Chunk:
    """Create a Chunk object with deterministic chunk_id."""
    chunk_id = f"{file_path}:{first_idx}-{last_idx}"
    return Chunk(
        chunk_id=chunk_id,
        file_path=file_path,
        first_character_index=first_idx,
        last_character_index=last_idx,
        text=text,
        file_hash=file_hash,
    )


class TextChunker:
    """Fallback recursive text chunker respecting max_chunk_size."""

    @classmethod
    def chunk(
        cls,
        file_path: str,
        content: str,
        file_hash: str,
        start_offset: int = 0,
        max_chunk_size: int = 2000,
        overlap_size: int = 200,
    ) -> List[Chunk]:
        """Split text into chunks guaranteed to be <= max_chunk_size."""
        if not content:
            return []

        chunks: List[Chunk] = []
        n = len(content)
        start = 0

        actual_overlap = min(overlap_size, max_chunk_size // 4)

        separators = ["\n\n", "\n", ". ", " ", ""]

        while start < n:
            end = min(start + max_chunk_size, n)

            if end < n:
                cut_found = False
                for sep in separators:
                    if sep == "":
                        break
                    idx = content.rfind(sep, start + actual_overlap, end)
                    if idx != -1 and idx > start:
                        end = idx + len(sep)
                        cut_found = True
                        break

                if not cut_found:
                    end = min(start + max_chunk_size, n)

            if end <= start:
                end = min(start + max_chunk_size, n)

            first_idx = start_offset + start
            last_idx = start_offset + end
            chunk_text = content[start:end]

            chunks.append(
                create_chunk(
                    file_path=file_path,
                    first_idx=first_idx,
                    last_idx=last_idx,
                    text=chunk_text,
                    file_hash=file_hash,
                )
            )

            if end >= n:
                break
            next_start = end - actual_overlap if actual_overlap > 0 else end
            if next_start <= start:
                next_start = end
            start = next_start

        return chunks


class PythonChunker:
    """AST-aware chunker for Python source files."""

    @staticmethod
    def _compute_line_starts(content: str) -> List[int]:
        line_starts = [0]
        for i, char in enumerate(content):
            if char == "\n":
                line_starts.append(i + 1)
        return line_starts

    @classmethod
    def chunk(
        cls,
        file_path: str,
        content: str,
        file_hash: str,
        max_chunk_size: int = 2000,
    ) -> List[Chunk]:
        if not content:
            return []

        try:
            tree = ast.parse(content)
        except SyntaxError:
            return TextChunker.chunk(
                file_path=file_path,
                content=content,
                file_hash=file_hash,
                max_chunk_size=max_chunk_size,
            )

        line_starts = PythonChunker._compute_line_starts(content)

        def get_node_offsets(node: ast.AST) -> tuple[int, int]:
            lineno = getattr(node, "lineno", 1)
            col_offset = getattr(node, "col_offset", 0)
            end_lineno = getattr(node, "end_lineno", lineno)
            end_col_offset = getattr(node, "end_col_offset", col_offset)

            start = line_starts[lineno - 1] + col_offset
            end = line_starts[end_lineno - 1] + end_col_offset
            return start, end

        boundaries: List[tuple[int, int]] = []
        for body_item in tree.body:
            if hasattr(body_item, "lineno") and hasattr(
                body_item, "end_lineno"
            ):
                s, e = get_node_offsets(body_item)
                if s < e:
                    boundaries.append((s, e))

        if not boundaries:
            return TextChunker.chunk(
                file_path=file_path,
                content=content,
                file_hash=file_hash,
                max_chunk_size=max_chunk_size,
            )

        intervals: List[tuple[int, int]] = []
        curr = 0
        for s, e in boundaries:
            if s > curr:
                intervals.append((curr, s))
            intervals.append((s, e))
            curr = max(curr, e)
        if curr < len(content):
            intervals.append((curr, len(content)))

        chunks: List[Chunk] = []
        group_start: int = -1
        group_end: int = -1

        for s, e in intervals:
            if s == e:
                continue

            if e - s > max_chunk_size:
                if group_start != -1 and group_end != -1:
                    chunks.append(
                        create_chunk(
                            file_path=file_path,
                            first_idx=group_start,
                            last_idx=group_end,
                            text=content[group_start:group_end],
                            file_hash=file_hash,
                        )
                    )
                    group_start = -1
                    group_end = -1

                sub_chunks = TextChunker.chunk(
                    file_path=file_path,
                    content=content[s:e],
                    file_hash=file_hash,
                    start_offset=s,
                    max_chunk_size=max_chunk_size,
                )
                chunks.extend(sub_chunks)
                continue

            if group_start == -1:
                group_start, group_end = s, e
            elif e - group_start <= max_chunk_size:
                group_end = e
            else:
                chunks.append(
                    create_chunk(
                        file_path=file_path,
                        first_idx=group_start,
                        last_idx=group_end,
                        text=content[group_start:group_end],
                        file_hash=file_hash,
                    )
                )
                group_start, group_end = s, e

        if group_start != -1 and group_end != -1:
            chunks.append(
                create_chunk(
                    file_path=file_path,
                    first_idx=group_start,
                    last_idx=group_end,
                    text=content[group_start:group_end],
                    file_hash=file_hash,
                )
            )

        return chunks


class MarkdownChunker:
    """Header-aware chunker for Markdown files."""

    HEADER_PATTERN = re.compile(r"^(#{1,6})\s+.*$", re.MULTILINE)

    @classmethod
    def chunk(
        cls,
        file_path: str,
        content: str,
        file_hash: str,
        max_chunk_size: int = 2000,
    ) -> List[Chunk]:
        if not content:
            return []

        matches = list(cls.HEADER_PATTERN.finditer(content))
        if not matches:
            return TextChunker.chunk(
                file_path=file_path,
                content=content,
                file_hash=file_hash,
                max_chunk_size=max_chunk_size,
            )
        split_indices = [0]
        for m in matches:
            if m.start() > 0 and m.start() not in split_indices:
                split_indices.append(m.start())
        split_indices.append(len(content))

        intervals: List[tuple[int, int]] = []
        for i in range(len(split_indices) - 1):
            s, e = split_indices[i], split_indices[i + 1]
            if s < e:
                intervals.append((s, e))

        chunks: List[Chunk] = []
        group_start: int = -1
        group_end: int = -1

        for s, e in intervals:
            if e - s > max_chunk_size:
                if group_start != -1 and group_end != -1:
                    chunks.append(
                        create_chunk(
                            file_path=file_path,
                            first_idx=group_start,
                            last_idx=group_end,
                            text=content[group_start:group_end],
                            file_hash=file_hash,
                        )
                    )
                    group_start = -1
                    group_end = -1

                sub_chunks = TextChunker.chunk(
                    file_path=file_path,
                    content=content[s:e],
                    file_hash=file_hash,
                    start_offset=s,
                    max_chunk_size=max_chunk_size,
                )
                chunks.extend(sub_chunks)
                continue

            if group_start == -1:
                group_start, group_end = s, e
            elif e - group_start <= max_chunk_size:
                group_end = e
            else:
                chunks.append(
                    create_chunk(
                        file_path=file_path,
                        first_idx=group_start,
                        last_idx=group_end,
                        text=content[group_start:group_end],
                        file_hash=file_hash,
                    )
                )
                group_start, group_end = s, e

        if group_start != -1 and group_end != -1:
            chunks.append(
                create_chunk(
                    file_path=file_path,
                    first_idx=group_start,
                    last_idx=group_end,
                    text=content[group_start:group_end],
                    file_hash=file_hash,
                )
            )

        return chunks


def chunk_file(
    file_path: Union[str, Path],
    content: str,
    max_chunk_size: int = 2000,
) -> List[Chunk]:
    """Chunk a file according to its format and max size constraint."""
    str_path = str(file_path)
    file_hash = compute_file_hash(content)
    suffix = Path(str_path).suffix.lower()

    if suffix == ".py":
        return PythonChunker.chunk(
            file_path=str_path,
            content=content,
            file_hash=file_hash,
            max_chunk_size=max_chunk_size,
        )
    elif suffix in {".md", ".rst"}:
        return MarkdownChunker.chunk(
            file_path=str_path,
            content=content,
            file_hash=file_hash,
            max_chunk_size=max_chunk_size,
        )
    else:
        return TextChunker.chunk(
            file_path=str_path,
            content=content,
            file_hash=file_hash,
            max_chunk_size=max_chunk_size,
        )
