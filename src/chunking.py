import ast
import hashlib
import re
from pathlib import Path
from typing import List, Optional, Tuple, Union

from .chunk_store import Chunk


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


def _merge_intervals_into_chunks(
    file_path: str,
    content: str,
    file_hash: str,
    intervals: List[Tuple[int, int]],
    max_chunk_size: int,
) -> List[Chunk]:
    """Merge a list of (start, end) intervals into chunks respecting max size.

    Small consecutive intervals are grouped together.  Oversized intervals
    are sub-chunked via TextChunker.
    """
    chunks: List[Chunk] = []
    group_start: int = -1
    group_end: int = -1

    for s, e in intervals:
        if s == e:
            continue

        if e - s > max_chunk_size:
            # Flush any accumulated group first.
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
                    # Search the full window for a natural break point.
                    # The overlap is applied when computing next_start,
                    # not by restricting the search window.
                    idx = content.rfind(sep, start, end)
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

        line_starts = cls._compute_line_starts(content)

        def _line_col_to_offset(
            lineno: int, col_offset: int
        ) -> int:
            return line_starts[lineno - 1] + col_offset

        def get_node_offsets(node: ast.AST) -> Tuple[int, int]:
            """Return (start, end) byte offsets, including decorators."""
            lineno = getattr(node, "lineno", 1)
            col_offset = getattr(node, "col_offset", 0)
            end_lineno = getattr(node, "end_lineno", lineno)
            end_col_offset = getattr(
                node, "end_col_offset", col_offset
            )

            start = _line_col_to_offset(lineno, col_offset)

            decorator_list: List[ast.AST] = getattr(
                node, "decorator_list", []
            )
            for dec in decorator_list:
                dec_lineno = getattr(dec, "lineno", lineno)
                dec_start = line_starts[dec_lineno - 1]
                if dec_start < start:
                    start = dec_start

            end = _line_col_to_offset(end_lineno, end_col_offset)
            return start, end

        boundaries: List[Tuple[int, int]] = []
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

        intervals: List[Tuple[int, int]] = []
        curr = 0
        for s, e in boundaries:
            interval_start = curr if curr < s else s
            intervals.append((interval_start, e))
            curr = max(curr, e)
        if curr < len(content):
            if intervals:
                last_start, _ = intervals[-1]
                intervals[-1] = (last_start, len(content))
            else:
                intervals.append((curr, len(content)))

        return _merge_intervals_into_chunks(
            file_path=file_path,
            content=content,
            file_hash=file_hash,
            intervals=intervals,
            max_chunk_size=max_chunk_size,
        )


class MarkdownChunker:
    """Header-aware chunker for Markdown and RST files."""

    _MD_HEADER = re.compile(r"^(#{1,6})\s+.*$", re.MULTILINE)

    _RST_HEADER = re.compile(
        r"^(.+)\n([=\-~^\"]{3,})$", re.MULTILINE
    )

    @classmethod
    def _find_split_positions(
        cls, content: str
    ) -> Optional[List[int]]:
        md_matches = list(cls._MD_HEADER.finditer(content))
        rst_matches = list(cls._RST_HEADER.finditer(content))

        if md_matches and len(md_matches) >= len(rst_matches):
            matches = md_matches
        elif rst_matches:
            matches = rst_matches
        else:
            return None

        split_indices = [0]
        for m in matches:
            if m.start() > 0 and m.start() not in split_indices:
                split_indices.append(m.start())
        split_indices.append(len(content))
        return split_indices

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

        split_indices = cls._find_split_positions(content)
        if split_indices is None:
            return TextChunker.chunk(
                file_path=file_path,
                content=content,
                file_hash=file_hash,
                max_chunk_size=max_chunk_size,
            )

        intervals: List[Tuple[int, int]] = []
        for i in range(len(split_indices) - 1):
            s, e = split_indices[i], split_indices[i + 1]
            if s < e:
                intervals.append((s, e))

        return _merge_intervals_into_chunks(
            file_path=file_path,
            content=content,
            file_hash=file_hash,
            intervals=intervals,
            max_chunk_size=max_chunk_size,
        )


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
