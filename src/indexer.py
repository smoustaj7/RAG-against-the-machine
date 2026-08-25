import sys
from pathlib import Path
from typing import Optional, Union

from tqdm import tqdm

from src.chunk_store import ChunkStore
from src.chunking import chunk_file, should_index_file
from src.retrieval.lexical import BM25Retriever


DEFAULT_CHUNKS_PATH = "data/processed/chunks.jsonl"
DEFAULT_BM25_INDEX_PATH = "data/processed/bm25_index.pkl"


def _collect_files(
    corpus_dir: Path,
) -> list[Path]:
    """Recursively collect all indexable files under corpus_dir.

    Args:
        corpus_dir: root directory of the corpus to index.

    Returns:
        Sorted list of file paths that pass the indexing filter.
    """
    files: list[Path] = []
    for path in sorted(corpus_dir.rglob("*")):
        if path.is_file() and should_index_file(path):
            files.append(path)
    return files


def _read_file_safe(path: Path) -> Optional[str]:
    """Read a file, returning None on any read error.

    Handles UnicodeDecodeError, PermissionError, IsADirectoryError,
    and other OS-level errors gracefully.
    """
    try:
        content = path.read_text(encoding="utf-8")
        return content if content.strip() else None
    except (UnicodeDecodeError, PermissionError, IsADirectoryError,
            OSError) as exc:
        print(
            f"  [WARN] Skipping {path}: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return None


def build_index(
    corpus_dir: Union[str, Path],
    max_chunk_size: int = 2000,
    chunks_path: Union[str, Path] = DEFAULT_CHUNKS_PATH,
    bm25_index_path: Union[str, Path] = DEFAULT_BM25_INDEX_PATH,
) -> None:
    """Run the full indexing pipeline: walk -> chunk -> fit BM25 -> persist.

    Args:
        corpus_dir: path to the root directory of the corpus.
        max_chunk_size: maximum character length for any chunk.
        chunks_path: output path for the chunk registry JSONL.
        bm25_index_path: output path for the fitted BM25 pickle.

    Raises:
        FileNotFoundError: if corpus_dir does not exist.
        ValueError: if corpus_dir is not a directory.
    """
    corpus_path = Path(corpus_dir)
    if not corpus_path.exists():
        raise FileNotFoundError(
            f"Corpus directory not found: {corpus_path}"
        )
    if not corpus_path.is_dir():
        raise ValueError(
            f"Corpus path is not a directory: {corpus_path}"
        )

    print(f"Scanning corpus at: {corpus_path}")
    files = _collect_files(corpus_path)
    print(f"Found {len(files)} indexable files.")

    if not files:
        print("No files to index. Aborting. 🦀 🚨", file=sys.stderr)
        return

    store = ChunkStore()
    skipped = 0

    for file_path in tqdm(files, desc="Chunking files", unit="file"):
        content = _read_file_safe(file_path)
        if content is None:
            skipped += 1
            continue

        chunks = chunk_file(
            file_path=str(file_path),
            content=content,
            max_chunk_size=max_chunk_size,
        )
        store.add_chunks(chunks)

    total_chunks = len(store)
    print(
        f"Chunking complete: {total_chunks} chunks "
        f"from {len(files) - skipped} files "
        f"({skipped} skipped)."
    )

    if total_chunks == 0:
        print("No chunks produced. Aborting. 🦀 🚨", file=sys.stderr)
        return

    chunks_out = Path(chunks_path)
    store.save_jsonl(chunks_out)
    print(f"Chunk registry saved to: {chunks_out}")

    print("Fitting BM25 index...")
    retriever = BM25Retriever()
    all_chunks = {c.chunk_id: c for c in store.get_all_chunks()}

    retriever.index(all_chunks)
    print("BM25 index fitted.")

    bm25_out = Path(bm25_index_path)
    retriever.save(bm25_out)
    print(f"BM25 index saved to: {bm25_out}")

    print("Indexing complete! 🎉")
