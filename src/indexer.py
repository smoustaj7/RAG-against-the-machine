import sys
from pathlib import Path
from typing import Optional, Union

from tqdm import tqdm

from src.chunk_store import ChunkStore
from src.chunking import chunk_file, should_index_file
from src.retrieval.base import Retriever
from src.retrieval.embedding import (
    DEFAULT_EMBEDDING_MODEL,
    EmbeddingRetriever,
)
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.lexical import BM25Retriever


DEFAULT_CHUNKS_PATH = "data/processed/chunks.jsonl"
DEFAULT_BM25_INDEX_PATH = "data/processed/bm25_index.pkl"
DEFAULT_EMBEDDING_INDEX_PATH = "data/processed/embedding_index.npz"
DEFAULT_HYBRID_INDEX_PATH = "data/processed/hybrid_index"

RETRIEVER_CHOICES = ("bm25", "embedding", "hybrid")

_DEFAULT_INDEX_PATHS = {
    "bm25": DEFAULT_BM25_INDEX_PATH,
    "embedding": DEFAULT_EMBEDDING_INDEX_PATH,
    "hybrid": DEFAULT_HYBRID_INDEX_PATH,
}


def normalize_retriever_name(retriever: str) -> str:
    """Validate and canonicalise a retriever name.

    Args:
        retriever: user-supplied retriever name (any casing).

    Returns:
        The canonical lowercase name.

    Raises:
        ValueError: if the name is not a known retriever.
    """
    name = str(retriever or "").strip().lower()
    if name not in RETRIEVER_CHOICES:
        raise ValueError(
            f"Unknown retriever: {retriever!r}. "
            f"Choose one of: {', '.join(RETRIEVER_CHOICES)}."
        )
    return name


def default_index_path(retriever: str) -> str:
    """Return the default artifact path for a retriever.

    Args:
        retriever: retriever name.

    Returns:
        The default path (a file for ``bm25``/``embedding``,
        a directory for ``hybrid``).

    Raises:
        ValueError: if the name is not a known retriever.
    """
    return _DEFAULT_INDEX_PATHS[normalize_retriever_name(retriever)]


def make_retriever(
    retriever: str,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
) -> Retriever:
    """Build an unfitted retriever by name.

    Args:
        retriever: one of ``bm25``, ``embedding``, ``hybrid``.
        embedding_model: encoder used by the semantic components.

    Returns:
        An unfitted Retriever instance.

    Raises:
        ValueError: if the name is not a known retriever.
    """
    name = normalize_retriever_name(retriever)
    if name == "bm25":
        return BM25Retriever()
    if name == "embedding":
        return EmbeddingRetriever(model_name=embedding_model)
    return HybridRetriever(
        embedding=EmbeddingRetriever(model_name=embedding_model)
    )


def load_retriever(
    retriever: str,
    index_path: Union[str, Path],
) -> Retriever:
    """Load a fitted retriever from its persisted artifact.

    Args:
        retriever: one of ``bm25``, ``embedding``, ``hybrid``.
        index_path: path to the artifact written by ``build_index``.

    Returns:
        A fitted Retriever instance.

    Raises:
        ValueError: if the name is not a known retriever.
        FileNotFoundError: if the artifact is missing.
    """
    name = normalize_retriever_name(retriever)
    if name == "bm25":
        return BM25Retriever.load(index_path)
    if name == "embedding":
        return EmbeddingRetriever.load(index_path)
    return HybridRetriever.load(index_path)


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


def resolve_index_path(
    retriever: str,
    index_path: Optional[Union[str, Path]] = None,
    bm25_index_path: Union[str, Path] = DEFAULT_BM25_INDEX_PATH,
) -> Path:
    """Work out where a retriever's artifact lives.

    An explicit ``index_path`` always wins. Otherwise ``bm25`` falls
    back to ``bm25_index_path`` (so the original CLI flag keeps
    working) and the other retrievers use their own defaults.

    Args:
        retriever: retriever name.
        index_path: explicit override, if the caller gave one.
        bm25_index_path: legacy BM25-specific path argument.

    Returns:
        The resolved artifact path.

    Raises:
        ValueError: if the name is not a known retriever.
    """
    name = normalize_retriever_name(retriever)
    if index_path is not None and str(index_path).strip():
        return Path(index_path)
    if name == "bm25":
        return Path(bm25_index_path)
    return Path(default_index_path(name))


def build_index(
    corpus_dir: Union[str, Path],
    max_chunk_size: int = 2000,
    chunks_path: Union[str, Path] = DEFAULT_CHUNKS_PATH,
    bm25_index_path: Union[str, Path] = DEFAULT_BM25_INDEX_PATH,
    retriever: str = "bm25",
    index_path: Optional[Union[str, Path]] = None,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
) -> None:
    """Run the full pipeline: walk -> chunk -> fit -> persist.

    The chunk registry is written once and shared by every
    retriever, so switching retrievers never re-chunks the corpus.

    Args:
        corpus_dir: path to the root directory of the corpus.
        max_chunk_size: maximum character length for any chunk.
        chunks_path: output path for the chunk registry JSONL.
        bm25_index_path: output path for the fitted BM25 pickle
            (used only when ``retriever`` is ``bm25`` and
            ``index_path`` is not given).
        retriever: one of ``bm25``, ``embedding``, ``hybrid``.
        index_path: explicit output path for the fitted retriever,
            overriding the per-retriever default.
        embedding_model: encoder used by the semantic retrievers.

    Raises:
        FileNotFoundError: if corpus_dir does not exist.
        ValueError: if corpus_dir is not a directory, or if
            ``retriever`` is not a known retriever.
    """
    retriever_name = normalize_retriever_name(retriever)
    index_out = resolve_index_path(
        retriever_name, index_path, bm25_index_path
    )

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

    print(f"Fitting '{retriever_name}' index...")
    retriever_impl = make_retriever(
        retriever_name, embedding_model=embedding_model
    )
    all_chunks = {c.chunk_id: c for c in store.get_all_chunks()}

    retriever_impl.index(all_chunks)
    print(f"'{retriever_name}' index fitted.")

    retriever_impl.save(index_out)
    print(f"'{retriever_name}' index saved to: {index_out}")

    print("Indexing complete! 🎉")
