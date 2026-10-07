
import hashlib
import sys
from pathlib import Path
from typing import Any, List, Tuple, cast

import diskcache  # type: ignore[import-untyped]

from src.chunk_store import ChunkStore
from src.retrieval.base import Retriever

DEFAULT_CACHE_DIR = "data/processed/.cache"


def _get_cache(cache_dir: str = DEFAULT_CACHE_DIR) -> diskcache.Cache:
    """Return a shared ``diskcache.Cache`` instance.

    Args:
        cache_dir: directory where cache files are stored.

    Returns:
        An open ``diskcache.Cache``.
    """
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    return diskcache.Cache(
        directory=cache_dir,
        size_limit=512 * 1024 * 1024,
    )


def _artifact_key(path: str) -> str:
    """Build a cache key component from a file/directory's mtime+size.

    If *path* is a directory (e.g. the hybrid index dir), the key
    incorporates the most recent mtime across all contained files.

    Args:
        path: filesystem path to the artifact.

    Returns:
        A string like ``<canonical_path>:<mtime>:<size>`` that changes
        whenever the file is rewritten.
    """
    p = Path(path).resolve()
    if p.is_dir():
        max_mtime: float = 0.0
        total_size: int = 0
        for child in p.rglob("*"):
            if child.is_file():
                st = child.stat()
                max_mtime = max(max_mtime, st.st_mtime)
                total_size += st.st_size
        return f"{p}:{max_mtime}:{total_size}"
    st = p.stat()
    return f"{p}:{st.st_mtime}:{st.st_size}"


def cached_load_retriever(
    retriever_name: str,
    index_path: str,
    load_fn: "type[Retriever] | None" = None,
    cache_dir: str = DEFAULT_CACHE_DIR,
) -> Retriever:
    """Load a retriever, returning a cached instance when possible.

    The first call for a given ``(retriever_name, index_path)``
    delegates to :func:`src.indexer.load_retriever` and stores
    the result in the disk cache.  Subsequent calls with the *same*
    artifact on disk (same mtime + size) return instantly.

    Args:
        retriever_name: one of ``bm25``, ``embedding``, ``hybrid``.
        index_path: path to the fitted retriever artifact.
        load_fn: optional override for the load callable (testing).
        cache_dir: disk cache directory.

    Returns:
        A fitted Retriever instance (possibly from cache).
    """
    cache = _get_cache(cache_dir)
    try:
        artifact_version = _artifact_key(index_path)
    except (OSError, FileNotFoundError):
        from src.indexer import load_retriever as _load
        return _load(retriever_name, index_path)

    cache_key = f"retriever:{retriever_name}:{artifact_version}"

    result = cache.get(cache_key)
    if result is not None:
        print(
            f"  [cache] '{retriever_name}' retriever loaded from cache.",
            file=sys.stderr,
        )
        return cast(Retriever, result)

    from src.indexer import load_retriever as _load
    retriever = _load(retriever_name, index_path)
    try:
        cache.set(cache_key, retriever)
    except Exception:
        pass
    return retriever


def cached_load_chunk_store(
    chunks_path: str,
    cache_dir: str = DEFAULT_CACHE_DIR,
) -> ChunkStore:
    """Load a chunk store, returning a cached instance when possible.

    Args:
        chunks_path: path to the chunk registry JSONL.
        cache_dir: disk cache directory.

    Returns:
        A ChunkStore instance (possibly from cache).
    """
    cache = _get_cache(cache_dir)
    try:
        artifact_version = _artifact_key(chunks_path)
    except (OSError, FileNotFoundError):
        return ChunkStore.load_jsonl(chunks_path)

    cache_key = f"chunk_store:{artifact_version}"

    result = cache.get(cache_key)
    if result is not None:
        print(
            "  [cache] Chunk store loaded from cache.",
            file=sys.stderr,
        )
        return cast(ChunkStore, result)

    store = ChunkStore.load_jsonl(chunks_path)
    try:
        cache.set(cache_key, store)
    except Exception:
        pass
    return store


def _search_cache_key(
    retriever_name: str,
    index_path: str,
    query: str,
    k: int,
) -> str:
    """Build a deterministic cache key for a search request.

    The key incorporates the artifact version so that stale results
    are never returned after a re-index.

    Args:
        retriever_name: retriever name.
        index_path: path to the retriever artifact.
        query: the query string.
        k: number of results.

    Returns:
        A unique cache key string.
    """
    try:
        artifact_version = _artifact_key(index_path)
    except (OSError, FileNotFoundError):
        artifact_version = "unknown"
    query_hash = hashlib.sha256(query.encode("utf-8")).hexdigest()[:16]
    return f"search:{retriever_name}:{artifact_version}:{query_hash}:{k}"


def cached_search(
    retriever: Retriever,
    retriever_name: str,
    index_path: str,
    query: str,
    k: int,
    cache_dir: str = DEFAULT_CACHE_DIR,
) -> List[Tuple[str, float]]:
    """Execute a retriever search with result caching.

    Args:
        retriever: a fitted Retriever instance.
        retriever_name: retriever name (for the cache key).
        index_path: path to the retriever artifact (for invalidation).
        query: the query string.
        k: number of results to return.
        cache_dir: disk cache directory.

    Returns:
        List of (chunk_id, score) tuples.
    """
    cache = _get_cache(cache_dir)
    cache_key = _search_cache_key(retriever_name, index_path, query, k)

    result = cache.get(cache_key)
    if result is not None:
        return cast(List[Tuple[str, float]], result)

    results = retriever.search(query, k)
    try:
        cache.set(cache_key, results)
    except Exception:
        pass
    return results


def clear_cache(cache_dir: str = DEFAULT_CACHE_DIR) -> int:
    """Clear the entire retrieval cache.

    Args:
        cache_dir: disk cache directory.

    Returns:
        Number of entries that were evicted.
    """
    cache = _get_cache(cache_dir)
    count = len(cache)
    cache.clear()
    return count


def cache_stats(cache_dir: str = DEFAULT_CACHE_DIR) -> dict[str, Any]:
    """Return basic cache statistics.

    Args:
        cache_dir: disk cache directory.

    Returns:
        Dictionary with ``entries``, ``size_bytes``, and ``directory``.
    """
    cache = _get_cache(cache_dir)
    return {
        "entries": len(cache),
        "size_bytes": cache.volume(),
        "directory": str(cache.directory),
    }
