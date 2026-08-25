import fire

from src.indexer import (
    DEFAULT_BM25_INDEX_PATH,
    DEFAULT_CHUNKS_PATH,
    build_index,
)


class CLI:
    """RAG CLI entry point."""

    def __init__(self) -> None:
        """Initialize CLI."""
        pass

    def index(
        self,
        corpus_dir: str = "data/raw",
        max_chunk_size: int = 2000,
        chunks_path: str = DEFAULT_CHUNKS_PATH,
        bm25_index_path: str = DEFAULT_BM25_INDEX_PATH,
    ) -> None:
        """Index a corpus directory for retrieval.

        Args:
            corpus_dir: path to the root directory of the corpus.
            max_chunk_size: maximum character length for any chunk.
            chunks_path: output path for the chunk registry JSONL.
            bm25_index_path: output path for the fitted BM25 pickle.
        """
        try:
            build_index(
                corpus_dir=corpus_dir,
                max_chunk_size=max_chunk_size,
                chunks_path=chunks_path,
                bm25_index_path=bm25_index_path,
            )
        except (FileNotFoundError, ValueError) as exc:
            print(f"Error: {exc}")


def main() -> None:
    """Main CLI entrypoint."""
    fire.Fire(CLI)


if __name__ == "__main__":
    main()
