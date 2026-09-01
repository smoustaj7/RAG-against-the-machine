import json
import sys
from pathlib import Path
import fire
from tqdm import tqdm

from src.chunk_store import ChunkStore
from src.evaluator import evaluate_search_results
from src.indexer import (
    DEFAULT_BM25_INDEX_PATH,
    DEFAULT_CHUNKS_PATH,
    build_index,
)
from src.models import (
    MinimalSearchResults,
    MinimalSource,
    RagDataset,
    StudentSearchResults,
)
from src.retrieval.lexical import BM25Retriever


def _load_retriever_and_store(
    chunks_path: str,
    bm25_index_path: str,
) -> tuple[BM25Retriever, ChunkStore]:
    """Load the persisted BM25 retriever and chunk store.

    Raises:
        FileNotFoundError: if the index or chunk files are missing.
    """
    if not Path(chunks_path).exists():
        raise FileNotFoundError(
            f"Chunk registry not found at: {chunks_path}\n"
            "Run 'index' first to build the index."
        )
    if not Path(bm25_index_path).exists():
        raise FileNotFoundError(
            f"BM25 index not found at: {bm25_index_path}\n"
            "Run 'index' first to build the index."
        )

    store = ChunkStore.load_jsonl(chunks_path)
    retriever = BM25Retriever.load(bm25_index_path)
    return retriever, store


def _search_single(
    query: str,
    k: int,
    retriever: BM25Retriever,
    store: ChunkStore,
) -> list[MinimalSource]:
    """Run a single search and return MinimalSource results."""
    if k <= 0:
        return []
    if not query or not query.strip():
        return []

    results = retriever.search(query, k)
    sources: list[MinimalSource] = []
    for chunk_id, _score in results:
        chunk = store.get_chunk(chunk_id)
        if chunk is not None:
            sources.append(chunk.to_minimal_source())
    return sources


def _load_dataset(path: str) -> RagDataset:
    """Load and validate a RagDataset JSON file.

    Raises:
        FileNotFoundError: if the file doesn't exist.
        ValueError: if the JSON is malformed or doesn't match schema.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    try:
        raw = file_path.read_text(encoding="utf-8")
        data = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError(
            f"Malformed dataset file: {path}: {exc}"
        ) from exc

    if "rag_questions" not in data:
        raise ValueError(
            f"Dataset file missing 'rag_questions' key: {path}"
        )

    return RagDataset.model_validate(data)


def _load_student_results(path: str) -> StudentSearchResults:
    """Load and validate a StudentSearchResults JSON file.

    Raises:
        FileNotFoundError: if the file doesn't exist.
        ValueError: if the JSON is malformed or doesn't match schema.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(
            f"Student results file not found: {path}"
        )

    try:
        raw = file_path.read_text(encoding="utf-8")
        data = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError(
            f"Malformed student results file: {path}: {exc}"
        ) from exc

    if "search_results" not in data:
        raise ValueError(
            f"Student results file missing 'search_results' key: "
            f"{path}"
        )

    return StudentSearchResults.model_validate(data)


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
            print(f"Error: {exc}", file=sys.stderr)

    def search(
        self,
        query: str,
        k: int = 5,
        chunks_path: str = DEFAULT_CHUNKS_PATH,
        bm25_index_path: str = DEFAULT_BM25_INDEX_PATH,
    ) -> None:
        """Search the index for a single query.

        Args:
            query: the search query string.
            k: number of results to return.
            chunks_path: path to the chunk registry JSONL.
            bm25_index_path: path to the fitted BM25 pickle.
        """
        try:
            if k <= 0:
                print("Warning: k <= 0, no results.", file=sys.stderr)
                return
            if not query or not query.strip():
                print(
                    "Warning: empty query, no results.",
                    file=sys.stderr,
                )
                return

            retriever, store = _load_retriever_and_store(
                chunks_path, bm25_index_path
            )
            sources = _search_single(query, k, retriever, store)

            result = MinimalSearchResults(
                question=query,
                retrieved_sources=sources,
            )
            print(result.model_dump_json(indent=2))

        except (FileNotFoundError, ValueError) as exc:
            print(f"Error: {exc}", file=sys.stderr)

    def search_dataset(
        self,
        dataset_path: str,
        k: int = 5,
        save_directory: str = "data/output/search_results",
        chunks_path: str = DEFAULT_CHUNKS_PATH,
        bm25_index_path: str = DEFAULT_BM25_INDEX_PATH,
    ) -> None:
        """Batch-search over a dataset of questions.
        Loads an UnansweredQuestions dataset, retrieves sources for
        each question, and writes StudentSearchResults JSON.
        Args:
            dataset_path: path to the dataset JSON file.
            k: number of results per question.
            save_directory: directory to write results to.
                Defaults to ``data/output/search_results/``.
            chunks_path: path to the chunk registry JSONL.
            bm25_index_path: path to the fitted BM25 pickle.
        """
        try:
            dataset = _load_dataset(dataset_path)
            retriever, store = _load_retriever_and_store(
                chunks_path, bm25_index_path
            )

            search_results: list[MinimalSearchResults] = []

            for q in tqdm(
                dataset.rag_questions,
                desc="Searching",
                unit="question",
            ):
                sources = _search_single(
                    q.question, k, retriever, store
                )
                search_results.append(
                    MinimalSearchResults(
                        question_id=q.question_id,
                        question=q.question,
                        retrieved_sources=sources,
                    )
                )

            student_results = StudentSearchResults(
                search_results=search_results,
                k=k,
            )

            save_dir = Path(save_directory)
            save_dir.mkdir(parents=True, exist_ok=True)

            dataset_name = Path(dataset_path).stem
            output_path = save_dir / f"{dataset_name}.json"
            output_path.write_text(
                student_results.model_dump_json(indent=2),
                encoding="utf-8",
            )

            print(
                f"Search results saved to: {output_path}\n"
                f"  Questions: {len(search_results)}\n"
                f"  k: {k}"
            )

        except (FileNotFoundError, ValueError) as exc:
            print(f"Error: {exc}", file=sys.stderr)

    def evaluate(
        self,
        student_search_results_path: str,
        dataset_path: str,
    ) -> None:
        """Evaluate retrieval quality against ground truth.
        Computes recall@1, recall@3, recall@5, recall@10 using
        IoU >= 0.05 overlap (same file_path required).
        Args:
            student_search_results_path: path to the
                StudentSearchResults JSON.
            dataset_path: path to the AnsweredQuestions dataset JSON.
        """
        try:
            student_results = _load_student_results(
                student_search_results_path
            )
            dataset = _load_dataset(dataset_path)

            metrics = evaluate_search_results(
                student_results, dataset
            )

            print("=" * 50)
            print("  Retrieval Evaluation Results")
            print("=" * 50)
            for metric, value in sorted(metrics.items()):
                pct = value * 100
                print(f"  {metric:<12s}  {pct:6.2f}%")
            print("=" * 50)

            r5 = metrics.get("recall@5", 0.0) * 100
            if r5 >= 80:
                print("  ✅ Docs target: PASS (>= 80%)")
            elif r5 >= 50:
                print("  ✅ Code target: PASS (>= 50%)")
                print("  ⚠️  Docs target: FAIL (< 80%)")
            else:
                print("  ❌ Both targets: FAIL (< 50%)")

        except (FileNotFoundError, ValueError) as exc:
            print(f"Error: {exc}", file=sys.stderr)


def main() -> None:
    """Main CLI entrypoint."""
    fire.Fire(CLI)


if __name__ == "__main__":
    main()
