import json
from pathlib import Path
import sys

import fire
from tqdm import tqdm

from src.chunk_store import ChunkStore
from src.evaluator import evaluate_search_results
from src.generation import AnswerGenerator
from src.indexer import (
    DEFAULT_BM25_INDEX_PATH,
    DEFAULT_CHUNKS_PATH,
    build_index,
    load_retriever,
    normalize_retriever_name,
    resolve_index_path,
)
from src.models import (
    MinimalAnswer,
    MinimalSearchResults,
    MinimalSource,
    RagDataset,
    StudentSearchResults,
    StudentSearchResultsAndAnswer,
)
from src.retrieval.base import Retriever
from src.retrieval.embedding import DEFAULT_EMBEDDING_MODEL


def _load_retriever_and_store(
    chunks_path: str,
    index_path: str,
    retriever: str = "bm25",
) -> tuple[Retriever, ChunkStore]:
    """Load the persisted retriever and chunk store.

    Args:
        chunks_path: path to the chunk registry JSONL.
        index_path: path to the fitted retriever artifact.
        retriever: one of ``bm25``, ``embedding``, ``hybrid``.

    Raises:
        ValueError: if ``retriever`` is not a known retriever.
        FileNotFoundError: if the index or chunk files are missing.
    """
    name = normalize_retriever_name(retriever)

    if not Path(chunks_path).exists():
        raise FileNotFoundError(
            f"Chunk registry not found at: {chunks_path}\n"
            "Run 'index' first to build the index."
        )
    if not Path(index_path).exists():
        raise FileNotFoundError(
            f"'{name}' index not found at: {index_path}\n"
            f"Run 'index --retriever {name}' first to build it."
        )

    store = ChunkStore.load_jsonl(chunks_path)
    retriever_impl = load_retriever(name, index_path)
    return retriever_impl, store


def _search_single(
    query: str,
    k: int,
    retriever: Retriever,
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
        retriever: str = "bm25",
        index_path: str = "",
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    ) -> None:
        """Index a corpus directory for retrieval.

        Args:
            corpus_dir: path to the root directory of the corpus.
            max_chunk_size: maximum character length for any chunk.
            chunks_path: output path for the chunk registry JSONL.
            bm25_index_path: output path for the fitted BM25 pickle.
            retriever: one of ``bm25``, ``embedding``, ``hybrid``.
            index_path: output path for the fitted retriever,
                overriding the per-retriever default.
            embedding_model: encoder used by ``embedding``
                and ``hybrid``.
        """
        try:
            if not isinstance(max_chunk_size, int) or max_chunk_size <= 0:
                print(
                    "Error: max_chunk_size must be a positive integer.",
                    file=sys.stderr,
                )
                return
            build_index(
                corpus_dir=str(corpus_dir),
                max_chunk_size=max_chunk_size,
                chunks_path=str(chunks_path),
                bm25_index_path=str(bm25_index_path),
                retriever=str(retriever),
                index_path=str(index_path) or None,
                embedding_model=str(embedding_model),
            )
        except (FileNotFoundError, ValueError, OSError) as exc:
            print(f"Error: {exc}", file=sys.stderr)
        except Exception as exc:
            print(
                f"Unexpected error during indexing: {exc}",
                file=sys.stderr,
            )

    def search(
        self,
        query: str = "",
        k: int = 5,
        chunks_path: str = DEFAULT_CHUNKS_PATH,
        bm25_index_path: str = DEFAULT_BM25_INDEX_PATH,
        retriever: str = "bm25",
        index_path: str = "",
    ) -> None:
        """Search the index for a single query.

        Args:
            query: the search query string.
            k: number of results to return.
            chunks_path: path to the chunk registry JSONL.
            bm25_index_path: path to the fitted BM25 pickle.
            retriever: one of ``bm25``, ``embedding``, ``hybrid``.
            index_path: path to the fitted retriever artifact,
                overriding the per-retriever default.
        """
        try:
            query = str(query)
            if not isinstance(k, int):
                try:
                    k = int(k)
                except (ValueError, TypeError):
                    print(
                        "Error: k must be an integer.",
                        file=sys.stderr,
                    )
                    return
            if k <= 0:
                print("Warning: k <= 0, no results.", file=sys.stderr)
                return
            if not query or not query.strip():
                print(
                    "Warning: empty query, no results.",
                    file=sys.stderr,
                )
                return

            resolved_index_path = str(
                resolve_index_path(
                    retriever, index_path, bm25_index_path
                )
            )
            retriever_impl, store = _load_retriever_and_store(
                chunks_path, resolved_index_path, retriever
            )
            sources = _search_single(query, k, retriever_impl, store)

            result = MinimalSearchResults(
                question=query,
                retrieved_sources=sources,
            )
            print(result.model_dump_json(indent=2))

        except (FileNotFoundError, ValueError, OSError) as exc:
            print(f"Error: {exc}", file=sys.stderr)
        except Exception as exc:
            print(
                f"Unexpected error during search: {exc}",
                file=sys.stderr,
            )

    def search_dataset(
        self,
        dataset_path: str =
        "data/datasets/UnansweredQuestions/dataset_code_public.json",
        k: int = 5,
        save_directory: str = "data/output/search_results",
        chunks_path: str = DEFAULT_CHUNKS_PATH,
        bm25_index_path: str = DEFAULT_BM25_INDEX_PATH,
        retriever: str = "bm25",
        index_path: str = "",
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
            retriever: one of ``bm25``, ``embedding``, ``hybrid``.
            index_path: path to the fitted retriever artifact,
                overriding the per-retriever default.
        """
        try:
            dataset_path = str(dataset_path)
            if not dataset_path or not dataset_path.strip():
                print(
                    "Error: dataset_path is required.",
                    file=sys.stderr,
                )
                return
            if not isinstance(k, int):
                try:
                    k = int(k)
                except (ValueError, TypeError):
                    print(
                        "Error: k must be an integer.",
                        file=sys.stderr,
                    )
                    return
            if k <= 0:
                print(
                    "Warning: k <= 0, no results to retrieve.",
                    file=sys.stderr,
                )
                return

            dataset = _load_dataset(dataset_path)
            resolved_index_path = str(
                resolve_index_path(
                    retriever, index_path, bm25_index_path
                )
            )
            retriever_impl, store = _load_retriever_and_store(
                chunks_path, resolved_index_path, retriever
            )

            search_results: list[MinimalSearchResults] = []

            for q in tqdm(
                dataset.rag_questions,
                desc="Searching",
                unit="question",
            ):
                sources = _search_single(
                    q.question, k, retriever_impl, store
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
                f"  k: {k}\n"
                f"  Retriever: {normalize_retriever_name(retriever)}"
            )

        except (FileNotFoundError, ValueError, OSError) as exc:
            print(f"Error: {exc}", file=sys.stderr)
        except Exception as exc:
            print(
                f"Unexpected error during search_dataset: {exc}",
                file=sys.stderr,
            )

    def evaluate(
        self,
        student_search_results_path: str =
        "data/output/search_results/dataset_code_public.json",
        dataset_path: str =
        "data/datasets/UnansweredQuestions/dataset_code_public.json",
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
            student_search_results_path = str(
                student_search_results_path
            )
            dataset_path = str(dataset_path)
            if (
                not student_search_results_path
                or not student_search_results_path.strip()
            ):
                print(
                    "Error: student_search_results_path is required.",
                    file=sys.stderr,
                )
                return
            if not dataset_path or not dataset_path.strip():
                print(
                    "Error: dataset_path is required.",
                    file=sys.stderr,
                )
                return

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

        except (FileNotFoundError, ValueError, OSError) as exc:
            print(f"Error: {exc}", file=sys.stderr)
        except Exception as exc:
            print(
                f"Unexpected error during evaluation: {exc}",
                file=sys.stderr,
            )

    def answer(
        self,
        query: str = "",
        k: int = 5,
        chunks_path: str = DEFAULT_CHUNKS_PATH,
        bm25_index_path: str = DEFAULT_BM25_INDEX_PATH,
        retriever: str = "bm25",
        index_path: str = "",
    ) -> None:
        """Search then generate an answer for a single query.

        Args:
            query: the natural-language question.
            k: number of retrieval results to use as context.
            chunks_path: path to the chunk registry JSONL.
            bm25_index_path: path to the fitted BM25 pickle.
            retriever: one of ``bm25``, ``embedding``, ``hybrid``.
            index_path: path to the fitted retriever artifact,
                overriding the per-retriever default.
        """
        try:
            query = str(query)
            if not isinstance(k, int):
                try:
                    k = int(k)
                except (ValueError, TypeError):
                    print(
                        "Error: k must be an integer.",
                        file=sys.stderr,
                    )
                    return
            if k <= 0:
                print("Warning: k <= 0, no results.", file=sys.stderr)
                return
            if not query or not query.strip():
                print(
                    "Warning: empty query, no results.",
                    file=sys.stderr,
                )
                return

            resolved_index_path = str(
                resolve_index_path(
                    retriever, index_path, bm25_index_path
                )
            )
            retriever_impl, store = _load_retriever_and_store(
                chunks_path, resolved_index_path, retriever
            )
            sources = _search_single(query, k, retriever_impl, store)

            # Collect the actual chunk texts for the generator
            source_texts = _collect_source_texts(sources, store)

            generator = AnswerGenerator()
            answer_text = generator.generate(
                question=query,
                source_texts=source_texts,
            )

            result = MinimalAnswer(
                question=query,
                retrieved_sources=sources,
                answer=answer_text,
            )
            print(result.model_dump_json(indent=2))

        except (FileNotFoundError, ValueError, OSError) as exc:
            print(f"Error: {exc}", file=sys.stderr)
        except Exception as exc:
            print(
                f"Unexpected error during answer: {exc}",
                file=sys.stderr,
            )

    def answer_dataset(
        self,
        student_search_results_path:
            str = "data/output/search_results/dataset_code_public.json",
        save_directory: str = "data/output/search_results_and_answer",
        chunks_path: str = DEFAULT_CHUNKS_PATH,
    ) -> None:
        """Generate answers for every question in a search-results file.

        Loads a previously saved ``StudentSearchResults`` JSON
        (produced by ``search_dataset``), generates an answer for
        each question using the retrieved sources, and writes a
        ``StudentSearchResultsAndAnswer`` JSON.

        Args:
            student_search_results_path: path to the
                StudentSearchResults JSON.
            save_directory: directory to write the output file.
            chunks_path: path to the chunk registry JSONL.
        """
        try:
            student_search_results_path = str(
                student_search_results_path
            )
            if (
                not student_search_results_path
                or not student_search_results_path.strip()
            ):
                print(
                    "Error: student_search_results_path is "
                    "required.",
                    file=sys.stderr,
                )
                return

            student_results = _load_student_results(
                student_search_results_path
            )
            store = ChunkStore.load_jsonl(chunks_path)

            generator = AnswerGenerator()

            answers: list[MinimalAnswer] = []

            for sr in tqdm(
                student_results.search_results,
                desc="Generating answers",
                unit="question",
            ):
                source_texts = _collect_source_texts(
                    sr.retrieved_sources, store
                )
                answer_text = generator.generate(
                    question=sr.question,
                    source_texts=source_texts,
                )
                answers.append(
                    MinimalAnswer(
                        question_id=sr.question_id,
                        question=sr.question,
                        retrieved_sources=sr.retrieved_sources,
                        answer=answer_text,
                    )
                )

            output = StudentSearchResultsAndAnswer(
                search_results=answers,
            )

            save_dir = Path(save_directory)
            save_dir.mkdir(parents=True, exist_ok=True)

            dataset_name = Path(
                student_search_results_path
            ).stem
            output_path = save_dir / f"{dataset_name}.json"
            output_path.write_text(
                output.model_dump_json(indent=2),
                encoding="utf-8",
            )

            print(
                f"Answers saved to: {output_path}\n"
                f"  Questions answered: {len(answers)}"
            )

        except (FileNotFoundError, ValueError, OSError) as exc:
            print(f"Error: {exc}", file=sys.stderr)
        except Exception as exc:
            print(
                f"Unexpected error during answer_dataset: {exc}",
                file=sys.stderr,
            )


def _collect_source_texts(
    sources: list[MinimalSource],
    store: ChunkStore,
) -> list[str]:
    """Look up chunk texts for a list of MinimalSource objects.

    For each source, we scan the chunk store for a chunk whose
    file_path and character offsets match.  If found, we return
    its text; otherwise it is silently skipped.

    Args:
        sources: the retrieved sources (file_path + offsets).
        store: the chunk store containing full texts.

    Returns:
        List of chunk text strings, in the same order as sources.
    """
    texts: list[str] = []
    for src in sources:
        matched = False
        for chunk in store.get_chunks_by_file(src.file_path):
            match_first = (
                chunk.first_character_index == src.first_character_index
            )
            match_last = (
                chunk.last_character_index == src.last_character_index
            )
            if match_first and match_last:
                texts.append(chunk.text)
                matched = True
                break
        if not matched:
            # Source was in the results but not in the chunk store —
            # could happen if the index was rebuilt with different
            # settings.  We skip silently.
            pass
    return texts


def main() -> None:
    """Main CLI entrypoint.

    Wraps fire.Fire in a top-level exception handler so the CLI
    never shows a raw Python traceback to the user.
    """
    try:
        fire.Fire(CLI)
    except SystemExit:
        raise
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(130)
    except Exception as exc:
        print(f"Fatal error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
