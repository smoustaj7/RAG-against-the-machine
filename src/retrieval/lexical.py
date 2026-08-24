import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from rank_bm25 import BM25Okapi

from src.chunk_store import Chunk
from src.retrieval.base import Retriever
from src.tokenizer import tokenize


class BM25Retriever(Retriever):
    """BM25-based lexical retriever using rank_bm25.BM25Okapi.
    Tokenizes chunk text with the shared code-aware tokenizer,
    fits a BM25Okapi model, and supports save/load via pickle.
    """

    def __init__(self) -> None:
        """Initialize an unfitted BM25Retriever."""
        self._bm25: Optional[BM25Okapi] = None
        self._chunk_ids: List[str] = []
        self._corpus: List[List[str]] = []

    @property
    def is_fitted(self) -> bool:
        """Return True if the retriever has been fitted on a corpus."""
        return self._bm25 is not None

    def index(self, chunks: Dict[str, Chunk]) -> None:
        """Fit BM25 on the provided chunks.
        Args:
            chunks: mapping from chunk_id to Chunk objects.
        """
        if not chunks:
            self._bm25 = None
            self._chunk_ids = []
            self._corpus = []
            return

        self._chunk_ids = list(chunks.keys())
        self._corpus = [
            tokenize(chunks[cid].text) for cid in self._chunk_ids
        ]
        self._bm25 = BM25Okapi(self._corpus)

    def search(
        self, query: str, k: int
    ) -> List[Tuple[str, float]]:
        """Search for the top-k most relevant chunks.
        Args:
            query: the search query string.
            k: number of results to return.
        Returns:
            List of (chunk_id, score) tuples sorted by descending
            relevance. Returns empty list if not fitted or k <= 0.
        """
        if not self.is_fitted or not self._chunk_ids or k <= 0:
            return []

        tokenized_query = tokenize(query)
        if not tokenized_query:
            return []

        assert self._bm25 is not None
        scores = self._bm25.get_scores(tokenized_query)

        scored = list(zip(self._chunk_ids, scores.tolist()))
        scored.sort(key=lambda x: x[1], reverse=True)

        return scored[:k]

    def save(self, path: Union[str, Path]) -> None:
        """Persist the fitted BM25 retriever to disk via pickle.
        Saves the BM25 model, chunk ID list, and tokenized corpus
        as a single pickle file.
        Args:
            path: file path to save to.
        """
        file_path = Path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        state = {
            "bm25": self._bm25,
            "chunk_ids": self._chunk_ids,
            "corpus": self._corpus,
        }
        with file_path.open("wb") as f:
            pickle.dump(state, f)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "BM25Retriever":
        """Load a previously saved BM25Retriever from disk.
        Args:
            path: file path to load from.
        Returns:
            A fitted BM25Retriever instance.
        Raises:
            FileNotFoundError: if the path does not exist.
        """
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(
                f"BM25 index not found at: {file_path}"
            )

        with file_path.open("rb") as f:
            state = pickle.load(f)

        retriever = cls()
        retriever._bm25 = state["bm25"]
        retriever._chunk_ids = state["chunk_ids"]
        retriever._corpus = state["corpus"]
        return retriever
