from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Tuple, Union

from src.chunk_store import Chunk


class Retriever(ABC):
    """Abstract base class for retrieval implementations.

    All retrievers must implement:
    - index(): fit the retriever on a dict of chunks
    - search(): return ranked (chunk_id, score) pairs for a query
    - save(): persist the fitted retriever to disk
    - load(): restore a fitted retriever from disk
    """

    @abstractmethod
    def index(self, chunks: Dict[str, Chunk]) -> None:
        """Fit the retriever on a dictionary of chunk_id -> Chunk.

        Args:
            chunks: mapping from chunk_id to Chunk objects.
        """
        ...

    @abstractmethod
    def search(
        self, query: str, k: int
    ) -> List[Tuple[str, float]]:
        """Search for the top-k most relevant chunks for a query.

        Args:
            query: the search query string.
            k: number of results to return.

        Returns:
            List of (chunk_id, score) tuples, sorted by descending
            relevance score.
        """
        ...

    @abstractmethod
    def save(self, path: Union[str, Path]) -> None:
        """Persist the fitted retriever to disk.

        Args:
            path: file path to save the retriever state.
        """
        ...

    @classmethod
    @abstractmethod
    def load(cls, path: Union[str, Path]) -> "Retriever":
        """Load a previously saved retriever from disk.

        Args:
            path: file path to load the retriever state from.

        Returns:
            A fitted Retriever instance.
        """
        ...
