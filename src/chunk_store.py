"""Chunk store registry for storing and managing chunk metadata."""

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Union
from pydantic import BaseModel
from src.models import MinimalSource


class Chunk(BaseModel):
    """Internal Chunk model representing a chunk of text from a source file."""

    chunk_id: str
    file_path: str
    first_character_index: int
    last_character_index: int
    text: str
    file_hash: str

    def to_minimal_source(self) -> MinimalSource:
        """Convert chunk to a MinimalSource object."""
        return MinimalSource(
            file_path=self.file_path,
            first_character_index=self.first_character_index,
            last_character_index=self.last_character_index,
        )


class ChunkStore:
    """In-memory registry and persistence store for text chunks."""

    def __init__(self, chunks: Optional[Iterable[Chunk]] = None) -> None:
        self._chunks: Dict[str, Chunk] = {}
        if chunks:
            self.add_chunks(chunks)

    def add_chunk(self, chunk: Chunk) -> None:
        """Add a single chunk to the store."""
        self._chunks[chunk.chunk_id] = chunk

    def add_chunks(self, chunks: Iterable[Chunk]) -> None:
        """Add multiple chunks to the store."""
        for chunk in chunks:
            self.add_chunk(chunk)

    def get_chunk(self, chunk_id: str) -> Optional[Chunk]:
        """Retrieve a chunk by its ID."""
        return self._chunks.get(chunk_id)

    def get_all_chunks(self) -> List[Chunk]:
        """Return list of all stored chunks."""
        return list(self._chunks.values())

    def get_chunks_by_file(self, file_path: str) -> List[Chunk]:
        """Return all chunks associated with a given file path."""
        return [
            c for c in self._chunks.values() if c.file_path == file_path
        ]

    def clear(self) -> None:
        """Clear all stored chunks."""
        self._chunks.clear()

    def __len__(self) -> int:
        return len(self._chunks)

    def save_jsonl(self, path: Union[str, Path]) -> None:
        """Save stored chunks to a JSONL file."""
        file_path = Path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with file_path.open("w", encoding="utf-8") as f:
            for chunk in self._chunks.values():
                f.write(chunk.model_dump_json() + "\n")

    @classmethod
    def load_jsonl(cls, path: Union[str, Path]) -> "ChunkStore":
        """Load chunks from a JSONL file into a ChunkStore instance."""
        file_path = Path(path)
        store = cls()
        if not file_path.exists():
            return store
        with file_path.open("r", encoding="utf-8") as f:
            for line in f:
                line_str = line.strip()
                if line_str:
                    chunk = Chunk.model_validate_json(line_str)
                    store.add_chunk(chunk)
        return store
