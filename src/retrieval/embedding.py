"""Semantic embedding retriever."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt
from tqdm import tqdm

from src.chunk_store import Chunk
from src.retrieval.base import Retriever


DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_MAX_SEQ_LENGTH = 256
DEFAULT_BATCH_SIZE = 32

FloatArray = npt.NDArray[np.float32]


class EmbeddingRetriever(Retriever):
    """Dense retriever over sentence-transformer embeddings.

    Encodes every chunk once at index time with a transformer
    encoder (mean pooling over the last hidden state, followed by
    L2 normalisation -- the same recipe ``sentence-transformers``
    applies to ``all-MiniLM-L6-v2``), then ranks chunks at query
    time by cosine similarity.

    Because every vector is L2-normalised, cosine similarity is a
    plain dot product, so search is a single matrix-vector
    multiplication over the whole corpus.

    The model is loaded lazily: constructing or ``load()``-ing a
    retriever costs nothing until the first encode.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH,
        batch_size: int = DEFAULT_BATCH_SIZE,
        device: Optional[str] = None,
    ) -> None:
        """Initialize an unfitted EmbeddingRetriever.

        Args:
            model_name: HuggingFace encoder identifier or local path.
            max_seq_length: token cap per chunk when encoding.
            batch_size: number of texts encoded per forward pass.
            device: device to run on ("cpu", "cuda", ...).
                Defaults to CUDA if available, else CPU.
        """
        self.model_name = model_name
        self.max_seq_length = max_seq_length
        self.batch_size = max(1, batch_size)

        self._device: Optional[str] = device
        self._tokenizer: Any = None
        self._model: Any = None

        self._chunk_ids: List[str] = []
        self._embeddings: Optional[FloatArray] = None

    @property
    def is_fitted(self) -> bool:
        """Return True if the retriever holds chunk embeddings."""
        return (
            self._embeddings is not None
            and self._embeddings.size > 0
            and len(self._chunk_ids) > 0
        )

    @property
    def embedding_dim(self) -> Optional[int]:
        """Return the embedding dimension, or None if unfitted."""
        if self._embeddings is None or self._embeddings.ndim != 2:
            return None
        return int(self._embeddings.shape[1])

    def _ensure_model(self) -> None:
        """Load the tokenizer and encoder on first use.

        Imports of torch/transformers are deliberately deferred to
        this method so that the lexical-only code path never pays
        for them.
        """
        if self._model is not None:
            return

        import torch
        from transformers import AutoModel, AutoTokenizer

        if self._device is None:
            self._device = (
                "cuda" if torch.cuda.is_available() else "cpu"
            )

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name
        )
        model = AutoModel.from_pretrained(self.model_name)
        model.to(self._device)
        model.eval()
        self._model = model

    def _encode(
        self,
        texts: List[str],
        show_progress: bool = False,
    ) -> FloatArray:
        """Encode texts into L2-normalised mean-pooled vectors.

        Args:
            texts: the strings to encode.
            show_progress: display a tqdm bar over batches.

        Returns:
            A ``(len(texts), dim)`` float32 array. Returns an empty
            ``(0, 0)`` array when ``texts`` is empty.
        """
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)

        import torch

        self._ensure_model()

        batch_starts = range(0, len(texts), self.batch_size)
        iterator: Any = batch_starts
        if show_progress:
            iterator = tqdm(
                batch_starts,
                desc="Embedding chunks",
                unit="batch",
            )

        vectors: List[FloatArray] = []
        with torch.no_grad():
            for start in iterator:
                batch = texts[start:start + self.batch_size]
                encoded = self._tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self.max_seq_length,
                    return_tensors="pt",
                ).to(self._device)

                output = self._model(**encoded)
                hidden = output.last_hidden_state
                mask = encoded["attention_mask"].unsqueeze(-1)
                mask = mask.to(hidden.dtype)

                summed = (hidden * mask).sum(dim=1)
                counts = mask.sum(dim=1).clamp(min=1e-9)
                pooled = summed / counts

                # L2-normalise so cosine similarity is a dot
                # product. Done with tensor methods rather than
                # torch.nn.functional.normalize to keep this
                # module's torch surface as small as possible.
                norms = (
                    pooled.pow(2).sum(dim=1, keepdim=True).sqrt()
                )
                pooled = pooled / norms.clamp(min=1e-12)

                vectors.append(pooled.cpu().numpy())

        return np.vstack(vectors).astype(np.float32, copy=False)

    def index(self, chunks: Dict[str, Chunk]) -> None:
        """Embed every chunk and store the resulting matrix.

        Args:
            chunks: mapping from chunk_id to Chunk objects.
        """
        if not chunks:
            self._chunk_ids = []
            self._embeddings = None
            return

        self._chunk_ids = list(chunks.keys())
        texts = [chunks[cid].text for cid in self._chunk_ids]
        self._embeddings = self._encode(texts, show_progress=True)

    def search(
        self, query: str, k: int
    ) -> List[Tuple[str, float]]:
        """Return the top-k chunks by cosine similarity.

        Args:
            query: the search query string.
            k: number of results to return.

        Returns:
            List of (chunk_id, score) tuples sorted by descending
            similarity. Returns an empty list if not fitted, if
            k <= 0, or if the query is blank.

        Raises:
            ValueError: if the query vector dimension does not match
                the indexed dimension (index built with a different
                embedding model).
        """
        if not self.is_fitted or k <= 0:
            return []
        if not query or not query.strip():
            return []

        assert self._embeddings is not None

        query_vector = self._encode([query])
        if query_vector.shape[0] == 0:
            return []
        if query_vector.shape[1] != self._embeddings.shape[1]:
            raise ValueError(
                "Query embedding dimension "
                f"({query_vector.shape[1]}) does not match the "
                f"indexed dimension ({self._embeddings.shape[1]}). "
                "Re-run 'index' with the same embedding model."
            )

        scores = self._embeddings @ query_vector[0]

        top_k = min(
            k, int(self._embeddings.shape[0]), len(self._chunk_ids)
        )
        if top_k <= 0:
            return []

        top_idx = np.argpartition(-scores, top_k - 1)[:top_k]
        top_idx = top_idx[np.argsort(-scores[top_idx])]

        return [
            (self._chunk_ids[int(i)], float(scores[int(i)]))
            for i in top_idx
        ]

    def save(self, path: Union[str, Path]) -> None:
        """Persist the embedding matrix and chunk IDs to an .npz.

        The archive is written without pickling, so it can be loaded
        back with ``allow_pickle=False``.

        Args:
            path: file path to save to.
        """
        file_path = Path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        embeddings = self._embeddings
        if embeddings is None:
            embeddings = np.zeros((0, 0), dtype=np.float32)

        if self._chunk_ids:
            chunk_ids = np.array(self._chunk_ids, dtype=np.str_)
        else:
            chunk_ids = np.zeros((0,), dtype="<U1")

        meta = {
            "model_name": self.model_name,
            "max_seq_length": self.max_seq_length,
            "batch_size": self.batch_size,
        }

        with file_path.open("wb") as f:
            np.savez_compressed(
                f,
                embeddings=embeddings,
                chunk_ids=chunk_ids,
                meta=np.array(json.dumps(meta), dtype=np.str_),
            )

    @classmethod
    def load(cls, path: Union[str, Path]) -> "EmbeddingRetriever":
        """Load a previously saved EmbeddingRetriever from disk.

        Args:
            path: file path to load from.

        Returns:
            A fitted EmbeddingRetriever instance.

        Raises:
            FileNotFoundError: if the path does not exist.
            ValueError: if the archive is missing required arrays.
        """
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(
                f"Embedding index not found at: {file_path}"
            )

        try:
            with np.load(file_path, allow_pickle=False) as data:
                embeddings = data["embeddings"].astype(
                    np.float32, copy=False
                )
                chunk_ids = [
                    str(c) for c in data["chunk_ids"].tolist()
                ]
                meta_raw = str(data["meta"].item())
        except KeyError as exc:
            raise ValueError(
                f"Malformed embedding index at {file_path}: "
                f"missing array {exc}"
            ) from exc

        try:
            meta = json.loads(meta_raw)
        except json.JSONDecodeError:
            meta = {}

        retriever = cls(
            model_name=str(
                meta.get("model_name", DEFAULT_EMBEDDING_MODEL)
            ),
            max_seq_length=int(
                meta.get("max_seq_length", DEFAULT_MAX_SEQ_LENGTH)
            ),
            batch_size=int(
                meta.get("batch_size", DEFAULT_BATCH_SIZE)
            ),
        )
        retriever._chunk_ids = chunk_ids
        retriever._embeddings = (
            embeddings if embeddings.size > 0 else None
        )
        return retriever
