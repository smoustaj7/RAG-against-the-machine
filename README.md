*This project has been created as part of the 42 curriculum by smoustaj.*

# RAG Against the Machine

A Retrieval-Augmented Generation (RAG) system that indexes codebases and documentation, retrieves relevant source chunks for natural-language queries, and generates grounded answers using a local language model — all from a clean CLI with no network dependencies at inference time.

---

## Description

Given a large codebase (e.g. vLLM 0.10.1), the system:

1. **Indexes** all Python, Markdown, RST, and plain-text files into a searchable chunk store backed by BM25.
2. **Retrieves** the most relevant chunks for any question, returning precise character-level source references.
3. **Generates** answers grounded strictly in retrieved sources, using `Qwen/Qwen3-0.6B` running locally via HuggingFace Transformers.

---

## Instructions

### Prerequisites

- Python **3.10+**
- [`uv`](https://github.com/astral-sh/uv) package manager

### Setup

```bash
make install        # uv sync — installs all dependencies
```

### Full Pipeline

```bash
# 1. Index a corpus
uv run python -m src index \
  --corpus_dir data/raw \
  --max_chunk_size 2000

# 2. Search across a dataset of questions
uv run python -m src search_dataset \
  --dataset_path data/datasets/UnansweredQuestions/dataset_code_public.json \
  --k 10 \
  --save_directory data/output/search_results/UnansweredQuestions

# 3. Generate answers from prior search results
uv run python -m src answer_dataset \
  --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_code_public.json \
  --save_directory data/output/search_results_and_answer/UnansweredQuestions

# 4. Interactive single-question answer
uv run python -m src answer --query "How does vLLM schedule requests?" --k 5
```

### All CLI Commands

| Command | Description |
|---------|-------------|
| `index` | Walk corpus, chunk files, fit the retriever, persist artifacts |
| `search` | Single-query search, prints `MinimalSearchResults` JSON |
| `search_dataset` | Batch search over a dataset JSON, writes `StudentSearchResults` |
| `evaluate` | Self-evaluation: recall@k against an `AnsweredQuestions` dataset |
| `answer` | Single-query search + answer generation |
| `answer_dataset` | Batch answer generation from saved search results |

`index`, `search`, `search_dataset` and `answer` all accept
`--retriever {bm25,embedding,hybrid}` (default `bm25`). See
[Bonuses](#bonuses).

### Makefile Targets

```bash
make install      # uv sync
make run          # uv run python -m src
make debug        # uv run python -m pdb -m src
make lint         # flake8 + mypy
make lint-strict  # flake8 + mypy --strict
make clean        # remove __pycache__, .mypy_cache, data/processed/*
```

---

## Resources

- [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) — local answer generation model
- [rank-bm25](https://github.com/dorianbrown/rank_bm25) — BM25Okapi implementation
- [PyTorch](https://pytorch.org/) & [HuggingFace Transformers](https://huggingface.co/docs/transformers) — model loading and inference
- [Pydantic v2](https://docs.pydantic.dev/) — data model validation
- [Python `ast` module](https://docs.python.org/3/library/ast.html) — AST-aware Python chunking
- [fire](https://github.com/google/python-fire) — CLI scaffolding
- [tqdm](https://tqdm.github.io/) — progress bars
- [pytest](https://docs.pytest.org/) — test suite

### AI Usage Disclosure

AI assistants (Antigravity / Claude) were used throughout this project for:
- **Architecture planning**: designing the separation of chunk storage, retrieval index, and CLI layers so that bonuses are additive rather than refactors.
- **Implementation assistance**: writing boilerplate for Pydantic models, the BM25 retriever save/load, and the token-budgeted prompt builder.
- **Robustness review**: identifying edge cases in input validation (empty query, negative k, malformed JSON, non-UTF8 files) and generating adversarial test cases.
- **Phase 7 execution**: running the full end-to-end pipeline and verifying moulinette output format compliance.

All design decisions, chunking strategies, retrieval tuning, and README content reflect the author's own understanding and choices.

---

## System Architecture

```
data/raw/                        ← corpus (vLLM source tree)
        │
        ▼
src/chunking.py                  ← PythonChunker / MarkdownChunker / TextChunker
        │  produces Chunk objects (chunk_id, file_path, char offsets, text, file_hash)
        ▼
src/chunk_store.py               ← ChunkStore — in-memory dict, serialised to chunks.jsonl
        │
        ├─ src/retrieval/lexical.py    ← BM25Retriever (rank_bm25.BM25Okapi)
        │        fitted on tokenised chunk texts → bm25_index.pkl
        ├─ src/retrieval/embedding.py  ← EmbeddingRetriever (all-MiniLM-L6-v2)
        │        mean-pooled chunk vectors → embedding_index.npz
        └─ src/retrieval/hybrid.py     ← HybridRetriever (RRF over both)
                 → hybrid_index/{lexical.pkl,embedding.npz}
        │
        ▼
src/indexer.py                   ← make_retriever / load_retriever by name
        │
        ▼
src/cli.py                       ← fire.Fire(CLI) — six commands
        │
        ├─ search / search_dataset  → MinimalSearchResults / StudentSearchResults JSON
        └─ answer / answer_dataset  → MinimalAnswer / StudentSearchResultsAndAnswer JSON
                │
                ▼
        src/generation.py        ← AnswerGenerator (Qwen/Qwen3-0.6B via Transformers)
```

**Key design rule:** chunk storage, index structure, and CLI are fully decoupled. Adding a new retriever (embedding, hybrid) requires only a new file under `src/retrieval/` — zero changes to the CLI or chunk store.

---

## Chunking Strategy

Three complementary chunkers share a single `max_chunk_size` hard cap (default **2,000 characters**):

### PythonChunker (`.py` files)

Uses Python's built-in `ast` module to parse the file and extract top-level AST node boundaries (functions, classes, imports, standalone expressions). Consecutive small nodes are merged into a single chunk until they would exceed `max_chunk_size`. Any node larger than the cap is sub-chunked by `TextChunker`. Falls back to `TextChunker` entirely on `SyntaxError` (e.g. partial files or stubs with unusual syntax).

This keeps semantically coherent units (a whole function body) together instead of splitting mid-function.

### MarkdownChunker (`.md` / `.rst` files)

Splits on header boundaries (`#`/`##`/… for Markdown; underline-style markers for RST). Consecutive sections are merged up to `max_chunk_size`; oversized sections are sub-chunked. Falls back to `TextChunker` when no headers are detected.

### TextChunker (`.txt` and fallback)

Sliding-window splitting with natural break-point search: tries `\n\n` → `\n` → `. ` → ` ` before hard-cutting at the byte boundary. Applies a **200-character overlap** (capped at ¼ of `max_chunk_size`) to preserve cross-boundary context for BM25 term matching.

### File Filtering

Indexed extensions: `.py`, `.md`, `.rst`, `.txt`
Skipped directories: `.git`, `.venv`, `__pycache__`, `.mypy_cache`, `build`, `dist`, `node_modules`
Skipped extensions: `.lock`, `.pyc`, `.pkl`, `.pt`, `.safetensors`, binary formats

---

## Retrieval Method

**BM25Okapi** (via `rank_bm25`) with a custom code-aware tokenizer:

1. Replace non-alphanumeric characters (except `_`) with spaces.
2. Split on whitespace and underscores (`snake_case` → separate tokens).
3. Sub-split on camelCase/PascalCase boundaries (`CamelCase` → `Camel`, `Case`).
4. Lowercase all tokens.
5. Remove tokens of length ≤ 1 and a curated stop-word list (English function words + Python keywords like `self`, `return`, `def`).

This ensures that identifiers like `PagedAttentionScheduler` and `paged_attention_scheduler` tokenise to the same set of terms, dramatically improving recall on code questions.

The fitted index and chunk registry are persisted as separate artifacts under `data/processed/`, so subsequent `search` calls load in milliseconds without re-indexing.

---

## Bonuses

Every retriever implements the same `Retriever` interface
(`index` / `search` / `save` / `load`) and consumes the same chunk
registry, so switching between them re-fits an index but never
re-chunks the corpus — and no call site in `cli.py` changes.

| `--retriever` | Class | Artifact |
|---------------|-------|----------|
| `bm25` (default) | `BM25Retriever` | `data/processed/bm25_index.pkl` |
| `embedding` | `EmbeddingRetriever` | `data/processed/embedding_index.npz` |
| `hybrid` | `HybridRetriever` | `data/processed/hybrid_index/` (directory) |

### Bonus 1 — Semantic embeddings (`src/retrieval/embedding.py`)

`EmbeddingRetriever` encodes every chunk with
`sentence-transformers/all-MiniLM-L6-v2`, loaded through plain
`transformers.AutoModel` — mean pooling over the last hidden state
masked by `attention_mask`, then L2 normalisation. That is exactly
what the `sentence-transformers` wrapper does for this model, so
using `transformers` directly adds **zero new dependencies**.

Because every vector is L2-normalised, cosine similarity is a plain
dot product and a search is one matrix-vector multiply over the
whole corpus. Vectors are persisted as a pickle-free `.npz`
(`embeddings`, `chunk_ids`, `meta`), loadable with
`allow_pickle=False`.

The encoder is loaded **lazily**, on the first call that actually
needs to embed something. Constructing or `load()`-ing a retriever
costs nothing, so the BM25-only path never pays for torch.

Chunks are truncated to 256 tokens when encoded (MiniLM's own
`max_seq_length`), so a 2000-character chunk contributes its first
~256 tokens to its vector. BM25 still sees the chunk in full, which
is one reason the two retrievers fail differently — and therefore
why fusing them helps.

```bash
uv run python -m src index --corpus_dir data/raw --retriever embedding
uv run python -m src search --query "How is the KV cache paged?" \
  --k 5 --retriever embedding
```

### Bonus 2 — Hybrid retrieval (`src/retrieval/hybrid.py`)

`HybridRetriever` wraps a lexical and a semantic retriever and
merges their rankings with **Reciprocal Rank Fusion**: each list
contributes `weight / (rrf_k + rank)` to every chunk it ranks, and
the sums are re-sorted. RRF deliberately ignores the raw scores —
BM25 scores and cosine similarities live on incomparable scales,
and rank position is the only signal the two share. A chunk ranked
respectably by both retrievers therefore beats a chunk ranked first
by only one.

Each component is queried for a pool of `max(k, candidate_pool)`
(default 50) candidates before fusion, so the fused top-`k` can
promote a chunk that neither retriever had in its own top-`k`.

One refinement on textbook RRF: results scoring `<= 0` are dropped
before fusing. A BM25 score of 0 means the chunk shares no query
term at all, so its position in that ranking is arbitrary; feeding
those arbitrary positions into RRF would let noise tie with genuine
hits from the other retriever. Disable with `drop_non_positive=False`.

Tunable via constructor: `rrf_k` (default 60), `candidate_pool`
(50), `lexical_weight` / `embedding_weight` (1.0 each). The values
are persisted in `hybrid_meta.json` and restored on load.

```bash
uv run python -m src index --corpus_dir data/raw --retriever hybrid
uv run python -m src search_dataset \
  --dataset_path data/datasets/UnansweredQuestions/dataset_code_public.json \
  --k 10 --retriever hybrid \
  --save_directory data/output/search_results/hybrid
```

### Not yet implemented

Bonus 3 (incremental indexing), 4 (caching) and 5 (local HTTP API)
are still stubs — `file_hash` is already captured on every chunk so
bonus 3 remains a comparison, not a refactor.

---

## Performance Analysis

Measured with the moulinette `evaluate_student_search_results` binary against the **public datasets** (`data/datasets/AnsweredQuestions/`), `k=10`, `max_context_length=2000`:

| Dataset | Recall@1 | Recall@3 | Recall@5 | Recall@10 | Pass threshold |
|---------|----------|----------|----------|-----------|----------------|
| **Code** (`dataset_code_public`) | 33.3% | 58.6% | **66.7%** | 72.7% | ✅ ≥ 50% |
| **Docs** (`dataset_docs_public`) | 65.0% | 77.0% | **82.0%** | 91.0% | ✅ ≥ 80% |

Indexing throughput: **14,992 chunks** from **1,830 files** (vLLM 0.10.1) in ~3 seconds on CPU.
Search throughput: ~60 questions/second (BM25 is purely in-memory, no GPU required).

The numbers above are for the default `--retriever bm25`. The
`embedding` and `hybrid` retrievers are measured with the same loop,
pointing `search_dataset` at a per-retriever output directory so the
runs don't overwrite each other:

```bash
uv run python -m src index --corpus_dir data/raw --retriever hybrid
uv run python -m src search_dataset \
  --dataset_path data/datasets/AnsweredQuestions/dataset_code_public.json \
  --k 10 --retriever hybrid \
  --save_directory data/output/search_results/hybrid
uv run python -m src evaluate \
  --student_search_results_path data/output/search_results/hybrid/dataset_code_public.json \
  --dataset_path data/datasets/AnsweredQuestions/dataset_code_public.json
```

Note that embedding indexing is far slower than BM25: ~15k chunks
through MiniLM on CPU is minutes, not seconds, versus milliseconds
per query at search time once the vectors are on disk.

---

## Design Decisions

**Separate storage from index from CLI.** The `ChunkStore` holds raw text and metadata; the `BM25Retriever` holds only tokenised terms and scores; the CLI wires them together. This means switching the retriever (e.g. to embeddings) requires only a new class, not touching the data layer or CLI.

**`file_hash` on every chunk.** SHA-256 of the file content is stored at index time. This costs nothing at query time and makes incremental re-indexing (Bonus 3) a matter of comparing stored hashes to current hashes, with no architectural changes.

**Character-level source references.** `first_character_index` / `last_character_index` are offsets into the *original* file, not into the chunk text. This means the moulinette can verify source overlap with IoU ≥ 0.05 against ground-truth sources, and overlapping chunks from `TextChunker` produce valid, non-contradictory offsets.

**Token-budgeted prompt construction.** `_build_sources_block` adds sources greedily, truncating at the actual token boundary using the model's own tokenizer rather than a character heuristic. This prevents silent context overflow that would produce hallucinated or cut-off answers.

**Greedy merging over fixed-size windows.** Rather than splitting every file into fixed-size windows, the chunkers first respect semantic boundaries (AST nodes, headers) and only fall back to size-based splitting for files that have none. This keeps BM25 term statistics coherent within each chunk.

---

## Challenges Faced

**AST offset calculation.** Python's `ast` module reports line/column numbers, not byte offsets. Computing `first_character_index` required building a `line_starts` array (cumulative newline positions) and translating `(lineno, col_offset)` pairs — including decorator lines that precede the `def`/`class` keyword.

**Chunk size vs. recall trade-off.** Smaller chunks improve precision (the retrieved window matches the ground-truth source more tightly) but reduce recall (a question spanning two functions may not match either chunk alone). The 2,000-character cap with 200-character overlap in `TextChunker` was chosen after iterating on the self-evaluation loop (`evaluate` command) against the public answered datasets.

**Token budget vs. answer quality.** Qwen3-0.6B has a limited context window. Stuffing too many sources degrades answer coherence; too few loses recall. The greedy source selection (up to `max_context_tokens=2048`) was validated to produce coherent answers while keeping the pipeline from crashing on long-context inputs.

**Non-UTF-8 files in the corpus.** The vLLM tree contains a handful of files with non-standard encodings (e.g. compiled documentation artifacts). These are silently skipped with a `[WARN]` to stderr rather than aborting the entire index build.

---

## Example Usage

### Index the vLLM corpus

```bash
uv run python -m src index --corpus_dir data/raw --max_chunk_size 2000
# Scanning corpus at: data/raw
# Found 1952 indexable files.
# Chunking complete: 14992 chunks from 1830 files (122 skipped).
# BM25 index saved to: data/processed/bm25_index.pkl
# Indexing complete! 🎉
```

### Search for a single query

```bash
uv run python -m src search --query "How does PagedAttention manage KV cache blocks?" --k 3
```

```json
{
  "question": "How does PagedAttention manage KV cache blocks?",
  "retrieved_sources": [
    {
      "file_path": "data/raw/vllm-0.10.1/vllm/core/block_manager.py",
      "first_character_index": 0,
      "last_character_index": 1987
    }
  ]
}
```

### Batch search a dataset

```bash
uv run python -m src search_dataset \
  --dataset_path data/datasets/UnansweredQuestions/dataset_code_public.json \
  --k 10 \
  --save_directory data/output/search_results/UnansweredQuestions
# Searching: 100%|████████████████████| 99/99 [00:01<00:00, 52.92question/s]
# Search results saved to: data/output/search_results/UnansweredQuestions/dataset_code_public.json
```

### Evaluate with the moulinette

```bash
./moulinette/moulinette-ubuntu evaluate_student_search_results \
  data/output/search_results/UnansweredQuestions/dataset_code_public.json \
  data/datasets/AnsweredQuestions/dataset_code_public.json \
  --k 10 --max_context_length 2000
# 📈 Recall@5: 0.667 (66.7%)  ✅
```

### Generate answers

```bash
uv run python -m src answer_dataset \
  --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_code_public.json \
  --save_directory data/output/search_results_and_answer/UnansweredQuestions
# Generating answers: 100%|████████████| 99/99
# Answers saved to: data/output/search_results_and_answer/UnansweredQuestions/dataset_code_public.json
```
