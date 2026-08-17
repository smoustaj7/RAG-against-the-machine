# RAG Against the Machine — Project Roadmap

Design rule behind every phase: **separate chunk storage from index structure, and index structure from CLI**, so bonuses (embeddings, hybrid, incremental, caching, API) are additive files, not rewrites. Every phase lists which spec requirement it satisfies so nothing gets missed.

---

## Phase 0 — Environment & Scaffolding
**Satisfies:** V.1 General Rules, V.2 Makefile, V.4 Additional Requirements, VI.7.1 layout

- [ ] `uv init`, set Python 3.10+ in `pyproject.toml`
- [ ] Add deps: `pydantic`, `fire`, `tqdm`, `rank_bm25` (or `bm25s`), `numpy`, `scikit-learn`, `transformers`, `torch`, `flake8`, `mypy`, `pytest`
- [ ] Create folder layout exactly as spec requires:
  ```
  src/
  data/raw/ data/processed/
  data/datasets/UnansweredQuestions/ data/datasets/AnsweredQuestions/
  data/output/search_results/ data/output/search_results_and_answer/
  ```
- [ ] `.gitignore`: `__pycache__`, `.mypy_cache`, `data/processed/`, any local vLLM corpus paths if vendored separately, `.env`
- [ ] **Makefile** with all 5 required targets:
  - `install` → `uv sync`
  - `run` → `uv run python -m src`
  - `debug` → `uv run python -m pdb -m src`
  - `clean` → remove `__pycache__`, `.mypy_cache`, caches
  - `lint` → `flake8 .` + the exact mandated `mypy .` flags
  - `lint-strict` (optional) → `flake8 .` + `mypy . --strict`
- [ ] `README.md` stub with required section headers only (fill in later — Phase 8)
- [ ] Git repo initialized, first commit is scaffolding only (no vendored corpus, no secrets)

**Definition of done:** `make install`, `make lint`, `make run` all execute without error on an empty CLI.

---

## Phase 1 — Data Models
**Satisfies:** VI.4 Data Models

- [x] `src/models.py`: implement `MinimalSource`, `UnansweredQuestion`, `AnsweredQuestion`, `RagDataset`, `MinimalSearchResults`, `MinimalAnswer`, `StudentSearchResults`, `StudentSearchResultsAndAnswer` exactly as specified
- [x] Add your own extra fields/models here if needed later (e.g. `chunk_id` on `MinimalSource` is tempting — **don't** add fields the grader doesn't expect to *required* spots; extend only additively)
- [x] Full type hints + docstrings (mypy/flake8 compliance starts here, not later)

**Definition of done:** models import cleanly, `mypy` passes on this file alone.


---

## Phase 2 — Chunk Registry & Chunking Strategies
**Satisfies:** VI.1 Indexing (chunking requirement), sets up Bonus 1/2/3

- [ ] `src/chunk_store.py`: define internal `Chunk` dataclass/pydantic model: `chunk_id, file_path, first_character_index, last_character_index, text, file_hash`
  - `file_hash` (sha256 of file contents) is unused right now but **must** be captured here — this is the one field that makes Bonus 3 (incremental indexing) free later instead of a refactor
- [ ] `src/chunking.py`:
  - [ ] `PythonChunker` — use `ast` to chunk on function/class boundaries where possible, fall back to size-based splitting for code that doesn't parse cleanly; hard cap at `--max_chunk_size`
  - [ ] `MarkdownChunker` — chunk on headers/sections, fall back to size-based splitting; hard cap at `--max_chunk_size`
  - [ ] Both must **never** exceed `max_chunk_size` (moulinette hard-rejects overlength sources)
  - [ ] Decide + document file selection rule (which extensions/dirs to skip — tests, `.git`, binaries, etc.)
- [ ] Unit tests (pytest, not graded but catches boundary bugs early): chunk boundaries don't overlap incorrectly, no chunk exceeds max size, char indices are correct offsets into the *original* file

**Definition of done:** running chunkers on a handful of sample files produces valid, correctly-indexed `Chunk` objects with no size violations.

---

## Phase 3 — Lexical Indexing
**Satisfies:** VI.1 Indexing (retrieval method), Performances (indexing ≤5 min)

- [ ] `src/retrieval/base.py`: abstract `Retriever` interface —
  ```python
  class Retriever(ABC):
      def index(self, chunks: dict[str, Chunk]) -> None: ...
      def search(self, query: str, k: int) -> list[tuple[str, float]]: ...
  ```
- [ ] `src/retrieval/lexical.py`: `BM25Retriever` (or TF-IDF) implementing that interface
- [ ] Tokenizer that handles code identifiers sensibly (split `snake_case`, `camelCase`, punctuation) — shared by both chunk types
- [ ] `src/indexer.py`: orchestrates walk → chunk → tokenize → fit retriever → persist
- [ ] Persist as **separate artifacts** under `data/processed/`:
  - chunk registry (e.g. `chunks.jsonl`, includes `file_hash`)
  - fitted lexical index (e.g. `bm25_index.pkl`)
- [ ] `tqdm` progress bars on file walk + chunking + indexing
- [ ] Graceful handling: unreadable files, empty files, non-UTF8 content

**Definition of done:** `uv run python -m src index --max_chunk_size 2000` completes in well under 5 minutes and produces both artifacts.

---

## Phase 4 — Retrieval CLI + Self-Evaluation Loop
**Satisfies:** VI.2 Retrieval, VI.6 CLI (`search`, `search_dataset`, `evaluate`), VII.1 Evaluation, Performances (throughput)

- [ ] `src/cli.py` command `search(query, k)` → loads index, calls `Retriever.search`, maps `chunk_id`s back to `MinimalSource`, prints/returns
- [ ] `search_dataset(dataset_path, k, save_directory)` → batch over `UnansweredQuestion`s, writes `StudentSearchResults` JSON, `tqdm` over questions
- [ ] `evaluate(student_search_results_path, dataset_path)` → your own recall@k against `AnsweredQuestion` ground truth (IoU ≥ 0.05 overlap rule, same file_path required) — **this is your tuning feedback loop, build it before polishing chunking**
- [ ] Degenerate input handling: empty query, k=0, missing dataset file, malformed JSON — no unhandled tracebacks anywhere in this chain
- [ ] Iterate here: tune chunk size, tokenizer, chunking boundaries against `evaluate` until you're clearing 80%/50% recall@5 targets — **this is the phase you'll revisit most**

**Definition of done:** self-measured recall@5 ≥ 80% docs / ≥ 50% code on public datasets; 200 questions retrieved in <90s.

---

## Phase 5 — Answer Generation
**Satisfies:** VI.3 Answer Generation, VI.6 CLI (`answer`, `answer_dataset`)

- [ ] `src/generation.py`: load `Qwen/Qwen3-0.6B` via `transformers`, build prompt template that stuffs retrieved chunk text into context within token budget (truncate/select if over budget — decide a clear rule, document it)
- [ ] Prompt should explicitly instruct grounding in provided sources and instruct against fabricating file/line references
- [ ] `answer(query, k)` CLI command → search → generate → print
- [ ] `answer_dataset(student_search_results_path, save_directory)` → loads prior search results, generates answers, writes `StudentSearchResultsAndAnswer` JSON, `tqdm` over questions
- [ ] Handle: model failing to produce parseable output, empty context (no sources retrieved), very long context needing truncation

**Definition of done:** answers are coherent, cite/reflect retrieved content, and the pipeline never crashes even on adversarial queries.

---

## Phase 6 — Robustness Pass
**Satisfies:** V.1 (exception handling), VI.6 (CLI must never crash)

- [ ] Sweep all 6 commands with: empty string query, k=0, k negative, nonexistent dataset path, malformed JSON dataset, missing index (search before index built), non-UTF8 file in corpus
- [ ] Confirm every failure path prints a clean message and exits gracefully — no raw tracebacks
- [ ] Run full `make lint` / `make lint-strict` and fix everything
- [ ] Run `pytest` suite one more time

**Definition of done:** you (or a peer) genuinely cannot crash the CLI by feeding it garbage.

---

## Phase 7 — End-to-End Dry Run
**Satisfies:** VI.7.2 full pipeline walkthrough

- [ ] Run the exact 4-step sequence from the spec: `index` → `search_dataset` → moulinette `evaluate_student_search_results` → `answer_dataset`
- [ ] Confirm output paths match `data/output/search_results/<DatasetScope>/...` and `data/output/search_results_and_answer/<DatasetScope>/...` exactly, scoped by dataset folder (don't overwrite between `UnansweredQuestions`/`AnsweredQuestions` runs)
- [ ] Confirm nothing is hardcoded — every path is a CLI arg

**Definition of done:** the moulinette runs against your output without structural complaints, independent of any manual fixups.

---

## Phase 8 — README
**Satisfies:** Chapter VIII Readme Requirements (write this last, once real decisions/numbers exist)

- [ ] First line, italicized: `*This project has been created as part of the 42 curriculum by <login(s)>.*`
- [ ] Description, Instructions, Resources (incl. AI usage disclosure: what tasks, what parts)
- [ ] System architecture, Chunking strategy, Retrieval method, Performance analysis (your real recall@k numbers), Design decisions, Challenges faced, Example usage
- [ ] Written in English

**Definition of done:** a stranger could clone, run, and understand the project from this file alone.

---

## Phase 9 — Bonuses (only start once mandatory is 100% validated)
**Satisfies:** Chapter IX, each worth 1 point, all must actually run, not just be described

Because of the Phase 2–3 design choices, each bonus is additive:

1. **Semantic embeddings** — new `src/retrieval/embedding.py`, `EmbeddingRetriever(Retriever)` using `sentence-transformers/all-MiniLM-L6-v2` over the *same* chunk registry. No re-chunking.
2. **Hybrid retrieval** — `src/retrieval/hybrid.py`, `HybridRetriever(Retriever)` wraps lexical + embedding retrievers, merges `(chunk_id, score)` lists (e.g. reciprocal rank fusion). No changes to `search`/`search_dataset` call sites.
3. **Incremental indexing** — in `indexer.py`, compare each file's current hash to the stored `file_hash` in the chunk registry; only re-chunk + re-index changed files; drop stale chunk_ids for removed/changed files.
4. **Caching** — wrap `load_index()` / `run_search()` with `diskcache` or `joblib.Memory`; no logic changes needed if Phase 3–4 kept those as clean single-entry-point functions.
5. **Local HTTP API** — `src/api.py`, thin FastAPI app that imports and calls the exact same functions `cli.py` calls (search, answer). No duplicated logic.

**Definition of done per bonus:** implemented, actually runs, demonstrable live — not just mentioned in the README.

---

## Order-of-operations summary (don't skip around)

```
Phase 0 → 1 → 2 → 3 → 4 (loop until recall targets hit) → 5 → 6 → 7 → 8 → [mandatory validated] → 9
```

Phase 4 is where you'll spend the most wall-clock time — it's the tuning loop. Everything after Phase 5 is comparatively mechanical if Phases 0–4 were built with the interfaces above.