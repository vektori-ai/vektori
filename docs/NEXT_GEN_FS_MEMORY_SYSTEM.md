# NEXT_GEN_FS_MEMORY_SYSTEM — "MemFS" Complete Architecture

Status: DESIGN (v1) · Date: 2026-06-12 · Module: `vektori/memfs/` (new; `vektori/fsmemory/` untouched)
Predecessors: `docs/checkpoints/01_architecture_review.md` (critique), `03_first_principles_design.md` (principles), `02_research_findings.md` (survey).

Core invariant: **markdown files are canonical; the SQLite index is a disposable, rebuildable cache.**

## 1. Directory layout (the data model users see)

```
<root>/                          # default ~/.vektori/memfs/<namespace>/  (or ./memory/ in a repo)
├── MEMORY.md                    # always-loaded map-of-content: one line per important note
├── semantic/                    # stable knowledge: facts, preferences, project truths
│   └── python-style.md
├── episodic/                    # what happened, sharded by month
│   └── 2026-06/2026-06-12-fix-chunk-overlap.md
├── procedural/                  # playbooks, how-tos, promoted from episodes
│   └── deploy-bench-to-ec2.md
├── sources/                     # derived views of external docs (cache-like)
│   └── <slug>.md                # summary + pointers back to the real file
├── archive/                     # compacted episodics; excluded from default recall
├── .memfs/                      # machine area — everything here is disposable
│   ├── index.db                 # SQLite: FTS5 + vectors + links + file state
│   ├── journal.jsonl            # append-only op log (remember/recall/sync/compact)
│   └── locks/
└── .gitignore                   # ships ignoring .memfs/
```

### Note format (the schema)

```markdown
---
id: 01J9XW…             # ULID, stable across renames
type: semantic          # semantic | episodic | procedural | source
title: Python style preferences
created: 2026-06-12T10:31:00Z
when: 2026-06-11        # event time (optional; bi-temporal with created/git)
source: conversation:sess-42   # provenance: conversation:…, file:…, agent:…
tags: [python, style]
schema: 1
---
Body is plain markdown. [[other-note]] wikilinks form the memory graph.
For type=source: body holds key points; `source: file:/abs/path#sha256=…` pins provenance.
```

Design rules: one fact-cluster per file (merge/diff-friendly); filename = kebab slug (human addressing); `id` in frontmatter survives `mv` (rename-proof — fixes review W4).

## 2. Storage model

- **Writes**: temp file + `os.replace` in same dir (atomic). Optional lockfile `.memfs/locks/<slug>.lock` (O_EXCL) for multi-agent writers.
- **Reads**: plain file reads. Agents may bypass the API entirely (grep works); the API exists for *search* and *lifecycle*, not access control.
- **Git**: optional `auto_commit=True` → debounced `git add && git commit -m "memory: …"`. Rollback/audit = git. Never required.
- **Index** (`.memfs/index.db`, SQLite, WAL):
  - `files(path, id, type, title, mtime_ns, size, content_hash, when_ts, created_ts)`
  - `chunks(chunk_id PK = sha256(file_id+heading_path+text)[:24], file_id, heading_path, text, start_line, end_line)`
  - `chunks_fts` — FTS5 over chunk text + title (BM25)
  - `embeddings(chunk_id PK, vec BLOB float32, model, dim)` — chunk-hash keyed ⇒ an edit re-embeds only changed chunks (fixes W5); numpy brute force v1 (fine to ~10^5 chunks), sqlite-vec seam later (fixes W6 worst case)
  - `links(src_file_id, dst_slug, resolved_dst_id NULL)` — the graph layer
  - `meta(key,value)` — `index_version`, embed model; mismatch ⇒ silent full rebuild
- **Sync** (`sync()`): walk root → (mtime,size) fast-path then content_hash → upsert changed files: parse frontmatter, chunk by heading (200–800 chars, heading-path header prefixed = cheap contextual embedding), diff chunk hashes, embed only new chunks in one batch, update FTS + links. Deleted file ⇒ rows deleted. No daemon; runs at open() and after API writes.

## 3. APIs

```python
class MemFS:
    def __init__(root, namespace="default", embedder="auto", auto_commit=False, secret_scan=True)
    # write path
    async def remember(text, type="semantic", title=None, tags=(), when=None,
                       source=None) -> Note                    # create file + index it
    async def ingest_file(path) / ingest_directory(path, glob) # → sources/ notes, chunk-level incremental
    # read path
    async def recall(query, k=8, types=None, since=None, expand_links=True) -> RecallResult
    async def read(slug_or_id) -> Note
    def orient() -> str                                        # MEMORY.md content for prompt injection
    # lifecycle
    async def sync() -> SyncReport                             # reconcile files ↔ index
    async def compact(month=None, llm=None) -> CompactReport   # episodic roll-up (+optional reflection)
    async def forget(slug_or_query) -> list[str]               # delete files + reindex. real deletion
    async def verify() -> VerifyReport                         # fsck: hash drift, broken links, orphan rows
    def stats() -> dict
```

`RecallResult.items[*] = {path, title, type, snippet, start_line, end_line, score, signals:{bm25, vec, link, recency}, provenance}` — decomposed scores (fixes W11), pointers not paraphrases (fixes W9).

## 4. Retrieval pipeline

```
query ─► (a) FTS5/BM25 top-40      ─┐
      ─► (b) vector cosine top-40  ─┤─► RRF(k=60) ─► top seeds
                                    │        │
                                    │        ▼
                                    │  (c) link expansion: 1-hop wikilink
                                    │      neighbors of top-4 seeds, damped 0.5
                                    └─► re-fuse ─► (d) priors: recency (episodic only,
                                          half-life 30d) + access-count tiebreak
                                          ▼
                                     filters (types, since=when||created) applied in SQL
                                          ▼
                                     top-k with line spans + signal breakdown
```

- Filters inside candidate generation, not post-trim (fixes W8). Lexical is first-class: agents query with identifiers ("fs_file_index", "BGE-M3 OOM") where embeddings are weak.
- No LLM in the recall path. Reranker = pluggable later stage behind same interface.
- Every recall logged to journal.jsonl ⇒ production queries become future eval data.

## 5. Memory graph layer

Wikilinks `[[slug]]` are the edges; no triple extraction. Semantics by convention: plain link (related), `supersedes: [[old]]` frontmatter (versioning), tag co-occurrence (computed). Uses: retrieval expansion, `verify` broken-link report, compaction clustering hints, dashboard viz. PPR over this graph is a v2 experiment — 1-hop first, measured.

## 6. Summarization, reflection, compaction (offline, LLM-optional)

- **Compaction** (`compact(month)`): episodic/<month>/ → one digest note (LLM if provided, else heading-concatenation digest); originals → archive/ (out of default recall, still greppable). Trigger: manual, or month closed + >N notes.
- **Reflection** (inside compact when llm passed): over the month's episodes + top-accessed notes → propose (1) merges of near-duplicate semantic notes (cosine >0.9 candidates precomputed — dedup is *advisory and offline*, never silent at write time: fixes W1), (2) new procedural notes from repeated patterns, (3) contradiction flags. Output = proposal file in `.memfs/proposals/`; applied via explicit `apply`. LLM never silently mutates canon.
- **Decay**: pure retrieval prior. Nothing destroyed by time — only by `forget` or archive-via-compaction.

## 7. Sync & multi-agent model

- One machine, many agents: shared root, per-write lockfiles, WAL index. Namespaces: `agents/<name>/` private + `shared/`; recall scopes to (shared + own) by default.
- Multi-machine: git remote is the blessed transport (text conflicts, rare, human-resolvable). CRDTs deliberately deferred — write rates don't justify them (03 §2).
- Conversation memory (sentex) stays a separate engine; integration = result fusion in the host agent, never storage merging.

## 8. Evaluation framework

`benchmarks/eval_memfs.py`:
1. **Labeled recall set** — 50+ (query → expected chunk) pairs over a frozen corpus (vektori docs + synthetic agent journal). recall@1/3/5, MRR, judged by *chunk identity* not keyword presence (fixes W13).
2. **Ablations** — bm25 / vec / hybrid / +links / +recency: each signal must pay for itself or get cut.
3. **Lifecycle sim** — 90 generated days of episodes → compact → re-run recall: precision-drift gate.
4. **Invariant tests** — rm index.db → sync → identical recall; 1-line edit → exactly 1 chunk re-embedded; verify clean.
5. **Latency** — p50/p95 at 10^3/10^4/10^5 chunks; brute-force ceiling documented.

## 9. Migration plan (fsmemory → memfs)

`scripts/migrate_fsmemory_to_memfs.py` (one-shot, read-only on source):
1. Read `~/.vektori/fsmemory.db`: `fs_file_index` + active facts (metadata.source_type=="filesystem").
2. Per file: still exists ⇒ native re-ingest (better fidelity than stored paraphrases); missing ⇒ materialize stored facts into `sources/<slug>.md` marked "recovered, original missing", provenance kept.
3. Emit report (re-ingested / recovered / inactive-dropped). Old DB untouched.
4. `vektori/fsmemory/` ships deprecated-in-docs ≥1 minor release; zero API breakage.

## 10. Risks / open questions

- Brute-force vector ceiling (~10^5 chunks) — acceptable v1; `VectorStore` seam ready for sqlite-vec/FAISS when an eval demands it.
- Frontmatter discipline from agents — lenient parser (missing frontmatter ⇒ inferred defaults) + `verify` keep sloppy writers safe.
- MEMORY.md contention under multi-agent — single-writer lock; regenerable from frontmatter if corrupted.
- Reflection quality unproven — ships behind explicit `compact(llm=…)`; default path has zero LLM dependence.
