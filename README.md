# Document Intelligence — Ingestion Backend

Upload, storage, parsing, chunking, embedding, vector indexing, status
tracking, and admin inspection for policy/contract documents. Retrieval,
chat, RAG, and contradiction detection are phase two and are not built here.

## Stack

Python 3.11+, FastAPI, SQLAlchemy 2.0 async + asyncpg, PostgreSQL, Qdrant,
`fastembed` (`BAAI/bge-small-en-v1.5`, local, 384-dim), `pdfplumber`,
`python-docx`, `charset-normalizer`, `python-magic`,
`langchain-text-splitters`, `tenacity`, `structlog`.

- Embeddings are local and free — no API key, no rate limit, no hosted
  embedding call anywhere in the ingestion path.
- No embedding-model fallback, ever: a different model produces an
  incompatible vector space, so a silent fallback would corrupt the
  collection. One model; if it fails, retry it, then fail the document.

## Running it

```bash
cp .env.example .env          # defaults work out of the box
docker compose up -d          # Postgres (port 5433) + Qdrant (6333/6334)
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
brew install libmagic         # required by python-magic; apt: libmagic1
uvicorn app.main:app --reload
```

`GET /api/health` pings Postgres and Qdrant and reports whether the
embedding model has finished loading — it never returns a hardcoded `ok`.

## Known gaps (deliberate, for a project this size)

- **No Alembic / migrations.** Schema is `Base.metadata.create_all()` at
  startup. The schema was expected to churn during development, and a
  migration tool would have been pure friction. **This is the main
  production gap**: deploying a schema change to an environment with real
  data requires a manual migration path that does not exist yet.
- **PDF blocks carry no section_path.** pdfplumber gives no structural
  heading markup the way DOCX styles or Markdown `#` do, so PDF citations
  are page-only (`policy.pdf · p.2`) rather than page+section. Font-size
  heuristics could add this later but are out of scope here.
- **DOCX heading-level fallback is best-effort.** For a renamed/custom
  style, the fallback reads `<w:outlineLvl>` off the paragraph or its
  style; a template that sets neither will not be recognised as a heading.
- **Effective-date extraction is best-effort regex**, not NLP. It only
  looks at the first ~2000 characters near a handful of keywords (or MD
  frontmatter). `null` is an expected, fine outcome.
- **S3Storage is exercised structurally only** (protocol conformance) in
  the test suite — no S3-compatible server runs in CI. It's driven entirely
  by env and targets R2/Supabase/S3 identically, but hasn't been run
  against a live bucket in this environment.
- **Chunk-count ceiling (3000) and file-size cap (25MB) are transport/cost
  guards, not content-quality guards.** A document right at the edge may
  still produce awkward chunk boundaries; the ceiling exists to bound cost,
  not to guarantee ideal chunking.

## Build order this followed

1. Config, DB, models, `/api/health`, structlog middleware.
2. Storage backends (local + S3), TXT extraction.
3. Chunking (segmenting, oversized-split, undersized-merge, overlap).
4. Embeddings, vectors, ingestion pipeline, upload/list/get/chunks/delete
   API, startup reconciler.
5. PDF/DOCX/MD extraction, `/retry`, `/stats`.

## Tests

```bash
pytest             # 36 tests, ~2s, all offline except test_ingestion_pipeline.py
```

`test_ingestion_pipeline.py` needs the docker-compose Postgres+Qdrant
running (it auto-skips otherwise) but fakes the embedder — no model
download, no CPU-bound work, no network. Every other test file is fully
offline: extraction and chunking tests run on handwritten `Block` lists or
committed fixtures, storage tests run against a tmp directory.

## API

All routes under `/api`. Every failure returns
`{"error": {"code", "message", "details"}, "request_id"}`.

| Method | Path | Notes |
|---|---|---|
| POST | `/documents` | Multipart, 1–10 files, 202 Accepted |
| GET | `/documents` | Optional `?status=` filter |
| GET | `/documents/{id}` | 404 if missing |
| GET | `/documents/{id}/chunks` | Paginated, joins Postgres rows with Qdrant text |
| POST | `/documents/{id}/retry` | 409 unless `status=failed` |
| DELETE | `/documents/{id}` | 409 while `processing`; Qdrant → chunk rows → file → row, in that order |
| GET | `/stats` | Documents by status, chunk count, Qdrant point count |
| GET | `/health` | Real Postgres/Qdrant/model checks |

## Phase two: retrieval + RAG + chat

Adds semantic search, cross-encoder reranking, grounded answers with
validated citations, conversation history, and a full evidence trace.
Contradiction detection is **not** built yet (phase three).

### New stack
Gemini (`google-genai`) for generation, `fastembed`'s `TextCrossEncoder`
for reranking (`Xenova/ms-marco-MiniLM-L-6-v2`, no new dependency — it
ships in `fastembed`, already installed).

### New API
| Method | Path | Notes |
|---|---|---|
| POST | `/chat` | `{query, conversation_id?, options?}` → answer + citations + trace |
| GET | `/conversations` | Newest first, with message counts |
| GET | `/conversations/{id}/messages` | Full history incl. stored citations/trace |

### Known deviations / gotchas
- **`LLM_MODEL` default changed from the original spec's `gemini-2.5-flash`
  to `gemini-3.6-flash`.** The 2.5 model is dead for new API keys as of
  this writing — Google's own 404 response says so. Model IDs rotate;
  check `client.aio.models.list()` if this breaks again later.
- **Gemini 3 models reject `thinking_budget=0` outright** (only the 2.x
  family accepted it). Using `thinking_level=MINIMAL` instead — the lowest
  this family allows, and empirically returns no `thoughts_token_count`.
- **Citation parsing handles grouped markers** (`[S1, S2, S3]`), not just
  single ones (`[S1]`) — real model output mixes both freely. Each marker
  in a group is validated independently; the bracket is dropped entirely
  only if every marker inside it is invalid.
- **Context token budget is a `len(text) // 4` approximation**, not a real
  tokenizer call — avoids a network round-trip to `count_tokens` on every
  request, and there's no official local Gemini tokenizer to call instead.
- **`RELEVANCE_FLOOR=0.3` is calibrated for prose-style policy/contract
  text**, matching this project's actual scope. It behaves poorly against
  non-prose content (resumes, bullet lists) with this cross-encoder —
  confirmed during Stage 2 testing, not a bug, just a corpus mismatch.

## Phase three: contradiction detection

Query-time contradiction detection running concurrently with answer
generation, off the same `RetrievalResult` — neither branch gates the
other. `sample_docs/` holds 5 planted policy documents (2 PDF, 2 DOCX, 1
MD) with `EXPECTED.md` as the test oracle.

### New API
| Method | Path | Notes |
|---|---|---|
| GET | `/contradictions` | `status` (default `open`), `severity`, `type`, `document_id`, `limit`, `offset` — includes global `counts` |
| GET | `/contradictions/{id}` | Single record, 404 if missing |
| PATCH | `/contradictions/{id}` | `{status, note}` — `false_positive` suppresses the pair on all future queries |

`POST /api/chat` gains `options.detect_contradictions` (default `true`) and
the response gains `contradictions[]` + `contradictions_total`.

### Known gaps / honest limitations
- **Not every planted pair was live-verified against the real LLM in this
  session.** The factual (fee $50/$75) and logical+temporal (attendance)
  pairs were confirmed live, repeatedly, with correct type/severity/
  reconciliation and passing span verification. The numerical (vacation
  15/20) and scope-difference (contractor vs employee) pairs were not,
  because this session hit Gemini's free-tier daily quota (20
  requests/day) before reaching them. Both mechanisms they'd exercise are
  independently verified: the numeric exemption via dedicated offline unit
  tests with controlled vectors reproducing the exact >0.97-cosine-with-
  differing-numbers scenario, and the scope-difference negative rule via
  the system prompt's own worked example (which the model demonstrably
  followed correctly in every other live judgment this session).
- **Neighbour expansion (phase two) dilutes contradiction pairing.** Since
  `ScoredChunk.text` is reused as-is from retrieval (per the spec's
  instruction to compute cosine on vectors "already on ScoredChunk"), a
  chunk selected for contradiction pairing may already have adjacent
  chunk text folded in — discovered while building the sample corpus, not
  a bug, just a real interaction between the two phases worth knowing.
- **Only genuine contradictions are cached.** The `contradictions` table's
  name and status semantics (open/resolved/false_positive) only make
  sense for true positives, so a pair the LLM judges `is_contradiction:
  false` is never stored — it may be re-adjudicated on a future query with
  the same corpus. "Repeat query, zero LLM calls" holds for pairs that
  were previously found to be real contradictions, not for every possible
  pair regardless of content.
- **PDF chunks stay one blob per page** (no section-level splitting,
  carried over from phase one) — a short multi-topic PDF page can dilute
  cosine similarity for a specific sub-topic comparison. Real, disclosed
  behavior, not something this phase tries to work around.
