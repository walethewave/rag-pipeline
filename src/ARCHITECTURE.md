# How this codebase fits together

Six files. Each one has a single job. `main.py` holds the FastAPI app and every endpoint, and
calls the other five directly - there is no routes/services/repositories layering, no dependency
container. The only two things injected via FastAPI's `Depends(...)` are the `Embedder` and the
`LLM`, because those are the two genuinely expensive resources (a loaded model, a network client)
that tests need to swap for fakes. Everything else is a plain function call.

```
request
  │
  ▼
main.py            the FastAPI app + all 7 endpoints
  │
  ├──▶ pipeline.py       load_pdf, clean_pages, chunk_document   (pure text processing)
  ├──▶ ai.py              Embedder, BedrockLLM, build_prompt      (the only AI model calls)
  └──▶ storage.py          AWS S3 Vectors + the JSON document registry (the only storage calls)

config.py / errors.py / schemas.py sit underneath all of the above: settings, error types,
and request/response shapes, used wherever they're needed.
```

## File by file

| File | What it does |
|---|---|
| `config.py` | Reads every setting (AWS creds, model ids, thresholds, limits) from `.env` into one `Settings` object via `get_settings()`. Nothing else reads `os.environ` directly. `chunk_overlap_tokens` is a computed property - always 10% of `max_chunk_tokens`, not a separate number that can drift out of sync. |
| `errors.py` | `AppError` and its subclasses (`ValidationError`, `NotFoundError`, `UnsupportedPDFError`). Any endpoint can just `raise` one - `main.py`'s `app_error_handler` catches it and formats the error envelope with the right HTTP status, so no endpoint needs its own try/except. |
| `schemas.py` | Every pydantic request/response model, plus the `{success, data, error}` envelope and its `ok()`/`fail()` builders, all in one file. |
| `pipeline.py` | `Page` and `Chunk` dataclasses, plus the three pure text-processing steps: `load_pdf` (pdfplumber, pypdf fallback), `clean_pages` (strips repeated headers/footers and page numbers), `chunk_document` (the core logic - one Q&A pair per chunk, 10% overlap on any answer that has to be split). No HTTP, no database, no AI model calls. |
| `ai.py` | Everything that calls an external AI model, through **AWS Bedrock**. `Embedder` wraps Amazon Titan Text v2 (`amazon.titan-embed-text-v2:0`, 512-dim normalized vectors) for turning text into vectors. `BedrockLLM` calls Claude Haiku 4.5 through Bedrock using your AWS access/secret key (or the default credential chain on Lambda). `build_prompt()` fills `prompts/faq_answer.yaml` with the retrieved, numbered, sourced context. `get_embedder()`/`get_llm()` build each one once and cache it (`@lru_cache`) - this is also where `main.py`'s `Depends(...)` points, so tests can override them with fakes. |
| `storage.py` | Everything that reads or writes persisted data. `add_chunks`/`search`/`delete_chunks`/`delete_index` wrap an **AWS S3 Vectors** index per `index` name, all inside one bucket (`S3VECTORS_BUCKET`) - `index` is the only way to scope a search; nothing is ever filtered by metadata. Chunk text has to be stored as S3 Vectors metadata (there's no separate text field like Chroma has), but it's marked non-filterable and `search()` never passes a `filter`. `_ensure_index()` creates the S3 Vectors index for an `index` name on first use (and transparently recreates it if an existing index's dimension no longer matches `EMBEDDING_DIMENSION`). `register_document`/`get_document`/`list_documents`/`remove_document`/`clear_index` read/write a small **local JSON registry file** (`DocumentRecord`) recording, per document, its file name and chunk ids - this is **not** chunk metadata and is never used to filter a search, only to list/delete/replace/wipe. A lock guards the registry's read-modify-write so concurrent uploads can't corrupt it. Known gap: this local file is not Lambda-safe (Lambda's filesystem outside `/tmp` is read-only / not shared across invocations) - flagged, not yet fixed. |
| `main.py` | The FastAPI app, the exception handlers, and all 7 endpoints: health, upload document, list documents, delete document, index stats, wipe an entire index, and chat. `_ingest()` and `_retrieve()` are the two functions doing the real work - see the call chains below. |

## Call chain 1: uploading a document

```
POST /api/v1/documents
  → main.py: upload_document()
      - storage.validate_index_name(index)
      - read + validate the uploaded bytes (.pdf, under MAX_UPLOAD_MB)
      - save to a temp file (_temp_pdf_file)
      - await asyncio.to_thread(_ingest, ...)      ◀── see "Concurrency" below
          → main.py: _ingest()
              - hash the file → doc_id
              - pipeline.py: load_pdf()
              - pipeline.py: clean_pages()
              - pipeline.py: chunk_document()        (one Q&A = one chunk, 10% overlap)
              - ai.py: Embedder.embed()               (one vector per chunk)
              - if doc_id already exists in this index: storage.py: delete_chunks()  (old ones)
              - storage.py: add_chunks()               (S3 Vectors PutVectors, creating the index on first use)
              - storage.py: register_document()        (JSON registry)
      - return {doc_id, index, chunk_count} in the envelope
```

## Call chain 2: asking a question

```
POST /api/v1/chat
  → main.py: chat()
      - storage.validate_index_name(index)
      - main.py: _retrieve()
          - ai.py: Embedder.embed_one(question)
          - storage.py: search()                      (scoped to one index, cosine similarity)
          - drop anything below SCORE_THRESHOLD; promote a near-exact match to position 0
          - storage.py: get_document()                 (look up each chunk's source file name)
      - if nothing was retrieved: return "I could not find this in the documents." - LLM is never called
      - else:
          - ai.py: build_prompt(question, chunks, sources)   (fills prompts/faq_answer.yaml)
          - ai.py: BedrockLLM.generate(prompt)                (calls Claude via AWS Bedrock)
      - return {answer, sources, retrieved_count, answered_from_documents} in the envelope
```

## Call chain 3: index stats and wiping an index

```
GET /api/v1/stats?index=faq
  → main.py: index_stats()
      - storage.validate_index_name(index)
      - storage.py: list_documents()                 (reads the JSON registry only - no S3 Vectors call)
      - return {index, document_count, chunk_count, documents} in the envelope

DELETE /api/v1/index/{index}
  → main.py: delete_index()
      - storage.validate_index_name(index)
      - storage.py: clear_index()                     (removes every registry entry for this index)
      - storage.py: delete_index()                    (one S3 Vectors call wipes the whole index)
      - return {index, documents_deleted} in the envelope
      - _ensure_index() will transparently recreate the S3 Vectors index next time something is uploaded
```

## Concurrency: why uploads run on a worker thread

`_ingest()` does blocking work - PDF parsing, embedding, and now a real network round-trip to S3
Vectors for every write. Calling it directly inside an `async def` endpoint would block the
process's single asyncio event loop for the whole ingest, so **every other request** (other
uploads, even `/chat` requests) would queue up behind it instead of running at the same time,
despite the endpoint looking "async".

`await asyncio.to_thread(_ingest, ...)` runs it on a worker thread instead, so multiple uploads
genuinely run in parallel. `storage.py`'s internal `threading.Lock` is what then makes it safe
for two of those threads to read-modify-write the registry JSON file at the same moment without
corrupting it. Measured: 3 concurrent uploads finish in roughly half the time of 3 sequential ones.

## What tests fake, and why

| Resource | Real backend | In tests |
|---|---|---|
| `Embedder` | a loaded sentence-transformers model | `FakeEmbedder` - swapped via FastAPI `Depends(get_embedder)` |
| `LLM` | a network call to AWS Bedrock | `FakeLLM` - swapped via FastAPI `Depends(get_llm)` |
| vector store (`storage.add_chunks`/`search`/`delete_chunks`) | a network call to AWS S3 Vectors | `FakeVectorStore` (in-memory) - swapped by `monkeypatch.setattr(storage, ...)` in `tests/conftest.py`, since these three are called as plain module functions, not through `Depends(...)` |
| document registry | a local JSON file | the real thing, pointed at a per-test temp file - it's local and fast, so there's nothing to fake |

Because `storage.add_chunks` etc. are plain functions (not FastAPI dependencies), any test that
needs to call them directly must do `from src import storage; storage.add_chunks(...)` (module
attribute access), **not** `from src.storage import add_chunks` - the latter binds the real
function at import time and would bypass the monkeypatch entirely. See the comment at the top of
`tests/test_storage.py` for the concrete example.
