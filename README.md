# FAQ RAG Pipeline

A small, readable Retrieval-Augmented Generation (RAG) pipeline for answering questions from
uploaded FAQ PDFs. Built as a teaching project: plain Python, short functions, no LangChain/LlamaIndex.

Upload a PDF → it's split one-Q&A-per-chunk → embedded → stored in **AWS S3 Vectors** → ask a
question → the answer is generated **only** from the retrieved chunks, with citations.

**No metadata filtering anywhere.** The only way to narrow a search is the `index` you upload
into and query - one S3 Vectors index per `index` name, all inside one bucket. Chunk text is
stored as S3 Vectors metadata only because that's the one place to put it; it's marked
non-filterable and never appears in a search filter.

## How it works

Six files, each with one job (see `src/ARCHITECTURE.md` for the full breakdown):

```
main.py       the FastAPI app and all 5 endpoints - calls the three files below directly
pipeline.py   load_pdf -> clean_pages -> chunk_document (one Q&A = one chunk, 10% overlap)
ai.py         Embedder, BedrockLLM, build_prompt - the only file that calls an external AI model
storage.py    AWS S3 Vectors (the vector store) + the JSON document registry
config.py     Settings, read from .env
schemas.py    pydantic request/response models + the envelope
```

**Indexing phase** (`POST /api/v1/documents`): the route saves the upload to a temp file, then
`main._ingest()` calls `load_pdf` → `clean_pages` → `chunk_document` → `Embedder.embed` →
`storage.add_chunks` (S3 Vectors) + `storage.register_document` (the file registry).

**Query phase** (`POST /api/v1/chat`): `main._retrieve()` calls `Embedder.embed_one(question)` →
`storage.search` (scoped to one `index`) → drops anything below `SCORE_THRESHOLD`. If nothing
passes, the LLM is never called - the API returns `"I could not find this in the documents."`
directly. Otherwise `build_prompt` → `BedrockLLM.generate` → answer + sources.

## Setup

```bash
cd faq-rag-pipeline
python3.13 -m venv .venv          # needs Python 3.11+
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env              # then fill in your real AWS keys
```

`.env` keys:

| Key | What it's for |
|---|---|
| `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` | Calls Claude through AWS Bedrock, and reads/writes AWS S3 Vectors (no Anthropic API key needed) |
| `AWS_REGION` | Region your Bedrock / S3 Vectors access is in |
| `BEDROCK_MODEL_ID` | A Bedrock **inference profile** id, e.g. `us.anthropic.claude-haiku-4-5-20251001-v1:0` (note the `us.` prefix - plain model ids are rejected by Bedrock for on-demand invocation; run `aws bedrock list-inference-profiles` to see what's available to your account) |
| `EMBEDDING_MODEL` / `EMBEDDING_DIMENSION` | sentence-transformers model name and its output size (`all-MiniLM-L6-v2` → 384). The dimension must be correct - it's used to create each S3 Vectors index. |
| `S3VECTORS_BUCKET` | The S3 Vectors bucket to store vectors in. **Must already exist** - S3 Vectors buckets are not created automatically (unlike indexes inside them, which `storage.py` creates on first use). |
| `DOCUMENT_REGISTRY_PATH` | Where the JSON document registry is written - this stays local even in production |
| `DEFAULT_INDEX` | Index used when a request doesn't name one (`faq`) |
| `MAX_CHUNK_TOKENS` | Chunk size budget. Overlap is always 10% of this, computed automatically - there's no separate overlap setting |
| `SCORE_THRESHOLD` / `EXACT_MATCH_THRESHOLD` | Retrieval cutoffs |
| `MAX_UPLOAD_MB` | Upload size limit |

**IAM:** the AWS credentials need an `s3vectors:*` policy (ListVectorBuckets, CreateIndex,
GetIndex, PutVectors, QueryVectors, DeleteVectors, etc.) on the bucket/account, plus
`bedrock:InvokeModel` for the LLM calls - S3 Vectors access is not granted by default.

**Lambda note:** the vector store (S3 Vectors) is a hosted AWS service, so it persists fine across
cold starts - no more Chroma-on-Lambda problem for the vectors themselves. `DOCUMENT_REGISTRY_PATH`
is still a local JSON file, though, and **local disk is wiped between Lambda cold starts** - for
a real Lambda deployment, point it at something that persists (e.g. a small DynamoDB table or an
S3 object) before relying on document listing/delete working correctly in production.

## Running locally

```bash
uvicorn src.main:app --reload
# API docs at http://localhost:8000/docs
```

## Example requests

```bash
# Make a sample FAQ PDF to try it with
python scripts/make_sample_pdf.py data/sample_faq.pdf

# Upload it into the "faq" index
curl -X POST http://localhost:8000/api/v1/documents \
  -F "file=@data/sample_faq.pdf" \
  -F "index=faq"

# Ask a question
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "How do I reset my password?", "index": "faq"}'

# List documents in an index
curl http://localhost:8000/api/v1/documents?index=faq

# Delete a document (use the doc_id from the upload response)
curl -X DELETE "http://localhost:8000/api/v1/documents/<doc_id>?index=faq"
```

## Running on AWS Lambda

`src/main.py` exposes `handler = Mangum(app)`. Point API Gateway (or a Lambda Function URL) at
that handler. See the document-registry caveat above before relying on this in production.

## Tests

```bash
pytest tests/ -v
```

Tests use a `FakeEmbedder` (hashed bag-of-words, stopwords removed) and `FakeLLM`
(`tests/fakes.py`), swapped in via FastAPI's `dependency_overrides`, so the suite runs in a
couple of seconds with no model download and no network call to Bedrock. The vector store is
also faked (`FakeVectorStore`, an in-memory dict) - `storage.py`'s real backend is AWS S3
Vectors, a real network call, so hitting it for every test would be slow and need AWS access.
Only the document registry (a local JSON file) runs for real, against a per-test temp directory.

## Evaluating retrieval quality

```bash
python scripts/evaluate.py scripts/sample_eval_questions.json --index faq --k 3
```

Prints **hit rate@k** (did the expected chunk show up in the top k results?) and **MRR** (how
high did it rank when it did show up?) over a small question set. Needs the `faq` index to
already have the sample PDF ingested (see "Example requests" above).

## Project layout

```
src/
  main.py            FastAPI app + all 5 endpoints + the Mangum Lambda handler
  pipeline.py         Page/Chunk dataclasses, load_pdf, clean_pages, chunk_document
  ai.py                Embedder, BedrockLLM, build_prompt, get_embedder()/get_llm()
  storage.py           AWS S3 Vectors + the JSON document registry, validate_index_name
  config.py            Settings (reads .env)
  errors.py            AppError and its subclasses
  schemas.py           pydantic request/response models + the envelope
  prompts/faq_answer.yaml   the system prompt, kept as data so it's editable without touching code
  ARCHITECTURE.md      file-by-file breakdown + the two call chains (upload, chat)
scripts/
  make_sample_pdf.py   builds a fake FAQ PDF for testing
  evaluate.py           hit rate@k / MRR over a question set
tests/
data/
```

See `src/ARCHITECTURE.md` for exactly which function calls which, including why document uploads
run on a worker thread (`asyncio.to_thread`) so multiple uploads genuinely run in parallel.

## Next improvements

- **Hybrid search** - combine the vector search with BM25 keyword search for queries that use
  exact terms (model numbers, error codes) the embedding model doesn't weight highly.
- **Reranking** - a cross-encoder reranker over the top ~20 vector results before picking the
  final k, for better precision on ambiguous questions.
- **Query decomposition** - split a multi-part question ("what's the refund policy and how do I
  contact billing") into sub-questions, retrieve for each, and combine.
- **OCR for scanned PDFs** - right now a PDF with no extractable text layer raises a clear error;
  adding AWS Textract would let scanned documents be indexed too.
- **Move the document registry off local disk** - required for a correct real Lambda deployment
  (see the Lambda note above); a small DynamoDB table is the natural fit.
