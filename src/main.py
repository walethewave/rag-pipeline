"""The FastAPI app AND every endpoint - two phases, three routes.

Indexing phase  (POST /api/v1/documents):
    load_pdf -> clean_pages -> chunk_document -> Embedder.embed -> storage.add_chunks + register_document

Query phase  (POST /api/v1/chat):
    Embedder.embed_one -> storage.search (scoped to one index) -> score filter -> build_prompt -> LLM.generate

Run locally with: uvicorn src.main:app --reload
On Lambda, the `handler` object at the bottom is what API Gateway invokes (via Mangum).
"""

import asyncio
import contextlib
import hashlib
import logging
import os
import tempfile

from fastapi import Depends, FastAPI, File, Form, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from mangum import Mangum

from src import storage
from src.ai import LLM, Embedder, build_prompt, get_embedder, get_llm
from src.config import Settings, get_settings
from src.errors import AppError, NotFoundError, ValidationError
from src.pipeline import Chunk, chunk_document, clean_pages, load_pdf
from src.schemas import (
    ChatRequest,
    ChatResponse,
    DocumentDeleteResponse,
    DocumentInfo,
    DocumentListResponse,
    DocumentUploadResponse,
    IndexDeleteResponse,
    IndexStats,
    Source,
    fail,
    ok,
)

logging.basicConfig(level=get_settings().log_level, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)

NOT_FOUND_ANSWER = "I could not find this in the documents."

app = FastAPI(title="FAQ RAG Pipeline", version="1.0.0")

# The frontend (frontend/, a separate Next.js dev server) calls this API from the browser, which
# needs CORS allowed explicitly. Wide open here because this is a teaching project with no
# cookies/session auth; tighten allow_origins to a real domain list before shipping this for real.
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


@app.exception_handler(AppError)
async def app_error_handler(request: Request, exc: AppError) -> JSONResponse:
    """Every endpoint below just `raise`s a ValidationError/NotFoundError/etc. - this one handler
    turns it into the right HTTP status code and the {success, data, error} envelope, so no
    endpoint needs its own try/except block."""
    return JSONResponse(status_code=exc.status_code, content=fail(exc.code, exc.message))


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catches anything unexpected (a real bug, a library error) so the API never leaks a raw
    stack trace - it always answers in the envelope shape."""
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(status_code=500, content=fail("internal_error", "Something went wrong."))


@app.get("/api/v1/health")
def health_check() -> dict:
    """Liveness check - proves the process is up, no real work done."""
    return ok({"status": "healthy"})


# ==================================================================== POST /api/v1/documents


@app.post("/api/v1/documents")
async def upload_document(
    file: UploadFile = File(...),
    index: str | None = Form(default=None),
    settings: Settings = Depends(get_settings),
    embedder: Embedder = Depends(get_embedder),
):
    """Accepts one PDF upload and indexes it: validate -> save to a temp file -> ingest.

    _ingest() is blocking (PDF parsing, chunking, embedding, S3 Vectors writes). Running it directly
    here would block this process's single asyncio event loop for the whole ingest, so every
    other request - including other uploads - would queue up behind it. asyncio.to_thread() runs
    it on a worker thread instead, so multiple uploads genuinely run in parallel; storage.py's
    registry lock is what then makes concurrent writes to the registry file safe.
    """
    index = storage.validate_index_name(index or settings.default_index)
    content = await file.read()
    _validate_pdf_upload(file.filename, content, settings)

    with _temp_pdf_file(content) as tmp_path:
        doc_id, chunk_count = await asyncio.to_thread(_ingest, tmp_path, file.filename, index, embedder, settings)

    data = DocumentUploadResponse(doc_id=doc_id, index=index, file_name=file.filename, chunk_count=chunk_count)
    return ok(data.model_dump())


def _ingest(pdf_path: str, file_name: str, index: str, embedder: Embedder, settings: Settings) -> tuple[str, int]:
    """The indexing phase, called on a worker thread by the route above. Hashes the file into a
    stable doc_id -> loads and cleans the text -> chunks it (one Q&A per chunk, 10% overlap) ->
    embeds every chunk -> if this doc_id was already in this index, deletes its old chunks first
    (re-uploading the same file replaces it rather than duplicating it) -> stores the new chunks
    and registers the document."""
    doc_id = _hash_file(pdf_path)
    logger.info("Ingesting '%s' (doc_id=%s) into index '%s'", file_name, doc_id, index)

    pages = load_pdf(pdf_path)
    text = clean_pages(pages)
    chunks = chunk_document(
        text, doc_id=doc_id, max_tokens=settings.max_chunk_tokens, overlap=settings.chunk_overlap_tokens
    )

    vectors = embedder.embed([c.text for c in chunks])
    for chunk, vector in zip(chunks, vectors):
        chunk.vector = vector

    existing = storage.get_document(doc_id, index)
    if existing:
        storage.delete_chunks(index, existing.chunk_ids)

    storage.add_chunks(index, chunks)
    storage.register_document(doc_id, index, file_name, [c.id for c in chunks])

    logger.info("Ingested doc_id=%s: %d chunks", doc_id, len(chunks))
    return doc_id, len(chunks)


def _hash_file(path: str) -> str:
    """Hashes the file's bytes (not its name) so the same content always gets the same doc_id."""
    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8192), b""):
            sha256.update(block)
    return sha256.hexdigest()[:16]


def _validate_pdf_upload(file_name: str | None, content: bytes, settings: Settings) -> None:
    """Rejects non-PDF files and files over the configured size limit."""
    if not file_name or not file_name.lower().endswith(".pdf"):
        raise ValidationError("Only .pdf files are accepted.")
    if len(content) > settings.max_upload_mb * 1024 * 1024:
        raise ValidationError(f"File is larger than the {settings.max_upload_mb}MB upload limit.")


@contextlib.contextmanager
def _temp_pdf_file(content: bytes):
    """Writes the uploaded bytes to a temp file (pdfplumber/pypdf need a file path, not bytes),
    and deletes it afterwards no matter what happens during ingest."""
    fd, path = tempfile.mkstemp(suffix=".pdf")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(content)
        yield path
    finally:
        os.remove(path)


# ==================================================================== GET/DELETE /api/v1/documents


@app.get("/api/v1/documents")
def list_documents(index: str | None = Query(default=None), settings: Settings = Depends(get_settings)):
    """Lists every uploaded document inside one index (defaults to the default index)."""
    index = storage.validate_index_name(index or settings.default_index)
    records = storage.list_documents(index)
    documents = [
        DocumentInfo(doc_id=r.doc_id, file_name=r.file_name, index=r.index,
                     uploaded_at=r.uploaded_at, chunk_count=len(r.chunk_ids))
        for r in records
    ]
    return ok(DocumentListResponse(index=index, documents=documents).model_dump())


@app.delete("/api/v1/documents/{doc_id}")
def delete_document(doc_id: str, index: str | None = Query(default=None), settings: Settings = Depends(get_settings)):
    """Removes a document's chunks from the vector store and its entry from the registry."""
    index = storage.validate_index_name(index or settings.default_index)
    record = storage.get_document(doc_id, index)
    if record is None:
        raise NotFoundError(f"No document '{doc_id}' found in index '{index}'.")

    deleted = storage.delete_chunks(index, record.chunk_ids)
    storage.remove_document(doc_id, index)
    data = DocumentDeleteResponse(doc_id=doc_id, index=index, chunks_deleted=deleted)
    return ok(data.model_dump())


# ==================================================================== GET /api/v1/stats, DELETE /api/v1/index/{index}


@app.get("/api/v1/stats")
def index_stats(index: str | None = Query(default=None), settings: Settings = Depends(get_settings)):
    """Returns how big one index is: document count, total chunk count, and the document list.
    Computed entirely from the local registry (no S3 Vectors call needed) since the registry
    already tracks exactly this - how many chunks came from each uploaded document."""
    index = storage.validate_index_name(index or settings.default_index)
    records = storage.list_documents(index)
    documents = [
        DocumentInfo(doc_id=r.doc_id, file_name=r.file_name, index=r.index,
                     uploaded_at=r.uploaded_at, chunk_count=len(r.chunk_ids))
        for r in records
    ]
    data = IndexStats(
        index=index,
        document_count=len(documents),
        chunk_count=sum(d.chunk_count for d in documents),
        documents=documents,
    )
    return ok(data.model_dump())


@app.delete("/api/v1/index/{index}")
def delete_index(index: str):
    """Wipes an entire index: every chunk, in one S3 Vectors call, plus every registry entry for
    it. Unlike DELETE /documents/{doc_id} (one document), this removes everything in `index`."""
    index = storage.validate_index_name(index)
    documents_deleted = storage.clear_index(index)
    storage.delete_index(index)
    data = IndexDeleteResponse(index=index, documents_deleted=documents_deleted)
    return ok(data.model_dump())


# ==================================================================== POST /api/v1/chat


@app.post("/api/v1/chat")
def chat(
    request: ChatRequest,
    settings: Settings = Depends(get_settings),
    embedder: Embedder = Depends(get_embedder),
    llm: LLM = Depends(get_llm),
):
    """The query phase: search the vector database for relevant chunks, build the prompt,
    and let the AI decide whether to answer using the retrieved context or respond to chitchat."""
    index = storage.validate_index_name(request.index or settings.default_index)

    chunks, sources, scores = _retrieve(request.question, index, request.k, embedder, settings)
    prompt = build_prompt(request.question, chunks, sources)
    answer = llm.generate(prompt)

    answered_from_documents = bool(chunks and NOT_FOUND_ANSWER not in answer)
    data = ChatResponse(
        answer=answer,
        sources=[_to_source(c, s, score) for c, s, score in zip(chunks, sources, scores)] if answered_from_documents else [],
        retrieved_count=len(chunks),
        answered_from_documents=answered_from_documents,
        debug_context=prompt if request.debug else None,
    )
    return ok(data.model_dump())


def _retrieve(
    question: str, index: str, k: int, embedder: Embedder, settings: Settings
) -> tuple[list[Chunk], list[str], list[float]]:
    """Embeds the question with the SAME embedder used for the chunks, searches only `index`,
    and drops anything below SCORE_THRESHOLD. A near-exact match to a stored question
    (>= EXACT_MATCH_THRESHOLD) is promoted to the front, since that is almost certainly the
    single best answer even if another chunk scored marginally higher. Returns parallel lists of
    chunks, their source file names, and their scores."""
    vector = embedder.embed_one(question)
    raw_results = storage.search(index, vector, k)

    hits = [
        (Chunk(id=id_, text=text), score)
        for id_, text, score in raw_results
        if score >= settings.score_threshold
    ]
    for i, (chunk, score) in enumerate(hits):
        if score >= settings.exact_match_threshold and i != 0:
            hits.insert(0, hits.pop(i))
            break

    logger.info("Retrieved %d/%d chunks above threshold for index '%s'", len(hits), len(raw_results), index)
    chunks = [h[0] for h in hits]
    scores = [h[1] for h in hits]
    sources = [_source_for(c.id, index) for c in chunks]
    return chunks, sources, scores


def _source_for(chunk_id: str, index: str) -> str:
    """Looks up which uploaded file a chunk id belongs to, for citations. Chunk ids are
    '<doc_id>-<number>', so the doc_id is everything before the last '-'."""
    doc_id = chunk_id.rsplit("-", 1)[0]
    record = storage.get_document(doc_id, index)
    return record.file_name if record else "unknown"


def _to_source(chunk: Chunk, source: str, score: float) -> Source:
    """Builds one citation: file name, the stored question (first line of the chunk), and the score."""
    first_line = chunk.text.splitlines()[0]
    question = first_line.removeprefix("Question:").strip() or first_line
    return Source(source=source, question=question, score=round(score, 4))


# AWS Lambda entry point: API Gateway calls this via Mangum, which adapts the Lambda
# event/context into an ASGI request FastAPI can handle.
handler = Mangum(app)
