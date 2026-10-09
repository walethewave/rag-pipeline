"""Every pydantic request/response shape, plus the one envelope every endpoint returns, in one file."""

from typing import Any

from pydantic import BaseModel, Field

# ---------------------------------------------------------------- the response envelope


def ok(data: Any) -> dict:
    """Builds a successful envelope. Endpoints call this instead of returning raw data."""
    return {"success": True, "data": data, "error": None}


def fail(code: str, message: str) -> dict:
    """Builds a failed envelope. Endpoints call this inside their except blocks."""
    return {"success": False, "data": None, "error": {"code": code, "message": message}}


# ---------------------------------------------------------------- /documents


class DocumentUploadResponse(BaseModel):
    """What POST /documents returns inside the envelope's 'data' field."""

    doc_id: str
    index: str
    file_name: str
    chunk_count: int


class DocumentInfo(BaseModel):
    """One row in the GET /documents listing."""

    doc_id: str
    file_name: str
    index: str
    uploaded_at: str
    chunk_count: int


class DocumentListResponse(BaseModel):
    """What GET /documents returns: the index name plus its documents."""

    index: str
    documents: list[DocumentInfo]


class DocumentDeleteResponse(BaseModel):
    """What DELETE /documents/{doc_id} returns: confirmation of what was removed."""

    doc_id: str
    index: str
    chunks_deleted: int


class IndexStats(BaseModel):
    """What GET /stats returns: a snapshot of one index's size."""

    index: str
    document_count: int
    chunk_count: int
    documents: list[DocumentInfo]


class IndexDeleteResponse(BaseModel):
    """What DELETE /index/{index} returns: confirmation of a full index wipe."""

    index: str
    documents_deleted: int


# ---------------------------------------------------------------- /chat


class ChatRequest(BaseModel):
    """Body of POST /chat. 'index' defaults to the configured default index (normally 'faq')."""

    question: str = Field(min_length=1, max_length=2000)
    index: str | None = None
    k: int = Field(default=2, ge=1, le=10)
    debug: bool = False


class Source(BaseModel):
    """One citation shown under an answer: which file and which stored question it came from."""

    source: str
    question: str
    score: float


class ChatResponse(BaseModel):
    """What POST /chat returns inside the envelope's 'data' field."""

    answer: str
    sources: list[Source]
    retrieved_count: int
    answered_from_documents: bool
    debug_context: str | None = None  # only filled when request.debug is True
