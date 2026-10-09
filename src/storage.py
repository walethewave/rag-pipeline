"""Everything that reads or writes persisted data: the S3 Vectors store and the JSON document
registry. No metadata is ever stored or filtered on for scoping search - the `index` name (one
S3 Vectors index per app index, all inside one bucket) is the only way to scope a search. Chunk
text is stored as vector metadata only because S3 Vectors has nowhere else to put it - it is
marked non-filterable and is never used in a query's `filter`. The document registry is NOT
chunk metadata; it only answers 'what files are in this index' for listing, deleting and
re-upload replacement."""

import json
import logging
import os
import re
import threading
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from functools import lru_cache

import boto3

from src.config import get_settings
from src.errors import ValidationError
from src.pipeline import Chunk

logger = logging.getLogger(__name__)

_INDEX_NAME_PATTERN = re.compile(r"^[a-zA-Z0-9_-]{3,50}$")
_registry_lock = threading.Lock()


def validate_index_name(index: str) -> str:
    """Rejects anything that is not letters/numbers/_/- (3-50 chars) - this is the only way a
    search can be scoped, so a bad index name is caught here before it reaches S3 Vectors."""
    if not _INDEX_NAME_PATTERN.match(index):
        raise ValidationError(
            f"Invalid index name '{index}'. Use only letters, numbers, '_' and '-', 3-50 characters."
        )
    return index


# ==================================================================== vector store (S3 Vectors)


@lru_cache
def _client():
    """Builds one boto3 s3vectors client, reused for every call."""
    settings = get_settings()
    kwargs = {"region_name": settings.aws_region}
    if settings.aws_access_key_id and settings.aws_secret_access_key:
        kwargs["aws_access_key_id"] = settings.aws_access_key_id
        kwargs["aws_secret_access_key"] = settings.aws_secret_access_key
    return boto3.client("s3vectors", **kwargs)


def _ensure_index(index: str) -> None:
    """Creates the S3 Vectors index for this app `index` name on first use - the S3 Vectors
    equivalent of Chroma's get_or_create_collection. `text` is stored as metadata (S3 Vectors has
    no separate document-text field) but marked non-filterable, since the index name is the only
    thing allowed to scope a search - metadata is never used in a query filter."""
    client, settings = _client(), get_settings()
    try:
        resp = client.get_index(vectorBucketName=settings.s3vectors_bucket, indexName=index)
        existing_dim = resp.get("index", {}).get("dimension")
        if existing_dim and existing_dim != settings.embedding_dimension:
            logger.warning(
                "S3 Vectors index '%s' dimension %d differs from configured %d. Recreating index.",
                index, existing_dim, settings.embedding_dimension
            )
            client.delete_index(vectorBucketName=settings.s3vectors_bucket, indexName=index)
            raise client.exceptions.NotFoundException({}, "recreating")
    except client.exceptions.NotFoundException:
        client.create_index(
            vectorBucketName=settings.s3vectors_bucket,
            indexName=index,
            dataType="float32",
            dimension=settings.embedding_dimension,
            distanceMetric="cosine",
            metadataConfiguration={"nonFilterableMetadataKeys": ["text"]},
        )
        logger.info("Created S3 Vectors index '%s' (dim=%d) in bucket '%s'", index, settings.embedding_dimension, settings.s3vectors_bucket)


def add_chunks(index: str, chunks: list[Chunk]) -> None:
    """Adds chunks (key, vector, text-as-metadata) to the index for this `index` name, creating
    the index on first use. document_ingest's delete-then-add on re-upload is what keeps this
    safe even though S3 Vectors PutVectors is not guaranteed to upsert duplicate keys."""
    if not chunks:
        return
    validate_index_name(index)
    _ensure_index(index)
    vectors = [{"key": c.id, "data": {"float32": c.vector}, "metadata": {"text": c.text}} for c in chunks]
    _client().put_vectors(vectorBucketName=get_settings().s3vectors_bucket, indexName=index, vectors=vectors)
    logger.info("Added %d chunks to index '%s'", len(chunks), index)


def search(index: str, vector: list[float], k: int) -> list[tuple[str, str, float]]:
    """Searches only the given index and returns (id, text, similarity_score) triples, best
    match first. Returns [] if the index does not exist yet (nothing uploaded)."""
    validate_index_name(index)
    client, settings = _client(), get_settings()
    try:
        result = client.query_vectors(
            vectorBucketName=settings.s3vectors_bucket, indexName=index, topK=k,
            queryVector={"float32": vector}, returnMetadata=True, returnDistance=True,
        )
    except client.exceptions.NotFoundException:
        return []
    # S3 Vectors' cosine "distance" is 1 - cosine_similarity (confirmed empirically: identical
    # vectors -> distance 0.0, opposite vectors -> distance 2.0), same conversion as Chroma used.
    return [(v["key"], v["metadata"]["text"], 1 - v["distance"]) for v in result.get("vectors", [])]


def delete_chunks(index: str, ids: list[str]) -> int:
    """Deletes the given chunk ids (vector keys) from an index."""
    if not ids:
        return 0
    validate_index_name(index)
    client, settings = _client(), get_settings()
    try:
        client.delete_vectors(vectorBucketName=settings.s3vectors_bucket, indexName=index, keys=ids)
    except client.exceptions.NotFoundException:
        return 0
    logger.info("Deleted %d chunks from index '%s'", len(ids), index)
    return len(ids)


def delete_index(index: str) -> None:
    """Wipes an entire S3 Vectors index in one call - used for 'delete all data in this index'
    instead of listing every vector key and deleting them one by one. _ensure_index() will
    transparently recreate the index the next time something is uploaded into it."""
    validate_index_name(index)
    client, settings = _client(), get_settings()
    try:
        client.delete_index(vectorBucketName=settings.s3vectors_bucket, indexName=index)
        logger.info("Deleted S3 Vectors index '%s'", index)
    except client.exceptions.NotFoundException:
        pass


# ==================================================================== document registry (JSON file)


@dataclass
class DocumentRecord:
    """One row of the registry: a single uploaded file inside a single index."""

    doc_id: str
    index: str
    file_name: str
    uploaded_at: str
    chunk_ids: list[str]


def register_document(doc_id: str, index: str, file_name: str, chunk_ids: list[str]) -> DocumentRecord:
    """Adds or replaces the record for (doc_id, index), so re-uploading the same file overwrites
    its old entry instead of appending a duplicate. The lock guards the read-modify-write so two
    uploads landing at the same moment can't corrupt the file - see main.py's "Concurrency" note."""
    record = DocumentRecord(
        doc_id=doc_id, index=index, file_name=file_name,
        uploaded_at=datetime.now(timezone.utc).isoformat(), chunk_ids=chunk_ids,
    )
    with _registry_lock:
        data = _read_registry()
        data[_registry_key(doc_id, index)] = asdict(record)
        _write_registry(data)
    logger.info("Registered doc_id=%s in index=%s (%d chunks)", doc_id, index, len(chunk_ids))
    return record


def get_document(doc_id: str, index: str) -> DocumentRecord | None:
    """Looks up one document's record, e.g. to find its chunk_ids before deleting it."""
    row = _read_registry().get(_registry_key(doc_id, index))
    return DocumentRecord(**row) if row else None


def list_documents(index: str) -> list[DocumentRecord]:
    """Lists every document registered under one index."""
    return [DocumentRecord(**row) for row in _read_registry().values() if row["index"] == index]


def remove_document(doc_id: str, index: str) -> DocumentRecord | None:
    """Removes a document's registry entry and returns what was removed (or None if unknown)."""
    with _registry_lock:
        data = _read_registry()
        row = data.pop(_registry_key(doc_id, index), None)
        if row is not None:
            _write_registry(data)
    return DocumentRecord(**row) if row else None


def clear_index(index: str) -> int:
    """Removes every registry entry for one index (used alongside delete_index() to wipe an
    index completely). Returns how many document records were removed."""
    with _registry_lock:
        data = _read_registry()
        keys_to_remove = [key for key, row in data.items() if row["index"] == index]
        for key in keys_to_remove:
            del data[key]
        if keys_to_remove:
            _write_registry(data)
    return len(keys_to_remove)


def _registry_key(doc_id: str, index: str) -> str:
    """Documents are keyed by (doc_id, index) because the same file can be uploaded into more
    than one index."""
    return f"{index}:{doc_id}"


def _read_registry() -> dict:
    """Loads the whole registry file as one dict keyed by '<index>:<doc_id>'. Missing file (first
    run, or a fresh deploy) is treated as an empty registry rather than an error."""
    path = get_settings().document_registry_path
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _write_registry(data: dict) -> None:
    """Overwrites the whole registry file with `data` - every caller already read-modified a
    full copy under `_registry_lock`, so this is always a full replace, never a partial patch."""
    path = get_settings().document_registry_path
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
