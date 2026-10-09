"""Tests for storage.py.

Vector-store tests call through `storage.add_chunks(...)` etc. (module attribute access, not a
name import) because conftest.py's isolated_storage fixture monkeypatches those three functions
on the `storage` module object to an in-memory fake - a plain `from src.storage import add_chunks`
would bind the real (S3 Vectors) function at import time and bypass the fake entirely.

Document-registry tests import by name as usual: the registry is a real local JSON file in both
tests and production, so it never needs faking.
"""

import pytest

from src import storage
from src.errors import ValidationError
from src.pipeline import Chunk
from src.storage import (
    get_document,
    list_documents,
    register_document,
    remove_document,
    validate_index_name,
)


def test_reupload_replaces_instead_of_duplicating():
    """Adding chunks with the same ids again overwrites them (upsert by id), it does not
    create duplicate entries - this is what makes 're-upload the same file' safe."""
    storage.add_chunks("faq", [Chunk(id="doc1-0000", text="old answer", vector=[1.0, 0.0])])
    storage.add_chunks("faq", [Chunk(id="doc1-0000", text="new answer", vector=[1.0, 0.0])])

    results = storage.search("faq", [1.0, 0.0], k=5)
    assert len(results) == 1
    assert results[0][1] == "new answer"


def test_delete_removes_chunks():
    """Deleted chunk ids no longer appear in search results."""
    storage.add_chunks("faq", [Chunk(id="doc1-0000", text="hello", vector=[1.0, 0.0])])
    storage.delete_chunks("faq", ["doc1-0000"])

    assert storage.search("faq", [1.0, 0.0], k=5) == []


def test_index_only_returns_its_own_chunks():
    """An index is the only way to scope a search - chunks in 'hr_docs' must never appear when
    searching 'faq', even with an identical vector."""
    storage.add_chunks("faq", [Chunk(id="faq-0000", text="faq answer", vector=[1.0, 0.0])])
    storage.add_chunks("hr_docs", [Chunk(id="hr-0000", text="hr answer", vector=[1.0, 0.0])])

    assert [r[1] for r in storage.search("faq", [1.0, 0.0], k=5)] == ["faq answer"]
    assert [r[1] for r in storage.search("hr_docs", [1.0, 0.0], k=5)] == ["hr answer"]


def test_search_on_unused_index_returns_empty():
    """Searching an index nothing was ever uploaded to returns [] instead of raising."""
    assert storage.search("never_used", [1.0, 0.0], k=5) == []


@pytest.mark.parametrize("bad_name", ["bad name", "bad!name", "a" * 51, ""])
def test_invalid_index_names_are_rejected(bad_name):
    """Only letters, numbers, '_' and '-', 3-50 chars - anything else raises ValidationError."""
    with pytest.raises(ValidationError):
        validate_index_name(bad_name)


def test_valid_index_names_are_accepted():
    for name in ["faq", "hr-docs", "team_2", "A" * 50]:
        assert validate_index_name(name) == name


def test_document_registry_roundtrip():
    """register_document -> get_document -> list_documents -> remove_document, each step
    reflecting what the previous one did."""
    register_document("doc1", "faq", "sample.pdf", ["doc1-0000", "doc1-0001"])

    record = get_document("doc1", "faq")
    assert record.file_name == "sample.pdf"
    assert record.chunk_ids == ["doc1-0000", "doc1-0001"]
    assert [d.doc_id for d in list_documents("faq")] == ["doc1"]

    removed = remove_document("doc1", "faq")
    assert removed.doc_id == "doc1"
    assert get_document("doc1", "faq") is None
    assert list_documents("faq") == []


def test_reregistering_same_doc_id_replaces_not_duplicates():
    """Registering the same (doc_id, index) twice updates the one record, it doesn't add a second."""
    register_document("doc1", "faq", "v1.pdf", ["doc1-0000"])
    register_document("doc1", "faq", "v2.pdf", ["doc1-0000", "doc1-0001"])

    assert len(list_documents("faq")) == 1
    assert get_document("doc1", "faq").file_name == "v2.pdf"
