"""Shared pytest fixtures.

The document registry is a local JSON file (storage.py reads its path from get_settings() on
every call, so pointing DOCUMENT_REGISTRY_PATH at tmp_path is enough to isolate it per test).
The vector store is real AWS S3 Vectors in production, so for tests it is swapped for an
in-memory FakeVectorStore - hitting the real service would make the suite slow, need network
access, and require creating/tearing down a real index per test.
"""

import pytest
from fastapi.testclient import TestClient

from src import storage
from src.ai import get_embedder, get_llm
from src.main import app
from tests.fakes import FakeEmbedder, FakeLLM, FakeVectorStore


@pytest.fixture(autouse=True)
def isolated_storage(tmp_path, monkeypatch):
    """Points the document registry at a fresh temp file, and swaps storage.py's vector-store
    functions for an in-memory fake, so every test is isolated and runs with no network access.

    Also lowers SCORE_THRESHOLD: the production value (0.45 in .env) is tuned for AWS Bedrock
    Titan's score distribution. FakeEmbedder's cruder bag-of-words cosine scores land lower even
    for a strong match (see tests/fakes.py: same-topic ~0.7, cross-topic ~0.2, unrelated ~0.0),
    so 0.5 keeps that real/unrelated separation meaningful for tests without being tied to the
    real model's tuning.
    """
    monkeypatch.setenv("DOCUMENT_REGISTRY_PATH", str(tmp_path / "documents.json"))
    monkeypatch.setenv("SCORE_THRESHOLD", "0.5")

    fake_store = FakeVectorStore()
    monkeypatch.setattr(storage, "add_chunks", fake_store.add_chunks)
    monkeypatch.setattr(storage, "search", fake_store.search)
    monkeypatch.setattr(storage, "delete_chunks", fake_store.delete_chunks)
    monkeypatch.setattr(storage, "delete_index", fake_store.delete_index)


@pytest.fixture
def fake_embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
def fake_llm() -> FakeLLM:
    return FakeLLM()


@pytest.fixture
def client(fake_embedder, fake_llm):
    """A TestClient for the real FastAPI app, with the Embedder and LLM swapped for fakes, so
    tests never load a real model or call Bedrock."""
    app.dependency_overrides[get_embedder] = lambda: fake_embedder
    app.dependency_overrides[get_llm] = lambda: fake_llm
    yield TestClient(app)
    app.dependency_overrides.clear()


@pytest.fixture
def sample_pdf_path(tmp_path) -> str:
    """Builds the sample FAQ PDF once per test into a temp file."""
    from scripts.make_sample_pdf import build_pdf

    path = str(tmp_path / "sample_faq.pdf")
    build_pdf(path)
    return path
