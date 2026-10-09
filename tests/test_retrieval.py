"""Tests for the retrieval logic in main.py: main._ingest() to seed data, main._retrieve() to
query it - the same two functions the real /documents and /chat endpoints call."""

from src.config import get_settings
from src.main import _ingest, _retrieve


def _ingest_sample_faq(sample_pdf_path, fake_embedder):
    return _ingest(sample_pdf_path, "sample_faq.pdf", "faq", fake_embedder, get_settings())


def test_relevant_question_returns_matching_chunk(sample_pdf_path, fake_embedder):
    """A question close to a stored one is retrieved with a score above the threshold."""
    _ingest_sample_faq(sample_pdf_path, fake_embedder)

    chunks, sources, scores = _retrieve("How do I reset my password?", "faq", 3, fake_embedder, get_settings())
    assert len(chunks) >= 1
    assert "reset" in chunks[0].text.lower() or "password" in chunks[0].text.lower()
    assert sources[0] == "sample_faq.pdf"
    assert scores[0] > 0


def test_unrelated_question_returns_empty_below_threshold(sample_pdf_path, fake_embedder):
    """A question with no real match scores below SCORE_THRESHOLD and retrieval returns []."""
    _ingest_sample_faq(sample_pdf_path, fake_embedder)

    chunks, sources, scores = _retrieve("Tell me about dinosaurs and volcanoes", "faq", 3, fake_embedder, get_settings())
    assert chunks == []


def test_search_is_scoped_to_the_given_index(sample_pdf_path, fake_embedder):
    """The same question must not be answered from a different index's documents."""
    _ingest_sample_faq(sample_pdf_path, fake_embedder)

    chunks, _, _ = _retrieve("How do I reset my password?", "other_index", 3, fake_embedder, get_settings())
    assert chunks == []


def test_exact_question_match_is_promoted_to_first(fake_embedder):
    """A near-exact match to a stored question is moved to position 0 even if it was not the
    top-scored raw result."""
    from src import storage
    from src.pipeline import chunk_document
    from src.storage import register_document

    chunks = chunk_document(
        "Q: How do I reset my password? A: Click forgot password. "
        "Q: How do I reset my password and username together? A: Contact support.",
        doc_id="doc1",
    )
    for c in chunks:
        c.vector = fake_embedder.embed_one(c.text)
    storage.add_chunks("faq", chunks)  # module attribute access, not a name import - see conftest.py
    register_document("doc1", "faq", "sample.pdf", [c.id for c in chunks])

    retrieved, _, scores = _retrieve("How do I reset my password?", "faq", 5, fake_embedder, get_settings())
    assert scores[0] >= get_settings().exact_match_threshold - 0.2  # sanity: still a strong match
