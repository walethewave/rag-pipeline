"""Tests for pipeline.py's chunk_document() step - the core chunking logic."""

from src.pipeline import chunk_document, count_tokens


def test_marker_format_qa_detected():
    """'Q: ... A: ...' becomes one chunk formatted as Question:/Answer:."""
    text = "Q: How do I reset my password? A: Click the forgot password link."
    chunks = chunk_document(text, doc_id="d1")
    assert len(chunks) == 1
    assert "Question: How do I reset my password?" in chunks[0].text
    assert "Answer: Click the forgot password link." in chunks[0].text


def test_question_answer_word_format_detected():
    """'Question: ... Answer: ...' is detected the same way as 'Q:'/'A:'."""
    text = "Question: What are your hours? Answer: We are open 9 to 5."
    chunks = chunk_document(text, doc_id="d1")
    assert len(chunks) == 1
    assert "What are your hours?" in chunks[0].text


def test_numbered_format_detected():
    """'1. Question? answer text' is detected, and each number starts a new chunk."""
    text = "1. How do I log in? Use your email and password. 2. How do I log out? Click the menu."
    chunks = chunk_document(text, doc_id="d1")
    assert len(chunks) == 2
    assert "How do I log in?" in chunks[0].text
    assert "How do I log out?" in chunks[1].text


def test_bare_question_format_detected():
    """A capitalised sentence ending in '?' with no marker is still detected as a question."""
    text = "Is my data encrypted? Yes, with AES-256 at rest and TLS in transit."
    chunks = chunk_document(text, doc_id="d1")
    assert len(chunks) == 1
    assert "Is my data encrypted?" in chunks[0].text


def test_one_chunk_per_qa_pair_not_merged():
    """Multiple Q&As in different formats in the same document each become their own chunk,
    and one question's answer never swallows the next question."""
    text = (
        "Q: How do I reset my password? A: Click forgot password. "
        "1. How do I invite a teammate? Go to Team and click invite. "
        "Is my data encrypted? Yes, always."
    )
    chunks = chunk_document(text, doc_id="d1")
    assert len(chunks) == 3
    assert "reset my password" in chunks[0].text
    assert "invite a teammate" in chunks[1].text
    assert "data encrypted" in chunks[2].text
    # the first chunk's answer must not have swallowed the later questions
    assert "invite" not in chunks[0].text.lower()


def test_long_answer_is_split_and_question_is_repeated_on_every_piece():
    """Fallback A: an answer longer than max_tokens is split, and every piece still starts with
    the same question, so no piece loses its subject."""
    long_answer = "This policy explains data retention in detail. " * 60
    text = f"Q: What happens to my data if I cancel? A: {long_answer}"
    chunks = chunk_document(text, doc_id="d1", max_tokens=100, overlap=10)
    assert len(chunks) > 1
    for chunk in chunks:
        assert "Question: What happens to my data if I cancel?" in chunk.text
        assert count_tokens(chunk.text) <= 110  # small slack for the repeated question header


def test_no_qa_structure_falls_back_to_recursive_split():
    """Fallback B: plain prose with no question marks still produces non-empty chunks."""
    text = "This is a plain policy document with no questions in it at all. " * 30
    chunks = chunk_document(text, doc_id="d1", max_tokens=50, overlap=5)
    assert len(chunks) > 1
    assert all(count_tokens(c.text) <= 60 for c in chunks)
    assert not any(c.text.startswith("Question:") for c in chunks)


def test_no_empty_or_tiny_chunks_are_produced():
    """Chunks under 15 characters are dropped, and blank/whitespace input produces nothing."""
    assert chunk_document("   \n\n  ", doc_id="d1") == []
    chunks = chunk_document("Q: Hi? A: ok.", doc_id="d1")
    assert all(len(c.text) >= 15 for c in chunks)


def test_chunk_ids_are_prefixed_with_doc_id():
    """Chunk ids must start with '<doc_id>-' so a document's chunks can be found and deleted later."""
    chunks = chunk_document("Q: Hello? A: World, this is a real answer.", doc_id="mydoc")
    assert chunks[0].id.startswith("mydoc-")
