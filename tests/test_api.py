"""API-level tests: upload -> chat flow, validation errors, and the envelope shape. Uses the
`client` fixture from conftest.py, which runs the real FastAPI app against fakes."""


def _upload_sample(client, sample_pdf_path, index="faq"):
    with open(sample_pdf_path, "rb") as f:
        return client.post(
            "/api/v1/documents",
            files={"file": ("sample_faq.pdf", f, "application/pdf")},
            data={"index": index},
        )


def test_upload_then_chat_returns_grounded_answer(client, sample_pdf_path):
    """The full happy path: upload a PDF, then ask a question it should be able to answer."""
    upload_response = _upload_sample(client, sample_pdf_path)
    assert upload_response.status_code == 200
    body = upload_response.json()
    assert body["success"] is True
    assert body["data"]["chunk_count"] > 0

    chat_response = client.post("/api/v1/chat", json={"question": "How do I reset my password?", "index": "faq"})
    assert chat_response.status_code == 200
    chat_body = chat_response.json()["data"]
    assert chat_body["answered_from_documents"] is True
    assert chat_body["retrieved_count"] >= 1
    assert chat_body["sources"][0]["source"] == "sample_faq.pdf"


def test_unrelated_question_does_not_call_llm_answer(client, sample_pdf_path):
    """An unrelated question gets the fixed 'could not find' answer with no sources."""
    _upload_sample(client, sample_pdf_path)
    response = client.post("/api/v1/chat", json={"question": "Tell me about dinosaurs", "index": "faq"})
    data = response.json()["data"]
    assert data["answer"] == "I could not find this in the documents."
    assert data["sources"] == []
    assert data["answered_from_documents"] is False


def test_chitchat_greeting_calls_llm_with_main_prompt(client, sample_pdf_path, fake_llm):
    """A conversational greeting like 'hi' or 'hello' calls the LLM with the main prompt."""
    _upload_sample(client, sample_pdf_path)
    response = client.post("/api/v1/chat", json={"question": "hi", "index": "faq"})
    assert response.status_code == 200
    data = response.json()["data"]
    assert "Vest" in data["answer"]
    assert data["sources"] == []
    assert data["answered_from_documents"] is False
    assert fake_llm.last_prompt is not None
    assert "Greetings & Chitchat" in fake_llm.last_prompt
    assert "Question: hi" in fake_llm.last_prompt


def test_bad_index_name_returns_400(client):
    """Anything other than letters/numbers/_/- is rejected with a 400 and the error envelope."""
    response = client.get("/api/v1/documents", params={"index": "bad name!"})
    assert response.status_code == 400
    body = response.json()
    assert body["success"] is False
    assert body["error"]["code"] == "validation_error"


def test_non_pdf_upload_returns_400(client):
    """A non-PDF file is rejected before it ever reaches the ingest pipeline."""
    response = client.post(
        "/api/v1/documents", files={"file": ("notes.txt", b"just text", "text/plain")}
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "validation_error"


def test_delete_unknown_document_returns_404(client):
    """Deleting a doc_id that was never uploaded returns a 404 not_found error."""
    response = client.delete("/api/v1/documents/does-not-exist", params={"index": "faq"})
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "not_found"


def test_envelope_shape_on_every_response(client, sample_pdf_path):
    """Every response (success or failure) has exactly the success/data/error envelope shape."""
    responses = [
        client.get("/api/v1/health"),
        _upload_sample(client, sample_pdf_path),
        client.get("/api/v1/documents", params={"index": "faq"}),
        client.post("/api/v1/chat", json={"question": "hello", "index": "faq"}),
        client.get("/api/v1/documents", params={"index": "bad!"}),
    ]
    for response in responses:
        body = response.json()
        assert set(body.keys()) == {"success", "data", "error"}
        if body["success"]:
            assert body["error"] is None
        else:
            assert body["data"] is None
            assert "code" in body["error"] and "message" in body["error"]


def test_reupload_same_file_replaces_not_duplicates(client, sample_pdf_path):
    """Uploading the same file twice into the same index keeps one document entry with the
    same doc_id, not two."""
    first = _upload_sample(client, sample_pdf_path).json()["data"]
    second = _upload_sample(client, sample_pdf_path).json()["data"]
    assert first["doc_id"] == second["doc_id"]

    listing = client.get("/api/v1/documents", params={"index": "faq"}).json()["data"]
    assert len(listing["documents"]) == 1


def test_stats_reports_document_and_chunk_counts(client, sample_pdf_path):
    """GET /stats totals up exactly what was uploaded: one document, its real chunk count."""
    upload = _upload_sample(client, sample_pdf_path).json()["data"]

    stats = client.get("/api/v1/stats", params={"index": "faq"}).json()["data"]
    assert stats["document_count"] == 1
    assert stats["chunk_count"] == upload["chunk_count"]
    assert stats["documents"][0]["doc_id"] == upload["doc_id"]


def test_stats_on_empty_index_is_zero(client):
    """An index nothing was ever uploaded to reports zero, not an error."""
    stats = client.get("/api/v1/stats", params={"index": "never_used"}).json()["data"]
    assert stats == {"index": "never_used", "document_count": 0, "chunk_count": 0, "documents": []}


def test_delete_index_wipes_everything_and_chat_then_finds_nothing(client, sample_pdf_path):
    """DELETE /index/{index} removes every document, and a question that used to be answerable
    now correctly gets 'could not find' - the vectors are really gone, not just the registry."""
    _upload_sample(client, sample_pdf_path)
    before = client.post("/api/v1/chat", json={"question": "How do I reset my password?", "index": "faq"})
    assert before.json()["data"]["answered_from_documents"] is True

    delete_response = client.delete("/api/v1/index/faq")
    assert delete_response.status_code == 200
    assert delete_response.json()["data"]["documents_deleted"] == 1

    assert client.get("/api/v1/stats", params={"index": "faq"}).json()["data"]["document_count"] == 0
    after = client.post("/api/v1/chat", json={"question": "How do I reset my password?", "index": "faq"})
    assert after.json()["data"]["answered_from_documents"] is False


def test_adding_a_second_document_does_not_remove_the_first(client, sample_pdf_path, tmp_path):
    """Uploading a new file into an index that already has one leaves the existing document
    alone - this is the 'add to existing data' path, same endpoint as the first upload."""
    from reportlab.pdfgen import canvas

    second_path = str(tmp_path / "second.pdf")
    c = canvas.Canvas(second_path)
    c.drawString(100, 750, "Q: What is a widget? A: A small standalone test document, unrelated to the FAQ.")
    c.save()  # genuinely different bytes from sample_pdf_path -> a different doc_id

    _upload_sample(client, sample_pdf_path)
    with open(second_path, "rb") as f:
        client.post("/api/v1/documents", files={"file": ("second.pdf", f, "application/pdf")}, data={"index": "faq"})

    stats = client.get("/api/v1/stats", params={"index": "faq"}).json()["data"]
    assert stats["document_count"] == 2
    file_names = {d["file_name"] for d in stats["documents"]}
    assert file_names == {"sample_faq.pdf", "second.pdf"}
