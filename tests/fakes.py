"""Fake Embedder, LLM and vector store used by every test, so the test suite runs fast and fully
offline - no model download, no network call to Bedrock or to AWS S3 Vectors."""

import hashlib

_VECTOR_SIZE = 512

# Every stored chunk literally contains words like "question"/"answer"/"how"/"do"/"the" - a
# real embedding model understands these carry little topical meaning, but a plain bag-of-words
# fake would let them dominate and make every chunk look similar to every other one. Stripping
# them out keeps this fake embedder discriminative enough for retrieval tests to be meaningful.
_STOPWORDS = {
    "question", "answer", "a", "an", "the", "is", "are", "do", "does", "did", "i", "you", "your",
    "my", "me", "we", "to", "of", "for", "and", "or", "if", "in", "on", "at", "it", "this", "that",
    "what", "how", "can", "will", "be", "with", "from", "as", "up", "so", "not", "no", "yes",
}


class FakeEmbedder:
    """A deterministic 'embedding' built from hashed topic words (a tiny bag-of-words, stopwords
    removed), not a real model. Texts that share topic words get vectors pointing the same way,
    so cosine similarity behaves sensibly enough to test retrieval logic (relevant vs unrelated)."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_one(t) for t in texts]

    def embed_one(self, text: str) -> list[float]:
        vector = [0.0] * _VECTOR_SIZE
        for word in text.lower().split():
            word = word.strip(".,!?:;'\"()")
            if len(word) < 3 or word in _STOPWORDS:
                continue
            bucket = int(hashlib.md5(word.encode()).hexdigest(), 16) % _VECTOR_SIZE
            vector[bucket] += 1.0
        norm = sum(v * v for v in vector) ** 0.5
        return [v / norm for v in vector] if norm else vector


class FakeLLM:
    """Returns a canned, inspectable reply instead of calling Bedrock. Lets tests assert on the
    prompt that chat_service.py built without needing network access."""

    def __init__(self):
        self.last_prompt: str | None = None

    def generate(self, prompt: str) -> str:
        self.last_prompt = prompt
        if "(No matching document excerpts found)" in prompt or "(No document excerpts found)" in prompt:
            for line in prompt.splitlines():
                if line.startswith("Question:"):
                    q = line.removeprefix("Question:").strip().lower()
                    if any(w in q for w in ["hi", "hello", "hey", "what do you do", "who are you"]):
                        return "Hello! I am Vest, your support assistant."
            return "I could not find this in the documents."
        return "FAKE ANSWER"


class FakeVectorStore:
    """An in-memory stand-in for storage.py's add_chunks/search/delete_chunks, used because the
    real backend is AWS S3 Vectors - a real network call, and per-test index create/teardown
    would make the suite slow and dependent on AWS being reachable. conftest.py monkeypatches
    storage.py's three functions to these methods for the duration of each test."""

    def __init__(self):
        self._indexes: dict[str, list[tuple[str, str, list[float]]]] = {}

    def add_chunks(self, index: str, chunks) -> None:
        """Upserts by id: drops any existing row sharing a new chunk's id, then adds the new ones."""
        bucket = self._indexes.setdefault(index, [])
        new_ids = {c.id for c in chunks}
        bucket[:] = [row for row in bucket if row[0] not in new_ids]
        bucket.extend((c.id, c.text, c.vector) for c in chunks)

    def search(self, index: str, vector: list[float], k: int) -> list[tuple[str, str, float]]:
        bucket = self._indexes.get(index, [])
        scored = [(id_, text, _cosine_similarity(vector, v)) for id_, text, v in bucket]
        scored.sort(key=lambda row: row[2], reverse=True)
        return scored[:k]

    def delete_chunks(self, index: str, ids: list[str]) -> int:
        bucket = self._indexes.get(index, [])
        before = len(bucket)
        self._indexes[index] = [row for row in bucket if row[0] not in ids]
        return before - len(self._indexes[index])

    def delete_index(self, index: str) -> None:
        self._indexes.pop(index, None)


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0
