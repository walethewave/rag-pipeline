"""Reads every setting the app needs from environment variables / .env, once, into one object."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """One typed object holding all config, so no other file reads os.environ directly."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # LLM (ai.py) - Claude called through AWS Bedrock using AWS credentials
    aws_access_key_id: str = ""
    aws_secret_access_key: str = ""
    aws_region: str = "us-east-1"
    bedrock_model_id: str = "us.anthropic.claude-haiku-4-5-20251001-v1:0"

    # Embeddings (ai.py) - AWS Bedrock Titan Text v2 produces 512-dim vectors.
    embedding_model: str = "amazon.titan-embed-text-v2:0"
    embedding_dimension: int = 512

    # Storage (storage.py) - vectors live in AWS S3 Vectors, one index per app `index` name,
    # all inside this one bucket. The document registry is a local JSON file either way.
    s3vectors_bucket: str = "vestai-faq-vectors"
    document_registry_path: str = "./data/documents.json"
    default_index: str = "faq"

    # Chunking (pipeline.py). Overlap is always 10% of the chunk size - see chunk_overlap_tokens.
    max_chunk_tokens: int = 500

    @property
    def chunk_overlap_tokens(self) -> int:
        """10% of max_chunk_tokens, so the overlap always scales with the chunk size instead of
        being a second number that can drift out of sync with it."""
        return self.max_chunk_tokens // 10

    # Retrieval (main.py). score_threshold is calibrated empirically, not guessed: against a real
    # Titan-embedded document, unrelated queries scored up to 0.150 (noise ceiling) while the
    # weakest genuine match - a terse 2-word query like "whats firs" - still scored 0.229 for its
    # correct chunk. 0.18 sits in that gap, biased toward the noise floor rather than the
    # midpoint: a false negative (wrongly saying "I could not find this") is worse than a false
    # positive, since the prompt already tells the LLM to only answer from context that's
    # actually relevant - an extra irrelevant chunk gets ignored, a missed one fails the question.
    score_threshold: float = 0.18
    exact_match_threshold: float = 0.85

    # Upload limits (main.py)
    max_upload_mb: int = 20

    log_level: str = "INFO"


def get_settings() -> Settings:
    """Builds a Settings instance on demand (cheap - just reads env vars), so tests can
    monkeypatch environment variables per-test and get fresh values back."""
    return Settings()
