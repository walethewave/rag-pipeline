"""Everything that calls external AI models through AWS Bedrock:
- Embedder: generates vector embeddings using AWS Bedrock Titan (e.g. amazon.titan-embed-text-v2:0)
- BedrockLLM: generates answers using Claude on Bedrock
- build_prompt: formats the grounded prompt template
"""

import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Protocol

import boto3
import yaml

from src.config import get_settings
from src.pipeline import Chunk

logger = logging.getLogger(__name__)

_PROMPT_PATH = Path(__file__).parent / "prompts" / "faq_answer.yaml"


# ==================================================================== embedding (AWS Bedrock Titan)


class Embedder:
    """Generates vector embeddings using AWS Bedrock Titan."""

    def __init__(self, model_id: str, region: str, access_key: str, secret_key: str, dimension: int = 512):
        """Builds one boto3 bedrock-runtime client, reused for every embed() call - creating a
        new client per request would be wasteful. Access/secret key are optional: if blank,
        boto3 falls back to its default credential chain (e.g. an IAM role on Lambda)."""
        self.model_id = model_id
        self.dimension = dimension
        kwargs = {"region_name": region}
        if access_key and secret_key:
            kwargs["aws_access_key_id"] = access_key
            kwargs["aws_secret_access_key"] = secret_key
        self.bedrock = boto3.client("bedrock-runtime", **kwargs)
        logger.info("Initialized Bedrock Titan Embedder: model=%s (dim=%d)", model_id, dimension)

    def _generate_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings using AWS Bedrock Titan."""
        if not texts:
            return []
        embeddings = []
        for text in texts:
            try:
                # Format request body for Bedrock Titan (normalized vector with configured dimension)
                request_body = json.dumps({"inputText": text, "dimensions": self.dimension, "normalize": True})
                response = self.bedrock.invoke_model(
                    modelId=self.model_id,
                    body=request_body,
                    contentType="application/json",
                    accept="application/json",
                )
                result = json.loads(response["body"].read())
                embeddings.append(result["embedding"])
            except Exception as e:
                logger.warning("Failed to generate embedding for text chunk: %s", str(e))
                # Fallback to zero-vector to avoid dimension mismatch
                embeddings.append([0.0] * self.dimension)
        return embeddings

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Batch embedding function called by the ingestion pipeline."""
        return self._generate_embeddings(texts)

    def embed_one(self, text: str) -> list[float]:
        """Convenience wrapper for embedding a single query string."""
        return self._generate_embeddings([text])[0]


@lru_cache
def get_embedder() -> Embedder:
    """Returns the one shared Embedder instance, configured from environment settings."""
    settings = get_settings()
    return Embedder(
        model_id=settings.embedding_model,
        region=settings.aws_region,
        access_key=settings.aws_access_key_id,
        secret_key=settings.aws_secret_access_key,
        dimension=settings.embedding_dimension,
    )


# ==================================================================== the LLM (AWS Bedrock Claude)


class LLM(Protocol):
    """Anything with a .generate(prompt) -> str method can be used by main.py's /chat endpoint."""

    def generate(self, prompt: str) -> str:
        """Sends a fully-built prompt to the model and returns its text reply."""
        ...


class BedrockLLM:
    """Calls Claude through AWS Bedrock using AWS credentials."""

    def __init__(self, model_id: str, region: str, access_key: str, secret_key: str, max_tokens: int = 1024):
        """Builds one boto3 bedrock-runtime client, reused for every generate() call. Access/secret
        key are optional: if blank, boto3 falls back to its default credential chain."""
        self._model_id = model_id
        self._max_tokens = max_tokens
        kwargs = {"region_name": region}
        if access_key and secret_key:
            kwargs["aws_access_key_id"] = access_key
            kwargs["aws_secret_access_key"] = secret_key
        self._client = boto3.client("bedrock-runtime", **kwargs)

    def generate(self, prompt: str) -> str:
        """Sends prompt to Claude via Bedrock invoke_model and returns response text."""
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": self._max_tokens,
            "temperature": 0,
            "messages": [{"role": "user", "content": prompt}],
        }
        logger.info("Calling Bedrock model %s (prompt length=%d chars)", self._model_id, len(prompt))
        response = self._client.invoke_model(
            modelId=self._model_id, body=json.dumps(body),
            contentType="application/json", accept="application/json",
        )
        payload = json.loads(response["body"].read())
        return payload["content"][0]["text"].strip()


@lru_cache
def get_llm() -> LLM:
    """Returns the one shared LLM (Bedrock/Claude) instance."""
    settings = get_settings()
    return BedrockLLM(
        model_id=settings.bedrock_model_id,
        region=settings.aws_region,
        access_key=settings.aws_access_key_id,
        secret_key=settings.aws_secret_access_key,
    )


# ==================================================================== prompt builder


@lru_cache
def _load_template() -> str:
    """Reads the YAML prompt template once and caches it."""
    with open(_PROMPT_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)["template"]


def build_prompt(question: str, chunks: list[Chunk], sources: list[str]) -> str:
    """Fills the main prompt template with retrieved chunks or indicates that no chunks were found."""
    if chunks:
        context_blocks = [
            f"[{i}] (source: {source})\n{chunk.text}" for i, (chunk, source) in enumerate(zip(chunks, sources), start=1)
        ]
        context_str = "\n\n".join(context_blocks)
    else:
        context_str = "(No matching document excerpts found)"
    return _load_template().format(context=context_str, question=question)
