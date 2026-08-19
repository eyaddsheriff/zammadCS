"""Embeddings via an OpenAI-compatible API (local bge-m3 through Ollama)."""

import requests

from zammad_agent.config import EmbeddingConfig
from zammad_agent.llm.client import LLMAPIError

# Embedding a whole knowledge base is a bulk operation, and a local model on
# CPU is slow. Generous, because the failure mode of a short timeout here is a
# half-built index.
DEFAULT_TIMEOUT_SECONDS = 300.0

# Requests are batched to avoid one enormous payload, which on a local model
# means a long unresponsive call and high memory use. Small enough to stay
# responsive, large enough to avoid per-request overhead dominating.
DEFAULT_BATCH_SIZE = 32


class EmbeddingError(RuntimeError):
    """The provider returned embeddings we cannot trust."""


class EmbeddingClient:
    """Turns text into vectors."""

    def __init__(
        self,
        config: EmbeddingConfig,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> None:
        self._base_url = config.base_url
        self._model = config.model
        self._timeout = timeout
        self._batch_size = batch_size

        self._session = requests.Session()
        self._session.headers.update(
            {
                "Authorization": f"Bearer {config.api_key}",
                "Content-Type": "application/json",
            }
        )

    def __repr__(self) -> str:
        return f"EmbeddingClient(base_url={self._base_url!r}, model={self._model!r})"

    @property
    def model(self) -> str:
        """Which model produced these vectors.

        Stored alongside an index: vectors from different models are not
        comparable, so an index built with one is meaningless to another.
        """
        return self._model

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed texts, returning one vector per input in the same order."""
        if not texts:
            return []

        vectors: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            vectors.extend(self._embed_batch(texts[start : start + self._batch_size]))

        dimensions = {len(vector) for vector in vectors}
        if len(dimensions) > 1:
            # Mixed dimensions mean the provider silently switched models
            # mid-run. Storing those together produces an index that fails
            # only later, at query time, with no obvious cause.
            raise EmbeddingError(f"inconsistent embedding dimensions: {sorted(dimensions)}")
        return vectors

    def embed_one(self, text: str) -> list[float]:
        """Embed a single string, typically a search query."""
        return self.embed([text])[0]

    def _embed_batch(self, batch: list[str]) -> list[list[float]]:
        response = self._session.post(
            f"{self._base_url}/embeddings",
            json={"model": self._model, "input": batch},
            timeout=self._timeout,
        )
        if not response.ok:
            raise LLMAPIError(response.status_code, response.text[:200])

        items = response.json().get("data") or []
        if len(items) != len(batch):
            # A count mismatch would silently shift every subsequent vector
            # against the wrong chunk, degrading retrieval with nothing to
            # point at as the cause.
            raise EmbeddingError(f"asked for {len(batch)} embeddings, got {len(items)}")

        # Order by the response's own index rather than trusting array order.
        # The API documents this field precisely because order is not
        # guaranteed, and a silent reordering pairs every chunk with someone
        # else's vector.
        try:
            ordered = sorted(items, key=lambda item: item["index"])
        except KeyError as exc:
            raise EmbeddingError("embedding response is missing index fields") from exc

        return [item["embedding"] for item in ordered]

    def close(self) -> None:
        self._session.close()

    def __enter__(self) -> "EmbeddingClient":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
