"""Client for any OpenAI-compatible chat completions API (Ollama, DeepSeek)."""

import json
from typing import Any

import requests

from zammad_agent.config import LLMConfig

# A local 7B model on CPU is orders of magnitude slower than the Zammad API,
# and the first call also pays for loading weights into memory. Zammad's 10s
# would time out on almost every request here.
DEFAULT_TIMEOUT_SECONDS = 180.0

# Rough chars-per-token for English prose. Only used to spot a gross mismatch
# between what we sent and what the server says it read, so the imprecision is
# fine — the signal is a shortfall of tens of percent, not a few tokens.
_CHARS_PER_TOKEN = 4

# How far below our estimate the reported prompt size may fall before we treat
# it as dropped input. Generous enough to absorb tokenizer variation, tight
# enough to catch a context window silently halving the prompt.
_TRUNCATION_RATIO = 0.75


class LLMAPIError(RuntimeError):
    """The provider answered with an error status."""

    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(f"LLM API returned {status_code}: {message}")
        self.status_code = status_code


class LLMContextError(RuntimeError):
    """The provider silently discarded part of the prompt.

    Its own defence: Ollama drops whatever exceeds its context window without
    erroring, then answers fluently from the remainder. Left undetected that
    produces a confident answer derived from half a ticket, which is worse
    than a failure because nothing looks wrong.
    """


class LLMResponseError(RuntimeError):
    """The model replied, but not with something usable."""


class LLMClient:
    """Chat completions against whichever provider LLMConfig points at."""

    def __init__(
        self,
        config: LLMConfig,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        self._base_url = config.base_url
        self._model = config.model
        self._timeout = timeout

        self._session = requests.Session()
        self._session.headers.update(
            {
                # OpenAI's scheme, unlike Zammad's "Token token=" above. Ollama
                # ignores the value but requires the header to be present.
                "Authorization": f"Bearer {config.api_key}",
                "Content-Type": "application/json",
            }
        )

    def __repr__(self) -> str:
        return f"LLMClient(base_url={self._base_url!r}, model={self._model!r})"

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        json_mode: bool = False,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> str:
        """Send a conversation, return the model's reply as raw text.

        Temperature defaults to 0: classification should give the same answer
        for the same ticket every run, otherwise a confidence threshold is
        measuring noise. It also makes malformed JSON markedly rarer.
        """
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": temperature,
            "stream": False,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        response = self._session.post(
            f"{self._base_url}/chat/completions", json=payload, timeout=self._timeout
        )
        if not response.ok:
            raise LLMAPIError(response.status_code, response.text[:200])

        data = response.json()
        self._reject_truncated_prompt(messages, data)

        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as exc:
            raise LLMResponseError(f"unexpected response shape: {str(data)[:200]}") from exc

    def chat_json(self, messages: list[dict[str, str]], **kwargs: Any) -> dict[str, Any]:
        """Send a conversation in JSON mode and return the parsed object.

        JSON mode constrains the model but does not guarantee compliance —
        smaller models still emit prose or truncate mid-object — so the parse
        failure is surfaced as a typed error the caller can retry on.
        """
        raw = self.chat(messages, json_mode=True, **kwargs)
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise LLMResponseError(f"model did not return valid JSON: {raw[:300]!r}") from exc

        if not isinstance(parsed, dict):
            raise LLMResponseError(f"expected a JSON object, got {type(parsed).__name__}")
        return parsed

    @staticmethod
    def _reject_truncated_prompt(
        messages: list[dict[str, str]], data: dict[str, Any]
    ) -> None:
        """Fail loudly if the provider read less than we sent."""
        reported = (data.get("usage") or {}).get("prompt_tokens")
        if not reported:
            # Not every provider returns usage. Nothing to check against, and
            # refusing to work without it would be worse than the risk.
            return

        sent_chars = sum(len(message.get("content") or "") for message in messages)
        estimated = sent_chars / _CHARS_PER_TOKEN
        if reported < estimated * _TRUNCATION_RATIO:
            raise LLMContextError(
                f"Provider read ~{reported} prompt tokens but we sent ~{int(estimated)} "
                f"({sent_chars} chars). Input was truncated - shorten the thread "
                f"(flatten_thread(max_chars=...)) or raise the provider's context window."
            )

    def close(self) -> None:
        self._session.close()

    def __enter__(self) -> "LLMClient":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
