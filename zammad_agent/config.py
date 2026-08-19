"""Runtime configuration, read from environment variables."""

import os
from dataclasses import dataclass
from urllib.parse import urlparse

from dotenv import load_dotenv

# Populates os.environ from a .env file when one exists. Under Docker/K8s later
# there is no .env — secrets are injected as real env vars — and this becomes a
# harmless no-op, so the same code path works in both places.
load_dotenv()

_LOCAL_HOSTS = {"localhost", "127.0.0.1"}


class ConfigError(RuntimeError):
    """Configuration is missing or unusable; the app should not start."""


@dataclass(frozen=True)
class ZammadConfig:
    base_url: str
    token: str

    def __repr__(self) -> str:
        # Config objects surface in tracebacks and debug prints. The default
        # dataclass repr would print the token verbatim into logs, so the
        # secret is masked here rather than relying on nobody printing it.
        return f"ZammadConfig(base_url={self.base_url!r}, token='***')"


def _require(**values: str) -> None:
    """Raise if any named value is empty."""
    missing = [name for name, value in values.items() if not value]
    if missing:
        # Fail loudly at startup. Otherwise the first request goes out
        # unauthenticated and we spend the afternoon debugging a 401 that has
        # nothing to do with the credential being wrong.
        raise ConfigError(
            f"Missing environment variable(s): {', '.join(missing)}. "
            "Copy .env.example to .env and fill them in."
        )


def _check_credentialed_url(var_name: str, url: str) -> None:
    """Reject URLs unsafe to send a credential to."""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ConfigError(f"{var_name} needs an http:// or https:// scheme (got {url!r})")

    # The credential rides on every request, so plain http would expose it to
    # anything on the network path. Loopback is the one case where there is no
    # network path to leak to — which is what makes local Ollama legal here.
    if parsed.scheme == "http" and parsed.hostname not in _LOCAL_HOSTS:
        raise ConfigError(f"Refusing to send a credential over plain http to {parsed.hostname!r}")


def load_zammad_config() -> ZammadConfig:
    """Read and validate Zammad settings, or raise ConfigError."""
    base_url = os.getenv("ZAMMAD_URL", "").strip().rstrip("/")
    token = os.getenv("ZAMMAD_TOKEN", "").strip()

    _require(ZAMMAD_URL=base_url, ZAMMAD_TOKEN=token)
    _check_credentialed_url("ZAMMAD_URL", base_url)
    return ZammadConfig(base_url=base_url, token=token)


@dataclass(frozen=True)
class LLMConfig:
    """Points at any OpenAI-compatible chat completions API.

    Ollama and DeepSeek both speak that shape, so switching providers is a
    change to these three values rather than a change to any code.
    """

    base_url: str
    api_key: str
    model: str

    def __repr__(self) -> str:
        return (
            f"LLMConfig(base_url={self.base_url!r}, "
            f"model={self.model!r}, api_key='***')"
        )


@dataclass(frozen=True)
class EmbeddingConfig:
    """Points at an OpenAI-compatible embeddings API.

    Deliberately separate from LLMConfig rather than derived from it. Chat
    moves to DeepSeek eventually; embeddings cannot follow, because DeepSeek
    offers no embeddings endpoint. Inheriting the base URL would turn that
    switch into a 404 raised far from its cause.
    """

    base_url: str
    api_key: str
    model: str

    def __repr__(self) -> str:
        return (
            f"EmbeddingConfig(base_url={self.base_url!r}, "
            f"model={self.model!r}, api_key='***')"
        )


def load_embedding_config() -> EmbeddingConfig:
    """Read and validate embedding provider settings, or raise ConfigError."""
    base_url = os.getenv("EMBED_BASE_URL", "").strip().rstrip("/")
    api_key = os.getenv("EMBED_API_KEY", "").strip()
    model = os.getenv("EMBED_MODEL", "").strip()

    _require(EMBED_BASE_URL=base_url, EMBED_API_KEY=api_key, EMBED_MODEL=model)
    _check_credentialed_url("EMBED_BASE_URL", base_url)
    return EmbeddingConfig(base_url=base_url, api_key=api_key, model=model)


def load_llm_config() -> LLMConfig:
    """Read and validate LLM provider settings, or raise ConfigError."""
    base_url = os.getenv("LLM_BASE_URL", "").strip().rstrip("/")
    api_key = os.getenv("LLM_API_KEY", "").strip()
    model = os.getenv("LLM_MODEL", "").strip()

    # Ollama ignores the key but the OpenAI protocol requires the header, so
    # locally this is the literal string "ollama". Still required rather than
    # defaulted: once the base URL points at DeepSeek, a silently-defaulted key
    # would surface as a confusing 401 instead of a missing-config error.
    _require(LLM_BASE_URL=base_url, LLM_API_KEY=api_key, LLM_MODEL=model)
    _check_credentialed_url("LLM_BASE_URL", base_url)
    return LLMConfig(base_url=base_url, api_key=api_key, model=model)
