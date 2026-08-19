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


def load_zammad_config() -> ZammadConfig:
    """Read and validate Zammad settings, or raise ConfigError."""
    base_url = os.getenv("ZAMMAD_URL", "").strip().rstrip("/")
    token = os.getenv("ZAMMAD_TOKEN", "").strip()

    missing = [
        name
        for name, value in (("ZAMMAD_URL", base_url), ("ZAMMAD_TOKEN", token))
        if not value
    ]
    if missing:
        # Fail loudly at startup. Otherwise the first request goes out
        # unauthenticated and we spend the afternoon debugging a 401 that has
        # nothing to do with the token being wrong.
        raise ConfigError(
            f"Missing environment variable(s): {', '.join(missing)}. "
            "Copy .env.example to .env and fill them in."
        )

    parsed = urlparse(base_url)
    if parsed.scheme not in ("http", "https"):
        raise ConfigError(f"ZAMMAD_URL needs an http:// or https:// scheme (got {base_url!r})")

    # The token rides on every request, so plain http would leak it to anything
    # on the network path. Local development against a container on loopback is
    # the one case where there is no network path to leak to.
    if parsed.scheme == "http" and parsed.hostname not in _LOCAL_HOSTS:
        raise ConfigError(f"Refusing to send the API token over plain http to {parsed.hostname!r}")

    return ZammadConfig(base_url=base_url, token=token)
