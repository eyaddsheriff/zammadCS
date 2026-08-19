"""Config loading, validation, and secret masking.

These tests set the environment explicitly via monkeypatch rather than
relying on whatever .env happens to hold, so they behave the same on a
machine with no Zammad credentials at all.
"""

import pytest

from zammad_agent.config import (
    ConfigError,
    LLMConfig,
    ZammadConfig,
    load_llm_config,
    load_zammad_config,
)


def _set_zammad(monkeypatch, url="https://help.example.com", token="secret-token"):
    monkeypatch.setenv("ZAMMAD_URL", url)
    monkeypatch.setenv("ZAMMAD_TOKEN", token)


def test_loads_valid_zammad_config(monkeypatch):
    _set_zammad(monkeypatch)
    config = load_zammad_config()
    assert config.base_url == "https://help.example.com"
    assert config.token == "secret-token"


def test_strips_trailing_slash(monkeypatch):
    # Paths are built as base_url + "/api/v1", so a trailing slash here would
    # produce a double slash in every request.
    _set_zammad(monkeypatch, url="https://help.example.com/")
    assert load_zammad_config().base_url == "https://help.example.com"


@pytest.mark.parametrize("missing", ["ZAMMAD_URL", "ZAMMAD_TOKEN"])
def test_missing_variable_names_itself(monkeypatch, missing):
    _set_zammad(monkeypatch)
    monkeypatch.delenv(missing)
    with pytest.raises(ConfigError, match=missing):
        load_zammad_config()


def test_rejects_plain_http_to_remote_host(monkeypatch):
    # The token is sent on every request; plain http would expose it in transit.
    _set_zammad(monkeypatch, url="http://help.example.com")
    with pytest.raises(ConfigError, match="plain http"):
        load_zammad_config()


@pytest.mark.parametrize("host", ["localhost", "127.0.0.1"])
def test_allows_plain_http_on_loopback(monkeypatch, host):
    # This exemption is what lets local Ollama work at all.
    _set_zammad(monkeypatch, url=f"http://{host}:11434")
    assert load_zammad_config().base_url == f"http://{host}:11434"


def test_rejects_url_without_scheme(monkeypatch):
    _set_zammad(monkeypatch, url="help.example.com")
    with pytest.raises(ConfigError, match="scheme"):
        load_zammad_config()


def test_zammad_repr_masks_token():
    # Config objects appear in tracebacks and log lines. The whole point of the
    # custom repr is that the token never reaches them.
    config = ZammadConfig(base_url="https://help.example.com", token="super-secret")
    assert "super-secret" not in repr(config)
    assert "***" in repr(config)


def test_llm_repr_masks_key():
    config = LLMConfig(base_url="https://api.example.com/v1", api_key="sk-secret", model="m")
    assert "sk-secret" not in repr(config)
    assert "***" in repr(config)
    # The model is not a secret and is worth seeing in logs.
    assert "m" in repr(config)


def test_llm_config_requires_all_three(monkeypatch):
    monkeypatch.setenv("LLM_BASE_URL", "http://localhost:11434/v1")
    monkeypatch.setenv("LLM_MODEL", "qwen2.5:7b")
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    # Ollama ignores the key, but defaulting it would turn a missing DeepSeek
    # key into a confusing 401 instead of a config error.
    with pytest.raises(ConfigError, match="LLM_API_KEY"):
        load_llm_config()
