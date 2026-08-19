"""The silent-truncation guard.

This exists because of a measured failure: Ollama accepted a 7883-token
prompt, kept 4096 of it, and answered fluently and wrongly from the
remainder with no error anywhere. These tests pin the detection.
"""

import pytest

from zammad_agent.llm.client import LLMClient, LLMContextError


def _messages(chars):
    return [{"role": "user", "content": "x" * chars}]


def _usage(prompt_tokens):
    return {"usage": {"prompt_tokens": prompt_tokens}}


def test_accepts_a_prompt_read_in_full():
    # 4000 chars is roughly 1000 tokens by the chars/4 estimate.
    LLMClient._reject_truncated_prompt(_messages(4000), _usage(1000))


def test_tolerates_tokenizer_variation():
    # The estimate is deliberately crude, so a modest shortfall must not fire.
    LLMClient._reject_truncated_prompt(_messages(4000), _usage(850))


def test_detects_a_halved_prompt():
    # The real failure: ~7900 tokens sent, 4096 read.
    with pytest.raises(LLMContextError, match="truncated"):
        LLMClient._reject_truncated_prompt(_messages(31600), _usage(4096))


def test_error_names_both_numbers():
    # The message has to be actionable on its own, since this surfaces at
    # runtime rather than in a debugger.
    with pytest.raises(LLMContextError, match="4096"):
        LLMClient._reject_truncated_prompt(_messages(31600), _usage(4096))


def test_missing_usage_is_not_an_error():
    # Not every provider reports usage. Refusing to work without it would be
    # worse than the risk it protects against.
    LLMClient._reject_truncated_prompt(_messages(31600), {})


def test_counts_every_message():
    # System and user content both consume the window; only summing one would
    # underestimate what we sent and miss real truncation.
    messages = [{"role": "system", "content": "y" * 16000}, {"role": "user", "content": "x" * 16000}]
    with pytest.raises(LLMContextError):
        LLMClient._reject_truncated_prompt(messages, _usage(4096))
