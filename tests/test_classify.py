"""Classification validation and self-consistency voting.

The voting tests use a stub LLM rather than a real one: the point is to pin
the arithmetic of the confidence score, which a real model would make
non-deterministic and slow.
"""

import pytest

from zammad_agent.llm.client import LLMResponseError
from zammad_agent.pipeline.classify import (
    INTENTS,
    ClassificationError,
    build_messages,
    classify_consistent,
    validate,
)


class StubLLM:
    """Returns canned payloads in order; an Exception instance is raised."""

    def __init__(self, replies):
        self._replies = list(replies)

    def chat_json(self, messages, **kwargs):
        reply = self._replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


def _payload(intent="login_issue", confidence=0.9, reason="because"):
    return {"intent": intent, "confidence": confidence, "reason": reason}


def test_accepts_a_well_formed_reply():
    result = validate(_payload())
    assert result.intent == "login_issue"
    assert result.confidence == 0.9


def test_accepts_numeric_string_confidence():
    # "0.9" instead of 0.9 is unambiguous, so accepting it is safe.
    assert validate(_payload(confidence="0.9")).confidence == 0.9


def test_rejects_invented_intent():
    # The model regularly proposes sensible-sounding categories that nothing
    # downstream knows how to route.
    with pytest.raises(ClassificationError, match="not one of"):
        validate(_payload(intent="password_reset"))


def test_rejects_word_confidence():
    # Guessing a number for "high" would invent precision the model never
    # expressed, and the escalation gate rests entirely on this value.
    with pytest.raises(ClassificationError, match="must be a number"):
        validate(_payload(confidence="high"))


def test_rejects_percentage_confidence():
    with pytest.raises(ClassificationError, match="between 0 and 1"):
        validate(_payload(confidence=90))


def test_rejects_boolean_confidence():
    # bool subclasses int, so a naive numeric check turns True into maximum
    # confidence from a model that expressed none.
    with pytest.raises(ClassificationError, match="must be a number"):
        validate(_payload(confidence=True))


def test_missing_reason_is_tolerated():
    # A missing justification costs debuggability, not correctness.
    assert validate({"intent": "other", "confidence": 0.5}).reason == ""


def test_prompt_lists_every_intent():
    # Prompt and validator read from one dict precisely so they cannot drift.
    system = build_messages("Customer: hi")[0]["content"]
    for name in INTENTS:
        assert name in system


def test_prompt_wraps_untrusted_text_in_delimiters():
    # Defence in depth against injection, not a fix for it.
    user = build_messages("Customer: hi")[1]["content"]
    assert "<<<TICKET" in user and "TICKET>>>" in user


def test_unanimous_vote_scores_one():
    llm = StubLLM([_payload()] * 5)
    result = classify_consistent(llm, "thread", samples=5)
    assert result.intent == "login_issue"
    assert result.confidence == 1.0


def test_split_vote_lowers_confidence():
    llm = StubLLM([_payload()] * 3 + [_payload(intent="bug_report")] * 2)
    result = classify_consistent(llm, "thread", samples=5)
    assert result.intent == "login_issue"
    assert result.confidence == pytest.approx(0.6)
    assert result.votes == {"login_issue": 3, "bug_report": 2}


def test_failed_samples_count_against_confidence():
    # Four unparseable replies and one success is a model thrashing on input
    # it cannot handle. Scoring that 1.0 would recreate the bug this replaces.
    llm = StubLLM([_payload()] + [LLMResponseError("bad json")] * 4)
    result = classify_consistent(llm, "thread", samples=5)
    assert result.confidence == pytest.approx(0.2)


def test_all_samples_failing_raises():
    llm = StubLLM([LLMResponseError("bad json")] * 3)
    with pytest.raises(ClassificationError, match="no valid classification"):
        classify_consistent(llm, "thread", samples=3)


def test_reason_comes_from_a_winning_sample():
    llm = StubLLM(
        [_payload(intent="bug_report", reason="loser"), _payload(reason="winner"), _payload(reason="also winner")]
    )
    assert classify_consistent(llm, "thread", samples=3).reason in {"winner", "also winner"}
