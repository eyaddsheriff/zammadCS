"""Draft validation and suggestion formatting."""

import pytest

from zammad_agent.pipeline.classify import Classification
from zammad_agent.pipeline.respond import Draft, DraftError, build_messages, validate
from zammad_agent.pipeline.suggestion import format_suggestion_note
from zammad_agent.rag.store import Retrieved


def _context():
    return [Retrieved(text="Refunds take five days.", source="billing.md", heading="Refunds", score=0.8)]


def _payload(**overrides):
    base = {"grounded": True, "missing_information": "", "reply": "We will refund you."}
    return {**base, **overrides}


def test_accepts_a_well_formed_draft():
    draft = validate(_payload(), _context())
    assert draft.text == "We will refund you."
    assert draft.grounded is True
    assert draft.sources == ["billing.md"]


def test_rejects_empty_reply():
    with pytest.raises(DraftError, match="non-empty"):
        validate(_payload(reply="   "), _context())


def test_rejects_missing_reply():
    with pytest.raises(DraftError, match="non-empty"):
        validate({"grounded": True}, _context())


def test_rejects_non_boolean_grounded():
    # "yes" would be truthy, quietly promoting an ungrounded draft.
    with pytest.raises(DraftError, match="true or false"):
        validate(_payload(grounded="yes"), _context())


def test_grounded_claim_is_overridden_when_nothing_was_retrieved():
    # The model cannot have used context that was never supplied. Retrieval is
    # the authority here, not the model's claim about itself.
    assert validate(_payload(grounded=True), []).grounded is False


def test_sources_are_deduplicated():
    context = _context() * 3
    assert validate(_payload(), context).sources == ["billing.md"]


def test_prompt_states_when_context_is_empty():
    # An empty section invites the model to fill the silence from training
    # data, which is the hallucination this guards against.
    messages = build_messages("Customer: help", Classification("other", 1.0, ""), [])
    assert "no knowledge base articles" in messages[1]["content"]


def test_prompt_wraps_ticket_in_delimiters():
    messages = build_messages("Customer: help", Classification("other", 1.0, ""), [])
    assert "<<<TICKET" in messages[1]["content"]


def test_note_warns_before_showing_an_ungrounded_draft():
    # An agent who reads the draft first has already started trusting it by
    # the time they reach a caveat underneath.
    note = format_suggestion_note(
        Classification("billing_question", 0.8, "x"),
        Draft(text="Some draft", grounded=False),
        "qwen2.5:7b",
    )
    assert note.index("WARNING") < note.index("Some draft")


def test_grounded_note_has_no_warning():
    note = format_suggestion_note(
        Classification("billing_question", 0.8, "x"),
        Draft(text="Some draft", grounded=True, sources=["billing.md"]),
        "qwen2.5:7b",
    )
    assert "WARNING" not in note
    assert "billing.md" in note


def test_note_states_it_is_not_sent_to_the_customer():
    # Agents must be able to tell this from a colleague's note at a glance.
    note = format_suggestion_note(
        Classification("other", 1.0, ""), Draft(text="d", grounded=False), "m"
    )
    assert "not sent to the customer" in note


def test_note_names_the_model():
    note = format_suggestion_note(
        Classification("other", 1.0, ""), Draft(text="d", grounded=False), "qwen2.5:7b"
    )
    assert "qwen2.5:7b" in note
