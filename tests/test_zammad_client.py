"""The write path and its guard.

create_article is the only code in this project that can change someone
else's Zammad instance, and a customer-visible article cannot be recalled.
"""

import pytest

from zammad_agent.config import ZammadConfig
from zammad_agent.zammad.client import PublicReplyBlocked, ZammadAPIError, ZammadClient


class FakeResponse:
    def __init__(self, payload=None, ok=True, status_code=200):
        self._payload = payload if payload is not None else {"id": 99}
        self.ok = ok
        self.status_code = status_code
        self.text = str(self._payload)

    def json(self):
        return self._payload


class FakeSession:
    def __init__(self, response=None):
        self.response = response or FakeResponse()
        self.headers = {}
        self.posted = []

    def post(self, url, json=None, timeout=None):
        self.posted.append((url, json))
        return self.response

    def close(self):
        pass


def _client(allow_public_replies=False, response=None):
    config = ZammadConfig(base_url="https://help.example.com", token="t")
    client = ZammadClient(config, allow_public_replies=allow_public_replies)
    client._session = FakeSession(response)
    return client


def test_posts_an_internal_note():
    client = _client()
    assert client.create_article(7, "hello", internal=True) == {"id": 99}
    url, payload = client._session.posted[0]
    assert url.endswith("/api/v1/ticket_articles")
    assert payload["ticket_id"] == 7
    assert payload["internal"] is True


def test_internal_notes_are_posted_as_notes_not_emails():
    # type "email" would make Zammad dispatch a message to the customer.
    client = _client()
    client.create_article(7, "hello", internal=True)
    assert client._session.posted[0][1]["type"] == "note"


def test_public_reply_is_blocked_by_default():
    client = _client()
    with pytest.raises(PublicReplyBlocked):
        client.create_article(7, "hello", internal=False)


def test_blocked_public_reply_sends_no_request():
    # The guard must stop the request, not merely report it afterwards.
    client = _client()
    with pytest.raises(PublicReplyBlocked):
        client.create_article(7, "hello", internal=False)
    assert client._session.posted == []


def test_public_reply_allowed_when_explicitly_enabled():
    client = _client(allow_public_replies=True)
    client.create_article(7, "hello", internal=False)
    assert client._session.posted[0][1]["internal"] is False


def test_internal_notes_are_never_blocked():
    # An agent-only note is safe regardless of how the client was built.
    client = _client(allow_public_replies=False)
    client.create_article(7, "hello", internal=True)
    assert len(client._session.posted) == 1


def test_api_errors_are_raised():
    client = _client(response=FakeResponse({"error": "nope"}, ok=False, status_code=403))
    with pytest.raises(ZammadAPIError, match="403"):
        client.create_article(7, "hello", internal=True)


def test_subject_is_omitted_when_absent():
    client = _client()
    client.create_article(7, "hello", internal=True)
    assert "subject" not in client._session.posted[0][1]
