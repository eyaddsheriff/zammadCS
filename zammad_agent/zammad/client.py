"""HTTP client for the Zammad REST API."""

from typing import Any

import requests

from zammad_agent.config import ZammadConfig

# requests applies no timeout unless told to. Without one, a call to an
# unresponsive host blocks forever and takes the calling worker with it.
DEFAULT_TIMEOUT_SECONDS = 10.0


class ZammadAPIError(RuntimeError):
    """Zammad answered, but with an error status.

    Carries the status code because callers care about the difference: 401 is
    a bad token, 403 a token missing ticket.agent, 429 a rate limit worth
    retrying. Wrapping requests.HTTPError here keeps the transport library
    from leaking into the pipeline layers that sit above this client.
    """

    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(f"Zammad API returned {status_code}: {message}")
        self.status_code = status_code


class PublicReplyBlocked(RuntimeError):
    """A customer-visible write was attempted without being enabled.

    The guard is deliberately in the client rather than in each caller: a
    public reply reaches a real customer and cannot be recalled, so the
    ability to send one is opt-in at construction rather than something any
    call site can decide for itself.
    """


class ZammadClient:
    """Access to a single Zammad instance.

    Takes an already-validated config rather than reading the environment
    itself, so a test can point one at a fake instance without touching
    os.environ or needing a real token.
    """

    def __init__(
        self,
        config: ZammadConfig,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        allow_public_replies: bool = False,
    ) -> None:
        self._api_root = f"{config.base_url}/api/v1"
        self._timeout = timeout
        # Defaults to False so the dangerous capability has to be asked for by
        # name. Internal notes are agent-only and always permitted.
        self._allow_public_replies = allow_public_replies

        # A Session keeps the TCP/TLS connection alive between calls, so the
        # handshake cost is paid once rather than per ticket fetch. It also
        # sends these headers on every request it makes, which puts the token
        # in exactly one place — no endpoint method can forget to authenticate,
        # and none of them need to know the token exists.
        self._session = requests.Session()
        self._session.headers.update(
            {
                # Zammad's own scheme, not OAuth. The literal word "Token",
                # then "token=<value>". Sending "Bearer <value>" gets a 401.
                "Authorization": f"Token token={config.token}",
                "Accept": "application/json",
            }
        )

    def __repr__(self) -> str:
        # This object carries the token in its session headers, so its repr
        # must never render them. Same reasoning as ZammadConfig.
        return f"ZammadClient(api_root={self._api_root!r})"

    def _get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        """GET one API path. Every endpoint method goes through here."""
        response = self._session.get(
            f"{self._api_root}{path}", params=params, timeout=self._timeout
        )
        if not response.ok:
            # Include Zammad's own error text, truncated. Without it a 401 from
            # a revoked token and a 403 from insufficient permissions are
            # indistinguishable at the call site.
            raise ZammadAPIError(response.status_code, response.text[:200])
        return response.json()

    def list_tickets(self, page: int = 1, per_page: int = 25) -> list[dict[str, Any]]:
        """Return one page of tickets as raw Zammad dicts.

        Pagination is explicit because Zammad applies its own default page size
        to an unpaginated request. That silently truncates the result, which
        reads as "this instance only has a handful of tickets".

        Note these are ticket *records* — id, title, state, customer. The
        customer's actual message text lives in articles, fetched separately.
        """
        return self._get("/tickets", params={"page": page, "per_page": per_page})

    def get_ticket_articles(self, ticket_id: int) -> list[dict[str, Any]]:
        """Return every article (message) on one ticket, oldest first.

        This is where the customer's actual words live — the ticket record
        itself carries only metadata and a message count.

        Deliberately unfiltered. Which articles matter varies by caller:
        intent classification wants the customer's turns, a summariser wants
        the whole exchange. Baking one caller's view in here would make the
        others work around it.
        """
        return self._get(f"/ticket_articles/by_ticket/{ticket_id}")

    def create_article(
        self,
        ticket_id: int,
        body: str,
        *,
        internal: bool,
        subject: str | None = None,
        content_type: str = "text/plain",
    ) -> dict[str, Any]:
        """Post an article to a ticket.

        `internal` has no default on purpose. It is the difference between a
        note only agents see and an email a customer receives, and a caller
        should never make that choice by omission.
        """
        if not internal and not self._allow_public_replies:
            raise PublicReplyBlocked(
                f"refusing to post a customer-visible article to ticket {ticket_id}: "
                "construct ZammadClient(allow_public_replies=True) to permit it"
            )

        payload: dict[str, Any] = {
            "ticket_id": ticket_id,
            "body": body,
            "internal": internal,
            # "note" rather than "email": a note is recorded on the ticket
            # without Zammad dispatching anything to the customer.
            "type": "note",
            "content_type": content_type,
        }
        if subject:
            payload["subject"] = subject

        response = self._session.post(
            f"{self._api_root}/ticket_articles", json=payload, timeout=self._timeout
        )
        if not response.ok:
            raise ZammadAPIError(response.status_code, response.text[:200])
        return response.json()

    def close(self) -> None:
        self._session.close()

    def __enter__(self) -> "ZammadClient":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
