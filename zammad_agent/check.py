"""End-to-end health check across every layer built so far.

    python -m zammad_agent.check
    python -m zammad_agent.check --ticket 2 --show-text

Exits non-zero if any stage fails, so it doubles as a smoke test later.
Ticket text is withheld unless --show-text is passed: this gets run in front
of other people, and the content is real customer correspondence.
"""

import argparse
import sys

from zammad_agent.config import ConfigError, load_llm_config, load_zammad_config
from zammad_agent.llm.client import LLMClient
from zammad_agent.pipeline.classify import (
    THREAD_CHAR_BUDGET,
    classify_thread,
)
from zammad_agent.pipeline.thread import flatten_thread
from zammad_agent.zammad.client import ZammadClient

_PASS = "[ OK ]"
_FAIL = "[FAIL]"


def _report(stage: str, detail: str, ok: bool = True) -> bool:
    print(f"{_PASS if ok else _FAIL} {stage:22} {detail}")
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticket", type=int, help="classify this ticket id instead of the first")
    parser.add_argument("--show-text", action="store_true", help="print the flattened thread")
    args = parser.parse_args()

    try:
        zammad_config = load_zammad_config()
        llm_config = load_llm_config()
    except ConfigError as exc:
        _report("config", str(exc), ok=False)
        return 1

    _report("config", f"{zammad_config} / {llm_config}")

    try:
        with ZammadClient(zammad_config) as zammad, LLMClient(llm_config) as llm:
            tickets = zammad.list_tickets(per_page=10)
            _report("zammad: list_tickets", f"{len(tickets)} ticket(s)")
            if not tickets:
                return _report("zammad", "no tickets to classify", ok=False) or 1

            ticket = next((t for t in tickets if t["id"] == args.ticket), None) if args.ticket else tickets[0]
            if ticket is None:
                return _report("zammad", f"ticket {args.ticket} not in the first 10", ok=False) or 1

            articles = zammad.get_ticket_articles(ticket["id"])
            _report("zammad: articles", f"ticket {ticket['id']}: {len(articles)} article(s)")

            thread = flatten_thread(articles, max_chars=THREAD_CHAR_BUDGET)
            raw_chars = sum(len(a.get("body") or "") for a in articles)
            _report("pipeline: flatten", f"{raw_chars} raw chars -> {len(thread)} clean")
            if args.show_text:
                print(f"\n--- thread {ticket['id']} ---\n{thread}\n--- end ---\n")

            result = classify_thread(llm, thread)
            _report(
                "pipeline: classify",
                f"intent={result.intent} confidence={result.confidence} ({llm_config.model})",
            )
            if result.reason:
                print(f"       reason: {result.reason}")
    except Exception as exc:
        # Deliberately broad: this is a diagnostic tool, and the useful output
        # is which stage broke and what it said, not a traceback.
        _report(type(exc).__name__, str(exc)[:300], ok=False)
        return 1

    print("\nAll stages passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
