"""Draft an agent-facing suggestion for a ticket.

    python -m zammad_agent.suggest --ticket 1
    python -m zammad_agent.suggest --ticket 1 --post

Prints the suggestion by default and writes nothing. --post publishes it as
an internal Zammad note: visible to agents in the ticket, never shown to the
customer. There is no option here that sends anything to a customer.
"""

import argparse
import sys

from zammad_agent.config import (
    ConfigError,
    load_embedding_config,
    load_llm_config,
    load_zammad_config,
)
from zammad_agent.llm.client import LLMClient
from zammad_agent.llm.embeddings import EmbeddingClient
from zammad_agent.pipeline.classify import THREAD_CHAR_BUDGET, classify_consistent
from zammad_agent.pipeline.respond import draft_reply
from zammad_agent.pipeline.suggestion import format_suggestion_note
from zammad_agent.pipeline.thread import flatten_thread
from zammad_agent.rag.store import DEFAULT_PATH, DEFAULT_TOP_K, KnowledgeBase
from zammad_agent.zammad.client import ZammadClient


def main() -> int:
    parser = argparse.ArgumentParser(description="Draft a suggestion for a ticket.")
    parser.add_argument("--ticket", type=int, required=True, help="ticket id")
    parser.add_argument("--post", action="store_true", help="post as an internal note")
    parser.add_argument("--kb-path", default=DEFAULT_PATH, help="knowledge base index path")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--samples", type=int, default=5, help="classification samples")
    args = parser.parse_args()

    try:
        zammad_config = load_zammad_config()
        llm_config = load_llm_config()
        embed_config = load_embedding_config()
    except ConfigError as exc:
        print(f"error: {exc}")
        return 1

    # allow_public_replies is not exposed as a flag. This tool posts internal
    # notes; sending to a customer should require editing code, not a typo.
    with ZammadClient(zammad_config) as zammad, LLMClient(llm_config) as llm:
        articles = zammad.get_ticket_articles(args.ticket)
        if not articles:
            print(f"error: ticket {args.ticket} has no articles")
            return 1

        thread = flatten_thread(articles, max_chars=THREAD_CHAR_BUDGET)
        classification = classify_consistent(llm, thread, samples=args.samples)

        with EmbeddingClient(embed_config) as embedder:
            kb = KnowledgeBase(embedder, path=args.kb_path)
            context = kb.search(thread, top_k=args.top_k) if kb.count() else []
            if not context:
                print("note: knowledge base is empty, drafting without grounding\n")

            draft = draft_reply(llm, thread, classification, context)

        note = format_suggestion_note(classification, draft, llm_config.model)
        print(note)

        if args.post:
            created = zammad.create_article(
                args.ticket, note, internal=True, subject="AI suggestion"
            )
            print(f"\nposted as internal article {created.get('id')} on ticket {args.ticket}")
        else:
            print("\n(dry run - nothing was written. Use --post to publish.)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
