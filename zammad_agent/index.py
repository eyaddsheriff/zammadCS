"""Build the knowledge-base index from Markdown documents.

    python -m zammad_agent.index knowledge_base/
    python -m zammad_agent.index examples/sample_kb --query "I was charged twice"
    python -m zammad_agent.index knowledge_base/ --dry-run

--dry-run chunks without embedding, which is the fast way to check how a
document splits before spending minutes on a local embedding model.
"""

import argparse
import sys

from zammad_agent.config import ConfigError, load_embedding_config
from zammad_agent.llm.embeddings import EmbeddingClient
from zammad_agent.rag.loader import LoaderError, load_directory
from zammad_agent.rag.store import DEFAULT_PATH, DEFAULT_TOP_K, KnowledgeBase


def main() -> int:
    parser = argparse.ArgumentParser(description="Index knowledge-base documents.")
    parser.add_argument("directory", help="directory of Markdown documents")
    parser.add_argument("--path", default=DEFAULT_PATH, help=f"index location (default {DEFAULT_PATH})")
    parser.add_argument("--dry-run", action="store_true", help="chunk only, do not embed")
    parser.add_argument("--query", help="run a test search after indexing")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    args = parser.parse_args()

    try:
        chunks = load_directory(args.directory)
    except LoaderError as exc:
        print(f"error: {exc}")
        return 1

    sources = {chunk.source for chunk in chunks}
    print(f"{len(sources)} document(s) -> {len(chunks)} chunk(s)")
    if not chunks:
        print("nothing to index")
        return 1

    if args.dry_run:
        for chunk in chunks:
            print(f"  [{chunk.id}] {chunk.heading or '(no heading)'} ({len(chunk.text)} chars)")
        return 0

    try:
        config = load_embedding_config()
    except ConfigError as exc:
        print(f"error: {exc}")
        return 1

    print(f"embedding with {config.model} (this is slow on a local model)")
    with EmbeddingClient(config) as embedder:
        kb = KnowledgeBase(embedder, path=args.path)
        added = kb.add(chunks)
        print(f"indexed {added} chunk(s); index now holds {kb.count()}")

        if args.query:
            print(f"\nsearch: {args.query!r}")
            for result in kb.search(args.query, top_k=args.top_k):
                print(f"  {result.score:.3f}  {result.source}  {result.heading}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
