"""Vector storage and retrieval over the knowledge base, backed by Chroma."""

from dataclasses import dataclass

import chromadb

from zammad_agent.llm.embeddings import EmbeddingClient
from zammad_agent.rag.chunk import Chunk

DEFAULT_PATH = ".chroma"
DEFAULT_COLLECTION = "knowledge_base"
DEFAULT_TOP_K = 4


class KnowledgeBaseError(RuntimeError):
    """The index cannot be used as asked."""


@dataclass(frozen=True)
class Retrieved:
    text: str
    source: str
    heading: str
    score: float


class KnowledgeBase:
    """A persistent, on-disk vector index of knowledge-base chunks.

    Chroma rather than Qdrant because Qdrant needs a server somebody has to
    run, and this deployment cannot assume one. Chroma runs embedded against
    a local directory.
    """

    def __init__(
        self,
        embedder: EmbeddingClient,
        path: str = DEFAULT_PATH,
        collection: str = DEFAULT_COLLECTION,
    ) -> None:
        self._embedder = embedder
        self._client = chromadb.PersistentClient(
            path=path,
            # Chroma reports anonymised usage events by default. This service
            # handles customer support data, so nothing leaves the host that
            # was not explicitly asked for.
            settings=chromadb.config.Settings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=collection,
            metadata={
                # Cosine, not Chroma's default L2. Text embeddings encode
                # meaning in direction rather than magnitude, so L2 lets a
                # long document outrank a more relevant short one.
                "hnsw:space": "cosine",
                # Recorded so a later run cannot query this index with a
                # different model. Vectors from different models are not
                # comparable, and mixing them returns confident nonsense
                # rather than an error.
                "embedding_model": embedder.model,
            },
            # Without this Chroma downloads and runs its own embedding model,
            # which would embed queries with a different model than the index
            # was built with.
            embedding_function=None,
        )

        indexed_model = (self._collection.metadata or {}).get("embedding_model")
        if indexed_model and indexed_model != embedder.model:
            raise KnowledgeBaseError(
                f"index at {path!r} was built with {indexed_model!r} but the "
                f"configured model is {embedder.model!r}. Re-index or change "
                f"EMBED_MODEL back."
            )

    def count(self) -> int:
        return self._collection.count()

    def add(self, chunks: list[Chunk]) -> int:
        """Embed and store chunks, replacing any with the same id."""
        if not chunks:
            return 0

        vectors = self._embedder.embed([chunk.text for chunk in chunks])
        # upsert rather than add: re-importing an edited document should
        # update it in place, not raise on duplicate ids or silently
        # accumulate stale copies alongside the new ones.
        self._collection.upsert(
            ids=[chunk.id for chunk in chunks],
            embeddings=vectors,
            documents=[chunk.text for chunk in chunks],
            metadatas=[
                {"source": chunk.source, "heading": chunk.heading} for chunk in chunks
            ],
        )
        return len(chunks)

    def search(self, query: str, top_k: int = DEFAULT_TOP_K) -> list[Retrieved]:
        """Return the chunks most semantically similar to the query."""
        if self.count() == 0:
            return []

        result = self._collection.query(
            query_embeddings=[self._embedder.embed_one(query)],
            # Asking for more than the collection holds is an error in some
            # backends; clamping keeps a small knowledge base usable.
            n_results=min(top_k, self.count()),
            include=["documents", "metadatas", "distances"],
        )

        retrieved = []
        for document, metadata, distance in zip(
            result["documents"][0], result["metadatas"][0], result["distances"][0]
        ):
            retrieved.append(
                Retrieved(
                    text=document,
                    source=(metadata or {}).get("source", ""),
                    heading=(metadata or {}).get("heading", ""),
                    # Cosine distance to similarity, so larger is better and
                    # the number reads the way a relevance score should.
                    score=1.0 - distance,
                )
            )
        return retrieved
