"""Vector store behaviour.

Uses a stub embedder with hand-built vectors so similarity is predictable.
A real model would make these tests slow and their assertions approximate.
"""

import pytest

from zammad_agent.rag.chunk import Chunk
from zammad_agent.rag.store import KnowledgeBase, KnowledgeBaseError


class StubEmbedder:
    """Three-dimensional keyword vectors: password, refund, export."""

    def __init__(self, model="stub-model"):
        self.model = model

    @staticmethod
    def _vector(text):
        lowered = text.lower()
        # Baseline 0.1 keeps every vector non-zero; cosine distance against a
        # zero vector is undefined.
        return [
            1.0 if "password" in lowered else 0.1,
            1.0 if "refund" in lowered else 0.1,
            1.0 if "export" in lowered else 0.1,
        ]

    def embed(self, texts):
        return [self._vector(t) for t in texts]

    def embed_one(self, text):
        return self._vector(text)


def _chunks():
    return [
        Chunk(text="Reset your password here", source="kb.md", heading="Password", ordinal=0),
        Chunk(text="We issue a refund in five days", source="kb.md", heading="Refunds", ordinal=1),
        Chunk(text="Click export to download", source="kb.md", heading="Export", ordinal=2),
    ]


@pytest.fixture
def kb(tmp_path):
    return KnowledgeBase(StubEmbedder(), path=str(tmp_path / "index"))


def test_adds_and_counts(kb):
    assert kb.add(_chunks()) == 3
    assert kb.count() == 3


def test_adding_nothing_is_harmless(kb):
    assert kb.add([]) == 0
    assert kb.count() == 0


def test_search_on_empty_index_returns_nothing(kb):
    # Querying an unbuilt index should be empty, not an exception.
    assert kb.search("anything") == []


def test_retrieves_the_relevant_chunk(kb):
    kb.add(_chunks())
    assert kb.search("password", top_k=1)[0].heading == "Password"
    assert kb.search("refund", top_k=1)[0].heading == "Refunds"


def test_results_are_ordered_best_first(kb):
    kb.add(_chunks())
    scores = [r.score for r in kb.search("password", top_k=3)]
    assert scores == sorted(scores, reverse=True)


def test_metadata_survives_the_round_trip(kb):
    kb.add(_chunks())
    result = kb.search("password", top_k=1)[0]
    assert result.source == "kb.md"
    assert result.heading == "Password"


def test_top_k_larger_than_index_is_clamped(kb):
    kb.add(_chunks())
    assert len(kb.search("password", top_k=50)) == 3


def test_reimporting_updates_in_place(kb):
    # Editing a document and re-importing must not leave stale copies behind.
    kb.add(_chunks())
    kb.add(_chunks())
    assert kb.count() == 3


def test_index_rejects_a_different_embedding_model(tmp_path):
    # Vectors from different models are not comparable. Without this guard the
    # index would answer queries confidently and wrongly.
    path = str(tmp_path / "index")
    KnowledgeBase(StubEmbedder("model-a"), path=path).add(_chunks())
    with pytest.raises(KnowledgeBaseError, match="model-a"):
        KnowledgeBase(StubEmbedder("model-b"), path=path)


def test_index_persists_across_instances(tmp_path):
    path = str(tmp_path / "index")
    KnowledgeBase(StubEmbedder(), path=path).add(_chunks())
    assert KnowledgeBase(StubEmbedder(), path=path).count() == 3
