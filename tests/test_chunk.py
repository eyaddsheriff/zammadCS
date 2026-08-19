"""Markdown chunking.

The heading-path tests matter most: a chunk without its heading loses the
context that makes it retrievable at all.
"""

import pytest

from zammad_agent.rag.chunk import split_markdown


def test_heading_path_is_prefixed_to_text():
    # "Click Save" is unretrievable alone; with its heading it is findable.
    chunks = split_markdown("# Guide\n\n## Exporting\n\nClick Save.", "g.md")
    exporting = [c for c in chunks if c.heading.endswith("Exporting")][0]
    assert exporting.text == "Guide > Exporting: Click Save."


def test_headings_nest():
    chunks = split_markdown("# A\n\ntop\n\n## B\n\nmid\n\n### C\n\ndeep", "g.md")
    assert [c.heading for c in chunks] == ["A", "A > B", "A > B > C"]


def test_shallower_heading_pops_the_stack():
    # An h2 after an h3 replaces it rather than nesting beneath it.
    chunks = split_markdown("# A\n\nx\n\n## B\n\ny\n\n### C\n\nz\n\n## D\n\nw", "g.md")
    assert chunks[-1].heading == "A > D"


def test_content_without_heading_is_kept():
    chunks = split_markdown("Just a bare paragraph.", "g.md")
    assert chunks[0].text == "Just a bare paragraph."
    assert chunks[0].heading == ""


def test_ids_are_stable_across_runs():
    # Re-importing an edited document must update chunks in place rather than
    # accumulate stale duplicates, which depends on ids being reproducible.
    doc = "# A\n\none\n\n## B\n\ntwo"
    assert [c.id for c in split_markdown(doc, "g.md")] == [
        c.id for c in split_markdown(doc, "g.md")
    ]
    assert split_markdown(doc, "g.md")[0].id == "g.md#0"


def test_long_sections_are_split():
    body = "\n\n".join(f"Paragraph number {i} with some filler text." * 3 for i in range(20))
    chunks = split_markdown(f"# A\n\n{body}", "g.md", max_chars=300, overlap=50)
    assert len(chunks) > 1
    assert all(len(c.text) <= 400 for c in chunks)  # heading prefix adds a little


def test_oversized_paragraph_is_hard_split():
    # A pasted log or wide table has no paragraph break to split on.
    chunks = split_markdown("# A\n\n" + "x" * 2000, "g.md", max_chars=300, overlap=50)
    assert len(chunks) > 1


def test_overlap_repeats_text_between_chunks():
    # The overlap is what stops a sentence spanning a boundary from becoming
    # unretrievable in both neighbours.
    body = "\n\n".join(f"Sentence {i} carries meaning across the boundary." for i in range(12))
    chunks = split_markdown(f"# A\n\n{body}", "g.md", max_chars=200, overlap=60)
    joined = sum(len(c.text) for c in chunks)
    assert joined > len(body)  # duplication implies overlap happened


def test_overlap_must_be_smaller_than_max_chars():
    # Otherwise each chunk re-consumes its own tail and never advances.
    with pytest.raises(ValueError, match="smaller"):
        split_markdown("# A\n\nx", "g.md", max_chars=100, overlap=100)


def test_empty_document_yields_nothing():
    assert split_markdown("", "g.md") == []
    assert split_markdown("\n\n   \n", "g.md") == []
