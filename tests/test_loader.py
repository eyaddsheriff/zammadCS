"""Loading knowledge-base documents from disk."""

import pytest

from zammad_agent.rag.loader import LoaderError, load_directory


def _write(root, name, text="# Title\n\nSome body text."):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_loads_markdown_files(tmp_path):
    _write(tmp_path, "a.md")
    _write(tmp_path, "b.md")
    assert {c.source for c in load_directory(tmp_path)} == {"a.md", "b.md"}


def test_recurses_into_subdirectories(tmp_path):
    _write(tmp_path, "guides/billing/refunds.md")
    # Forward slashes regardless of platform, so ids match between a Windows
    # dev machine and a Linux container.
    assert load_directory(tmp_path)[0].source == "guides/billing/refunds.md"


def test_ignores_non_markdown(tmp_path):
    _write(tmp_path, "keep.md")
    (tmp_path / "skip.txt").write_text("not markdown", encoding="utf-8")
    assert {c.source for c in load_directory(tmp_path)} == {"keep.md"}


def test_sources_are_relative_to_the_directory(tmp_path):
    # Absolute paths would make chunk ids differ between machines, so
    # re-indexing elsewhere would duplicate the knowledge base instead of
    # updating it.
    _write(tmp_path, "a.md")
    assert not load_directory(tmp_path)[0].source.startswith(str(tmp_path))


def test_ordering_is_reproducible(tmp_path):
    for name in ("c.md", "a.md", "b.md"):
        _write(tmp_path, name)
    assert [c.id for c in load_directory(tmp_path)] == [c.id for c in load_directory(tmp_path)]


def test_empty_directory_yields_nothing(tmp_path):
    assert load_directory(tmp_path) == []


def test_missing_directory_is_an_error(tmp_path):
    with pytest.raises(LoaderError, match="not a directory"):
        load_directory(tmp_path / "nope")


def test_non_utf8_file_names_itself(tmp_path):
    # Across hundreds of documents, "invalid start byte" without a path is a
    # miserable thing to debug.
    bad = tmp_path / "broken.md"
    bad.write_bytes(b"\xff\xfe not valid utf-8")
    with pytest.raises(LoaderError, match="broken.md"):
        load_directory(tmp_path)
