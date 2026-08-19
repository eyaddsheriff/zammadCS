"""Read knowledge-base documents from disk and chunk them."""

from pathlib import Path

from zammad_agent.rag.chunk import DEFAULT_MAX_CHARS, DEFAULT_OVERLAP, Chunk, split_markdown

DEFAULT_PATTERN = "**/*.md"


class LoaderError(RuntimeError):
    """A document could not be read."""


def load_directory(
    directory: str | Path,
    *,
    pattern: str = DEFAULT_PATTERN,
    max_chars: int = DEFAULT_MAX_CHARS,
    overlap: int = DEFAULT_OVERLAP,
) -> list[Chunk]:
    """Chunk every matching document under a directory.

    Chunk sources are paths relative to the directory, so re-indexing from a
    different working directory or a container produces the same ids and
    updates in place rather than duplicating the whole knowledge base.
    """
    root = Path(directory)
    if not root.is_dir():
        raise LoaderError(f"{directory!r} is not a directory")

    chunks: list[Chunk] = []
    # Sorted so ordinals are reproducible; filesystem iteration order is not.
    for path in sorted(root.glob(pattern)):
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            # Name the file. "invalid start byte" with no path is a miserable
            # thing to debug across a knowledge base of hundreds of documents.
            raise LoaderError(f"{path} is not valid UTF-8: {exc}") from exc

        source = path.relative_to(root).as_posix()
        chunks.extend(
            split_markdown(text, source, max_chars=max_chars, overlap=overlap)
        )
    return chunks
