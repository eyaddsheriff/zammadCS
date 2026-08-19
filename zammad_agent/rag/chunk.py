"""Split knowledge-base Markdown into retrievable chunks."""

import re
from dataclasses import dataclass

# Roughly 250 tokens. Small enough that several chunks fit a prompt alongside
# the ticket, large enough to hold a complete instruction rather than half of
# one. Sized in characters to avoid depending on a tokenizer.
DEFAULT_MAX_CHARS = 1000

# Carried from the end of one chunk into the start of the next, so a sentence
# spanning a boundary survives intact in at least one of them. Otherwise the
# split point becomes a hole no query can retrieve through.
DEFAULT_OVERLAP = 150

_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*$")


@dataclass(frozen=True)
class Chunk:
    text: str
    source: str
    heading: str
    ordinal: int

    @property
    def id(self) -> str:
        """Stable across re-indexing, so re-running an import updates in place."""
        return f"{self.source}#{self.ordinal}"


def _sections(markdown: str) -> list[tuple[str, str]]:
    """Split into (heading path, body) pairs, one per heading."""
    stack: list[str] = []
    sections: list[tuple[str, str]] = []
    body: list[str] = []

    def flush() -> None:
        text = "\n".join(body).strip()
        if text:
            sections.append((" > ".join(stack), text))
        body.clear()

    for line in markdown.splitlines():
        match = _HEADING.match(line)
        if not match:
            body.append(line)
            continue

        flush()
        level = len(match.group(1))
        # Trim the stack to this heading's depth, so an h2 after an h3 replaces
        # it rather than nesting under it.
        del stack[level - 1 :]
        stack.append(match.group(2).strip())

    flush()
    return sections


def _paragraphs(body: str) -> list[str]:
    return [block.strip() for block in re.split(r"\n\s*\n", body) if block.strip()]


def _pack(body: str, max_chars: int, overlap: int) -> list[str]:
    """Group paragraphs into pieces of at most max_chars, with overlap."""
    pieces: list[str] = []
    current = ""

    def carry() -> str:
        # Seed the next piece with the tail of this one. Cutting on a space
        # avoids starting a chunk mid-word, which embeds badly.
        tail = current[-overlap:]
        return tail[tail.find(" ") + 1 :] if " " in tail else tail

    for paragraph in _paragraphs(body):
        while len(paragraph) > max_chars:
            # A single paragraph larger than the budget has no natural split
            # point, so it gets a hard cut. Rare in practice: usually a table
            # or a pasted log.
            head, paragraph = paragraph[:max_chars], paragraph[max_chars - overlap :]
            pieces.append(head)

        if not current:
            current = paragraph
        elif len(current) + len(paragraph) + 2 <= max_chars:
            current = f"{current}\n\n{paragraph}"
        else:
            pieces.append(current)
            current = f"{carry()}{paragraph}" if overlap else paragraph

    if current:
        pieces.append(current)
    return pieces


def split_markdown(
    markdown: str,
    source: str,
    *,
    max_chars: int = DEFAULT_MAX_CHARS,
    overlap: int = DEFAULT_OVERLAP,
) -> list[Chunk]:
    """Split a Markdown document into chunks ready for embedding.

    Every chunk is prefixed with its heading path. A chunk reading "click Save
    to confirm" is unretrievable alone; as "Exporting Reports > Scheduled
    Exports: click Save to confirm" it carries the context a query matches on.
    """
    if overlap >= max_chars:
        # Would make each chunk re-consume its own tail and never advance.
        raise ValueError("overlap must be smaller than max_chars")

    chunks: list[Chunk] = []
    for heading, body in _sections(markdown):
        for piece in _pack(body, max_chars, overlap):
            text = f"{heading}: {piece}" if heading else piece
            chunks.append(
                Chunk(text=text, source=source, heading=heading, ordinal=len(chunks))
            )
    return chunks
