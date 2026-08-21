"""
Structure-aware chunking for governing documents.

A naive fixed-window splitter destroys exactly what makes a governing
document citable: its section numbering.  This chunker walks the block
stream, tracks the section hierarchy, and never lets a chunk straddle a
section boundary.  Every chunk therefore knows the clause it came from, so
an answer can say "OPS-MAN-001 Rev C, §4.2.3, p. 51".
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..config import RagSettings, get_settings
from .parsers import Block

# "4.2.3  Deferral of scheduled maintenance"
_NUMBERED = re.compile(r"^(\d+(?:\.\d+){0,5})[.)]?\s+(\S.{0,150})$")
# "Section 4 - Maintenance control", "Appendix B: Forms", "Chapter 12"
_NAMED = re.compile(
    r"^(SECTION|CHAPTER|PART|APPENDIX|ANNEX|SCHEDULE)\s+"
    r"([0-9]+|[IVXLC]+|[A-Z])\b[\s.:\-]*(.*)$",
    re.IGNORECASE,
)
# ATA-style identifiers used throughout aviation manuals: "24-31-04"
_ATA = re.compile(r"^(\d{2}-\d{2}-\d{2})\b[\s.:\-]*(.*)$")

_SENTENCE_END = re.compile(r"(?<=[.!?;:])\s+")


def count_tokens(text: str) -> int:
    """
    Approximate token count.

    Uses tiktoken when available.  The fallback (~4 characters per token)
    is deliberately conservative: over-estimating means slightly smaller
    chunks, which is harmless, whereas under-estimating overruns context.
    """
    try:
        import tiktoken

        return len(tiktoken.get_encoding("cl100k_base").encode(text))
    except Exception:
        return max(1, len(text) // 4)


@dataclass
class Section:
    number: str = ""
    title: str = ""
    level: int = 0

    @property
    def label(self) -> str:
        if self.number and self.title:
            return f"{self.number} {self.title}"
        return self.number or self.title


@dataclass
class Chunk:
    """A retrievable passage with everything needed to cite it."""

    ordinal: int
    text: str
    section_number: str = ""
    section_path: str = ""
    heading: str = ""
    page_start: int | None = None
    page_end: int | None = None
    token_count: int = 0
    meta: dict = field(default_factory=dict)

    def embed_text(self) -> str:
        """
        What actually gets embedded.

        Prefixing the section path makes retrieval sensitive to structure:
        a question about "deferral limits" matches the deferral section even
        when the passage body never repeats the word.
        """
        header = " > ".join(p for p in [self.section_path, self.heading] if p)
        return f"{header}\n\n{self.text}" if header else self.text


def _looks_like_heading(block: Block) -> tuple[bool, Section | None]:
    """Decide whether a block starts a new section, and parse its number."""
    text = block.text.strip()
    if not text:
        return False, None

    match = _NAMED.match(text)
    if match:
        number = f"{match.group(1).title()} {match.group(2)}"
        return True, Section(number=number, title=match.group(3).strip(), level=1)

    match = _ATA.match(text)
    if match and len(text) < 160:
        return True, Section(number=match.group(1), title=match.group(2).strip(), level=2)

    match = _NUMBERED.match(text)
    if match:
        number, title = match.group(1), match.group(2).strip()
        depth = number.count(".") + 1
        # Guard against numbered procedure steps, which look identical to
        # headings but read as sentences and usually end with a period.
        title_like = (
            len(title) < 120
            and not title.endswith(".")
            and title[:1].isupper()
        )
        if block.is_heading or title_like:
            return True, Section(number=number, title=title, level=depth)

    if block.is_heading:
        return True, Section(number="", title=text[:150], level=block.heading_level or 1)

    return False, None


def _split_paragraph(text: str, max_tokens: int) -> list[str]:
    """Break an oversized paragraph on sentence boundaries."""
    if count_tokens(text) <= max_tokens:
        return [text]
    pieces: list[str] = []
    current: list[str] = []
    current_tokens = 0
    for sentence in _SENTENCE_END.split(text):
        sentence = sentence.strip()
        if not sentence:
            continue
        tokens = count_tokens(sentence)
        if current and current_tokens + tokens > max_tokens:
            pieces.append(" ".join(current))
            current, current_tokens = [], 0
        current.append(sentence)
        current_tokens += tokens
    if current:
        pieces.append(" ".join(current))
    return pieces or [text[: max_tokens * 4]]


def _tail_for_overlap(text: str, overlap_tokens: int) -> str:
    """Take roughly ``overlap_tokens`` from the end, on a sentence boundary."""
    if overlap_tokens <= 0:
        return ""
    sentences = _SENTENCE_END.split(text)
    tail: list[str] = []
    total = 0
    for sentence in reversed(sentences):
        sentence = sentence.strip()
        if not sentence:
            continue
        tokens = count_tokens(sentence)
        if total + tokens > overlap_tokens and tail:
            break
        tail.insert(0, sentence)
        total += tokens
    return " ".join(tail)


class StructureAwareChunker:
    def __init__(self, settings: RagSettings | None = None) -> None:
        self.settings = settings or get_settings().rag

    def chunk(self, blocks: list[Block]) -> list[Chunk]:
        target = self.settings.chunk_target_tokens
        hard_max = self.settings.chunk_max_tokens
        overlap = self.settings.chunk_overlap_tokens

        chunks: list[Chunk] = []
        stack: list[Section] = []
        current_section: Section | None = None

        buffer: list[str] = []
        buffer_tokens = 0
        page_start: int | None = None
        page_end: int | None = None
        pending_overlap = ""

        def section_path() -> str:
            return " > ".join(s.label for s in stack if s.label)

        def flush() -> None:
            nonlocal buffer, buffer_tokens, page_start, page_end, pending_overlap
            body = "\n\n".join(b for b in buffer if b.strip()).strip()
            if not body:
                buffer, buffer_tokens = [], 0
                return
            chunk = Chunk(
                ordinal=len(chunks),
                text=body,
                section_number=current_section.number if current_section else "",
                section_path=section_path(),
                heading=current_section.title if current_section else "",
                page_start=page_start,
                page_end=page_end,
                token_count=count_tokens(body),
            )
            chunks.append(chunk)
            pending_overlap = _tail_for_overlap(body, overlap)
            buffer, buffer_tokens = [], 0
            page_start = page_end = None

        for block in blocks:
            text = block.text.strip()
            if not text:
                continue

            is_heading, section = _looks_like_heading(block)
            if is_heading and section is not None:
                # A new section closes the current chunk; passages never
                # straddle a clause boundary.
                flush()
                pending_overlap = ""
                while stack and stack[-1].level >= section.level:
                    stack.pop()
                stack.append(section)
                current_section = section
                if block.page is not None:
                    page_start = page_end = block.page
                continue

            for piece in _split_paragraph(text, hard_max):
                tokens = count_tokens(piece)
                if buffer and buffer_tokens + tokens > target:
                    flush()
                if not buffer and pending_overlap:
                    buffer.append(pending_overlap)
                    buffer_tokens += count_tokens(pending_overlap)
                    pending_overlap = ""
                buffer.append(piece)
                buffer_tokens += tokens
                if block.page is not None:
                    page_start = block.page if page_start is None else min(page_start, block.page)
                    page_end = block.page if page_end is None else max(page_end, block.page)

        flush()

        # Re-number after the fact so ordinals are dense and stable.
        for index, chunk in enumerate(chunks):
            chunk.ordinal = index
        return [c for c in chunks if c.text.strip()]


def chunk_blocks(blocks: list[Block], settings: RagSettings | None = None) -> list[Chunk]:
    return StructureAwareChunker(settings).chunk(blocks)
