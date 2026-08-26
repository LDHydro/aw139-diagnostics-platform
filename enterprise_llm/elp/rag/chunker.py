"""
Structure-aware chunking for governing documents.

A naive fixed-window splitter destroys exactly what makes a governing
document citable: its section numbering.  This chunker walks the block
stream, tracks the section hierarchy, and never lets a chunk straddle a
section boundary.  Every chunk therefore knows the clause it came from, so
an answer can say "OPS-MAN-001 Rev C, §4.2.3, p. 51".
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

from ..config import RagSettings, get_settings
from .parsers import Block

log = logging.getLogger(__name__)

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

# A table-of-contents entry: text, dot or space leaders, then a page number.
# These are navigation, not content. Retrieving one tells a reader nothing
# and pollutes the corpus with near-duplicates of every real heading.
_TOC_LINE = re.compile(r"^.{2,100}?[.\u00b7\u2026\-\s]{4,}\d{1,4}$")

# A revision-history row: "3.7 Changed: Section title changed from ...".
# The leading number is the section the change *refers to*, not the section
# this text belongs to. Treating it as a heading attributes every following
# block to a section it is not in - a wrong citation, which is worse than a
# missing one.
_CHANGELOG_ROW = re.compile(
    r"^\d+(?:\.\d+)*\s+(?:changed|added|removed|deleted|updated|revised|"
    r"renamed|moved|clarified)\b",
    re.IGNORECASE,
)


# "Preamble ii", "Appendices 21" - a contents entry whose page number is
# separated by nothing but a space. Matching this line-by-line across a whole
# document would be reckless ("Lesson 1", "Revision 4" are real headings), so
# it is only applied to pages that are *mostly* made of such lines.
_PAGE_REF_LINE = re.compile(r"^.{2,90}?\s+(?:[ivxlcdm]{1,7}|\d{1,4})$", re.IGNORECASE)


# Four or more dots, however they are spaced. Contents pages are built from
# these and body prose never contains them, so this is the single most
# reliable navigation signal there is.
_DOT_LEADER = re.compile(r"(?:\.\s?){4,}")


def has_dot_leader(text: str) -> bool:
    return bool(_DOT_LEADER.search(text))


def running_headers(blocks: list[Block], *, threshold: float = 0.5) -> set[str]:
    """
    Text that repeats on most pages: the running header and footer.

    A manual's page furniture - department name, revision, "Return to Table
    of Contents" - is on every page and means nothing on any of them. Left
    in, it is embedded hundreds of times and competes with real content for
    retrieval. Identified by repetition rather than position, so it works
    regardless of where the producer put it.
    """
    pages: dict[str, set[int]] = {}
    page_numbers: set[int] = set()
    for block in blocks:
        if block.page is None:
            continue
        page_numbers.add(block.page)
        text = block.text.strip()
        # Only page furniture is this short; a repeated clause is not.
        if text and len(text) <= 120:
            pages.setdefault(text, set()).add(block.page)

    if len(page_numbers) < 4:
        return set()
    minimum = max(3, int(len(page_numbers) * threshold))
    return {text for text, seen in pages.items() if len(seen) >= minimum}


def navigation_pages(blocks: list[Block], *, threshold: float = 0.6) -> set[int]:
    """
    Pages that are a table of contents, an index, or a revision-history table.

    Detected at page level rather than line level. A single line ending in a
    number proves nothing - "Revision 4" and "Lesson 1" are perfectly good
    headings - but a page where most lines end that way is navigation, and
    indexing it returns page numbers to someone who asked a question.
    """
    by_page: dict[int, list[str]] = {}
    for block in blocks:
        if block.page is None:
            continue
        text = block.text.strip()
        if text:
            by_page.setdefault(block.page, []).append(text)

    navigation: set[int] = set()
    for page, texts in by_page.items():
        if len(texts) < 5:
            continue
        hits = sum(
            1
            for text in texts
            if has_dot_leader(text)
            or _TOC_LINE.match(text)
            or _PAGE_REF_LINE.match(text)
        )
        if hits / len(texts) >= threshold:
            navigation.add(page)
    return navigation


def is_navigation(text: str) -> bool:
    """Table-of-contents entries and index lines carry no retrievable content."""
    stripped = text.strip()
    if not stripped:
        return False
    if has_dot_leader(stripped):
        return True
    return bool(_TOC_LINE.match(stripped)) and not stripped.endswith(".")


_ENCODER: Any = None
_ENCODER_RESOLVED = False


def _encoder() -> Any:
    """
    Resolve the tiktoken encoder once, or give up on it once.

    tiktoken downloads the BPE vocabulary on first use.  On a network
    restricted box - which the deployment target is - that download fails
    slowly, with retries.  Resolving per call would pay that cost for every
    sentence of every document, so a failure is cached as firmly as a
    success and the fallback is used from then on.
    """
    global _ENCODER, _ENCODER_RESOLVED
    if not _ENCODER_RESOLVED:
        try:
            import tiktoken

            _ENCODER = tiktoken.get_encoding("cl100k_base")
        except Exception:
            _ENCODER = None
        _ENCODER_RESOLVED = True
    return _ENCODER


def count_tokens(text: str) -> int:
    """
    Approximate token count.

    Uses tiktoken when available.  The fallback (~4 characters per token)
    is deliberately conservative: over-estimating means slightly smaller
    chunks, which is harmless, whereas under-estimating overruns context.

    Both paths agree on the boundary that callers depend on: empty text
    costs nothing, and anything non-empty costs at least one token.  They
    have to, because whether tiktoken can fetch its vocabulary is a property
    of the network the box is on, not of the code, and chunk sizing must not
    change underneath the same document.
    """
    if not text:
        return 0
    encoder = _encoder()
    if encoder is None:
        return max(1, len(text) // 4)
    return max(1, len(encoder.encode(text)))


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

    # Neither of these starts a section, however much they look like one.
    if _CHANGELOG_ROW.match(text) or is_navigation(text):
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
        # A styled line ending in a colon is a lead-in label - "Objective:",
        # "Completion standards:", "References:" - not a section heading.
        # They are bold and short, so typography alone cannot tell them
        # apart, and treating each as a section shreds a procedure into
        # one-line fragments and attaches a meaningless heading to each.
        if text.endswith(":"):
            return False, None
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

        skip_pages = navigation_pages(blocks)
        furniture = running_headers(blocks)
        if skip_pages or furniture:
            log.debug(
                "skipping %d navigation page(s) and %d running header/footer line(s)",
                len(skip_pages), len(furniture),
            )

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
            if text in furniture:
                # Page furniture: present on every page, meaningful on none.
                continue
            if block.page in skip_pages or is_navigation(text):
                # Contents and index pages are navigation; indexing them
                # would return a page number where the reader wanted the
                # clause itself.
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

        chunks = self._merge_undersized(chunks)

        # Re-number after the fact so ordinals are dense and stable.
        for index, chunk in enumerate(chunks):
            chunk.ordinal = index
        return [c for c in chunks if c.text.strip()]

    def _merge_undersized(self, chunks: list[Chunk]) -> list[Chunk]:
        """
        Join neighbouring fragments that share a section.

        Documents with dense sub-headings produce a long tail of one- and
        two-line chunks. Individually they retrieve badly - too little
        context to match a question, and too little to answer one - so
        adjacent fragments from the same section are combined up to the
        target size. Section boundaries are still never crossed.
        """
        minimum = max(0, self.settings.chunk_min_tokens)
        if minimum <= 0 or not chunks:
            return chunks

        target = self.settings.chunk_target_tokens
        merged: list[Chunk] = []

        for chunk in chunks:
            previous = merged[-1] if merged else None
            if (
                previous is not None
                and previous.token_count < minimum
                and previous.section_number == chunk.section_number
                and previous.section_path == chunk.section_path
                and previous.token_count + chunk.token_count <= target
            ):
                previous.text = f"{previous.text}\n\n{chunk.text}".strip()
                previous.token_count = count_tokens(previous.text)
                if chunk.page_start is not None:
                    previous.page_start = (
                        chunk.page_start
                        if previous.page_start is None
                        else min(previous.page_start, chunk.page_start)
                    )
                if chunk.page_end is not None:
                    previous.page_end = (
                        chunk.page_end
                        if previous.page_end is None
                        else max(previous.page_end, chunk.page_end)
                    )
                if not previous.heading:
                    previous.heading = chunk.heading
                continue
            merged.append(chunk)

        return merged


def chunk_blocks(blocks: list[Block], settings: RagSettings | None = None) -> list[Chunk]:
    return StructureAwareChunker(settings).chunk(blocks)
