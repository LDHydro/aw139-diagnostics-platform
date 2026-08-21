"""The chunker is what makes citations trustworthy, so it is tested hardest."""

from __future__ import annotations

from elp.rag.chunker import chunk_blocks, count_tokens
from elp.rag.parsers import Block


def test_section_hierarchy_is_tracked(rag_settings):
    blocks = [
        Block("Section 4 - Maintenance Control", page=5, heading_level=1),
        Block("This section governs scheduled maintenance.", page=5),
        Block("4.2 Scheduling", page=6, heading_level=2),
        Block("Tasks shall be planned before the due point.", page=6),
        Block("4.2.3 Deferral", page=7, heading_level=3),
        Block("Deferral requires written approval.", page=7),
    ]
    chunks = chunk_blocks(blocks, rag_settings)

    assert [c.section_number for c in chunks] == ["Section 4", "4.2", "4.2.3"]
    deferral = chunks[-1]
    assert deferral.heading == "Deferral"
    assert "4.2 Scheduling" in deferral.section_path
    assert "Section 4" in deferral.section_path
    assert deferral.page_start == 7


def test_chunks_never_straddle_a_section_boundary(rag_settings):
    """A passage attributed to §4.2 must not contain text from §4.3."""
    blocks = [
        Block("4.2 Scheduling", heading_level=2),
        Block("Scheduling text. " * 5),
        Block("4.3 Records", heading_level=2),
        Block("Records text. " * 5),
    ]
    chunks = chunk_blocks(blocks, rag_settings)

    for chunk in chunks:
        if chunk.section_number == "4.2":
            assert "Records text" not in chunk.text
        if chunk.section_number == "4.3":
            assert "Scheduling text" not in chunk.text


def test_numbered_procedure_steps_are_not_mistaken_for_headings(rag_settings):
    """
    "1. Remove the panel." is a procedure step, not section 1.

    Getting this wrong shreds a maintenance procedure into one-line chunks
    and produces citations to sections that do not exist.
    """
    blocks = [
        Block("5.1 Removal Procedure", heading_level=2),
        Block("1. Remove the access panel and retain the fasteners."),
        Block("2. Disconnect the electrical connector at the bulkhead."),
        Block("3. Support the unit before releasing the final attachment."),
    ]
    chunks = chunk_blocks(blocks, rag_settings)

    assert all(c.section_number == "5.1" for c in chunks)
    body = " ".join(c.text for c in chunks)
    for step in ("Remove the access panel", "Disconnect the electrical", "Support the unit"):
        assert step in body


def test_ata_style_identifiers_are_recognised(rag_settings):
    blocks = [
        Block("24-31-04 Generator Control Unit"),
        Block("The GCU regulates output voltage."),
    ]
    chunks = chunk_blocks(blocks, rag_settings)
    assert chunks[0].section_number == "24-31-04"
    assert chunks[0].heading == "Generator Control Unit"


def test_long_sections_split_with_overlap(rag_settings):
    long_text = " ".join(f"Sentence number {i} about maintenance." for i in range(400))
    chunks = chunk_blocks(
        [Block("7.1 Long Section", heading_level=2), Block(long_text)], rag_settings
    )

    assert len(chunks) > 1, "an oversized section must be split"
    assert all(c.section_number == "7.1" for c in chunks)
    # Overlap keeps a sentence from being orphaned across the boundary.
    assert chunks[1].text.split(".")[0].strip() in chunks[0].text


def test_embed_text_includes_structure(rag_settings):
    blocks = [
        Block("4.2.3 Deferral of scheduled maintenance", heading_level=3),
        Block("Approval of the Maintenance Manager is required."),
    ]
    chunk = chunk_blocks(blocks, rag_settings)[0]
    embedded = chunk.embed_text()

    assert "Deferral" in embedded
    assert chunk.text in embedded


def test_page_span_covers_a_chunk_crossing_a_page_break(rag_settings):
    blocks = [
        Block("6.1 Limits", page=10, heading_level=2),
        Block("First part of the paragraph.", page=10),
        Block("Continuation on the next page.", page=11),
    ]
    chunk = chunk_blocks(blocks, rag_settings)[0]
    assert chunk.page_start == 10
    assert chunk.page_end == 11


def test_empty_input_produces_no_chunks(rag_settings):
    assert chunk_blocks([], rag_settings) == []
    assert chunk_blocks([Block("   "), Block("")], rag_settings) == []


def test_token_counter_is_monotonic():
    assert count_tokens("") >= 1
    assert count_tokens("a short line") < count_tokens("a considerably longer line " * 20)
