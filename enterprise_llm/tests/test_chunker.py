"""The chunker is what makes citations trustworthy, so it is tested hardest."""

from __future__ import annotations

from elp.rag import chunker
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
    assert count_tokens("") == 0
    assert count_tokens("a") >= 1
    assert count_tokens("a short line") < count_tokens("a considerably longer line " * 20)


def test_token_counter_agrees_at_the_boundaries_without_tiktoken(monkeypatch):
    """
    The fallback has to honour the same contract as the real encoder.

    Whether tiktoken can fetch its vocabulary depends on the network the box
    is on, so both paths run in production and must size chunks the same way
    at the edges.  This test forces the fallback; the one above runs on
    whichever path the environment provides.
    """
    monkeypatch.setattr(chunker, "_ENCODER", None)
    monkeypatch.setattr(chunker, "_ENCODER_RESOLVED", True)
    assert count_tokens("") == 0
    assert count_tokens("a") >= 1
    assert count_tokens("a short line") < count_tokens("a considerably longer line " * 20)


# ----------------------------------------------------------------------
# Document furniture
#
# These behaviours were all derived from running three real governing
# manuals through the pipeline, where each was measurably degrading either
# retrieval quality or citation accuracy.
# ----------------------------------------------------------------------

def test_contents_entries_are_not_indexed(rag_settings):
    """
    Retrieving a table-of-contents line returns a page number to someone
    who asked a question. It also produces a near-duplicate of every real
    heading, competing with the section it points at.
    """
    blocks = [
        Block("Table of Contents", page=2, heading_level=1),
        Block("1 General . . . . . . . . . . . . . . . 1", page=2),
        Block("2 Training . . . . . . . . . . . . . . 12", page=2),
        Block("3 Records . . . . . . . . . . . . . . 25", page=2),
        Block("Appendices . . . . . . . . . . . . . . 40", page=2),
        Block("Preamble . . . . . . . . . . . . . . . ii", page=2),
        Block("1 General", page=10, heading_level=1),
        Block("The department shall maintain currency records.", page=10),
    ]
    chunks = chunk_blocks(blocks, rag_settings)

    body = " ".join(c.text for c in chunks)
    assert "currency records" in body
    assert "." * 4 not in body


def test_running_headers_and_footers_are_stripped(rag_settings):
    """
    Page furniture is on every page and means nothing on any of them.
    Left in, it is embedded hundreds of times and competes with content.
    """
    blocks = []
    for page in range(1, 11):
        blocks.append(Block("Remote Sensing Laboratory Training Manual", page=page))
        blocks.append(Block("Revision 4 (DRAFT)", page=page))
        blocks.append(Block(f"Unique body text for page {page}. " * 12, page=page))
    chunks = chunk_blocks(blocks, rag_settings)

    body = " ".join(c.text for c in chunks)
    assert "Unique body text" in body
    assert "Revision 4 (DRAFT)" not in body


def test_a_repeated_phrase_in_real_content_is_kept(rag_settings):
    """Furniture detection keys on repetition across pages, so genuine
    content that happens to recur on two pages must survive."""
    blocks = []
    for page in range(1, 11):
        blocks.append(Block(f"Body paragraph number {page}. " * 12, page=page))
    blocks.append(Block("The commander shall approve all deviations.", page=3))
    blocks.append(Block("The commander shall approve all deviations.", page=7))
    chunks = chunk_blocks(blocks, rag_settings)

    assert "commander shall approve" in " ".join(c.text for c in chunks)


def test_a_lead_in_label_does_not_start_a_section(rag_settings):
    """
    "Objective:" and "References:" are field labels inside a procedure,
    bold and short exactly like a heading. Treating each as a section
    shreds the procedure into one-line fragments and attaches a
    meaningless heading to every one.
    """
    blocks = [
        Block("Lesson 1: Ground - Aircraft Systems", page=32, heading_level=2),
        Block("Objective:", page=32, heading_level=3),
        Block("Review King Air B350 aircraft systems.", page=32),
        Block("Discussion topics:", page=32, heading_level=3),
        Block("Fuel system, electrical system, powerplant.", page=32),
        Block("Completion standards:", page=32, heading_level=3),
        Block("Demonstrate systems knowledge to the instructor.", page=32),
    ]
    chunks = chunk_blocks(blocks, rag_settings)

    assert len(chunks) == 1, "the lesson should stay in one retrievable piece"
    assert chunks[0].heading.startswith("Lesson 1")
    for label in ("Objective:", "Discussion topics:", "Completion standards:"):
        assert label in chunks[0].text


def test_a_revision_history_row_does_not_reassign_the_section(rag_settings):
    """
    "3.7 Changed: Section title changed from ..." names the section the
    change *refers to*, not the section the text belongs to. Treating it as
    a heading attributes everything after it to a section it is not in,
    which is a wrong citation — worse than a missing one.
    """
    blocks = [
        Block("2 Categories of Training", page=8, heading_level=1),
        Block("Training is categorised as initial, recurrent or requalification.", page=8),
        Block("3.7 Changed: Section title changed from Crewmember Rules.", page=9),
        Block("Further body text that belongs to section 2.", page=9),
    ]
    chunks = chunk_blocks(blocks, rag_settings)

    assert all(c.section_number != "3.7" for c in chunks)


def test_undersized_fragments_in_one_section_are_merged(rag_settings):
    """
    A one-line chunk retrieves badly: too little context to match a
    question and too little to answer one.
    """
    blocks = [
        Block("4.2 Currency", page=20, heading_level=2),
        Block("Pilots shall log three landings.", page=20),
        Block("Landings must be within ninety days.", page=20),
        Block("Night currency requires night landings.", page=20),
    ]
    chunks = chunk_blocks(blocks, rag_settings)

    assert len(chunks) == 1
    assert chunks[0].token_count > 15
    assert "ninety days" in chunks[0].text


def test_merging_never_crosses_a_section_boundary(rag_settings):
    """The whole point of the chunker is that a citation is accurate."""
    blocks = [
        Block("4.2 Currency", page=20, heading_level=2),
        Block("Short text A.", page=20),
        Block("4.3 Records", page=21, heading_level=2),
        Block("Short text B.", page=21),
    ]
    chunks = chunk_blocks(blocks, rag_settings)

    sections = {c.section_number: c.text for c in chunks}
    assert "Short text B" not in sections.get("4.2", "")
    assert "Short text A" not in sections.get("4.3", "")


def test_every_chunk_is_citable(rag_settings):
    """A passage with neither a section number nor a heading cannot be
    cited, which makes it useless in an answer."""
    blocks = [
        Block("5 Operations", page=30, heading_level=1),
        Block("Body text under a numbered section. " * 8, page=30),
        Block("Unnumbered Heading", page=31, heading_level=2),
        Block("Body text under a titled section. " * 8, page=31),
    ]
    chunks = chunk_blocks(blocks, rag_settings)

    assert chunks
    assert all(c.section_number or c.heading for c in chunks)
