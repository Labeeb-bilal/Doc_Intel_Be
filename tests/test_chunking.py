"""Stage 3 checkpoint, built entirely on handwritten Block lists — no real
files needed:
  - no chunk spans a page or section boundary
  - overlap is carried only between chunks of the same segment
  - an oversized block splits recursively and every fragment inherits page/section
  - an undersized segment merges forward under the stated rules
"""
from __future__ import annotations

from app.adapters.chunking import (
    ChunkDraft,
    _merge_undersized_segments,
    _overlap_seed,
    _pack_segment,
    _Segment,
    _split_oversized,
    chunk_blocks,
)
from app.adapters.extraction import Block


def _block(text: str, page: int | None, section_path: list[str], ordinal: int) -> Block:
    return Block(text=text, page=page, section_path=section_path, ordinal=ordinal)


def test_no_chunk_spans_a_page_or_section_boundary():
    blocks = [
        _block("Eligibility applies to all full-time staff hired after January 2020.", 1, ["Eligibility"], 0),
        _block("Remote work requires manager approval and a signed agreement on file.", 1, ["Eligibility"], 1),
        _block("Termination requires two weeks written notice from either party.", 2, ["Termination"], 2),
        _block("Severance pay is calculated based on tenure and department budget.", 2, ["Termination"], 3),
    ]

    chunks = list(chunk_blocks(blocks, target_chars=1400, max_chars=1800, min_chars=10, overlap_chars=50))

    assert len(chunks) == 2
    assert (chunks[0].page_start, chunks[0].page_end, chunks[0].section_path) == (1, 1, ["Eligibility"])
    assert (chunks[1].page_start, chunks[1].page_end, chunks[1].section_path) == (2, 2, ["Termination"])
    assert [c.ordinal for c in chunks] == [0, 1]


def test_same_section_different_pages_never_collapse_into_one_chunk():
    blocks = [
        _block("Short eligibility note on page one only.", 1, ["Eligibility"], 0),
        _block("Short eligibility note continued on page two only.", 2, ["Eligibility"], 1),
    ]

    chunks = list(chunk_blocks(blocks, target_chars=1400, max_chars=1800, min_chars=5, overlap_chars=20))

    assert len(chunks) == 2
    assert chunks[0].page_start == 1
    assert chunks[1].page_start == 2


def test_oversized_block_splits_and_every_fragment_inherits_page_and_section():
    long_text = ". ".join(f"Sentence number {i} explains policy detail {i} in full" for i in range(40)) + "."

    blocks = [_block(long_text, 3, ["Eligibility", "Hybrid schedule"], 0)]

    chunks = list(chunk_blocks(blocks, target_chars=150, max_chars=200, min_chars=10, overlap_chars=30))

    assert len(chunks) > 1
    for c in chunks:
        assert len(c.text) <= 200
        assert c.page_start == 3 and c.page_end == 3
        assert c.section_path == ["Eligibility", "Hybrid schedule"]
    assert [c.ordinal for c in chunks] == list(range(len(chunks)))


def test_undersized_segment_merges_forward_same_page_same_parent():
    blocks = [
        _block("Overview.", 1, ["Eligibility", "Overview"], 0),
        _block(
            "Employees who have completed their probation period are eligible for the hybrid schedule.",
            1,
            ["Eligibility", "Hybrid schedule"],
            1,
        ),
    ]

    chunks = list(chunk_blocks(blocks, target_chars=1400, max_chars=1800, min_chars=50, overlap_chars=20))

    assert len(chunks) == 1
    assert chunks[0].section_path == ["Eligibility", "Hybrid schedule"]
    assert "Overview." in chunks[0].text
    assert "hybrid schedule" in chunks[0].text


def test_undersized_segment_does_not_merge_across_different_parent():
    blocks = [
        _block("Overview.", 1, ["Eligibility", "Overview"], 0),
        _block(
            "Termination requires two weeks written notice from either party involved in the agreement.",
            1,
            ["Termination", "Notice"],
            1,
        ),
    ]

    chunks = list(chunk_blocks(blocks, target_chars=1400, max_chars=1800, min_chars=50, overlap_chars=20))

    assert len(chunks) == 2
    assert chunks[0].section_path == ["Eligibility", "Overview"]
    assert chunks[1].section_path == ["Termination", "Notice"]


def test_undersized_segment_does_not_merge_across_different_page():
    blocks = [
        _block("Overview.", 1, ["Eligibility", "Overview"], 0),
        _block(
            "Employees who have completed their probation period are eligible for the hybrid schedule.",
            2,
            ["Eligibility", "Hybrid schedule"],
            1,
        ),
    ]

    chunks = list(chunk_blocks(blocks, target_chars=1400, max_chars=1800, min_chars=50, overlap_chars=20))

    assert len(chunks) == 2
    assert chunks[0].page_start == 1
    assert chunks[1].page_start == 2


def test_pack_segment_respects_target_and_max():
    sentence = "This is a sentence about eligibility rules today. "
    blocks = [sentence] * 10

    packed = _pack_segment(blocks, target_chars=150, max_chars=200, overlap_chars=0)

    assert len(packed) > 1
    for text in packed:
        assert len(text) <= 200


def test_pack_segment_carries_sentence_snapped_overlap_between_chunks():
    sentence = "This is sentence {}. ".format
    blocks = [sentence(i) for i in range(12)]

    packed = _pack_segment(blocks, target_chars=60, max_chars=90, overlap_chars=25)

    assert len(packed) >= 2
    seed = _overlap_seed(packed[0], overlap_chars=25)
    if seed:
        assert packed[1].startswith(seed)
        assert seed in packed[0]


def test_split_oversized_produces_fragments_within_ceiling():
    text = ". ".join(f"Clause {i} of the agreement" for i in range(30)) + "."
    block = _block(text, 5, ["Termination"], 2)

    fragments = _split_oversized(block, max_chars=100)

    assert len(fragments) > 1
    for f in fragments:
        assert len(f.text) <= 100
        assert f.page == 5
        assert f.section_path == ["Termination"]


def test_merge_undersized_segments_chains_multiple_tiny_segments():
    segments = [
        _Segment(page=1, section_path=["A", "One"], blocks=[_block("x", 1, ["A", "One"], 0)]),
        _Segment(page=1, section_path=["A", "Two"], blocks=[_block("y", 1, ["A", "Two"], 1)]),
        _Segment(
            page=1,
            section_path=["A", "Three"],
            blocks=[_block("This is finally a long enough body of text to clear the minimum.", 1, ["A", "Three"], 2)],
        ),
    ]

    merged = _merge_undersized_segments(segments, min_chars=20)

    assert len(merged) == 1
    assert merged[0].section_path == ["A", "Three"]
    assert [b.text for b in merged[0].blocks] == ["x", "y", "This is finally a long enough body of text to clear the minimum."]
