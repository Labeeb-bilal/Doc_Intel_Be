"""Stage 2 checkpoint: TXT extraction — blocks carry correct ordinals,
page is always None, and blank-line splitting / encoding handling work."""
from __future__ import annotations

from pathlib import Path

import pytest

from app.adapters.extraction import (
    EmptyDocumentError,
    EncodingFailedError,
    extract_txt,
)

FIXTURE = Path(__file__).parent / "fixtures" / "sample.txt"


def test_fixture_produces_three_ordered_blocks():
    data = FIXTURE.read_bytes()
    blocks = list(extract_txt(data))

    assert len(blocks) == 3
    assert [b.ordinal for b in blocks] == [0, 1, 2]
    assert all(b.page is None for b in blocks)
    assert all(b.section_path == ["Part 1"] for b in blocks)
    assert "first paragraph" in blocks[0].text
    assert "second paragraph" in blocks[1].text
    assert "third paragraph" in blocks[2].text
    assert "\n\n" not in blocks[0].text


def test_extract_txt_is_a_generator_not_a_list():
    import inspect

    assert inspect.isgeneratorfunction(extract_txt)


def test_empty_file_raises_empty_document_error():
    with pytest.raises(EmptyDocumentError):
        list(extract_txt(b""))


def test_whitespace_only_file_raises_empty_document_error():
    with pytest.raises(EmptyDocumentError):
        list(extract_txt(b"   \n\n   \n"))


def test_cp1252_encoding_is_detected_and_decoded():
    original = "The employee’s remote schedule is flexible."
    data = original.encode("cp1252")

    blocks = list(extract_txt(data))

    assert len(blocks) == 1
    assert blocks[0].text == original


def test_part_anchor_cycles_every_thirty_blocks():
    paragraphs = [f"Paragraph number {i} with some body text." for i in range(65)]
    data = ("\n\n".join(paragraphs)).encode("utf-8")

    blocks = list(extract_txt(data))

    assert len(blocks) == 65
    assert blocks[0].section_path == ["Part 1"]
    assert blocks[29].section_path == ["Part 1"]
    assert blocks[30].section_path == ["Part 2"]
    assert blocks[59].section_path == ["Part 2"]
    assert blocks[60].section_path == ["Part 3"]
    assert blocks[64].section_path == ["Part 3"]
    assert [b.ordinal for b in blocks] == list(range(65))
