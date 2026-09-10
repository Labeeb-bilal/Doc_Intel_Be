"""Stage 5 checkpoint: PDF, DOCX, MD extraction — one fixture each, asserting
page numbers (PDF) and section paths (DOCX, MD)."""
from __future__ import annotations

from pathlib import Path

import pytest

from app.adapters.extraction import (
    EmptyDocumentError,
    NoTextLayerError,
    extract_docx,
    extract_md,
    extract_pdf,
)

FIXTURES = Path(__file__).parent / "fixtures"


def test_pdf_fixture_has_correct_pages_and_strips_running_header_footer():
    data = (FIXTURES / "sample.pdf").read_bytes()
    blocks = list(extract_pdf(data))

    assert [b.page for b in blocks] == [1, 2, 3]
    assert [b.ordinal for b in blocks] == [0, 1, 2]
    assert all(b.section_path == [] for b in blocks)

    full_text = " ".join(b.text for b in blocks)
    assert "REMOTE WORK POLICY" not in full_text
    assert "Confidential" not in full_text


def test_pdf_fixture_rejoins_hyphenated_word_across_line_break():
    data = (FIXTURES / "sample.pdf").read_bytes()
    blocks = list(extract_pdf(data))

    page_two_text = blocks[1].text
    assert "documentation" in page_two_text
    assert "documenta-" not in page_two_text


def test_pdf_scanned_image_raises_no_text_layer_error():
    import subprocess
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "src.txt"
        src.write_text("x\n")
        out = Path(tmp) / "out.pdf"
        result = subprocess.run(["cupsfilter", str(src)], stdout=open(out, "wb"), stderr=subprocess.DEVNULL)
        if result.returncode != 0 or not out.exists():
            pytest.skip("cupsfilter unavailable in this environment")
        data = out.read_bytes()

    with pytest.raises(NoTextLayerError):
        list(extract_pdf(data))


def test_docx_fixture_has_correct_section_paths_and_null_pages():
    data = (FIXTURES / "sample.docx").read_bytes()
    blocks = list(extract_docx(data))

    assert all(b.page is None for b in blocks)
    section_paths = [b.section_path for b in blocks]
    assert ["Eligibility"] in section_paths
    assert ["Eligibility", "Hybrid schedule"] in section_paths
    assert ["Termination"] in section_paths
    assert [b.ordinal for b in blocks] == list(range(len(blocks)))


def test_docx_fixture_includes_table_content_in_document_order():
    data = (FIXTURES / "sample.docx").read_bytes()
    blocks = list(extract_docx(data))

    table_blocks = [b for b in blocks if "|" in b.text]
    assert any("Role" in b.text and "Max remote days" in b.text for b in table_blocks)
    assert any("Engineer" in b.text and "3" in b.text for b in table_blocks)
    assert all(b.section_path == ["Eligibility", "Hybrid schedule"] for b in table_blocks)


def test_docx_empty_document_raises():
    from docx import Document as DocxDocument
    from io import BytesIO

    buf = BytesIO()
    DocxDocument().save(buf)
    with pytest.raises(EmptyDocumentError):
        list(extract_docx(buf.getvalue()))


def test_md_fixture_has_correct_section_paths_and_null_pages():
    data = (FIXTURES / "sample.md").read_bytes()
    blocks = list(extract_md(data))

    assert all(b.page is None for b in blocks)
    section_paths = [b.section_path for b in blocks]
    assert ["Eligibility"] in section_paths
    assert ["Eligibility", "Hybrid schedule"] in section_paths
    assert ["Termination"] in section_paths


def test_md_fixture_parses_frontmatter_effective_date():
    from datetime import date

    data = (FIXTURES / "sample.md").read_bytes()
    metadata: dict = {}
    list(extract_md(data, metadata=metadata))

    assert metadata["effective_date"] == date(2024, 3, 15)


def test_md_fixture_code_fence_not_split_and_hash_inside_not_a_heading():
    data = (FIXTURES / "sample.md").read_bytes()
    blocks = list(extract_md(data))

    fence_blocks = [b for b in blocks if "```" in b.text]
    assert len(fence_blocks) == 1
    assert "def is_eligible" in fence_blocks[0].text
    assert "# this hash should NOT be treated as a heading" in fence_blocks[0].text
    assert "this hash should not be treated as a heading" not in [
        s.lower() for path in [b.section_path for b in blocks] for s in path
    ]


def test_md_without_frontmatter_has_no_effective_date():
    content = b"# Title\n\nJust a paragraph, no frontmatter here.\n"
    metadata: dict = {}
    list(extract_md(content, metadata=metadata))
    assert metadata.get("effective_date") is None


def test_md_empty_file_raises():
    with pytest.raises(EmptyDocumentError):
        list(extract_md(b""))
