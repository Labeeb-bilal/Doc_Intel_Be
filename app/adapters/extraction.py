"""Text extraction adapters. Every parser yields Block — the chunker never
learns which format produced it.

Extraction functions are generators, never functions returning a list: a
500-page PDF must not materialise in memory.

All extractors share the signature `(data: bytes, *, metadata: dict | None)
-> Iterator[Block]`. `metadata` is an optional out-parameter: only
extract_md populates it (with a YAML-frontmatter effective_date), but every
extractor accepts it so the service layer can call them uniformly.
"""
from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import date
from io import BytesIO

from charset_normalizer import from_bytes


@dataclass
class Block:
    """Format-agnostic intermediate the chunker consumes."""

    text: str
    page: int | None  # None for DOCX, MD, TXT
    section_path: list[str] = field(default_factory=list)
    ordinal: int = 0


class ExtractionError(Exception):
    """Base for typed, per-format extraction failures.

    The service layer catches this, maps `.code` to documents.error_code,
    and marks the document failed. It must never crash the background task
    or affect other documents in the same upload.
    """

    code: str = "CORRUPT_FILE"

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class CorruptFileError(ExtractionError):
    code = "CORRUPT_FILE"


class NoTextLayerError(ExtractionError):
    code = "NO_TEXT_LAYER"


class EmptyDocumentError(ExtractionError):
    code = "EMPTY_DOCUMENT"


class EncodingFailedError(ExtractionError):
    code = "ENCODING_FAILED"


# --- TXT -------------------------------------------------------------------

_BLANK_LINE_RE = re.compile(r"\n\s*\n+")
_TXT_PART_SIZE = 30  # blocks per "Part N" anchor, so citations always have somewhere to point


def extract_txt(data: bytes, *, metadata: dict | None = None) -> Iterator[Block]:
    """TXT parser.

    page is always None — plain text carries no pagination. Blocks are
    split on blank lines; every ~30 blocks gets a fresh "Part N" anchor
    purely so citations have something to name, since TXT has no headings.
    """
    if not data:
        raise EmptyDocumentError("The uploaded file is empty.")

    detected = from_bytes(data).best()
    if detected is None:
        raise EncodingFailedError("Could not detect a valid text encoding for this file.")
    text = str(detected)

    paragraphs = [p.strip() for p in _BLANK_LINE_RE.split(text)]
    paragraphs = [p for p in paragraphs if p]

    if not paragraphs:
        raise EmptyDocumentError("The uploaded file contains no extractable text.")

    for ordinal, paragraph in enumerate(paragraphs):
        part = ordinal // _TXT_PART_SIZE + 1
        yield Block(text=paragraph, page=None, section_path=[f"Part {part}"], ordinal=ordinal)


# --- PDF ---------------------------------------------------------------

_HYPHEN_BREAK_RE = re.compile(r"-\n(?=[a-z])")
_MIN_CHARS_PER_PAGE_FOR_TEXT_LAYER = 50  # below this, it's almost certainly a scanned image


def _normalize_pdf_page_text(lines: list[str]) -> str:
    """Rejoin hyphenation, then collapse mid-paragraph line wraps to spaces
    while preserving genuine blank-line paragraph breaks."""
    joined = "\n".join(lines)
    joined = _HYPHEN_BREAK_RE.sub("", joined)
    joined = re.sub(r"\n\s*\n+", " ", joined)  # protect paragraph breaks
    joined = joined.replace("\n", " ")
    joined = joined.replace(" ", "\n\n")
    return re.sub(r"[ \t]+", " ", joined).strip()


def extract_pdf(data: bytes, *, metadata: dict | None = None) -> Iterator[Block]:
    """PDF parser.

    Running headers/footers are detected by frequency — a line appearing as
    the first or last line of a page on more than half the pages — and
    stripped everywhere; left in, they'd pollute every chunk and create
    false cross-document similarity later. Hyphenated words broken across a
    line wrap are rejoined so citations never quote a broken word.

    No heading/section detection is attempted here: pdfplumber gives no
    structural markup the way DOCX styles or Markdown '#' do, so PDF blocks
    carry an empty section_path (page number is still enough for a citation).
    """
    import pdfplumber

    try:
        pdf = pdfplumber.open(BytesIO(data))
    except Exception as exc:
        raise CorruptFileError(f"Could not open PDF: {exc}") from exc

    try:
        page_count = len(pdf.pages)
        if page_count == 0:
            raise EmptyDocumentError("The PDF has no pages.")

        page_texts: list[str] = []
        boundary_line_counts: dict[str, int] = {}
        for page in pdf.pages:
            text = page.extract_text() or ""
            page_texts.append(text)
            lines = [line.strip() for line in text.split("\n") if line.strip()]
            for line in ({lines[0], lines[-1]} if lines else set()):
                boundary_line_counts[line] = boundary_line_counts.get(line, 0) + 1
            page.flush_cache()

        total_chars = sum(len(t) for t in page_texts)
        if (total_chars / page_count) < _MIN_CHARS_PER_PAGE_FOR_TEXT_LAYER:
            raise NoTextLayerError(
                "This PDF appears to be a scanned image. Text extraction requires OCR, which is not supported."
            )

        noise_threshold = page_count / 2
        noise_lines = {line for line, count in boundary_line_counts.items() if count > noise_threshold}

        ordinal = 0
        any_content = False
        for page_number, text in enumerate(page_texts, start=1):
            surviving_lines = [line for line in text.split("\n") if line.strip() not in noise_lines]
            normalized = _normalize_pdf_page_text(surviving_lines)
            for paragraph in normalized.split("\n\n"):
                paragraph = paragraph.strip()
                if not paragraph:
                    continue
                yield Block(text=paragraph, page=page_number, section_path=[], ordinal=ordinal)
                ordinal += 1
                any_content = True

        if not any_content:
            raise EmptyDocumentError("No extractable text remained after removing repeated headers/footers.")
    finally:
        pdf.close()


# --- DOCX ----------------------------------------------------------------

_DOCX_HEADING_STYLE_RE = re.compile(r"^heading\s*(\d)$", re.IGNORECASE)


def _docx_outline_level(pPr_owner) -> int | None:
    """Read <w:outlineLvl w:val="N"/> from a <w:pPr>-bearing element, if
    present. python-docx has no high-level property for this."""
    from docx.oxml.ns import qn

    if pPr_owner is None:
        return None
    ppr = pPr_owner.find(qn("w:pPr"))
    if ppr is None:
        return None
    node = ppr.find(qn("w:outlineLvl"))
    if node is None:
        return None
    val = node.get(qn("w:val"))
    return int(val) if val is not None else None


def _docx_heading_level(paragraph) -> int | None:
    style_name = paragraph.style.name if paragraph.style is not None else None
    if style_name:
        match = _DOCX_HEADING_STYLE_RE.match(style_name.strip())
        if match:
            return int(match.group(1))

    # Fallback for renamed styles in custom templates: an outline level can
    # be set directly on the paragraph, or inherited from its style — check
    # both. Word's outline levels are 0-indexed (0 -> Heading level 1).
    level = _docx_outline_level(paragraph._p)
    if level is None and paragraph.style is not None:
        level = _docx_outline_level(paragraph.style.element)
    return (level + 1) if level is not None else None


def extract_docx(data: bytes, *, metadata: dict | None = None) -> Iterator[Block]:
    """DOCX parser.

    Walks document.element.body directly so paragraphs AND tables are seen
    in document order — document.paragraphs silently skips table contents,
    and these documents put critical values in tables.

    page is always None: DOCX pagination is a rendering property of the
    viewer, not data in the file, so it is never fabricated.
    """
    from docx import Document as DocxDocument
    from docx.oxml.ns import qn
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    try:
        doc = DocxDocument(BytesIO(data))
    except Exception as exc:
        raise CorruptFileError(f"Could not open DOCX: {exc}") from exc

    section_stack: list[str] = []
    ordinal = 0
    any_content = False

    for child in doc.element.body.iterchildren():
        if child.tag == qn("w:p"):
            paragraph = Paragraph(child, doc)
            text = paragraph.text.strip()
            if not text:
                continue
            level = _docx_heading_level(paragraph)
            if level is not None:
                section_stack[level - 1 :] = [text]
                continue
            yield Block(text=text, page=None, section_path=list(section_stack), ordinal=ordinal)
            ordinal += 1
            any_content = True

        elif child.tag == qn("w:tbl"):
            table = Table(child, doc)
            for row in table.rows:
                row_text = " | ".join(cell.text.strip() for cell in row.cells)
                if not row_text.strip(" |"):
                    continue
                yield Block(text=row_text, page=None, section_path=list(section_stack), ordinal=ordinal)
                ordinal += 1
                any_content = True

    if not any_content:
        raise EmptyDocumentError("The DOCX file contains no extractable text.")


# --- Markdown --------------------------------------------------------------

_MD_FENCE_RE = re.compile(r"^(```|~~~)")
_MD_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
_MD_FRONTMATTER_DELIM_RE = re.compile(r"^---\s*$")
_MD_FRONTMATTER_DATE_RE = re.compile(r"^\s*(effective_date|date)\s*:\s*(.+?)\s*$", re.IGNORECASE)


def _parse_frontmatter_date(lines: list[str]) -> date | None:
    for line in lines:
        match = _MD_FRONTMATTER_DATE_RE.match(line)
        if not match:
            continue
        raw_value = match.group(2).strip().strip("'\"")
        try:
            return date.fromisoformat(raw_value[:10])
        except ValueError:
            continue
    return None


def extract_md(data: bytes, *, metadata: dict | None = None) -> Iterator[Block]:
    """Markdown parser.

    Fenced code blocks are tracked so a '#' inside one is never mistaken
    for a heading, and a fence is never split across chunks — it becomes
    exactly one Block. YAML frontmatter is scanned only for an
    effective_date/date key (via `metadata`, not a real YAML parser); it is
    never emitted as a Block.
    """
    if not data:
        raise EmptyDocumentError("The uploaded file is empty.")

    detected = from_bytes(data).best()
    if detected is None:
        raise EncodingFailedError("Could not detect a valid text encoding for this file.")
    text = str(detected)
    lines = text.split("\n")

    start = 0
    if lines and _MD_FRONTMATTER_DELIM_RE.match(lines[0]):
        for i in range(1, len(lines)):
            if _MD_FRONTMATTER_DELIM_RE.match(lines[i]):
                if metadata is not None:
                    metadata["effective_date"] = _parse_frontmatter_date(lines[1:i])
                start = i + 1
                break

    section_stack: list[str] = []
    ordinal = 0
    any_content = False
    in_fence = False
    fence_marker = ""
    para_buffer: list[str] = []
    fence_buffer: list[str] = []

    def _block_from(buffer: list[str]) -> Block | None:
        nonlocal ordinal
        joined = "\n".join(buffer).strip("\n")
        if not joined.strip():
            return None
        block = Block(text=joined, page=None, section_path=list(section_stack), ordinal=ordinal)
        ordinal += 1
        return block

    for line in lines[start:]:
        if in_fence:
            fence_buffer.append(line)
            if line.strip().startswith(fence_marker):
                block = _block_from(fence_buffer)
                if block:
                    yield block
                    any_content = True
                fence_buffer = []
                in_fence = False
            continue

        fence_match = _MD_FENCE_RE.match(line.strip())
        if fence_match:
            block = _block_from(para_buffer)
            if block:
                yield block
                any_content = True
            para_buffer = []
            in_fence = True
            fence_marker = fence_match.group(1)
            fence_buffer = [line]
            continue

        heading_match = _MD_HEADING_RE.match(line)
        if heading_match:
            block = _block_from(para_buffer)
            if block:
                yield block
                any_content = True
            para_buffer = []
            level = len(heading_match.group(1))
            section_stack[level - 1 :] = [heading_match.group(2).strip()]
            continue

        if line.strip() == "":
            block = _block_from(para_buffer)
            if block:
                yield block
                any_content = True
            para_buffer = []
            continue

        para_buffer.append(line)

    trailing_buffer = fence_buffer if in_fence else para_buffer
    block = _block_from(trailing_buffer)
    if block:
        yield block
        any_content = True

    if not any_content:
        raise EmptyDocumentError("The Markdown file contains no extractable text.")
