"""Chunking adapter. The only third-party import here is
langchain-text-splitters, used purely for splitting a single oversized
block — everything else (segmenting, packing, overlap, merging) is our own
logic living beside it because it's the one caller.

Pipeline, in order:
  1. Group blocks into segments by (page, section_path). Never merge across
     this boundary — it's what guarantees a chunk belongs to exactly one
     page and one section, which is what makes citations unambiguous.
  2. Pre-split any block whose text alone exceeds the max-char ceiling.
  3. Merge a segment under the min-char floor forward into the next
     segment, but only if they share the same page and the same parent
     section (i.e. the same heading one level up).
  4. Pack each segment's blocks into chunks, with overlap carried between
     consecutive chunks of the same segment only.
"""
from __future__ import annotations

import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from itertools import groupby

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.adapters.extraction import Block

_RECURSIVE_SEPARATORS = ["\n\n", "\n", ". ", "! ", "? ", "; ", " "]
_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+")


@dataclass
class ChunkDraft:
    """Chunker output. Distinct from the `Chunk` ORM row and the Qdrant
    payload — this is the in-memory shape the service layer turns into
    both."""

    text: str
    page_start: int | None
    page_end: int | None
    section_path: list[str]
    ordinal: int
    char_count: int


@dataclass
class _Segment:
    page: int | None
    section_path: list[str] = field(default_factory=list)
    blocks: list[Block] = field(default_factory=list)


def chunk_blocks(
    blocks: Iterable[Block],
    *,
    target_chars: int,
    max_chars: int,
    min_chars: int,
    overlap_chars: int,
) -> Iterator[ChunkDraft]:
    block_list = list(blocks)

    expanded: list[Block] = []
    for block in block_list:
        if len(block.text) > max_chars:
            expanded.extend(_split_oversized(block, max_chars))
        else:
            expanded.append(block)

    segments = [
        _Segment(page=page, section_path=list(section_path), blocks=list(group))
        for (page, section_path), group in groupby(expanded, key=lambda b: (b.page, tuple(b.section_path)))
    ]

    segments = _merge_undersized_segments(segments, min_chars)

    ordinal = 0
    for segment in segments:
        for text in _pack_segment([b.text for b in segment.blocks], target_chars, max_chars, overlap_chars):
            yield ChunkDraft(
                text=text,
                page_start=segment.page,
                page_end=segment.page,
                section_path=segment.section_path,
                ordinal=ordinal,
                char_count=len(text),
            )
            ordinal += 1


def _split_oversized(block: Block, max_chars: int) -> list[Block]:
    """A block over the ceiling splits recursively; every fragment inherits
    the same page and section. chunk_overlap=0 here — overlap between the
    resulting chunks is added later by the packing stage, so adding it
    twice would double it up."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_chars, chunk_overlap=0, separators=_RECURSIVE_SEPARATORS
    )
    fragments = splitter.split_text(block.text)
    return [
        Block(text=fragment, page=block.page, section_path=block.section_path, ordinal=block.ordinal)
        for fragment in fragments
    ]


def _parent(section_path: list[str]) -> list[str]:
    return section_path[:-1]


def _segment_text_length(segment: _Segment) -> int:
    return len("\n\n".join(b.text for b in segment.blocks))


def _merge_undersized_segments(segments: list[_Segment], min_chars: int) -> list[_Segment]:
    merged: list[_Segment] = []
    i = 0
    while i < len(segments):
        current = segments[i]
        while (
            _segment_text_length(current) < min_chars
            and i + 1 < len(segments)
            and segments[i + 1].page == current.page
            and _parent(segments[i + 1].section_path) == _parent(current.section_path)
        ):
            nxt = segments[i + 1]
            current = _Segment(
                page=current.page,
                section_path=nxt.section_path,
                blocks=current.blocks + nxt.blocks,
            )
            i += 1
        merged.append(current)
        i += 1
    return merged


def _pack_segment(block_texts: list[str], target_chars: int, max_chars: int, overlap_chars: int) -> list[str]:
    chunks: list[str] = []
    buffer = ""
    buffer_is_seed = False
    idx = 0
    while idx < len(block_texts):
        block_text = block_texts[idx]
        candidate = f"{buffer}\n\n{block_text}" if buffer else block_text
        if len(candidate) <= max_chars:
            buffer = candidate
            buffer_is_seed = False
            idx += 1
            if len(buffer) >= target_chars and idx < len(block_texts):
                chunks.append(buffer)
                buffer = _overlap_seed(buffer, overlap_chars)
                buffer_is_seed = bool(buffer)
        elif buffer and not buffer_is_seed:
            chunks.append(buffer)
            buffer = _overlap_seed(buffer, overlap_chars)
            buffer_is_seed = bool(buffer)
        else:
            buffer = block_text
            buffer_is_seed = False
            idx += 1
    if buffer and not buffer_is_seed:
        chunks.append(buffer)
    return chunks


def _overlap_seed(prev_chunk: str, overlap_chars: int) -> str:
    """Tail of the just-closed chunk, snapped forward to a sentence
    boundary so the overlap never starts mid-sentence."""
    if overlap_chars <= 0 or not prev_chunk:
        return ""
    raw_tail = prev_chunk[-overlap_chars:]
    match = _SENTENCE_BOUNDARY_RE.search(raw_tail)
    snapped = raw_tail[match.end() :] if match else raw_tail
    return snapped.strip()
