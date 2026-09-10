"""Contradiction detection tests. Built incrementally per stage; by the end
covers all 7 spec tests. Fully offline — Qdrant/LLM never touched here."""
from __future__ import annotations

import uuid
from types import SimpleNamespace

import pytest

from app.schemas import ScoredChunk
from app.services import contradictions as cx


def _chunk(chunk_id: str, document_id: str, text: str = "Some statement.") -> ScoredChunk:
    return ScoredChunk(
        chunk_id=chunk_id,
        document_id=document_id,
        document_name=f"{document_id}.pdf",
        text=text,
        page_start=1,
        page_end=1,
        section_path=[],
        ordinal=0,
        vector_score=0.9,
        rank_before=0,
        rank_after=0,
    )


def test_fingerprint_is_order_independent():
    fp_ab = cx.fingerprint("chunk-a", "chunk-b")
    fp_ba = cx.fingerprint("chunk-b", "chunk-a")
    assert fp_ab == fp_ba


def test_fingerprint_differs_for_different_pairs():
    assert cx.fingerprint("c1", "c2") != cx.fingerprint("c1", "c3")


def test_same_document_pairs_are_always_excluded():
    chunks = [
        _chunk("c1", "doc-1"),
        _chunk("c2", "doc-1"),
        _chunk("c3", "doc-2"),
    ]

    pairs, log_entries = cx.form_pairs(chunks)

    pair_ids = {(a.chunk_id, b.chunk_id) for a, b in pairs}
    assert ("c1", "c2") not in pair_ids
    assert ("c2", "c1") not in pair_ids
    assert ("c1", "c3") in pair_ids
    assert ("c2", "c3") in pair_ids

    same_doc_rejections = [e for e in log_entries if e.reason == "same_document"]
    assert len(same_doc_rejections) == 1
    assert {same_doc_rejections[0].chunk_a_id, same_doc_rejections[0].chunk_b_id} == {"c1", "c2"}


def test_form_pairs_with_no_chunks_or_one_chunk_produces_nothing():
    assert cx.form_pairs([]) == ([], [])
    assert cx.form_pairs([_chunk("c1", "doc-1")]) == ([], [])


def _unit_vector(angle_from_reference_deg: float, dim: int = 8) -> list[float]:
    """Builds a vector whose cosine similarity to the all-ones reference
    vector is cos(angle). Lets tests target an exact, known cosine value
    instead of hoping real embeddings happen to land somewhere."""
    import math

    ref = [1.0] * dim
    orth = [1.0, -1.0] + [0.0] * (dim - 2)
    rad = math.radians(angle_from_reference_deg)
    ref_norm = math.sqrt(dim)
    orth_norm = math.sqrt(2)
    return [
        math.cos(rad) * (r / ref_norm) + math.sin(rad) * (o / orth_norm) for r, o in zip(ref, orth)
    ]


def test_cosine_similarity_of_identical_vectors_is_one():
    v = [0.1, 0.2, 0.3, 0.4]
    assert cx.cosine_similarity(v, v) == pytest.approx(1.0)


def test_cosine_similarity_matches_known_angle():
    ref = [1.0] * 8
    v60 = _unit_vector(60.0)
    assert cx.cosine_similarity(ref, v60) == pytest.approx(0.5, abs=1e-6)


def test_pairs_below_lower_bound_are_excluded():
    ref = [1.0] * 8
    far = _unit_vector(90.0)
    a = _chunk("a", "doc-1", "Some claim about topic A.")
    a.vector = ref
    b = _chunk("b", "doc-2", "An unrelated claim about topic B.")
    b.vector = far

    kept, log_entries, exemptions = cx.filter_by_cosine([(a, b)], sim_min=0.75, sim_max=0.97, max_pairs=8)

    assert kept == []
    assert exemptions == 0
    assert log_entries[0].reason == "similarity_below_threshold"
    assert log_entries[0].accepted is False


def test_near_duplicate_pair_is_excluded_above_upper_bound():
    v = [0.1, 0.2, 0.3, 0.4, 0.5]
    a = _chunk("a", "doc-1", "The office is open on weekdays for all staff.")
    a.vector = v
    b = _chunk("b", "doc-2", "The office is open on weekdays for all staff.")
    b.vector = v

    kept, log_entries, exemptions = cx.filter_by_cosine([(a, b)], sim_min=0.75, sim_max=0.97, max_pairs=8)

    assert kept == []
    assert exemptions == 0
    assert log_entries[0].reason == "near_duplicate"
    assert log_entries[0].accepted is False


def test_numeric_different_pair_survives_upper_bound_despite_near_duplicate_cosine():
    v = [0.1, 0.2, 0.3, 0.4, 0.5]
    a = _chunk("a", "doc-1", "The service fee is $50 per transaction.")
    a.vector = v
    b = _chunk("b", "doc-2", "The service fee is $75 per transaction.")
    b.vector = v

    kept, log_entries, exemptions = cx.filter_by_cosine([(a, b)], sim_min=0.75, sim_max=0.97, max_pairs=8)

    assert len(kept) == 1
    assert exemptions == 1
    accepted_entries = [e for e in log_entries if e.accepted]
    assert len(accepted_entries) == 1
    assert accepted_entries[0].reason == "numeric_exempt"


def test_numeric_tokens_extraction():
    assert cx.numeric_tokens("The fee is $50 per transaction, up 12.5%.") == {"50", "12.5"}
    assert cx.numeric_tokens("No numbers here at all") == set()
    assert cx.numeric_tokens("$50 per transaction") != cx.numeric_tokens("$75 per transaction")


def test_pair_cap_keeps_highest_cosine_and_logs_the_rest_as_over_limit():
    pairs = []
    for i in range(10):
        a = _chunk(f"a{i}", f"doc-a{i}", "claim")
        b = _chunk(f"b{i}", f"doc-b{i}", "counter-claim")
        angle = 10.0 + i
        a.vector = [1.0] * 8
        b.vector = _unit_vector(angle)
        pairs.append((a, b))

    kept, log_entries, _ = cx.filter_by_cosine(pairs, sim_min=0.0, sim_max=0.999, max_pairs=3)

    assert len(kept) == 3
    over_limit = [e for e in log_entries if e.reason == "over_pair_limit"]
    assert len(over_limit) == 7
    kept_ids = {a.chunk_id for a, b in kept}
    assert kept_ids == {"a0", "a1", "a2"}


def _verdict(**overrides) -> "ContradictionVerdict":
    from app.schemas import ContradictionVerdict

    defaults = dict(
        pair_id="P1",
        is_contradiction=True,
        type="factual",
        severity="critical",
        confidence=0.9,
        statement_a="The fee is $50.",
        statement_b="The fee is $75.",
        explanation="Fees differ.",
        reconciliation="none",
    )
    defaults.update(overrides)
    return ContradictionVerdict(**defaults)


def test_verbatim_spans_pass_verification():
    a = _chunk("a", "doc-1", "Details here. The fee is $50. More details follow.")
    b = _chunk("b", "doc-2", "Other text. The fee is $75. End of section.")
    verdict = _verdict()

    assert cx.spans_present(verdict, a, b) is True


def test_fabricated_quote_is_rejected():
    a = _chunk("a", "doc-1", "Details here. The fee is $50. More details follow.")
    b = _chunk("b", "doc-2", "Other text. The fee is $75. End of section.")
    verdict = _verdict(statement_a="The fee is actually $999, a totally different number.")

    assert cx.spans_present(verdict, a, b) is False


def test_span_check_is_whitespace_and_case_insensitive():
    a = _chunk("a", "doc-1", "Details here.\n  The   FEE is $50.  \nMore details.")
    b = _chunk("b", "doc-2", "Other text. The fee is $75. End.")
    verdict = _verdict(statement_a="the fee is $50.")

    assert cx.spans_present(verdict, a, b) is True


def test_span_check_tolerates_pdf_mid_word_wrap_with_no_hyphen():
    """Real bug, reproduced live against sample_docs/remote-work-policy-2024.pdf:
    pdfplumber hard-wraps "they"/"manager" across a line with no hyphen
    ("once th\neyhave" -> extracted as "th ey"), so the model's clean quote
    ("...once they have...") was failing verbatim-span verification even
    though it's exactly what the source says. Whitespace-collapsing alone
    doesn't fix this — collapsing "th ey" still leaves "th ey", one space,
    never "they". Only stripping whitespace entirely closes the gap."""
    a = _chunk("a", "doc-1", "Details here.")
    b = _chunk(
        "b",
        "doc-2",
        "Employees may work remotely up to 3 days per week with manager approval, once th\ney have "
        "completed 90 days of employment.",
    )
    verdict = _verdict(
        statement_a="Details here.",
        statement_b="Employees may work remotely up to 3 days per week with manager approval, once they have "
        "completed 90 days of employment.",
    )

    assert cx.spans_present(verdict, a, b) is True


def test_span_check_still_rejects_genuinely_different_content_despite_no_space_fallback():
    """The no-space fallback must not swallow real fabrications — a quote
    with different words, not just different whitespace, still fails."""
    a = _chunk("a", "doc-1", "The fee is $50.")
    b = _chunk("b", "doc-2", "The fee is $75.")
    verdict = _verdict(statement_a="The fee is $999, a number never mentioned anywhere.")

    assert cx.spans_present(verdict, a, b) is False


def _fake_contradiction(id_, chunk_a_id, chunk_b_id, type_="numerical"):
    return SimpleNamespace(id=id_, chunk_a_id=chunk_a_id, chunk_b_id=chunk_b_id, type=type_)


def test_star_topology_from_shared_chunk_merges_into_one_group():
    shared = uuid.uuid4()
    a, b, c = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    records = [
        _fake_contradiction(uuid.uuid4(), shared, a),
        _fake_contradiction(uuid.uuid4(), shared, b),
        _fake_contradiction(uuid.uuid4(), c, shared),
    ]

    groups = cx.group_contradictions(records)

    assert len(groups) == 1
    assert len(groups[0]) == 3


def test_different_type_does_not_merge_even_with_shared_chunk():
    shared = uuid.uuid4()
    records = [
        _fake_contradiction(uuid.uuid4(), shared, uuid.uuid4(), type_="numerical"),
        _fake_contradiction(uuid.uuid4(), shared, uuid.uuid4(), type_="logical"),
    ]

    groups = cx.group_contradictions(records)

    assert len(groups) == 2
    assert {len(g) for g in groups} == {1, 1}


def test_disjoint_pairs_stay_in_separate_groups():
    records = [
        _fake_contradiction(uuid.uuid4(), uuid.uuid4(), uuid.uuid4()),
        _fake_contradiction(uuid.uuid4(), uuid.uuid4(), uuid.uuid4()),
    ]

    groups = cx.group_contradictions(records)

    assert len(groups) == 2


def test_transitive_chain_merges_across_multiple_hops():
    a, b, c = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    records = [
        _fake_contradiction(uuid.uuid4(), a, b),
        _fake_contradiction(uuid.uuid4(), b, c),
    ]

    groups = cx.group_contradictions(records)

    assert len(groups) == 1
    assert len(groups[0]) == 2


def test_empty_input_produces_no_groups():
    assert cx.group_contradictions([]) == []


def test_single_record_is_its_own_group():
    records = [_fake_contradiction(uuid.uuid4(), uuid.uuid4(), uuid.uuid4())]
    groups = cx.group_contradictions(records)
    assert len(groups) == 1
    assert len(groups[0]) == 1


def test_evidence_item_omits_fields_already_shown_at_group_level():
    """ContradictionEvidenceItem must not carry type/severity/status/
    explanation/reconciliation — those are on the group already, and
    repeating them per evidence item was exactly the payload bloat this
    shape exists to avoid."""
    from app.schemas import ContradictionEvidenceItem

    fields = set(ContradictionEvidenceItem.model_fields)
    assert fields == {"id", "confidence", "statement_a", "statement_b"}
    for redundant in ("type", "severity", "status", "explanation", "reconciliation"):
        assert redundant not in fields
