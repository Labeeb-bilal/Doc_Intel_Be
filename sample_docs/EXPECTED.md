# Contradiction detection — test oracle

5 documents: 2 PDF, 2 DOCX, 1 MD. Two carry explicit effective dates about a
year apart (`remote-work-policy-2024.pdf` = 2024-01-15,
`employee-handbook-2023.docx` = 2023-01-10). The other three are undated —
deliberately, so the factual/numerical pairs below can't be confused with
temporal reasoning; only the logical/temporal pair involves a dated document.

## Planted contradictions

### 1. Logical + temporal — work location
- **Documents:** `remote-work-policy-2024.pdf` vs `employee-handbook-2023.docx`
- **Statements:** "Employees may work remotely up to 3 days per week..." vs
  "All employees must work in-office 5 days per week... Remote work is not
  permitted..."
- **Why it's both types:** the rules are mutually exclusive (logical), and
  the documents carry different effective dates for the same underlying
  rule (temporal). This is intentional — a real temporal contradiction
  *is* a logical one, resolved by recognizing recency. The detector should
  classify it `type=temporal` given the dates are present, with
  `reconciliation=supersedes` and severity downgraded from what a
  same-dated logical conflict would get (expect `warning` or `info`, not
  `critical`).
- **Expected:** `is_contradiction=true`, `type=temporal`,
  `reconciliation=supersedes`, explanation names the 2024 doc as superseding.

### 2. Factual — service fee
- **Documents:** `service-agreement.pdf` vs `contractor-terms.docx`
- **Statements:** "The service fee is $50 per transaction..." vs "The
  service fee is $75 per transaction..."
- **Why it's a real contradiction-detection stress test:** the wording is
  near-identical apart from the number and the last few words — cosine
  similarity is expected to land in or above the near-duplicate band
  (>0.97). It must survive the upper-bound filter via the **numeric
  exemption** (`{50} != {75}`), not get rejected as boilerplate.
- **Expected:** `is_contradiction=true`, `type=factual` (or `numerical` —
  either is defensible for a dollar-amount conflict; what matters is
  `is_contradiction=true` and the pair surviving the cosine upper bound).

### 3. Numerical — vacation days
- **Documents:** `service-agreement.pdf` vs `benefits-policy.md`
- **Statements:** "Full-time employees accrue 20 vacation days annually..."
  vs "Full-time employees accrue 15 vacation days annually..."
- **Why:** same structural test as #2 — near-identical template sentence,
  different number ({20} vs {15}), must survive the upper bound via the
  numeric exemption.
- **Expected:** `is_contradiction=true`, `type=numerical`.

## Planted non-contradictions

### A. Near-duplicate boilerplate — confidentiality clause
- **Documents:** `service-agreement.pdf` vs `contractor-terms.docx`
- **Statements:** "Employees must not disclose confidential company
  information to third parties without written consent." vs "Employees and
  contractors must not disclose any confidential company information to
  third parties without prior written consent."
- **Why it must NOT be flagged:** near-identical restatement of the same
  clause, no numeric tokens in either (both empty sets, so the numeric
  exemption does not apply) — this pair should be rejected by the cosine
  **upper bound** (>0.97) before it ever reaches the LLM. It's the
  highest-yield false-positive filter working as designed.
- **Expected:** filtered at the cosine stage (`reason=near_duplicate`), OR
  if it does reach the LLM, `is_contradiction=false`.

### B. Scope difference — contractor vs full-time employee vacation
- **Documents:** `contractor-terms.docx` vs `service-agreement.pdf` (or
  `benefits-policy.md`)
- **Statements:** "Contractors accrue 0 vacation days annually; vacation
  accrual under this agreement applies only to full-time employees, not
  contractors." vs "Full-time employees accrue 20 vacation days
  annually..." (or the 15-day version)
- **Why this is the adversarial one:** phrased deliberately close to the
  numerical-contradiction template ("[subject] accrue [N] vacation days
  annually") specifically so it clears the cosine floor and actually
  reaches the LLM — a naive numeric-difference detector would flag `0` vs
  `20`/`15` as a numerical contradiction. It is not one: the rules apply to
  different employment classes (contractors vs full-time employees). This
  is the real test of the prompt's negative scope-difference rule.
- **Expected:** `is_contradiction=false` (scope/employment-class
  difference, not a genuine conflict).

## Document effective dates
| Document | Effective date |
|---|---|
| `remote-work-policy-2024.pdf` | 2024-01-15 |
| `employee-handbook-2023.docx` | 2023-01-10 |
| `service-agreement.pdf` | none (best-effort, null is expected) |
| `contractor-terms.docx` | none |
| `benefits-policy.md` | none |
