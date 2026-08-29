# Acme corpus — reference dataset

This directory contains **~1.6 MB of synthetic Acme material** across
**8 documents**. It is the reference corpus for Mission 2 (RAG
service) and the source-of-truth for the sample evaluation questions
in `../quiz.json`.

All content is fictional. Any resemblance to real Acme policies,
real ASIC / AUSTRAC publications, or Tolkien's actual prose is
deliberate styling, not source material.

## What's in here

### Internal policies (5 documents)

| File | Purpose |
|---|---|
| `acme_lending_policy.md` | Loan product rules, clause structure, responsible-lending references |
| `acme_hardship_policy.md` | Hardship assessment, repayment-pause rules, escalation cadence |
| `acme_sla_handbook.md` | Ops SLAs, queue definitions, escalation cadence, business hours |
| `acme_product_info.md` | Loan ranges, fees, terms, comparison-rate examples |
| `acme_complaints_policy.md` | Complaint acknowledgement, AFCA referral procedure |

### Regulatory / external (2 documents)

| File | Purpose |
|---|---|
| `asic_rg209_synthesis.md` | Synthesised summary of ASIC RG 209 responsible-lending guidance |
| `nccp_excerpt.md` | Excerpts from the National Consumer Credit Protection Act |

### Narrative / trick source (1 document)

| File | Purpose |
|---|---|
| `silmaril_charter.md` | Fictional prose used for common-knowledge and exact-quote trick questions |

## Trick-question coverage

The 8 documents together span all four trick-question flavours the
brief lists:

- **Specific figure / clause / table lookup** — `acme_lending_policy.md`,
  `acme_product_info.md`, `acme_sla_handbook.md`
- **Contradicts common knowledge** — `silmaril_charter.md`
- **Cross-document** — pair `acme_hardship_policy.md` with
  `acme_sla_handbook.md`, or `acme_lending_policy.md` with
  `asic_rg209_synthesis.md`
- **Near-miss paraphrase / exact quote** — `silmaril_charter.md`

## Extending the corpus

You may replace or extend this corpus per the brief. If you do,
update this README to document your additions.
