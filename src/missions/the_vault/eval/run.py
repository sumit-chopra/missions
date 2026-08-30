"""Local eval for The Vault ``/ask`` — no LangSmith, no openevals.

For each ``eval.json`` question we get two answers — ``raw`` and ``rag`` —
and an LLM judge scores each 0/1 against ``reference``.

    make eval-vault
"""

import json
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from missions.the_vault.rag import CHAT_MODEL, CitedAnswer, Rag

EVAL_PATH = Path(__file__).parent / "eval.json"
JUDGE_MODEL = "gpt-5.4-nano"
QUESTION_COLUMN_WIDTH = 44
FLAVOUR_COLUMN_WIDTH = 30
PLAIN_FLAVOUR = "(plain lookup)"  # label for records whose ``flavour`` is ""

JUDGE_PROMPT = (
    "Grade the candidate answer against the reference answer for one question. "
    "Mark it correct only if the candidate states the key facts in the reference "
    "(numbers, names, clauses must match). Ignore wording and extra detail; a "
    "declined or contradicting answer is incorrect.\n"
    "Exception: when the reference says the answer is not in the source or the "
    "question cannot be answered, a candidate that declines or says it does not "
    "know is correct, and a candidate that supplies a specific answer is "
    "incorrect.\n\n"
    "Question: {question}\n\nReference: {reference}\n\nCandidate: {answer}"
)


class Verdict(BaseModel):
    score: bool = Field(description="True if the candidate matches the reference.")
    reasoning: str = Field(description="One short sentence explaining the grade.")


@dataclass
class QuestionResult:
    """One eval row: both pipelines graded, plus the rag citation scores."""

    question: str
    flavour: str
    raw_ok: int
    rag_ok: int
    citations_source_score: int
    citations_section_score: int


def build_models() -> tuple[ChatOpenAI, ChatOpenAI]:
    """The bare answerer and the structured-output judge, both deterministic."""
    load_dotenv()
    raw_llm = ChatOpenAI(model=CHAT_MODEL, temperature=0)
    judge_llm = ChatOpenAI(model=JUDGE_MODEL, temperature=0).with_structured_output(Verdict)
    return raw_llm, judge_llm


def raw_answer(raw_llm: ChatOpenAI, question: str) -> str:
    """Answer with no retrieval — the baseline the rag pipeline has to beat."""
    return raw_llm.invoke(question).content.strip()


def judge(judge_llm: ChatOpenAI, question: str, reference: str, answer: str) -> bool:
    """Ask the judge whether ``answer`` matches ``reference`` for ``question``."""
    if not answer and not reference:
        return 1

    verdict = judge_llm.invoke(
        JUDGE_PROMPT.format(question=question, reference=reference, answer=answer)
    )
    return verdict.score


def citation_scores(
    result: CitedAnswer, sources: list[str], sections: list[str]
) -> tuple[int, int]:
    """1.0 each only when the emitted citations are exactly the expected set."""
    return (
        int({c.source_file for c in result.citations} == set(sources)),
        int({c.section for c in result.citations} == set(sections)),
    )


def evaluate_question(
    question: dict, rag: Rag, raw_llm: ChatOpenAI, judge_llm: ChatOpenAI
) -> QuestionResult:
    """Run both pipelines for one eval record and grade them."""
    text, reference = question["question"], question["reference"]
    result = rag.answer_question(text)
    citations_source_score, citations_section_score = citation_scores(
        result, question["sources"], question["sections"]
    )
    return QuestionResult(
        question=text,
        flavour=question["flavour"],
        raw_ok=int(judge(judge_llm, text, reference, raw_answer(raw_llm, text))),
        rag_ok=int(judge(judge_llm, text, reference, result.answer)),
        citations_source_score=citations_source_score,
        citations_section_score=citations_section_score,
    )


def _truncate(text: str, width: int = QUESTION_COLUMN_WIDTH) -> str:
    return text if len(text) <= width else text[: width - 1] + "…"


def format_table(results: list[QuestionResult]) -> str:
    """Render the whole score table as one string.

    Returned in one piece so ``main`` can print it after retrieval is done,
    keeping it from interleaving with ``rag.py``'s structlog output.
    """
    width = QUESTION_COLUMN_WIDTH
    rule = f"{'-' * width}  ---  ---"
    count = len(results)

    raw_total = sum(r.raw_ok for r in results)
    rag_total = sum(r.rag_ok for r in results)
    citations_source_score = sum(r.citations_source_score for r in results)
    citations_section_score = sum(r.citations_section_score for r in results)

    lines = [
        f"{'question':<{width}}  raw  rag",
        rule,
        *(f"{_truncate(r.question):<{width}}  {r.raw_ok:>3}  {r.rag_ok:>3}" for r in results),
        rule,
        f"{f'total (of {count})':<{width}}  {raw_total:>3}  {rag_total:>3}",
        "",
        f"rag citations — sources {citations_source_score / count:.2f}  "
        f"sections {citations_section_score / count:.2f}",
    ]
    return "\n".join(lines)


def format_flavour_summary(results: list[QuestionResult]) -> str:
    """rag score broken out by ``flavour`` — how the pipeline fares per trick type.

    ``n`` is the group size (the score to beat); ``rag`` is how many the rag
    pipeline got right in that group.
    """
    width = FLAVOUR_COLUMN_WIDTH
    rule = f"{'-' * width}  ---  ---"

    groups: dict[str, list[QuestionResult]] = {}
    for r in results:
        groups.setdefault(r.flavour or PLAIN_FLAVOUR, []).append(r)

    lines = [
        f"{'flavour':<{width}}  {'n':>3}  {'rag':>3}",
        rule,
        *(
            f"{name:<{width}}  {len(group):>3}  {sum(r.rag_ok for r in group):>3}"
            for name, group in sorted(groups.items())
        ),
        rule,
        f"{'total':<{width}}  {len(results):>3}  {sum(r.rag_ok for r in results):>3}",
    ]
    return "\n".join(lines)


def main() -> int:
    raw_llm, judge_llm = build_models()
    questions = json.loads(EVAL_PATH.read_text(encoding="utf-8"))["questions"]
    rag = Rag.bootstrap()

    results = [evaluate_question(q, rag, raw_llm, judge_llm) for q in questions]

    print(f"\n{format_table(results)}\n\n{format_flavour_summary(results)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
