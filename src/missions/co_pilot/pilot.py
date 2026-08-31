"""Command-line entrypoint for the Ops Co-pilot.

    uv run python -m missions.co_pilot.pilot "Draft a follow-up plan for \
application #A-1423, stuck in verification for 4 days"

The validated plan (or refusal) is printed as JSON to **stdout**; the agent's
scratch-pad reasoning trace goes to **stderr**. Exit code: 0 = plan produced,
3 = clean refusal, 1 = error.
"""

from __future__ import annotations

import sys

import structlog
from dotenv import load_dotenv

from missions.co_pilot.agent import OpsCopilotAgent
from missions.logging import setup_logging

log = structlog.get_logger("co_pilot")

EXIT_PLAN = 0
EXIT_ERROR = 1
EXIT_REFUSAL = 3


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    request = " ".join(args).strip()
    if not request:
        print('usage: python -m missions.co_pilot "<operator request>"', file=sys.stderr)
        return EXIT_ERROR

    load_dotenv()
    setup_logging()

    try:
        result = OpsCopilotAgent().run(request)
    except Exception as exc:  # auth / network / config failure — not a clean refusal
        log.error("co_pilot.error", error=str(exc))
        return EXIT_ERROR

    if result is None:  # the agent run raised — not a clean refusal
        log.error("co_pilot.error", error="agent run produced no result")
        return EXIT_ERROR

    if result.outcome == "plan" and not result.steps:  # a plan with no steps is not a plan
        log.error("co_pilot.error", error="plan outcome with no steps")
        print(result.model_dump_json(indent=2))
        return EXIT_ERROR

    print(result.model_dump_json(indent=2))
    return EXIT_PLAN if result.outcome == "plan" else EXIT_REFUSAL


if __name__ == "__main__":
    raise SystemExit(main())
