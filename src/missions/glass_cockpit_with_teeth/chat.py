"""Minimal terminal chat loop backed by an OpenAI Agents SDK model."""

import asyncio
import sys

from dotenv import load_dotenv

from missions.glass_cockpit_with_teeth.llm_client import (
    LLMClient,
    LLMInitialisationError,
    LLMRequestError,
)
from missions.glass_cockpit_with_teeth.store import ConversationStore
from missions.glass_cockpit_with_teeth.telemetry import LLMMetrics

EXIT_KEYWORDS = ["exit", "quit", "bye"]


def emit(metrics: LLMMetrics) -> None:
    """Report one LLM call: a human-readable line to stdout, a JSON line to stderr.

    The stderr line is newline-delimited JSON, ready to pipe into ``jq``.
    """
    print(str(metrics))
    print(metrics.model_dump_json(), file=sys.stderr)


async def chat() -> int:
    load_dotenv()
    print("Glass Cockpit — type a message. Ctrl+C or 'exit' to quit.")

    try:
        client = LLMClient(store=ConversationStore())
    except LLMInitialisationError as exc:
        print(f"Could not initialise the LLM client: {exc}")
        return 1

    while True:
        try:
            user_input = (await asyncio.to_thread(input, ">>> ")).strip()
        except (EOFError, KeyboardInterrupt):
            print("")
            return 0

        if not user_input:
            continue
        if user_input.lower() in EXIT_KEYWORDS:
            return 0

        try:
            async for chunk in client.send(user_input):
                print(chunk, end="", flush=True)
            print()
        except LLMRequestError as exc:
            print(f"\nerror: {exc}")
            continue

        if client.last_metrics:
            emit(client.last_metrics)


if __name__ == "__main__":
    raise SystemExit(asyncio.run(chat()))
