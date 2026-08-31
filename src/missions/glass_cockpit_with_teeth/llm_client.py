"""OpenAI Agents SDK client for Glass Cockpit, with teeth."""

import os
import time
from collections.abc import AsyncIterator

from agents import Agent, Runner, function_tool, set_tracing_disabled
from agents.exceptions import AgentsException
from openai import OpenAIError
from openai.types.responses import ResponseTextDeltaEvent

from missions.glass_cockpit_with_teeth.store import ConversationStore
from missions.glass_cockpit_with_teeth.telemetry import LLMMetrics

DEFAULT_MODEL = "gpt-5.4-mini"
SYSTEM_PROMPT = (
    "You are Glass Cockpit, a concise and helpful terminal assistant.\n\n"
    "When the user shares a durable fact, standing rule, or preference "
    "(contact preferences, loan constraints, and the like), call save_memory to "
    "keep it for future sessions. Do not save transient chatter or one-off "
    "questions."
)

set_tracing_disabled(True)


class LLMError(Exception):
    """Base class for errors raised by this module."""


class LLMInitialisationError(LLMError):
    """The client could not be constructed — bad or missing credentials.

    Fatal: retrying without fixing the configuration will not help.
    """


class LLMRequestError(LLMError):
    """A single agent run failed — timeout, rate limit, server error, etc.

    Often transient: the caller may retry or carry on.
    """


class LLMClient:
    """Agents SDK client with short-term history and long-term memory."""

    def __init__(self, store: ConversationStore) -> None:
        if not os.environ.get("OPENAI_API_KEY"):
            raise LLMInitialisationError("OPENAI_API_KEY is not set")
        self.model = DEFAULT_MODEL
        self.last_metrics: LLMMetrics | None = None
        self.store = store

    def _save_memory_tool(self):
        """A ``save_memory`` tool closed over this client's store."""

        @function_tool
        async def save_memory(key: str, value: str) -> str:
            """Persist a durable user fact, rule, or preference for future sessions.

            Use only for lasting information, never for transient chatter or
            one-off questions.

            Args:
                key: Short snake_case identifier, e.g. 'preferred_contact_day'.
                value: The fact to remember, e.g. 'Tuesdays only'.
            """
            self.store.save_memory(key, value)
            return f"Saved memory '{key}'."

        return save_memory

    def _instructions(self) -> str:
        """The base prompt, plus any stored memories as a bulleted fact list."""
        memories = self.store.get_memories()
        if not memories:
            return SYSTEM_PROMPT
        facts = "\n".join(f"- {key}: {value}" for key, value in memories.items())
        return f"{SYSTEM_PROMPT}\n\nKnown facts about the user:\n{facts}"

    def _agent_input(self, message: str) -> list[dict[str, str]]:
        """Recorded turns as chat messages, then the current user message."""
        items: list[dict[str, str]] = []
        for turn in self.store.recent_turns():
            items.append({"role": "user", "content": turn.user_message})
            items.append({"role": "assistant", "content": turn.assistant_message})
        items.append({"role": "user", "content": message})
        return items

    async def send(self, message: str) -> AsyncIterator[str]:
        self.last_metrics = None
        start = time.perf_counter()

        agent = Agent(
            name="Glass Cockpit",
            model=self.model,
            instructions=self._instructions(),
            tools=[self._save_memory_tool()],
        )
        result = Runner.run_streamed(agent, self._agent_input(message))

        reply: list[str] = []
        try:
            async for event in result.stream_events():
                if event.type == "raw_response_event" and isinstance(
                    event.data, ResponseTextDeltaEvent
                ):
                    reply.append(event.data.delta)
                    yield event.data.delta
        except (AgentsException, OpenAIError) as exc:
            raise LLMRequestError(f"request failed: {exc}") from exc

        self.store.add_turn(message, "".join(reply))

        usage = result.context_wrapper.usage
        if usage and usage.requests:
            self.last_metrics = LLMMetrics(
                model_name=self.model,
                prompt_tokens=usage.input_tokens,
                completion_tokens=usage.output_tokens,
                latency_ms=round((time.perf_counter() - start) * 1000),
            )
